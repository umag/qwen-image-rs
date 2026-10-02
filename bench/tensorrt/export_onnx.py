#!/usr/bin/env python3
"""Export DiTCore (bf16, or ModelOpt-quantized) to ONNX with external data.

  export_onnx.py --quant none  [--exporter dynamo|torchscript]   -> onnx/bf16/
  export_onnx.py --quant fp8   [--mha]                           -> onnx/fp8[-mha]/
  export_onnx.py --quant int8sq                                  -> onnx/int8sq/

Fixed shapes: B=1, 4096 image tokens, L=21 text tokens (the oracle prompt).
Quantized variants are calibrated in PyTorch on calib_samples() (16 real DiT
inputs: 3 prompts x 5 noise levels + the oracle sample), the fake-quant model's
cosine vs the oracle is printed, then exported with Q/DQ through ModelOpt's
TorchScript symbolics (the path ModelOpt supports for FP8/INT8 Q/DQ) and the
weights are folded to real FP8/INT8 by ModelOpt's ONNX exporters.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import onnx
import torch

from common import (DiTCore, Timer, WORK, calib_samples, core_inputs, cosine, load_dit_io,
                    load_transformer)

NAMES_IN = ["hidden_states", "encoder_hidden_states", "timestep", "cos", "sin"]
NAMES_OUT = ["output"]
# Outside the 32 blocks: tiny or precision-critical layers stay bf16 (the same
# choice qwen-image-rs makes: only the per-block GEMMs are INT8).
KEEP_BF16 = ["*img_in*", "*txt_in*", "*time_text_embed*", "*modulation*", "*norm_out*", "*proj_out*",
             # ModelOpt also wraps nn.LayerNorm with an input quantizer; that
             # would quantize the residual stream (amax ~8.5e3) — we quantize
             # only the GEMM inputs, like qwen-image-rs.
             "*img_norm1*", "*img_norm2*"]
MHA_QUANTIZERS = ("*q_bmm_quantizer", "*k_bmm_quantizer", "*v_bmm_quantizer", "*softmax_quantizer")


def quantize(core, quant, mha, alpha=None):
    import copy

    import modelopt.torch.quantization as mtq
    if mha:
        from modelopt.torch.quantization.nn import QuantModuleRegistry
        from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantAttention
        from common import ImgAttn
        if QuantModuleRegistry.get(ImgAttn) is None:
            QuantModuleRegistry.register({ImgAttn: "ImgAttn"})(_QuantAttention)
    if quant == "fp8":
        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    elif quant == "int8sq":
        cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        if alpha is not None:
            cfg["algorithm"] = {"method": "smoothquant", "alpha": alpha}
    elif quant == "int8":
        cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    else:
        raise SystemExit(quant)
    qc = cfg["quant_cfg"]
    if isinstance(qc, dict):
        for pat in KEEP_BF16:
            qc[pat] = {"enable": False}
        if mha:  # FP8 attention: quantize the SDPA operands (ModelOpt's FP8 MHA)
            for q in MHA_QUANTIZERS:
                qc[q] = {"num_bits": (4, 3), "axis": None}
            qc["*bmm2_output_quantizer"] = {"enable": False}
    else:  # newer list-style configs
        for pat in KEEP_BF16:
            qc.append({"quantizer_name": pat, "enable": False})
        if mha:
            for q in MHA_QUANTIZERS:
                qc.append({"quantizer_name": q, "cfg": {"num_bits": (4, 3), "axis": None}})
            qc.append({"quantizer_name": "*bmm2_output_quantizer", "enable": False})
    print("quant_cfg:", json.dumps(qc, default=str)[:1500], flush=True)
    samples = calib_samples()

    def loop(m):
        for s in samples:
            m(*s)

    with Timer() as tq, torch.inference_mode():
        mtq.quantize(core, cfg, forward_loop=loop)
    print(f"calibration {tq.s:.1f}s on {len(samples)} samples", flush=True)
    if mha:  # ModelOpt's FP8 MHA symbolic needs the Q/DQ "high precision" type = the 16-bit model dtype
        for m in core.img_attn:
            for qn in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer", "softmax_quantizer"):
                getattr(m, qn).trt_high_precision_dtype = "BFloat16"
    mtq.print_quant_summary(core) if hasattr(mtq, "print_quant_summary") else None
    return core


class _QDQ(torch.autograd.Function):
    """Symmetric INT8 fake quant whose ONNX form is a plain QuantizeLinear ->
    DequantizeLinear pair (bf16 scale, int8 zero point; axis 0 when per-channel)."""

    @staticmethod
    def forward(ctx, x, scale, zp):
        s = scale.view(-1, *([1] * (x.dim() - 1))) if scale.numel() > 1 else scale
        return (torch.clamp(torch.round(x.float() / s.float()), -128, 127) * s.float()).to(x.dtype)

    @staticmethod
    def symbolic(g, x, scale, zp):
        q = g.op("QuantizeLinear", x, scale, zp, axis_i=0)
        return g.op("DequantizeLinear", q, scale, zp, axis_i=0)


class QDQLinear(torch.nn.Module):
    """A calibrated ModelOpt INT8 QuantLinear re-expressed with _QDQ: same
    (smoothed) weight, SmoothQuant pre-scale, per-tensor activation scale and
    per-output-channel weight scale. ModelOpt's own INT8 TorchScript export
    segfaults in torch 2.14's tracer (see docs/TENSORRT.md), this one traces."""

    def __init__(self, m):
        super().__init__()
        iq, wq = m.input_quantizer, m.weight_quantizer
        dt = m.weight.dtype
        pqs = getattr(iq, "pre_quant_scale", None)
        self.register_buffer("pqs", None if pqs is None else pqs.detach().to(dt).reshape(-1))
        self.register_buffer("a_scale", (iq.amax.detach().float().reshape(()) / 127).to(dt))
        self.register_buffer("a_zp", torch.zeros((), dtype=torch.int8))
        ws = (wq.amax.detach().float().reshape(-1) / 127).clamp_min(1e-12).to(dt)
        self.register_buffer("w_scale", ws)
        self.register_buffer("w_zp", torch.zeros(ws.shape, dtype=torch.int8))
        self.weight = torch.nn.Parameter(m.weight.detach(), requires_grad=False)

    def forward(self, x):
        if self.pqs is not None:
            x = x * self.pqs
        x = _QDQ.apply(x, self.a_scale, self.a_zp)
        w = _QDQ.apply(self.weight, self.w_scale, self.w_zp)
        return torch.nn.functional.linear(x, w)


def int8_to_plain_qdq(core):
    n = 0
    for name, m in list(core.named_modules()):
        iq = getattr(m, "input_quantizer", None)
        if iq is None or not iq.is_enabled or not hasattr(m, "weight") or m.weight.dim() != 2:
            continue
        parent = core.get_submodule(name.rsplit(".", 1)[0]) if "." in name else core
        setattr(parent, name.rsplit(".", 1)[-1], QDQLinear(m))
        n += 1
    print(f"re-expressed {n} INT8 linears as plain Q/DQ", flush=True)
    return core


def qdq_scales_to_bf16(g):
    """ModelOpt's FP8 ONNX exporter inserts Q/DQ after Softmax with FLOAT scales,
    so the dequantized P is f32 while V is bf16; a strongly typed TensorRT
    network rejects that MatMul. Recast every FLOAT Q/DQ scale initializer to
    BF16 (weights' scales already are) so the graph stays bf16 end to end."""
    from onnx import numpy_helper
    inits = {i.name: i for i in g.graph.initializer}
    n = 0
    for node in g.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) > 1:
            t = inits.get(node.input[1])
            if t is not None and t.data_type == onnx.TensorProto.FLOAT:
                a = torch.from_numpy(numpy_helper.to_array(t).copy()).to(torch.bfloat16)
                t.ClearField("float_data")
                t.data_type = onnx.TensorProto.BFLOAT16
                t.raw_data = a.view(torch.int16).numpy().tobytes()
                n += 1
    print(f"recast {n} FLOAT Q/DQ scales to BF16", flush=True)
    return g


def strip_softmax_qdq(g):
    """Undo ModelOpt's FP8 exporter's Q/DQ-after-Softmax (inserted for TRT's FP8
    MHA fusion). Without FP8 q/k/v it is a lone FP8 P that blocks TensorRT's
    bf16 fused-MHA pattern, so attention would run unfused. Used when --mha is off."""
    by_out = {o: n for n in g.graph.node for o in n.output}
    drop, remap = set(), {}
    for n in g.graph.node:
        if n.op_type == "DequantizeLinear":
            q = by_out.get(n.input[0])
            if q is not None and q.op_type == "QuantizeLinear":
                src = by_out.get(q.input[0])
                if src is not None and src.op_type == "Softmax":
                    remap[n.output[0]] = q.input[0]
                    drop.update((id(n), id(q)))
    keep = [n for n in g.graph.node if id(n) not in drop]
    for n in keep:
        for i, x in enumerate(n.input):
            if x in remap:
                n.input[i] = remap[x]
    del g.graph.node[:]
    g.graph.node.extend(keep)
    print(f"stripped {len(remap)} Softmax Q/DQ pairs", flush=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="none", choices=["none", "fp8", "int8sq", "int8"])
    ap.add_argument("--mha", action="store_true", help="also FP8-quantize attention (ModelOpt FP8 MHA)")
    ap.add_argument("--exporter", default=None, choices=["dynamo", "torchscript"])
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--cos-only", action="store_true", help="calibrate + print cosine, no export")
    ap.add_argument("--alpha", type=float, default=None, help="SmoothQuant alpha (int8sq)")
    args = ap.parse_args()
    exporter = args.exporter or ("dynamo" if args.quant == "none" else "torchscript")
    name = args.name or (args.quant if args.quant != "none" else "bf16") + ("-mha" if args.mha else "") \
        + ("" if (args.exporter is None) else f"-{exporter}")
    outdir = WORK / "onnx" / name
    t = load_transformer()
    io = load_dit_io()
    xin = core_inputs(io)
    core = DiTCore(t).eval()
    meta = {"name": name, "quant": args.quant, "mha": args.mha, "exporter": exporter}
    if args.quant != "none":
        quantize(core, args.quant, args.mha, args.alpha)
        with torch.inference_mode():
            c = cosine(core(*xin).float(), io["output"])
        meta["fakequant_cos"] = c
        meta["alpha"] = args.alpha
        print(f"fake-quant (PyTorch) cos={c[0]:.6f} tok_cos={c[1]:.6f}", flush=True)
        if args.quant.startswith("int8"):
            int8_to_plain_qdq(core)
            with torch.inference_mode():
                c2 = cosine(core(*xin).float(), io["output"])
            meta["plain_qdq_cos"] = c2
            print(f"plain Q/DQ re-expression cos={c2[0]:.6f} (must match fake-quant)", flush=True)

    if args.cos_only:
        print(json.dumps(meta), flush=True)
        return
    tmp = Path(tempfile.mkdtemp(prefix="dit_onnx_", dir=WORK))
    raw = tmp / "model.onnx"
    kw = dict(input_names=NAMES_IN, output_names=NAMES_OUT, dynamo=(exporter == "dynamo"))
    if args.opset:
        kw["opset_version"] = args.opset
    if exporter == "dynamo":
        kw["external_data"] = True
    with Timer() as te, torch.inference_mode():
        torch.onnx.export(core, xin, str(raw), **kw)
    print(f"torch.onnx.export ({exporter}) {te.s:.1f}s", flush=True)

    with Timer() as tp:
        g = onnx.load(str(raw), load_external_data=True)
        if args.quant == "fp8":
            from modelopt.torch._deploy.utils.torch_onnx import quantize_weights
            g = quantize_weights(core, g)
            if not args.mha:
                g = strip_softmax_qdq(g)
            g = qdq_scales_to_bf16(g)
        elif args.mha:  # INT8 linears + ModelOpt FP8 MHA
            g = qdq_scales_to_bf16(g)
        outdir.mkdir(parents=True, exist_ok=True)
        for f in outdir.iterdir():
            f.unlink()
        onnx.save_model(g, str(outdir / "model.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024)
    shutil.rmtree(tmp)
    ops = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    meta.update(export_s=round(te.s, 1), post_s=round(tp.s, 1),
                opset=[(o.domain, o.version) for o in g.opset_import], ops=ops,
                bytes=sum(f.stat().st_size for f in outdir.iterdir()))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
