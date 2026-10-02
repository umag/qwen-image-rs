#!/usr/bin/env python3
"""PyTorch baselines for one Qwen-Image-2.1 DiT forward at 1024² (B=1).

  bench_torch.py --modes eager,core,compile,compile-max [--n 60]

eager       diffusers QwenImage21Transformer2DModel.forward (default processor,
            native SDPA = flash/efficient attention), bf16
core        DiTCore (bench/tensorrt/common.py) eager — the graph we export
compile     torch.compile(DiTCore) inductor, default mode
compile-max torch.compile(DiTCore, mode="max-autotune") (inductor GEMM
            autotune + CUDA graphs)
Each mode: cosine vs the oracle `output`, sustained s/forward, peak VRAM.
"""
import argparse
import json

import torch

from common import (DiTCore, Timer, core_inputs, cosine, fmt, load_dit_io, load_transformer,
                    sustained, WORK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="eager,core,compile,compile-max")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # torch default
    with Timer() as tl:
        t = load_transformer()
    print(f"load {tl.s:.1f}s  weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    io = load_dit_io()
    ref = io["output"]
    xin = core_inputs(io)
    results = {}

    def report(name, fn, extra=None):
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = fn()
            cos = cosine(out.float(), ref)
            r = sustained(fn, n=args.n)
        r.update(cos_overall=cos[0], cos_img_tok=cos[1],
                 peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30, **(extra or {}))
        results[name] = r
        print(f"[{name}] {fmt(r)} cos={cos[0]:.6f} tok_cos={cos[1]:.6f} peak={r['peak_vram_gib']:.2f}GiB {extra or ''}",
              flush=True)

    modes = args.modes.split(",")
    if "eager" in modes:
        img_mask = io["img_mask"].bool()
        report("torch-eager-diffusers", lambda: t(
            hidden_states=xin[0], encoder_hidden_states=xin[1], timestep=io["timestep"].to(torch.bfloat16),
            img_shapes=[[(1, 64, 64)]], img_mask=img_mask, return_dict=False)[0])
    core = DiTCore(t).eval()
    if "core" in modes:
        report("torch-eager-core", lambda: core(*xin))
    for m in ("compile", "compile-max"):
        if m not in modes:
            continue
        torch._dynamo.reset()
        kw = {"mode": "max-autotune"} if m == "compile-max" else {}
        c = torch.compile(core, fullgraph=True, dynamic=False, **kw)
        with Timer() as tc, torch.inference_mode():
            c(*xin)
            c(*xin)
            torch.cuda.synchronize()
        report(f"torch-{m}", lambda: c(*xin), {"compile_s": round(tc.s, 1)})
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "results_torch.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
