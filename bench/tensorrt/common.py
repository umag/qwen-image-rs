"""Shared pieces for the TensorRT / PyTorch DiT benchmarks.

`DiTCore` is an export-friendly re-statement of
`QwenImage21Transformer2DModel.forward` for the text-to-image case (text prefix,
then one target image, no KV cache — the same full joint-sequence forward that
qwen-image-rs runs every step). It REUSES the diffusers submodules (same
weights) and only replaces the parts ONNX/TensorRT cannot take:

* complex RoPE  -> real cos/sin (precomputed per text length, graph inputs)
* the Python-list block-causal prefill -> two SDPA calls: the text queries
  causal over the text keys, the image queries over all keys (exactly what
  diffusers' default `QwenImage21AttnProcessor` computes for this layout)
* data-dependent token metadata -> static slicing (text first, image last)

`check_core()` in bench_torch.py proves DiTCore == diffusers forward.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HOME = Path(os.path.expanduser("~"))
SNAP = Path(os.environ.get(
    "QIR_SNAP",
    HOME / "dev_tmp/weights/hf/hub/models--Qwen--Qwen-Image-2.1/snapshots/b3179ad355be050328e483a9dfdd9e60cd62adfa"))
ORACLE = Path(os.environ.get("QIR_ORACLE", HOME / "dev_tmp/oracle_out"))
WORK = Path(os.environ.get("TRT_BENCH", HOME / "dev_tmp/trt-bench"))
IMG_H = IMG_W = 64  # 1024px / 16 -> 64x64 latent tokens = 4096


def load_transformer(device="cuda", dtype=torch.bfloat16):
    from diffusers import QwenImage21Transformer2DModel
    t = QwenImage21Transformer2DModel.from_pretrained(SNAP / "transformer", torch_dtype=dtype)
    return t.to(device).eval().requires_grad_(False)


def load_dit_io(device="cuda"):
    from safetensors.torch import load_file
    d = load_file(str(ORACLE / "dit_io.safetensors"))
    return {k: v.to(device) for k, v in d.items()}


def rope_cos_sin(text_len: int, device="cuda"):
    """Real cos/sin (S, 64) f32 for [text_len text tokens, 64x64 image].
    Uses diffusers' own (weight-free) QwenImage21Rope, axes (16, 56, 56)."""
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    pos_embed = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    img_mask = torch.zeros(text_len + IMG_H * IMG_W // 4, dtype=torch.bool, device=device)
    img_mask[text_len:] = True
    repeats = torch.where(img_mask, 4, 1)
    pad = torch.repeat_interleave(img_mask, repeats)
    freqs = pos_embed([(1, IMG_H, IMG_W)], pad, device=device)  # (S, 64) complex64
    return freqs.real.float().contiguous(), freqs.imag.float().contiguous()


def _rope(x, cos, sin):
    # x (B,S,H,D) ; pairs are interleaved (x0,x1) like view_as_complex
    xf = x.float().unflatten(-1, (-1, 2))
    x0, x1 = xf[..., 0], xf[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)
    return out.to(x.dtype)


def sdpa(q, k, v, causal=False):
    """SDPA; under the legacy TorchScript ONNX exporter (the one ModelOpt's Q/DQ
    symbolics need) it is spelled out with every tensor in the input dtype —
    that exporter's SDPA symbolic mixes f32 and bf16, which a strongly typed
    TensorRT network rejects. TensorRT re-fuses the pattern into its MHA kernel."""
    if not (torch.onnx.is_in_onnx_export() and not _is_dynamo_export()):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    s = (q * (1.0 / math.sqrt(int(q.shape[-1])))) @ k.transpose(-1, -2)  # python float: no traced Pow/Cast
    if causal:
        n = q.shape[-2]
        s = s + torch.full((n, n), float("-inf"), dtype=s.dtype, device=s.device).triu(1)
    return s.softmax(-1) @ v


def _is_dynamo_export():
    try:
        return torch.compiler.is_compiling() or torch.compiler.is_exporting()
    except AttributeError:
        return torch.compiler.is_compiling()


class ImgAttn(torch.nn.Module):
    """The image-query attention (4096 queries x all keys) as its own module so
    ModelOpt can attach FP8 q/k/v quantizers to it (export_onnx.py --mha); once
    quantized, ModelOpt swaps F.scaled_dot_product_attention for its FP8SDPA."""

    def forward(self, q, k, v):
        if hasattr(self, "q_bmm_quantizer"):
            return F.scaled_dot_product_attention(q, k, v)
        return sdpa(q, k, v)


class DiTCore(torch.nn.Module):
    """forward(hidden_states (B,4096,64), encoder_hidden_states (B,L,4096),
    timestep (B,), cos (S,64), sin (S,64)) -> (B, L+4096, 64)."""

    def __init__(self, transformer):
        super().__init__()
        self.t = transformer
        self.img_attn = torch.nn.ModuleList(ImgAttn() for _ in transformer.transformer_blocks)

    def forward(self, hidden_states, encoder_hidden_states, timestep, cos, sin):
        t = self.t
        B = hidden_states.shape[0]
        L = encoder_hidden_states.shape[1]
        img = t.img_in(hidden_states)
        txt = t.txt_in(encoder_hidden_states)
        x = torch.cat([txt, img], dim=1)  # text prefix, then the target image

        ts = timestep.to(img.dtype)
        ts = torch.cat([ts, ts.new_zeros(1)], dim=0)  # causal_condition: extra t=0 row
        temb = t.time_text_embed(ts, img)  # (B+1, D)
        mod = t.modulation(temb)  # (B+1, 4D)
        real = mod[:B].unsqueeze(1)  # image tokens: their sample's t
        zero = mod[B:].unsqueeze(0)  # text tokens: t=0 row

        def rows(p_real, p_zero):  # (B, S, n) per-token modulation
            return torch.cat([p_zero.expand(B, L, -1), p_real.expand(B, x.shape[1] - L, -1)], dim=1)

        mod1r, mod2r = real.chunk(2, dim=-1)
        mod1z, mod2z = zero.chunk(2, dim=-1)
        s1r, g1r = mod1r.chunk(2, dim=-1)
        s1z, g1z = mod1z.chunk(2, dim=-1)
        s2r, g2r = mod2r.chunk(2, dim=-1)
        s2z, g2z = mod2z.chunk(2, dim=-1)
        sc1, gt1 = 1 + rows(s1r, s1z), rows(g1r, g1z).tanh()
        sc2, gt2 = 1 + rows(s2r, s2z), rows(g2r, g2z).tanh()

        for blk, img_attn in zip(t.transformer_blocks, self.img_attn):
            a = blk.attn
            h = blk.img_norm1(x) * sc1
            q = a.norm_q(a.to_q(h).unflatten(-1, (a.heads, -1)))
            k = a.norm_k(a.to_k(h).unflatten(-1, (a.heads, -1)))
            v = a.to_v(h).unflatten(-1, (a.heads, -1))
            q, k = _rope(q.to(v.dtype), cos, sin), _rope(k.to(v.dtype), cos, sin)
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (B,H,S,D)
            o_txt = sdpa(q[:, :, :L], k[:, :, :L], v[:, :, :L], causal=True)
            o_img = img_attn(q[:, :, L:], k, v)
            o = torch.cat([o_txt, o_img], dim=2).transpose(1, 2).flatten(2, 3)
            x = x + gt1 * a.to_out[0](o)
            x = x + gt2 * blk.img_mlp(blk.img_norm2(x) * sc2)

        scale = t.norm_out.linear(t.norm_out.silu(temb).to(x.dtype))
        scale = rows(scale[:B].unsqueeze(1), scale[B:].unsqueeze(0))
        x = t.norm_out.norm(x) * (1 + scale)
        return t.proj_out(x)


def core_inputs(io, device="cuda", dtype=torch.bfloat16):
    """DiTCore inputs from the oracle dit_io (B=1, L=21)."""
    L = io["encoder_hidden_states"].shape[1]
    cos, sin = rope_cos_sin(L, device)
    return (io["hidden_states"].to(dtype), io["encoder_hidden_states"].to(dtype),
            io["timestep"].float(), cos, sin)


def cosine(a: torch.Tensor, b: torch.Tensor):
    """(overall cosine, per-image-token mean cosine) in f64, like compare_dit.py."""
    a = a.reshape(-1, a.shape[-1]).double()
    b = b.reshape(-1, b.shape[-1]).double()
    overall = float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
    ai, bi = a[-4096:], b[-4096:]
    per_tok = float(F.cosine_similarity(ai, bi, dim=1).mean())
    return overall, per_tok


def sustained(fn, n=60, warmup=5):
    """Time `fn` n times back to back after warmup; CUDA-event per call.
    Returns dict(median, p90, mean, min) in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) / 1000 for s, e in evs)
    return {"median": ts[n // 2], "p90": ts[int(n * 0.9)], "mean": sum(ts) / n, "min": ts[0], "n": n}


def fmt(r):
    return f"median={r['median']:.4f}s p90={r['p90']:.4f}s min={r['min']:.4f}s n={r['n']}"


def calib_samples(n_t=(1.0, 0.85, 0.6, 0.35, 0.1), device="cuda", dtype=torch.bfloat16):
    """Realistic DiT inputs for PTQ calibration: the 3 oracle prompts' embeddings
    x flow-matching interpolants x_t=(1-t)*x0 + t*noise of the oracle's final
    latents, plus the dit_io sample itself. Yields DiTCore arg tuples."""
    from safetensors.torch import load_file
    g = torch.Generator(device="cpu").manual_seed(1234)
    out = [core_inputs(load_dit_io(device), device, dtype)]
    for i in range(3):
        emb = load_file(str(ORACLE / f"{i:02d}.embeds.safetensors"))["embeds"][None].to(device, dtype)
        x0 = load_file(str(ORACLE / f"{i:02d}.latent.safetensors"))["latent"].to(device)
        cos, sin = rope_cos_sin(emb.shape[1], device)
        for t in n_t:
            noise = torch.randn(x0.shape, generator=g).to(device)
            xt = ((1 - t) * x0 + t * noise).to(dtype)
            out.append((xt, emb, torch.tensor([t], device=device), cos, sin))
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
