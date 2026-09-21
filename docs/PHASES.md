# qwen-image-rs — phase tracker

Each phase is driven through the `@magistr/issue-lifecycle` swamp model (one
issue per phase). Status here mirrors the model state; the model is source of
truth (`swamp model method run <issue> hydrate`).

Locked decisions (2026-09-21): **candle spine + FFI kernels**; **correct bf16
end-to-end before any optimization**; **GGUF supported alongside safetensors**;
4090(Ada)-tuned optimizations.

| # | Phase | Component | Oracle gate | IL issue | Status |
|---|-------|-----------|-------------|----------|--------|
| 0 | Scaffold + host | repo, CUDA build, latest Rust | candle GPU smoke passes on 4090 | (folded into oracle) | ✅ done |
| 1 | Oracle harness | diffusers reference set | 3 PNGs + latents, fixed seed | `qwen-image-rs-oracle` | ✅ complete (attested 4302633) |
| 2 | VAE decode | `AutoencoderKLQwenImage21` (2.1, all 2D convs) | latent→RGBA matches oracle | `qwen-image-rs-vae` | ✅ complete (PSNR 51–53 dB) |
| 3 | Text encoder | `Qwen3VLForConditionalGeneration` (text-only) | embeds match within tol | `qwen-image-rs-text-encoder` | ✅ complete (cosine 0.9993) |
| 4 | DiT + sampler | `QwenImage21Transformer2DModel` + FlowMatchEuler | first e2e image ≈ oracle @ bf16 | `qwen-image-rs-dit` | ✅ complete (DiT cos 0.99996; e2e img 30.85 dB) |
| 5 | Optimization | FP8 → SageAttention → GGUF → VAE tiling → CUDA-graph | each ≥ bf16 quality, faster | `qwen-image-rs-optimize` | not started |
| 4b | Standalone generate | tokenizer + VRAM sequencing | prompt → PNG, no dumps | (folded) | ✅ done (~69s, no OOM) |
| 6 | Bench + package | CLI, latency/VRAM table vs ComfyUI | reproducible bench | `qwen-image-rs-bench` | not started |

## Findings so far
- **Oracle runs fast**: 15B model via `enable_model_cpu_offload` decodes ~37s/img (2.03 it/s, 40 steps) on the 4090 — **no OOM**, quick iteration viable.
- **Latent geometry**: 1024² → 64×64×64 f32 (16× compression, 64 channels), confirmed from `oracle_out/*.latent.pt`.
- **VAE is Wan-style 3D causal-conv** (`AutoencoderKLQwenImage`): `CausalConv3d`, RMS-norm, residual blocks, mid-block attention, nearest-exact resample. Decodes single frames (T=1).
- **candle 0.11 has no conv3d** — turned out to be a non-issue: the *2.1* `QwenImage21CausalConv3d` subclasses `nn.Conv2d` (folds T away), so the whole decoder is 2D. (The older `AutoencoderKLQwenImage` uses real 3D convs; the 2.1 model does not.)
- **VAE decoder validated** (candle vs diffusers bf16 oracle): PSNR 53.3/51.4/52.8 dB across the 3 latents — sub-LSB MAE, the residual is the f32-vs-bf16 storage floor. Run: `qwen-image-rs vae-decode --weights <vae> --latent <x>.safetensors --out x.png`.
- **2.1 latent unpack** ≠ old pipeline: `(B, seq, C).transpose(1,2).reshape(B,C,√seq,√seq)`; unnormalize `z = latent*std + mean` (raw std).
- **DiT**: 32 layers, 32×128=4096 hidden, context_in_dim 4096, 3D RoPE [16,56,56], mlp_ratio 3, causal_condition.
- **Text encoder validated** (candle vs oracle): per-token cosine mean 0.9993 across 3 prompts (bf16 on the 4090; 8B f32 OOMs). Standard Qwen3 decoder (36L, GQA 32/8, q/k RMSNorm, SwiGLU, RoPE θ=5e6); returns **pre-final-norm** hidden states (pipeline hooks the final norm away), drops the **14** system-prefix tokens. t2i template: system "Comprehend and analyze the provided prompt." Run: `qwen-image-rs text-encode --weights <text_encoder> --input-ids <ids>.safetensors --drop 14 --out e.safetensors`. NOTE: candle-transformers ships `qwen3_vl/text.rs` but returns post-norm+last-token only — wrote our own for pre-norm all-positions.

- **DiT + full pipeline validated** (candle vs oracle): single DiT forward cosine **0.99996** (MSE 2.7e-4); 32-layer MMDiT with complex 3-axis RoPE (via `rope_i`), block-causal attention (`allowed=(j≤i)∨both-image`), `causal_condition` modulation (image tokens use t, text t=0), ZeroCenterRMSNorm. FlowMatchEuler: sigmas linspace(1,1/N,N) → exp shift by `mu` → stretch to terminal 0.02 → Euler `x+=Δσ·v`. **End-to-end: 40-step denoise → latent cosine 0.9977 → VAE → image PSNR 30.85 dB vs oracle** (bf16 drift over 40 steps). First fully-Rust image: `docs/first_generated.png`. CLI: `dit-forward`, `denoise`.

## Phase 5 optimization order (4090-specific)
1. **FP8 e4m3fn** weight-only on DiT + text-encoder linears (native Ada FP8 TC).
2. **SageAttention INT8** QK (~2.1x over FA2 on 4090). FFI the upstream `.cu`; do not rewrite.
3. **GGUF** Q4_K/Q5_K/Q8_0 low-VRAM path (candle `quantized::gguf_file`) — fits 24 GB without offload.
4. **VAE tiling/slicing** for ≥1024² decode.
5. **CUDA-graph** capture of the denoise loop.

Each is a feature flag, validated against the Phase-4 bf16 oracle before it counts.

## Host facts (recon 2026-09-21)
- WSL, RTX 4090 24 GB, driver 616.92, CUDA 13.3 (nvcc), 913 GB free.
- Rust 1.98 → being updated to latest stable. cuda-oxide/cargo-oxide present (iris-rs); unused here.
- Model **not gated** — `Qwen/Qwen-Image-2.1` downloads without a token.
- Oracle needs a Python 3.12 venv (torch has no 3.14 wheels) — see scripts/setup-host.sh.
