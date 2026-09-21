# qwen-image-rs — phase tracker

Each phase is driven through the `@magistr/issue-lifecycle` swamp model (one
issue per phase). Status here mirrors the model state; the model is source of
truth (`swamp model method run <issue> hydrate`).

Locked decisions (2026-09-21): **candle spine + FFI kernels**; **correct bf16
end-to-end before any optimization**; **GGUF supported alongside safetensors**;
4090(Ada)-tuned optimizations.

| # | Phase | Component | Oracle gate | IL issue | Status |
|---|-------|-----------|-------------|----------|--------|
| 0 | Scaffold + host | repo, CUDA build, latest Rust | candle GPU smoke passes on 4090 | `qwen-image-rs-scaffold` | in progress |
| 1 | Oracle harness | diffusers reference set | 3 PNGs + latents + embeds, fixed seed | `qwen-image-rs-oracle` | pending env |
| 2 | VAE decode | `AutoencoderKLQwenImage21` | latent→RGBA matches oracle | `qwen-image-rs-vae` | not started |
| 3 | Text encoder | `Qwen3VLForConditionalGeneration` | embeds match within tol | `qwen-image-rs-text-encoder` | not started |
| 4 | DiT + sampler | `QwenImage21Transformer2DModel` + FlowMatchEuler | first e2e image ≈ oracle @ bf16 | `qwen-image-rs-dit` | not started |
| 5 | Optimization | FP8 → SageAttention → GGUF → VAE tiling → CUDA-graph | each ≥ bf16 quality, faster | `qwen-image-rs-optimize` | not started |
| 6 | Bench + package | CLI, latency/VRAM table vs ComfyUI | reproducible bench | `qwen-image-rs-bench` | not started |

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
