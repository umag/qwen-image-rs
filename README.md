# qwen-image-rs

Rust + CUDA inference for **Qwen-Image-2.1** on the RTX 4090 (Ada), built on
the [candle](https://github.com/huggingface/candle) spine with ComfyUI-class
optimizations (FP8, SageAttention, GGUF) ported/FFI'd in.

Research build only — the weights are under the **Qwen Research License**.

## Pipeline (`QwenImage21Pipeline`)
| Component | Class | ~Size | Module |
|-----------|-------|-------|--------|
| Text/vision encoder | `Qwen3VLForConditionalGeneration` | 8B | `model::text_encoder` |
| Transformer (DiT) | `QwenImage21Transformer2DModel` (32-layer single-stream) | 7B | `model::dit` |
| VAE | `AutoencoderKLQwenImage21` (64-ch RGBA, 16x) | — | `model::vae` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — | `model::scheduler` |

## Approach
1. **Correct bf16 end-to-end first**, validated against a diffusers oracle.
2. **Then optimize**, 4090-tuned, each behind a feature flag and re-checked
   against the bf16 oracle: FP8 e4m3fn → SageAttention INT8 → GGUF quant →
   VAE tiling → CUDA-graph.

Full plan and status: [docs/PHASES.md](docs/PHASES.md).

## Layout
```
src/model/     component ports (vae, text_encoder, dit, scheduler, config)
src/loader/    safetensors + GGUF weight resolution
src/device.rs  CPU/CUDA selection
scripts/       check.sh (Mac CPU checks) · host.sh (drive the 4090 via swamp)
               setup-host.sh (latest Rust + oracle venv) · oracle.py (references)
```

## Status: working end-to-end
Prompt → 1024² PNG in **~69 s** on a 4090 (unoptimized bf16). All three model
components ported to candle and numerically validated against a diffusers oracle:

| Component | Match vs oracle |
|-----------|-----------------|
| VAE decoder | PSNR 51–53 dB |
| Text encoder (Qwen3-VL) | per-token cosine 0.9993 |
| DiT (single forward) | cosine 0.99996 |
| Full pipeline (40 steps) | latent 0.9977 → image 30.85 dB |

See `docs/generate_standalone.png` (a red mug, from the prompt below).

## Build / run
```sh
# Mac: CPU-only checks (fmt, clippy, check, test)
scripts/check.sh

# WSL 4090 (via the swamp wsl-drills ssh model — never raw ssh):
scripts/host.sh smoke          # build --features cuda + GPU matmul smoke test

# Text-to-image (on the GPU host, --features cuda):
qwen-image-rs generate --model <snapshot> \
  --prompt "a red ceramic coffee mug on a wooden table, soft morning light" \
  --steps 40 --seed 42 --out out.png
```
Features: `cuda`, `cudnn`, `flash-attn`. Default build is CPU-only (macOS-safe).

## Host
WSL, RTX 4090 24 GB, CUDA 13.3, latest stable Rust. Shared with the paused
`iris-rs` CUDA port — coordinate GPU windows. Reach it only through the
`wsl-drills` `@swamp/ssh` model.
