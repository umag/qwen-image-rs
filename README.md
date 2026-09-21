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

## Build / run
```sh
# Mac: CPU-only checks (fmt, clippy, check, test)
scripts/check.sh

# WSL 4090 (via the swamp wsl-drills ssh model — never raw ssh):
scripts/host.sh smoke          # build --features cuda + GPU matmul smoke test
scripts/host.sh build          # release build on the GPU host
```
Features: `cuda`, `cudnn`, `flash-attn`. Default build is CPU-only (macOS-safe).

## Host
WSL, RTX 4090 24 GB, CUDA 13.3, latest stable Rust. Shared with the paused
`iris-rs` CUDA port — coordinate GPU windows. Reach it only through the
`wsl-drills` `@swamp/ssh` model.
