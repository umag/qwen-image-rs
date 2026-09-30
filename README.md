# qwen-image-rs

Rust + CUDA inference for **Qwen-Image-2.1** on the RTX 4090 (Ada), built on
the [candle](https://github.com/huggingface/candle) spine. The three model
components are ported from scratch and validated against a diffusers oracle;
speed comes from a stack of custom CUDA kernels (INT8 linears, SageAttention,
fused norms, a CUTLASS INT8-GEMM epilogue, BSHD-native attention), each gated
against that oracle.

**~2.5× faster than the bf16 baseline: 0.62 → 0.247 s/step** at 1024², with
`dit-forward` cosine held at 0.999934 through every optimization.

Research build only — this repo ships **no weights**. The Qwen-Image-2.1 weights
are under the **Qwen Research License** (research-only) and must be obtained
separately.

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

## Status: working end-to-end + optimized
All three components ported to candle and validated against a diffusers oracle:

| Component | Match vs oracle |
|-----------|-----------------|
| VAE decoder | PSNR 51–53 dB |
| Text encoder (Qwen3-VL) | per-token cosine 0.9993 |
| DiT (single forward) | cosine 0.99996 |
| Full pipeline (40 steps) | latent 0.9977 → image 30.85 dB |

### Optimization scoreboard (1024², RTX 4090, per denoise step)
Each step was chosen from an `nsys` profile (attack the largest kernel), then
re-checked against the oracle before the next one.

| Step | Feature | s/step |
|---|---|---|
| bf16 + FlashAttention-2 (baseline) | `flash-attn` | 0.62 |
| ConvRot W8A8 INT8 linears | `convrot` | 0.51 |
| SageAttention INT8-QK / FP16-PV | `sage` | 0.48 |
| Fused LayerNorm+AdaLN | `fusednorm` | 0.445 |
| Fused activation quantizer | `convrot` | 0.38 |
| Dequant → CUTLASS EVT epilogue | `convrot` | 0.34 |
| Fused RMSNorm×weight + gated residual | `fusednorm` | 0.27 |
| BSHD-native attention (no transpose copies) | `sage` | **0.247** |

Plus: bf16 VAE decode (1.57×) and tiled decode (constant memory); batched
multi-seed generation (`generate --batch N`, B=4 sweet spot). Recommended fast
build: `--features convrot,sage,fusednorm`. Resident batch ≈ 11 s/image.

See `docs/generate_standalone.png` and `docs/batch/` for samples.

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
Features: `cuda`, `cudnn`, `flash-attn`, `convrot`, `sage`, `fusednorm`. Default
build is CPU-only (macOS-safe).

## Weights
Not included (Qwen Research License). Download `Qwen/Qwen-Image-2.1` and, if you
want the low-VRAM/resident path, convert it — see **[docs/WEIGHTS.md](docs/WEIGHTS.md)**
(links, `scripts/convert.sh`, and the `prequantize-text` / `prequantize-convrot`
verbs).

## Host
WSL, RTX 4090 24 GB, CUDA 13.3, latest stable Rust.

## License & credits
Apache-2.0 (see `LICENSE`). Attributions in `NOTICE`: this is an independent
port of **Qwen-Image-2.1** (Alibaba/Qwen, Qwen Research License — weights not
included); it vendors **thu-ml SageAttention** (Apache-2.0) under
`kernels/sage/vendor/`, uses **NVIDIA CUTLASS** for the INT8 GEMM, and is built
on **candle** (Hugging Face).
