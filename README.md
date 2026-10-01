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
| BSHD-native attention (no transpose copies) | `sage` | 0.247 |
| RoPE fused into INT8 quant; V born f16 | `sage`,`convrot` | 0.2266 |
| SageAttention2: INT8-QK per-thread + K smoothing / FP8-PV | `sage2` | 0.2149 (same-session A/B 0.2259 → 0.2149, −4.8%) |
| SA2 quant: ten per-layer quant launches → three (L2-ordered, bit-identical) | `sage2` | 0.2127 (same-session A/B 0.2159 → 0.2127, −1.5%) |
| Hadamard rotation fused into the INT8 activation quantizer (no bf16 rotation GEMM) | `convrot` | 0.1911 (same-session A/B 0.2125 → 0.1911, −10.1%) |
| SwiGLU `silu(g)·p` fused into the MLP-out rotate+quantize (no bf16 h, no silu/mul passes) | `convrot` | **0.1746** (same-session A/B 0.1905 → 0.1746, −8.4%) |

Plus: bf16 VAE decode (1.57×) and tiled decode (constant memory); batched
multi-seed generation (`generate --batch N`, B=4 sweet spot). Recommended (default)
build: `--features convrot,sage,fusednorm,sage2`. Most accurate: drop `sage2`
(dit-forward vs oracle 0.999944). The `sage2` (`convrot,sage,fusednorm,sage2`, dit-forward 0.999911 — FP8
P·V; every DiT linear except img_in runs ConvRot INT8); in a `sage2` build `QIR_SAGE=1` selects SageAttention v1 (bit-identical to
the build without `sage2`), `QIR_SAGE=2f32` SA2 with fp32 P·V accumulation.
Resident batch ≈ 11 s/image. `--convrot` loads a cached prequantized DiT
(built once on first use, ~26 s): DiT load 7.0 → 2.7 s cold, 2.1 → 1.1 s warm.

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
Features: `cuda`, `cudnn`, `flash-attn`, `convrot`, `sage`, `sage2`, `fusednorm`. Default
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
