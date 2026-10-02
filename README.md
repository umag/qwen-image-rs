# qwen-image-rs

Rust + CUDA inference for **Qwen-Image-2.1** text-to-image on one RTX 4090 (Ada,
sm89). Built on the [candle](https://github.com/huggingface/candle) spine. All
three model components are ported from scratch and validated against a diffusers
oracle. The speed comes from a stack of custom CUDA kernels, each one chosen
from an `nsys` profile and gated against that oracle before the next.

**Denoise: 0.62 → 0.148 s/step (~4.2×) at 1024². VAE decode: 1.15 → 0.27 s.
A resident pipeline renders a 1024² image every ~6.2 s.**

Research build only. This repo ships **no weights**. The Qwen-Image-2.1 weights
are under the **Qwen Research License** (research-only); see
[docs/WEIGHTS.md](docs/WEIGHTS.md).

---

## Performance

Measured 2026-10-02 on HEAD, RTX 4090 24 GB, WSL2, CUDA 13.3, 1024², 40 steps,
guidance 1. Recommended build (`--features convrot,sage,fusednorm,sage2,cudnn`).

### End to end

| Mode | Wall time | Per image | Peak VRAM |
|---|---|---|---|
| `generate`, 1 image, warm file cache | 18.1 s | 18.1 s | 13.8 GB |
| `generate`, 1 image, cold file cache | 27.0 s | 27.0 s | 13.8 GB |
| `generate --batch 4` (4 seeds, one denoise) | 38.2 s | 9.6 s | 13.8 GB |
| `batch --resident`, 4 prompts | 38.3 s | **6.2 s steady** (6.4 s before `-gemm-tiling-push`) | 21.3 GB |

Where a warm single `generate` spends its 18.1 s:

| Phase | Time |
|---|---|
| process start + tokenizer + text encoder (8 GB Q8_0 GGUF) load + encode | 10.4 s |
| DiT load (from the prequantized ConvRot cache) | 0.7 s |
| denoise, 40 steps × 0.148 s | 5.9 s |
| VAE load + decode (incl. ~0.2 s first-call cuDNN warm-up) | 0.7 s |

`batch --resident` keeps all three models loaded. Each image then costs encode
24 ms + denoise 5.91 s + decode 0.27 s (6.20 s; same-session A/B before the GEMM
tiling push: 6.39 s). Model loading is paid once.

`generate --batch N` is **throughput-neutral**: 0.616 s/step for 4 lanes is
0.154 s/step per image, the same as one image. The INT8 GEMMs already saturate
the GPU at B=1. Batching saves the per-process load and gives an SDXL-style seed
grid. B=4 is the efficient limit on 24 GB; from B=5 each image costs more.

### Inside one denoise step (~148 ms, `nsys`, B=1)

| Kernel | ms/step | Share | Note |
|---|---|---|---|
| INT8 GEMMs (CUTLASS, dequant in the epilogue) | 95.6 | 65% | 580–610 of ~660 INT8 TOPS kernel time; row-split plans (`-gemm-tiling-push`, was 100.5) |
| SageAttention2 (INT8 Q·K, FP8 P·V) | 18.7 | 12% | |
| SwiGLU `silu(g)·p` + rotate + INT8 quant | 9.3 | 6% | one fused kernel |
| gated residual + LayerNorm×(scale+1) | 7.8 | 5% | one fused kernel |
| SA2 quant (RoPE + RMSNorm + per-thread INT8, FP8 V) | 6.8 | 4% | one kernel, 3 L2-ordered launches |
| activation rotate + INT8 quant | 2.8 | 2% | |
| copies / misc | ~6 | 4% | |

### Inside one VAE decode (~0.27 s, 1024², bf16, whole image)

cuDNN implicit-GEMM conv 108 ms · cuDNN NCHW↔NHWC transforms 31 ms · bias adds
27 ms · fused channel RmsNorm(+SiLU) ~9 ms · fused bias+residual 6 ms · upsample
9 ms · copies.

### Quality (vs the diffusers bf16 oracle)

| Check | Result |
|---|---|
| Text encoder (Qwen3-VL, bf16) | per-token cosine 0.9993 (Q8_0 GGUF: 0.9977) |
| DiT single forward, recommended build (`--convrot`, SA2) | cosine 0.999911 |
| DiT single forward, `QIR_SAGE=1` (SageAttention v1) | cosine 0.999881 |
| DiT single forward, no `--convrot` | cosine 0.999945 |
| VAE decode | PSNR 56.4 dB |
| Determinism | same seed → byte-identical image; `--batch` lane 0 == the single image |

### vs TensorRT / PyTorch (one DiT forward, 1024², B=1, same session)

| Engine | s/forward | vs ours | cosine vs oracle |
|---|---|---|---|
| **qwen-image-rs** (INT8 ConvRot + SageAttention2) | **0.148** | 1.00× | 0.999911 |
| TensorRT 11.3 INT8 SmoothQuant (ModelOpt, α=0.8) | 0.216 | 1.46× slower | 0.998214 |
| TensorRT FP8 linears + FP8 attention (ModelOpt) | 0.253 | 1.72× | 0.999624 |
| TensorRT FP8 linears | 0.265 | 1.80× | 0.999699 |
| TensorRT bf16 | 0.428 | 2.90× | 0.999988 |
| `torch.compile` max-autotune, bf16 | 0.428 | 2.90× | 0.999983 |
| diffusers eager, bf16 | 0.506 | 3.43× | 1.000000 |

TensorRT on Ada refuses our INT8-GEMM + FP8-attention mix, and its static
per-tensor INT8 loses quality that ConvRot keeps. Method, per-kernel breakdown
and caveats: [docs/TENSORRT.md](docs/TENSORRT.md).

---

## Quick start

```sh
# 1. weights (accept the Qwen Research License on Hugging Face first)
hf download Qwen/Qwen-Image-2.1 --local-dir weights/qwen-image-2.1

# 2. host setup: latest Rust, the oracle venv, user-space cuDNN 9 (no root)
scripts/setup-host.sh

# 3. build the fast path (CUTLASS include dir required)
export CUTLASS_DIR=$HOME/dev_tmp/cutlass/include
cargo build --release --features convrot,sage,fusednorm,sage2,cudnn

# 4. optional: pre-convert (text encoder -> Q8_0 GGUF, warm the INT8 DiT cache)
scripts/convert.sh weights/qwen-image-2.1

# 5. one image
qwen-image-rs generate --model weights/qwen-image-2.1 \
  --prompt "a red ceramic coffee mug on a wooden table, soft morning light" \
  --convrot --text-gguf weights/qwen-image-2.1/qir/qir-text.gguf --out mug.png

# 6. four seed variations in one denoise
qwen-image-rs generate --model weights/qwen-image-2.1 --prompt "..." \
  --convrot --text-gguf weights/qwen-image-2.1/qir/qir-text.gguf \
  --batch 4 --seed 42 --out-dir grid/          # writes 000.png..003.png (seeds 42..45)

# 7. many prompts, models loaded once (~6.4 s/image)
qwen-image-rs batch --model weights/qwen-image-2.1 --prompts prompts.txt \
  --resident --convrot --text-gguf weights/qwen-image-2.1/qir/qir-text.gguf \
  --out-dir out/
```

The first `--convrot` run builds the prequantized INT8 DiT once (~26 s, a
7.1 GB cache entry under `~/.cache/qwen-image-rs/convrot/`). Later runs load it
in ~1 s. Full weights and conversion guide: **[docs/WEIGHTS.md](docs/WEIGHTS.md)**.

---

## CLI

| Verb | What it does |
|---|---|
| `generate` | prompt → PNG. Loads text encoder → DiT → VAE one at a time (fits 24 GB). |
| `batch` | many prompts from a file; `--resident` keeps all three models loaded |
| `prequantize-text` | text encoder → Q8_0 GGUF (`--text-gguf`) |
| `prequantize-convrot` | DiT → ConvRot INT8 file; without `--out` it warms the `--convrot` cache |
| `text-encode`, `denoise`, `dit-forward`, `vae-decode` | single stages, for validation against the oracle |
| `convrot-test`, `sage-test`, `fusednorm-test`, `cudnn-test`, `vae-fused-test` | kernel self-tests (bit-identity / cosine vs references) |
| `gemm-bench`, `bench`, `smoke`, `info` | micro-benchmarks, GPU smoke test, weight-set info |

Main `generate` / `batch` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model <snapshot>` | — | dir with `processor/ text_encoder/ transformer/ vae/` |
| `--prompt` / `--prompts <file>` | — | one prompt / one prompt per line |
| `--size` | 1024 | output side in pixels (multiple of 16) |
| `--steps` | 40 | flow-match Euler steps |
| `--seed` | 42 | noise seed; with `--batch N`, lanes use seed … seed+N−1 |
| `--batch N` | 1 | N images in one batched denoise (`generate`); needs `--out-dir` when N>1 |
| `--guidance` / `--negative` | 1.0 / "" | true CFG when > 1 (two DiT forwards per step) |
| `--convrot` | off | INT8 DiT (ConvRot W8A8); loads/creates the prequantized cache |
| `--text-gguf <file>` | — | load the text encoder from a Q8_0 GGUF (needed for `--resident`) |
| `--quant-text` / `--quant` | off | quantize text encoder / DiT to Q8_0 on load (VRAM tools, slow load) |
| `--vae-tile auto\|0\|N` | auto | `auto` decodes the whole image when it fits in free VRAM, else 32-latent tiles |
| `--vae-f32` | off | f32 VAE reference path |
| `--resident` | off | (`batch`) keep all models in VRAM |
| `--convrot-cache <dir>`, `--no-convrot-cache`, `--rebuild-convrot-cache` | — | control the INT8 DiT cache |
| `--emit-latents` | off | also save each image's final latent (debug) |

Environment:

| Variable | Effect |
|---|---|
| `QIR_SAGE=1\|2\|2f32` | attention in a `sage2` build: v1 (INT8/FP16, bit-identical to a non-`sage2` build), SA2 (default), SA2 with fp32 P·V accumulation |
| `QIR_CUDNN=0` | disable the cuDNN VAE conv (old im2col path, byte-identical to a non-`cudnn` build) |
| `QIR_CUDNN_LIB` | cuDNN 9 lib dir at build time (default: the user-space wheel from `setup-host.sh`) |
| `QIR_CONVROT_CACHE` | INT8 DiT cache root (default `$XDG_CACHE_HOME` or `~/.cache/qwen-image-rs/convrot`) |
| `QIR_VAE_FUSED`, `QIR_CUDNN_ALGO`, `QIR_CUDNN_DEBUG` | diagnostics |

### Cargo features

| Feature | What it adds |
|---|---|
| (none) | CPU-only build; compiles on macOS (used for `scripts/check.sh`) |
| `cuda` | candle CUDA backend |
| `convrot` | ConvRot W8A8 INT8 linears: CUTLASS INT8 GEMM with fused dequant epilogue, fused rotate+quant, fused SwiGLU, merged q\|k and gate\|proj GEMMs, prequant cache |
| `sage` | vendored SageAttention v1 (INT8 Q·K / FP16 P·V), BSHD-native, RoPE fused into the quant |
| `sage2` | SageAttention2 for sm89 (INT8 Q·K per-thread + K smoothing / FP8 P·V); default attention when built |
| `fusednorm` | fused LayerNorm+AdaLN with gated residual, fused RMSNorms, fused VAE channel RmsNorm(+SiLU) and bias/residual |
| `cudnn` | VAE convs on cuDNN 9 tensor-core implicit GEMM (own bridge; candle's built-in cuDNN conv is slower) |
| `flash-attn` | FlashAttention-2 attention path (the original bf16 baseline) |

Recommended: `convrot,sage,fusednorm,sage2,cudnn`. For the most accurate fast
build, drop `sage2` (DiT cosine 0.999944 instead of 0.999911).

---

## How it got fast

Every step was picked from an `nsys` profile (attack the largest kernel). Each
one was then checked against the oracle and driven through a reviewed, attested
issue lifecycle. Times are denoise s/step at 1024²; the later steps are
same-session A/B because absolute timings drift between sessions.

| Step | Feature | s/step |
|---|---|---|
| bf16 + FlashAttention-2 (baseline) | `flash-attn` | 0.62 |
| ConvRot W8A8 INT8 linears (Hadamard-rotated) | `convrot` | 0.51 |
| SageAttention INT8-QK / FP16-PV | `sage` | 0.48 |
| fused LayerNorm+AdaLN | `fusednorm` | 0.445 |
| fused activation quantizer | `convrot` | 0.38 |
| dequant moved into the CUTLASS EVT epilogue | `convrot` | 0.34 |
| fused RMSNorm×weight + gated residual | `fusednorm` | 0.27 |
| BSHD-native attention (no transpose copies) | `sage` | 0.247 |
| RoPE fused into the INT8 Q/K quant; V written f16 by its GEMM | `sage`,`convrot` | 0.2266 |
| SageAttention2 (sm89, FP8 P·V) | `sage2` | 0.2149 |
| SA2 quant: 10 launches → 3 (bit-identical) | `sage2` | 0.2127 |
| Hadamard rotation fused into the activation quant (in-register FWHT) | `convrot` | 0.1911 |
| SwiGLU fused into the MLP-out rotate+quant | `convrot` | 0.1746 |
| q\|k and gate\|proj merged GEMMs; per-shape CUTLASS tiles (bit-identical) | `convrot` | 0.1703 |
| per-head q/k RMSNorm fused into the SA2 quant (bit-identical) | `fusednorm`,`sage2` | 0.1647 |
| gated residual fused into the next LayerNorm+AdaLN (bit-identical) | `fusednorm` | 0.1522 |
| INT8 GEMM row-split plans: text rows past the 4096 image rows in their own small launch, 128x256 tiles (bit-identical) | `convrot` | **0.1477** (A/B 0.1524 → 0.1477, −3.1%; `--batch 2` −1.7%) |

VAE decode (1024², bf16):

| Step | Feature | `vae-decode` (whole) | resident decode |
|---|---|---|---|
| f32 decoder | — | — | 1.84 s |
| bf16 decoder | — | 0.95 s | 1.15 s (tiled) |
| cuDNN tensor-core implicit-GEMM conv (no im2col buffer; peak 15.7 → 8.9 GB) | `cudnn` | 0.51 s | 0.68 s |
| fused channel RmsNorm×γ(+SiLU), bias+residual in one pass (byte-identical) | `fusednorm` | **0.27 s** | 0.38 s |
| `--vae-tile auto`: whole-image decode when it fits (no seams, 48.5 → 56.4 dB) | — | — | **0.27 s** |

Other wins: text encoder as Q8_0 GGUF (resident VRAM), prequantized INT8 DiT
cache (DiT load 7.0 → 2.7 s cold), batched multi-seed generation, true CFG.

**Correctness bugs found along the way:**
- **Stale shared memory in the vendored SageAttention.** It loaded key rows past
  the end of the sequence without zero-filling them. The patch is marked
  LOCAL PATCH in `kernels/sage/vendor/qattn/attn_utils.cuh`.
- **Ignored view offsets at B=1.** Custom ops did not apply the tensor view's
  start offset. At B=1 that conditioned images on the system prompt. Every
  raw-pointer op now applies the offset through `src/layout.rs`, with tests.

Each fix is covered by a regression test. Both bugs had looked like
"non-determinism" until they were understood. The full log, with every number,
is in [HANDOVER.md](HANDOVER.md).

---

## What is left

| Lever | Expected gain | Note |
|---|---|---|
| text encoder load (single-shot `generate`) | up to ~8 s per process | 10.4 s of an 18 s run is loading the 8 GB GGUF; resident mode already avoids it |
| SwiGLU in the gate\|proj GEMM epilogue | ~3% denoise | large CUTLASS epilogue work |
| LayerNorm folded into the activation quant loader | ~1.7% denoise | plumbing through three code paths |
| NHWC end to end in the VAE | ~50 ms/decode | removes cuDNN layout transforms, lets cuDNN fuse the bias |
| distilled / lightweight VAE decoder | ~4–6× decode | lossy; needs distillation for this 64-ch RGBA VAE |
| CUDA graphs | launch overhead | blocked: candle 0.11 has no stream-capture hook |

The INT8 GEMMs (65% of a step, ~90% of the INT8 peak in kernel time) and
SageAttention2 are close to the 4090's limits. cuBLASLt IMMA is no faster at
these shapes and cannot fuse the per-row × per-col dequant (see HANDOVER "GEMM tiling push").
From here, denoise gains come in single-digit percent.

---

## Layout

```
src/model/        ports: text_encoder (Qwen3-VL), dit (QwenImage21 DiT), vae (2.1 decoder),
                  scheduler (flow-match Euler), rotation (Regular Hadamard), config
src/convrot.rs    ConvRot INT8 linears + bridges   src/convrot_cache.rs  prequant cache
src/gemm_tiles.rs per-shape CUTLASS tile table     src/layout.rs         view-offset helpers
src/sage.rs       SageAttention v1 bridge          src/sage2.rs          SageAttention2 bridge
src/rope.rs       BSHD RoPE                         src/fusednorm.rs      fused norm bridges
src/cudnn_conv.rs cuDNN conv bridge                 src/vae_fused.rs      fused VAE ops
src/main.rs       CLI                               src/loader/           safetensors + GGUF
kernels/convrot/  INT8 GEMM (CUTLASS EVT), rotate/quant kernels
kernels/sage/     SageAttention v1/v2 launchers, RoPE; vendor/ = thu-ml kernels (Apache-2.0)
kernels/fusednorm/ fused norm / residual / VAE norm kernels
scripts/          check.sh (Mac CPU checks) · host.sh (drive the GPU host) · setup-host.sh
                  convert.sh (weight conversion) · oracle.py (diffusers references)
                  compare_dit.py · compare_latent.py · compare_png.py
docs/             WEIGHTS.md · PHASES.md · sample images
```

## Validation

- `scripts/check.sh`: fmt, clippy `-D warnings`, check, unit tests on the CPU build (macOS).
- `scripts/oracle.py`: regenerates the diffusers reference set (fixed seeds: latents, embeddings, images).
- `dit-forward` + `scripts/compare_dit.py`, `vae-decode` + `scripts/compare_png.py`: oracle parity.
- `*-test` verbs: every custom kernel checked bit-for-bit or by cosine against a
  reference, including on offset views and NaN-poisoned shared memory.

## Hardware

Developed and measured on an RTX 4090 24 GB (sm89) under WSL2, CUDA 13.3, latest
stable Rust. The INT8/FP8 kernels target Ada. Other GPUs are untested. The vendored
SageAttention kernels and the CUTLASS configs are sm80+/sm89-specific.

## License & credits

Apache-2.0 (see `LICENSE`). Attributions in `NOTICE`:
- an independent port of **Qwen-Image-2.1** (Alibaba/Qwen, Qwen Research License, weights not included);
- vendors **thu-ml SageAttention** v1 and v2 (Apache-2.0) under `kernels/sage/vendor/`, with one marked local patch;
- uses **NVIDIA CUTLASS** for the INT8 GEMM and **cuDNN** for the VAE convs;
- built on **candle** (Hugging Face);
- the ConvRot scheme follows pytorch/ao's Regular Hadamard rotation for DiTs.
