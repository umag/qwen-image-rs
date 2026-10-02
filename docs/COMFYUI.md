# qwen-image-rs vs ComfyUI

Measured 2026-10-02 in one session on the same host: RTX 4090 24 GB (480 W cap),
WSL2, CUDA 13.3 driver, 117 GB RAM. Workload: **Qwen-Image-2.1 text-to-image,
1024², 40 steps, cfg 1 (one DiT forward per step)**. Both sides use the same 4
prompts. qwen-image-rs is HEAD `c05396d`, rebuilt for this run with
`--features convrot,sage,fusednorm,sage2,cudnn`. ComfyUI is `master` at
`65787d6` (2026-10-01), torch 2.14.1+cu130, comfy-kitchen 0.2.36,
SageAttention 2.2.0 (built from source for sm89). Scripts and the API-format
workflows: [`bench/comfyui/`](../bench/comfyui/).

## Result

| Variant | ComfyUI flags | DiT weights / attention | s/step | per image, steady | cold first image ¹ | GPU mem ² | vs ours (step / image) |
|---|---|---|---|---|---|---|---|
| **qwen-image-rs** `batch --resident` | – | INT8 ConvRot / SageAttention2 (ours) | **0.147** ³ | **6.20 s** | 18.5–18.9 s | 19.2 GB | 1.00× |
| A. stock | – | bf16 / PyTorch SDPA | 0.419 | 19.54 s | 32.1 s | 21.8 GB ⁴ | 2.85× / 3.15× slower |
| official template default | – | INT8 ConvRot / PyTorch SDPA | 0.182 | 8.25 s | 15.8 s | 16.3 GB | 1.24× / 1.33× |
| B. fp8 + sage | `--use-sage-attention` | bf16 cast to `fp8_e4m3fn` / SageAttention 2.2 | 0.419 | 17.87 s | 26.0 s | 16.4 GB | 2.85× / 2.88× |
| C. B + fast | `--use-sage-attention --fast` | `fp8_e4m3fn` / SageAttention 2.2 | 0.419 | 17.79 s | 30.9 s | 16.8 GB | 2.85× / 2.87× |
| D. C + `TorchCompileModel` | as C | | **fails** ⁵ | | | | |
| D′. C + compile, DynamicVRAM off | C + `--disable-dynamic-vram` | `fp8_e4m3fn` (compiled) / SageAttention 2.2 | 0.243 | 10.62 s | 53.3 s (≈22 s compile) ⁶ | 22.3 GB | 1.65× / 1.71× |
| INT8 + sage | `--use-sage-attention` | INT8 ConvRot / SageAttention 2.2 | 0.156 | 7.24 s | 13.8 s | 16.3 GB | 1.06× / 1.17× |
| INT8 + comfy-kitchen attention | `--use-ck-attention` | INT8 ConvRot / comfy-kitchen INT8 | 0.156 | 7.26 s | 13.6 s | 16.3 GB | 1.06× / 1.17× |
| INT8 + sage + fast ⁷ | `--use-sage-attention --fast` | INT8 ConvRot / SageAttention 2.2 | 0.156 | 7.19 s | 16.7–18.3 s | 19.9 GB | 1.06× / 1.16× |
| INT8 + ck + fast | `--use-ck-attention --fast` | INT8 ConvRot / comfy-kitchen INT8 | 0.157 | 7.24 s | 17.2 s | 19.8 GB | 1.07× / 1.17× |
| **best ComfyUI**: INT8 + sage + fast, DynamicVRAM off | `--use-sage-attention --fast --disable-dynamic-vram` | INT8 ConvRot / SageAttention 2.2 | **0.156** | **6.94 s** | 28.4 s | 21.4 GB | **1.06× / 1.12×** |
| INT8 + `TorchCompileModel` (DynamicVRAM on or off) | | | **fails** ⁵ | | | | |
| E. NVIDIA FP8 checkpoint | | | **none exists** ⁸ | | | | |
| F. Nunchaku SVDQuant INT4 | | | **cannot run in ComfyUI** ⁹ | | | | |

No variant offloads during the steady images, with one exception: A. Its bf16
text encoder and bf16 DiT do not fit in 24 GB together (see ⁴).

**Not comparable at 40 steps** (a different, 6-step distilled checkpoint):

| Variant | flags | steps | s/step | per image, steady | cold first image | GPU mem |
|---|---|---|---|---|---|---|
| G. Viggle turbo v0.3, merged INT8 ConvRot DiT + its `ViggleTurboSigmas` node | `--use-sage-attention --fast` | 6 | 0.144 | **1.84 s** | 15.4 s | 17.0 GB |

The best Lightning-style option for 2.1 is Viggle turbo. Its card says "about 5×
faster" than 40 steps; here it is 3.9× faster than the best 40-step ComfyUI
config. lightx2v has published Lightning LoRAs only for Qwen-Image,
-2512 and -Edit-2511, not for 2.1.

### Definitions

- **s/step**: the median over the steady images of `(t(step N) − t(step 1)) / (N−1)`.
  The times come from ComfyUI's websocket progress events, so the first step's
  setup is excluded. Ours: `denoise_ms / 40`.
- **per image, steady**: the median `execution_start → execution_success`
  time of images 2–8. Eight prompts (4 prompts × 2 seeds) are queued at once
  and all models stay resident. Each image covers text encode, 40 steps, VAE
  decode and PNG save. Ours: the time between consecutive `done (resident)`
  lines of a 4-prompt `batch --resident` run (images 1–3), which covers the
  same work.
- All runs are sustained: during the denoise the card sits at its
  460–475 W cap (turbo: 396 W mean, because its runs are short).

¹ ComfyUI: the first prompt after a fresh server is up, including all model
loads and first-use setup. The server's own start-up (Python imports, node
registration) adds another 6–10 s before that. Ours: one `generate` process from
start to PNG, warm file cache (two runs; the first run after ComfyUI took
26.7 s, because ComfyUI pins 108 GB of host RAM and that pushes our weights out
of the page cache).
² Device-wide `nvidia-smi` peak minus the idle desktop baseline. The baseline
moves between runs (0.8–1.6 GB), so the values are ±0.8 GB. Ours is 21.3 GB raw
(the README value), 19.2 GB minus baseline, the same in both runs.
³ 0.1475 (suite 1) and 0.1471 (suite 2). Per image 6.21 and 6.19 s.
⁴ The bf16 text encoder (16.7 GB) and bf16 DiT (13.6 GB) do not both fit.
ComfyUI's DynamicVRAM streams them from pinned host RAM. Nothing fails, but text
encode costs 1.07 s per prompt instead of 0.37 s.
⁵ With DynamicVRAM on (the default), the first KSampler call raises
`AttributeError: 'torch.Stream' object has no attribute 'cuda_stream'`. Dynamo
traces into the DynamicVRAM allocator (`comfy/model_prefetch.py:malloc_graph_begin`
→ `comfy_aimdo/malloc_graph.py`). With `--disable-dynamic-vram` the fp8 path
compiles (D′). The INT8 path still fails:
`TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_method __dlpack__ …`. The comfy-kitchen INT8 kernels take their inputs
through DLPack, and dynamo cannot trace that. **`torch.compile` therefore cannot
be combined with the fast INT8 path in this ComfyUI build.**
⁶ Image 1 includes the inductor compile. One steady image (16.7 s) recompiled
once more; the median ignores it.
⁷ Two runs in the same session: 0.1565 / 0.1562 s/step, 7.20 / 7.19 s/image.
⁸ NVIDIA has published no FP8 ("NVFP8") Qwen-Image-2.1 checkpoint (`nvidia/*`
on Hugging Face has only Qwen-Image-Edit-NVPCB and Qwen-Image-Flash).
Community ModelOpt FP8 builds such as `HangGlidersRule/…-ModelOpt-FP8` are in
diffusers format, and ComfyUI has no loader for them. Comfy-Org ships no fp8
file: its quantized format is INT8 ConvRot. In ComfyUI, the only FP8 route is
B/C/D′, `weight_dtype fp8_e4m3fn` cast from the bf16 file. NVFP4 is
Blackwell-only.
⁹ ComfyUI-nunchaku (v1.2.0) supports Qwen-Image, -Edit and -Edit-2509, but not
2.1, whose DiT is a different architecture. The two 2.1 SVDQuant checkpoints
on HF are `catplusplus/nunchaku-qwen-image-2.1` (NVFP4, Blackwell-only) and
`BlazeMCworld/Qwen-Image-2.1-nunchaku-lite-int4` (diffusers from `main`, no
ComfyUI node).

## What the numbers say

- **The official ComfyUI template is already fast.** It loads Comfy-Org's
  INT8 ConvRot DiT and text encoder. That is the same ConvRot technique we use,
  run through NVIDIA's comfy-kitchen CUDA kernels: INT8 GEMM with fused
  rotate and quant, fused RMSNorm+RoPE, fused AdaLN. With PyTorch SDPA
  attention it does 0.182 s/step, 2.3× faster than stock bf16 ComfyUI.
- **One flag closes most of the gap.** `--use-sage-attention` (or
  `--use-ck-attention`) moves attention to INT8 and gives 0.156 s/step.
  That is **6% slower per step than qwen-image-rs** (0.147).
- **Per image, the best ComfyUI config is 12% slower** (6.94 s vs 6.20 s).
  That config adds `--disable-dynamic-vram`, which cuts ComfyUI's text encode
  from 0.37 to 0.20 s and its decode from 0.24 to 0.20 s. The remaining 0.74 s
  breaks down as: sampler node 6.35 s vs our denoise 5.89 s (+0.46 s), text
  encode 0.195 s vs 0.025 s (+0.17 s), executor/PNG overhead (~+0.18 s).
  **ComfyUI won the VAE decode** here: 0.20 s against our 0.27 s (−0.07 s).
  Since `qwen-image-rs-vae-nhwc` (after this table was measured) ours decodes
  in **0.13 s** (resident 0.14 s, per image 6.35 → 6.05 s same session), so
  the decode is now ours by 0.07 s. See "VAE decode" below.
- **`--fast` does nothing for this model.** It enables fp16 accumulation,
  fp8 matmul, cuBLAS ops and autotune. INT8: 0.1561 → 0.1562 s/step. fp8:
  0.4186 → 0.4188.
- **In eager mode, fp8 weights give no speed.** `weight_dtype fp8_e4m3fn`
  only halves the DiT memory. The log says `model weight dtype
  torch.float8_e4m3fn, manual cast: torch.bfloat16`: every linear is cast
  back to bf16. B and C run at the bf16 tensor-core limit (~0.419 s/step,
  the same as TensorRT bf16 in [TENSORRT.md](TENSORRT.md)), so SageAttention
  has no visible effect either. Only compiled (D′) does fp8 pay off: 0.243
  s/step, close to TensorRT FP8 (0.253–0.265). It needs ~22 s of compile and
  2.6 GB more memory, and it is still 1.65× slower than INT8.
- **ComfyUI uses less VRAM in its default mode.** It needs 16.3 GB for DiT,
  text encoder and VAE together, against our 19.2 GB. DynamicVRAM even runs the
  31 GB bf16 pair on a 24 GB card. Our resident mode keeps a bf16 activation
  workspace and allocates a 5 GB whole-image VAE decode buffer.
- **The fastest ComfyUI image is the turbo checkpoint**, 1.84 s per image in
  6 steps. That is a different model (a distillation). We have not tried it.
  Its diffusers-format transformer has the 2.1 architecture, but our CLI would
  need its custom 6-node sigma schedule.

### Images

![grid](comfyui_grid.jpg)

Rows: ours, best ComfyUI (INT8), ComfyUI fp8+compile, ComfyUI turbo 6-step.
Columns: the 4 prompts. All are clean, prompt-faithful 1024² images. For the same
seed, the ComfyUI variants give almost the same picture (bf16, fp8, INT8 and
the template differ only in fine detail), so the INT8 path costs no visible
quality. Ours differ in composition because the noise RNG and schedule differ:
ComfyUI uses `euler` + `simple` with the model's shift, ours uses the diffusers
FlowMatch Euler schedule with dynamic shift and its own RNG. This benchmark does
not measure quality.

## What ComfyUI does that we don't

- **Prefix KV cache** (`QwenImage21Cache`, on by default). The text rows (and
  any reference-image rows) go through the DiT once. Each step then runs only
  the 4096 target rows against the cached prefix K/V. For t2i the prefix is
  ~21 tokens, so the saving is <1%. For edit with up to 10 reference images it
  is large. We recompute all rows every step.
- **DynamicVRAM**: weights stream from pinned host RAM, so the full bf16
  pipeline (31 GB) runs on 24 GB. We need the INT8 DiT and the Q8_0 text
  encoder to stay resident.
- **More features**: graph caching (unchanged nodes do not re-run), LoRA on
  INT8 weights (requantized), ControlNet, the prompt enhancer, edit and RGBA
  workflows, the turbo checkpoint, a UI and many other models. We do t2i only.

## What we do that ComfyUI doesn't

- **6% faster denoise** with the same recipe (INT8 ConvRot GEMMs, INT8/FP8
  attention). Our parts: CUTLASS INT8 GEMMs with the dequant in the epilogue,
  SageAttention2 with RoPE+RMSNorm+quant fused into one kernel, and one fused
  kernel each for SwiGLU+rotate+quant and gated residual+LayerNorm.
- **~8–15× faster text encode** per prompt (25 ms vs 195–370 ms).
- One static binary: no Python, no server. Same seed, byte-identical image.

ComfyUI wins on: VRAM in its default mode (16.3 vs 19.2 GB), the prefix
cache, and features. (It won the VAE decode, 0.20 vs 0.27 s, until
`-vae-nhwc`: ours is now 0.13 s.)

### VAE decode (same 4090, 1024² latent, bf16, after `-vae-nhwc`)

ComfyUI's decoder called directly (`first_stage_model.decode` on a resident
GPU latent, 5 runs) against `vae-decode --iters 4`, both steady state:

| | wall | GPU busy | conv | layout transforms | elementwise / other |
|---|---|---|---|---|---|
| ComfyUI (PyTorch 2, cuDNN 9.24) | 203 ms | 198 ms | 131 ms | 17 ms (cuDNN NCHW↔NHWC) | ~48 ms (F.normalize, SiLU, adds, upsample) |
| qwen-image-rs, NHWC | **133 ms** | 132 ms | 110 ms | 0 | ~22 ms (fused norm, fused bias epilogues, upsample, attention) |

ComfyUI does **not** run channels-last: its Qwen-Image-2.1 VAE
(`comfy/ldm/wan/vae2_2.py`) calls `torch.cudnn_convolution` on NCHW tensors,
so cuDNN wraps each conv in transform kernels, and runs each ResidualBlock in
spatial strips (`strip_apply`, halo 2) to bound memory (1.98 GB peak
allocated). Its edge was PyTorch's caching allocator: no time lost mapping
memory. Ours lost ~50 ms per decode to `cuMemAllocAsync` until the CUDA pool
was told to keep freed memory.

## Method

1. `bench/comfyui/setup.sh` installs ComfyUI, its venv, SageAttention and the
   Comfy-Org weights (into the shared HF cache).
2. `bench/comfyui/run_all.sh` runs ours first, then each variant on a
   **fresh ComfyUI server**. The server is killed after each run, with a 20 s
   pause between runs. `nvidia-smi` is sampled every 250 ms. Suites 2 and 3
   repeated ours and the INT8+sage+fast config: both matched suite 1 within
   0.3%.
3. `bench.py` builds the official `image_qwen_image_2_1_t2i` template's graph
   in API format (`bench/comfyui/workflows/`), with the subgraph flattened and
   the prompt enhancer off (the template default). It queues all 8 prompts
   and times them from websocket events. ComfyUI's `/history` timestamps
   agree within 10 ms.
4. Text encoder: bf16 for A, INT8 ConvRot (the template default) for all
   other variants. VAE: bf16 for all.

## Caveats

- **cfg**: the template defaults to **cfg 1** with an empty negative prompt,
  so ComfyUI already skips the unconditional pass. No variant here doubles the
  DiT forwards. A user who raises cfg above 1 pays ~2× per step in ComfyUI.
  Ours has no CFG path (guidance 1 only).
- The two sides use different samplers and RNG, so the images differ. The
  work per step is the same: one DiT forward over 4096 image tokens.
- The template's default step count is 25, not 40. At 25 steps, the best
  ComfyUI config would take ≈4.6 s per image and ours ≈4.0 s.
- The text encoders differ: ComfyUI's is the INT8 ConvRot Qwen3-VL-8B, ours is
  a Q8_0 GGUF of the same model.
- ComfyUI's "per image" includes PNG encode and save plus its executor
  overhead (~10–20 ms). Ours includes the PNG write too.
- `fp8_e4m3fn` here is a plain cast on load. No scaled-fp8 ComfyUI file exists
  for 2.1.
