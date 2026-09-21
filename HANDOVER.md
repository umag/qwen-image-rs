# qwen-image-rs — session handover

Rust/candle inference engine for **Qwen-Image-2.1** on the RTX 4090. Built from
scratch, reverse-engineered from diffusers, every component validated vs the
reference. ~36 commits. This doc = pick-up point for a fresh session.

## Where things run
- **Repo (source of truth):** `/Users/mag1/dev_tmp/qwen-image-rs` on the Mac (git).
- **GPU host:** WSL 4090, reached ONLY via the swamp ssh models — `wsl` and
  `wsl-drills` (`swamp model method run wsl-drills exec --input '{...}'`), never raw
  ssh. Read stdout: `swamp data get <model> run-exec-wsl --json` then extract the
  `stdout` field (result table truncates). CUDA 13.3, nvcc, latest stable Rust.
- **Build pattern:** edit on Mac → `git archive HEAD -o /tmp/qwen-image-rs.tar` →
  `wsl-drills copy` to host → extract to `~/dev_tmp/qwen-image-rs` →
  `cargo build --release --features <...>` with `CARGO_TARGET_DIR=~/.cache/qwen-image-rs-target`.
  Mac does CPU-only `scripts/check.sh` (candle needs no CUDA on Mac for the default build).
- **Model weights:** on host, HF cache `~/dev_tmp/weights/hf`; full snapshot
  `models--Qwen--Qwen-Image-2.1/snapshots/b3179ad3.../` (has processor/, text_encoder/,
  transformer/, vae/). NOT gated. Oracle references + dumps in `~/dev_tmp/oracle_out/`.
- **CUTLASS:** fresh clone at `~/dev_tmp/cutlass` (host). The flash-attn-vendored
  checkout has a `matrix.h` bug CUDA 13.3 rejects — DO NOT use it. `CUTLASS_DIR=~/dev_tmp/cutlass/include`.

## Status: DONE + validated
Pipeline works end-to-end: `generate --model <snapshot> --prompt "..." --out x.png`
(~48 s/image, 1024², 40 steps). Components vs oracle: VAE 51–53 dB · text encoder
cos 0.9993 · DiT cos 0.99996 · full image PSNR 30.85 dB.

CLI verbs (all in `src/main.rs`): `generate`, `batch`, `denoise`, `dit-forward`,
`text-encode`, `vae-decode`, `smoke`, `bench`, `convrot-test`.

Optimizations, each measured:
- **FlashAttention-2** (`--features flash-attn`): 1.85× denoise (1.15→0.62 s/step),
  MORE accurate. Split: image queries → full flash, text prefix → causal flash
  (= block-causal for t2i, no S² matrix). candle-flash-attn 0.11, ~19 min first build.
- **Q8_0 GGUF** (`--quant`): ½ DiT VRAM, near-lossless. NOTE: quantize-ON-LOAD costs
  ~62 s → it's a VRAM tool, NOT speed. Load-speed needs a pre-quantized GGUF file.
- **ConvRot W8A8 INT8** (`--features convrot`): see below.

## ConvRot INT8 — WIRED INTO THE DiT + validated end to end (commit 3ff66f6)
`QLinear` gained a feature-gated `Convrot(ConvRotLinear)` variant; a runtime
`convrot: bool` threads `QwenImageDit::load → Block → Attention/SwiGlu` (mirrors
`quant`). When set and in-features % 256 == 0, attn q/k/v/o + SwiGlu proj/gate/out
run rotated INT8; norm_out/proj_out/modulation/img_in/txt_in stay bf16 (they're
plain `Linear`, untouched). The 256×256 Regular Hadamard is built once per load,
shared by handle clone. `--convrot` added to generate/batch/denoise/dit-forward.
- **dit-forward --convrot vs oracle: overall cosine 0.999945** (per-token image
  mean 0.99995) — matches the bf16 baseline's 0.99996. Wiring is numerically exact.
- **Full generate --convrot: coherent, on-prompt image.** vs the bf16 image:
  PSNR 20.8 dB — LOW only because of 40-step flow-match trajectory divergence in
  fine detail (mug interior lighting, coffee level); both images are clean red
  mugs, same composition. Per-step accuracy is the 0.9999 above.
- **Speed (naive-attn path, 40 steps @1024²): convrot ~40.8s vs bf16 ~44.7s,
  ~1.02 vs 1.12 s/step (~1.1×).** The isolated MLP win is 1.59×; it's diluted
  here because naive S² attention (unchanged) dominates each step.
- **`--features convrot,flash-attn` TOGETHER (DONE, builds in ~1m with flash
  cached): convrot denoise ~19.8s vs bf16 ~24.2s, 0.51 vs 0.62 s/step (~1.22×)**
  — the win grows once flash makes attention cheap, as predicted. AND the image
  is far closer: **bf16-vs-convrot PSNR 38.2 dB** (naive-attn was only 20.8),
  because flash's stable attention keeps the INT8-perturbed 40-step trajectory
  near the bf16 one. dit-forward cosine 0.999942 (unchanged). This is the
  recommended fast path.

### Pre-quantized convrot weights (commit adds `prequantize-convrot`)
`prequantize-convrot --weights <transformer> --out <file.safetensors>` rotates+
INT8-quantizes the 224 convrot linears ONCE, writing `<prefix>.weight_i8` (U8) +
`<prefix>.col_scale` (f32) beside the 73 bf16 tensors. `QwenImageDit::load`
auto-detects `weight_i8` (`VarBuilder::contains_tensor`/`get_unchecked_dtype`,
native dtype, no cast) → `ConvRotLinear::from_prequantized`, skipping the load-
time rotate+quant. No new flag; point a transformer dir at the file and use
`--convrot`.
- **Bit-exact:** prequant vs on-the-fly convrot cosine 1.000000, MSE 0.0;
  prequant vs oracle 0.999942 (identical to on-the-fly).
- **File 6.8 GB vs ~15 GB bf16 (~55% smaller);** rotate+quant eliminated at load.
- Warm load ties (~3.5s both, cheap compute + hot cache) — the win is cold-start
  I/O (halved), VRAM, and repeated loads (serve).

### SageAttention INT8-QK / FP16-PV (STARTED — kernel validated, integrated)
Vendored thu-ml/SageAttention's sm80 fused kernel into `kernels/sage/vendor/`
(torch stripped) + `kernels/sage/sage_ffi.cu` raw-pointer launchers (head_dim
128, per-BLOCK INT8 scales, float SV accum, bf16 out) + INT8 quant via their
`QuantInt8Kernel`. `src/sage.rs` bridges via candle CustomOp3 (q,k bf16 + v f16
→ o bf16). `sage` feature, `sage-test` verb.
- **Kernel builds torch-free under CUDA 13.3 (~1 min)** — needed `#include
  <cassert>` before cuda_fp8/fp6/fp4 headers (CUDA 13 `__assert_fail`).
- **Self-test: INT8 attention cosine 0.99991 (non-causal) / 0.99994 (causal)**
  vs f32 softmax reference.
- Wired into DiT `attend()` (sage feature takes precedence over flash-attn):
  causal SageAttention over the text prefix, non-causal over image queries —
  same block-causal split as the flash path. Benchmark build: `--features
  convrot,sage` (INT8 linears + INT8 attention).
- **Measured (`convrot,sage`, 40 steps @1024²): denoise ~19.4s / 0.48 s/step;
  dit-forward cosine 0.999924 vs oracle; image PSNR 34.7 dB vs bf16.** vs
  `convrot,flash` 19.8s / 0.51 s/step → only ~2-4% faster.
- **Why marginal:** the `bench` verb (seq 4117, bf16) shows attention S² 12.1ms
  > mlp 8.4ms > attn_proj 3.6ms — attention-bound *before* flash. But once
  flash/sage make attention cheap, the GEMMs (mlp+proj ≈ 12ms) dominate, and
  those are convrot's target. So convrot is the big denoise lever here and
  sage-over-flash is small AT THIS SHAPE. Sage will pay off more on
  attention-heavier configs (condition images, higher resolution → larger S²).
- Per-BLOCK granularity (not per-warp) so quant scale counts match the kernel's
  per-block indexing without padding. Per-warp is a later accuracy refinement
  (needs the padded ceil(N/128)*4 scale layout + their per-warp quant kernel).

### Future levers
- **Quantize the TEXT ENCODER too (resident-VRAM mode).** Today `generate`
  loads Qwen3-VL 8B (bf16 ~16 GB) → frees → DiT → frees → VAE, reloading each
  per run. Quantize the text encoder (Q8_0/INT8) so text-enc + DiT (convrot ~7
  GB) + VAE all fit in 24 GB **resident at once** — no reload between images,
  big win for batch/serve. Mirrors the `--quant`/prequant work on the DiT;
  apply Q8_0 (candle QMatMul) to the Qwen3 decoder linears in
  `src/model/text_encoder.rs`, add a resident pipeline that loads all three once.
- **CUDA-graph capture** — candle 0.11 exposes no stream-capture hook and its
  op path allocates fresh tensors per step (pointers move), so replay is fragile.
  Not viable without patching candle. Deprioritized.
- **SageAttention per-warp / fused v-scale** — accuracy/speed refinement over the
  per-block first cut above.
- Build: `cargo build --release --features convrot` (CUTLASS_DIR set), ~1m53s.
  Runtime compare needs the oracle venv python (`~/dev_tmp/qwen-image-oracle/.venv/bin/python`
  has safetensors/PIL/numpy; the system python3 does not).
- `scripts/host.sh` fixed: `--repo-dir` is a swamp SUBCOMMAND option, not a global
  flag — it now follows `model method run` / `data get`, not `swamp` itself.

## ConvRot INT8 (earlier) — validated + fast, DiT wiring now DONE (above)
Method = pytorch/ao#4695 Group-wise Regular Hadamard Rotation for DiTs. Reference
kernel: sglang#38040 (vendored `kernels/convrot/convrot_int8_gemm.cu`, but it's
torch+sgl-extension coupled — we used a SIMPLER stock-CUTLASS path instead).

Built + PROVEN on the 4090 (`convrot-test`):
- `kernels/convrot/int8_gemm.cu` — stock-CUTLASS 2.x int8 GEMM (sm89, mma.sync
  16×8×32, int32 accum). Bit-exact vs CPU.
- `kernels/convrot/quant_ops.cu` — per-row bf16→int8 quantize + i32→bf16 dequant
  (route around candle's missing i32→f32 cast; bf16→f32 DOES exist).
- `src/model/rotation.rs` — 256×256 orthonormal Regular Hadamard `(H4/2)^⊗4`,
  `rotate`/`fold_weight`. Unit-tested (R Rᵀ=I, output-preserving).
- `src/convrot.rs` — `int8_gemm` (CustomOp2), `QuantizeRows` (CustomOp2), `Dequant`
  (CustomOp3), `ConvRotLinear` (rotate→quant→int8 GEMM→dequant). Bridge pattern:
  `storage.as_cuda_slice::<T>()` + `dev.cuda_stream()` + `slice.device_ptr(&stream)`
  + `dev.alloc` + `CudaStorage::wrap_cuda_slice` (copied from candle-flash-attn).
  `build.rs` compiles the .cu via `cc.cuda(true)` + `CUTLASS_DIR`.
- **Results:** full ConvRot linear cosine **0.99993** vs bf16; **1.59× faster** at
  the MLP shape (4117×4096@12288: 1.65 vs 2.62 ms, incl. rotate+quant+dequant).

### THE NEXT STEP (mechanical, no unknowns)
Wire `ConvRotLinear` into the DiT's 32 blocks and validate end to end:
1. In `src/model/dit.rs`, add a ConvRot variant to `QLinear`
   (`#[cfg(feature="convrot")] Convrot(crate::convrot::ConvRotLinear)`), built from
   the bf16 weight + `regular_hadamard_256` at load. Thread a runtime `convrot: bool`
   (like the existing `quant` flag) through `QwenImageDit::load` → `Block` →
   `Attention`/`SwiGlu`. Start with the **SwiGlu** (biggest weights); optionally the
   attn q/k/v/o. Mixed precision: keep norm_out/proj_out/modulation bf16.
2. Add `--convrot` to `dit-forward`/`denoise`/`generate` (thread to `QwenImageDit::load`).
3. Validate: `dit-forward --convrot` vs the dumped oracle output (`~/dev_tmp/oracle_out/dit_io.safetensors`)
   using `/tmp/compare_dit.py` — expect cosine ~0.999x. Then a full `generate --convrot`
   image + PSNR vs bf16 (target the paper's ~29 dB). Bench the step time.
Caveat: K must be a multiple of 256 for the rotation (MLP: 4096, 12288 both OK;
attn: 4096 OK). ConvRotLinear input dim K = the linear's in_features.

## Gotchas banked (don't re-learn)
- candle DType has I16/I32/I64 but **no I8** → int8 rides in **U8** tensors (same bytes).
- candle's cuda build here lacks the **I32→F32 cast** ("named symbol not found") →
  dequant is a custom kernel; host-compare in tests.
- device_ptr guards borrow the slice → **scope them** before `wrap_cuda_slice` moves it.
- swamp exec JSON: avoid embedded `"` in the command (breaks the JSON) — write a
  script file + copy it, or keep commands quote-free.
- CUDA 13.3 is strict: fresh CUTLASS only; flash-attn compiles fine (just ~19 min).
- Two model snapshots exist (b3179ad3 full, 790c9263 partial) — glob carefully.

## Issue-lifecycle issues (state in swamp)
`qwen-image-rs-oracle` (complete), `-vae` (complete), `-text-encoder` (complete),
`-dit` (complete), `-convrot` (planned — the DiT-wiring work above goes under it).
Resume any: `swamp model method run <issue> hydrate`.

## Also-planned / future levers
Pre-quantized GGUF *file* (cut ~24 s load); SageAttention INT8 attention; prefix
KV cache (helps most with condition images); resident serve mode.

Memory: `[[project_qwen_image_rs]]`, `[[lesson_qwen_image_rs_attention_bound_profile]]`.
