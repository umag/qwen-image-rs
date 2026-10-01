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

## Status: DONE + heavily optimized (all accuracy-neutral vs oracle)
End-to-end works: `generate --model <snapshot> --prompt "..." --out x.png`.
**Fastest build: `--features convrot,sage,fusednorm,sage2`** (SageAttention2,
dit-forward 0.999911 since `-fused-swiglu` (0.999879 after `-fused-rotate-quant`, chaos) — FP8 P·V;
denoise 0.1746 s/step after `-fused-swiglu`; `QIR_SAGE=1` in that binary = SageAttention v1,
bit-identical to the build without `sage2`). **Most accurate fast build: `--features
convrot,sage,fusednorm`** (0.999944). **DEFAULT (human decision 2026-10-01): the
SA2 build** — accepted at 0.999894 for -4.8% denoise. (bf16 VAE is the
CUDA default; `--convrot --text-gguf <gguf> --vae-tile 32`). Oracle parity held
at **dit-forward cos 0.999935** through every optimization; VAE decode 53–55 dB.

### Speed today (1024², RTX 4090, measured 2026-09-21)
- **Denoise: 0.247 s/step** (was bf16+flash 0.62) — **~2.5×**. VAE tiled bf16
  decode **1.19 s** (was f32 1.84).
- **Single `generate` (loads all 3 models each run): ~25 s** (40 steps); the
  ~14 s beyond compute is one-time model load (8 GB text GGUF mmap + DiT + VAE).
- **`batch --resident` (models loaded once): ~11.1 s/image @40 steps** (enc
  ~25 ms + denoise 40×0.247 s + tiled decode ~1.19 s). Steady, no VRAM drift.

### The optimization scoreboard (all via issue-lifecycle, all oracle-neutral)
| lever | feature | denoise s/step |
|---|---|---|
| bf16 + FlashAttention-2 (start) | flash-attn | 0.62 |
| ConvRot W8A8 INT8 linears | convrot | 0.51 |
| SageAttention INT8-QK/FP16-PV | sage | 0.48 |
| fused LayerNorm+AdaLN | fusednorm | 0.445 |
| fused activation quantizer | convrot | 0.38 |
| dequant→CUTLASS-EVT epilogue | convrot | 0.34 |
| fused RMSNorm×weight + gated residual | fusednorm | 0.27 |
| BSHD-native attention (no transpose copies) | sage | 0.247 |
| RoPE fused into the INT8 Q/K quantizer | sage | 0.2295 (A/B same session: 0.2336 → 0.2295, −1.8%) |
| V born f16 in the ConvRot epilogue (no V cast) + sage partial-tile zero-fill | convrot,sage | 0.2266 (A/B same session: 0.2292 → 0.2266, −1.1%) |
| SageAttention2 sm89: INT8-QK per-thread + K smoothing, FP8-PV (fp32+fp16 accum) | sage2 | 0.2149 (A/B same binary/session, `QIR_SAGE=1` vs `2`: 0.2259 → 0.2149, −4.8%; dit-forward --convrot 0.999944 → 0.999894) |
| every DiT linear a `QLinear`, tail linears via ConvRot (uniformity) | convrot | 0.2150 (flat, as predicted; dit-forward --convrot 0.999894 → 0.999898) |
| SA2 quant fused: 10 per-layer launches → 3 (one 512-thread kernel, L2-ordered), bit-identical | sage2 | 0.2127 (A/B same session: 0.2159 → 0.2127, −1.5%) |
| Hadamard rotation fused into the activation quantizer (f32 in registers; no bf16 rotation GEMM) | convrot | 0.1911 (A/B same session: 0.2125 → 0.1911, −10.1%; dit-forward --convrot 0.999898 → 0.999879, chaos — see section) |
| SwiGLU `silu(g)·p` fused into the MLP-out rotate+quantize (f32; no bf16 h, no usilu/bmul passes) | convrot | **0.1746** (A/B same session: 0.1905 → 0.1746, −8.4%; dit-forward --convrot SA2 0.999879 → 0.999911, v1 0.999900 → 0.999881, chaos) |

Plus (not per-step): bf16 VAE decode 1.57×; text encoder Q8_0 GGUF (resident
VRAM); VAE tiling (constant decode memory); true CFG (`--guidance`/`--negative`,
2× denoise when >1).

### Batched multi-seed generation (DONE — `qwen-image-rs-batch-seeds`)
`generate --batch N` renders one prompt as N images with N different seeds
(`seed, seed+1, …`) in ONE batched DiT denoise — the SDXL "batch count" grid.
Encode once → free TE → broadcast the embed to `(N,txt,4096)` → N seed-noises
(lane i = `set_seed(seed+i)`, RNG reset per lane) → batched denoise (`forward` /
`guided_noise_pred` operate at any B; `forward` broadcasts the per-token
modulation to B, guarded so B=1 is byte-identical) → load VAE once, loop tiled
decode → `--out-dir`/`{i:03}.png` (B=1 keeps `--out`). `--emit-latents` saves
each lane's latent. All batch kernels are per-lane (sage via `stride_bz`, rope
grid `B*S*H`, convrot/fusednorm CTA-per-row).
- **Sweet spot B=4** (spike: DiT-only peak 14.2 GB, throughput-neutral vs
  sequential; knee at B=5 — B≥5 pays a growing per-image tax, B=8 +72 %). It's
  ergonomic, not faster: the GEMMs already saturate the 4090, so 8 images = ~2×
  the time of 4, same as sequential.
- **B=1 is bit-reproducible and matches lane 0 of a batch** (measured after
  `qwen-image-rs-b1-off-prompt`, convrot,sage,fusednorm): same-seed B=1 twice
  = cosine **1.0000000, mse 0**; B=1 vs B=2 lane 0 latent cosine **0.99997**
  (batch-size GEMM tiling differs, so not bit-exact). The older notes here
  ("same-seed B=1 twice = 0.981", "a lane does NOT reproduce a single run",
  "B=1 with convrot lands off-prompt at some seeds — non-determinism
  artifact") were TWO BUGS, not chaos: the sage partial-tile stale-smem NaN
  (fixed in `-bf16-v-pv`) and the B=1 off-prompt offset bug (see Gotchas).
  `scripts/compare_latent.py` (thr 0.99) gates lane self-consistency.

CLI verbs (all in `src/main.rs`): `generate`, `batch` (has `--resident`,
`--vae-tile`, `--guidance`/`--negative`, `--text-gguf`, `--quant-text`),
`denoise`, `dit-forward`, `text-encode`, `vae-decode` (`--bf16`), `smoke`,
`bench`, `convrot-test`, `sage-test`, `fusednorm-test`, `prequantize-convrot`,
`prequantize-text`.

### NEXT SESSION — remaining optimization backlog (ranked; latest nsys below)
The denoise is now a **flat tail — no dominant kernel** (10-step trace, bf16 VAE):
`im2col_bf16` 10.1% (VAE convs) · `ucopy_bf16` 8.4% (attention-layout copies) ·
sage attn 8.0% · bf16 GEMMs ~11% (non-convrot linears + VAE) · our 4 fused
kernels ~11% total (each 1.8–3.6%, efficient) · silu 3% · one-time load
weight-quant ~6.6% (224-inst kernels — NOT per-step). Remaining levers, by
impact/effort:
1. ~~**BSHD-native fused attention**~~ **DONE** (`qwen-image-rs-bshd-attention`,
   commit a081ad5). See "BSHD-native fused attention" below — denoise
   0.27→0.247 s/step (~8.5%), oracle 0.999934 unchanged, sage BSHD vs BHSD
   bit-exact. Follow-on Q/K-quant-into-rope fusion **DONE**
   (`qwen-image-rs-rope-quant-fusion`, −1.8%). The V f16 cast is gone too
   (`qwen-image-rs-bf16-v-pv`, −1.1%: to_v's ConvRot epilogue stores f16).
2. **VAE conv algorithm** (`im2col_bf16` ~10%). bf16 already; further needs
   implicit-GEMM / Winograd convs or a fused decoder — large.
3. ~~**Per-warp sage quant**~~ **DONE as SageAttention2** (`-sageattention2`,
   per-thread INT8 + K smoothing + FP8 P·V, −4.8%; its 5 quant passes cost
   ~4.7 ms/step — cut by `-sage2-quant-fusion`, −1.5%). **SageAttention on
   attention-heavier configs** (condition images / higher res) where S² matters
   more — sage's win grows there.
4. ~~**Convrot the remaining ≥256-dim bf16 linears**~~ **NEGATIVE — not worth it**
   (`qwen-image-rs-convrot-tail-linears`): they are 0.13% of a step; see
   "Tail linears through ConvRot" below.
5. ~~Pre-quantized convrot DiT as default~~ **DONE** (`-prequant-default`: cached
   file, DiT load cold 7.0 → 2.7 s); the residual is the 8 GB text-GGUF cold mmap.
Re-profile after each: `nsys profile -o /tmp/p --trace=cuda <bin> generate ...
--steps 10 ...` then `nsys stats --report cuda_gpu_kern_sum --format table
/tmp/p.nsys-rep`. Diminishing returns — each remaining lever is substantial work
for a single-digit-% or structural slice.

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
run rotated INT8. (Since `-tail-linears-unify` every DiT linear is a `QLinear`
and the tail linears rotate too, except img_in — see "Tail linears through
ConvRot".) The 256×256 Regular Hadamard is built once per load,
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

### Rotation fused into the ConvRot activation quantizer (DONE — `qwen-image-rs-fused-rotate-quant`)
Every ConvRot linear used to rotate its activation with a bf16 GEMM
(`ampere_bf16_s1688gemm_bf16_128x128..nn`, one per INT8 linear, 195.5 ms per
10-step generate) and write a bf16 `x R` that `quantize_rows_fused_k` re-read
(128.2 ms). `rotate_quantize_rows_k<CPW>` (kernels/convrot/quant_ops.cu,
`RotateQuantizeFused` in src/convrot.rs) does both: CTA per row, 8 warps, a
warp per 256-wide chunk (lane = 8 contiguous elements, one 16-B load, CPW
chunks per warp kept in f32 registers, K <= 16384), the Regular Hadamard as 4
radix-4 stages — R256 = (H4/2)^⊗4 acts per base-4 digit as `y_a = Σx/2 − x_a`
(digit 0 in-register, digit 1 = e bit 2 + lane xor 1, digits 2/3 = lane xor
2,4 / 8,16) — then row amax, scale = amax/127, `__float2int_rn`, 8-B int8
stores. Weights unchanged (fold_weight at load), so `PREQUANT_FORMAT` and the
convrot cache stay valid. The old rotate + quantize path is
`ConvRotLinear::forward_as_unfused`, kept only as the self-test oracle.
- **Self-test (`convrot-test`):** fused vs an f64-accumulated host reference
  (row 0 zero, row 1 with 60× outliers): **0 int8 differences** at (M,K) =
  (37,256), (300,4096), (64,12288), (5,16384), scale rel ≤ 1.1e-7. vs the old
  bf16-GEMM path: int8 within ±1 on 4–6% of elements (the old path's bf16
  rounding of `x R`), scale rel ≤ 3.8e-3 (bf16). Linear fused vs unfused
  cosine 0.99998 / 0.99997. Rejects K=128/384/16640; M=0 → empty. Offset view
  bit-identical. Activation path alone (M=4117): K=4096 0.133 → 0.027 ms,
  K=12288 0.495 → 0.199 ms.
- **Oracle (dit-forward --convrot vs dit_io):** default SA2 0.999898 →
  **0.999879**; but `QIR_SAGE=1` 0.999867 → 0.999900 and `QIR_SAGE=2f32`
  0.999832 → 0.999868 (mean of the three +1.7e-5). The per-op math is MORE
  exact (matches f64), so the −1.9e-5 on the default config is the known
  chaotic response of 32 blocks of INT8/FP8 rounding to any ~1e-5 input
  change (see "Tail linears"), not an error. vs own no-convrot output
  0.999912 → 0.999907. No-convrot byte-identical; run-to-run byte-identical.
- **Images:** B=1 mug and `--batch 2` clean, on-prompt; B=1 == lane 0
  (latent cos 1.0000000, PNG identical); vs prior B=1 latent cos 0.99993.
- **Speed:** denoise A/B (batch --resident, 2nd image, two interleaved
  rounds): 8500 / 8497 → 7644 / 7651 ms = **0.2125 → 0.1911 s/step (−10.1%)**.
  nsys 10-step generate: rotation GEMM gone; `rotate_quantize_rows_k` 119.5 ms
  total (<2> 1980× 35 us, <6> 320× 154 us ≈ 1 TB/s) vs 323.7 ms for
  GEMM + quant before.

### SwiGLU fused into the MLP-out quantizer (DONE — `qwen-image-rs-fused-swiglu`)
`SwiGlu::forward` used to run candle `usilu_bf16` + `bmul_bf16` over `(M, 12288)`
and materialize a bf16 `h` that the out linear's `rotate_quantize_rows_k<6>` re-read
(per block ~657 MB of traffic). Now `rotate_quantize_rows_k<CPW, Loader>`
(kernels/convrot/quant_ops.cu) is templated on a row loader: `PlainRow` (as before,
byte-identical output) or `SwiGluRow` (gate and proj rows through independent row
strides, `silu(g) = g/(1+expf(-g))` times `p` in f32, then the same R256 + amax +
int8). `ConvRotLinear::forward_swiglu(g, p, out)` (src/convrot.rs,
`SwiGluRotateQuantize` CustomOp3) feeds the usual GEMM + epilogue tail
(`gemm_dequant`); `SwiGlu::forward` takes it when `out` is ConvRot, else the
unchanged candle ops. g/p may be **row-strided views** (`layout::row_strided_2d`:
unit inner stride, `ld >= K`, `ld % 8 == 0`) — two tensors, or the two column halves
of one merged `(M, 2N)` gate|proj output (for `-gemm-merge-tune`). Weights and the
prequant cache untouched (no `PREQUANT_FORMAT` bump).
- **Self-test (`convrot-test`):** vs an f64 host reference (silu·p and R in f64):
  max |dq| 1 on 0–4 int8s per shape at (M,K) = (37,256), (300,4096), (64,12288),
  (5,16384), scale rel ≤ 1.9e-7; vs the candle bf16 silu·mul + fused rotate+quant:
  ±1 on 5–10% (the bf16 roundings of silu(g) and of h), scale rel ≤ 2.7e-3. Merged
  `(drop+M, 2K)` column halves (row offset, ld = 2K) vs dense copies: bit-identical.
  Rejects K=128, ld % 8 != 0, g/p shape mismatch; M=0 → empty. Linear
  `forward_swiglu` vs `forward(silu(g)*p)` cosine 0.99996 (3-D and merged inputs).
  Activation path (M=4117, K=12288): **0.913 → 0.327 ms**.
- **Oracle (dit-forward --convrot vs dit_io):** SA2 0.999879 → **0.999911**,
  `QIR_SAGE=1` 0.999900 → 0.999881, `2f32` 0.999868 → 0.999890 (mixed signs, mean
  +1.2e-5: the chaos pattern; per-op math is closer to f64). vs own no-convrot
  0.999907 → 0.999931; vs the prior build 0.999942 / 0.999940 / 0.999950.
  No-convrot byte-identical; run-to-run byte-identical.
- **Images:** B=1 mug + `--batch 2` clean, on-prompt; B=1 == lane 0 (latent cos
  1.0000000, PNG identical); vs prior B=1 latent cos 0.99992.
- **Speed:** denoise A/B (batch --resident, 2nd image, two interleaved rounds):
  7611 / 7628 → 6980 / 6987 ms = **0.1905 → 0.1746 s/step (−8.4%)**. nsys 10-step
  generate: `usilu_bf16` 701 → 381 inst, `bmul_bf16` 680 → 360 (the rest are VAE /
  text), `rotate_quantize_rows_k<6>` replaced by the SwiGlu instantiation 320 × 287 us
  (202 MB in + 50 MB out ≈ 0.88 TB/s).
- Re-profile (HEAD before this change): INT8 EVT GEMMs 100 ms/step = ~575 TOPS
  effective, near the 4090's INT8 dense peak, so GEMM tile tuning has little room.
  Next visible lever: `fused_rmsnorm_scale_kernel` for the q/k head norms, 640 inst ×
  199 us ≈ 13 ms/step at only ~340 GB/s (N=128 rows, CTA-per-row).

### Prequantized ConvRot DiT = the DEFAULT `--convrot` load (DONE — `qwen-image-rs-prequant-default`)
`src/convrot_cache.rs`: every `--convrot` DiT load (generate, batch both paths,
denoise, dit-forward) goes through `resolve_dit_files`, which returns the cache
entry `<root>/<snapshot>-<fnv(canonical dir)>/transformer_convrot.safetensors`
(root: `--convrot-cache` > `$QIR_CONVROT_CACHE` > `$XDG_CACHE_HOME/qwen-image-rs/convrot`
> `~/.cache/qwen-image-rs/convrot`), building it first when missing/stale with
the SAME builder as `prequantize-convrot` (`convrot_cache::build`, streams the
bf16 sources via mmap, `ConvRotLinear::from_weight`, atomic tmp+fsync+rename).
- **Validity** = safetensors `__metadata__` `qir.policy` == `dit::convrot_policy_tag()`
  (`PREQUANT_FORMAT` + GROUP + BLOCK_LINEARS + TAIL_LINEARS with rot flags — bump
  `PREQUANT_FORMAT` if `from_weight`'s math/layout changes) AND `qir.source` ==
  FNV of (canonical dir, file names, sizes, mtimes) AND header-implied length ==
  file length (torn-file check). Else rebuild in place. Any build failure (no CUDA,
  read-only, ENOSPC) warns and falls back to on-load quant (same output).
- Flags: `--no-convrot-cache` (old path), `--rebuild-convrot-cache`, `--convrot-cache DIR`.
  A `--weights` dir already holding `*.weight_i8` is used as-is.
  `prequantize-convrot --weights <transformer>` without `--out` pre-warms the
  cache entry (`scripts/convert.sh` does that).
- **Bit-identical:** dit-forward --convrot on build-run, cache hit, rebuild,
  `--no-convrot-cache` and an explicit `--out` file: `cmp`-identical to the
  prior build; no-convrot identical; B=1 and `--batch 2` PNGs identical to the
  prior build, B=1 == lane 0.
- **Load (dit-forward --convrot, same session, `drop_caches` for cold):**
  cold 7.00 → **2.73 s**, warm 2.08 → **1.12 s**. One-time build 25.9 s,
  entry 7.12 GB. nsys 10-step generate: `uabs_bf16` and `quantize_rows_i8_k`
  gone, `fast_max_bf16` 240 → 9 (the rest are not weight quant). Denoise flat
  (0.2122 vs 0.2126 s/step, noise).
- Not byte-reproducible FILES: the header's `__metadata__` is a HashMap (safetensors
  serializes it in hash order), so two builds can differ in header bytes; the
  tensor data are identical (outputs above are).

### Pre-quantized convrot weights (commit adds `prequantize-convrot`)
`prequantize-convrot --weights <transformer> --out <file.safetensors>` rotates+
INT8-quantizes the convrot linears ONCE (231 since `-tail-linears-unify`: 224
block + 7 tail; the set is `dit::is_convrot_target`, the loader's own policy),
writing `<prefix>.weight_i8` (U8) + `<prefix>.col_scale` (f32) beside the 66
remaining bf16 tensors. Files written before `-tail-linears-unify` still load
(the tail layers find their bf16 `weight` and rotate at load) but regenerate them. `QwenImageDit::load`
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

### Text encoder Q8_0 (DONE — enables resident VRAM)
`src/model/text_encoder.rs` gained a `QLinear` (Full/Q8_0) mirroring the DiT;
`QwenTextEncoder::load(cfg, quant, vb)` quantizes the Qwen3 decoder linears to
Q8_0 (weight-only). Exposed as `--quant-text` (generate/batch) and `--quant`
(text-encode). Token embedding stays bf16.
- **VRAM: text-enc bf16 ~16 GB → Q8_0 ~9 GB** (embed 1.2 GB bf16 + linears ~7.4
  GB Q8). All three resident: ~9 (text Q8) + ~7 (DiT convrot) + ~0.7 (VAE f32)
  ≈ **16.7 GB < 24 GB** (bf16 text would be ~23.7 GB — too tight). So Q8_0 text
  is what makes an all-resident pipeline fit.
- **Quality: Q8 vs bf16 embeds cosine 0.9977** (bf16-vs-oracle was 0.9993; Q8
  error accumulates over 36 layers). End-to-end `--convrot --quant-text` image
  is coherent + on-prompt; PSNR 22.3 dB vs bf16 (the embed shift changes the
  conditioning globally — same subject, different fine detail). A quality knob:
  `--quant-text` for residency, bf16 for max fidelity.
- **Load cost: ~58 s** (quantize-on-load, like the DiT `--quant`) — it's a VRAM
  tool, not a speed one. A pre-quantized text-encoder file (mirror
  `prequantize-convrot`, or a Q8_0 GGUF) would cut that; TODO.

### Resident pipeline + pre-quantized text GGUF (DONE)
- **`batch --resident`**: loads text-enc + DiT + VAE ONCE and streams prompts
  through encode→denoise→decode without freeing. `src/main.rs` batch().
- **`prequantize-text --weights <text_encoder> --out <file.gguf>`** writes a Q8_0
  GGUF (decoder linears Q8_0, embed+norms F16, via `gguf_file::write`; text-only
  tensors); **`--text-gguf <file>`** on generate/batch loads it via
  `QwenTextEncoder::load_gguf` (candle quantized VarBuilder → QMatMul from
  Arc<QTensor>). File 8.1 GB.
- **Why the GGUF is REQUIRED for resident:** quantize-on-load (`--quant-text`)
  materializes bf16 weights on the GPU transiently, which inflates candle's CUDA
  pool to **~24 GB (at the limit)** — a differently-sized 2nd prompt then can't
  allocate and renders BLANK (identical prompts reuse the pooled buffers and are
  fine). `--text-gguf` never puts a bf16 weight on the GPU → **pool ~18.9 GB**,
  ~5.6 GB headroom, every prompt renders. Validated: mug→vase resident, both
  correct (was blank on vase).
- **Measured per-phase (resident, instrumented):** encode **0.2–1.4 s**,
  denoise **20.3 s**, VAE decode **~35 s**. So resident 57 s/image is dominated
  by the VAE DECODE, not the encode (the earlier "~35 s Q8 encode" caveat was a
  MIS-ATTRIBUTION). The decode is slow ONLY in resident: sequential
  `--text-gguf` is 35 s total because it frees te+DiT before decoding, giving
  the VAE full VRAM; resident keeps all three co-resident (~18.9 GB), so the
  VAE's 1024² upsampling feature maps thrash. embed quality: Q8 vs bf16 cosine
  0.9977.
- **So caching bf16 encoder weights is the WRONG fix** (encode is 0.2–1.4 s, ~1%
  of the time) and would add ~7.5 GB → back over 24 GB (blank returns).

### VAE tiled decode (DONE — makes resident the fast path)
`decode_tiled` (src/model/vae.rs) splits the latent into overlapping tile×tile
(latent) windows, decodes each, and feather-blends into a canvas — peak decode
memory scales with tile², not image². `--vae-tile <N>` on generate/batch
(0=off; overlap = tile/4).
- **Seamless: tiled vs non-tiled PSNR 48.3 dB** (maxabs 22), imperceptible.
- **Resident `--vae-tile 32`: decode 35 s → 1.8 s (19×), per-image 57 s → 22 s.**
  Both mug+vase correct. This is now the FASTEST path: resident+gguf+tile
  **22 s/img** < sequential `--text-gguf` 35 s < non-tiled resident 57 s.
- Recommended resident invocation:
  `batch --resident --convrot --text-gguf <gguf> --vae-tile 32 ...`

### CFG + step count (DONE)
- **`--steps` was already a param** (generate/batch/denoise). Denoise is a flat
  ~0.5 s/step: 25 steps ≈ 12.6 s denoise (~14.5 s/image resident+tile), 40 steps
  ≈ 19.9 s (~21.7 s/image).
- **True CFG (`--guidance <scale>` + `--negative <prompt>`)**: `guided_noise_pred`
  in main.rs. guidance ≤ 1 = single forward (fast path, default 1.0); guidance >
  1 encodes the negative once and runs the DiT twice per step, combining
  `v = v_uncond + g·(v_cond − v_uncond)`. Validated: denoise 12.6 s → 24.9 s
  (1.98× at 25 steps) — exactly the expected 2×; images coherent, stronger
  adherence at g=4. The oracle used pipeline defaults (guidance≈1), which is why
  the single-forward port matched it.

### Reference: comfy-kitchen (Comfy-Org, Apache-2.0)
Python/CUDA/HIP diffusion kernel lib: FP8/NVFP4/MXFP8/INT8 quant, attention,
RoPE, **AdaLN/RMS-AdaLN**, GEMM. torch-coupled (QuantizedTensor subclass) so not
drop-in — its .cu kernels would need the same vendor-and-strip-torch-FFI as
SageAttention. Interesting future source for a **fused AdaLN/RMS-AdaLN** kernel
(our DiT does norm+modulation as separate candle ops) or an alt attention.
NVFP4/MXFP8/MXFP4 are Blackwell (sm120) — no use on Ada; our INT8 convrot+sage
already cover the 4090 wins. No model impls, no ConvRot scheme.

### Fused LayerNorm+AdaLN kernel (DONE — `fusednorm` feature)
Driven through the issue-lifecycle (`qwen-image-rs-fused-adaln`). nsys showed
~22% of denoise GPU time in unfused norm/modulation ops + ~11k bf16<->f32 casts.
`kernels/fusednorm/fused_norm.cu` fuses `norm_no_affine(x)*(scale+1)` into one
CTA-per-row kernel (f32 mean/var in shared mem, bf16 in/out) + AdaLN affine;
candle `CustomOp2` bridge in `src/fusednorm.rs`; `norm_mod` helper in dit.rs
wires it at the 2 block norms + norm_out behind the `fusednorm` feature (candle
fallback unchanged). `fusednorm-test` verb.
- **Unit: fused vs candle cosine 0.999996.** Integration: dit-forward vs oracle
  0.999929 (unchanged from convrot+sage 0.999924).
- **Denoise 20.3 s -> 17.8 s (40 steps) = ~12% faster** (0.51 -> 0.445 s/step) on
  the convrot,sage,fusednorm path. Recommended fast build now:
  `--features convrot,sage,fusednorm`.
- Scope: LayerNorm+affine only (the dominant bucket). RMSNorm variants
  (ZeroCenterRmsNorm/HeadRmsNorm) + gated residuals were the follow-on — see
  "Fused RMSNorm×weight + gated residual" below.

### Fused activation quantizer (DONE — extends `convrot`)
Driven through the issue-lifecycle (`qwen-image-rs-fused-actquant`). nsys showed
~18.5% of denoise in the per-forward activation quant (uabs_bf16 5.0% +
fast_max_bf16 5.5% + quantize_rows_i8 2.7%). `quantize_rows_fused_k`
(kernels/convrot/quant_ops.cu) does the CTA-per-row `max(|x|)` reduction +
`scale=amax/127` + int8 quantize in one pass, emitting int8 + the per-row scale.
`QuantizeFused` CustomOp2 (src/convrot.rs) writes `row_scale` in place into a
pre-allocated input (fresh zeros, no aliasing/autograd — safe in inference);
`ConvRotLinear::forward` calls it, dropping candle's abs+max_keepdim+recip.
`from_weight` (load-time) unchanged.
- **convrot-test cosine 0.99994 (int8-gemm exact); dit-forward vs oracle 0.999934
  (unchanged).**
- **Denoise 17.8 s -> 15.2 s (40 steps) = ~14.5% faster** (0.445 -> 0.380 s/step)
  on convrot,sage,fusednorm,+actquant. Cumulative fastest path denoise: bf16
  0.62 -> 0.38 s/step.
- Remaining fusion levers: RMSNorm variants + gated residuals (both now DONE —
  see "Fused RMSNorm×weight + gated residual" below).

### Fused dequant epilogue (DONE — `qwen-image-rs-dequant-epilogue`, extends `convrot`)
Driven through the issue-lifecycle. nsys showed ~7.4% of a 10-step denoise trace
(2240 instances = 224 convrot linears × 10 steps) in the standalone
`dequant_i32_bf16_k` kernel applying `out=i32*s_row[m]*s_col[n]->bf16` after the
INT8 GEMM, plus the i32 intermediate written to and read back from global memory.
Fused the per-token(row) × per-channel(col) dequant INTO the GEMM's CUTLASS
epilogue so the INT8 GEMM emits bf16 directly.
- `kernels/convrot/int8_gemm.cu`: new `int8_gemm_dequant_bf16` built on the
  **CUTLASS 4.8 SM80 Epilogue Visitor Tree (`Sm80EVT`)** — `VisitorAccFetch`
  (i32) → `VisitorCompute<multiplies>` with `VisitorColBroadcast` s_row[m] →
  `VisitorCompute<multiplies>` with `VisitorRowBroadcast` s_col[n] →
  `VisitorAuxStore` bf16 — via `DefaultGemmWithVisitor` + `GemmUniversalAdapter`,
  modeled on `examples/47`. **INT8 needs `arch::OpMultiplyAddSaturate`** (there is
  no plain-`OpMultiplyAdd` s8 16×8×32 mma). The raw `int8_gemm_s32` is kept for
  the bit-exact self-test. `build.rs` gained `--expt-extended-lambda` (EVT
  visitors use device lambdas).
- `quant_ops.cu`: the `dequant_i32_bf16_k` kernel + launcher are removed.
- `src/convrot.rs`: `int8_gemm` + `Dequant` collapse into one `CustomOp3`
  (`Int8GemmDequant`). candle has no `CustomOp4`, so the two scale vectors ride
  in one packed f32 tensor `cat(col_scale, row_scale)` — **col first** so the
  vectorized `RowBroadcast` load of s_col starts at the 32-B-aligned buffer base
  (s_row then sits at offset N, aligned because N is always a mult of 8). Packing
  `row` first mis-aligns s_col at odd M and faults `CUDA_ERROR_MISALIGNED_ADDRESS`.
- **convrot-test cosine 0.99994 (int8-gemm bit-exact, max diff 0); dit-forward vs
  oracle 0.999934 (UNCHANGED). MLP-shape isolated linear 2.68 -> 0.935 ms.**
- **Denoise 15.3 s -> ~13.5 s (40 steps) = ~11-12% faster** on
  convrot,sage,fusednorm. The win exceeds the 7.4% dequant fraction because the
  fusion also removes the i32 tensor's global write+readback and 2240 kernel
  launches/trace.

### Fused RMSNorm×weight + gated residual (DONE — `qwen-image-rs-fused-rmsnorm-gate`, extends `fusednorm`)
Driven through the issue-lifecycle. nsys of the fast path showed ~10% of denoise
GPU time spread across bmul_f32 (6.2%) / bmul_bf16 (4.6%) / fast_sum_f32 (3.6%) /
usqr_f32 (2.3%) / badd_bf16 (2.8%) — the RMSNorm variants and the gated residuals
the LayerNorm+AdaLN fusion left untouched. Two more kernels in
`kernels/fusednorm/fused_norm.cu`, both behind `fusednorm`:
- `fused_rmsnorm_scale_kernel` — CTA-per-row f32 sum-of-squares reduction,
  `out[m,n] = x[m,n] * rsqrt(mean_n(x^2)+eps) * W[n]` with a per-COLUMN weight
  vector `W` (f32). ONE kernel serves both RMSNorm sites: `ZeroCenterRmsNorm`
  (txt_in, N=4096, `W=weight+1`) and `HeadRmsNorm` (q/k norm, N=128, `W=weight`).
  `W` rides as f32 so it matches candle's `weight.to_dtype(F32)(+1)` exactly;
  ZeroCenter bakes `weight+1` in f32 at load, Head upcasts weight at load.
- `fused_gated_residual_kernel` — elementwise `out = h + tanh(gate)*y` (f32 tanh +
  fma, bf16 in/out), the 2×/block gated residuals in `Block::forward`.
- `src/fusednorm.rs`: `FusedRmsnormScale` (CustomOp2) + `FusedGatedResidual`
  (CustomOp3) bridges + `self_test_rmsnorm`/`self_test_gated`. `src/model/dit.rs`:
  `rmsnorm_scale` + `gated_residual` helpers (mirror `norm_mod`) wired behind
  `cfg(fusednorm)`; candle fallback unchanged. `fusednorm-test` prints the new
  cosines.
- **Self-tests vs candle: RMSNorm×weight cosine 0.999995 (N=4096) / 0.999999
  (N=128); gated residual 0.999996.** dit-forward vs oracle overall_cos 0.999935
  (UNCHANGED from the 0.999934 baseline).
- **Denoise ~13.5 s -> ~10.8 s (40 steps, mean of 10679/10881 ms) = ~20% faster**
  (0.338 -> 0.270 s/step) on convrot,sage,fusednorm. The win exceeds the ~10%
  kernel fraction because the fusions also drop many small op launches and
  bf16<->f32 casts. Cumulative fastest path: bf16 0.62 -> 0.27 s/step.
- Remaining fusion levers: none obvious in the norm/residual buckets; attention
  and the resident VAE decode dominate what's left.

### BSHD-native fused attention (DONE — `qwen-image-rs-bshd-attention`, commit a081ad5)
Kills the transpose→contiguous copies the copy-reduction audit (below) proved
unremovable at the Rust level. The sage path now keeps q/k/v in `(B,S,H,D)` end
to end:
- `kernels/sage/rope_bshd.cu` + `src/rope.rs` — a BSHD interleaved-RoPE
  `CustomOp3` replaces candle `rope_i` (which forces `(B,H,S,D)`), so q/k are
  never transposed for RoPE. Parity vs candle `rope_i`: cosine **0.999996**.
- `kernels/sage/sage_ffi.cu` — stride-parameterized launchers
  (`sage_quant_q/k_bshd` read bf16 in BSHD, write int8 **HND**;
  `sage_attn_bshd` reads int8 q/k HND, V + O in BSHD). The vendored kernels are
  UNCHANGED — they already honor independent `stride_bz/seq/h` for q/k/v/o
  (base ptrs at `qk_int_sv_f16_sm80.cuh:198-200`, O store `:634`) and separate
  quant in/out strides (`fused_quant.cuh:65-92`). Only head_dim need be stride-1
  (true in HND and BSHD).
- `src/sage.rs` `sage_attention_bshd` — CustomOp3 that honors each view's
  `Layout::start_offset()` + strides, so the block-causal S-axis narrows (text
  prefix / image queries) are **zero-copy views** (offsets the device ptr by
  `start_offset*2` bytes). `debug_assert!(b==1)` marks the validated envelope
  (stride-honoring would also be correct for B>1, but that's untested). The old
  `(B,H,S,D)` `sage_attention` is retained ONLY as the equivalence oracle.
- `src/model/dit.rs` — `forward()` branches on `cfg(sage)`: BSHD path (no
  transpose, `rope_i_bshd`, `attend_bshd` with dim-1 narrows + `Tensor::cat`,
  BSHD out); flash/naive paths unchanged.
- **Validation:** sage BSHD vs the retained BHSD path (same block-causal split)
  **cosine 1.000000, maxabs 0.0** — bit-exact, so images are identical to the
  pre-change fast path. dit-forward `--convrot` vs oracle **0.999934**
  (UNCHANGED). Denoise **0.270→0.247 s/step** (~8.5%, 9885 ms/40 resident) —
  matches the predicted ~8.4% ucopy fraction. nsys: `ucopy_bf16` fell from
  ~8.4% (2785 inst / 10-step denoise) to ~2.5% (377 inst) in a FULL generate
  trace that also includes VAE decode + the bf16 text encoder; the residual is
  VAE + the `ot||oi` cat, not attention transposes.
- Follow-ons (not done): fuse Q/K quant INTO the rope kernel (thu-ml's own IO
  trick — avoids a separate rope pass); bf16-native-V PV kernel (the V f16 cast
  survives, but no longer behind a transpose).

### RoPE fused into the INT8 Q/K quantizer (DONE — `qwen-image-rs-rope-quant-fusion`)
SageAttention's own IO trick: q/k are rotated and INT8-quantized in ONE pass.
- `kernels/sage/sage_ffi.cu` `RopeQuantInt8Kernel` / `sage_rope_quant_bshd_launch`:
  reads PRE-rope bf16 q/k (an S-axis BSHD view + the matching cos/sin rows),
  rotates each 8-element pack's 4 interleaved pairs, rounds to bf16 (what the
  unfused rope kernel stored), then runs the verbatim `QuantInt8Kernel` tail
  (block amax → scale amax/127 → int8). int8 HND + scales laid out exactly as
  before. `kernels/sage/rope_pair.cuh` is the shared no-FMA (`__fmul_rn` etc.)
  rotation used by both `rope_i_bshd` (kept as oracle) and the fused kernel.
- `src/sage.rs`: `QuantizedQk` VO (one U8 tensor = int8 region + f32 scales,
  with dims + `QkRole::{Query(128-token blocks), Key(64)}`); `rope_quant_bshd`
  (fused), `quant_bshd` (unfused oracle), `sage_attention_quantized`
  (attention over pre-quantized q/k). `sage_attention_bshd` = quant_bshd ×2 +
  sage_attention_quantized (oracle path, one attention launch path remains).
- Block alignment: `attend_bshd` fuses PER NARROW — q[0:txt] (blk 128),
  k[0:txt] (blk 64) with cos/sin[0:txt]; q[txt:S] with cos/sin[txt:S]; full k.
  Scales start at each call's first token, as the kernel's `q_scale_idx` /
  `k_scale_idx` expect. k is still quantized twice (as before).
- **Validation:** `sage-test` new line — fused vs rope→quant on all 4 narrows,
  B=1 and B=2, S=293/txt=37 (partial blocks): **0 int8 mismatches, 0 scale
  mismatches, attention cosine 1.000000 maxabs 0** (sage-test now exits nonzero
  on mismatch). dit-forward WITHOUT `--convrot` (deterministic path): output
  **bit-identical** to the pre-change build (cos 1.000000, mse 0; vs oracle
  0.999965 both). With `--convrot`: in-distribution with baseline (see gotcha
  below): 8 runs 0.99952–0.999931 vs baseline 8 runs 0.99942–0.999940.
- **Speed (A/B, same session, batch --resident 2 prompts ×40 steps, 2nd image):**
  baseline 9330–9367 ms → fused 9171–9189 ms = **0.2336 → 0.2295 s/step
  (−1.8%)**, matching the prediction (rope kernel = 53 ms / 10-step trace,
  ~2%). nsys: `rope_i_bshd_kernel` gone; `RopeQuantInt8Kernel` 18.97 + 18.74 ms
  ≈ the old `QuantInt8Kernel` 19.0 + 19.9 ms — the rotation is free.
- Note: the absolute s/step on 2026-09-30 is lower than the 0.247 recorded
  earlier for the same baseline commit (clock/thermal drift) — compare A/B only.

### V born f16 in the ConvRot epilogue (DONE — `qwen-image-rs-bf16-v-pv`)
The sage FP16 P·V takes f16 V, and `attend_bshd` used to cast V bf16→f16 every
block, every step. nsys (10 steps): `cast_bf16_f16` 320 inst × 76 us = 24.4 ms =
2.4 ms/step (~1%).
- `kernels/convrot/int8_gemm.cu`: the Sm80EVT types are a template
  `Evt<ElementOutput>` (bf16 | f16); `int8_gemm_dequant_impl<E>` holds the host
  launcher; `int8_gemm_dequant_bf16` and the new `int8_gemm_dequant_f16` wrap it.
  The f16 store rounds the f32 epilogue value once (the old path rounded twice:
  f32→bf16→f16), so it is at least as accurate.
- `src/convrot.rs`: `EpilogueOut::{Bf16, F16}`; `Int8GemmDequant(EpilogueOut)`;
  `ConvRotLinear::forward_as(x, out)` (`forward` = Bf16).
- `src/model/dit.rs`: `QLinear::forward_f16` (sage only): ConvRot → f16
  epilogue; Full/Quant → forward + `to_dtype(F16)` (the old op, so the
  no-convrot path is byte-identical). The sage branch uses it for `to_v`;
  `attend_bshd` takes V already f16.
- Tests (`convrot-test`): both store types bit-exact vs a host reference at
  M=37 (odd M, alignment); f16 epilogue vs bf16 epilogue + cast at the to_v
  shape (M=4117, K=N=4096) within 1 bf16 ulp. convrot-test now exits nonzero
  on a bit-exactness failure.
- **This change exposed the sage partial-tile NaN bug** (see Gotchas): the cast
  kernel used to be the last kernel before the rope-quant/attention calls; with
  it gone, the text-prefix attention read NaN stale shared memory every run
  (dit-forward --convrot 0.9059). Fixed in the same issue (zero-fill).
- **Validation:** no-convrot dit-forward byte-identical to the prior build
  (vs oracle 0.999965). `--convrot`: 0.999944 on 4/4 runs (deterministic now;
  prior build 0.99980–0.99995 on the same 4 runs). Speed A/B (same session,
  batch --resident, 2 prompts × 40 steps, 2nd image): base 9169 / 9168 ms →
  new 9057 / 9068 ms = **0.2292 → 0.2266 s/step (−1.1%)**. nsys: `cast_bf16_f16`
  gone; to_v shows as a separate f16-epilogue GEMM (320 inst, 245 us).
- Images match the prior build closely. NOTE: both A/B prompts ("a red mug on
  a wooden table", "a lighthouse on a cliff at sunset") came out clean but
  OFF-PROMPT with BOTH builds (a purple city poster, a pencil-sketch dining
  room). This is the known "convrot B=1 lands off-prompt at some seeds" issue;
  not caused by this change, still open.

### SageAttention2 sm89 (DONE — `qwen-image-rs-sageattention2`, feature `sage2`)
INT8 Q·K with **per-thread** scales + **K smoothing**, **FP8 e4m3 P·V** with
per-channel V scales and fp32+fp16 two-level accumulation (thu-ml's sm89 default).
- **Feature + runtime switch:** `sage2` (implies `sage`) compiles
  `kernels/sage/sage2_ffi.cu` (own TU — the sm80/sm89 vendored kernels share
  `PACK_SIZE_*` macro names). In a `sage2` build `QIR_SAGE` picks the attention
  at run time: `2` (default) SA2 fp16-accum, `2f32` SA2 fp32-accum, `1` v1.
  Parsed once in `sage2::attention_impl()` (unknown value = hard error), logged
  once at info (`attention implementation attention=Sage2(F16)`). Why a runtime
  switch on top of the feature: same-binary, same-session A/B with no rebuild,
  and v1 stays a zero-cost fallback (`QIR_SAGE=1` output is BIT-IDENTICAL to
  the pre-change build, convrot and not).
- **Vendored:** only `vendor/qattn/qk_int_sv_f8_sm89.cuh` (upstream
  `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh` @ d1a57a5, torch include stripped). It
  reuses the already-vendored `attn_utils.cuh`, so the **kFillZero LOCAL PATCH
  covers its predicated Q/K loads**. Its V loads are UNPREDICATED (upstream
  assumes padded V) — our V quant writes every padded column as 0, so no
  stale/uninitialized bytes are ever read.
- **Ours (upstream does these in Triton/torch):** `Sage2KSumPartialKernel`
  (RoPE + per-256-token channel sums, fixed-order reduce → deterministic) →
  `Sage2RopeQuantKernel<Key>` (RoPE − the call's key mean → per-thread INT8;
  4 scales / 64-key block, group `(tok%8)/2`); `Sage2RopeQuantKernel<Query>`
  (RoPE → per-thread INT8; 8 scales / 32-query warp block, group `tok%8`; grid
  padded to `ceil(n/128)*4` blocks so every scale the kernel reads is written);
  `Sage2VAmaxPartialKernel` + `Sage2VQuantKernel` (per-channel amax → FP8,
  transposed to `(B,H,D,Lpad64)` with upstream's 16-token fp8-mma permute
  `0,1,4,5,8,9,12,13,2,3,…`; amax floored at 1e-7 — upstream's MeanScaleKernel
  NaNs on an all-zero channel). Group mappings are derived from the mma fragment
  layout (rows `lane/4+8j`, cols `2*(lane%4)+8j`) = upstream `quant_per_thread.py`.
- **Rust (`src/sage2.rs`):** VOs `Sage2Qk` (int8 + per-thread scales + Key
  scratch, one U8 alloc) and `Sage2V` (fp8 + per-channel scales + scratch,
  **carries its `PvAccum`** — the attention takes the accumulation mode from V,
  so V's scale_max (2.25 fp16-accum / 448 fp32) can never mismatch the kernel).
  Scratch lives in the result allocation (outlives the async kernels). Every
  launcher returns `cudaGetLastError`; every op honors `start_offset` + strides.
  `attend_block_causal` keeps v1's per-narrow structure (text causal on
  `[0,txt)`, image queries over full k/v), so K is smoothed with each call's
  own key mean. `dit.rs::attend_bshd` dispatches on `attention_impl()`.
- **sage-test (new lines, exits nonzero on failure):** block-causal txt=37/S=293,
  B=1,2, real-RoPE angles, per-channel K bias ×4, zeroed V channel + K head-dim:
  vs f32 ref **cos 0.999373 (fp16 accum) / 0.999363 (fp32)** (v1 on the same
  inputs 0.999615 — random Gaussian scores are diffuse, FP8 P dominates);
  non-finite 0. B=2 lanes vs B=1 (lane 1 = nonzero batch offset) 0 mismatches;
  S-offset views vs copies 0 (bytes + outputs); run-twice 0; NaN-poisoned smem
  partial tile (txt=21) NaN 0, maxabs 0.
- **dit-forward vs oracle:** `QIR_SAGE=2` --convrot **0.999894** (v1 0.999944),
  no-convrot 0.999945 (v1 0.999965); `2f32` 0.999901 / 0.999943. vs the prior
  build's output: SA2 0.999911 (convrot) / 0.999966 (no-convrot). Below the
  0.99993 standing gate, above the ~0.9995 stop line → shipped behind the
  feature/selector; default-path choice left to the human.
- **Speed (batch --resident, 2 prompts × 40 steps, 2nd image, SAME binary, two
  interleaved rounds):** v1 9031 / 9041 ms → SA2 8590 / 8605 ms = **0.2259 →
  0.2149 s/step (−4.8%)**; `2f32` 8761 ms (−3.0%). nsys (10 steps): image
  attention kernel **1.035 → 0.553 ms/call (1.87×), 331 → 177 ms**; quant side
  v1 `RopeQuantInt8Kernel` 34 ms → SA2 five kernels 81 ms (K-sum 16.5, K-quant
  15.7, Q-quant 16.4, V-amax 16.1, V-quant 16.6) — the extra ~4.7 ms/step of
  bandwidth-bound passes eats a third of the attention win.
- **Images:** B=1 mug (seed 42) and `--batch 2` both on-prompt; B=1 vs B=2 lane 0
  latent cos 0.99963; SA2 vs v1 B=1 latent 0.99894, PSNR 32.0 dB (same image,
  fine-detail drift). Lighthouse A/B image clean.
- **Follow-ons:** DONE as `-sage2-quant-fusion` (next section). Per-warp Q
  (fewer scales) is not needed — per-thread already ships.

### SA2 quant fusion (DONE — `qwen-image-rs-sage2-quant-fusion`, bit-identical)
The six quantized operands of one block-causal SA2 attention (Q/K/V for the
text-prefix call, Q(img)/K(full)/V(full) for the image call) used ten per-op
launches per layer. Now `sage2::quant_layer` builds them in **three launches of
one generic 512-thread kernel** (`Sage2QuantKernel`, `sage2_quant_layer_launch`),
each over a task table (one task per CTA, flat `blockIdx.x` → (task, block)):
1. V-amax partials (V full + V txt);
2. V quant + K-sum partials;
3. K quant + Q quant.
- **Where the time went (measured, dit-forward nsys, us/layer):** old 250 =
  K-sum 51, V-amax 50, V quant 50, K quant 48, Q quant 47, + five text-prefix
  launches ~11. The passes were already near DRAM bandwidth in isolation; the
  waste was (a) 1024-thread CTAs = ONE CTA per SM, so every per-CTA barrier /
  serial prologue (summing 17 chunk partials, the 4-thread group max) idled the
  SM's memory pipe, and (b) launch order: V's amax pass ran after Q and K had
  evicted V from L2. **512-thread CTAs (3 per SM, 40 regs, 26 KB smem)** hide
  the barrier stalls; **order V → K → Q** (V was written last by to_v) makes the
  V-amax pass an L2 hit (12.5 us for 34 MB) and each quant pass follows its
  partial pass. Seven orders measured: 233–255 us with 1024-thread CTAs, 168.5
  with 512 (order above best; [K quant + Q] alone in launch 3 > other splits).
- **Result:** quant **250 → 168.5 us/layer (−33%)**; generate 10-step trace
  98.5 → 53.9 ms. Denoise A/B (same session, batch --resident, 2nd image, two
  interleaved rounds): 8644 / 8628 → 8516 / 8503 ms = **0.2159 → 0.2127 s/step
  (−1.5%)** (the absolute s/step drifts by session; compare A/B only).
- **Bit-identical:** every task is the per-op kernel's math op for op, the
  K-mean in the same chunk order. `sage-test` new line: fused vs the untouched
  per-op kernels (kept as the oracle: `rope_quant`/`quant_v`,
  `attend_block_causal_unfused`) — payload + scale bytes of all six operands and
  the attention outputs, B=1/B=2 lanes, S-offset views, txt=37 and 21, fp16 and
  fp32 accum: **0 mismatches**. dit-forward --convrot / no-convrot / `QIR_SAGE=2f32`
  / `QIR_SAGE=1`: byte-identical to the prior build (`cmp`); B=1 and `--batch 2`
  PNGs byte-identical to the prior build; B=1 == lane 0.
- **ABI:** `S2Task` / `S2Tasks` are `#[repr(C)]` mirrors, size-asserted on both
  sides (96 / 592 B); the launcher re-checks each task's grid bookkeeping
  (`nblocks` vs kind/ncta/nchunk) and returns cudaErrorInvalidValue on drift.
  `quant_layer` is a multi-input/multi-output bridge outside CustomOp
  (`storage_and_layout` + `Tensor::from_storage`): same view checks
  (`check_qk_view`, `check_rope_tables`) and start_offset + stride pointers as the
  CustomOp bridges; guards held until the launch is enqueued.
- **Not done (measured / argued, not worth it):** V-amax in the to_v ConvRot
  EVT epilogue — needs per-batch-lane AND per-call-range (text [0,txt) vs full)
  column maxima with atomics + zero-init, and the pass it would remove is now
  the 12.5 us L2-hit one. Reusing the full K/V quant for the text call changes
  the math (mean/scales over the whole sequence) and needs a vendored-kernel
  stride patch; the text tasks now cost a few CTAs inside shared launches.
  Keeping a whole head on chip to drop the second K read is impossible
  (1 MB/head; the K mean is per channel and the INT8 scale is per token across
  channels — a grid-wide dependency).

### Tail linears through ConvRot (`qwen-image-rs-convrot-tail-linears` NEGATIVE for speed; then DONE for uniformity in `qwen-image-rs-tail-linears-unify`)
**Now (`-tail-linears-unify`): every DiT linear is a `QLinear`.** One policy,
`dit::linear_precision(prefix, quant, convrot)` (`BLOCK_LINEARS` + the
`TAIL_LINEARS` table of `(prefix, K, N, rotate)`), picks Full / Q8_0 / ConvRot
from the weight prefix (`vb.prefix()`); `prequantize-convrot` calls the same
`is_convrot_target`, so file and loader cannot disagree. An unknown prefix is a
load error; a policy-Full layer whose file has only `weight_i8` is a load error
naming the layer. Q8_0 (`--quant`) stays block-only (unchanged). Load logs
`DiT linears loaded convrot_linears=231 total_linears=232`.
- Rotated under `--convrot`: txt_in.in_layer/out_layer, time_embed linear_1
  (K=256) / linear_2, modulation.1 (M=2, N=16384), norm_out.linear, proj_out
  (N=64: a partial 128-wide N tile; N%8 meets the 128-bit store alignment and
  s_row stays 32-B aligned). **img_in stays bf16** (K=64, cannot rotate).
  `convrot-test` checks the epilogue bit-exact vs the host ref at
  (M,N,K) = (2,4096,256), (2,16384,4096), (37,64,4096): 0 mismatches.
- **Accuracy:** dit-forward --convrot vs oracle **0.999894 → 0.999898**
  (vs the no-convrot output 0.999904 → 0.999912). No-convrot dit-forward
  byte-identical to the prior build (Full = the same candle Linear). Prequant
  (231 linears, 66 bf16 tensors) vs on-the-fly: byte-identical.
- **Per-layer attribution is below the noise floor — do not over-read it.**
  Measured with a temporary knob (not shipped): each upstream tail layer ALONE
  moves the oracle cosine by −5.3e-5 … +4.8e-5 (linear_2 0.999841, txt_in.in
  0.999942, modulation 0.999905, txt_in.out 0.999881, linear_1 0.999888);
  the downstream ones barely move it (norm_out 0.999894, proj_out 0.999889;
  vs the prior output 0.999997 / 0.999990). The effects do not add: keeping
  the two "worst" (linear_2, txt_in.out) bf16 and rotating the rest gave
  0.999849, keeping only linear_2 bf16 0.999791, all seven 0.999898. With the
  block linears forced bf16, each tail layer alone still scatters the cosine
  0.999909–0.999957 around the bf16 0.999945. So any ~1e-4 perturbation
  upstream re-rolls the SA2 INT8/FP8 and block INT8 rounding through 32
  blocks; the per-layer delta measures that chaos, not the layer's error.
  Decision: rotate all seven (the only tested set within 1e-5 of the
  baseline, and it improves both references). Flip a `TAIL_LINEARS` bool to
  keep a layer bf16.
- **Side effect: B=1 is now bit-identical to `--batch 2` lane 0** (latent cos
  1.0000000, mse 0; was 0.9986 on the prior build) — the remaining
  M-dependent cuBLAS bf16 GEMMs (txt_in at M=B·txt, proj_out at M=B·S) were
  what made batch lanes differ; the INT8 CUTLASS GEMMs are per-row invariant.
- **Speed: flat** as predicted (same session, batch --resident, 2nd image,
  two interleaved rounds): base 8599 / 8595 ms → new 8611 / 8601 ms =
  0.2149 → 0.2150 s/step (+0.1%, inside the ~0.4% noise). Images: B=1 mug
  and `--batch 2` on-prompt; new vs prior B=1 latent cos 0.99884.

Original measurement (`-convrot-tail-linears`), before building: nsys, 10-step trace, fast build (`convrot,sage,fusednorm`),
denoise GPU busy 2301 ms (= 230 ms/step). The small-M bf16 GEMMs, attributed by grid:
- `modulation` (M=2, K=4096, N=16384): `cutlass_80_wmma..16x16_128x2` grid (8,128),
  10 inst, **168 us** each.
- N=4096 small-M GEMMs: same kernel, grid (8,32), 50 inst = 5/step (`txt_in.in_layer`,
  `txt_in.out_layer` at M≈txt, `norm_out.linear` + `time_embed` 2 linears at M=2), **42 us** each.
- In scope (modulation + norm_out + txt_in ×2): **0.29 ms/step = 0.13%** of a step.
They are already weight-bandwidth-bound (134 MB / 168 us, 33.5 MB / 42 us ≈ 800 GB/s).
INT8 halves the weight bytes at best → ceiling ≤ 0.15 ms/step (**0.06%**), less the
rotate + quant + dequant launches it adds. The same-session A/B noise is ~0.4%, so the
gain cannot even be measured. Holds at any B and with CFG: `temb` is always (2, INNER),
so modulation/norm_out stay M=2; CFG doubles every GEMM (ratio unchanged).
Related idea, also negligible: `txt_in` is step-invariant (text embed only) and could
be cached across steps — saves 2×42 us/step (0.04%). Not done.

### Copy-reduction audit (`qwen-image-rs-reduce-copies` issue — NEGATIVE RESULT, superseded by BSHD attention above)
nsys of the fast path (`convrot,sage,fusednorm`, bf16 VAE) showed `ucopy_bf16` at
6.9% (2785 instances / 10-step denoise ≈ 278/step) — plain `.contiguous()` memory
copies. Audited every fast-path `.contiguous()` for redundant (already-contiguous)
or avoidable (consumer tolerates a view) removal **without changing numerics**.
**Conclusion: all are load-bearing; nothing safely removable at the Rust level.**

Grounded in candle 0.11 source (verified, not assumed):
- `rope_i` **bails** on non-contiguous input (`rotary_emb.rs:278`). So the
  `qh`/`kh` `transpose(1,2)?.contiguous()?` at `dit.rs` 255/260 are REQUIRED — the
  transpose makes them strided and rope over the full sequence needs contiguous.
- `Tensor::contiguous()` on an already-contiguous tensor is a **free clone**, no
  copy kernel (`tensor.rs:2466`). So an already-contiguous `.contiguous()` emits
  ZERO `ucopy` — it is never part of the 6.9%.
- `Tensor::reshape` on a **non-contiguous** tensor **copies** (`copy_strided_src`,
  `tensor.rs` else-branch); on contiguous it is a free view. So `dit.rs:302`
  `out.transpose(1,2)?.contiguous()?` is not redundant: dropping it just moves the
  identical single copy into the `reshape((b,s,INNER))` at `dit.rs:266`.
  **Empirically confirmed** (test-removed 302, rebuilt): `overall_cos` 0.999934
  UNCHANGED, denoise 15333 ms vs 15394 ms baseline = −0.4% (noise, no win). Reverted.
- `to_dtype` output is **always contiguous** (`cuda_backend to_dtype` writes linear
  output). So `sage.rs:163` `.contiguous()` after `to_dtype(F16)` is a no-op today —
  but it is a **defensive raw-pointer FFI guard** (the sage kernel reads a bare
  device ptr assuming contiguous; if `v` ever arrives already-F16 non-contiguous it
  would silently corrupt). Same for `fusednorm.rs:89-90`. **Kept as guards.**
- `sage_attention` already contiguizes its inputs internally, and `attend()` passes
  **views** (`vv.transpose(1,2)` at `dit.rs:288`, narrows) — a prior pass removed the
  double-copies; none remain. `ConvRotLinear::forward` has **no** `.contiguous()`
  (rotate uses matmul + reshape on contiguous views).

The 6.9% `ucopy_bf16` is **structural**: per block ≈ qh + kh full-S contiguizations
(for rope_i) + sage narrow→contiguous (text q/k, image q; whole-k is a no-op) + the
`ot||oi` cat + the out linearization. Reducing it needs a fused attention kernel that
ingests (B,S,H,D) and does rope/transpose internally — **out of scope** (Rust-only).
- **Considered & rejected:** move `rope_i` into `attend()` per-narrow to drop the
  full-S `q` pre-contiguous (255). Ceiling ~≤1% denoise (only `q`'s full-S copy goes;
  `k` still needs the whole-S contiguize for the image call), and it touches all
  THREE `attend` variants (sage/flash/naive) + narrowed cos/sin correctness, with no
  way to validate flash/naive on the fast-path build. High-risk / low-reward — not done.

### Future levers
- **Faster resident encode** — the Q8 te forward dominates resident per-image
  time; cache dequantized bf16 weights once after load (trades VRAM), or use a
  narrower activation-aware path.
- **CUDA-graph capture** — not viable in candle 0.11 (no stream-capture hook).
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
- New `.cu`: `#include <cassert>` BEFORE any cuda_fp8/fp6/fp4 headers (CUDA 13
  `__assert_fail`). Name every `extern "C"` launcher distinctly from Rust fns
  (a `..._launch` suffix — a same-name collision is a compile error).
- **INT8 CUTLASS mma needs `arch::OpMultiplyAddSaturate`** (no plain-`OpMultiplyAdd`
  s8 16×8×32); `device::Gemm` defaults it, `DefaultGemmWithVisitor` (EVT) does not.
- **EVT scale vectors pack col-first** (`cat(col_scale,row_scale)`): the vectorized
  RowBroadcast base must be 32-B aligned; row-first faulted MISALIGNED at M=4117.
  Build the convrot .cu with `--expt-extended-lambda` (EVT visitors).
- **dit-forward `--convrot` noise was a NaN bug — FIXED, now deterministic**
  (`qwen-image-rs-bf16-v-pv`). Before: 8 runs vs oracle scattered
  0.99942–0.999940. Root cause: the vendored sage kernel loaded K/V rows
  `>= kv_len` of the last 64-key tile with `SharedMemFillMode::kNoFill`, so they
  kept stale shared memory from whatever kernel ran before on that SM. Masked
  scores are exactly 0, but `0 * NaN = NaN` in P·V. The text-prefix causal call
  (kv_len = txt_len < 64) always has such rows, so the text outputs went NaN
  whenever the stale bytes were NaN/Inf (it depends on which kernel ran just
  before on each SM → run-to-run noise). The convrot INT8 activation quantizer
  then hid it (fmaxf ignores NaN, NaN→int8 gives 0); the no-convrot path had
  finite stale bytes by luck. Fix: `kFillZero` in the predicated
  `load_global_to_share` (`kernels/sage/vendor/qattn/attn_utils.cuh`, LOCAL
  PATCH). After: `--convrot` gives 0.999944 on every run; no-convrot stays
  byte-identical. Regression test: `sage-test` "partial last K/V tile under
  NaN-poisoned smem" (fills every SM's smem with f16 NaN first).
  compute-sanitizer memcheck/initcheck do NOT catch this (shared memory, and
  initcheck only tracks global memory) — poison shared memory to test for it.
- **SA2 sm89 kernel loads V UNPREDICATED** (upstream assumes V padded to 64
  tokens): whatever produces the FP8 V buffer must write every padded column
  (ours writes 0). Q/K go through the patched `load_global_to_share` (kFillZero).
- **Fused CUDA bridges MUST honor `Layout::start_offset`** (`qwen-image-rs-b1-off-prompt`).
  `CudaStorage::device_ptr` is the base of the ALLOCATION, not the view. candle's
  `contiguous()` is a no-op for any `is_contiguous` view and size-1 dims are skipped
  in that check, so `(1,S,D).narrow(1, drop, n).contiguous()` stays a zero-copy view
  at offset `drop*D`. generate's B=1 prompt embeds are exactly that (TE output past
  the system prefix); the fusednorm/convrot bridges passed the base pointer, so
  txt_in's fused RMSNorm read the SYSTEM-PROMPT rows → every B=1 image was
  conditioned on the system prompt (clean but off-prompt: ink-wash portraits,
  gibberish banners). B>1 `broadcast_as().contiguous()` made a fresh copy, so batches
  were fine; dit-forward/denoise from files had offset 0, so parity never saw it.
  It hit every B=1 build since text_norm went through `fused_rmsnorm_scale`, sage or
  not, convrot or not. Fix: every bridge adds `crate::layout::dense_byte_offset::<T>`
  (errors on non-dense views); the CUTLASS GEMM bridges also bail on a misaligned
  view. Regression: `fusednorm-test` / `convrot-test` run each op on an offset view
  vs a fresh copy (must be bit-identical; exit nonzero otherwise) + CPU unit tests in
  `src/layout.rs`. Isolation recipe that found it: same seed B=1 vs `--batch 2` lane
  0 (identical noise + embeds) → if they differ in CONTENT, suspect a B-dependent
  input layout, not numerics.
- `scripts/compare_dit.py <ours> <oracle>` = the dit-forward oracle compare
  (overall_cos, MSE, per-token cos over the last 4096 rows); run it with the
  oracle venv python.
- **Fused-norm parity: pass the weight as f32** (ZeroCenter bakes `weight+1` in f32
  at load) to match candle's `weight.to_dtype(F32)`.

### Session/tooling workflow (host, verify runner, issue-lifecycle)
- **Build/run on the 4090 ONLY via `scripts/host.sh`** (git-archive→copy→build over
  the `wsl-drills` @swamp/ssh model). COMMIT before `sync` (uses `git archive HEAD`).
  Host build: `export CUTLASS_DIR=$HOME/dev_tmp/cutlass/include; cargo build
  --release --features convrot,sage,fusednorm`. Read stdout via the host.sh `run`
  wrapper; filter noise with `grep -viE "WRN|system │|Source path|seaweedfs|Committed|Syncing|Wrote|Preparing|Running|warning|cutlass|cute|include"`.
- **Mac cargo: ALWAYS `CARGO_TARGET_DIR=$HOME/.cache/qwen-image-rs-target`** — else a
  stray in-tree `target/` appears and `scripts/check.sh` fails exit 3 ("a target/
  directory appeared"); fix `rm -rf target`. Default (no-feature) build must always
  compile on the Mac.
- **`--repo-dir` is a swamp SUBCOMMAND option** (after `model method run`/`data get`),
  NOT global. Run swamp from `/Users/mag1/dev_tmp/swamp`.
- **Issue-lifecycle Phase-4b `verify` runner** (Mac, `runner:"local"`): use
  `command:"bash", args:["-c","export PATH=$HOME/.cargo/bin:$PATH; export
  QIR_LOCAL_TARGET_DIR=$HOME/.cache/qwen-image-rs-target; scripts/check.sh <stage>"]`,
  `cwd:"."` (relative), `repoDir` absolute; one `verify` call with all 4 controls
  (fmt/lint/check/test). (~/.bash_profile was fixed this session; toolchain pins
  rustfmt/clippy.)
- **IL method arg quirks:** `implement --input branch="main"`; `iterate_verification
  --input source=auto --input reason=...`; `record_review reviewer/verdict/--input-file`;
  `resolve_findings --input-file` (`resolutions:` map); `approve_plan` no inputs;
  `attest --input '{commitSha,repoDir,configPaths:[agent-constraints/*.md,Cargo.toml],producedBy}'`.
  Standing PRE-APPROVAL (agent-constraints/iteration-limits.md): approve plans +
  resolve findings yourself. Attestation `data get` writes ~empty (cosmetic; model
  state is authoritative). Then `.attestations/<sha>.json` + `complete`.
- **Subagents:** run optimization lifecycles SEQUENTIALLY (parallel forks clobber the
  shared host build dir `~/dev_tmp/qwen-image-rs`, the out-of-tree target, and the
  `main` branch; `wsl-drills` model lock serializes GPU work anyway).

## Issue-lifecycle issues (state in swamp) — resume: `swamp model method run <issue> hydrate`
Phase ports (complete): `-oracle`, `-vae`, `-text-encoder`, `-dit`.
Optimizations (complete): `-fused-adaln` (LayerNorm+AdaLN), `-fused-actquant`,
`-dequant-epilogue` (CUTLASS EVT), `-fused-rmsnorm-gate` (RMSNorm+gated residual),
`-vae-bf16`, `-bshd-attention` (BSHD-native fused attention),
`-rope-quant-fusion` (RoPE fused into the sage INT8 Q/K quantizer),
`-bf16-v-pv` (V born f16 in the ConvRot epilogue + sage partial-tile zero-fill),
`-sageattention2` (SA2 sm89 behind `sage2` + `QIR_SAGE`),
`-sage2-quant-fusion` (SA2 quant: 10 launches/layer → 3, bit-identical, −1.5%),
`-prequant-default` (cached prequantized DiT = default `--convrot` load; load 7.0 → 2.7 s cold),
`-fused-rotate-quant` (Hadamard rotation fused into the activation quantizer, −10.1%),
`-fused-swiglu` (SwiGLU silu·p fused into the MLP-out rotate+quantize, −8.4%).
Bugs (complete): `-b1-off-prompt` (B=1 generate ignored the prompt: fused bridges
ignored the view offset — see Gotchas).
Non-code outcomes: `-reduce-copies` (complete, NEGATIVE — all fast-path
`.contiguous()` load-bearing; see "Copy-reduction audit"); `-convrot-tail-linears`
(complete, NEGATIVE — 0.13% of a step; see "Tail linears through ConvRot");
`-tail-linears-unify` (complete — every DiT linear a `QLinear`, one precision
policy, speed flat, 0.999898); `-fused-dequant`
(closed — superseded by `-dequant-epilogue` which shipped it).

## Also-planned / future levers
Pre-quantized GGUF *file* (cut ~24 s load); SageAttention INT8 attention; prefix
KV cache (helps most with condition images); resident serve mode.

Memory: `[[project_qwen_image_rs]]`, `[[lesson_qwen_image_rs_attention_bound_profile]]`.

### VAE bf16 decode (DONE — `vae-bf16` issue)
VAE decoder runs in bf16 on CUDA by default (was F32 — the top GPU-time bucket).
`QwenImageVae` got a `dtype` field; decode unnormalizes z*std+mean in f32 then
casts to the conv dtype; `RmsNorm` (F.normalize over C) f32-accumulates.
`--vae-f32` fallback; `vae-decode --bf16`.
- **Quality-neutral: bf16 vs f32 53.6 dB; bf16 vs oracle 55.2 dB** (f32 was
  53.3 — bf16 is marginally closer to the bf16 diffusers oracle).
- **Decode 1.84 s -> 1.17 s (~1.57x)** tiled, resident 40 steps. Per-image ~16.3 s.
