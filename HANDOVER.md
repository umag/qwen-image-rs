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

## ConvRot INT8 (the last active work) — validated + fast, needs DiT wiring
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
