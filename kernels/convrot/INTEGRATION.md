# ConvRot INT8 kernel integration (from sglang #38040)

Vendored: `convrot_int8_gemm.cu` — the reference ConvRot W8A8 kernel from
[sglang PR #38040](https://github.com/sgl-project/sglang/pull/38040)
("ConvRot INT8 online W8A8 for DiTs"). It is torch+CUTLASS coupled; integrating
it into this candle project is scoped below.

## What the kernel provides
1. **`convrot_rotate_quantize_activation_kernel<GroupSize, GeluInput, ...>`**
   (raw pointers already): `bf16 x (M,K) -> int8 x_q (M,K) + float row_scale (M)`.
   Fuses the group-wise **Regular Hadamard rotation** (`H / sqrt(GroupSize)`,
   which keeps `(Ux)·(Uw) == x·w`) **and** per-token (per-row) INT8 quantization
   with an amax scale. GeluInput folds a GELU in for fused MLP paths.
   Launcher `launch_rotate_quantize_kernel(x, x_q, row_scale, M, K, stream)`.
2. **CUTLASS INT8 GEMM** (torch::Tensor interface): `out = dequant(x_q @ w_qᵀ)`
   with per-token `x_scale` and per-channel `w_scale`, optional bias. Dispatches
   to CUTLASS 3.x collective (SM100) or CUTLASS 2.x `mma.sync` (Ada/Ampere, our
   sm89 path). int8 in, int32 accumulate, bf16 out.

## Build finding (important)
The sgl `.cu` depends on **sgl-kernel's own CUTLASS extensions**, not just stock
CUTLASS: `cutlass_extensions/{epilogue_per_row_per_col_scale.h,
gemm_universal_base_compat.h, gemm_with_epilogue_visitor.h}` + a torch-coupled
`utils.h` (`device_sm_version`, `CHECK_INPUT`). A faithful FFI must vendor that
whole slice + torch-decouple `utils.h`. For sm89 only the CUTLASS **2.x**
`run_convrot_int8_gemm_mma_sync` path is used (drop all Sm90/Sm100 3.x/cute code).

## RECOMMENDED simpler path (avoids the sgl extension headers)
The sgl kernel fuses per-row×per-col dequant into a custom CUTLASS epilogue
visitor (hence the extension headers). We don't need that fusion:
1. **Rotate + quantize in candle** (no custom kernel): rotation is done
   (`rotation.rs`); add per-token amax INT8 quant (candle ops). Same for the
   weight offline (per-channel INT8), folded with `fold_weight`.
2. **One stock-CUTLASS 2.x INT8 GEMM** `C(int32) = A(int8) @ Bᵀ(int8)` via
   `cutlass::gemm::device::Gemm<int8,RowMajor, int8,ColumnMajor, int32,RowMajor,
   int32, OpClassTensorOp, Sm80, ...>` + trivial `LinearCombination<int32>` — the
   canonical CUTLASS int8 example, **stock headers only**, builds for sm89.
3. **Dequant in candle**: `out_bf16 = int32.to(f32) * x_scale[m] * w_scale[n]
   (+bias)` — cheap elementwise pass.
This shrinks the FFI to ONE small stock-CUTLASS kernel; everything else is candle.

## Integration steps (the remaining work)
1. **Decouple torch**: drop `torch/all.h`, `ATen/cuda/CUDAContext.h`, `c10/...`;
   replace `TORCH_CHECK` with `assert`; rewrite the two GEMM launchers to take
   raw pointers (`void* out, const int8_t* x_q, const int8_t* w_q,
   const float* x_scale, const float* w_scale, const float* bias, M, N, K,
   stream`) instead of `torch::Tensor&`. The rotate/quantize launcher is already
   raw-pointer (just drop its TORCH_CHECK).
2. **extern "C" wrappers**: `convrot_rotate_quantize(...)` and
   `convrot_int8_gemm(...)` dispatching the GroupSize=256 templates.
3. **build.rs** via `bindgen_cuda`: compile the `.cu` for `sm_89`, `-std=c++17`,
   with the CUTLASS include path. Host has a CUTLASS checkout at
   `~/.cudaforge/git/checkouts/cutlass-*` (from the candle-flash-attn build) —
   pin/vendor a known CUTLASS rather than depend on that artifact.
4. **Rust binding + candle bridge**: `extern "C"` fns; get device pointers from
   candle CUDA tensors (via the cuda backend / cudarc `CudaSlice`), pass the
   stream, wrap outputs back into candle Tensors.
5. **Wire `ConvRotLinear`**: fold `Rᵀ` into weights offline + quantize weights to
   int8 per-channel (once, at load); at forward, call rotate_quantize on the
   activation then the int8 GEMM. Behind `--convrot`; mixed precision keeps
   attention-out + value-proj at bf16 (per the paper).
6. **Validate**: single ConvRotLinear vs bf16 (cosine within int8 tol); DiT
   forward + full image PSNR vs bf16 (target the paper's ~29 dB / 0.96 SSIM);
   bench the int8 GEMM vs bf16 (expect ~2x on the MLP).

## Status
- ✅ Rotation foundation in candle (`src/model/rotation.rs`) validated.
- ✅ Kernel vendored + interface understood; sgl extension-header dependency found.
- ✅ **INT8 tensor-core GEMM PROVEN** (`kernels/convrot/int8_gemm.cu`, the simpler
  stock-CUTLASS path): compiles for sm89 + runs + **bit-exact vs CPU** on the 4090.
  This was the biggest risk — it works. `int8_gemm_s32(int32* C, int8* A, int8* B,
  M, N, K, stream)`.
- ⚠️ **CUTLASS build note:** the flash-attn-vendored CUTLASS checkout has a
  `matrix.h` `set_slice3x3` bug that **CUDA 13.3 rejects**. Use a FRESH CUTLASS:
  `git clone --depth 1 https://github.com/NVIDIA/cutlass` (cloned to
  `~/dev_tmp/cutlass` on the host). Compile: `nvcc -arch=sm_89 -std=c++17
  --expt-relaxed-constexpr -I~/dev_tmp/cutlass/include`.
- ✅ **`build.rs` + candle CustomOp2 bridge WORK** (`src/convrot.rs`, `--features
  convrot`): compiles the kernel (cc + `CUTLASS_DIR`), launches it from candle via
  `as_cuda_slice`→`device_ptr(&stream)`→`wrap_cuda_slice` (flash-attn pattern),
  U8-holds-int8 in / I32 out. `convrot-test` CLI: candle→kernel→candle **bit-exact
  (maxdiff 0)** on the 4090. The whole FFI chain is proven.
- ⚠️ candle's cuda build here lacks the **I32→F32 cast** kernel ("named symbol not
  found"); the self-test compares on host. The int32→bf16 dequant must avoid that
  cast — do it in a tiny custom op/kernel, or copy-to-host, or a supported path.
- ⬜ **Remaining (no unknowns, pure wiring):** (1) rotate+per-token INT8 quant of
  activations (candle rotation is done; add amax→int8 into a U8 tensor — small
  custom op, since candle can't emit signed int8 via `to_dtype`); (2) per-channel
  INT8 weight quant offline (once, at load); (3) int32→bf16 dequant with row/col
  scales (custom op / host, per the cast note); (4) `ConvRotLinear` behind
  `--convrot`, mixed precision (attn-out + value-proj stay bf16); (5) validate
  DiT/image vs bf16 (target ~29 dB / 0.96 SSIM) + bench vs bf16.
