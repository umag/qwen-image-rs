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
- ✅ Rotation foundation in candle (`src/model/rotation.rs`) validated (this is
  the same H/sqrt(N) rotation the kernel fuses — useful as a CPU oracle).
- ✅ Kernel vendored + interface understood.
- ⬜ Steps 1–6 above (the FFI build + bridge + validate) — a focused session.
  Main risk: the CUTLASS build config for the sm89 int8 path, and the
  candle↔cudarc device-pointer bridge.
