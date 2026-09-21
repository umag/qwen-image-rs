// Minimal stock-CUTLASS 2.x INT8 tensor-core GEMM for ConvRot.
// C(int32) = A(int8, M x K row-major) @ B(int8, N x K row-major = K x N col-major).
// Per-row/per-col dequant is applied in candle afterward, so the epilogue is a
// trivial identity (alpha=1, beta=0) — no custom epilogue visitor, stock headers.
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,  // A: (M, K)
    int8_t,
    cutlass::layout::ColumnMajor,  // B: K x N col-major (== weight (N, K) row-major)
    int32_t,
    cutlass::layout::RowMajor,  // C: (M, N)
    int32_t,                    // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,  // Ada (sm89) runs the Sm80 int8 tensor-core path
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,  // int8 mma.sync instruction
    cutlass::epilogue::thread::LinearCombination<int32_t, 128 / 32, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

extern "C" int int8_gemm_s32(
    int32_t* c, const int8_t* a, const int8_t* b, int m, int n, int k, cudaStream_t stream) {
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {a, k},  // A row-major, lda = K
      {b, k},  // B col-major, ldb = K
      {c, n},  // C row-major, ldc = N
      {c, n},
      {1, 0});  // alpha = 1, beta = 0 -> raw int32 accumulate
  cutlass::Status s = gemm(args, nullptr, stream);
  return s == cutlass::Status::kSuccess ? 0 : static_cast<int>(s);
}
