// Stock-CUTLASS INT8 tensor-core GEMMs for ConvRot (sm89, runs the Sm80 int8
// path: mma.sync 16x8x32, int32 accum).
//
// Two entry points:
//   * int8_gemm_s32          — raw C(int32) = A(int8) @ Bᵀ(int8). Kept for the
//                              bit-exact bridge self-test (host int reference).
//   * int8_gemm_dequant_cfg_launch — FUSED GEMM + per-token(row) × per-channel(col)
//                              dequant in the CUTLASS epilogue, emitting bf16
//                              directly: D = out( acc·s_row[m]·s_col[n] ), out =
//                              bf16 or f16 (the f16 variant feeds the sage FP16
//                              P·V with V born f16 — no separate cast kernel).
//                              Removes the separate dequant_i32_bf16_k kernel and
//                              the intermediate i32 tensor. Templated on a tile
//                              config (TileCfg, picked per shape in Rust). Built on the SM80
//                              Epilogue Visitor Tree (Sm80EVT), modeled on CUTLASS
//                              examples/47 (ampere_gemm_universal_streamk_broadcast).
#include <cassert>  // precede any cuda fp headers (CUDA 13 __assert_fail)
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

// EVT (epilogue visitor tree) machinery for the fused path.
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>

// ---------------------------------------------------------------------------
// Raw INT8 GEMM: C(int32) = A(int8, M×K row-major) @ B(int8, N×K row-major =
// K×N col-major). Identity epilogue (alpha=1, beta=0). Kept for the self-test.
// ---------------------------------------------------------------------------
using GemmRaw = cutlass::gemm::device::Gemm<
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
  GemmRaw gemm;
  typename GemmRaw::Arguments args(
      {m, n, k},
      {a, k},  // A row-major, lda = K
      {b, k},  // B col-major, ldb = K
      {c, n},  // C row-major, ldc = N
      {c, n},
      {1, 0});  // alpha = 1, beta = 0 -> raw int32 accumulate
  cutlass::Status s = gemm(args, nullptr, stream);
  return s == cutlass::Status::kSuccess ? 0 : static_cast<int>(s);
}

// ---------------------------------------------------------------------------
// Fused INT8 GEMM + dequant epilogue (SM80 EVT).
//   D[m,n] = out( float(acc[m,n]) * s_row[m] * s_col[n] ),  out = bf16 | f16
// s_row is per-token (length M, contiguous along M → ColBroadcast).
// s_col is per-output-channel (length N, contiguous along N → RowBroadcast).
// ---------------------------------------------------------------------------
namespace convrot_evt {
using namespace cute;

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;  // weight (N,K) row-major == K×N col-major
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute = float;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;  // int8 mma.sync
constexpr int EVTEpilogueStages = 1;
constexpr int AlignmentA = 16;  // 128-bit / 8-bit
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;   // 128-bit / 16-bit (bf16 | f16) store (all convrot N are mult of 8)

// One tile configuration (qwen-image-rs-gemm-merge-tune): threadblock and
// warp tile, pipeline stages. INT8 accumulation is exact and the epilogue is
// per element, so every config computes bit-identical outputs; the choice is
// speed only (picked per shape on the Rust side, `convrot::gemm_config`).
// Swizzle = GemmIdentityThreadblockSwizzle<Swizzle>: CTAs are rastered in
// groups of up to Swizzle N-tiles so a wave reuses A (and B) tiles from L2
// instead of streaming all of A once per N-column wave.
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int Stages, class Swizzle>
struct TileCfg {
  using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
  using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
  static constexpr int NumStages = Stages;
  using Swz = Swizzle;
};
template <int N>
using Raster = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<N>;
using StreamK = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

// Everything downstream of the output element is templated on it (bf16 | f16)
// and on the tile config; ElementC doubles as the notional C/D element the
// thread map is built for.
template <typename ElementOutput, class Cfg>
struct Evt {
  using ThreadblockShape = typename Cfg::ThreadblockShape;
  using WarpShape = typename Cfg::WarpShape;
  using ElementC = ElementOutput;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

  // Fetch the int32 accumulator.
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  // Per-row activation scale s_row[m]: a column vector (varies along M, broadcast
  // along N). Stride<_1,_0,int> = M contiguous, N broadcast, batch stride = M.
  using SRow = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, float, cute::Stride<_1, _0, int32_t>>;

  // Per-channel weight scale s_col[n]: a row vector (varies along N, broadcast
  // along M). Stride<_0,_1,int> = M broadcast, N contiguous, batch stride = N.
  using SCol = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, float, cute::Stride<_0, _1, int32_t>>;

  // acc * s_row  (int32 acc and float scale converted to float).
  using MulRow = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTMulRow = cutlass::epilogue::threadblock::Sm80EVT<MulRow, Accum, SRow>;

  // (acc * s_row) * s_col.
  using MulCol = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTMulCol = cutlass::epilogue::threadblock::Sm80EVT<MulCol, EVTMulRow, SCol>;

  // Store D (ElementOutput, row-major). Stride<int64,_1,int64> = row stride N, N contiguous.
  using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, _1, int64_t>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVTMulCol>;

  using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
      ElementC, LayoutC, AlignmentC,
      ElementAccumulator,
      ElementCompute,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTD,
      typename Cfg::Swz,
      Cfg::NumStages,
      cutlass::arch::OpMultiplyAddSaturate,  // int8 tensor-op mma (no plain OpMultiplyAdd for s8)
      EVTEpilogueStages>::GemmKernel;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;
};
}  // namespace convrot_evt

// One process-lifetime device workspace shared by every launch (stream-K
// partials + barrier flags; the data-parallel configs need none). Launches are
// stream-ordered, so reuse is safe; it only grows (the old buffer is freed
// after the stream drains). The barrier part is re-zeroed by CUTLASS
// (cudaMemsetAsync in init_workspace) on every launch. 0 = ok, -1 = alloc fail.
static int workspace_for(size_t bytes, cudaStream_t stream, void** out) {
  static void* ws = nullptr;
  static size_t ws_bytes = 0;
  *out = nullptr;
  if (bytes == 0) return 0;
  if (bytes > ws_bytes) {
    if (ws) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) return -1;
      cudaFree(ws);
      ws = nullptr;
      ws_bytes = 0;
    }
    if (cudaMalloc(&ws, bytes) != cudaSuccess) return -1;
    ws_bytes = bytes;
  }
  *out = ws;
  return 0;
}

// D(out, M×N row-major) = out( acc · s_row[m] · s_col[n] ), out = bf16 | f16.
// a int8 M×K
// row-major; b int8 N×K row-major (== K×N col-major). s_row length M, s_col
// length N. Returns 0 on success, else the cutlass::Status as int.
template <typename ElementOutput, class Cfg>
static int int8_gemm_dequant_impl(
    void* d,
    const int8_t* a,
    const int8_t* b,
    const float* s_row,
    const float* s_col,
    int m, int n, int k, cudaStream_t stream) {
  using namespace convrot_evt;
  using E = Evt<ElementOutput, Cfg>;
  using EVTD = typename E::EVTD;
  using DeviceGemm = typename E::DeviceGemm;
  cutlass::gemm::GemmCoord problem(m, n, k);

  // EVT callback arguments: children first, node last, mirroring the tree.
  typename EVTD::Arguments callback_args{
      {                                                                       // EVTMulCol
          {                                                                   // EVTMulRow
              {},                                                             // Accum
              {const_cast<float*>(s_row), 0.f, {_1{}, _0{}, int32_t(m)}},     // SRow
              {}                                                              // MulRow
          },
          {const_cast<float*>(s_col), 0.f, {_0{}, _1{}, int32_t(n)}},         // SCol
          {}                                                                  // MulCol
      },
      {reinterpret_cast<ElementOutput*>(d), {int64_t(n), _1{}, int64_t(m) * n}},  // StoreD
  };

  typename DeviceGemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      1,  // batch count
      callback_args,
      static_cast<const void*>(a),
      static_cast<const void*>(b),
      nullptr,  // ptr_C unused (no source tensor)
      nullptr,  // ptr_D unused (store goes through the visitor)
      int64_t(m) * k,  // batch_stride_A
      int64_t(n) * k,  // batch_stride_B
      int64_t(0),      // batch_stride_C
      int64_t(0),      // batch_stride_D
      int64_t(k),      // lda (A row-major, MxK)
      int64_t(k),      // ldb (B col-major, KxN)
      int64_t(0),      // ldc (unused)
      int64_t(0));     // ldd (unused)

  DeviceGemm gemm;
  cutlass::Status s = gemm.can_implement(args);
  if (s != cutlass::Status::kSuccess) return static_cast<int>(s);

  void* workspace = nullptr;
  if (workspace_for(DeviceGemm::get_workspace_size(args), stream, &workspace) != 0) return -1;

  s = gemm.initialize(args, workspace, stream);
  if (s == cutlass::Status::kSuccess) s = gemm(stream);
  return s == cutlass::Status::kSuccess ? 0 : static_cast<int>(s);
}

// The tile configs, by index (TB MxNxK / warp MxNxK / stages / swizzle).
// Keep in sync with `gemm_tiles::GEMM_CONFIGS` (selection: `gemm_config`).
using convrot_evt::Raster;
using convrot_evt::StreamK;
using Cfg0 = convrot_evt::TileCfg<128, 128, 64, 64, 64, 64, 3, Raster<1>>;
using Cfg1 = convrot_evt::TileCfg<128, 128, 64, 64, 64, 64, 3, Raster<4>>;
using Cfg2 = convrot_evt::TileCfg<64, 128, 64, 32, 64, 64, 4, Raster<1>>;
using Cfg3 = convrot_evt::TileCfg<128, 256, 64, 64, 64, 64, 3, Raster<4>>;
// SPIKE (gemm-tiling-push): stream-K and alternates
using Cfg4 = convrot_evt::TileCfg<256, 128, 64, 64, 64, 64, 4, Raster<1>>;
using Cfg5 = convrot_evt::TileCfg<128, 256, 64, 64, 64, 64, 4, Raster<1>>;
using Cfg6 = convrot_evt::TileCfg<256, 128, 64, 64, 64, 64, 3, Raster<1>>;
using Cfg7 = convrot_evt::TileCfg<64, 64, 64, 32, 32, 64, 6, Raster<1>>;
using Cfg8 = convrot_evt::TileCfg<128, 256, 64, 64, 64, 64, 3, Raster<1>>;
using Cfg9 = convrot_evt::TileCfg<256, 128, 64, 64, 64, 64, 3, Raster<2>>;
using Cfg10 = convrot_evt::TileCfg<128, 128, 64, 64, 64, 64, 4, Raster<1>>;
using Cfg11 = convrot_evt::TileCfg<128, 256, 64, 64, 64, 64, 3, Raster<2>>;
static constexpr int kNumCfgs = 12;

template <typename ElementOutput>
static int int8_gemm_dequant_dispatch(
    int cfg, void* d, const int8_t* a, const int8_t* b, const float* s_row, const float* s_col,
    int m, int n, int k, cudaStream_t stream) {
#define QIR_CFG(i) \
  case i: return int8_gemm_dequant_impl<ElementOutput, Cfg##i>(d, a, b, s_row, s_col, m, n, k, stream);
  switch (cfg) {
    QIR_CFG(0) QIR_CFG(1) QIR_CFG(2) QIR_CFG(3) QIR_CFG(4) QIR_CFG(5) QIR_CFG(6) QIR_CFG(7) QIR_CFG(8) QIR_CFG(9) QIR_CFG(10) QIR_CFG(11)
  }
#undef QIR_CFG
  return -2;  // unknown config
}

// D(out, M×N row-major) = out( acc · s_row[m] · s_col[n] ) with tile config
// `cfg` (0..kNumCfgs). out_kind 0 = bf16, 1 = f16 (rounded once from the f32
// epilogue value; feeds the sage FP16 P·V with V born f16). Returns 0 on
// success, -2 for an unknown cfg / out_kind, else the cutlass::Status as int.
extern "C" int int8_gemm_dequant_cfg_launch(
    int out_kind, int cfg, void* d, const int8_t* a, const int8_t* b, const float* s_row,
    const float* s_col, int m, int n, int k, cudaStream_t stream) {
  if (cfg < 0 || cfg >= kNumCfgs) return -2;
  switch (out_kind) {
    case 0:
      return int8_gemm_dequant_dispatch<cutlass::bfloat16_t>(cfg, d, a, b, s_row, s_col, m, n, k, stream);
    case 1:
      return int8_gemm_dequant_dispatch<cutlass::half_t>(cfg, d, a, b, s_row, s_col, m, n, k, stream);
    default:
      return -2;
  }
}

// ---------------------------------------------------------------------------
// SPIKE (gemm-tiling-push): epilogue cost probes. variant 0 = stock device::Gemm
// 128x256x64 s3, int32 row-major out; 1 = EVT store-only (acc -> bf16, no
// scales) 128x256x64 s3; 2 = EVT store-only 128x128x64 s3.
// ---------------------------------------------------------------------------
using GemmRaw256 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, int32_t,
    cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 128 / 32, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

namespace convrot_evt {
template <class Cfg>
struct EvtStoreOnly {
  using ThreadblockShape = typename Cfg::ThreadblockShape;
  using WarpShape = typename Cfg::WarpShape;
  using ElementC = cutlass::bfloat16_t;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementC, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, _1, int64_t>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, Accum>;
  using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB,
      cutlass::ComplexTransform::kNone, AlignmentB, ElementC, LayoutC, AlignmentC,
      ElementAccumulator, ElementCompute, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EVTD, typename Cfg::Swz, Cfg::NumStages,
      cutlass::arch::OpMultiplyAddSaturate, EVTEpilogueStages>::GemmKernel;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;
};
}  // namespace convrot_evt

template <class Cfg>
static int spike_store_only(void* d, const int8_t* a, const int8_t* b, int m, int n, int k,
                            cudaStream_t stream) {
  using namespace convrot_evt;
  using E = EvtStoreOnly<Cfg>;
  using DeviceGemm = typename E::DeviceGemm;
  typename E::EVTD::Arguments cb{{}, {reinterpret_cast<cutlass::bfloat16_t*>(d), {int64_t(n), _1{}, int64_t(m) * n}}};
  typename DeviceGemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, cb,
                                      a, b, nullptr, nullptr, int64_t(m) * k, int64_t(n) * k, 0, 0,
                                      int64_t(k), int64_t(k), 0, 0);
  DeviceGemm gemm;
  cutlass::Status s = gemm.can_implement(args);
  if (s == cutlass::Status::kSuccess) s = gemm.initialize(args, nullptr, stream);
  if (s == cutlass::Status::kSuccess) s = gemm(stream);
  return s == cutlass::Status::kSuccess ? 0 : static_cast<int>(s);
}

extern "C" int int8_gemm_spike_launch(int variant, void* d, const int8_t* a, const int8_t* b,
                                      int m, int n, int k, cudaStream_t stream) {
  switch (variant) {
    case 0: {
      GemmRaw256 gemm;
      typename GemmRaw256::Arguments args({m, n, k}, {a, k}, {b, k},
                                          {static_cast<int32_t*>(d), n},
                                          {static_cast<int32_t*>(d), n}, {1, 0});
      cutlass::Status s = gemm(args, nullptr, stream);
      return s == cutlass::Status::kSuccess ? 0 : static_cast<int>(s);
    }
    case 1: return spike_store_only<Cfg8>(d, a, b, m, n, k, stream);
    case 2: return spike_store_only<Cfg0>(d, a, b, m, n, k, stream);
    default: return -2;
  }
}
