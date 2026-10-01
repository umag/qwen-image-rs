// Small element-wise quant/dequant kernels for ConvRot, routing around candle's
// missing signed-int8 emission and i32->f32 cast on this CUDA build.
#include <cassert> // precede cuda fp headers (CUDA 13 __assert_fail)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// out[m,k] = clamp(round(x[m,k] * inv_scale[m]), -127, 127), stored as int8 bytes
// in a uint8 buffer. Per-row (per-token) scale.
__global__ void quantize_rows_i8_k(
    unsigned char* out, const __nv_bfloat16* x, const float* inv_scale, long M, long K) {
  long total = M * K;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    long row = idx / K;
    float v = __bfloat162float(x[idx]) * inv_scale[row];
    int q = __float2int_rn(v);
    q = max(-127, min(127, q));
    out[idx] = (unsigned char)(signed char)q;
  }
}

// (The i32->bf16 dequant is now fused into the INT8 GEMM's CUTLASS epilogue —
// see int8_gemm_dequant_bf16 in int8_gemm.cu — so the standalone dequant kernel
// and its launcher have been removed.)

// Fused per-row activation quantize: one CTA per row computes amax = max(|x|)
// over K, then scale = amax/127, then quantizes each element to int8 — in a
// single pass. Emits int8 bytes (out) and writes the per-row f32 scale
// (row_scale, pre-allocated). Replaces candle abs + max_keepdim + recip +
// the plain quantize_rows_i8 kernel inside ConvRotLinear::forward.
static constexpr int QFUSE_THREADS = 256;

__global__ void quantize_rows_fused_k(
    unsigned char* out, const __nv_bfloat16* x, float* row_scale, long K) {
  __shared__ float red[QFUSE_THREADS];
  long row = blockIdx.x;
  const __nv_bfloat16* xr = x + row * K;
  unsigned char* orow = out + row * K;
  int tid = threadIdx.x;

  float local = 0.f;
  for (long i = tid; i < K; i += QFUSE_THREADS) {
    local = fmaxf(local, fabsf(__bfloat162float(xr[i])));
  }
  red[tid] = local;
  __syncthreads();
  for (int s = QFUSE_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float amax = red[0];
  float scale = amax * (1.0f / 127.0f);
  float inv = (amax > 0.f) ? (127.0f / amax) : 0.f; // zero row -> int8 zeros
  if (tid == 0) row_scale[row] = scale;
  for (long i = tid; i < K; i += QFUSE_THREADS) {
    int q = __float2int_rn(__bfloat162float(xr[i]) * inv);
    q = max(-127, min(127, q));
    orow[i] = (unsigned char)(signed char)q;
  }
}

// ---------------------------------------------------------------------------
// Fused rotate + quantize (qwen-image-rs-fused-rotate-quant).
//
// ConvRot rotates each 256-wide chunk of an activation row by the Regular
// Hadamard R256 = (H4/2)^{⊗4} (src/model/rotation.rs: R[i][j] = Π_d r4(i_d, j_d)
// over the base-4 digits d of i and j, r4 = -1/2 on the diagonal, +1/2 off it).
// R is symmetric, so x Rᵀ = R x per chunk, and the Kronecker factor along digit
// d maps the four elements that share every other digit as
//     y_a = (x_0 + x_1 + x_2 + x_3) / 2 - x_a.
// One warp transforms one chunk: lane l holds elements i = 8l + e (e = 0..7),
// so i bits 0-2 = e, bits 3-7 = lane bits 0-4. Digit 0 (i bits 0-1) is
// in-register, digit 1 (bits 2-3) = e bit 2 + lane bit 0, digit 2 = lane bits
// 1-2, digit 3 = lane bits 3-4 (xor shuffles). All in f32; the old path ran
// the rotation as a bf16 GEMM with a bf16-rounded output.
__device__ __forceinline__ void hadamard256_warp(float (&v)[8]) {
  const unsigned FULL = 0xffffffffu;
  // digit 0: e bits 0-1
#pragma unroll
  for (int g = 0; g < 8; g += 4) {
    float s = (v[g] + v[g + 1]) + (v[g + 2] + v[g + 3]);
#pragma unroll
    for (int a = 0; a < 4; ++a) v[g + a] = 0.5f * s - v[g + a];
  }
  // digit 1: e bit 2 and lane bit 0
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    float p = v[e] + v[e + 4];
    float s = p + __shfl_xor_sync(FULL, p, 1);
    v[e] = 0.5f * s - v[e];
    v[e + 4] = 0.5f * s - v[e + 4];
  }
  // digit 2: lane bits 1-2; digit 3: lane bits 3-4
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float t = v[e] + __shfl_xor_sync(FULL, v[e], 2);
    float s = t + __shfl_xor_sync(FULL, t, 4);
    v[e] = 0.5f * s - v[e];
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float t = v[e] + __shfl_xor_sync(FULL, v[e], 8);
    float s = t + __shfl_xor_sync(FULL, t, 16);
    v[e] = 0.5f * s - v[e];
  }
}

static constexpr int RQ_WARPS = 8;
static constexpr int RQ_MAX_CPW = 8; // chunks per warp kept in registers -> K <= 16384

// One CTA per row. Warp w handles chunks w, w + 8, ... (up to CPW of them),
// keeping the rotated f32 values in registers until the row amax is known.
// Same output contract as quantize_rows_fused_k: int8 bytes + row_scale[row]
// = amax/127 (0 for an all-zero row, which quantizes to zeros).
template <int CPW>
__global__ void __launch_bounds__(RQ_WARPS * 32) rotate_quantize_rows_k(
    unsigned char* out, const __nv_bfloat16* x, float* row_scale, int K) {
  __shared__ float wmax[RQ_WARPS];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long row = blockIdx.x;
  const int nchunks = K >> 8;
  const __nv_bfloat16* xr = x + row * (long)K;
  unsigned char* orow = out + row * (long)K;

  float v[CPW][8];
  float local = 0.f;
#pragma unroll
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp + RQ_WARPS * c;
    // `chunk` depends only on the warp id: this branch is warp-uniform, so the
    // full-mask shuffles inside hadamard256_warp are safe.
    if (chunk < nchunks) {
      uint4 raw = *reinterpret_cast<const uint4*>(xr + chunk * 256 + lane * 8);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float2 f = __bfloat1622float2(h[e]);
        v[c][2 * e] = f.x;
        v[c][2 * e + 1] = f.y;
      }
      hadamard256_warp(v[c]);
#pragma unroll
      for (int e = 0; e < 8; ++e) local = fmaxf(local, fabsf(v[c][e]));
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
  if (lane == 0) wmax[warp] = local;
  __syncthreads();
  float amax = wmax[0];
#pragma unroll
  for (int w = 1; w < RQ_WARPS; ++w) amax = fmaxf(amax, wmax[w]);
  const float scale = amax * (1.0f / 127.0f);
  const float inv = (amax > 0.f) ? (127.0f / amax) : 0.f; // zero row -> int8 zeros
  if (threadIdx.x == 0) row_scale[row] = scale;
#pragma unroll
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp + RQ_WARPS * c;
    if (chunk < nchunks) {
      unsigned int w[2] = {0u, 0u};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int q = __float2int_rn(v[c][e] * inv);
        q = max(-127, min(127, q));
        w[e >> 2] |= (unsigned int)(unsigned char)(signed char)q << (8 * (e & 3));
      }
      *reinterpret_cast<uint2*>(orow + chunk * 256 + lane * 8) = make_uint2(w[0], w[1]);
    }
  }
}

static inline int grid_for(long total, int t) {
  long b = (total + t - 1) / t;
  return (int)(b > 65535 ? 65535 : b);
}

extern "C" void quantize_rows_i8(
    unsigned char* out, const __nv_bfloat16* x, const float* inv_scale, int M, int K, cudaStream_t s) {
  long total = (long)M * K;
  int t = 256;
  quantize_rows_i8_k<<<grid_for(total, t), t, 0, s>>>(out, x, inv_scale, M, K);
}

extern "C" void quantize_rows_fused_launch(
    unsigned char* out, const __nv_bfloat16* x, float* row_scale, int M, int K, cudaStream_t s) {
  quantize_rows_fused_k<<<M, QFUSE_THREADS, 0, s>>>(out, x, row_scale, (long)K);
}

// Fused rotate + quantize. Returns 0 on success, 1 for an unsupported K
// (K % 256 != 0 or K > 256 * RQ_WARPS * RQ_MAX_CPW), 2 for a misaligned x
// (16-B vector loads) or out (8-B stores), else the CUDA launch error.
extern "C" int rotate_quantize_rows_launch(
    unsigned char* out, const __nv_bfloat16* x, float* row_scale, int M, int K, cudaStream_t s) {
  if (K <= 0 || K % 256 != 0 || K > 256 * RQ_WARPS * RQ_MAX_CPW) return 1;
  if (((size_t)x & 15) != 0 || ((size_t)out & 7) != 0) return 2;
  if (M <= 0) return 0;
  const int nchunks = K / 256;
  const int cpw = (nchunks + RQ_WARPS - 1) / RQ_WARPS;
  const int t = RQ_WARPS * 32;
  switch (cpw) {
    case 1: rotate_quantize_rows_k<1><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 2: rotate_quantize_rows_k<2><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 3: rotate_quantize_rows_k<3><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 4: rotate_quantize_rows_k<4><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 5: rotate_quantize_rows_k<5><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 6: rotate_quantize_rows_k<6><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 7: rotate_quantize_rows_k<7><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    case 8: rotate_quantize_rows_k<8><<<M, t, 0, s>>>(out, x, row_scale, K); break;
    default: return 1;
  }
  return (int)cudaGetLastError();
}
