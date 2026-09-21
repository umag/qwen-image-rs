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
