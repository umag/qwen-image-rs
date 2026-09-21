// Small element-wise quant/dequant kernels for ConvRot, routing around candle's
// missing signed-int8 emission and i32->f32 cast on this CUDA build.
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

// out[m,n] = bf16(c[m,n] * row_scale[m] * col_scale[n]).
__global__ void dequant_i32_bf16_k(
    __nv_bfloat16* out, const int* c, const float* row_scale, const float* col_scale, long M, long N) {
  long total = M * N;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    long m = idx / N, n = idx % N;
    float v = (float)c[idx] * row_scale[m] * col_scale[n];
    out[idx] = __float2bfloat16(v);
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

extern "C" void dequant_i32_bf16(
    __nv_bfloat16* out, const int* c, const float* row_scale, const float* col_scale, int M, int N,
    cudaStream_t s) {
  long total = (long)M * N;
  int t = 256;
  dequant_i32_bf16_k<<<grid_for(total, t), t, 0, s>>>(out, c, row_scale, col_scale, M, N);
}
