// Fused LayerNorm(no-affine) + AdaLN (scale+1) modulation for the DiT.
// out[m,n] = ((x[m,n] - mean_m) * rsqrt(var_m + eps)) * (scale[m,n] + 1)
// where mean_m/var_m are over the row (last dim, N elements). bf16 in/out,
// f32 accumulation — mirrors `norm_no_affine(x) * (scale+1)` in src/model/dit.rs.
// One CTA per row; the row is staged in shared memory (N f32) and reduced in
// two f32 block passes (mean, then var). Only compiled under the `fusednorm`
// feature. Called from Rust (src/fusednorm.rs) via a candle CustomOp2.

#include <cassert> // precede any cuda fp headers (CUDA 13 __assert_fail)
#include <cuda_bf16.h>

static constexpr int THREADS = 256;

__global__ void fused_norm_mod_kernel(__nv_bfloat16 *__restrict__ out,
                                      const __nv_bfloat16 *__restrict__ x,
                                      const __nv_bfloat16 *__restrict__ scale,
                                      int n, float eps) {
  extern __shared__ float sh[]; // n floats: the row staged as f32
  __shared__ float red[THREADS];

  const int row = blockIdx.x;
  const size_t base = (size_t)row * n;
  const int tid = threadIdx.x;

  // Stage the row as f32 and accumulate a partial sum for the mean.
  float local = 0.f;
  for (int i = tid; i < n; i += THREADS) {
    float v = __bfloat162float(x[base + i]);
    sh[i] = v;
    local += v;
  }
  red[tid] = local;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float mean = red[0] / (float)n;
  __syncthreads();

  // Second pass: variance.
  local = 0.f;
  for (int i = tid; i < n; i += THREADS) {
    float d = sh[i] - mean;
    local += d * d;
  }
  red[tid] = local;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float rstd = rsqrtf(red[0] / (float)n + eps);
  __syncthreads();

  // Normalize + AdaLN affine (scale + 1).
  for (int i = tid; i < n; i += THREADS) {
    float normed = (sh[i] - mean) * rstd;
    float s = __bfloat162float(scale[base + i]) + 1.0f;
    out[base + i] = __float2bfloat16(normed * s);
  }
}

// out/x/scale are (M, N) bf16 row-major, contiguous. M rows, N cols.
extern "C" void fused_norm_mod_launch(void *out, const void *x,
                                      const void *scale, int m, int n, float eps,
                                      void *stream) {
  size_t shmem = (size_t)n * sizeof(float);
  fused_norm_mod_kernel<<<m, THREADS, shmem, (cudaStream_t)stream>>>(
      (__nv_bfloat16 *)out, (const __nv_bfloat16 *)x,
      (const __nv_bfloat16 *)scale, n, eps);
}

// Fused RMSNorm(no zero-centering) * per-channel weight, for the DiT's RMSNorm
// variants. out[m,n] = x[m,n] * rsqrt(mean_n(x[m,:]^2) + eps) * W[n], where W is
// a per-COLUMN weight vector of length N (f32). Mirrors
// `x * (1/sqrt(mean(x^2)+eps)) * W` in ZeroCenterRmsNorm/HeadRmsNorm
// (src/model/dit.rs). bf16 x/out, f32 accumulation. One CTA per row; the row is
// staged in shared memory (N f32) and reduced in one f32 sum-of-squares pass.
// W rides as f32 so it matches candle's `weight.to_dtype(F32) (+1)` exactly
// (ZeroCenter bakes weight+1 in f32 at load; Head passes weight upcast to f32).
__global__ void
fused_rmsnorm_scale_kernel(__nv_bfloat16 *__restrict__ out,
                           const __nv_bfloat16 *__restrict__ x,
                           const float *__restrict__ w, int n, float eps) {
  extern __shared__ float sh[]; // n floats: the row staged as f32
  __shared__ float red[THREADS];

  const int row = blockIdx.x;
  const size_t base = (size_t)row * n;
  const int tid = threadIdx.x;

  // Stage the row as f32 and accumulate a partial sum of squares.
  float local = 0.f;
  for (int i = tid; i < n; i += THREADS) {
    float v = __bfloat162float(x[base + i]);
    sh[i] = v;
    local += v * v;
  }
  red[tid] = local;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float rrms = rsqrtf(red[0] / (float)n + eps);
  __syncthreads();

  // Normalize and scale by the per-column weight.
  for (int i = tid; i < n; i += THREADS) {
    out[base + i] = __float2bfloat16(sh[i] * rrms * w[i]);
  }
}

// out/x are (M, N) bf16 row-major, contiguous; w is (N,) f32. M rows, N cols.
extern "C" void fused_rmsnorm_scale_launch(void *out, const void *x,
                                           const void *w, int m, int n,
                                           float eps, void *stream) {
  size_t shmem = (size_t)n * sizeof(float);
  fused_rmsnorm_scale_kernel<<<m, THREADS, shmem, (cudaStream_t)stream>>>(
      (__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, (const float *)w, n, eps);
}

// Fused gated residual: out[i] = h[i] + tanh(gate[i]) * y[i], elementwise over
// all `total` elements. Mirrors `h + gate.tanh() * y` in Block::forward
// (src/model/dit.rs), used twice per block. bf16 in/out, f32 tanh + fma.
__global__ void
fused_gated_residual_kernel(__nv_bfloat16 *__restrict__ out,
                            const __nv_bfloat16 *__restrict__ h,
                            const __nv_bfloat16 *__restrict__ gate,
                            const __nv_bfloat16 *__restrict__ y, size_t total) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float hv = __bfloat162float(h[i]);
    float gv = __bfloat162float(gate[i]);
    float yv = __bfloat162float(y[i]);
    out[i] = __float2bfloat16(hv + tanhf(gv) * yv);
  }
}

// out/h/gate/y are contiguous bf16 with `total` elements (same shape).
extern "C" void fused_gated_residual_launch(void *out, const void *h,
                                            const void *gate, const void *y,
                                            size_t total, void *stream) {
  const int threads = 256;
  size_t blocks = (total + threads - 1) / threads;
  fused_gated_residual_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      (__nv_bfloat16 *)out, (const __nv_bfloat16 *)h,
      (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)y, total);
}
