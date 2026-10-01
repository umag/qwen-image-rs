// Fused LayerNorm(no-affine) + AdaLN (scale+1) modulation for the DiT.
// out[m,n] = ((x[m,n] - mean_m) * rsqrt(var_m + eps)) * (scale[m,n] + 1)
// where mean_m/var_m are over the row (last dim, N elements). bf16 in/out,
// f32 accumulation — mirrors `norm_no_affine(x) * (scale+1)` in src/model/dit.rs.
// One CTA per row; the row is staged in shared memory (N f32) and reduced in
// two f32 block passes (mean, then var). Only compiled under the `fusednorm`
// feature. Called from Rust (src/fusednorm.rs) via a candle CustomOp2.

#include <cassert> // precede any cuda fp headers (CUDA 13 __assert_fail)
#include <cstdint>
#include <cuda_bf16.h>

#include "head_rmsnorm.cuh"

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
                           const float *__restrict__ w, int n, long ld,
                           float eps) {
  extern __shared__ float sh[]; // n floats: the row staged as f32
  __shared__ float red[THREADS];

  const int row = blockIdx.x;
  const size_t base = (size_t)row * n;    // out: dense rows
  const size_t xbase = (size_t)row * ld;  // x: rows ld >= n elements apart
  const int tid = threadIdx.x;

  // Stage the row as f32 and accumulate a partial sum of squares.
  float local = 0.f;
  for (int i = tid; i < n; i += THREADS) {
    float v = __bfloat162float(x[xbase + i]);
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

// N = 128 (the per-head q/k norm): an aligned 16-lane group per row, 8
// channels per lane (one 16-B load + one 16-B store), 16 rows per 256-thread
// CTA. The CTA-per-row kernel above leaves half its 256 threads idle at N=128
// and pays a block barrier tree per row (~295 GB/s); this one has no shared
// memory and no barriers. Bit-identical to it (head_rmsnorm.cuh: same f32
// sum-of-squares order, same rsqrt/scale expression).
constexpr int R128_THREADS = 256;
constexpr int R128_ROWS = R128_THREADS / 16;
__global__ void __launch_bounds__(R128_THREADS)
fused_rmsnorm128_kernel(__nv_bfloat16 *__restrict__ out,
                        const __nv_bfloat16 *__restrict__ x,
                        const float *__restrict__ w, int m, long ld, float eps) {
  const int row = blockIdx.x * R128_ROWS + threadIdx.x / 16;
  const int d0 = (threadIdx.x % 16) * 8;
  const bool live = row < m; // uniform across the row's 16-lane group
  float v[8];
  if (live) {
    __nv_bfloat16 xb[8];
    *(uint4 *)xb = *(const uint4 *)(x + (size_t)row * ld + d0);
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = __bfloat162float(xb[j]);
  } else {
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = 0.f;
  }
  __nv_bfloat16 o[8];
  head_rmsnorm8(v, w + d0, eps, o); // the whole group takes part in the shuffles
  if (live) *(uint4 *)(out + (size_t)row * 128 + d0) = *(const uint4 *)o;
}

static void rmsnorm_block(void *out, const void *x, const void *w, int m, int n,
                          long ld, float eps, cudaStream_t st) {
  size_t shmem = (size_t)n * sizeof(float);
  fused_rmsnorm_scale_kernel<<<m, THREADS, shmem, st>>>(
      (__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, (const float *)w, n, ld, eps);
}

// out is (M, N) bf16 row-major, dense; x rows are `ld` (>= N) elements apart
// (ld = N for a dense x; ld > N for a column view such as the q or k half of
// the head-interleaved merged q|k output); w is (N,) f32. M rows, N cols.
// N = 128 with a 16-B aligned x and ld % 8 == 0 takes the 16-lane kernel;
// anything else the CTA-per-row kernel (same bits either way).
extern "C" void fused_rmsnorm_scale_launch(void *out, const void *x,
                                           const void *w, int m, int n,
                                           long ld, float eps, void *stream) {
  cudaStream_t st = (cudaStream_t)stream;
  if (n == 128 && (uintptr_t)x % 16 == 0 && ld % 8 == 0 && (uintptr_t)out % 16 == 0) {
    fused_rmsnorm128_kernel<<<(m + R128_ROWS - 1) / R128_ROWS, R128_THREADS, 0, st>>>(
        (__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, (const float *)w, m, ld, eps);
  } else {
    rmsnorm_block(out, x, w, m, n, ld, eps, st);
  }
}

// The CTA-per-row kernel unconditionally: the bit-identity oracle for the
// N = 128 kernel in fusednorm-test.
extern "C" void fused_rmsnorm_scale_block_launch(void *out, const void *x,
                                                 const void *w, int m, int n,
                                                 long ld, float eps, void *stream) {
  rmsnorm_block(out, x, w, m, n, ld, eps, (cudaStream_t)stream);
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
