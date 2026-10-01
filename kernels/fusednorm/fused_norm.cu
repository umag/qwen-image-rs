// Fused LayerNorm(no-affine) + AdaLN (scale+1) modulation for the DiT.
// (The DiT now calls residual_norm_mod_kernel at the bottom of this file; this
// kernel and fused_gated_residual_kernel remain as its bit-identity oracles.)
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

// Fused gated residual + LayerNorm(no-affine) + AdaLN (scale+1), one CTA per
// row (`qwen-image-rs-residual-norm-fusion`). With a residual (HAS_RES):
//   h_out[r,:] = bf16(h + tanh(gate) * y)                       (the new stream)
//   x_out[r,:] = bf16(LN(float(h_out[r,:])) * (scale + 1))
// without one, x_out = LN(h) * (scale + 1). gate/scale are the DiT's (2, n)
// modulation rows, `gld`/`sld` elements apart: row 1 for text tokens
// (pos < txt_len), row 0 for image tokens, pos = r % seq (B lanes share it).
// Bit-identical to fused_gated_residual_kernel followed by
// fused_norm_mod_kernel on the materialized per-token tensors: the same h'
// expression (same TU and flags -> same fma contraction), and after a barrier
// each thread sums the same strided set sh[tid + k*THREADS] in k order with the
// same tree, so mean/var match to the bit. VEC stages 8 elements per 16-B load
// (needs 16-B aligned pointers, n % 8 == 0, ld % 8 == 0); the reduction is
// order-identical either way.
template <bool HAS_RES, bool VEC>
__global__ void __launch_bounds__(THREADS)
residual_norm_mod_kernel(__nv_bfloat16 *__restrict__ x_out,
                         __nv_bfloat16 *__restrict__ h_out,
                         const __nv_bfloat16 *__restrict__ h,
                         const __nv_bfloat16 *__restrict__ gate, long gld,
                         const __nv_bfloat16 *__restrict__ y,
                         const __nv_bfloat16 *__restrict__ scale, long sld,
                         int n, int seq, int txt_len, float eps) {
  extern __shared__ float sh[]; // n floats: the (rounded) row staged as f32
  __shared__ float red[THREADS];

  const int row = blockIdx.x;
  const size_t base = (size_t)row * n;
  const int tid = threadIdx.x;
  const long sel = (row % seq) < txt_len ? 1 : 0;
  const __nv_bfloat16 *g = gate + sel * gld; // unused without a residual
  const __nv_bfloat16 *sc = scale + sel * sld;

  // Stage h' (or h) as f32; write h' as the next residual stream.
  if (VEC) {
    for (int c = tid; c < n / 8; c += THREADS) {
      const int i = c * 8;
      __nv_bfloat16 hb[8];
      float v[8];
      *(uint4 *)hb = *(const uint4 *)(h + base + i);
      if (HAS_RES) {
        __nv_bfloat16 gb[8], yb[8], ob[8];
        *(uint4 *)gb = *(const uint4 *)(g + i);
        *(uint4 *)yb = *(const uint4 *)(y + base + i);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          float hv = __bfloat162float(hb[j]);
          float gv = __bfloat162float(gb[j]);
          float yv = __bfloat162float(yb[j]);
          ob[j] = __float2bfloat16(hv + tanhf(gv) * yv);
          v[j] = __bfloat162float(ob[j]);
        }
        *(uint4 *)(h_out + base + i) = *(const uint4 *)ob;
      } else {
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = __bfloat162float(hb[j]);
      }
      *(float4 *)(sh + i) = *(const float4 *)v;
      *(float4 *)(sh + i + 4) = *(const float4 *)(v + 4);
    }
  } else {
    for (int i = tid; i < n; i += THREADS) {
      float v = __bfloat162float(h[base + i]);
      if (HAS_RES) {
        float gv = __bfloat162float(g[i]);
        float yv = __bfloat162float(y[base + i]);
        __nv_bfloat16 o = __float2bfloat16(v + tanhf(gv) * yv);
        h_out[base + i] = o;
        v = __bfloat162float(o);
      }
      sh[i] = v;
    }
  }
  __syncthreads();

  // Mean: per-thread strided sum in fused_norm_mod_kernel's order, same tree.
  float local = 0.f;
  for (int i = tid; i < n; i += THREADS) local += sh[i];
  red[tid] = local;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float mean = red[0] / (float)n;
  __syncthreads();

  // Variance.
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

  // Normalize + AdaLN affine (scale + 1). Elementwise: any mapping is exact.
  if (VEC) {
    for (int c = tid; c < n / 8; c += THREADS) {
      const int i = c * 8;
      __nv_bfloat16 sb[8], ob[8];
      float v[8];
      *(uint4 *)sb = *(const uint4 *)(sc + i);
      *(float4 *)v = *(const float4 *)(sh + i);
      *(float4 *)(v + 4) = *(const float4 *)(sh + i + 4);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float normed = (v[j] - mean) * rstd;
        float s = __bfloat162float(sb[j]) + 1.0f;
        ob[j] = __float2bfloat16(normed * s);
      }
      *(uint4 *)(x_out + base + i) = *(const uint4 *)ob;
    }
  } else {
    for (int i = tid; i < n; i += THREADS) {
      float normed = (sh[i] - mean) * rstd;
      float s = __bfloat162float(sc[i]) + 1.0f;
      x_out[base + i] = __float2bfloat16(normed * s);
    }
  }
}

template <bool HAS_RES, bool VEC>
static void residual_norm_mod(void *x_out, void *h_out, const void *h,
                              const void *gate, long gld, const void *y,
                              const void *scale, long sld, int m, int n,
                              int seq, int txt_len, float eps, cudaStream_t st) {
  size_t shmem = (size_t)n * sizeof(float);
  residual_norm_mod_kernel<HAS_RES, VEC><<<m, THREADS, shmem, st>>>(
      (__nv_bfloat16 *)x_out, (__nv_bfloat16 *)h_out,
      (const __nv_bfloat16 *)h, (const __nv_bfloat16 *)gate, gld,
      (const __nv_bfloat16 *)y, (const __nv_bfloat16 *)scale, sld, n, seq,
      txt_len, eps);
}

static bool al16(const void *p) { return (uintptr_t)p % 16 == 0; }

// x_out (and h_out when y != null) are (m, n) bf16 dense; h, y are (m, n) bf16
// dense; gate/scale point at modulation row 0 with row 1 `gld`/`sld` elements
// further. y == null: no residual (gate and h_out unused). m rows = B * seq.
// `scalar` forces the scalar staging path (bit-identity test). Returns the
// launch's cudaError (0 on success).
extern "C" int fused_residual_norm_mod_launch(
    void *x_out, void *h_out, const void *h, const void *gate, long gld,
    const void *y, const void *scale, long sld, int m, int n, int seq,
    int txt_len, float eps, int scalar, void *stream) {
  cudaStream_t st = (cudaStream_t)stream;
  const bool res = y != nullptr;
  bool vec = !scalar && n % 8 == 0 && sld % 8 == 0 && al16(x_out) && al16(h) &&
             al16(scale);
  if (res) vec = vec && gld % 8 == 0 && al16(h_out) && al16(gate) && al16(y);
  if (res && vec)
    residual_norm_mod<true, true>(x_out, h_out, h, gate, gld, y, scale, sld, m, n, seq, txt_len, eps, st);
  else if (res)
    residual_norm_mod<true, false>(x_out, h_out, h, gate, gld, y, scale, sld, m, n, seq, txt_len, eps, st);
  else if (vec)
    residual_norm_mod<false, true>(x_out, h_out, h, gate, gld, y, scale, sld, m, n, seq, txt_len, eps, st);
  else
    residual_norm_mod<false, false>(x_out, h_out, h, gate, gld, y, scale, sld, m, n, seq, txt_len, eps, st);
  return (int)cudaGetLastError();
}
