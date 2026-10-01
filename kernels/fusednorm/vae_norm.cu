// VAE decoder fused ops (feature `fusednorm`; bridges in src/vae_fused.rs).
//
// 1. vae_rmsnorm_k: QwenImage21RMS_norm over the channel dim of an NCHW bf16
//    tensor, times gamma, optionally followed by SiLU — byte-identical to the
//    candle chain in src/model/vae.rs RmsNorm::forward (+ candle_nn silu):
//      xf = f32(x); s = fast_sum_f32(xf*xf over C); n = max(sqrt(s), 1e-12)
//      y = bf16(xf / n * sqrt(C) + 0) * gamma          (bf16 multiply)
//      y = y / (1 + hexp(-y))                          (bf16 ops, if silu)
//    candle's fast_sum runs one block per pixel with N = next_pow2(min(1024,C))
//    threads: thread t holds 0 + v[t] (+ v[t+1024] when C > 1024), then a
//    shared-memory tree `shr[t] += shr[t+s]`, s = N/2 .. 1. That association
//    equals a balanced adjacent-pair tree over the leaves in BIT-REVERSED
//    order, so here every aligned chunk of that order is a complete subtree:
//    a CTA is 32 pixels (x) by G lanes (y); lane g sums chunk g of the
//    bit-reversed order with a compile-time pairwise recursion, and the G
//    partials combine pairwise in lane order. Explicit _rn intrinsics keep nvcc
//    from contracting into FMAs (candle-kernels are built without fast-math).
//    Adjacent threads = adjacent pixels, so every channel load is coalesced.
//
// 2. vae_bias_residual_k: out = bf16(bf16(y + bias[c]) + r) — the conv bias
//    broadcast-add then the residual add, both bf16 roundings kept, one pass.

#include <cassert> // precede any cuda fp headers (CUDA 13 __assert_fail)
#include <cstdint>
#include <cuda_bf16.h>

static constexpr int PX = 32; // pixels per CTA (threadIdx.x)
static constexpr int G = 8;   // channel lanes per pixel (threadIdx.y)

struct NormArgs {
  const __nv_bfloat16 *x;
  const __nv_bfloat16 *gamma;
  __nv_bfloat16 *out;
  long long pixels; // B * H * W
  long long hw;     // H * W
  int c;            // channels
  float scale;      // (float)sqrt((double)C), as candle's affine gets it
};

__device__ __forceinline__ size_t addr(const NormArgs &a, long long p, int ch) {
  const long long b = p / a.hw, s = p - b * a.hw;
  return (size_t)((b * a.c + ch) * a.hw + s);
}

__device__ __forceinline__ float sq(const NormArgs &a, long long p, int ch) {
  const float v = __bfloat162float(a.x[addr(a, p, ch)]);
  return __fmul_rn(v, v);
}

// Leaf k of the bit-reversed order (LOG2N bits): candle thread t = brev(k)
// holds 0 + v[t] (+ v[t + 1024]); channels >= C contribute 0.
template <int LOG2N>
__device__ __forceinline__ float leaf(const NormArgs &a, long long p, int k) {
  const int t = (int)(__brev((unsigned)k) >> (32 - LOG2N));
  float s = 0.f;
  if (t < a.c) s = __fadd_rn(s, sq(a, p, t));
  if (LOG2N == 10 && t + 1024 < a.c) s = __fadd_rn(s, sq(a, p, t + 1024));
  return s;
}

// Sum of leaves [k0, k0 + M) as the balanced adjacent-pair tree.
template <int LOG2N, int M>
__device__ __forceinline__ float subtree(const NormArgs &a, long long p, int k0) {
  if constexpr (M == 1) {
    return leaf<LOG2N>(a, p, k0);
  } else {
    const float l = subtree<LOG2N, M / 2>(a, p, k0);
    const float r = subtree<LOG2N, M / 2>(a, p, k0 + M / 2);
    return __fadd_rn(l, r);
  }
}

template <int LOG2N, bool SILU>
__global__ void __launch_bounds__(PX *G) vae_rmsnorm_k(NormArgs a) {
  constexpr int N = 1 << LOG2N;
  constexpr int M = N / G; // leaves per lane (N >= G)
  __shared__ float part[G][PX];
  __shared__ float norm_s[PX];
  const int px = threadIdx.x, g = threadIdx.y;
  const long long p = (long long)blockIdx.x * PX + px;
  const bool live = p < a.pixels;

  part[g][px] = live ? subtree<LOG2N, M>(a, p, g * M) : 0.f;
  __syncthreads();
  if (g == 0) {
    // Lanes are consecutive chunks of the bit-reversed order: pairwise.
    float v[G];
#pragma unroll
    for (int i = 0; i < G; ++i) v[i] = part[i][px];
#pragma unroll
    for (int w = 1; w < G; w <<= 1)
#pragma unroll
      for (int i = 0; i < G; i += 2 * w) v[i] = __fadd_rn(v[i], v[i + w]);
    norm_s[px] = fmaxf(__fsqrt_rn(v[0]), 1e-12f);
  }
  __syncthreads();
  if (!live) return;
  const float n = norm_s[px];
  const __nv_bfloat16 one = static_cast<__nv_bfloat16>(1);
  for (int ch = g; ch < a.c; ch += G) {
    const size_t i = addr(a, p, ch);
    const float xf = __bfloat162float(a.x[i]);
    const float v = __fadd_rn(__fmul_rn(__fdiv_rn(xf, n), a.scale), 0.f);
    __nv_bfloat16 y = __float2bfloat16_rn(v) * a.gamma[ch];
    if (SILU) y = y / (one + hexp(-y));
    a.out[i] = y;
  }
}

template <int LOG2N>
static void launch_n(const NormArgs &a, bool silu, cudaStream_t st) {
  const dim3 block(PX, G);
  const unsigned grid = (unsigned)((a.pixels + PX - 1) / PX);
  if (silu)
    vae_rmsnorm_k<LOG2N, true><<<grid, block, 0, st>>>(a);
  else
    vae_rmsnorm_k<LOG2N, false><<<grid, block, 0, st>>>(a);
}

__global__ void vae_bias_residual_k(__nv_bfloat16 *__restrict__ out,
                                    const __nv_bfloat16 *__restrict__ y,
                                    const __nv_bfloat16 *__restrict__ bias,
                                    const __nv_bfloat16 *__restrict__ r,
                                    long long numel, long long hw, int c) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += (long long)gridDim.x * blockDim.x) {
    const int ch = (int)((i / hw) % c);
    const __nv_bfloat16 t = y[i] + bias[ch];
    out[i] = t + r[i];
  }
}

extern "C" {

// Returns 0 on launch, 1 if C is outside [G, 2048] (caller falls back).
int vae_rmsnorm_launch(void *out, const void *x, const void *gamma,
                       long long b, int c, long long hw, int silu,
                       void *stream) {
  if (c < G || c > 2048) return 1;
  NormArgs a;
  a.x = (const __nv_bfloat16 *)x;
  a.gamma = (const __nv_bfloat16 *)gamma;
  a.out = (__nv_bfloat16 *)out;
  a.pixels = b * hw;
  a.hw = hw;
  a.c = c;
  a.scale = (float)sqrt((double)c);
  int n = c < 1024 ? c : 1024, log2n = 0;
  while ((1 << log2n) < n) ++log2n;
  cudaStream_t st = (cudaStream_t)stream;
  const bool s = silu != 0;
  switch (log2n) {
  case 3: launch_n<3>(a, s, st); break;
  case 4: launch_n<4>(a, s, st); break;
  case 5: launch_n<5>(a, s, st); break;
  case 6: launch_n<6>(a, s, st); break;
  case 7: launch_n<7>(a, s, st); break;
  case 8: launch_n<8>(a, s, st); break;
  case 9: launch_n<9>(a, s, st); break;
  default: launch_n<10>(a, s, st); break;
  }
  return 0;
}

void vae_bias_residual_launch(void *out, const void *y, const void *bias,
                              const void *r, long long numel, long long hw,
                              int c, void *stream) {
  const int threads = 256;
  long long blocks = (numel + threads - 1) / threads;
  if (blocks > 65535LL * 32) blocks = 65535LL * 32;
  vae_bias_residual_k<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
      (__nv_bfloat16 *)out, (const __nv_bfloat16 *)y,
      (const __nv_bfloat16 *)bias, (const __nv_bfloat16 *)r, numel, hw, c);
}

} // extern "C"
