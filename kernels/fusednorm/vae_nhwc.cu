// VAE decoder fused ops on channels-last (NHWC) bf16 activations (feature
// `fusednorm`; bridges in src/vae_fused.rs). Every op reproduces the candle
// NCHW chain in src/model/vae.rs bit for bit — only the memory order differs.
//
// 1. vae_nhwc_rmsnorm_k: [bf16(x + bias[c])] -> QwenImage21RMS_norm over C ->
//    * gamma -> [SiLU]. The channel sum must associate exactly like candle's
//    fast_sum (see vae_norm.cu): a balanced adjacent-pair tree over the
//    leaves k = 0..N-1 (N = next_pow2(min(1024, C))), leaf k = channel
//    t = brev(k) (+ channel t + 1024 when C > 1024). Split the LOG2N-bit index
//    k = [g | j] into a chunk id g (top log2(G) bits, G = 32*VEC chunks) and a
//    position j: t = brev(j) * G + brev(g). So chunk g holds exactly the
//    channels with t mod G = rho = brev(g), and each chunk is a complete
//    subtree. One warp per pixel; lane L loads channels VEC*L .. VEC*L+VEC-1
//    (+ q*G): residues rho = VEC*L + e. Each lane sums its VEC chunks with a
//    compile-time pairwise recursion over j, then the G chunk sums combine as
//    the tree over g: tree level b pairs chunks whose g differs in bit b, i.e.
//    whose rho differs in bit log2(G)-1-b — lane bits 4..0 first (xor-16..1
//    shuffles), then the in-lane e bits high to low. fadd is commutative bit
//    for bit, so a butterfly gives every lane the exact same subtotal.
//    Explicit _rn intrinsics: no FMA contraction (candle-kernels: no fast-math).
//
// 2. vae_nhwc_epilogue_k: out = bf16(y + bias[c]), then + residual:
//      0 none | 1 r[i] | 2 bf16(r[i] + rbias[c]) | 3 DupUp3D(xc) gathered.
//    Same two bf16 roundings as candle's broadcast_add followed by add.
//
// 3. vae_nhwc_upsample2x_k: nearest 2x (== nearest-exact for 2x).

#include <cassert> // precede any cuda fp headers (CUDA 13 __assert_fail)
#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8; // pixels per CTA (one warp each)

__host__ __device__ constexpr int brev_bits(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; ++i) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

__host__ __device__ constexpr int ilog2(int v) {
  int r = 0;
  while ((1 << r) < v) ++r;
  return r;
}

template <int VEC> struct Vec;
template <> struct Vec<8> { using T = uint4; };
template <> struct Vec<4> { using T = uint2; };
template <> struct Vec<2> { using T = unsigned int; };
template <> struct Vec<1> { using T = unsigned short; };

template <int VEC>
__device__ __forceinline__ void load_vec(__nv_bfloat16 *dst, const __nv_bfloat16 *src) {
  using T = typename Vec<VEC>::T;
  *reinterpret_cast<T *>(dst) = *reinterpret_cast<const T *>(src);
}

template <int VEC>
__device__ __forceinline__ void store_vec(__nv_bfloat16 *dst, const __nv_bfloat16 *src) {
  using T = typename Vec<VEC>::T;
  *reinterpret_cast<T *>(dst) = *reinterpret_cast<const T *>(src);
}

__device__ __forceinline__ float sqf(__nv_bfloat16 v) {
  const float f = __bfloat162float(v);
  return __fmul_rn(f, f);
}

// Per-lane state of the norm: QS slots x VEC channels (bf16, bias added).
template <int LOG2N, int VEC> struct NormShape {
  static constexpr int N = 1 << LOG2N;
  static constexpr int G = 32 * VEC;            // chunks
  static constexpr int LOG2G = ilog2(G);
  static constexpr int M = N / G;               // leaves per chunk
  static constexpr int LOG2M = LOG2N - LOG2G;
  static constexpr int EXTRA = (LOG2N == 10) ? 1024 / G : 0; // slot offset of t+1024
  static constexpr int QS = (LOG2N == 10) ? 2048 / G : M;    // slots per lane
};

// Leaf j of chunk e: slot q = brev(j) (+ the t+1024 slot when C > 1024).
template <int LOG2N, int VEC, int J>
__device__ __forceinline__ float leaf(const __nv_bfloat16 (&v)[NormShape<LOG2N, VEC>::QS][VEC],
                                      const bool (&live)[NormShape<LOG2N, VEC>::QS], int e) {
  using S = NormShape<LOG2N, VEC>;
  constexpr int q = brev_bits(J, S::LOG2M);
  float s = 0.f;
  if (live[q]) s = __fadd_rn(s, sqf(v[q][e]));
  if constexpr (S::EXTRA > 0) {
    if (live[q + S::EXTRA]) s = __fadd_rn(s, sqf(v[q + S::EXTRA][e]));
  }
  return s;
}

template <int LOG2N, int VEC, int J0, int CNT>
__device__ __forceinline__ float subtree(const __nv_bfloat16 (&v)[NormShape<LOG2N, VEC>::QS][VEC],
                                         const bool (&live)[NormShape<LOG2N, VEC>::QS], int e) {
  if constexpr (CNT == 1) {
    return leaf<LOG2N, VEC, J0>(v, live, e);
  } else {
    const float l = subtree<LOG2N, VEC, J0, CNT / 2>(v, live, e);
    const float r = subtree<LOG2N, VEC, J0 + CNT / 2, CNT / 2>(v, live, e);
    return __fadd_rn(l, r);
  }
}

template <int LOG2N, int VEC, bool SILU, bool BIAS>
__global__ void __launch_bounds__(32 * WARPS)
    vae_nhwc_rmsnorm_k(__nv_bfloat16 *__restrict__ out, const __nv_bfloat16 *__restrict__ x,
                       const __nv_bfloat16 *__restrict__ gamma,
                       const __nv_bfloat16 *__restrict__ bias, long long pixels, int c,
                       float scale) {
  using S = NormShape<LOG2N, VEC>;
  const int lane = threadIdx.x & 31;
  const long long p = (long long)blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (p >= pixels) return; // whole warp exits together
  const __nv_bfloat16 *xp = x + p * c;
  __nv_bfloat16 v[S::QS][VEC];
  bool live[S::QS];
#pragma unroll
  for (int q = 0; q < S::QS; ++q) {
    const int base = q * S::G + VEC * lane; // c % VEC == 0: all VEC in or out
    live[q] = base < c;
    if (live[q]) {
      load_vec<VEC>(v[q], xp + base);
      if constexpr (BIAS) {
        __nv_bfloat16 bv[VEC];
        load_vec<VEC>(bv, bias + base);
#pragma unroll
        for (int e = 0; e < VEC; ++e) v[q][e] = v[q][e] + bv[e];
      }
    }
  }
  float part[VEC];
#pragma unroll
  for (int e = 0; e < VEC; ++e) part[e] = subtree<LOG2N, VEC, 0, S::M>(v, live, e);
  // Tree over g: lane bits 4..0 (rho's top bits) first ...
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
    for (int e = 0; e < VEC; ++e) part[e] = __fadd_rn(part[e], __shfl_xor_sync(0xffffffffu, part[e], m));
  // ... then the in-lane e bits, high to low.
#pragma unroll
  for (int w = VEC / 2; w >= 1; w >>= 1)
#pragma unroll
    for (int e = 0; e < VEC; ++e)
      if (e < w) part[e] = __fadd_rn(part[e], part[e + w]);
  const float n = fmaxf(__fsqrt_rn(part[0]), 1e-12f);
  const __nv_bfloat16 one = static_cast<__nv_bfloat16>(1);
  __nv_bfloat16 *op = out + p * c;
#pragma unroll
  for (int q = 0; q < S::QS; ++q) {
    if (!live[q]) continue;
    const int base = q * S::G + VEC * lane;
    __nv_bfloat16 gv[VEC], o[VEC];
    load_vec<VEC>(gv, gamma + base);
#pragma unroll
    for (int e = 0; e < VEC; ++e) {
      const float xf = __bfloat162float(v[q][e]);
      const float t = __fadd_rn(__fmul_rn(__fdiv_rn(xf, n), scale), 0.f);
      __nv_bfloat16 y = __float2bfloat16_rn(t) * gv[e];
      if (SILU) y = y / (one + hexp(-y));
      o[e] = y;
    }
    store_vec<VEC>(op + base, o);
  }
}

template <int LOG2N, int VEC>
void launch_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x, const __nv_bfloat16 *g,
                 const __nv_bfloat16 *bias, long long pixels, int c, bool silu, cudaStream_t st) {
  const unsigned grid = (unsigned)((pixels + WARPS - 1) / WARPS);
  const float scale = (float)sqrt((double)c);
  const dim3 block(32 * WARPS);
  if (bias) {
    if (silu)
      vae_nhwc_rmsnorm_k<LOG2N, VEC, true, true><<<grid, block, 0, st>>>(out, x, g, bias, pixels, c, scale);
    else
      vae_nhwc_rmsnorm_k<LOG2N, VEC, false, true><<<grid, block, 0, st>>>(out, x, g, bias, pixels, c, scale);
  } else {
    if (silu)
      vae_nhwc_rmsnorm_k<LOG2N, VEC, true, false><<<grid, block, 0, st>>>(out, x, g, bias, pixels, c, scale);
    else
      vae_nhwc_rmsnorm_k<LOG2N, VEC, false, false><<<grid, block, 0, st>>>(out, x, g, bias, pixels, c, scale);
  }
}

template <int LOG2N>
int launch_norm_n(__nv_bfloat16 *out, const __nv_bfloat16 *x, const __nv_bfloat16 *g,
                  const __nv_bfloat16 *bias, long long pixels, int c, bool silu, int vec,
                  cudaStream_t st) {
  constexpr int N = 1 << LOG2N;
  switch (vec) {
  case 8:
    if constexpr (N >= 256) { launch_norm<LOG2N, 8>(out, x, g, bias, pixels, c, silu, st); return 0; }
    return 1;
  case 4:
    if constexpr (N >= 128) { launch_norm<LOG2N, 4>(out, x, g, bias, pixels, c, silu, st); return 0; }
    return 1;
  case 2:
    if constexpr (N >= 64) { launch_norm<LOG2N, 2>(out, x, g, bias, pixels, c, silu, st); return 0; }
    return 1;
  default:
    launch_norm<LOG2N, 1>(out, x, g, bias, pixels, c, silu, st);
    return 0;
  }
}

// ---------------------------------------------------------------- epilogue

struct EpiArgs {
  __nv_bfloat16 *out;
  const __nv_bfloat16 *y;
  const __nv_bfloat16 *bias;
  const __nv_bfloat16 *r;     // residual (mode 1, 2) or the DupUp source (mode 3)
  const __nv_bfloat16 *rbias; // mode 2
  long long n_vec;            // numel / VEC
  int c;
  int mode;
  // DupUp3D (mode 3): out (B, 2h, 2w, c) from r = xc (B, h, w, in_c).
  int h, w, in_c, repeats, ft;
};

// Source channel of DupUp3D (first_chunk, temporal block ft-1) for output
// channel o at sub-pixel (i, j): see dup_up() in src/model/vae.rs.
__device__ __forceinline__ int dupup_src(int o, int i, int j, int ft, int repeats) {
  const int k = ((o * ft + (ft - 1)) * 2 + i) * 2 + j;
  return k / repeats;
}

template <int VEC> __global__ void vae_nhwc_epilogue_k(EpiArgs a) {
  for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x; v < a.n_vec;
       v += (long long)gridDim.x * blockDim.x) {
    const long long i0 = v * VEC;
    const int c0 = (int)(i0 % a.c);
    __nv_bfloat16 yv[VEC], bv[VEC], o[VEC];
    load_vec<VEC>(yv, a.y + i0);
    load_vec<VEC>(bv, a.bias + c0);
#pragma unroll
    for (int e = 0; e < VEC; ++e) o[e] = yv[e] + bv[e];
    if (a.mode == 1 || a.mode == 2) {
      __nv_bfloat16 rv[VEC];
      load_vec<VEC>(rv, a.r + i0);
      if (a.mode == 2) {
        __nv_bfloat16 rb[VEC];
        load_vec<VEC>(rb, a.rbias + c0);
#pragma unroll
        for (int e = 0; e < VEC; ++e) rv[e] = rv[e] + rb[e];
      }
#pragma unroll
      for (int e = 0; e < VEC; ++e) o[e] = o[e] + rv[e];
    } else if (a.mode == 3) {
      const long long pix = i0 / a.c; // over (B, 2h, 2w)
      const int w2 = 2 * a.w, h2 = 2 * a.h;
      const int X = (int)(pix % w2);
      const long long t = pix / w2;
      const int Y = (int)(t % h2);
      const long long b = t / h2;
      const long long src = ((b * a.h + Y / 2) * a.w + X / 2) * a.in_c;
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        o[e] = o[e] + a.r[src + dupup_src(c0 + e, Y & 1, X & 1, a.ft, a.repeats)];
    }
    store_vec<VEC>(a.out + i0, o);
  }
}

// ---------------------------------------------------------------- upsample

template <int VEC>
__global__ void vae_nhwc_upsample2x_k(__nv_bfloat16 *__restrict__ out,
                                      const __nv_bfloat16 *__restrict__ x, long long n_vec,
                                      int h, int w, int c) {
  const int cv = c / VEC;
  for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x; v < n_vec;
       v += (long long)gridDim.x * blockDim.x) {
    const int ci = (int)(v % cv);
    const long long pix = v / cv; // over (B, 2h, 2w)
    const int X = (int)(pix % (2 * w));
    const long long t = pix / (2 * w);
    const int Y = (int)(t % (2 * h));
    const long long b = t / (2 * h);
    const long long src = ((b * h + Y / 2) * w + X / 2) * c + (long long)ci * VEC;
    __nv_bfloat16 tmp[VEC];
    load_vec<VEC>(tmp, x + src);
    store_vec<VEC>(out + v * VEC, tmp);
  }
}

unsigned grid_for(long long n, int threads) {
  long long blocks = (n + threads - 1) / threads;
  if (blocks > 65535LL * 32) blocks = 65535LL * 32;
  return (unsigned)(blocks < 1 ? 1 : blocks);
}

int pick_vec(int c, uintptr_t align) {
  for (int v = 8; v > 1; v >>= 1)
    if (c % v == 0 && (align % (2 * v)) == 0) return v;
  return 1;
}

} // namespace

extern "C" {

// Returns 0 on launch; nonzero if the shape is unsupported (C outside
// [17, 2048] — the warp needs N >= 32 leaves); the caller falls back.
int vae_nhwc_rmsnorm_launch(void *out, const void *x, const void *gamma, const void *bias,
                            long long pixels, int c, int silu, void *stream) {
  if (c < 17 || c > 2048) return 1;
  int n = c < 1024 ? c : 1024, log2n = 0;
  while ((1 << log2n) < n) ++log2n;
  uintptr_t al = (uintptr_t)x | (uintptr_t)out | (uintptr_t)gamma | (bias ? (uintptr_t)bias : 0);
  int vec = pick_vec(c, al);
  while (vec > 1 && 32 * vec > (1 << log2n)) vec >>= 1;
  auto *o = (__nv_bfloat16 *)out;
  auto *xx = (const __nv_bfloat16 *)x;
  auto *g = (const __nv_bfloat16 *)gamma;
  auto *bb = (const __nv_bfloat16 *)bias;
  cudaStream_t st = (cudaStream_t)stream;
  const bool s = silu != 0;
  switch (log2n) {
  case 5: return launch_norm_n<5>(o, xx, g, bb, pixels, c, s, vec, st);
  case 6: return launch_norm_n<6>(o, xx, g, bb, pixels, c, s, vec, st);
  case 7: return launch_norm_n<7>(o, xx, g, bb, pixels, c, s, vec, st);
  case 8: return launch_norm_n<8>(o, xx, g, bb, pixels, c, s, vec, st);
  case 9: return launch_norm_n<9>(o, xx, g, bb, pixels, c, s, vec, st);
  case 10: return launch_norm_n<10>(o, xx, g, bb, pixels, c, s, vec, st);
  default: return 1;
  }
}

// mode: 0 none, 1 dense residual, 2 dense residual + rbias, 3 DupUp3D gather.
// numel = output elements; for mode 3, (h, w) are the SOURCE spatial dims.
int vae_nhwc_epilogue_launch(void *out, const void *y, const void *bias, const void *r,
                             const void *rbias, long long numel, int c, int mode, int h, int w,
                             int in_c, int repeats, int ft, void *stream) {
  if (c < 1 || numel % c != 0 || mode < 0 || mode > 3) return 1;
  if (mode == 3 && (h < 1 || w < 1 || in_c < 1 || repeats < 1 || ft < 1)) return 1;
  uintptr_t al = (uintptr_t)out | (uintptr_t)y | (uintptr_t)bias;
  if (mode == 1 || mode == 2) al |= (uintptr_t)r;
  if (mode == 2) al |= (uintptr_t)rbias;
  const int vec = pick_vec(c, al);
  EpiArgs a{(__nv_bfloat16 *)out, (const __nv_bfloat16 *)y, (const __nv_bfloat16 *)bias,
            (const __nv_bfloat16 *)r, (const __nv_bfloat16 *)rbias, numel / vec, c, mode,
            h, w, in_c, repeats, ft};
  const int threads = 256;
  const unsigned grid = grid_for(a.n_vec, threads);
  cudaStream_t st = (cudaStream_t)stream;
  switch (vec) {
  case 8: vae_nhwc_epilogue_k<8><<<grid, threads, 0, st>>>(a); break;
  case 4: vae_nhwc_epilogue_k<4><<<grid, threads, 0, st>>>(a); break;
  case 2: vae_nhwc_epilogue_k<2><<<grid, threads, 0, st>>>(a); break;
  default: vae_nhwc_epilogue_k<1><<<grid, threads, 0, st>>>(a); break;
  }
  return 0;
}

// x (B, h, w, c) -> out (B, 2h, 2w, c).
int vae_nhwc_upsample2x_launch(void *out, const void *x, long long b, int h, int w, int c,
                               void *stream) {
  if (b < 1 || h < 1 || w < 1 || c < 1) return 1;
  const int vec = pick_vec(c, (uintptr_t)out | (uintptr_t)x);
  const long long n_vec = b * 4LL * h * w * c / vec;
  const int threads = 256;
  const unsigned grid = grid_for(n_vec, threads);
  cudaStream_t st = (cudaStream_t)stream;
  auto *o = (__nv_bfloat16 *)out;
  auto *xx = (const __nv_bfloat16 *)x;
  switch (vec) {
  case 8: vae_nhwc_upsample2x_k<8><<<grid, threads, 0, st>>>(o, xx, n_vec, h, w, c); break;
  case 4: vae_nhwc_upsample2x_k<4><<<grid, threads, 0, st>>>(o, xx, n_vec, h, w, c); break;
  case 2: vae_nhwc_upsample2x_k<2><<<grid, threads, 0, st>>>(o, xx, n_vec, h, w, c); break;
  default: vae_nhwc_upsample2x_k<1><<<grid, threads, 0, st>>>(o, xx, n_vec, h, w, c); break;
  }
  return 0;
}

} // extern "C"
