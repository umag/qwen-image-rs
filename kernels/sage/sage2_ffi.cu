// SageAttention2 (sm89) raw-pointer FFI: INT8 Q.K with PER-THREAD scales + K
// smoothing, FP8 (e4m3) P.V with per-channel V scales. The attention kernel is
// thu-ml's vendored qk_int_sv_f8_attn_kernel (vendor/qattn/qk_int_sv_f8_sm89.cuh,
// torch stripped). Everything upstream does in Triton / torch around it is ours:
//
//   * Sage2KSumPartialKernel  - RoPE(K) per-256-token-chunk channel sums (the
//                               K-smoothing mean, deterministic fixed-order reduce)
//   * Sage2RopeQuantKernel    - fused RoPE (+ K: subtract the sequence mean) +
//                               per-thread INT8 quant, scale layout exactly as the
//                               kernel's q_scale_idx / k_scale_idx (kPerThread)
//   * Sage2VAmaxPartialKernel - per-channel |V| max per 256-token chunk
//   * Sage2VQuantKernel       - per-channel FP8 quant, transposed to
//                               (B, H, D, Lpad64) with upstream's 16-token permute
//                               (TransposePadPermuteKernel), padded columns = 0
//   * sage2_attn_launch       - the fused attention, fp16 or fp32 PV accumulation
//
// head_dim is fixed at 128. Every launcher returns cudaGetLastError() so the
// Rust bridge (src/sage2.rs) can fail loudly.

#include <cassert> // must precede cuda_fp8/fp6/fp4 headers (CUDA 13 __assert_fail)
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "vendor/qattn/qk_int_sv_f8_sm89.cuh" // qk_int_sv_f8_attn_kernel + enums
#include "rope_pair.cuh"                       // shared no-FMA RoPE rotation
#include "../fusednorm/head_rmsnorm.cuh"       // per-head q/k RMSNorm (same bits as fusednorm)

namespace sage2 {

constexpr uint32_t D = 128;            // head_dim
constexpr uint32_t TPT = D / 8;        // threads per token (8 elements each)
constexpr uint32_t CHUNK = 256;        // tokens per partial-reduction chunk
constexpr uint32_t ROWS = 16;          // token rows per partial-reduction CTA
constexpr uint32_t V_TILE = 64;        // CTA_K: V tokens per tile / pad unit

// Rotate one 8-element bf16 pack (4 interleaved pairs) of token `tok`, rounded
// to bf16 exactly like the v1 fused rope+quant (and the unfused rope kernel).
__device__ __forceinline__ void rope_bf16(const __nv_bfloat16 x[8],
                                          const __nv_bfloat16 *cos,
                                          const __nv_bfloat16 *sin, uint32_t tok,
                                          uint32_t sseq_cs, uint32_t d0, float out[8]) {
  const __nv_bfloat16 *c = cos + (size_t)tok * sseq_cs + d0 / 2;
  const __nv_bfloat16 *s = sin + (size_t)tok * sseq_cs + d0 / 2;
#pragma unroll
  for (uint32_t p = 0; p < 4; p++) {
    __nv_bfloat16 y0, y1;
    rope_pair_bf16(__bfloat162float(x[2 * p]), __bfloat162float(x[2 * p + 1]),
                   __bfloat162float(c[p]), __bfloat162float(s[p]), y0, y1);
    out[2 * p] = __bfloat162float(y0);
    out[2 * p + 1] = __bfloat162float(y1);
  }
}

__device__ __forceinline__ void rope_pack(const __nv_bfloat16 *src,
                                          const __nv_bfloat16 *cos,
                                          const __nv_bfloat16 *sin,
                                          uint32_t tok, uint32_t sseq_cs,
                                          uint32_t d0, float out[8]) {
  __nv_bfloat16 x[8];
  *(float4 *)(&x[0]) = *(const float4 *)src;
  rope_bf16(x, cos, sin, tok, sseq_cs, d0, out);
}

// rope_pack with the per-head q/k RMSNorm in front (qwen-image-rs-qk-norm-fusion):
// when `w` (the f32 norm weight, 128 channels) is set, the raw projection pack is
// normalized across the token's 16-lane group exactly as the standalone
// fused_rmsnorm_scale kernel does it (same f32 order, bf16-rounded), then
// rotated — so the result is bit-identical to norm-then-rope and the normalized
// q/k never round-trip through memory. ALL 16 lanes of the token's aligned
// half-warp must call (the norm reduces over them); `w` is uniform per task.
__device__ __forceinline__ void rope_norm_pack(const __nv_bfloat16 *src, const float *w,
                                               float eps, const __nv_bfloat16 *cos,
                                               const __nv_bfloat16 *sin, uint32_t tok,
                                               uint32_t sseq_cs, uint32_t d0,
                                               float out[8]) {
  __nv_bfloat16 x[8];
  *(float4 *)(&x[0]) = *(const float4 *)src;
  if (w != nullptr) {
    float v[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) v[j] = __bfloat162float(x[j]);
    head_rmsnorm8(v, w + d0, eps, x);
  }
  rope_bf16(x, cos, sin, tok, sseq_cs, d0, out);
}

// grid (nchunk, H, B), block ROWS*TPT = 256. partial[(b*H+h)*nchunk + c][D] =
// sum over the chunk's tokens of RoPE(k) (bf16-rounded), summed per channel in
// a fixed order (deterministic run to run).
__global__ void Sage2KSumPartialKernel(
    const __nv_bfloat16 *__restrict__ in, const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin, float *__restrict__ partial,
    uint32_t n, uint32_t sbz, uint32_t sseq, uint32_t sh, uint32_t sseq_cs) {
  const uint32_t c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const uint32_t nchunk = gridDim.x, nh = gridDim.y;
  const uint32_t row = threadIdx.x / TPT, d0 = threadIdx.x % TPT * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint32_t end = min((c + 1) * CHUNK, n);
  for (uint32_t t = c * CHUNK + row; t < end; t += ROWS) {
    float y[8];
    rope_pack(in + (size_t)b * sbz + (size_t)h * sh + (size_t)t * sseq + d0, cos,
              sin, t, sseq_cs, d0, y);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) acc[j] += y[j];
  }
  __shared__ float sm[ROWS][D];
#pragma unroll
  for (uint32_t j = 0; j < 8; j++) sm[row][d0 + j] = acc[j];
  __syncthreads();
  if (threadIdx.x < D) {
    float s = 0.f;
    for (uint32_t r = 0; r < ROWS; r++) s += sm[r][threadIdx.x];
    partial[((size_t)(b * nh + h) * nchunk + c) * D + threadIdx.x] = s;
  }
}

// Per-thread scale group of a token inside its quant block, as the sm89
// kernel's mma fragments see it:
//  Query (BLK 32 = one WARP_Q): a thread owns rows lane/4 + {0,8,16,24}
//        -> group = tok % 8, 8 scales per block.
//  Key   (BLK 64 = CTA_K = WARP_K): a thread owns cols 2*(lane%4) + {0,1} + 8j
//        -> group = (tok % 8) / 2, 4 scales per block.
// (= upstream sageattention/triton/quant_per_thread.py.)
template <bool IS_K> struct Gran {
  static constexpr uint32_t BLK = IS_K ? 64 : 32;
  static constexpr uint32_t NG = IS_K ? 4 : 8;
  __device__ static uint32_t group(uint32_t r) { return IS_K ? (r % 8) / 2 : r % 8; }
};

// grid (nblk, H, B), block BLK*TPT. Reads PRE-rope bf16 (B,S,H,D) with caller
// strides, writes int8 HND-contiguous (b, h, n, D) + scale[(b*H+h)*nscale +
// blk*NG + g] = amax_g / 127. For Q, nblk is padded to ceil(n/128)*4 so every
// scale the attention kernel reads is written (empty blocks -> 1e-7/127).
template <bool IS_K>
__global__ void Sage2RopeQuantKernel(
    const __nv_bfloat16 *__restrict__ in, const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin, const float *__restrict__ partial,
    int8_t *__restrict__ out, float *__restrict__ scale, uint32_t n,
    uint32_t nchunk, uint32_t sbz, uint32_t sseq, uint32_t sh, uint32_t sseq_cs) {
  using G = Gran<IS_K>;
  const uint32_t bx = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const uint32_t nh = gridDim.y, nblk = gridDim.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t row = tid / TPT, d0 = tid % TPT * 8;
  const uint32_t tok = bx * G::BLK + row;

  __shared__ float s_mean[D];
  __shared__ float s_tok_amax[G::BLK];
  __shared__ float s_gamax[G::NG];

  if constexpr (IS_K) {
    if (tid < D) {
      float s = 0.f;
      const float *p = partial + (size_t)(b * nh + h) * nchunk * D + tid;
      for (uint32_t c = 0; c < nchunk; c++) s += p[(size_t)c * D];
      s_mean[tid] = s / (float)n;
    }
    __syncthreads();
  }

  float x[8];
  if (tok < n) {
    rope_pack(in + (size_t)b * sbz + (size_t)h * sh + (size_t)tok * sseq + d0, cos,
              sin, tok, sseq_cs, d0, x);
    if constexpr (IS_K) {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) x[j] -= s_mean[d0 + j];
    }
  } else {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) x[j] = 0.f;
  }

  float amax = 0.0000001f; // prevent dividing by zero (as QuantInt8Kernel)
#pragma unroll
  for (uint32_t j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(x[j]));
  // reduce over the token's 16 threads (aligned 16-lane groups of a warp)
#pragma unroll
  for (uint32_t off = TPT / 2; off > 0; off /= 2)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  if (tid % TPT == 0) s_tok_amax[row] = amax;
  __syncthreads();
  if (tid < G::NG) {
    float m = 0.0000001f;
    for (uint32_t r = 0; r < G::BLK; r++)
      if (G::group(r) == tid) m = fmaxf(m, s_tok_amax[r]);
    s_gamax[tid] = m;
    scale[(size_t)(b * nh + h) * nblk * G::NG + bx * G::NG + tid] = m / 127.0f;
  }
  __syncthreads();
  if (tok < n) {
    const float inv = 127.0f / s_gamax[G::group(row)];
    char4 o[2];
    o[0] = make_char4(float_to_int8_rn(x[0] * inv), float_to_int8_rn(x[1] * inv),
                      float_to_int8_rn(x[2] * inv), float_to_int8_rn(x[3] * inv));
    o[1] = make_char4(float_to_int8_rn(x[4] * inv), float_to_int8_rn(x[5] * inv),
                      float_to_int8_rn(x[6] * inv), float_to_int8_rn(x[7] * inv));
    *reinterpret_cast<float2 *>(out + ((size_t)(b * nh + h) * n + tok) * D + d0) =
        *reinterpret_cast<float2 *>(&o[0]);
  }
}

// grid (nchunk, H, B), block 256. partial[(b*H+h)*nchunk + c][D] = max |v| over
// the chunk's tokens (max is order-independent -> deterministic).
__global__ void Sage2VAmaxPartialKernel(const half *__restrict__ v,
                                        float *__restrict__ partial, uint32_t n,
                                        uint32_t sbz, uint32_t sseq, uint32_t sh) {
  const uint32_t c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const uint32_t nchunk = gridDim.x, nh = gridDim.y;
  const uint32_t row = threadIdx.x / TPT, d0 = threadIdx.x % TPT * 8;
  float m[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint32_t end = min((c + 1) * CHUNK, n);
  for (uint32_t t = c * CHUNK + row; t < end; t += ROWS) {
    half x[8];
    *(float4 *)(&x[0]) =
        *(const float4 *)(v + (size_t)b * sbz + (size_t)h * sh + (size_t)t * sseq + d0);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) m[j] = fmaxf(m[j], fabsf(__half2float(x[j])));
  }
  __shared__ float sm[ROWS][D];
#pragma unroll
  for (uint32_t j = 0; j < 8; j++) sm[row][d0 + j] = m[j];
  __syncthreads();
  if (threadIdx.x < D) {
    float s = 0.f;
    for (uint32_t r = 0; r < ROWS; r++) s = fmaxf(s, sm[r][threadIdx.x]);
    partial[((size_t)(b * nh + h) * nchunk + c) * D + threadIdx.x] = s;
  }
}

// Column of token m (0..15) inside its 16-token group after upstream's fp8-mma
// permute: 0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15 (TransposePadPermuteKernel).
__device__ __forceinline__ uint32_t perm16(uint32_t m) {
  return (m / 8) * 2 + ((m / 2) % 4) * 4 + (m % 2);
}

// grid (lpad/64, H, B), block 64*TPT = 1024. out (b, h, d, lpad) fp8 e4m3 =
// v * scale_max / amax_d (padded tokens -> 0), vscale[(b*H+h)*D + d] =
// amax_d / scale_max. amax is floored at 1e-7 (an all-zero channel would be
// 0 * inf = NaN in upstream's MeanScaleKernel).
__global__ void Sage2VQuantKernel(const half *__restrict__ v,
                                  const float *__restrict__ partial,
                                  uint8_t *__restrict__ out,
                                  float *__restrict__ vscale, uint32_t n,
                                  uint32_t nchunk, uint32_t lpad, float scale_max,
                                  uint32_t sbz, uint32_t sseq, uint32_t sh) {
  const uint32_t bx = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const uint32_t nh = gridDim.y;
  const uint32_t tid = threadIdx.x;
  __shared__ float s_inv[D];
  __shared__ __align__(16) uint8_t s_out[D][V_TILE];
  if (tid < D) {
    float a = 0.0000001f;
    const float *p = partial + (size_t)(b * nh + h) * nchunk * D + tid;
    for (uint32_t c = 0; c < nchunk; c++) a = fmaxf(a, p[(size_t)c * D]);
    s_inv[tid] = scale_max / a;
    if (bx == 0) vscale[(size_t)(b * nh + h) * D + tid] = a / scale_max;
  }
  __syncthreads();
  const uint32_t row = tid / TPT, d0 = tid % TPT * 8;
  const uint32_t tok = bx * V_TILE + row;
  float x[8];
  if (tok < n) {
    half hv[8];
    *(float4 *)(&hv[0]) =
        *(const float4 *)(v + (size_t)b * sbz + (size_t)h * sh + (size_t)tok * sseq + d0);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) x[j] = __half2float(hv[j]);
  } else {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) x[j] = 0.f;
  }
  const uint32_t col = (row / 16) * 16 + perm16(row % 16);
#pragma unroll
  for (uint32_t j = 0; j < 8; j++) {
    __nv_fp8_storage_t q =
        __nv_cvt_float_to_fp8(x[j] * s_inv[d0 + j], __NV_SATFINITE, __NV_E4M3);
    s_out[d0 + j][col] = (uint8_t)q;
  }
  __syncthreads();
  if (tid < D * (V_TILE / 16)) {
    const uint32_t d = tid / (V_TILE / 16), part = tid % (V_TILE / 16);
    *(uint4 *)(out + ((size_t)(b * nh + h) * D + d) * lpad + bx * V_TILE + part * 16) =
        *(const uint4 *)(&s_out[d][part * 16]);
  }
}

// ---------------------------------------------------------------------------
// Fused per-layer quant (qwen-image-rs-sage2-quant-fusion). One block-causal
// SA2 attention needs six quantized operands: Q(txt), Q(img), K(txt), K(full),
// V(txt), V(full) — ten per-op launches (two passes per K and V operand: the
// per-channel partial reduction, then the quant). The DiT issues three
// launches of ONE generic 512-thread kernel instead, each over a table of
// tasks (one task per CTA, decoded from a flat blockIdx.x):
//   1. V-amax partials (V full, V txt)
//   2. V quant (full, txt) + K-sum partials (full, txt)
//   3. K quant (full, txt) + Q quant (img, txt)
// The order keeps each tensor's second read right after its first (V was
// written last by to_v, then K, then Q), so the quant pass re-reads from L2;
// the small text-prefix tasks ride along instead of costing five launches of
// their own (each a few us of GPU time plus a launch gap). The math of every
// task is the per-op kernel's, op for op and in the same summation order, so
// payloads + scales are byte-identical (sage-test checks it against the
// untouched per-op kernels above).

enum S2Kind : uint32_t { kS2Q = 0, kS2K = 1, kS2V = 2, kS2KSum = 3, kS2VAmax = 4 };

// One quant task. Pointers are already offset to the view's first element (and
// cos/sin to its first row). `w` (Q / K / K-sum only, else null): the per-head
// RMSNorm weight (128 f32) to apply to the raw input first, with `eps`. `nblk` = the scale layout's quant blocks per head
// (Q: ceil(n/128)*4 32-token blocks; K: ceil(n/64)); `ncta` = CTAs per head
// (Q/K/V) or 0 (partials); `nblocks` = the task's share of the flat grid.
struct S2Task {
  const void *in;
  const void *cos;
  const void *sin;
  float *partial;
  void *out;
  float *scale;
  const float *w;
  uint32_t kind, n, nchunk, nblk, ncta, nblocks;
  uint32_t sbz, sseq, sh, sseq_cs, lpad;
  float scale_max;
  float eps;
};

constexpr uint32_t S2_MAX_TASKS = 6;
struct S2Tasks {
  S2Task t[S2_MAX_TASKS];
  uint32_t ntask, H, B;
};
// ABI with the #[repr(C)] mirrors in src/sage2.rs (which assert the same sizes).
static_assert(sizeof(S2Task) == 112, "S2Task layout");
static_assert(sizeof(S2Tasks) == 688, "S2Tasks layout");

// 512-thread CTAs: up to three resident per SM, so one CTA's barrier stalls
// overlap the others' loads (a 1024-thread CTA is alone on its SM).
constexpr uint32_t S2_THREADS = 512;
constexpr uint32_t S2_ROWS = S2_THREADS / TPT;                   // 32 token rows
constexpr uint32_t S2_PART_PER_CTA = S2_THREADS / (ROWS * TPT); // 2 chunks

// Map a flat block index to (task, block within the task); false if past the end.
__device__ __forceinline__ bool s2_pick(const S2Tasks &P, uint32_t blk, S2Task &T,
                                        uint32_t &local) {
  for (uint32_t i = 0; i < P.ntask; i++) {
    if (blk < P.t[i].nblocks) {
      T = P.t[i];
      local = blk;
      return true;
    }
    blk -= P.t[i].nblocks;
  }
  return false;
}

// Partials: each 256-thread quarter of the CTA runs Sage2KSumPartialKernel /
// Sage2VAmaxPartialKernel's body for one (chunk, h, b) — same loop, same
// order (per row t = c*CHUNK + row + 16i ascending, then rows 0..15).
__device__ __forceinline__ void s2_partial(const S2Task &T, uint32_t local, uint32_t nh,
                                           uint32_t nb) {
  const uint32_t q = threadIdx.x / (ROWS * TPT), qt = threadIdx.x % (ROWS * TPT);
  const uint32_t li = local * S2_PART_PER_CTA + q;
  const bool live = li < T.nchunk * nh * nb;
  const uint32_t c = li % T.nchunk, h = (li / T.nchunk) % nh, b = li / (T.nchunk * nh);
  const uint32_t row = qt / TPT, d0 = qt % TPT * 8;
  const bool is_k = T.kind == kS2KSum;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (live) {
    const uint32_t end = min((c + 1) * CHUNK, T.n);
    for (uint32_t t = c * CHUNK + row; t < end; t += ROWS) {
      if (is_k) {
        float y[8];
        // t is uniform across the row's 16 lanes (row = qt / TPT), as the
        // in-group norm reduction needs.
        rope_norm_pack((const __nv_bfloat16 *)T.in + (size_t)b * T.sbz + (size_t)h * T.sh +
                           (size_t)t * T.sseq + d0,
                       T.w, T.eps, (const __nv_bfloat16 *)T.cos,
                       (const __nv_bfloat16 *)T.sin, t, T.sseq_cs, d0, y);
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) acc[j] += y[j];
      } else {
        half x[8];
        *(float4 *)(&x[0]) = *(const float4 *)((const half *)T.in + (size_t)b * T.sbz +
                                               (size_t)h * T.sh + (size_t)t * T.sseq + d0);
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) acc[j] = fmaxf(acc[j], fabsf(__half2float(x[j])));
      }
    }
  }
  __shared__ float sm[S2_PART_PER_CTA][ROWS][D];
#pragma unroll
  for (uint32_t j = 0; j < 8; j++) sm[q][row][d0 + j] = acc[j];
  __syncthreads();
  if (live && qt < D) {
    float s = 0.f;
    if (is_k) {
      for (uint32_t r = 0; r < ROWS; r++) s += sm[q][r][qt];
    } else {
      for (uint32_t r = 0; r < ROWS; r++) s = fmaxf(s, sm[q][r][qt]);
    }
    T.partial[((size_t)(b * nh + h) * T.nchunk + c) * D + qt] = s;
  }
}

// Q or K: RoPE (+ K: minus the key mean) + per-thread INT8 quant of one quant
// block: Q = a 32-token warp block (8 groups, one token per thread), K = a
// 64-token block (4 groups, two tokens per thread: rows r and r+32, which
// share a group since 32 % 8 == 0). Same per-element math and scale values as
// Sage2RopeQuantKernel (the mean is summed in the same chunk order; max is
// order-independent).
template <bool is_k>
__device__ __forceinline__ void s2_quant_qk(const S2Task &T, uint32_t local, uint32_t nh) {
  const uint32_t cx = local % T.ncta, h = (local / T.ncta) % nh, b = local / (T.ncta * nh);
  const uint32_t tid = threadIdx.x;
  constexpr uint32_t NT = is_k ? 2 : 1;                 // tokens per thread
  constexpr uint32_t blk_tok = S2_ROWS * NT, ng = is_k ? 4 : 8;
  const uint32_t row0 = tid / TPT, d0 = tid % TPT * 8;
  const uint32_t g = is_k ? (row0 % 8) / 2 : row0 % 8; // same for row0 + 32
  const size_t hb = (size_t)(b * nh + h);

  __shared__ float s_mean[D];
  __shared__ float s_tok_amax[64];
  __shared__ float s_gamax[8];
  if constexpr (is_k) {
    if (tid < D) {
      float s = 0.f;
      const float *p = T.partial + hb * T.nchunk * D + tid;
#pragma unroll 8
      for (uint32_t c = 0; c < T.nchunk; c++) s += p[(size_t)c * D];
      s_mean[tid] = s / (float)T.n;
    }
    __syncthreads();
  }
  float x[NT][8];
#pragma unroll
  for (uint32_t i = 0; i < NT; i++) {
    const uint32_t row = row0 + i * S2_ROWS, tok = cx * blk_tok + row;
    if (tok < T.n) {
      // tok < n is uniform across the token's 16 lanes (in-group norm).
      rope_norm_pack((const __nv_bfloat16 *)T.in + (size_t)b * T.sbz + (size_t)h * T.sh +
                         (size_t)tok * T.sseq + d0,
                     T.w, T.eps, (const __nv_bfloat16 *)T.cos,
                     (const __nv_bfloat16 *)T.sin, tok, T.sseq_cs, d0, x[i]);
      if constexpr (is_k) {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) x[i][j] -= s_mean[d0 + j];
      }
    } else {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) x[i][j] = 0.f;
    }
    float amax = 0.0000001f;
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(x[i][j]));
#pragma unroll
    for (uint32_t off = TPT / 2; off > 0; off /= 2)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    if (tid % TPT == 0) s_tok_amax[row] = amax;
  }
  __syncthreads();
  if (tid < ng) {
    float m = 0.0000001f;
#pragma unroll
    for (uint32_t r = 0; r < blk_tok; r++) {
      const uint32_t rg = is_k ? (r % 8) / 2 : r % 8;
      if (rg == tid) m = fmaxf(m, s_tok_amax[r]);
    }
    s_gamax[tid] = m;
    T.scale[hb * T.nblk * ng + cx * ng + tid] = m / 127.0f;
  }
  __syncthreads();
  const float inv = 127.0f / s_gamax[g];
#pragma unroll
  for (uint32_t i = 0; i < NT; i++) {
    const uint32_t tok = cx * blk_tok + row0 + i * S2_ROWS;
    if (tok < T.n) {
      char4 o[2];
      o[0] = make_char4(float_to_int8_rn(x[i][0] * inv), float_to_int8_rn(x[i][1] * inv),
                        float_to_int8_rn(x[i][2] * inv), float_to_int8_rn(x[i][3] * inv));
      o[1] = make_char4(float_to_int8_rn(x[i][4] * inv), float_to_int8_rn(x[i][5] * inv),
                        float_to_int8_rn(x[i][6] * inv), float_to_int8_rn(x[i][7] * inv));
      *reinterpret_cast<float2 *>((int8_t *)T.out + (hb * T.n + tok) * D + d0) =
          *reinterpret_cast<float2 *>(&o[0]);
    }
  }
}

// V: Sage2VQuantKernel's body for one 64-token tile, two tokens per thread
// (rows r and r+32).
__device__ __forceinline__ void s2_quant_v(const S2Task &T, uint32_t local, uint32_t nh) {
  const uint32_t bx = local % T.ncta, h = (local / T.ncta) % nh, b = local / (T.ncta * nh);
  const uint32_t tid = threadIdx.x;
  const size_t hb = (size_t)(b * nh + h);
  __shared__ float s_inv[D];
  __shared__ __align__(16) uint8_t s_out[D][V_TILE];
  if (tid < D) {
    float a = 0.0000001f;
    const float *p = T.partial + hb * T.nchunk * D + tid;
#pragma unroll 8
    for (uint32_t c = 0; c < T.nchunk; c++) a = fmaxf(a, p[(size_t)c * D]);
    s_inv[tid] = T.scale_max / a;
    if (bx == 0) T.scale[hb * D + tid] = a / T.scale_max;
  }
  __syncthreads();
  const uint32_t row0 = tid / TPT, d0 = tid % TPT * 8;
#pragma unroll
  for (uint32_t i = 0; i < V_TILE / S2_ROWS; i++) {
    const uint32_t row = row0 + i * S2_ROWS, tok = bx * V_TILE + row;
    float x[8];
    if (tok < T.n) {
      half hv[8];
      *(float4 *)(&hv[0]) = *(const float4 *)((const half *)T.in + (size_t)b * T.sbz +
                                              (size_t)h * T.sh + (size_t)tok * T.sseq + d0);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) x[j] = __half2float(hv[j]);
    } else {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) x[j] = 0.f;
    }
    const uint32_t col = (row / 16) * 16 + perm16(row % 16);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) {
      __nv_fp8_storage_t q =
          __nv_cvt_float_to_fp8(x[j] * s_inv[d0 + j], __NV_SATFINITE, __NV_E4M3);
      s_out[d0 + j][col] = (uint8_t)q;
    }
  }
  __syncthreads();
  static_assert(D * (V_TILE / 16) == S2_THREADS, "one 16-B store per thread");
  const uint32_t d = tid / (V_TILE / 16), part = tid % (V_TILE / 16);
  *(uint4 *)((uint8_t *)T.out + (hb * D + d) * T.lpad + bx * V_TILE + part * 16) =
      *(const uint4 *)(&s_out[d][part * 16]);
}

// One CTA = one (task, block). Every path keeps its __syncthreads uniform
// across the CTA (the whole CTA runs one task).
__global__ void __launch_bounds__(S2_THREADS) Sage2QuantKernel(const S2Tasks P) {
  S2Task T;
  uint32_t local;
  if (!s2_pick(P, blockIdx.x, T, local)) return;
  switch (T.kind) {
  case kS2KSum:
  case kS2VAmax: s2_partial(T, local, P.H, P.B); break;
  case kS2Q: s2_quant_qk<false>(T, local, P.H); break;
  case kS2K: s2_quant_qk<true>(T, local, P.H); break;
  default: s2_quant_v(T, local, P.H); break;
  }
}

static uint32_t s2_total_blocks(const S2Tasks &P) {
  uint32_t n = 0;
  for (uint32_t i = 0; i < P.ntask; i++) n += P.t[i].nblocks;
  return n;
}

template <MaskMode MM, bool F16_ACCUM>
static cudaError_t launch_attn(const int8_t *q, const int8_t *k, const int8_t *v,
                               __nv_bfloat16 *o, const float *qs, const float *ks,
                               const float *vs, int B, int Hq, int Hk, int qo,
                               int kv, int lpad, uint32_t sbz_o, uint32_t sseq_o,
                               uint32_t sh_o, float sm_scale, cudaStream_t st) {
  constexpr int CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
  const uint32_t sbz_q = (uint32_t)Hq * qo * D, sh_q = (uint32_t)qo * D, sseq_q = D;
  const uint32_t sbz_k = (uint32_t)Hk * kv * D, sh_k = (uint32_t)kv * D, sseq_k = D;
  const uint32_t sbz_v = (uint32_t)Hk * D * lpad, sh_v = (uint32_t)D * lpad,
                 sd_v = (uint32_t)lpad;
  const size_t smem = std::max((size_t)CTA_Q * D + CTA_K * D + CTA_K * D,
                               (size_t)CTA_Q * D * sizeof(half));
  auto kf = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, D, DataType::kInt8,
                                     QuantGranularity::kPerThread,
                                     QuantGranularity::kPerThread, float, true,
                                     __nv_bfloat16, ComputeUnit::kCudaCore, MM, false,
                                     true, false, F16_ACCUM>;
  cudaError_t e = cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return e;
  dim3 grid((qo + CTA_Q - 1) / CTA_Q, Hq, B);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  kf<<<grid, block, smem, st>>>((int8_t *)q, (int8_t *)k, (int8_t *)v, o, nullptr,
                                (float *)qs, (float *)ks, (float *)vs, nullptr, qo, kv,
                                Hq / Hk, sbz_q, sseq_q, sh_q, sbz_k, sseq_k, sh_k,
                                sbz_v, sh_v, sd_v, sbz_o, sseq_o, sh_o, sm_scale);
  return cudaGetLastError();
}

} // namespace sage2

using namespace sage2;

// Fused RoPE + per-thread INT8 quant. is_k=0: Query (scales (B,H,ceil(n/128)*32));
// is_k=1: Key with smoothing (scales (B,H,ceil(n/64)*4)); `partial` is a scratch
// of B*H*ceil(n/256)*128 f32 (Key only; may be null for Query).
extern "C" int sage2_rope_quant_launch(const void *in, const void *cos,
                                       const void *sin, void *out_i8, float *scale,
                                       float *partial, int B, int H, int N,
                                       unsigned sbz, unsigned sseq, unsigned sh,
                                       unsigned sseq_cs, int is_k, void *stream) {
  cudaStream_t st = (cudaStream_t)stream;
  const __nv_bfloat16 *x = (const __nv_bfloat16 *)in;
  const __nv_bfloat16 *c = (const __nv_bfloat16 *)cos;
  const __nv_bfloat16 *s = (const __nv_bfloat16 *)sin;
  const uint32_t n = (uint32_t)N;
  const uint32_t nchunk = (n + CHUNK - 1) / CHUNK;
  if (is_k) {
    Sage2KSumPartialKernel<<<dim3(nchunk, H, B), ROWS * TPT, 0, st>>>(
        x, c, s, partial, n, sbz, sseq, sh, sseq_cs);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    const uint32_t nblk = (n + 63) / 64;
    Sage2RopeQuantKernel<true><<<dim3(nblk, H, B), 64 * TPT, 0, st>>>(
        x, c, s, partial, (int8_t *)out_i8, scale, n, nchunk, sbz, sseq, sh, sseq_cs);
  } else {
    const uint32_t nblk = (n + 127) / 128 * 4; // padded to the kernel's warp blocks
    Sage2RopeQuantKernel<false><<<dim3(nblk, H, B), 32 * TPT, 0, st>>>(
        x, c, s, nullptr, (int8_t *)out_i8, scale, n, 0, sbz, sseq, sh, sseq_cs);
  }
  return (int)cudaGetLastError();
}

// Per-channel FP8 V quant of an f16 (B,S,H,D) view -> fp8 (B,H,D,lpad) +
// vscale (B,H,D). `partial`: scratch of B*H*ceil(n/256)*128 f32.
extern "C" int sage2_quant_v_launch(const void *v, void *out_fp8, float *vscale,
                                    float *partial, int B, int H, int N, int lpad,
                                    float scale_max, unsigned sbz, unsigned sseq,
                                    unsigned sh, void *stream) {
  cudaStream_t st = (cudaStream_t)stream;
  const uint32_t n = (uint32_t)N;
  const uint32_t nchunk = (n + CHUNK - 1) / CHUNK;
  Sage2VAmaxPartialKernel<<<dim3(nchunk, H, B), ROWS * TPT, 0, st>>>(
      (const half *)v, partial, n, sbz, sseq, sh);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return (int)e;
  Sage2VQuantKernel<<<dim3((uint32_t)lpad / V_TILE, H, B), V_TILE * TPT, 0, st>>>(
      (const half *)v, partial, (uint8_t *)out_fp8, vscale, n, nchunk, (uint32_t)lpad,
      scale_max, sbz, sseq, sh);
  return (int)cudaGetLastError();
}

// SA2 attention: int8 q (B,Hq,qo,D) / k (B,Hk,kv,D) HND + per-thread scales,
// fp8 v (B,Hk,D,lpad) + vscale (B,Hk,D); o bf16 with caller (B,S,H,D) strides.
// f16_accum selects the fp32+fp16 PV accumulation (V quantized with scale_max
// 2.25) vs fp32 (scale_max 448) — the caller guarantees they match.
extern "C" int sage2_attn_launch(const void *q_i8, const void *k_i8,
                                 const void *v_fp8, void *o_bf16,
                                 const float *q_scale, const float *k_scale,
                                 const float *v_scale, int B, int Hq, int Hk,
                                 int qo, int kv, int lpad, unsigned sbz_o,
                                 unsigned sseq_o, unsigned sh_o, float sm_scale,
                                 int is_causal, int f16_accum, void *stream) {
  auto q = (const int8_t *)q_i8;
  auto k = (const int8_t *)k_i8;
  auto v = (const int8_t *)v_fp8;
  auto o = (__nv_bfloat16 *)o_bf16;
  cudaStream_t st = (cudaStream_t)stream;
  cudaError_t e;
  if (is_causal) {
    e = f16_accum ? launch_attn<MaskMode::kCausal, true>(q, k, v, o, q_scale, k_scale,
                                                         v_scale, B, Hq, Hk, qo, kv, lpad,
                                                         sbz_o, sseq_o, sh_o, sm_scale, st)
                  : launch_attn<MaskMode::kCausal, false>(q, k, v, o, q_scale, k_scale,
                                                          v_scale, B, Hq, Hk, qo, kv,
                                                          lpad, sbz_o, sseq_o, sh_o,
                                                          sm_scale, st);
  } else {
    e = f16_accum ? launch_attn<MaskMode::kNone, true>(q, k, v, o, q_scale, k_scale,
                                                       v_scale, B, Hq, Hk, qo, kv, lpad,
                                                       sbz_o, sseq_o, sh_o, sm_scale, st)
                  : launch_attn<MaskMode::kNone, false>(q, k, v, o, q_scale, k_scale,
                                                        v_scale, B, Hq, Hk, qo, kv, lpad,
                                                        sbz_o, sseq_o, sh_o, sm_scale, st);
  }
  return (int)e;
}

// Fused per-layer quant: `ntables` launches of Sage2QuantKernel, in order, one
// per task table (sage2.rs quant_layer builds them). Validates the tables'
// grid bookkeeping against the kernel's fixed CTA shapes before launching.
extern "C" int sage2_quant_layer_launch(const S2Tasks *tables, int ntables,
                                        void *stream) {
  cudaStream_t st = (cudaStream_t)stream;
  for (int i = 0; i < ntables; i++) {
    const S2Tasks &P = tables[i];
    if (P.ntask == 0 || P.ntask > S2_MAX_TASKS) return (int)cudaErrorInvalidValue;
    for (uint32_t j = 0; j < P.ntask; j++) {
      const S2Task &T = P.t[j];
      const uint32_t heads = P.H * P.B;
      uint32_t want;
      switch (T.kind) {
      case kS2VAmax:
        if (T.w != nullptr) return (int)cudaErrorInvalidValue;
        [[fallthrough]];
      case kS2KSum:
        want = (T.nchunk * heads + S2_PART_PER_CTA - 1) / S2_PART_PER_CTA;
        break;
      case kS2Q:
        if (T.ncta != T.nblk) return (int)cudaErrorInvalidValue; // one warp block per CTA
        want = T.ncta * heads;
        break;
      case kS2K: want = T.ncta * heads; break;
      case kS2V:
        if (T.w != nullptr) return (int)cudaErrorInvalidValue; // no norm on V
        want = T.ncta * heads;
        break;
      default: return (int)cudaErrorInvalidValue;
      }
      if (T.nblocks != want) return (int)cudaErrorInvalidValue;
    }
    const uint32_t n = s2_total_blocks(P);
    if (n == 0) continue;
    Sage2QuantKernel<<<n, S2_THREADS, 0, st>>>(P);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
  }
  return (int)cudaSuccess;
}
