// Per-head RMSNorm (head_dim 128) on a 16-lane group: each of the 16 lanes of
// an aligned half-warp holds 8 consecutive channels (lane l -> d0 = 8*(l%16)).
// Shared by the standalone N=128 kernel (fused_norm.cu) and the SA2 Q/K quant
// prologue (sage2_ffi.cu) so both produce the SAME bits.
//
// Bit-identity with fused_rmsnorm_scale_kernel (CTA-per-row, 256 threads, one
// element per thread for N=128, shared-memory tree s = 128, 64, ..., 1): that
// tree adds element e to e+64 (index bit 6) first, then bit 5, ..., bit 0 last
// (the s = 128 level adds exact zeros). Here element bits 6..3 are the lane
// bits 3..0 -> xor-shuffles 8, 4, 2, 1 (in that order), then the in-lane bits
// 2, 1, 0. Float addition is commutative, so every lane ends with the same sum
// as the tree's red[0]. Squares use __fmul_rn so nvcc cannot contract them
// into an FMA with the first add (the old kernel's `0.f + v*v` rounds v*v).
#pragma once
#include <cuda_bf16.h>

// Mask of the caller's aligned 16-lane group (all 16 lanes must call).
__device__ __forceinline__ unsigned head_rms_mask() {
  return 0xffffu << (threadIdx.x & 16u);
}

// Sum of squares of the 128 channels of the group's row (every lane gets it).
__device__ __forceinline__ float head_rms_sumsq(const float v[8]) {
  const unsigned mask = head_rms_mask();
  float s[8];
#pragma unroll
  for (int j = 0; j < 8; j++) s[j] = __fmul_rn(v[j], v[j]);
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
    for (int j = 0; j < 8; j++) s[j] = s[j] + __shfl_xor_sync(mask, s[j], off);
  }
  const float a0 = s[0] + s[4], a1 = s[1] + s[5], a2 = s[2] + s[6], a3 = s[3] + s[7];
  return (a0 + a2) + (a1 + a3);
}

// out[j] = bf16(v[j] * rsqrt(mean(v^2) + eps) * w[d0 + j]) for the lane's 8
// channels; `w` points at the f32 weight of channel d0. Same expression order
// as fused_rmsnorm_scale_kernel: rsqrtf(sum / n + eps), (v * rrms) * w.
__device__ __forceinline__ void head_rmsnorm8(const float v[8], const float *w,
                                              float eps, __nv_bfloat16 out[8]) {
  const float rrms = rsqrtf(head_rms_sumsq(v) / 128.0f + eps);
#pragma unroll
  for (int j = 0; j < 8; j++) out[j] = __float2bfloat16(v[j] * rrms * __ldg(w + j));
}
