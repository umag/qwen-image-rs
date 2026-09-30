// Shared interleaved-pair RoPE rotation, used by BOTH the standalone BSHD rope
// kernel (rope_bshd.cu, retained as the equivalence oracle) and the fused
// rope+INT8-quant kernel (sage_ffi.cu). Explicit round-to-nearest intrinsics
// forbid FMA contraction, so the two call sites compute bit-identical results
// regardless of how nvcc schedules the surrounding code. The rotated value is
// rounded to bf16 (what the unfused path stores between rope and quant).
#pragma once
#include <cuda_bf16.h>

// (x0, x1) rotated by (c, sn) -> bf16-rounded (y0, y1).
static __device__ __forceinline__ void rope_pair_bf16(float x0, float x1,
                                                      float c, float sn,
                                                      __nv_bfloat16 &y0,
                                                      __nv_bfloat16 &y1) {
  y0 = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, sn)));
  y1 = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x0, sn), __fmul_rn(x1, c)));
}
