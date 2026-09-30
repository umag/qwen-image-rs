// BSHD interleaved-RoPE launcher. Applies the same interleaved-pair rotation as
// candle's `rope_i` but on a (B, S, H, D) tensor directly, so the sage attention
// path never has to transpose q/k into (B, H, S, D). cos/sin are (S, D/2) and are
// shared across heads (RoPE is head-independent). Computed in f32, bf16 in/out —
// at least as accurate as candle's bf16 rope_i (parity self-test gates it).
//
// Layout: (B,S,H,D) contiguous; element (b,s,h,d) at ((b*S+s)*H+h)*D+d. cos/sin
// indexed by [s * (D/2) + j] for pair j. One block per (b,s,h) token-head; D/2
// threads, each rotating one interleaved pair (x[2j], x[2j+1]).

#include <cuda_bf16.h>

#include "rope_pair.cuh" // shared no-FMA rotation (bit-identical to the fused path)

static __global__ void rope_i_bshd_kernel(__nv_bfloat16 *__restrict__ out,
                                          const __nv_bfloat16 *__restrict__ in,
                                          const __nv_bfloat16 *__restrict__ cos,
                                          const __nv_bfloat16 *__restrict__ sin,
                                          int S, int H, int Dhalf) {
  int token_head = blockIdx.x;   // 0 .. B*S*H
  int j = threadIdx.x;           // 0 .. Dhalf
  if (j >= Dhalf) return;
  // cos/sin are indexed by sequence position s (head-independent). b and h drop
  // out because the (B,S,H,D) base offset is linear in token_head.
  int s = (token_head / H) % S;
  int D = Dhalf * 2;
  const __nv_bfloat16 *xb = in + (size_t)token_head * D;
  __nv_bfloat16 *ob = out + (size_t)token_head * D;
  float c = __bfloat162float(cos[(size_t)s * Dhalf + j]);
  float sn = __bfloat162float(sin[(size_t)s * Dhalf + j]);
  float x0 = __bfloat162float(xb[2 * j]);
  float x1 = __bfloat162float(xb[2 * j + 1]);
  rope_pair_bf16(x0, x1, c, sn, ob[2 * j], ob[2 * j + 1]);
}

// out, in: (B,S,H,D) bf16 contiguous. cos,sin: (S, D/2) bf16 contiguous.
extern "C" void rope_i_bshd_launch(void *out, const void *in, const void *cos,
                                   const void *sin, int B, int S, int H,
                                   int Dhalf, void *stream) {
  dim3 grid((unsigned)B * S * H);
  dim3 block((unsigned)Dhalf);
  rope_i_bshd_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      (__nv_bfloat16 *)out, (const __nv_bfloat16 *)in,
      (const __nv_bfloat16 *)cos, (const __nv_bfloat16 *)sin, S, H, Dhalf);
}
