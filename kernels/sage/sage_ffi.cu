// Raw-pointer FFI launchers for SageAttention's INT8-QK / FP16-PV kernel
// (thu-ml/SageAttention, csrc/qattn/qk_int_sv_f16_cuda_sm80.cu). The vendored
// kernel and its headers under vendor/ are torch-free (the torch launchers were
// stripped); these `extern "C"` wrappers replicate the launch config for our
// fixed DiT shape — head_dim 128, per-warp INT8 scales, float SV accum, bf16
// output — and are called from Rust (src/sage.rs) via candle CustomOp3.
//
// Config (from the original launcher): CTA_Q 128, CTA_K 64, WARP_Q 32,
// WARP_K 64; per-warp quant means one Q scale per 32 tokens, one K scale per 64.
// Tensor layout is HND = (B, H, N, D) contiguous, so strides are
// bz = H*N*D, h = N*D, seq = D.

#include <cassert> // must precede cuda_fp8/fp6/fp4 headers (CUDA 13 __assert_fail)
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "vendor/qattn/qk_int_sv_f16_sm80.cuh" // qk_int_sv_f16_attn_kernel + enums
#include "vendor/fused/fused_quant.cuh"        // QuantInt8Kernel
#include "rope_pair.cuh"                      // shared no-FMA RoPE rotation

static constexpr uint32_t HEAD_DIM = 128;

// Per-BLOCK_SIZE INT8 quantization of a (B,H,N,128) bf16 tensor. Produces int8
// bytes + one f32 scale (amax/127) per BLOCK_SIZE tokens. No sm_scale fused
// (has_sm_scale=false) — the attention kernel applies sm_scale itself.
template <uint32_t BLOCK_SIZE>
static void launch_quant(const __nv_bfloat16 *in, int8_t *out, float *scale,
                         int B, int H, int N, cudaStream_t s) {
  uint32_t nblk = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(nblk, H, B);
  constexpr uint32_t num_pack = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
  dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack);
  uint32_t sbz_in = (uint32_t)H * N * HEAD_DIM, sh_in = (uint32_t)N * HEAD_DIM,
           sseq_in = HEAD_DIM;
  uint32_t sbz_sc = (uint32_t)H * nblk, sh_sc = nblk;
  QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack, false, false, __nv_bfloat16>
      <<<grid, block, 0, s>>>((__nv_bfloat16 *)in, nullptr, out, scale, 1.0f,
                              (uint32_t)N, sbz_in, sseq_in, sh_in, 0, 0, sbz_in,
                              sseq_in, sh_in, sbz_sc, sh_sc);
}

// Per-BLOCK granularity: Q scale per CTA_Q=128 tokens, K scale per CTA_K=64.
// Scale counts (ceil(N/128), ceil(N/64)) match the kernel's per-block indexing
// exactly — no padding, unlike per-warp.
extern "C" void sage_quant_q(const void *in, void *out_i8, float *scale, int B,
                             int H, int N, void *stream) {
  launch_quant<128>((const __nv_bfloat16 *)in, (int8_t *)out_i8, scale, B, H, N,
                    (cudaStream_t)stream);
}

extern "C" void sage_quant_k(const void *in, void *out_i8, float *scale, int B,
                             int H, int N, void *stream) {
  launch_quant<64>((const __nv_bfloat16 *)in, (int8_t *)out_i8, scale, B, H, N,
                   (cudaStream_t)stream);
}

// BSHD variant: the bf16 input is read with caller-supplied strides (a
// (B,S,H,D) tensor, possibly a narrowed view — see src/sage.rs), while the int8
// output + scales are written HND-contiguous (unchanged from the HND path, so
// the attention kernel indexes them identically). Only the input strides differ.
template <uint32_t BLOCK_SIZE>
static void launch_quant_bshd(const __nv_bfloat16 *in, int8_t *out, float *scale,
                              int B, int H, int N, uint32_t sbz_in,
                              uint32_t sseq_in, uint32_t sh_in,
                              cudaStream_t s) {
  uint32_t nblk = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(nblk, H, B);
  constexpr uint32_t num_pack = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
  dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack);
  uint32_t sbz_out = (uint32_t)H * N * HEAD_DIM, sh_out = (uint32_t)N * HEAD_DIM,
           sseq_out = HEAD_DIM;
  uint32_t sbz_sc = (uint32_t)H * nblk, sh_sc = nblk;
  QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack, false, false, __nv_bfloat16>
      <<<grid, block, 0, s>>>((__nv_bfloat16 *)in, nullptr, out, scale, 1.0f,
                              (uint32_t)N, sbz_in, sseq_in, sh_in, 0, 0, sbz_out,
                              sseq_out, sh_out, sbz_sc, sh_sc);
}

extern "C" void sage_quant_q_bshd(const void *in, void *out_i8, float *scale,
                                  int B, int H, int N, unsigned sbz_in,
                                  unsigned sseq_in, unsigned sh_in,
                                  void *stream) {
  launch_quant_bshd<128>((const __nv_bfloat16 *)in, (int8_t *)out_i8, scale, B,
                         H, N, sbz_in, sseq_in, sh_in, (cudaStream_t)stream);
}

extern "C" void sage_quant_k_bshd(const void *in, void *out_i8, float *scale,
                                  int B, int H, int N, unsigned sbz_in,
                                  unsigned sseq_in, unsigned sh_in,
                                  void *stream) {
  launch_quant_bshd<64>((const __nv_bfloat16 *)in, (int8_t *)out_i8, scale, B, H,
                        N, sbz_in, sseq_in, sh_in, (cudaStream_t)stream);
}

// Fused interleaved-RoPE + per-block INT8 quantization (SageAttention's own IO
// trick): reads PRE-rope bf16 q/k with caller strides (a (B,S,H,D) S-axis view),
// rotates each 8-element pack's 4 interleaved pairs with the view-local token's
// cos/sin row, rounds to bf16 (exactly what the unfused rope kernel stores), then
// runs the verbatim QuantInt8Kernel tail (has_sm_scale=false, sub_mean=false):
// block amax -> scale = amax/127 -> int8. Output int8 HND + scales exactly as
// launch_quant_bshd writes them, so the attention kernel indexes them unchanged.
// Bit-identical to rope_i_bshd -> sage_quant_*_bshd (gated by
// sage::self_test_rope_quant). cos/sin: rows of D/2 bf16, row stride
// `stride_seq_cs`, row 0 = the view's first token.
template <uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread>
__global__ void RopeQuantInt8Kernel(
    const __nv_bfloat16 *__restrict__ input, const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin, int8_t *__restrict__ output,
    float *__restrict__ scale, const uint32_t num_tokens,
    const uint32_t stride_bz_input, const uint32_t stride_seq_input,
    const uint32_t stride_h_input, const uint32_t stride_seq_cs,
    const uint32_t stride_bz_output, const uint32_t stride_seq_output,
    const uint32_t stride_h_output, const uint32_t stride_bz_scale,
    const uint32_t stride_h_scale) {
  constexpr uint32_t pack_size = 8; // float4 = 8 bf16 = 4 interleaved pairs
  constexpr uint32_t num_threads_per_token = HEAD_DIM / pack_size;
  static_assert(num_threads_per_token <= 32, "threads per token <= warp size");

  __nv_bfloat16 x_val[8];
  float x_val_float[num_pack_per_thread][8];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  uint32_t d0 = thread_id % num_threads_per_token * pack_size;
  const __nv_bfloat16 *input_ptr_base = input + batch_id * stride_bz_input +
                                        head_id * stride_h_input +
                                        thread_base_token * stride_seq_input + d0;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output +
                            head_id * stride_h_output +
                            thread_base_token * stride_seq_output + d0;
  float *scale_ptr_base =
      scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  for (uint32_t i = 0; i < num_pack_per_thread; i++) {
    uint32_t tok = thread_base_token + i * iter_stride;
    if (tok < num_tokens) {
      *(float4 *)(&x_val[0]) =
          *(const float4 *)(input_ptr_base + i * iter_stride * stride_seq_input);
      const __nv_bfloat16 *c_row = cos + (size_t)tok * stride_seq_cs + d0 / 2;
      const __nv_bfloat16 *s_row = sin + (size_t)tok * stride_seq_cs + d0 / 2;
#pragma unroll
      for (uint32_t p = 0; p < 4; p++) {
        __nv_bfloat16 y0, y1;
        rope_pair_bf16(__bfloat162float(x_val[2 * p]),
                       __bfloat162float(x_val[2 * p + 1]),
                       __bfloat162float(c_row[p]), __bfloat162float(s_row[p]),
                       y0, y1);
        x_val_float[i][2 * p] = __bfloat162float(y0);
        x_val_float[i][2 * p + 1] = __bfloat162float(y1);
      }
    } else {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
        x_val_float[i][j] = 0.0f;
    }
  }

  // --- verbatim QuantInt8Kernel tail ---
  float amax_val = 0.0000001f; // prevent from dividing by zero
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++) {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (thread_id == 0) {
    s_amax = block_amax_val;
    scale_ptr_base[0] = s_amax / 127.0f;
  }
  __syncthreads();

  float tmp_scale = 127.0f / s_amax;
  char4 o_val[num_pack_per_thread][2];
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++) {
#pragma unroll
    for (uint32_t j = 0; j < 2; j += 1) {
      o_val[i][j] = make_char4(float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
                               float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
                               float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
                               float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale));
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++) {
    if (thread_base_token + i * iter_stride < num_tokens) {
      *reinterpret_cast<float2 *>(output_ptr_base + i * iter_stride * stride_seq_output) =
          *reinterpret_cast<float2 *>(&o_val[i][0]);
    }
  }
}

template <uint32_t BLOCK_SIZE>
static void launch_rope_quant_bshd(const __nv_bfloat16 *in,
                                   const __nv_bfloat16 *cos,
                                   const __nv_bfloat16 *sin, int8_t *out,
                                   float *scale, int B, int H, int N,
                                   uint32_t sbz_in, uint32_t sseq_in,
                                   uint32_t sh_in, uint32_t sseq_cs,
                                   cudaStream_t s) {
  uint32_t nblk = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(nblk, H, B);
  constexpr uint32_t num_pack = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
  dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack);
  uint32_t sbz_out = (uint32_t)H * N * HEAD_DIM, sh_out = (uint32_t)N * HEAD_DIM,
           sseq_out = HEAD_DIM;
  uint32_t sbz_sc = (uint32_t)H * nblk, sh_sc = nblk;
  RopeQuantInt8Kernel<BLOCK_SIZE, num_pack><<<grid, block, 0, s>>>(
      in, cos, sin, out, scale, (uint32_t)N, sbz_in, sseq_in, sh_in, sseq_cs,
      sbz_out, sseq_out, sh_out, sbz_sc, sh_sc);
}

// is_q selects the Q block (128 tokens) vs K block (64 tokens) granularity.
extern "C" void sage_rope_quant_bshd_launch(const void *in, const void *cos,
                                            const void *sin, void *out_i8,
                                            float *scale, int B, int H, int N,
                                            unsigned sbz_in, unsigned sseq_in,
                                            unsigned sh_in, unsigned sseq_cs,
                                            int is_q, void *stream) {
  if (is_q)
    launch_rope_quant_bshd<128>(
        (const __nv_bfloat16 *)in, (const __nv_bfloat16 *)cos,
        (const __nv_bfloat16 *)sin, (int8_t *)out_i8, scale, B, H, N, sbz_in,
        sseq_in, sh_in, sseq_cs, (cudaStream_t)stream);
  else
    launch_rope_quant_bshd<64>(
        (const __nv_bfloat16 *)in, (const __nv_bfloat16 *)cos,
        (const __nv_bfloat16 *)sin, (int8_t *)out_i8, scale, B, H, N, sbz_in,
        sseq_in, sh_in, sseq_cs, (cudaStream_t)stream);
}

template <MaskMode MM>
static void launch_attn(const int8_t *q, const int8_t *k, const half *v,
                        __nv_bfloat16 *o, const float *qs, const float *ks,
                        int B, int Hq, int Hk, int qo_len, int kv_len,
                        float sm_scale, cudaStream_t stream) {
  constexpr int CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
  int num_kv_groups = Hq / Hk;
  uint32_t sbz_q = (uint32_t)Hq * qo_len * HEAD_DIM, sh_q = (uint32_t)qo_len * HEAD_DIM,
           sseq_q = HEAD_DIM;
  uint32_t sbz_k = (uint32_t)Hk * kv_len * HEAD_DIM, sh_k = (uint32_t)kv_len * HEAD_DIM,
           sseq_k = HEAD_DIM;
  size_t smem_max = std::max((size_t)CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                 CTA_K * HEAD_DIM * sizeof(int8_t) +
                                 CTA_K * HEAD_DIM * sizeof(half),
                             (size_t)CTA_Q * HEAD_DIM * sizeof(half));
  auto kf = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
                                      DataType::kInt8, QuantGranularity::kPerBlock,
                                      QuantGranularity::kPerBlock, float, false,
                                      __nv_bfloat16, ComputeUnit::kTensorCore, MM,
                                      false, false>;
  cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
  dim3 grid((qo_len + CTA_Q - 1) / CTA_Q, Hq, B);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  kf<<<grid, block, smem_max, stream>>>(
      (int8_t *)q, (int8_t *)k, (half *)v, o, nullptr, (float *)qs, (float *)ks,
      nullptr, qo_len, kv_len, num_kv_groups, sbz_q, sseq_q, sh_q, sbz_k, sseq_k,
      sh_k, sbz_k, sseq_k, sh_k, sbz_q, sseq_q, sh_q, sm_scale);
}

// INT8-QK / FP16-PV attention. q,k are int8 (in u8 storage); v is fp16; o is
// bf16. Layout HND. `is_causal` selects the block-causal mask.
extern "C" void sage_attn(const void *q_i8, const void *k_i8, const void *v_f16,
                          void *o_bf16, const float *q_scale,
                          const float *k_scale, int B, int Hq, int Hk,
                          int qo_len, int kv_len, float sm_scale, int is_causal,
                          void *stream) {
  if (is_causal)
    launch_attn<MaskMode::kCausal>((const int8_t *)q_i8, (const int8_t *)k_i8,
                                   (const half *)v_f16, (__nv_bfloat16 *)o_bf16,
                                   q_scale, k_scale, B, Hq, Hk, qo_len, kv_len,
                                   sm_scale, (cudaStream_t)stream);
  else
    launch_attn<MaskMode::kNone>((const int8_t *)q_i8, (const int8_t *)k_i8,
                                 (const half *)v_f16, (__nv_bfloat16 *)o_bf16,
                                 q_scale, k_scale, B, Hq, Hk, qo_len, kv_len,
                                 sm_scale, (cudaStream_t)stream);
}

// BSHD variant: int8 q/k + their scales stay HND-contiguous (as produced by the
// _bshd quant launchers above), so q/k strides are the HND ones. V (f16) and O
// (bf16) are read/written with caller-supplied (B,S,H,D) strides — V may be a
// narrowed view, O is a fresh contiguous (B, qo, Hq, D) buffer. The kernel
// honors stride_bz/seq/h independently for every operand; only head_dim must be
// stride-1 (satisfied in both layouts).
template <MaskMode MM>
static void launch_attn_bshd(const int8_t *q, const int8_t *k, const half *v,
                             __nv_bfloat16 *o, const float *qs, const float *ks,
                             int B, int Hq, int Hk, int qo_len, int kv_len,
                             uint32_t sbz_v, uint32_t sseq_v, uint32_t sh_v,
                             uint32_t sbz_o, uint32_t sseq_o, uint32_t sh_o,
                             float sm_scale, cudaStream_t stream) {
  constexpr int CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
  int num_kv_groups = Hq / Hk;
  uint32_t sbz_q = (uint32_t)Hq * qo_len * HEAD_DIM,
           sh_q = (uint32_t)qo_len * HEAD_DIM, sseq_q = HEAD_DIM;
  uint32_t sbz_k = (uint32_t)Hk * kv_len * HEAD_DIM,
           sh_k = (uint32_t)kv_len * HEAD_DIM, sseq_k = HEAD_DIM;
  size_t smem_max = std::max((size_t)CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                 CTA_K * HEAD_DIM * sizeof(int8_t) +
                                 CTA_K * HEAD_DIM * sizeof(half),
                             (size_t)CTA_Q * HEAD_DIM * sizeof(half));
  auto kf = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
                                      DataType::kInt8, QuantGranularity::kPerBlock,
                                      QuantGranularity::kPerBlock, float, false,
                                      __nv_bfloat16, ComputeUnit::kTensorCore, MM,
                                      false, false>;
  cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
  dim3 grid((qo_len + CTA_Q - 1) / CTA_Q, Hq, B);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  kf<<<grid, block, smem_max, stream>>>(
      (int8_t *)q, (int8_t *)k, (half *)v, o, nullptr, (float *)qs, (float *)ks,
      nullptr, qo_len, kv_len, num_kv_groups, sbz_q, sseq_q, sh_q, sbz_k, sseq_k,
      sh_k, sbz_v, sseq_v, sh_v, sbz_o, sseq_o, sh_o, sm_scale);
}

extern "C" void sage_attn_bshd(const void *q_i8, const void *k_i8,
                               const void *v_f16, void *o_bf16,
                               const float *q_scale, const float *k_scale, int B,
                               int Hq, int Hk, int qo_len, int kv_len,
                               unsigned sbz_v, unsigned sseq_v, unsigned sh_v,
                               unsigned sbz_o, unsigned sseq_o, unsigned sh_o,
                               float sm_scale, int is_causal, void *stream) {
  if (is_causal)
    launch_attn_bshd<MaskMode::kCausal>(
        (const int8_t *)q_i8, (const int8_t *)k_i8, (const half *)v_f16,
        (__nv_bfloat16 *)o_bf16, q_scale, k_scale, B, Hq, Hk, qo_len, kv_len,
        sbz_v, sseq_v, sh_v, sbz_o, sseq_o, sh_o, sm_scale,
        (cudaStream_t)stream);
  else
    launch_attn_bshd<MaskMode::kNone>(
        (const int8_t *)q_i8, (const int8_t *)k_i8, (const half *)v_f16,
        (__nv_bfloat16 *)o_bf16, q_scale, k_scale, B, Hq, Hk, qo_len, kv_len,
        sbz_v, sseq_v, sh_v, sbz_o, sseq_o, sh_o, sm_scale,
        (cudaStream_t)stream);
}
