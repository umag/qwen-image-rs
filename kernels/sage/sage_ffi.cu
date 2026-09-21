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
