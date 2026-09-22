//! SageAttention INT8-QK / FP16-PV bridge. A candle `CustomOp3` that, given
//! `q,k` (bf16) and `v` (fp16) in layout `(B, H, S, D)`, quantizes q,k to INT8
//! per-block, runs the vendored fused attention kernel (kernels/sage/), and
//! returns bf16 output. Only compiled under the `sage` feature (implies `cuda`).
//!
//! INT8 rides in U8 storage (candle has no I8), same as the convrot bridge.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::Result;

extern "C" {
    fn sage_quant_q(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        stream: *mut c_void,
    );
    fn sage_quant_k(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        stream: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    fn sage_attn(
        q_i8: *const c_void,
        k_i8: *const c_void,
        v_f16: *const c_void,
        o_bf16: *mut c_void,
        q_scale: *const f32,
        k_scale: *const f32,
        b: i32,
        hq: i32,
        hk: i32,
        qo: i32,
        kv: i32,
        sm_scale: f32,
        is_causal: i32,
        stream: *mut c_void,
    );
}

extern "C" {
    fn sage_quant_q_bshd(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        sbz_in: u32,
        sseq_in: u32,
        sh_in: u32,
        stream: *mut c_void,
    );
    fn sage_quant_k_bshd(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        sbz_in: u32,
        sseq_in: u32,
        sh_in: u32,
        stream: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    fn sage_attn_bshd(
        q_i8: *const c_void,
        k_i8: *const c_void,
        v_f16: *const c_void,
        o_bf16: *mut c_void,
        q_scale: *const f32,
        k_scale: *const f32,
        b: i32,
        hq: i32,
        hk: i32,
        qo: i32,
        kv: i32,
        sbz_v: u32,
        sseq_v: u32,
        sh_v: u32,
        sbz_o: u32,
        sseq_o: u32,
        sh_o: u32,
        sm_scale: f32,
        is_causal: i32,
        stream: *mut c_void,
    );
}

struct SageAttn {
    scale: f32,
    causal: bool,
}

impl candle_core::CustomOp3 for SageAttn {
    fn name(&self) -> &'static str {
        "sage-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("sage-attn is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        _vl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let (b, hq, s, d) = ql.shape().dims4()?;
        let (_, hk, skv, _) = kl.shape().dims4()?;
        let q_s = q.as_cuda_slice::<half::bf16>()?;
        let k_s = k.as_cuda_slice::<half::bf16>()?;
        let v_s = v.as_cuda_slice::<half::f16>()?;
        let stream = dev.cuda_stream();

        let nq_blk = s.div_ceil(128);
        let nk_blk = skv.div_ceil(64);
        let q_i8 = unsafe { dev.alloc::<u8>(b * hq * s * d)? };
        let k_i8 = unsafe { dev.alloc::<u8>(b * hk * skv * d)? };
        let q_scale = unsafe { dev.alloc::<f32>(b * hq * nq_blk)? };
        let k_scale = unsafe { dev.alloc::<f32>(b * hk * nk_blk)? };
        let o = unsafe { dev.alloc::<half::bf16>(b * hq * s * d)? };

        {
            let (qp, _a) = q_s.device_ptr(&stream);
            let (kp, _b) = k_s.device_ptr(&stream);
            let (vp, _c) = v_s.device_ptr(&stream);
            let (qip, _d) = q_i8.device_ptr(&stream);
            let (kip, _e) = k_i8.device_ptr(&stream);
            let (qsp, _f) = q_scale.device_ptr(&stream);
            let (ksp, _g) = k_scale.device_ptr(&stream);
            let (op, _h) = o.device_ptr(&stream);
            let st = stream.cu_stream() as *mut c_void;
            unsafe {
                sage_quant_q(
                    qp as *const c_void,
                    qip as *mut c_void,
                    qsp as *mut f32,
                    b as i32,
                    hq as i32,
                    s as i32,
                    st,
                );
                sage_quant_k(
                    kp as *const c_void,
                    kip as *mut c_void,
                    ksp as *mut f32,
                    b as i32,
                    hk as i32,
                    skv as i32,
                    st,
                );
                sage_attn(
                    qip as *const c_void,
                    kip as *const c_void,
                    vp as *const c_void,
                    op as *mut c_void,
                    qsp as *const f32,
                    ksp as *const f32,
                    b as i32,
                    hq as i32,
                    hk as i32,
                    s as i32,
                    skv as i32,
                    self.scale,
                    self.causal as i32,
                    st,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(o, dev), (b, hq, s, d).into()))
    }
}

/// INT8-QK / FP16-PV attention. `q,k,v` are `(B, H, S, D)` bf16 on CUDA;
/// `scale` is the softmax scale (1/sqrt(D)); `causal` selects the causal mask.
/// Returns `(B, H, S, D)` bf16. D must be a multiple of 64.
pub fn sage_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let vf = v.to_dtype(DType::F16)?.contiguous()?;
    Ok(q.apply_op3(&k, &vf, SageAttn { scale, causal })?)
}

/// BSHD-native SageAttention. `q,k` bf16, `v` f16, all `(B, S, H, D)` (may be
/// narrowed views on the S axis — start_offset + strides are honored, so the
/// block-causal narrows are zero-copy). The int8 q/k intermediate + scales stay
/// HND-contiguous (so the attention kernel indexes them exactly as the HND path
/// does); only V is read and O is written in BSHD. Returns `(B, S, H, D)` bf16.
struct SageAttnBshd {
    scale: f32,
    causal: bool,
}

impl candle_core::CustomOp3 for SageAttnBshd {
    fn name(&self) -> &'static str {
        "sage-attn-bshd"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("sage-attn-bshd is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let (b, sq, hq, d) = ql.shape().dims4()?;
        let (_, skv, hk, _) = kl.shape().dims4()?;
        // B>1 is exercised by `generate --batch N` (the stride-honoring bridge
        // reads each batch lane via stride_bz). B=1 remains the common path.
        // head_dim must be the innermost (stride-1) dimension — the kernel's
        // vectorized loads require it. True for (B,S,H,D) and its S-axis narrows.
        let qst = ql.stride();
        let kst = kl.stride();
        let vst = vl.stride();
        if qst[3] != 1 || kst[3] != 1 || vst[3] != 1 {
            candle_core::bail!("sage-attn-bshd requires stride-1 head_dim");
        }
        let q_s = q.as_cuda_slice::<half::bf16>()?;
        let k_s = k.as_cuda_slice::<half::bf16>()?;
        let v_s = v.as_cuda_slice::<half::f16>()?;
        let stream = dev.cuda_stream();

        let nq_blk = sq.div_ceil(128);
        let nk_blk = skv.div_ceil(64);
        let q_i8 = unsafe { dev.alloc::<u8>(b * hq * sq * d)? };
        let k_i8 = unsafe { dev.alloc::<u8>(b * hk * skv * d)? };
        let q_scale = unsafe { dev.alloc::<f32>(b * hq * nq_blk)? };
        let k_scale = unsafe { dev.alloc::<f32>(b * hk * nk_blk)? };
        let o = unsafe { dev.alloc::<half::bf16>(b * sq * hq * d)? };

        // Output is a fresh contiguous (B, sq, Hq, D) buffer: BSHD strides.
        let sbz_o = (sq * hq * d) as u32;
        let sseq_o = (hq * d) as u32;
        let sh_o = d as u32;
        {
            let (qp, _a) = q_s.device_ptr(&stream);
            let (kp, _b) = k_s.device_ptr(&stream);
            let (vp, _c) = v_s.device_ptr(&stream);
            let (qip, _d) = q_i8.device_ptr(&stream);
            let (kip, _e) = k_i8.device_ptr(&stream);
            let (qsp, _f) = q_scale.device_ptr(&stream);
            let (ksp, _g) = k_scale.device_ptr(&stream);
            let (op, _h) = o.device_ptr(&stream);
            // Offset the base pointers by each view's start_offset (in elements)
            // so narrowed inputs read the right slice without a copy.
            let qp = qp as usize + ql.start_offset() * std::mem::size_of::<half::bf16>();
            let kp = kp as usize + kl.start_offset() * std::mem::size_of::<half::bf16>();
            let vp = vp as usize + vl.start_offset() * std::mem::size_of::<half::f16>();
            let st = stream.cu_stream() as *mut c_void;
            unsafe {
                sage_quant_q_bshd(
                    qp as *const c_void,
                    qip as *mut c_void,
                    qsp as *mut f32,
                    b as i32,
                    hq as i32,
                    sq as i32,
                    qst[0] as u32,
                    qst[1] as u32,
                    qst[2] as u32,
                    st,
                );
                sage_quant_k_bshd(
                    kp as *const c_void,
                    kip as *mut c_void,
                    ksp as *mut f32,
                    b as i32,
                    hk as i32,
                    skv as i32,
                    kst[0] as u32,
                    kst[1] as u32,
                    kst[2] as u32,
                    st,
                );
                sage_attn_bshd(
                    qip as *const c_void,
                    kip as *const c_void,
                    vp as *const c_void,
                    op as *mut c_void,
                    qsp as *const f32,
                    ksp as *const f32,
                    b as i32,
                    hq as i32,
                    hk as i32,
                    sq as i32,
                    skv as i32,
                    vst[0] as u32,
                    vst[1] as u32,
                    vst[2] as u32,
                    sbz_o,
                    sseq_o,
                    sh_o,
                    self.scale,
                    self.causal as i32,
                    st,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(o, dev), (b, sq, hq, d).into()))
    }
}

/// INT8-QK / FP16-PV attention, BSHD-native. `q,k` are `(B,S,H,D)` bf16, `v` is
/// `(B,S,H,D)` f16 (all may be S-axis narrows). `scale` is 1/sqrt(D); `causal`
/// selects the causal mask. Returns `(B,S,H,D)` bf16. D must be a multiple of 64.
pub fn sage_attention_bshd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    Ok(q.apply_op3(k, v, SageAttnBshd { scale, causal })?)
}

/// Validate INT8 SageAttention vs an f32 softmax-attention reference: cosine.
pub fn self_test(causal: bool) -> Result<f32> {
    use candle_core::{Device, D};
    let dev = Device::new_cuda(0)?;
    let (b, h, s, d) = (1usize, 4usize, 256usize, 128usize);
    let scale = 1.0 / (d as f64).sqrt();
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;

    let y = sage_attention(&q, &k, &v, scale as f32, causal)?.to_dtype(DType::F32)?;

    // f32 reference.
    let qf = q.to_dtype(DType::F32)?;
    let kf = k.to_dtype(DType::F32)?;
    let vf = v.to_dtype(DType::F32)?;
    let mut scores = (qf.matmul(&kf.transpose(D::Minus1, D::Minus2)?)? * scale)?;
    if causal {
        let mut mask = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                mask[i * s + j] = f32::NEG_INFINITY;
            }
        }
        let m = Tensor::from_vec(mask, (1, 1, s, s), &dev)?;
        scores = scores.broadcast_add(&m)?;
    }
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    let yref = attn.matmul(&vf)?;

    let dot = (&y * &yref)?.sum_all()?.to_scalar::<f32>()?;
    let na = y.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = yref.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

/// A/B equivalence: the BSHD path must equal the retained BHSD `sage_attention`
/// (same kernel, same INT8 quant, only the read/write layout differs) on the
/// exact block-causal split the DiT uses (causal text prefix + non-causal image
/// queries). Exercises the S-axis narrows' start_offset. Returns (cosine, maxabs).
pub fn self_test_bshd() -> Result<(f32, f32)> {
    use candle_core::{DType, Device};
    let (b, h, s, d, txt) = (1usize, 4usize, 160usize, 128usize, 32usize);
    let dev = Device::new_cuda(0)?;
    dev.set_seed(42)?; // deterministic inputs
    let sc = (1.0 / (d as f64).sqrt()) as f32;
    let q = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let vf = v.to_dtype(DType::F16)?;

    // BHSD reference (existing path): transpose to (B,H,S,D), narrow on dim 2.
    let qh = q.transpose(1, 2)?.contiguous()?;
    let kh = k.transpose(1, 2)?.contiguous()?;
    let vh = vf.transpose(1, 2)?.contiguous()?;
    let ot = sage_attention(
        &qh.narrow(2, 0, txt)?,
        &kh.narrow(2, 0, txt)?,
        &vh.narrow(2, 0, txt)?,
        sc,
        true,
    )?;
    let qi = qh.narrow(2, txt, s - txt)?;
    let oi = sage_attention(&qi, &kh, &vh, sc, false)?;
    let refb = Tensor::cat(&[ot, oi], 2)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?; // (B,S,H,D)

    // BSHD path (new): narrow on dim 1, no transpose.
    let ot = sage_attention_bshd(
        &q.narrow(1, 0, txt)?,
        &k.narrow(1, 0, txt)?,
        &vf.narrow(1, 0, txt)?,
        sc,
        true,
    )?;
    let qi = q.narrow(1, txt, s - txt)?;
    let oi = sage_attention_bshd(&qi, &k, &vf, sc, false)?;
    let ours = Tensor::cat(&[ot, oi], 1)?.to_dtype(DType::F32)?; // (B,S,H,D)

    // Element-wise on the host, NaN-aware. This test proves the BSHD layout ==
    // the BHSD layout (same kernel, different read/write strides) — NOT that the
    // vendored INT8 kernel never NaNs. On some synthetic randn draws that kernel
    // emits a NaN row (a per-block-quant edge case; real model activations don't
    // trigger it — dit-forward stays clean). Both paths emit it identically, so
    // a position where BOTH are NaN is a MATCH; a one-sided NaN is a real
    // mismatch. maxabs (over finite pairs) is the bit-exactness signal; cosine
    // is over the finite pairs. A one-sided NaN forces maxabs -> NaN (fails).
    let a = ours.flatten_all()?.to_vec1::<f32>()?;
    let bvec = refb.flatten_all()?.to_vec1::<f32>()?;
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    let mut mismatched_nans = 0usize;
    for (x, y) in a.iter().zip(bvec.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        match (x.is_finite(), y.is_finite()) {
            (true, true) => {
                dot += x * y;
                na += x * x;
                nb += y * y;
                let diff = (x - y).abs();
                if diff > maxabs {
                    maxabs = diff;
                }
            }
            // both non-finite: a match only if the same class (NaN vs NaN, or
            // equal signed infinities).
            (false, false) if x.is_nan() == y.is_nan() && (x.is_nan() || x == y) => {}
            _ => mismatched_nans += 1,
        }
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        1.0
    };
    let maxabs = if mismatched_nans > 0 {
        f32::NAN
    } else {
        maxabs as f32
    };
    Ok((cos, maxabs))
}
