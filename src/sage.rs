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
