//! BSHD interleaved-RoPE: a candle `CustomOp3` that rotates a `(B, S, H, D)`
//! tensor in place of candle's `rope_i` (which requires `(B, H, S, D)`), so the
//! sage attention path keeps q/k in `(B, S, H, D)` and never pays the
//! transpose→contiguous copies. cos/sin are `(S, D/2)`, shared across heads.
//! Only compiled under the `sage` feature (implies `cuda`); the kernel lives in
//! `kernels/sage/rope_bshd.cu` and is linked into the `sage_attn` lib.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::Result;

extern "C" {
    fn rope_i_bshd_launch(
        out: *mut c_void,
        inp: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        b: i32,
        s: i32,
        h: i32,
        dhalf: i32,
        stream: *mut c_void,
    );
}

struct RopeBshd;

impl candle_core::CustomOp3 for RopeBshd {
    fn name(&self) -> &'static str {
        "rope-i-bshd"
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
        candle_core::bail!("rope-i-bshd is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
        sin: &CudaStorage,
        _sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (b, s, h, d) = xl.shape().dims4()?;
        let (cs, dhalf) = cl.shape().dims2()?;
        if cs != s || dhalf * 2 != d {
            candle_core::bail!("rope-i-bshd: cos {cs}x{dhalf} vs x ({s},{d})");
        }
        if xl.start_offset() != 0 || !xl.is_contiguous() {
            candle_core::bail!("rope-i-bshd expects a contiguous (B,S,H,D) input");
        }
        let x_s = x.as_cuda_slice::<half::bf16>()?;
        let cos_s = cos.as_cuda_slice::<half::bf16>()?;
        let sin_s = sin.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(b * s * h * d)? };
        {
            let (xp, _a) = x_s.device_ptr(&stream);
            let (cp, _b) = cos_s.device_ptr(&stream);
            let (sp, _c) = sin_s.device_ptr(&stream);
            let (op, _d) = out.device_ptr(&stream);
            unsafe {
                rope_i_bshd_launch(
                    op as *mut c_void,
                    xp as *const c_void,
                    cp as *const c_void,
                    sp as *const c_void,
                    b as i32,
                    s as i32,
                    h as i32,
                    dhalf as i32,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (b, s, h, d).into()))
    }
}

/// Interleaved RoPE on `x (B,S,H,D)` bf16 with `cos,sin (S, D/2)` bf16.
/// Returns `(B,S,H,D)` bf16 — the same rotation as candle `rope_i` applied to
/// the `(B,H,S,D)` transpose, but without the transpose.
pub fn rope_i_bshd(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok(x.apply_op3(cos, sin, RopeBshd)?)
}

/// Validate BSHD rope vs candle `rope_i` on the transposed reference: cosine.
pub fn self_test() -> Result<f32> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (b, s, h, d) = (1usize, 96usize, 4usize, 128usize);
    let dhalf = d / 2;
    let x = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let cos = Tensor::randn(0f32, 1f32, (s, dhalf), &dev)?.to_dtype(DType::BF16)?;
    let sin = Tensor::randn(0f32, 1f32, (s, dhalf), &dev)?.to_dtype(DType::BF16)?;

    let ours = rope_i_bshd(&x, &cos, &sin)?.to_dtype(DType::F32)?;

    // candle reference: rope_i on (B,H,S,D), transposed back to (B,S,H,D).
    let xh = x.transpose(1, 2)?.contiguous()?; // (B,H,S,D)
    let refh = candle_nn::rotary_emb::rope_i(&xh, &cos, &sin)?;
    let reff = refh.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let dot = (&ours * &reff)?.sum_all()?.to_scalar::<f32>()?;
    let na = ours.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = reff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}
