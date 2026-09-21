//! Fused LayerNorm(no-affine) + AdaLN `(scale+1)` bridge. A candle `CustomOp2`
//! that launches `fused_norm_mod` (kernels/fusednorm/fused_norm.cu), replacing
//! the ~8 candle ops of `norm_no_affine(x) * (scale+1)` with one kernel. bf16
//! in/out, f32 accumulation. Only compiled under the `fusednorm` feature.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::Result;

extern "C" {
    fn fused_norm_mod_launch(
        out: *mut c_void,
        x: *const c_void,
        scale: *const c_void,
        m: i32,
        n: i32,
        eps: f32,
        stream: *mut c_void,
    );
}

struct FusedNormMod {
    eps: f32,
}

impl candle_core::CustomOp2 for FusedNormMod {
    fn name(&self) -> &'static str {
        "fused-norm-mod"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused-norm-mod is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let dims = x_l.shape().dims().to_vec();
        if dims != scale_l.shape().dims() {
            candle_core::bail!(
                "fused-norm-mod shape mismatch: {:?} vs {:?}",
                dims,
                scale_l.shape().dims()
            );
        }
        let n = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x = x.as_cuda_slice::<half::bf16>()?;
        let scale = scale.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
        {
            let (xp, _a) = x.device_ptr(&stream);
            let (sp, _b) = scale.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            unsafe {
                fused_norm_mod_launch(
                    op as *mut c_void,
                    xp as *const c_void,
                    sp as *const c_void,
                    m as i32,
                    n as i32,
                    self.eps,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

/// `norm_no_affine(x) * (scale + 1)` fused. `x` and `scale` share the shape
/// `(..., N)`; normalization is over the last dim. bf16 in/out.
pub fn fused_norm_mod(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let scale = scale.contiguous()?;
    Ok(x.apply_op2(&scale, FusedNormMod { eps })?)
}

/// Validate the fused kernel vs the candle reference (LayerNorm no-affine, f32,
/// then `*(scale+1)`) at the real DiT shape: cosine similarity.
pub fn self_test() -> Result<f32> {
    use candle_core::{DType, Device, D};
    let dev = Device::new_cuda(0)?;
    let (m, n) = (4117usize, 4096usize); // seq × INNER
    let eps = 1e-6f32;
    let x = Tensor::randn(0f32, 1f32, (m, n), &dev)?.to_dtype(DType::BF16)?;
    let scale = Tensor::randn(0f32, 0.5f32, (m, n), &dev)?.to_dtype(DType::BF16)?;

    let y = fused_norm_mod(&x, &scale, eps)?.to_dtype(DType::F32)?;

    // candle reference: norm_no_affine(x, eps) * (scale + 1)
    let x32 = x.to_dtype(DType::F32)?;
    let mean = x32.mean_keepdim(D::Minus1)?;
    let xc = x32.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    let yref = normed.broadcast_mul(&(scale.to_dtype(DType::F32)? + 1.0)?)?;

    let dot = (&y * &yref)?.sum_all()?.to_scalar::<f32>()?;
    let na = y.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = yref.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}
