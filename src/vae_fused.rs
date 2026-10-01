//! VAE decoder fused ops (feature `fusednorm`): bridges to
//! kernels/fusednorm/vae_norm.cu.
//!
//! - [`rmsnorm`]: channel RmsNorm × gamma (+ SiLU) over NCHW bf16, one kernel,
//!   byte-identical to the candle chain in `model::vae` (see the .cu header for
//!   how candle's `fast_sum` association is reproduced).
//! - [`bias_residual`]: `(y + bias) + r` with both bf16 roundings, one pass.
//!
//! CUDA bf16 only; callers keep the candle path for anything else.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::layout::dense_byte_offset;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn vae_rmsnorm_launch(
        out: *mut c_void,
        x: *const c_void,
        gamma: *const c_void,
        b: i64,
        c: i32,
        hw: i64,
        silu: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn vae_bias_residual_launch(
        out: *mut c_void,
        y: *const c_void,
        bias: *const c_void,
        r: *const c_void,
        numel: i64,
        hw: i64,
        c: i32,
        stream: *mut c_void,
    );
}

/// Channel counts the norm kernel takes (`[8, 2048]`).
pub fn rmsnorm_supports(c: usize) -> bool {
    (8..=2048).contains(&c)
}

struct VaeRmsNorm {
    silu: bool,
}

impl candle_core::CustomOp2 for VaeRmsNorm {
    fn name(&self) -> &'static str {
        "vae-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("vae-rmsnorm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        g: &CudaStorage,
        g_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (b, c, h, w) = x_l.shape().dims4()?;
        if g_l.shape().elem_count() != c {
            candle_core::bail!(
                "vae-rmsnorm: gamma {:?} does not match C = {c}",
                g_l.shape()
            );
        }
        if !rmsnorm_supports(c) {
            candle_core::bail!("vae-rmsnorm: C = {c} outside [8, 2048]");
        }
        let xo = dense_byte_offset::<half::bf16>(x_l, "vae-rmsnorm x")?;
        let go = dense_byte_offset::<half::bf16>(g_l, "vae-rmsnorm gamma")?;
        let x = x.as_cuda_slice::<half::bf16>()?;
        let g = g.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(b * c * h * w)? };
        let rc = {
            let (xp, _a) = x.device_ptr(&stream);
            let (gp, _b) = g.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            unsafe {
                vae_rmsnorm_launch(
                    op as *mut c_void,
                    (xp + xo) as *const c_void,
                    (gp + go) as *const c_void,
                    b as i64,
                    c as i32,
                    (h * w) as i64,
                    self.silu as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            candle_core::bail!("vae-rmsnorm: launcher rejected C = {c}");
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

/// `F.normalize(x, dim=1) * sqrt(C) * gamma`, then SiLU if `silu`, for a
/// `(B, C, H, W)` bf16 CUDA tensor; `gamma` has `C` elements (bf16).
pub fn rmsnorm(x: &Tensor, gamma: &Tensor, silu: bool) -> candle_core::Result<Tensor> {
    if x.dtype() != DType::BF16 || gamma.dtype() != DType::BF16 {
        candle_core::bail!("vae-rmsnorm: bf16 only");
    }
    x.contiguous()?
        .apply_op2_no_bwd(&gamma.contiguous()?, &VaeRmsNorm { silu })
}

struct VaeBiasResidual;

impl candle_core::CustomOp3 for VaeBiasResidual {
    fn name(&self) -> &'static str {
        "vae-bias-residual"
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
        candle_core::bail!("vae-bias-residual is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        y: &CudaStorage,
        y_l: &Layout,
        bias: &CudaStorage,
        bias_l: &Layout,
        r: &CudaStorage,
        r_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = y.device().clone();
        let (b, c, h, w) = y_l.shape().dims4()?;
        if r_l.shape() != y_l.shape() || bias_l.shape().elem_count() != c {
            candle_core::bail!(
                "vae-bias-residual shape mismatch: y {:?} bias {:?} r {:?}",
                y_l.shape(),
                bias_l.shape(),
                r_l.shape()
            );
        }
        let yo = dense_byte_offset::<half::bf16>(y_l, "vae-bias-residual y")?;
        let bo = dense_byte_offset::<half::bf16>(bias_l, "vae-bias-residual bias")?;
        let ro = dense_byte_offset::<half::bf16>(r_l, "vae-bias-residual r")?;
        let y = y.as_cuda_slice::<half::bf16>()?;
        let bias = bias.as_cuda_slice::<half::bf16>()?;
        let r = r.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let n = b * c * h * w;
        let out = unsafe { dev.alloc::<half::bf16>(n)? };
        {
            let (yp, _a) = y.device_ptr(&stream);
            let (bp, _b) = bias.device_ptr(&stream);
            let (rp, _c) = r.device_ptr(&stream);
            let (op, _d) = out.device_ptr(&stream);
            unsafe {
                vae_bias_residual_launch(
                    op as *mut c_void,
                    (yp + yo) as *const c_void,
                    (bp + bo) as *const c_void,
                    (rp + ro) as *const c_void,
                    n as i64,
                    (h * w) as i64,
                    c as i32,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), y_l.shape().clone()))
    }
}

/// `y.broadcast_add(bias as (1,C,1,1)) + r`, both adds rounded to bf16 as the
/// two candle ops would. `y`, `r`: `(B, C, H, W)` bf16; `bias`: `C` elements.
pub fn bias_residual(y: &Tensor, bias: &Tensor, r: &Tensor) -> candle_core::Result<Tensor> {
    for t in [y, bias, r] {
        if t.dtype() != DType::BF16 {
            candle_core::bail!("vae-bias-residual: bf16 only");
        }
    }
    y.contiguous()?
        .apply_op3_no_bwd(&bias.contiguous()?, &r.contiguous()?, &VaeBiasResidual)
}

/// The candle reference chains (what `model::vae` runs without the fused ops).
pub fn rmsnorm_reference(x: &Tensor, gamma: &Tensor, silu: bool) -> candle_core::Result<Tensor> {
    let c = x.dim(1)?;
    let xf = x.to_dtype(DType::F32)?;
    let norm = xf
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    let y = (xf.broadcast_div(&norm)? * (c as f64).sqrt())?.to_dtype(x.dtype())?;
    let y = y.broadcast_mul(&gamma.reshape((1, c, 1, 1))?)?;
    if silu {
        candle_nn::ops::silu(&y)
    } else {
        Ok(y)
    }
}

fn mismatches(a: &Tensor, b: &Tensor) -> candle_core::Result<usize> {
    // Compare the raw bf16 bit patterns (catches -0 vs +0 too).
    let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(a.iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count())
}

/// `vae-fused-test`: every fused op vs its candle chain, bit for bit.
/// Returns `(case, mismatching elements)`.
pub fn self_test(dev: &candle_core::Device) -> candle_core::Result<Vec<(String, usize)>> {
    let mut out = Vec::new();
    // (B, C, H, W): VAE channel counts, an odd C, C > 1024, a tile-edge W.
    let shapes = [
        (1, 1152, 8, 8),
        (1, 576, 16, 12),
        (1, 288, 9, 37),
        (2, 144, 16, 16),
        (1, 64, 5, 7),
        (1, 100, 6, 6),
        (1, 1030, 4, 4),
        (1, 2048, 3, 3),
    ];
    for &(b, c, h, w) in &shapes {
        let mut x = (Tensor::randn(0f32, 1f32, (b, c, h, w), dev)? * 3.0)?;
        // A few exact zeros / negative zeros, and one all-zero pixel column.
        let mut v = x.flatten_all()?.to_vec1::<f32>()?;
        for (i, e) in v.iter_mut().enumerate() {
            if i % 97 == 0 {
                *e = 0.0;
            } else if i % 89 == 0 {
                *e = -0.0;
            }
        }
        let hw = h * w;
        for ch in 0..c {
            v[ch * hw] = if ch % 2 == 0 { 0.0 } else { -0.0 };
        }
        x = Tensor::from_vec(v, (b, c, h, w), dev)?;
        let x = x.to_dtype(DType::BF16)?;
        let gamma = (Tensor::randn(0f32, 1f32, c, dev)? + 1.0)?.to_dtype(DType::BF16)?;
        for silu in [false, true] {
            let ours = rmsnorm(&x, &gamma, silu)?;
            let refr = rmsnorm_reference(&x, &gamma, silu)?;
            out.push((
                format!("rmsnorm ({b},{c},{h},{w}) silu={silu}"),
                mismatches(&ours, &refr)?,
            ));
        }
        let bias = Tensor::randn(0f32, 1f32, c, dev)?.to_dtype(DType::BF16)?;
        let r = Tensor::randn(0f32, 2f32, (b, c, h, w), dev)?.to_dtype(DType::BF16)?;
        let ours = bias_residual(&x, &bias, &r)?;
        let refr = (x.broadcast_add(&bias.reshape((1, c, 1, 1))?)? + &r)?;
        out.push((
            format!("bias_residual ({b},{c},{h},{w})"),
            mismatches(&ours, &refr)?,
        ));
    }
    // Offset views: a B=1 narrow of a B=2 batch is dense at offset C*H*W.
    let big = Tensor::randn(0f32, 1f32, (2, 144, 8, 8), dev)?.to_dtype(DType::BF16)?;
    let view = big.narrow(0, 1, 1)?;
    let fresh = view.to_dtype(DType::F32)?.to_dtype(DType::BF16)?;
    let gamma = Tensor::ones(144, DType::BF16, dev)?;
    let off = view.layout().start_offset();
    let d = mismatches(
        &rmsnorm(&view, &gamma, true)?,
        &rmsnorm(&fresh, &gamma, true)?,
    )?;
    out.push((
        format!("rmsnorm offset view (start_offset {off})"),
        d + (off == 0) as usize,
    ));
    let d = mismatches(
        &bias_residual(&view, &gamma, &view)?,
        &bias_residual(&fresh, &gamma, &fresh)?,
    )?;
    out.push((
        format!("bias_residual offset view (start_offset {off})"),
        d + (off == 0) as usize,
    ));
    Ok(out)
}
