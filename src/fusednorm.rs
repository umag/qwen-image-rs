//! Fused LayerNorm(no-affine) + AdaLN `(scale+1)` bridge. A candle `CustomOp2`
//! that launches `fused_norm_mod` (kernels/fusednorm/fused_norm.cu), replacing
//! the ~8 candle ops of `norm_no_affine(x) * (scale+1)` with one kernel. bf16
//! in/out, f32 accumulation. Only compiled under the `fusednorm` feature.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::layout::{dense_byte_offset, dense_offset, row_strided_2d};
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
    #[allow(clippy::too_many_arguments)]
    fn fused_rmsnorm_scale_launch(
        out: *mut c_void,
        x: *const c_void,
        w: *const c_void,
        m: i32,
        n: i32,
        ld: std::ffi::c_long,
        eps: f32,
        stream: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    fn fused_rmsnorm_scale_block_launch(
        out: *mut c_void,
        x: *const c_void,
        w: *const c_void,
        m: i32,
        n: i32,
        ld: std::ffi::c_long,
        eps: f32,
        stream: *mut c_void,
    );
    fn fused_gated_residual_launch(
        out: *mut c_void,
        h: *const c_void,
        gate: *const c_void,
        y: *const c_void,
        total: usize,
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
        let xo = dense_byte_offset::<half::bf16>(x_l, "fused-norm-mod x")?;
        let so = dense_byte_offset::<half::bf16>(scale_l, "fused-norm-mod scale")?;
        let x = x.as_cuda_slice::<half::bf16>()?;
        let scale = scale.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
        {
            let (xp, _a) = x.device_ptr(&stream);
            let (sp, _b) = scale.device_ptr(&stream);
            let (xp, sp) = (xp + xo, sp + so);
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

/// Fused RMSNorm(no zero-centering) × per-channel weight: `out[m,n] = x[m,n] *
/// rsqrt(mean_n(x^2) + eps) * W[n]`. `x` is `(..., N)` bf16; `W` is `(N,)` f32.
/// Replaces `x * (1/sqrt(mean(x^2)+eps)) * W` (ZeroCenterRmsNorm/HeadRmsNorm) in
/// src/model/dit.rs. `W` is f32 so it matches candle's `weight.to_dtype(F32)`
/// (Head) / `weight.to_dtype(F32) + 1` (ZeroCenter, baked at load) exactly.
struct FusedRmsnormScale {
    eps: f32,
    /// Always the CTA-per-row kernel (the bit-identity oracle for the N = 128
    /// 16-lane kernel the launcher otherwise picks).
    force_block: bool,
}

impl candle_core::CustomOp2 for FusedRmsnormScale {
    fn name(&self) -> &'static str {
        "fused-rmsnorm-scale"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused-rmsnorm-scale is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        w: &CudaStorage,
        w_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let dims = x_l.shape().dims().to_vec();
        let n = *dims.last().unwrap();
        let wn = w_l.shape().dims1()?;
        if wn != n {
            candle_core::bail!("fused-rmsnorm-scale weight len {wn} != last dim {n}");
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        // 2-D x may be a row-strided view (unit inner stride, rows ld apart);
        // any other rank must be dense.
        let (xo, ld) = if dims.len() == 2 {
            row_strided_2d(x_l, "fused-rmsnorm-scale x")?
        } else {
            (dense_offset(x_l, "fused-rmsnorm-scale x")?, n)
        };
        let xo = (xo * std::mem::size_of::<half::bf16>()) as u64;
        let wo = dense_byte_offset::<f32>(w_l, "fused-rmsnorm-scale w")?;
        let x = x.as_cuda_slice::<half::bf16>()?;
        let w = w.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
        {
            let (xp, _a) = x.device_ptr(&stream);
            let (wp, _b) = w.device_ptr(&stream);
            let (xp, wp) = (xp + xo, wp + wo);
            let (op, _c) = out.device_ptr(&stream);
            let launch = if self.force_block {
                fused_rmsnorm_scale_block_launch
            } else {
                fused_rmsnorm_scale_launch
            };
            unsafe {
                launch(
                    op as *mut c_void,
                    xp as *const c_void,
                    wp as *const c_void,
                    m as i32,
                    n as i32,
                    ld as std::ffi::c_long,
                    self.eps,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

/// `x * rsqrt(mean(x^2)+eps) * W` fused. `x` is `(..., N)` bf16, `W` is `(N,)`
/// f32 (the effective per-channel weight). bf16 out, dense. A 2-D `x` with
/// unit-stride rows (e.g. the q or k half of the merged q|k projection,
/// viewed as `(M*H, 2*D)` and column-narrowed) is read in place; anything
/// else is made contiguous first.
pub fn fused_rmsnorm_scale(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    rmsnorm_scale_impl(x, w, eps, false)
}

fn rmsnorm_scale_impl(x: &Tensor, w: &Tensor, eps: f32, force_block: bool) -> Result<Tensor> {
    let x = if x.rank() == 2 && row_strided_2d(x.layout(), "x").is_ok() {
        x.clone()
    } else {
        x.contiguous()?
    };
    let w = w.contiguous()?;
    Ok(x.apply_op2(&w, FusedRmsnormScale { eps, force_block })?)
}

/// Fused gated residual: `out = h + tanh(gate) * y`, elementwise. `h`, `gate`,
/// `y` share the same shape (bf16). Replaces `h + gate.tanh() * y` (used twice
/// per block) in src/model/dit.rs.
struct FusedGatedResidual;

impl candle_core::CustomOp3 for FusedGatedResidual {
    fn name(&self) -> &'static str {
        "fused-gated-residual"
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
        candle_core::bail!("fused-gated-residual is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        h: &CudaStorage,
        h_l: &Layout,
        gate: &CudaStorage,
        gate_l: &Layout,
        y: &CudaStorage,
        y_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = h.device().clone();
        let dims = h_l.shape().dims().to_vec();
        if dims != gate_l.shape().dims() || dims != y_l.shape().dims() {
            candle_core::bail!(
                "fused-gated-residual shape mismatch: h {:?} gate {:?} y {:?}",
                dims,
                gate_l.shape().dims(),
                y_l.shape().dims()
            );
        }
        let total: usize = dims.iter().product();
        let ho = dense_byte_offset::<half::bf16>(h_l, "fused-gated-residual h")?;
        let go = dense_byte_offset::<half::bf16>(gate_l, "fused-gated-residual gate")?;
        let yo = dense_byte_offset::<half::bf16>(y_l, "fused-gated-residual y")?;
        let h = h.as_cuda_slice::<half::bf16>()?;
        let gate = gate.as_cuda_slice::<half::bf16>()?;
        let y = y.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(total)? };
        {
            let (hp, _a) = h.device_ptr(&stream);
            let (gp, _b) = gate.device_ptr(&stream);
            let (yp, _c) = y.device_ptr(&stream);
            let (op, _d) = out.device_ptr(&stream);
            let (hp, gp, yp) = (hp + ho, gp + go, yp + yo);
            unsafe {
                fused_gated_residual_launch(
                    op as *mut c_void,
                    hp as *const c_void,
                    gp as *const c_void,
                    yp as *const c_void,
                    total,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), h_l.shape().clone()))
    }
}

/// `h + tanh(gate) * y` fused. All three share shape (bf16 in/out).
pub fn fused_gated_residual(h: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    let h = h.contiguous()?;
    let gate = gate.contiguous()?;
    let y = y.contiguous()?;
    Ok(h.apply_op3(&gate, &y, FusedGatedResidual)?)
}

/// Validate the fused RMSNorm×weight kernel vs the candle reference (RMSNorm no
/// zero-center, f32, then `* W`) at row width `n`: cosine similarity. `n` is
/// 4096 (ZeroCenter) or 128 (Head) in the DiT.
pub fn self_test_rmsnorm(n: usize) -> Result<f32> {
    use candle_core::{DType, Device, D};
    let dev = Device::new_cuda(0)?;
    let m = 4117usize; // seq rows
    let eps = 1e-6f32;
    let x = Tensor::randn(0f32, 1f32, (m, n), &dev)?.to_dtype(DType::BF16)?;
    // Effective per-channel weight (f32), as the DiT passes it.
    let w = Tensor::randn(1f32, 0.2f32, n, &dev)?; // f32 (N,)

    let y = fused_rmsnorm_scale(&x, &w, eps)?.to_dtype(DType::F32)?;

    // candle reference: x * (1/sqrt(mean(x^2)+eps)) * w
    let x32 = x.to_dtype(DType::F32)?;
    let ms = x32.sqr()?.mean_keepdim(D::Minus1)?;
    let rrms = (ms + eps as f64)?.sqrt()?.recip()?;
    let yref = x32.broadcast_mul(&rrms)?.broadcast_mul(&w)?;

    let dot = (&y * &yref)?.sum_all()?.to_scalar::<f32>()?;
    let na = y.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = yref.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

/// Validate the fused gated-residual kernel vs candle `h + tanh(gate) * y`.
/// `gate` is drawn with a wide spread so tanh is exercised across its
/// saturating range, not only its near-linear region.
pub fn self_test_gated() -> Result<f32> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (b, s, inner) = (1usize, 4117usize, 4096usize);
    let h = Tensor::randn(0f32, 1f32, (b, s, inner), &dev)?.to_dtype(DType::BF16)?;
    let gate = Tensor::randn(0f32, 1.5f32, (b, s, inner), &dev)?.to_dtype(DType::BF16)?;
    let y = Tensor::randn(0f32, 1f32, (b, s, inner), &dev)?.to_dtype(DType::BF16)?;

    let out = fused_gated_residual(&h, &gate, &y)?.to_dtype(DType::F32)?;

    // candle reference.
    let oref = (&h + gate.tanh()?.broadcast_mul(&y)?)?.to_dtype(DType::F32)?;

    let dot = (&out * &oref)?.sum_all()?.to_scalar::<f32>()?;
    let na = out.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = oref.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

/// Offset-view regression (`qwen-image-rs-b1-off-prompt`). Each fused op runs
/// on a dense view with a NONZERO storage offset — built exactly like
/// `generate`'s B=1 prompt embeds: `(1, drop+n, N).narrow(1, drop, n)`, which
/// candle keeps as a zero-copy view — and on a fresh offset-0 copy of the same
/// values. The outputs must be bit-identical; a bridge that passes the storage
/// base pointer reads rows `0..n` instead of `drop..drop+n`. Returns
/// `(op, bit_identical)` per op.
pub fn self_test_offset_views() -> Result<Vec<(&'static str, bool)>> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (drop, n, inner) = (14usize, 21usize, 4096usize);
    // Draw a (1, drop+n, N) bf16 tensor and return its S-axis narrow (a view).
    let view = |std: f32| -> Result<Tensor> {
        let t = Tensor::randn(0f32, std, (1, drop + n, inner), &dev)?.to_dtype(DType::BF16)?;
        Ok(t.narrow(1, drop, n)?)
    };
    let bits = |t: &Tensor| -> Result<Vec<u16>> {
        Ok(t.flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    let (x, s, g) = (view(1.0)?, view(0.5)?, view(1.5)?);
    anyhow::ensure!(
        x.contiguous()?.layout().start_offset() == drop * inner,
        "test premise: the narrow must stay a zero-copy offset view"
    );
    let (xf, sf, gf) = (
        x.force_contiguous()?,
        s.force_contiguous()?,
        g.force_contiguous()?,
    );
    // An offset f32 weight too: (drop+N,) narrowed to N.
    let w = Tensor::randn(1f32, 0.2f32, drop + inner, &dev)?.narrow(0, drop, inner)?;
    let wf = w.force_contiguous()?;
    // Row-strided 2-D views (the merged q|k case): rows of 128 taken from a
    // (drop + R, 256) matrix, both halves, with a row offset.
    let qk = Tensor::randn(0f32, 1f32, (drop + 3 * n, 256), &dev)?
        .to_dtype(DType::BF16)?
        .narrow(0, drop, 3 * n)?;
    let wh = Tensor::randn(1f32, 0.2f32, 128, &dev)?;
    let mut strided_ok = true;
    for off in [0usize, 128] {
        let v = qk.narrow(1, off, 128)?;
        strided_ok &= v.stride()[0] == 256
            && bits(&fused_rmsnorm_scale(&v, &wh, 1e-6)?)?
                == bits(&fused_rmsnorm_scale(&v.force_contiguous()?, &wh, 1e-6)?)?;
    }
    Ok(vec![
        (
            "fused_rmsnorm_scale",
            bits(&fused_rmsnorm_scale(&x, &w, 1e-6)?)?
                == bits(&fused_rmsnorm_scale(&xf, &wf, 1e-6)?)?,
        ),
        ("fused_rmsnorm_scale (row-strided q/k halves)", strided_ok),
        (
            "fused_norm_mod",
            bits(&fused_norm_mod(&x, &s, 1e-6)?)? == bits(&fused_norm_mod(&xf, &sf, 1e-6)?)?,
        ),
        (
            "fused_gated_residual",
            bits(&fused_gated_residual(&x, &g, &s)?)?
                == bits(&fused_gated_residual(&xf, &gf, &sf)?)?,
        ),
    ])
}

/// Bit-identity of the N = 128 16-lane RMSNorm kernel (`qwen-image-rs-qk-norm-fusion`)
/// vs the CTA-per-row kernel it replaces for the per-head q/k norm: dense rows,
/// both row-strided halves of a head-interleaved `(R, 256)` q|k matrix with a
/// row offset, an odd row count (partial last CTA), and rows of extreme
/// magnitude (tiny -> eps-dominated, huge, all-zero). Returns `(case, mismatches)`.
pub fn self_test_rmsnorm128_bits() -> Result<Vec<(String, usize)>> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let bits = |t: &Tensor| -> Result<Vec<u16>> {
        Ok(t.flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    let cmp = |x: &Tensor, w: &Tensor| -> Result<usize> {
        let a = bits(&rmsnorm_scale_impl(x, w, 1e-6, false)?)?;
        let b = bits(&rmsnorm_scale_impl(x, w, 1e-6, true)?)?;
        Ok(a.iter().zip(&b).filter(|(p, q)| p != q).count() + a.len().abs_diff(b.len()))
    };
    let w = Tensor::randn(1f32, 0.2f32, 128, &dev)?;
    let rows = 4117 * 32 + 5; // a q/k call's rows + a partial last CTA
    let dense = Tensor::randn(0f32, 1f32, (rows, 128), &dev)?.to_dtype(DType::BF16)?;
    let qk = Tensor::randn(0f32, 3f32, (7 + 4117, 256), &dev)?
        .to_dtype(DType::BF16)?
        .narrow(0, 7, 4117)?;
    // Per-row magnitudes spanning 1e-6 .. 1e6 (rsqrt near eps, large sums), plus zero rows.
    let mag = Tensor::arange(0u32, 777, &dev)?
        .to_dtype(DType::F32)?
        .affine(24.0 / 776.0, -12.0)?
        .exp()?
        .reshape((777, 1))?;
    let extreme = Tensor::randn(0f32, 1f32, (777, 128), &dev)?
        .broadcast_mul(&mag)?
        .to_dtype(DType::BF16)?;
    let zeros = Tensor::zeros((33, 128), DType::BF16, &dev)?;
    Ok(vec![
        ("dense M=131749".into(), cmp(&dense, &w)?),
        (
            "row-strided q half (ld 256, row offset)".into(),
            cmp(&qk.narrow(1, 0, 128)?, &w)?,
        ),
        (
            "row-strided k half (ld 256, row offset)".into(),
            cmp(&qk.narrow(1, 128, 128)?, &w)?,
        ),
        ("magnitudes e^-12..e^12".into(), cmp(&extreme, &w)?),
        ("all-zero rows".into(), cmp(&zeros, &w)?),
    ])
}
