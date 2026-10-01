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
    #[allow(clippy::too_many_arguments)]
    fn fused_residual_norm_mod_launch(
        x_out: *mut c_void,
        h_out: *mut c_void,
        h: *const c_void,
        gate: *const c_void,
        gld: std::ffi::c_long,
        y: *const c_void,
        scale: *const c_void,
        sld: std::ffi::c_long,
        m: i32,
        n: i32,
        seq: i32,
        txt_len: i32,
        eps: f32,
        scalar: i32,
        stream: *mut c_void,
    ) -> i32;
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
        (
            "fused_residual_norm_mod (h, y, gate/scale rows offset)",
            residual_norm_mod_offset_ok(&x, &s, drop, &dev)?,
        ),
    ])
}

fn mr(rows: &Tensor, txt_len: usize) -> ModRows<'_> {
    ModRows { rows, txt_len }
}

/// [`self_test_offset_views`] case for [`fused_residual_norm_mod`]: `h`, `y`
/// are S-axis offset views; the gate / scale rows are row- and column-offset
/// views of a wider modulation matrix (row stride 3N). Both outputs, with and
/// without a residual, must match the fresh-copy run bit for bit.
fn residual_norm_mod_offset_ok(
    h: &Tensor,
    y: &Tensor,
    drop: usize,
    dev: &candle_core::Device,
) -> Result<bool> {
    use candle_core::DType;
    let n = h.dim(2)?;
    let wide = Tensor::randn(0f32, 1f32, (drop + 2, 3 * n), dev)?.to_dtype(DType::BF16)?;
    let (g, sc) = (
        wide.narrow(0, drop, 2)?.narrow(1, n, n)?,
        wide.narrow(0, drop, 2)?.narrow(1, 2 * n, n)?,
    );
    let (gf, sf) = (g.force_contiguous()?, sc.force_contiguous()?);
    let txt = 5;
    let both = |o: (Tensor, Tensor)| -> Result<Vec<u16>> {
        let cat = Tensor::cat(&[o.0, o.1], 0)?;
        Ok(cat
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    let (hf, yf) = (h.force_contiguous()?, y.force_contiguous()?);
    let res = both(fused_residual_norm_mod(
        h,
        Some((mr(&g, txt), y)),
        mr(&sc, txt),
        1e-6,
    )?)? == both(fused_residual_norm_mod(
        &hf,
        Some((mr(&gf, txt), &yf)),
        mr(&sf, txt),
        1e-6,
    )?)?;
    let plain = both(fused_residual_norm_mod(h, None, mr(&sc, txt), 1e-6)?)?
        == both(fused_residual_norm_mod(&hf, None, mr(&sf, txt), 1e-6)?)?;
    Ok(res && plain)
}

/// Bit-identity of [`fused_residual_norm_mod`] (`qwen-image-rs-residual-norm-fusion`)
/// vs what it replaces: [`fused_gated_residual`] then [`fused_norm_mod`] on the
/// materialized per-token modulation tensors. Cases cover B = 1 and 2, the
/// real DiT shape, text splits 0 / S, the vector and the scalar staging paths
/// (forced, and natural via a row width with n % 8 != 0), and the no-residual
/// variant. Gate rows have a wide spread (tanh saturation), h large values
/// (bf16 rounding of h' matters). Returns `(case, mismatched elements)`.
pub fn self_test_residual_norm_mod_bits() -> Result<Vec<(String, usize)>> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let eps = 1e-6f32;
    let bits = |t: &Tensor| -> Result<Vec<u16>> {
        Ok(t.flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    let diff = |a: &Tensor, b: &Tensor| -> Result<usize> {
        let (a, b) = (bits(a)?, bits(b)?);
        Ok(a.iter().zip(&b).filter(|(p, q)| p != q).count() + a.len().abs_diff(b.len()))
    };
    // (B, S, N) per-token tensor from (2, N) rows: text tokens row 1, image row 0.
    let tokens = |rows: &Tensor, b: usize, s: usize, txt: usize| -> Result<Tensor> {
        let n = rows.dim(1)?;
        let mut parts = Vec::new();
        if txt > 0 {
            parts.push(rows.narrow(0, 1, 1)?.broadcast_as((txt, n))?.contiguous()?);
        }
        if s > txt {
            parts.push(
                rows.narrow(0, 0, 1)?
                    .broadcast_as((s - txt, n))?
                    .contiguous()?,
            );
        }
        Ok(Tensor::cat(&parts, 0)?
            .unsqueeze(0)?
            .broadcast_as((b, s, n))?
            .contiguous()?)
    };
    let mut out = Vec::new();
    // (B, S, txt_len, N, forced scalar)
    let cases = [
        (1usize, 4117usize, 21usize, 4096usize, false),
        (1, 4117, 21, 4096, true),
        (2, 533, 21, 4096, false),
        (2, 533, 0, 4096, false),
        (1, 77, 77, 4096, false),
        (2, 61, 9, 1004, false),
    ];
    for (b, s, txt, n, scalar) in cases {
        // Modulation like the DiT: a (2, 4N) matrix, gate / scale its column chunks.
        let gsrc = Tensor::randn(0f32, 1.5f32, (2, 4 * n), &dev)?.to_dtype(DType::BF16)?;
        let ssrc = Tensor::randn(0f32, 0.5f32, (2, 4 * n), &dev)?.to_dtype(DType::BF16)?;
        let (g, sc) = (gsrc.narrow(1, n, n)?, ssrc.narrow(1, 2 * n, n)?);
        let h = Tensor::randn(0f32, 8f32, (b, s, n), &dev)?.to_dtype(DType::BF16)?;
        let y = Tensor::randn(0f32, 2f32, (b, s, n), &dev)?.to_dtype(DType::BF16)?;
        let (gt, st) = (tokens(&g, b, s, txt)?, tokens(&sc, b, s, txt)?);
        let tag = format!(
            "B={b} S={s} txt={txt} N={n}{}",
            if scalar { " scalar" } else { "" }
        );

        let h_ref = fused_gated_residual(&h, &gt, &y)?;
        let x_ref = fused_norm_mod(&h_ref, &st, eps)?;
        let (h2, x2) =
            residual_norm_mod_impl(&h, Some((mr(&g, txt), &y)), mr(&sc, txt), eps, scalar)?;
        out.push((format!("{tag} residual: h'"), diff(&h2, &h_ref)?));
        out.push((format!("{tag} residual: x"), diff(&x2, &x_ref)?));

        let x_ref = fused_norm_mod(&h, &st, eps)?;
        let (_, x2) = residual_norm_mod_impl(&h, None, mr(&sc, txt), eps, scalar)?;
        out.push((format!("{tag} plain: x"), diff(&x2, &x_ref)?));
    }
    Ok(out)
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

/// The DiT's AdaLN parameters for one modulation chunk (`scale` or `gate`):
/// a `(2, N)` bf16 row view — row 0 for image tokens (real timestep), row 1
/// for the `txt_len` text tokens (t = 0) that open every sequence. Rows may sit
/// any `ld >= N` apart (a column narrow of the `(2, 4N)` modulation output).
/// Every batch lane shares `txt_len` (one prompt broadcast to B).
#[derive(Clone, Copy)]
pub struct ModRows<'a> {
    pub rows: &'a Tensor,
    pub txt_len: usize,
}

/// `(h', LayerNorm(h') * (scale + 1))` with `h' = h + tanh(gate) * y` when a
/// residual `(gate, y)` is given, else `(h, LayerNorm(h) * (scale + 1))` — one
/// CTA-per-row kernel (`qwen-image-rs-residual-norm-fusion`). `h`, `y` are
/// `(B, S, N)` bf16; `gate`/`scale` are [`ModRows`] selected per token
/// position. Bit-identical to [`fused_gated_residual`] then [`fused_norm_mod`]
/// on the materialized per-token tensors. Both outputs are fresh dense tensors
/// (`h'` is `h` itself without a residual).
pub fn fused_residual_norm_mod(
    h: &Tensor,
    residual: Option<(ModRows, &Tensor)>,
    scale: ModRows,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    residual_norm_mod_impl(h, residual, scale, eps, false)
}

fn residual_norm_mod_impl(
    h: &Tensor,
    residual: Option<(ModRows, &Tensor)>,
    scale: ModRows,
    eps: f32,
    scalar: bool,
) -> Result<(Tensor, Tensor)> {
    use candle_core::cuda_backend::cudarc::driver::CudaSlice;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Storage};
    const WHAT: &str = "fused-residual-norm-mod";
    let (b, seq, n) = h.dims3()?;
    let m = b * seq;
    let check_rows = |r: &ModRows, what: &str| -> Result<()> {
        anyhow::ensure!(
            r.rows.dtype() == DType::BF16 && r.rows.dims() == [2, n],
            "{WHAT}: {what} rows must be bf16 (2, {n}), got {:?} {:?}",
            r.rows.dtype(),
            r.rows.dims()
        );
        anyhow::ensure!(
            r.txt_len <= seq,
            "{WHAT}: {what} txt_len {} > seq {seq}",
            r.txt_len
        );
        Ok(())
    };
    anyhow::ensure!(
        h.dtype() == DType::BF16,
        "{WHAT}: h must be bf16, got {:?}",
        h.dtype()
    );
    check_rows(&scale, "scale")?;
    if let Some((g, y)) = &residual {
        check_rows(g, "gate")?;
        anyhow::ensure!(
            g.txt_len == scale.txt_len,
            "{WHAT}: gate txt_len {} != scale txt_len {}",
            g.txt_len,
            scale.txt_len
        );
        anyhow::ensure!(
            y.dtype() == DType::BF16 && y.dims() == h.dims(),
            "{WHAT}: y must be bf16 {:?}, got {:?} {:?}",
            h.dims(),
            y.dtype(),
            y.dims()
        );
    }
    // The row is staged in shared memory as f32 (48 KiB without an opt-in).
    anyhow::ensure!(
        n * 4 <= 48 * 1024,
        "{WHAT}: row width {n} exceeds the smem stage"
    );
    let i32_of = |v: usize, what: &str| -> Result<i32> {
        i32::try_from(v).map_err(|_| anyhow::anyhow!("{WHAT}: {what} {v} overflows i32"))
    };
    let (m32, n32, s32, t32) = (
        i32_of(m, "rows")?,
        i32_of(n, "width")?,
        i32_of(seq, "seq")?,
        i32_of(scale.txt_len, "txt_len")?,
    );

    let h = h.contiguous()?;
    let y = residual.map(|(_, y)| y.contiguous()).transpose()?;
    let bf = std::mem::size_of::<half::bf16>();
    let (hs, hl) = h.storage_and_layout();
    let (ss, sl) = scale.rows.storage_and_layout();
    let ho = dense_byte_offset::<half::bf16>(hl, "fused-residual-norm-mod h")?;
    let (so, sld) = row_strided_2d(sl, "fused-residual-norm-mod scale rows")?;
    let hc = match &*hs {
        Storage::Cuda(c) => c,
        _ => anyhow::bail!("{WHAT}: h must be on CUDA"),
    };
    let sc = match &*ss {
        Storage::Cuda(c) => c,
        _ => anyhow::bail!("{WHAT}: scale rows must be on CUDA"),
    };
    let dev = hc.device().clone();
    let stream = dev.cuda_stream();
    let x_out: CudaSlice<half::bf16> = unsafe { dev.alloc::<half::bf16>(m * n)? };
    let h_out: Option<CudaSlice<half::bf16>> = match residual {
        Some(_) => Some(unsafe { dev.alloc::<half::bf16>(m * n)? }),
        None => None,
    };
    let rc = {
        let (hp, _g0) = hc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let (sp, _g1) = sc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let (xp, _g2) = x_out.device_ptr(&stream);
        // Residual operands; their read guards live until the launch is enqueued.
        let gy = match (&residual, &y) {
            (Some((g, _)), Some(y)) => Some((g.rows.storage_and_layout(), y.storage_and_layout())),
            _ => None,
        };
        let mut _guards = Vec::with_capacity(3);
        let (op, gp, gld, yp) = match (&gy, &h_out) {
            (Some(((gs, gl), (ys, yl))), Some(ho_)) => {
                let (go, gld) = row_strided_2d(gl, "fused-residual-norm-mod gate rows")?;
                let yo = dense_byte_offset::<half::bf16>(yl, "fused-residual-norm-mod y")?;
                let (gc, yc) = match (&**gs, &**ys) {
                    (Storage::Cuda(g), Storage::Cuda(y)) => (g, y),
                    _ => anyhow::bail!("{WHAT}: gate rows and y must be on CUDA"),
                };
                let (p_g, g_g) = gc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
                let (p_y, g_y) = yc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
                let (p_o, g_o) = ho_.device_ptr(&stream);
                _guards.extend([g_g, g_y, g_o]);
                (
                    p_o as *mut c_void,
                    (p_g + (go * bf) as u64) as *const c_void,
                    gld,
                    (p_y + yo) as *const c_void,
                )
            }
            _ => (std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null()),
        };
        unsafe {
            fused_residual_norm_mod_launch(
                xp as *mut c_void,
                op,
                (hp + ho) as *const c_void,
                gp,
                gld as std::ffi::c_long,
                yp,
                (sp + (so * bf) as u64) as *const c_void,
                sld as std::ffi::c_long,
                m32,
                n32,
                s32,
                t32,
                eps,
                i32::from(scalar),
                stream.cu_stream() as *mut c_void,
            )
        }
    };
    anyhow::ensure!(rc == 0, "{WHAT}: CUDA launch failed (cudaError {rc})");
    drop((hs, ss));
    let wrap = |o: CudaSlice<half::bf16>| -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(o, dev.clone())),
            (b, seq, n),
            BackpropOp::none(),
            false,
        )
    };
    let x = wrap(x_out);
    let h_new = match h_out {
        Some(o) => wrap(o),
        None => h,
    };
    Ok((h_new, x))
}
