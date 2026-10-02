//! VAE decoder fused ops (feature `fusednorm`): bridges to
//! kernels/fusednorm/vae_norm.cu.
//!
//! - [`rmsnorm`]: channel RmsNorm × gamma (+ SiLU) over NCHW bf16, one kernel,
//!   byte-identical to the candle chain in `model::vae` (see the .cu header for
//!   how candle's `fast_sum` association is reproduced).
//! - [`bias_residual`]: `(y + bias) + r` with both bf16 roundings, one pass.
//! - NHWC (channels-last) variants for the NHWC decoder (kernels/fusednorm/
//!   vae_nhwc.cu): [`rmsnorm_nhwc`] (optionally adding the previous conv's
//!   bias first), [`bias_epilogue_nhwc`] (conv bias + none / dense / biased /
//!   DupUp3D residual), [`upsample2x_nhwc`]. Same math, bit for bit.
//!
//! CUDA bf16 only; callers keep the candle path for anything else.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::layout::dense_byte_offset;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn vae_nhwc_rmsnorm_launch(
        out: *mut c_void,
        x: *const c_void,
        gamma: *const c_void,
        bias: *const c_void,
        pixels: i64,
        c: i32,
        silu: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn vae_nhwc_epilogue_launch(
        out: *mut c_void,
        y: *const c_void,
        bias: *const c_void,
        r: *const c_void,
        rbias: *const c_void,
        numel: i64,
        c: i32,
        mode: i32,
        h: i32,
        w: i32,
        in_c: i32,
        repeats: i32,
        ft: i32,
        stream: *mut c_void,
    ) -> i32;
    fn vae_nhwc_upsample2x_launch(
        out: *mut c_void,
        x: *const c_void,
        b: i64,
        h: i32,
        w: i32,
        c: i32,
        stream: *mut c_void,
    ) -> i32;
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

// ------------------------------------------------------------------ NHWC

/// Channel counts [`rmsnorm_nhwc`] takes (`[17, 2048]`: a warp needs N >= 32).
pub fn rmsnorm_nhwc_supports(c: usize) -> bool {
    (17..=2048).contains(&c)
}

/// Device pointer of a dense bf16 view (storage base + the view's start
/// offset), plus the cudarc access guard, which the caller keeps alive until
/// the launch is enqueued (it records the stream access on drop).
fn cuda_ptr<'a>(
    s: &'a CudaStorage,
    l: &Layout,
    stream: &'a std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    what: &str,
) -> candle_core::Result<(u64, impl Sized + 'a)> {
    let off = dense_byte_offset::<half::bf16>(l, what)?;
    let (p, guard) = s.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
    Ok((p + off, guard))
}

struct NhwcRmsNorm {
    silu: bool,
}

impl NhwcRmsNorm {
    fn run(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        g: &CudaStorage,
        g_l: &Layout,
        bias: Option<(&CudaStorage, &Layout)>,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let c = *x_l.shape().dims().last().unwrap_or(&0);
        if x_l.shape().rank() != 4 || g_l.shape().elem_count() != c {
            candle_core::bail!(
                "vae-nhwc-rmsnorm: x {:?} / gamma {:?} (want (B,H,W,C) and C)",
                x_l.shape(),
                g_l.shape()
            );
        }
        if let Some((_, bl)) = bias {
            if bl.shape().elem_count() != c {
                candle_core::bail!("vae-nhwc-rmsnorm: bias {:?} != C = {c}", bl.shape());
            }
        }
        if !rmsnorm_nhwc_supports(c) {
            candle_core::bail!("vae-nhwc-rmsnorm: C = {c} outside [17, 2048]");
        }
        let n = x_l.shape().elem_count();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(n)? };
        let (xp, _gx) = cuda_ptr(x, x_l, &stream, "vae-nhwc-rmsnorm x")?;
        let (gp, _gg) = cuda_ptr(g, g_l, &stream, "vae-nhwc-rmsnorm gamma")?;
        let bg = match bias {
            Some((b, bl)) => Some(cuda_ptr(b, bl, &stream, "vae-nhwc-rmsnorm bias")?),
            None => None,
        };
        let bp = bg.as_ref().map_or(0, |(p, _)| *p);
        let rc = {
            let (op, _o) = out.device_ptr(&stream);
            unsafe {
                vae_nhwc_rmsnorm_launch(
                    op as *mut c_void,
                    xp as *const c_void,
                    gp as *const c_void,
                    bp as *const c_void,
                    (n / c) as i64,
                    c as i32,
                    self.silu as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            candle_core::bail!("vae-nhwc-rmsnorm: launcher rejected C = {c}");
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

impl candle_core::CustomOp2 for NhwcRmsNorm {
    fn name(&self) -> &'static str {
        "vae-nhwc-rmsnorm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("vae-nhwc-rmsnorm is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        g: &CudaStorage,
        g_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(x, x_l, g, g_l, None)
    }
}

impl candle_core::CustomOp3 for NhwcRmsNorm {
    fn name(&self) -> &'static str {
        "vae-nhwc-rmsnorm-bias"
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
        candle_core::bail!("vae-nhwc-rmsnorm is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        g: &CudaStorage,
        g_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(x, x_l, g, g_l, Some((b, b_l)))
    }
}

fn check_bf16(ts: &[&Tensor], what: &str) -> candle_core::Result<()> {
    if ts.iter().any(|t| t.dtype() != DType::BF16) {
        candle_core::bail!("{what}: bf16 only");
    }
    Ok(())
}

/// Channel RmsNorm × gamma (+ SiLU) of a `(B, H, W, C)` bf16 CUDA tensor —
/// with `bias`, of `bf16(x + bias[c])` (the previous conv's bias, which then
/// never round-trips through memory). Bit-identical to the NCHW candle chain.
pub fn rmsnorm_nhwc(
    x: &Tensor,
    gamma: &Tensor,
    bias: Option<&Tensor>,
    silu: bool,
) -> candle_core::Result<Tensor> {
    check_bf16(&[x, gamma], "vae-nhwc-rmsnorm")?;
    let x = x.contiguous()?;
    let g = gamma.contiguous()?;
    match bias {
        Some(b) => {
            check_bf16(&[b], "vae-nhwc-rmsnorm")?;
            x.apply_op3_no_bwd(&g, &b.contiguous()?, &NhwcRmsNorm { silu })
        }
        None => x.apply_op2_no_bwd(&g, &NhwcRmsNorm { silu }),
    }
}

/// What [`bias_epilogue_nhwc`] adds after the conv bias.
pub enum Residual<'a> {
    /// Nothing: `bf16(y + bias)`.
    None,
    /// `+ r`, `r` shaped like `y`.
    Dense(&'a Tensor),
    /// `+ bf16(r + rbias)`: an un-biased shortcut conv output and its bias.
    DenseBiased(&'a Tensor, &'a Tensor),
    /// `+ DupUp3D(x)`: `x` (B, h, w, in_c) is the up block's input; `y` is
    /// (B, 2h, 2w, C); `factor_t` the temporal factor (1 or 2).
    DupUp(&'a Tensor, usize),
}

struct NhwcEpilogue {
    mode: i32,
    // DupUp3D source geometry (mode 3).
    h: usize,
    w: usize,
    in_c: usize,
    repeats: usize,
    ft: usize,
}

impl NhwcEpilogue {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        y: &CudaStorage,
        y_l: &Layout,
        bias: &CudaStorage,
        bias_l: &Layout,
        r: Option<(&CudaStorage, &Layout)>,
        rbias: Option<(&CudaStorage, &Layout)>,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = y.device().clone();
        let dims = y_l.shape().dims();
        let c = *dims.last().unwrap_or(&0);
        if dims.len() != 4 || bias_l.shape().elem_count() != c {
            candle_core::bail!(
                "vae-nhwc-epilogue: y {:?} bias {:?}",
                y_l.shape(),
                bias_l.shape()
            );
        }
        match (self.mode, r, rbias) {
            (1 | 2, Some((_, rl)), _) if rl.shape() != y_l.shape() => candle_core::bail!(
                "vae-nhwc-epilogue: residual {:?} != y {:?}",
                rl.shape(),
                y_l.shape()
            ),
            (2, _, Some((_, rbl))) if rbl.shape().elem_count() != c => {
                candle_core::bail!("vae-nhwc-epilogue: rbias {:?} != C = {c}", rbl.shape())
            }
            (3, Some((_, rl)), _) => {
                let want = [dims[0], self.h, self.w, self.in_c];
                if rl.shape().dims() != want || dims[1] != 2 * self.h || dims[2] != 2 * self.w {
                    candle_core::bail!(
                        "vae-nhwc-epilogue: DupUp source {:?} vs output {:?}",
                        rl.shape(),
                        y_l.shape()
                    );
                }
            }
            _ => {}
        }
        let n = y_l.shape().elem_count();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(n)? };
        let (yp, _gy) = cuda_ptr(y, y_l, &stream, "vae-nhwc-epilogue y")?;
        let (bp, _gb) = cuda_ptr(bias, bias_l, &stream, "vae-nhwc-epilogue bias")?;
        let rg = match r {
            Some((s, l)) => Some(cuda_ptr(s, l, &stream, "vae-nhwc-epilogue r")?),
            None => None,
        };
        let rbg = match rbias {
            Some((s, l)) => Some(cuda_ptr(s, l, &stream, "vae-nhwc-epilogue rbias")?),
            None => None,
        };
        let rp = rg.as_ref().map_or(0, |(p, _)| *p);
        let rbp = rbg.as_ref().map_or(0, |(p, _)| *p);
        let rc = {
            let (op, _o) = out.device_ptr(&stream);
            unsafe {
                vae_nhwc_epilogue_launch(
                    op as *mut c_void,
                    yp as *const c_void,
                    bp as *const c_void,
                    rp as *const c_void,
                    rbp as *const c_void,
                    n as i64,
                    c as i32,
                    self.mode,
                    self.h as i32,
                    self.w as i32,
                    self.in_c as i32,
                    self.repeats as i32,
                    self.ft as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            candle_core::bail!("vae-nhwc-epilogue: launcher rejected {:?}", y_l.shape());
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), y_l.shape().clone()))
    }
}

impl candle_core::CustomOp2 for NhwcEpilogue {
    fn name(&self) -> &'static str {
        "vae-nhwc-bias"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("vae-nhwc-epilogue is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        y: &CudaStorage,
        y_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(y, y_l, b, b_l, None, None)
    }
}

impl candle_core::CustomOp3 for NhwcEpilogue {
    fn name(&self) -> &'static str {
        "vae-nhwc-bias-residual"
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
        candle_core::bail!("vae-nhwc-epilogue is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        y: &CudaStorage,
        y_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        r: &CudaStorage,
        r_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if self.mode == 2 {
            // `b` holds [bias | rbias] (2C): split it as two views.
            let c = b_l.shape().elem_count() / 2;
            let bl = Layout::contiguous_with_offset(c, b_l.start_offset());
            let rbl = Layout::contiguous_with_offset(c, b_l.start_offset() + c);
            if !b_l.is_contiguous() || 2 * c != b_l.shape().elem_count() {
                candle_core::bail!("vae-nhwc-epilogue: [bias|rbias] {:?}", b_l.shape());
            }
            return self.run(y, y_l, b, &bl, Some((r, r_l)), Some((b, &rbl)));
        }
        self.run(y, y_l, b, b_l, Some((r, r_l)), None)
    }
}

/// `bf16(y + bias[c])` plus `residual` on a `(B, H, W, C)` bf16 CUDA tensor,
/// with the same bf16 roundings as the candle broadcast_add + add chain.
/// For [`Residual::DenseBiased`] pass `bias` as `[bias | rbias]` (2C).
pub fn bias_epilogue_nhwc(
    y: &Tensor,
    bias: &Tensor,
    residual: Residual<'_>,
) -> candle_core::Result<Tensor> {
    check_bf16(&[y, bias], "vae-nhwc-epilogue")?;
    let y = y.contiguous()?;
    let b = bias.contiguous()?;
    let op = |mode| NhwcEpilogue {
        mode,
        h: 0,
        w: 0,
        in_c: 0,
        repeats: 0,
        ft: 0,
    };
    match residual {
        Residual::None => y.apply_op2_no_bwd(&b, &op(0)),
        Residual::Dense(r) => {
            check_bf16(&[r], "vae-nhwc-epilogue")?;
            y.apply_op3_no_bwd(&b, &r.contiguous()?, &op(1))
        }
        Residual::DenseBiased(r, rbias) => {
            check_bf16(&[r, rbias], "vae-nhwc-epilogue")?;
            let both = Tensor::cat(&[&b.flatten_all()?, &rbias.flatten_all()?], 0)?;
            y.apply_op3_no_bwd(&both, &r.contiguous()?, &op(2))
        }
        Residual::DupUp(x, ft) => {
            check_bf16(&[x], "vae-nhwc-epilogue")?;
            let (_, h, w, in_c) = x.dims4()?;
            let out_c = y.dim(3)?;
            let factor = ft * 4;
            if ft == 0 || (out_c * factor) % in_c != 0 {
                candle_core::bail!("vae-nhwc-epilogue: DupUp {in_c} -> {out_c} (ft {ft})");
            }
            let e = NhwcEpilogue {
                mode: 3,
                h,
                w,
                in_c,
                repeats: out_c * factor / in_c,
                ft,
            };
            y.apply_op3_no_bwd(&b, &x.contiguous()?, &e)
        }
    }
}

struct NhwcUpsample2x;

impl candle_core::CustomOp1 for NhwcUpsample2x {
    fn name(&self) -> &'static str {
        "vae-nhwc-upsample2x"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("vae-nhwc-upsample2x is CUDA-only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, x_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (b, h, w, c) = x_l.shape().dims4()?;
        let n = b * 4 * h * w * c;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(n)? };
        let (xp, _gx) = cuda_ptr(x, x_l, &stream, "vae-nhwc-upsample2x x")?;
        let rc = {
            let (op, _o) = out.device_ptr(&stream);
            unsafe {
                vae_nhwc_upsample2x_launch(
                    op as *mut c_void,
                    xp as *const c_void,
                    b as i64,
                    h as i32,
                    w as i32,
                    c as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            candle_core::bail!("vae-nhwc-upsample2x: launcher rejected {:?}", x_l.shape());
        }
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev),
            Shape::from((b, 2 * h, 2 * w, c)),
        ))
    }
}

/// Nearest 2x upsample of a `(B, H, W, C)` bf16 CUDA tensor.
pub fn upsample2x_nhwc(x: &Tensor) -> candle_core::Result<Tensor> {
    check_bf16(&[x], "vae-nhwc-upsample2x")?;
    x.contiguous()?.apply_op1_no_bwd(&NhwcUpsample2x)
}

/// NHWC <-> NCHW helpers for the self-test.
fn to_nchw(x: &Tensor) -> candle_core::Result<Tensor> {
    x.permute((0, 3, 1, 2))?.contiguous()
}
fn to_nhwc(x: &Tensor) -> candle_core::Result<Tensor> {
    x.permute((0, 2, 3, 1))?.contiguous()
}

/// The NHWC ops vs the NCHW candle chains on permuted data, bit for bit.
fn self_test_nhwc(dev: &candle_core::Device) -> candle_core::Result<Vec<(String, usize)>> {
    let mut out = Vec::new();
    // (B, C, H, W): VAE channel counts, C > 1024, C = 2048, odd / small C
    // (vector width 1 / 2 / 4 paths), B = 2.
    let shapes = [
        (1, 1152, 8, 8),
        (1, 576, 16, 12),
        (1, 288, 9, 37),
        (2, 144, 16, 16),
        (1, 64, 5, 7),
        (1, 100, 6, 6),
        (1, 1030, 4, 4),
        (1, 2048, 3, 3),
        (1, 33, 4, 5),
        (1, 18, 3, 3),
        (1, 1036, 2, 3),
    ];
    for &(b, c, h, w) in &shapes {
        let mut v = (Tensor::randn(0f32, 1f32, (b, c, h, w), dev)? * 3.0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
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
        let x = Tensor::from_vec(v, (b, c, h, w), dev)?.to_dtype(DType::BF16)?;
        let xn = to_nhwc(&x)?;
        let gamma = (Tensor::randn(0f32, 1f32, c, dev)? + 1.0)?.to_dtype(DType::BF16)?;
        let bias = Tensor::randn(0f32, 1f32, c, dev)?.to_dtype(DType::BF16)?;
        let xb = x.broadcast_add(&bias.reshape((1, c, 1, 1))?)?;
        for silu in [false, true] {
            let ours = to_nchw(&rmsnorm_nhwc(&xn, &gamma, None, silu)?)?;
            let refr = rmsnorm_reference(&x, &gamma, silu)?;
            out.push((
                format!("nhwc rmsnorm ({b},{c},{h},{w}) silu={silu}"),
                mismatches(&ours, &refr)?,
            ));
            let ours = to_nchw(&rmsnorm_nhwc(&xn, &gamma, Some(&bias), silu)?)?;
            let refr = rmsnorm_reference(&xb, &gamma, silu)?;
            out.push((
                format!("nhwc bias+rmsnorm ({b},{c},{h},{w}) silu={silu}"),
                mismatches(&ours, &refr)?,
            ));
        }
        let r = Tensor::randn(0f32, 2f32, (b, c, h, w), dev)?.to_dtype(DType::BF16)?;
        let rbias = Tensor::randn(0f32, 1f32, c, dev)?.to_dtype(DType::BF16)?;
        let b4 = |t: &Tensor| t.reshape((1, c, 1, 1));
        let cases: [(&str, Tensor, Tensor); 3] = [
            (
                "bias",
                to_nchw(&bias_epilogue_nhwc(&xn, &bias, Residual::None)?)?,
                x.broadcast_add(&b4(&bias)?)?,
            ),
            (
                "bias+residual",
                to_nchw(&bias_epilogue_nhwc(
                    &xn,
                    &bias,
                    Residual::Dense(&to_nhwc(&r)?),
                )?)?,
                (x.broadcast_add(&b4(&bias)?)? + &r)?,
            ),
            (
                "bias+biased residual",
                to_nchw(&bias_epilogue_nhwc(
                    &xn,
                    &bias,
                    Residual::DenseBiased(&to_nhwc(&r)?, &rbias),
                )?)?,
                (x.broadcast_add(&b4(&bias)?)? + r.broadcast_add(&b4(&rbias)?)?)?,
            ),
        ];
        for (name, ours, refr) in cases {
            out.push((
                format!("nhwc {name} ({b},{c},{h},{w})"),
                mismatches(&ours, &refr)?,
            ));
        }
        let ours = to_nchw(&upsample2x_nhwc(&xn)?)?;
        let refr = x.upsample_nearest2d(2 * h, 2 * w)?;
        out.push((
            format!("nhwc upsample2x ({b},{c},{h},{w})"),
            mismatches(&ours, &refr)?,
        ));
    }
    // DupUp3D residual: the VAE's (in_c, out_c, factor_t) pairs, shrunk.
    for &(b, in_c, out_c, ft, h, w) in &[
        (1usize, 64usize, 64usize, 2usize, 4usize, 5usize),
        (2, 32, 16, 2, 3, 3),
        (1, 16, 8, 1, 4, 2),
        (1, 1152, 576, 2, 2, 2),
        (1, 288, 144, 1, 3, 2),
    ] {
        let xc = Tensor::randn(0f32, 1f32, (b, in_c, h, w), dev)?.to_dtype(DType::BF16)?;
        let y = Tensor::randn(0f32, 1f32, (b, out_c, 2 * h, 2 * w), dev)?.to_dtype(DType::BF16)?;
        let bias = Tensor::randn(0f32, 1f32, out_c, dev)?.to_dtype(DType::BF16)?;
        let ours = to_nchw(&bias_epilogue_nhwc(
            &to_nhwc(&y)?,
            &bias,
            Residual::DupUp(&to_nhwc(&xc)?, ft),
        )?)?;
        let sc = crate::model::vae::dup_up(&xc, in_c, out_c, ft)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let refr = (y.broadcast_add(&bias.reshape((1, out_c, 1, 1))?)? + sc)?;
        out.push((
            format!("nhwc bias+dupup ({b},{in_c}->{out_c},ft{ft},{h}x{w})"),
            mismatches(&ours, &refr)?,
        ));
    }
    // Offset views: a B=1 narrow of a B=2 NHWC batch is dense at H*W*C.
    let big = Tensor::randn(0f32, 1f32, (2, 8, 8, 144), dev)?.to_dtype(DType::BF16)?;
    let view = big.narrow(0, 1, 1)?;
    let fresh = view.to_dtype(DType::F32)?.to_dtype(DType::BF16)?;
    let g = Tensor::randn(0f32, 1f32, 144, dev)?.to_dtype(DType::BF16)?;
    let off = view.layout().start_offset();
    let bad = (off == 0) as usize;
    let pairs = [
        (
            "rmsnorm",
            rmsnorm_nhwc(&view, &g, Some(&g), true)?,
            rmsnorm_nhwc(&fresh, &g, Some(&g), true)?,
        ),
        (
            "bias+residual",
            bias_epilogue_nhwc(&view, &g, Residual::DenseBiased(&view, &g))?,
            bias_epilogue_nhwc(&fresh, &g, Residual::DenseBiased(&fresh, &g))?,
        ),
        (
            "upsample2x",
            upsample2x_nhwc(&view)?,
            upsample2x_nhwc(&fresh)?,
        ),
    ];
    for (name, a, b) in pairs {
        out.push((
            format!("nhwc {name} offset view (start_offset {off})"),
            mismatches(&a, &b)? + bad,
        ));
    }
    // DupUp source as an offset view.
    let src = Tensor::randn(0f32, 1f32, (2, 4, 4, 64), dev)?.to_dtype(DType::BF16)?;
    let sv = src.narrow(0, 1, 1)?;
    let sf = sv.to_dtype(DType::F32)?.to_dtype(DType::BF16)?;
    let y = Tensor::randn(0f32, 1f32, (1, 8, 8, 64), dev)?.to_dtype(DType::BF16)?;
    let gb = Tensor::randn(0f32, 1f32, 64, dev)?.to_dtype(DType::BF16)?;
    out.push((
        "nhwc dupup source offset view".to_string(),
        mismatches(
            &bias_epilogue_nhwc(&y, &gb, Residual::DupUp(&sv, 2))?,
            &bias_epilogue_nhwc(&y, &gb, Residual::DupUp(&sf, 2))?,
        )?,
    ));
    Ok(out)
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
    out.extend(self_test_nhwc(dev)?);
    Ok(out)
}
