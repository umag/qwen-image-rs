//! `AutoencoderKLQwenImage21` decoder, ported to candle. Decode-only (encode
//! not needed for text-to-image). See docs/PHASES.md Phase 2.
//!
//! The 2.1 VAE is a Wan-derived autoencoder specialized for images: its
//! `QwenImage21CausalConv3d` subclasses `nn.Conv2d` and folds the single frame
//! away (`squeeze(2) -> conv2d -> unsqueeze(2)`), so **every conv here is 2D**
//! and candle's lack of conv3d is irrelevant. At T=1 the temporal cache is
//! never populated and the temporal upsampling collapses to spatial-only, so
//! the whole decode is a 2D convnet.
//!
//! Decode path (matches diffusers `_decode`, `first_chunk=True`, `feat_cache`
//! all-None): unpack packed latent -> `z*std + mean` (per channel) ->
//! `post_quant_conv` (1x1) -> `Decoder3d` -> clamp[-1,1].
//!
//! Activation layout ([`ActLayout`], fixed at load): NCHW, or — with the
//! `cudnn` + `fusednorm` features on CUDA bf16 — NHWC (channels-last) end to
//! end. cuDNN's tensor-core conv engines are NHWC, so on NCHW tensors it
//! wraps every conv in layout transforms; on NHWC it does not, and every conv
//! bias folds into the fused kernel that reads the conv output next (the
//! following RmsNorm, the residual add, the DupUp shortcut add). Both layouts
//! compute the same bf16 math; `decode` takes and returns NCHW either way.

use candle_core::{DType, Tensor, D};
use candle_nn::{conv2d, Conv2dConfig, VarBuilder};

/// Memory order of the decoder's activations (and of its conv filters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActLayout {
    /// (B, C, H, W) activations, OIHW filters.
    Nchw,
    /// (B, H, W, C) activations, KRSC `(O, kh, kw, C)` filters.
    Nhwc,
}

/// The layout policy: NHWC only where its fused kernels and the cuDNN NHWC
/// conv exist (CUDA, bf16, both features built and not switched off).
pub fn layout_for(cuda: bool, dtype: DType, nhwc_available: bool) -> ActLayout {
    if cuda && dtype == DType::BF16 && nhwc_available {
        ActLayout::Nhwc
    } else {
        ActLayout::Nchw
    }
}

/// Whether this build + environment can run the NHWC decoder:
/// features `cudnn` + `fusednorm`, and none of `QIR_CUDNN=0`,
/// `QIR_VAE_FUSED=0`, `QIR_VAE_NHWC=0`.
fn nhwc_available() -> bool {
    #[cfg(all(feature = "cudnn", feature = "fusednorm"))]
    {
        cudnn_enabled()
            && vae_fused_enabled()
            && std::env::var("QIR_VAE_NHWC").map_or(true, |v| v != "0")
    }
    #[cfg(not(all(feature = "cudnn", feature = "fusednorm")))]
    {
        false
    }
}

/// `(B, C, H, W)` of an activation in `lay`.
fn dims_bchw(x: &Tensor, lay: ActLayout) -> candle_core::Result<(usize, usize, usize, usize)> {
    match lay {
        ActLayout::Nchw => x.dims4(),
        ActLayout::Nhwc => {
            let (b, h, w, c) = x.dims4()?;
            Ok((b, c, h, w))
        }
    }
}

fn to_nchw(x: &Tensor) -> candle_core::Result<Tensor> {
    x.permute((0, 3, 1, 2))?.contiguous()
}

fn to_nhwc(x: &Tensor) -> candle_core::Result<Tensor> {
    x.permute((0, 2, 3, 1))?.contiguous()
}

/// Warn (once per process) that an NHWC fused op / cuDNN conv failed and the
/// candle fallback ran: same bits, but slower.
#[cfg_attr(not(any(feature = "cudnn", feature = "fusednorm")), allow(dead_code))]
fn warn_nhwc_fallback(what: &str, e: &dyn std::fmt::Display) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(op = what, error = %e, "NHWC VAE op failed; using the candle fallback");
    });
}

/// What follows a conv's bias add on the NHWC path.
enum Res<'a> {
    None,
    /// `+ r`
    Dense(&'a Tensor),
    /// `+ bf16(r + rbias)` (a shortcut conv output whose bias is pending).
    DenseBiased(&'a Tensor, &'a Tensor),
    /// `+ dup_up(x, in_c, out_c, factor_t)` with `x` NHWC.
    DupUp(&'a Tensor, usize, usize, usize),
}

/// `bf16(y + bias)` then `res`, on NHWC `y` — one fused kernel under
/// `fusednorm`, else (or on a kernel error) the candle chain, same bits.
fn nhwc_epilogue(y: &Tensor, bias: &Tensor, res: Res<'_>) -> candle_core::Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    if y.device().is_cuda() && y.dtype() == DType::BF16 && vae_fused_enabled() {
        use crate::vae_fused::{bias_epilogue_nhwc, Residual};
        let r = match res {
            Res::None => Residual::None,
            Res::Dense(r) => Residual::Dense(r),
            Res::DenseBiased(r, rb) => Residual::DenseBiased(r, rb),
            Res::DupUp(x, _, _, ft) => Residual::DupUp(x, ft),
        };
        match bias_epilogue_nhwc(y, bias, r) {
            Ok(t) => return Ok(t),
            Err(e) => warn_nhwc_fallback("bias epilogue", &e),
        }
    }
    let t = y.broadcast_add(bias)?;
    match res {
        Res::None => Ok(t),
        Res::Dense(r) => t + r,
        Res::DenseBiased(r, rb) => t + r.broadcast_add(rb)?,
        Res::DupUp(x, in_c, out_c, ft) => {
            let sc = dup_up(&to_nchw(x)?, in_c, out_c, ft)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            t + to_nhwc(&sc)?
        }
    }
}

/// Nearest 2x upsample in `lay` (== nearest-exact for an integer scale).
fn upsample2x(x: &Tensor, lay: ActLayout) -> candle_core::Result<Tensor> {
    match lay {
        ActLayout::Nchw => {
            let (_, _, h, w) = x.dims4()?;
            x.upsample_nearest2d(h * 2, w * 2)
        }
        ActLayout::Nhwc => {
            #[cfg(feature = "fusednorm")]
            if x.device().is_cuda() && x.dtype() == DType::BF16 {
                match crate::vae_fused::upsample2x_nhwc(x) {
                    Ok(t) => return Ok(t),
                    Err(e) => warn_nhwc_fallback("upsample2x", &e),
                }
            }
            let (_, h, w, _) = x.dims4()?;
            to_nhwc(&to_nchw(x)?.upsample_nearest2d(h * 2, w * 2)?)
        }
    }
}

/// A VAE conv (stride 1): candle's conv2d (im2col + GEMM), or — with the
/// `cudnn` feature, on CUDA in bf16 — cuDNN with tensor-core math
/// ([`crate::cudnn_conv`]), on NCHW or NHWC activations. `QIR_CUDNN=0`
/// forces the candle path.
struct Conv {
    /// OIHW for [`ActLayout::Nchw`], KRSC for [`ActLayout::Nhwc`].
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    lay: ActLayout,
}

impl Conv {
    fn load(
        in_c: usize,
        out_c: usize,
        k: usize,
        padding: usize,
        lay: ActLayout,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            ..Default::default()
        };
        let inner = conv2d(in_c, out_c, k, cfg, vb)?;
        let weight = match lay {
            ActLayout::Nchw => inner.weight().clone(),
            // Reordered once here; the OIHW copy is dropped with `inner`.
            ActLayout::Nhwc => inner.weight().permute((0, 2, 3, 1))?.contiguous()?,
        };
        Ok(Self {
            weight,
            bias: inner.bias().cloned(),
            padding,
            lay,
        })
    }

    /// The convolution without its bias.
    fn conv_nobias(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self.lay {
            ActLayout::Nchw => {
                #[cfg(feature = "cudnn")]
                if x.device().is_cuda() && x.dtype() == DType::BF16 && cudnn_enabled() {
                    match crate::cudnn_conv::conv2d_bf16(x, &self.weight, self.padding) {
                        Ok(y) => return Ok(y),
                        // cuDNN refused this call (unsupported shape, workspace
                        // OOM): the im2col path still decodes (not bit-identical;
                        // slower).
                        Err(e) => warn_cudnn_fallback(x, &e),
                    }
                }
                // Exactly what candle_nn's Conv2d::forward does before its bias add.
                x.conv2d(&self.weight, self.padding, 1, 1, 1)
            }
            ActLayout::Nhwc => {
                #[cfg(feature = "cudnn")]
                match crate::cudnn_conv::conv2d_bf16_nhwc(x, &self.weight, self.padding) {
                    Ok(y) => return Ok(y),
                    Err(e) => warn_nhwc_fallback("cudnn conv", &e),
                }
                let w = self.weight.permute((0, 3, 1, 2))?.contiguous()?;
                let y = to_nchw(x)?.conv2d(&w, self.padding, 1, 1, 1)?;
                to_nhwc(&y)
            }
        }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.conv_nobias(x)?;
        match (&self.bias, self.lay) {
            (Some(b), ActLayout::Nchw) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?),
            (Some(b), ActLayout::Nhwc) => nhwc_epilogue(&y, b, Res::None),
            (None, _) => Ok(y),
        }
    }

    /// The conv output with its bias still to add: on NHWC the bias is
    /// returned so the next fused kernel adds it; on NCHW it is applied.
    fn forward_pending(&self, x: &Tensor) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        match self.lay {
            ActLayout::Nchw => Ok((self.forward(x)?, None)),
            ActLayout::Nhwc => Ok((self.conv_nobias(x)?, self.bias.clone())),
        }
    }

    /// `forward(x) + (r [+ r_bias])` (a residual skip). Under `fusednorm` on
    /// CUDA bf16 the bias add and the residual add run as one kernel with the
    /// same two (three, with `r_bias`) bf16 roundings (byte-identical).
    fn forward_residual(
        &self,
        x: &Tensor,
        r: &Tensor,
        r_bias: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let r_owned;
        let r = match (r_bias, self.lay) {
            (Some(rb), ActLayout::Nchw) => {
                r_owned = r.broadcast_add(&rb.reshape((1, rb.dim(0)?, 1, 1))?)?;
                &r_owned
            }
            (Some(rb), ActLayout::Nhwc) => {
                let y = self.conv_nobias(x)?;
                return match &self.bias {
                    Some(b) => nhwc_epilogue(&y, b, Res::DenseBiased(r, rb)),
                    None => y + r.broadcast_add(rb)?,
                };
            }
            (None, _) => r,
        };
        match (&self.bias, self.lay) {
            (Some(b), ActLayout::Nhwc) => nhwc_epilogue(&self.conv_nobias(x)?, b, Res::Dense(r)),
            #[cfg(feature = "fusednorm")]
            (Some(b), ActLayout::Nchw)
                if x.device().is_cuda() && x.dtype() == DType::BF16 && vae_fused_enabled() =>
            {
                crate::vae_fused::bias_residual(&self.conv_nobias(x)?, b, r)
            }
            _ => self.forward(x)? + r,
        }
    }
}

/// `QIR_VAE_FUSED=0` turns the fused VAE norm/residual kernels off.
#[cfg(feature = "fusednorm")]
fn vae_fused_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_VAE_FUSED").map_or(true, |v| v != "0"))
}

#[cfg(feature = "cudnn")]
fn warn_cudnn_fallback(x: &Tensor, e: &candle_core::Error) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(shape = ?x.dims(), error = %e, "cuDNN conv failed; falling back to im2col");
    });
}

#[cfg(feature = "cudnn")]
fn cudnn_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_CUDNN").map_or(true, |v| v != "0"))
}

use crate::model::config::VaeConfig;
use crate::Result;

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// `QwenImage21RMS_norm` over the channel dim of an activation.
/// Replicates `F.normalize(x, dim=C) * sqrt(C) * gamma` (RMS norm; eps 1e-12).
struct RmsNorm {
    gamma: Tensor, // (1, C, 1, 1)
    scale: f64,    // sqrt(C)
    lay: ActLayout,
}

impl RmsNorm {
    fn load(c: usize, images: bool, lay: ActLayout, vb: VarBuilder) -> Result<Self> {
        // gamma is stored (C,1,1) for images=True, (C,1,1,1) for images=False.
        let gamma = if images {
            vb.get((c, 1, 1), "gamma")?
        } else {
            vb.get((c, 1, 1, 1), "gamma")?
        };
        let gamma = gamma.reshape((1, c, 1, 1))?;
        Ok(Self {
            gamma,
            scale: (c as f64).sqrt(),
            lay,
        })
    }

    /// `forward(x [+ bias])`, then SiLU if `with_silu` — one fused kernel
    /// under `fusednorm` on CUDA bf16 (byte-identical to the candle chain
    /// below). `bias` is a pending conv bias (NHWC path), added in bf16 first.
    fn forward_silu(&self, x: &Tensor, bias: Option<&Tensor>, with_silu: bool) -> Result<Tensor> {
        if self.lay == ActLayout::Nhwc {
            #[cfg(feature = "fusednorm")]
            if x.device().is_cuda()
                && x.dtype() == DType::BF16
                && self.gamma.dtype() == DType::BF16
                && crate::vae_fused::rmsnorm_nhwc_supports(x.dim(3)?)
            {
                match crate::vae_fused::rmsnorm_nhwc(x, &self.gamma, bias, with_silu) {
                    Ok(t) => return Ok(t),
                    Err(e) => warn_nhwc_fallback("rmsnorm", &e),
                }
            }
            let x = match bias {
                Some(b) => x.broadcast_add(b)?,
                None => x.clone(),
            };
            let y = self.forward_nchw_unfused(&to_nchw(&x)?)?;
            let y = if with_silu { silu(&y)? } else { y };
            return Ok(to_nhwc(&y)?);
        }
        let x_owned;
        let x = match bias {
            Some(b) => {
                x_owned = x.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?)?;
                &x_owned
            }
            None => x,
        };
        #[cfg(feature = "fusednorm")]
        if x.device().is_cuda()
            && x.dtype() == DType::BF16
            && self.gamma.dtype() == DType::BF16
            && crate::vae_fused::rmsnorm_supports(x.dim(1)?)
            && vae_fused_enabled()
        {
            return Ok(crate::vae_fused::rmsnorm(x, &self.gamma, with_silu)?);
        }
        let y = self.forward_nchw_unfused(x)?;
        if with_silu {
            Ok(silu(&y)?)
        } else {
            Ok(y)
        }
    }

    /// The candle chain on an NCHW tensor.
    fn forward_nchw_unfused(&self, x: &Tensor) -> Result<Tensor> {
        // F.normalize: x / max(||x||_2 over C, 1e-12). Reduce in f32 (bf16 sums
        // over the channel dim lose precision / the 1e-12 eps vanishes), then
        // cast back to the working dtype and apply gamma.
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let norm = xf
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        let normalized = (xf.broadcast_div(&norm)? * self.scale)?.to_dtype(dt)?;
        Ok(normalized.broadcast_mul(&self.gamma)?)
    }
}

fn conv2d_k3(in_c: usize, out_c: usize, lay: ActLayout, vb: VarBuilder) -> Result<Conv> {
    Conv::load(in_c, out_c, 3, 1, lay, vb)
}

fn conv2d_k1(in_c: usize, out_c: usize, lay: ActLayout, vb: VarBuilder) -> Result<Conv> {
    Conv::load(in_c, out_c, 1, 0, lay, vb)
}

/// VAE decode tiling policy (`--vae-tile auto|0|N`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeTiling {
    /// Whole image when its estimated working set fits in free VRAM, else
    /// [`AUTO_TILE`]-latent tiles.
    Auto,
    /// Always the whole image.
    Off,
    /// Always `n`×`n` latent tiles (overlap n/4).
    Tile(usize),
}

impl std::str::FromStr for VaeTiling {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim() {
            "auto" => Ok(Self::Auto),
            "0" => Ok(Self::Off),
            n => n
                .parse::<usize>()
                .map(Self::Tile)
                .map_err(|_| format!("--vae-tile: expected auto, 0 or a tile size, got {n:?}")),
        }
    }
}

/// Tile size [`VaeTiling::Auto`] falls back to.
pub const AUTO_TILE: usize = 32;
/// Headroom kept free on top of the decode estimate (allocator slack,
/// fragmentation, the display / other processes).
pub const FIT_MARGIN: usize = 1 << 30;

/// Estimated whole-image decode working set per OUTPUT pixel, in bytes.
/// Measured at 1024² (vae-decode, nvidia-smi peak minus idle): cuDNN conv bf16
/// ~4.5 KiB/px; candle's im2col conv materializes 9x the conv input (~13.5
/// KiB/px); f32 doubles every activation. Rounded up.
pub fn decode_bytes_per_pixel(cudnn_conv: bool, f32_decoder: bool) -> usize {
    let base = if cudnn_conv { 5 << 10 } else { 15 << 10 };
    if f32_decoder {
        2 * base
    } else {
        base
    }
}

/// Does a whole-image decode of `pixels` output pixels fit in `free` bytes?
pub fn untiled_fits(pixels: usize, bytes_per_pixel: usize, free: usize) -> bool {
    pixels
        .saturating_mul(bytes_per_pixel)
        .saturating_add(FIT_MARGIN)
        <= free
}

/// Whether this build's VAE convs run on cuDNN (feature `cudnn`, not disabled).
fn conv_uses_cudnn() -> bool {
    #[cfg(feature = "cudnn")]
    {
        cudnn_enabled()
    }
    #[cfg(not(feature = "cudnn"))]
    {
        false
    }
}

/// `QwenImage21ResidualBlock`: norm1->silu->conv1(k3)->norm2->silu->conv2(k3) + shortcut.
struct ResidualBlock {
    norm1: RmsNorm,
    conv1: Conv,
    norm2: RmsNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl ResidualBlock {
    fn load(in_c: usize, out_c: usize, lay: ActLayout, vb: VarBuilder) -> Result<Self> {
        let shortcut = if in_c != out_c {
            Some(conv2d_k1(in_c, out_c, lay, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self {
            norm1: RmsNorm::load(in_c, false, lay, vb.pp("norm1"))?,
            conv1: conv2d_k3(in_c, out_c, lay, vb.pp("conv1"))?,
            norm2: RmsNorm::load(out_c, false, lay, vb.pp("norm2"))?,
            conv2: conv2d_k3(out_c, out_c, lay, vb.pp("conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // NHWC: the shortcut's and conv1's biases stay pending and are added by
        // conv2's residual epilogue / norm2 (same bf16 roundings).
        let (h, h_bias) = match &self.shortcut {
            Some(sc) => sc.forward_pending(x)?,
            None => (x.clone(), None),
        };
        let x = self.norm1.forward_silu(x, None, true)?;
        let (x, b1) = self.conv1.forward_pending(&x)?;
        let x = self.norm2.forward_silu(&x, b1.as_ref(), true)?;
        Ok(self.conv2.forward_residual(&x, &h, h_bias.as_ref())?)
    }
}

/// `QwenImage21AttentionBlock`: single-head self-attention over HW tokens.
struct AttentionBlock {
    norm: RmsNorm,
    to_qkv: Conv,
    proj: Conv,
    dim: usize,
    lay: ActLayout,
}

impl AttentionBlock {
    fn load(dim: usize, lay: ActLayout, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(dim, true, lay, vb.pp("norm"))?,
            to_qkv: conv2d_k1(dim, dim * 3, lay, vb.pp("to_qkv"))?,
            proj: conv2d_k1(dim, dim, lay, vb.pp("proj"))?,
            dim,
            lay,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x;
        let (b, c, h, w) = dims_bchw(x, self.lay)?;
        let xn = self.norm.forward_silu(x, None, false)?;
        let qkv = self.to_qkv.forward(&xn)?;
        // -> (B, HW, 3C), chunk on the last dim. NHWC already is token-major.
        let qkv = match self.lay {
            ActLayout::Nchw => qkv.reshape((b, 3 * c, h * w))?.transpose(1, 2)?,
            ActLayout::Nhwc => qkv.reshape((b, h * w, 3 * c))?,
        };
        let q = qkv.narrow(2, 0, c)?.contiguous()?;
        let k = qkv.narrow(2, c, c)?.contiguous()?;
        let v = qkv.narrow(2, 2 * c, c)?.contiguous()?;
        // scaled dot-product attention, single head, scale 1/sqrt(C)
        let scale = 1.0 / (self.dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?; // (B, HW, HW)
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // (B, HW, C)
        let out = match self.lay {
            ActLayout::Nchw => out.transpose(1, 2)?.reshape((b, c, h, w))?,
            ActLayout::Nhwc => out.reshape((b, h, w, c))?,
        };
        Ok(self.proj.forward_residual(&out, identity, None)?)
    }
}

/// `QwenImage21MidBlock`: resnet -> attn -> resnet.
struct MidBlock {
    resnet0: ResidualBlock,
    attn: AttentionBlock,
    resnet1: ResidualBlock,
}

impl MidBlock {
    fn load(dim: usize, lay: ActLayout, vb: VarBuilder) -> Result<Self> {
        let resnets = vb.pp("resnets");
        Ok(Self {
            resnet0: ResidualBlock::load(dim, dim, lay, resnets.pp("0"))?,
            attn: AttentionBlock::load(dim, lay, vb.pp("attentions").pp("0"))?,
            resnet1: ResidualBlock::load(dim, dim, lay, resnets.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

/// Nearest-exact 2x upsample (== nearest for integer scale) + conv2d(k3).
/// The learned resample conv of `QwenImage21Resample` (upsample2d/3d at T=1).
struct Upsampler {
    conv: Conv,
}

impl Upsampler {
    fn load(dim: usize, out_dim: usize, lay: ActLayout, vb: VarBuilder) -> Result<Self> {
        // Sequential(Upsample[0], Conv2d[1]) -> weights under "resample.1".
        Ok(Self {
            conv: conv2d_k3(dim, out_dim, lay, vb.pp("resample").pp("1"))?,
        })
    }

    /// `conv(upsample2x(x)) + dup_up(x_in)`, the up block's tail: on NHWC the
    /// conv bias and the DupUp3D shortcut add run as one gather epilogue.
    fn forward_dupup(
        &self,
        x: &Tensor,
        x_in: &Tensor,
        sc: (usize, usize, usize),
    ) -> Result<Tensor> {
        let (in_c, out_c, factor_t) = sc;
        let up = upsample2x(x, self.conv.lay)?;
        match (self.conv.lay, &self.conv.bias) {
            (ActLayout::Nhwc, Some(b)) => Ok(nhwc_epilogue(
                &self.conv.conv_nobias(&up)?,
                b,
                Res::DupUp(x_in, in_c, out_c, factor_t),
            )?),
            (ActLayout::Nhwc, None) => {
                let sc = to_nhwc(&dup_up(&to_nchw(x_in)?, in_c, out_c, factor_t)?)?;
                Ok((self.conv.conv_nobias(&up)? + sc)?)
            }
            (ActLayout::Nchw, _) => {
                Ok((self.conv.forward(&up)? + dup_up(x_in, in_c, out_c, factor_t)?)?)
            }
        }
    }
}

/// `QwenImage21DupUp3D` at T=1, `first_chunk=True`: parameter-free 2x spatial
/// upsample that maps `in_c -> out_c` via channel repeat + pixel rearrange.
/// NCHW in and out. Equivalent gather: output `(o, 2y+i, 2x+j)` =
/// `x[dup_up_src_channel(o, i, j, ..), y, x]`.
pub(crate) fn dup_up(x: &Tensor, in_c: usize, out_c: usize, factor_t: usize) -> Result<Tensor> {
    let factor = factor_t * 4; // fs*fs = 4
    let repeats = out_c * factor / in_c;
    let (b, _, h, w) = x.dims4()?;
    // repeat_interleave over channels by `repeats` -> (B, out*factor, H, W)
    let x = x
        .reshape((b, in_c, 1, h, w))?
        .broadcast_as((b, in_c, repeats, h, w))?
        .contiguous()?
        .reshape((b, in_c * repeats, h, w))?;
    // Channel nesting is [out][factor_t][fs*fs]. Split, pick the kept temporal
    // block (index factor_t-1, == first_chunk), then depth-to-space by 2.
    let x = x.reshape((b, out_c, factor_t, 4, h, w))?; // 6-D
    let x = x.narrow(2, factor_t - 1, 1)?; // (B, out, 1, 4, H, W)
    let x = x.reshape((b, out_c, 2, 2, h, w))?; // fs, fs, H, W
                                                // permute (B,out,fs_i,fs_j,H,W) -> (B,out,H,fs_i,W,fs_j)
    let x = x.permute((0, 1, 4, 2, 5, 3))?.contiguous()?;
    Ok(x.reshape((b, out_c, h * 2, w * 2))?)
}

/// The input channel [`dup_up`] copies into output channel `o` at sub-pixel
/// `(i, j)` (the formula the NHWC DupUp epilogue kernel gathers with).
#[cfg_attr(not(test), allow(dead_code))]
fn dup_up_src_channel(o: usize, i: usize, j: usize, in_c: usize, out_c: usize, ft: usize) -> usize {
    let repeats = out_c * ft * 4 / in_c;
    let k = ((o * ft + (ft - 1)) * 2 + i) * 2 + j;
    k / repeats
}

/// `QwenImage21ResidualUpBlock`: (num_res_blocks+1) resnets, optional learned
/// upsampler with its param-free DupUp residual shortcut.
struct ResidualUpBlock {
    resnets: Vec<ResidualBlock>,
    upsampler: Option<(Upsampler, (usize, usize, usize))>, // shortcut (in_c, out_c, factor_t)
}

impl ResidualUpBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_c: usize,
        out_c: usize,
        num_res_blocks: usize,
        up_flag: bool,
        temporal_upsample: bool,
        lay: ActLayout,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let mut resnets = Vec::new();
        let mut cur = in_c;
        for j in 0..num_res_blocks + 1 {
            resnets.push(ResidualBlock::load(cur, out_c, lay, rvb.pp(j.to_string()))?);
            cur = out_c;
        }
        let upsampler = if up_flag {
            let up = Upsampler::load(out_c, out_c, lay, vb.pp("upsampler"))?;
            let factor_t = if temporal_upsample { 2 } else { 1 };
            Some((up, (in_c, out_c, factor_t)))
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        match &self.upsampler {
            Some((up, sc)) => up.forward_dupup(&h, x, *sc),
            None => Ok(h),
        }
    }
}

/// `QwenImage21Decoder3d`.
struct Decoder3d {
    conv_in: Conv,
    mid_block: MidBlock,
    up_blocks: Vec<ResidualUpBlock>,
    norm_out: RmsNorm,
    conv_out: Conv,
}

impl Decoder3d {
    fn load(cfg: &VaeConfig, lay: ActLayout, vb: VarBuilder) -> Result<Self> {
        let dim = 144usize; // decoder_base_dim
        let z = cfg.latent_channels; // 64
        let out_c = cfg.in_channels; // 4 (RGBA)
        let dim_mult = &cfg.dim_mult; // [1,2,4,8,8]
        let num_res_blocks = 2usize;
        // temperal_upsample = reversed(temperal_downsample=[F,T,T,T]) = [T,T,T,F]
        let temporal_upsample = [true, true, true, false];

        // dims = [dim*u for u in [dim_mult[-1]] + reversed(dim_mult)]
        let mut mult = vec![*dim_mult.last().unwrap()];
        mult.extend(dim_mult.iter().rev().copied());
        let dims: Vec<usize> = mult.iter().map(|u| dim * u).collect();

        let conv_in = conv2d_k3(z, dims[0], lay, vb.pp("conv_in"))?;
        let mid_block = MidBlock::load(dims[0], lay, vb.pp("mid_block"))?;

        let mut up_blocks = Vec::new();
        let upvb = vb.pp("up_blocks");
        for i in 0..dims.len() - 1 {
            let in_c = dims[i];
            let out_c = dims[i + 1];
            let up_flag = i != dim_mult.len() - 1;
            let tu = if up_flag { temporal_upsample[i] } else { false };
            up_blocks.push(ResidualUpBlock::load(
                in_c,
                out_c,
                num_res_blocks,
                up_flag,
                tu,
                lay,
                upvb.pp(i.to_string()),
            )?);
        }
        let last = *dims.last().unwrap();
        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out: RmsNorm::load(last, false, lay, vb.pp("norm_out"))?,
            conv_out: conv2d_k3(last, out_c, lay, vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        x = self.mid_block.forward(&x)?;
        for up in &self.up_blocks {
            x = up.forward(&x)?;
        }
        x = self.norm_out.forward_silu(&x, None, true)?;
        Ok(self.conv_out.forward(&x)?)
    }
}

/// The decode half of `AutoencoderKLQwenImage21`.
pub struct QwenImageVae {
    post_quant_conv: Conv,
    decoder: Decoder3d,
    latents_mean: Tensor, // (1, z, 1, 1) f32
    latents_std: Tensor,  // (1, z, 1, 1) f32
    dtype: DType,         // conv working dtype (bf16 on CUDA, f32 fallback)
    lay: ActLayout,
}

impl QwenImageVae {
    /// Load the decoder + post_quant_conv from a diffusers VarBuilder rooted at
    /// the vae, plus the per-channel latents_mean/std from config.
    pub fn load(
        cfg: &VaeConfig,
        latents_mean: &[f32],
        latents_std: &[f32],
        vb: VarBuilder,
    ) -> Result<Self> {
        let z = cfg.latent_channels;
        let dev = vb.device().clone();
        let mean = Tensor::from_slice(latents_mean, (1, z, 1, 1), &dev)?;
        let std = Tensor::from_slice(latents_std, (1, z, 1, 1), &dev)?;
        let lay = layout_for(dev.is_cuda(), vb.dtype(), nhwc_available());
        tracing::info!(layout = ?lay, dtype = ?vb.dtype(), "vae decoder");
        Ok(Self {
            dtype: vb.dtype(),
            post_quant_conv: conv2d_k1(z, z, lay, vb.pp("post_quant_conv"))?,
            decoder: Decoder3d::load(cfg, lay, vb.pp("decoder"))?,
            latents_mean: mean,
            latents_std: std,
            lay,
        })
    }

    /// The decoder's activation layout.
    pub fn layout(&self) -> ActLayout {
        self.lay
    }

    /// Decode an already-unpacked, normalized latent `(B, z, H, W)` into an RGBA
    /// image `(B, 4, H*16, W*16)` in [-1, 1] (pre postprocess). On the NHWC
    /// path the result is a zero-copy `(B, 4, H, W)` view of NHWC storage.
    pub fn decode(&self, z_normalized: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = z_normalized.dims4()?;
        // Keep this decode's freed memory in the CUDA pool across syncs so the
        // next decode does not re-map it (bounded by the working-set estimate).
        let bpp = decode_bytes_per_pixel(conv_uses_cudnn(), self.dtype == DType::F32);
        crate::device::retain_pool(z_normalized.device(), (b * h * w * 256).saturating_mul(bpp));
        // Unnormalize in f32 (per-channel std/mean; latents_std stores 1/std),
        // then cast to the conv working dtype (bf16 on CUDA → tensor-core convs).
        let z = z_normalized
            .to_dtype(DType::F32)?
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?
            .to_dtype(self.dtype)?;
        let z = match self.lay {
            ActLayout::Nchw => z,
            ActLayout::Nhwc => to_nhwc(&z)?,
        };
        let x = self.post_quant_conv.forward(&z)?;
        let out = self.decoder.forward(&x)?.clamp(-1.0, 1.0)?;
        Ok(match self.lay {
            ActLayout::Nchw => out,
            ActLayout::Nhwc => out.permute((0, 3, 1, 2))?,
        })
    }

    /// Decode with a [`VaeTiling`] policy. `Auto` estimates the whole-image
    /// working set ([`decode_bytes_per_pixel`]) against free VRAM and tiles
    /// only when it would not fit: on WSL an over-subscribed decode does not
    /// fail, it silently spills to shared memory (35 s decodes, blank images).
    /// A whole-image attempt that errors anyway is retried tiled.
    pub fn decode_auto(&self, z_normalized: &Tensor, tiling: VaeTiling) -> Result<Tensor> {
        let tile = match tiling {
            VaeTiling::Off => return self.decode(z_normalized),
            VaeTiling::Tile(n) => return self.decode_tiled(z_normalized, n, (n / 4).max(1)),
            VaeTiling::Auto => AUTO_TILE,
        };
        let (b, _, h, w) = z_normalized.dims4()?;
        let pixels = b * h * w * 256; // 16x per side
        let bpp = decode_bytes_per_pixel(conv_uses_cudnn(), self.dtype == DType::F32);
        let need = pixels.saturating_mul(bpp);
        let fits = match crate::device::free_vram(z_normalized.device()) {
            Some((free, _total)) => {
                // Memory the CUDA pool retained from earlier work is reusable
                // by this decode but not in the driver's free count.
                let slack = crate::device::pool_slack(z_normalized.device());
                let free = free.saturating_add(slack);
                let fits = untiled_fits(pixels, bpp, free);
                tracing::info!(
                    need_mib = need >> 20,
                    free_mib = free >> 20,
                    pool_slack_mib = slack >> 20,
                    mode = if fits { "whole" } else { "tiled" },
                    "vae decode (auto)"
                );
                fits
            }
            None => true, // CPU: host RAM, no co-resident GPU models
        };
        if !fits {
            return self.decode_tiled(z_normalized, tile, (tile / 4).max(1));
        }
        match self.decode(z_normalized) {
            Ok(img) => Ok(img),
            Err(e) => {
                tracing::warn!(error = %e, "whole-image VAE decode failed; retrying tiled");
                self.decode_tiled(z_normalized, tile, (tile / 4).max(1))
            }
        }
    }

    /// Tiled decode: split the latent into overlapping `tile`×`tile` (latent)
    /// windows, decode each, and feather-blend them into the full image. Peak
    /// memory scales with the tile size², not the image size², so the decode
    /// fits alongside co-resident models (fixes the resident-mode VAE thrash).
    /// `overlap` (latent) is blended with a linear feather to hide seams.
    /// Scale factor latent→image is 16.
    pub fn decode_tiled(
        &self,
        z_normalized: &Tensor,
        tile: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let (b, _c, h, w) = z_normalized.dims4()?;
        if tile == 0 || (tile >= h && tile >= w) {
            return self.decode(z_normalized); // one tile — no benefit
        }
        const SCALE: usize = 16;
        let stride = tile.saturating_sub(overlap).max(1);
        let blend_px = (overlap * SCALE).max(1) as f32;
        let dev = z_normalized.device().clone();
        let (ph, pw) = (h * SCALE, w * SCALE);
        let out_c = 4usize; // RGBA
        let mut canvas = Tensor::zeros((b, out_c, ph, pw), DType::F32, &dev)?;
        let mut wsum = Tensor::zeros((1usize, 1usize, ph, pw), DType::F32, &dev)?;
        // Linear feather that plateaus at 1 in the interior and ramps toward the
        // edges; floored above 0 so the outer border (covered by one tile) is
        // recovered exactly after the division.
        let feather = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|k| (((k + 1).min(len - k)) as f32 / blend_px).clamp(1.0 / blend_px, 1.0))
                .collect()
        };
        let mut i = 0;
        loop {
            let th = tile.min(h - i);
            let mut j = 0;
            loop {
                let tw = tile.min(w - j);
                let ztile = z_normalized
                    .narrow(2, i, th)?
                    .narrow(3, j, tw)?
                    .contiguous()?;
                let dec = self.decode(&ztile)?.to_dtype(DType::F32)?; // (b,4,th*S,tw*S)
                let (pi, pj, pth, ptw) = (i * SCALE, j * SCALE, th * SCALE, tw * SCALE);
                let wy = Tensor::from_vec(feather(pth), (1, 1, pth, 1), &dev)?;
                let wx = Tensor::from_vec(feather(ptw), (1, 1, 1, ptw), &dev)?;
                let mask = wy.broadcast_mul(&wx)?; // (1,1,pth,ptw)
                let weighted = dec.broadcast_mul(&mask)?;
                let cregion = canvas.narrow(2, pi, pth)?.narrow(3, pj, ptw)?;
                canvas = canvas.slice_assign(
                    &[0..b, 0..out_c, pi..pi + pth, pj..pj + ptw],
                    &(cregion + weighted)?.contiguous()?,
                )?;
                let wregion = wsum.narrow(2, pi, pth)?.narrow(3, pj, ptw)?;
                wsum = wsum.slice_assign(
                    &[0..1, 0..1, pi..pi + pth, pj..pj + ptw],
                    &(wregion + mask)?.contiguous()?,
                )?;
                if tw >= w - j {
                    break;
                }
                j += stride;
            }
            if th >= h - i {
                break;
            }
            i += stride;
        }
        Ok(canvas.broadcast_div(&wsum)?.clamp(-1.0, 1.0)?)
    }
}

/// Unpack the 2.1 pipeline's token latent `(B, seq, C)` into `(B, C, H, W)`.
/// Mirrors `QwenImage21Pipeline._unpack_latents`: `transpose(1,2).reshape(B,C,H,W)`
/// with `H = W = sqrt(seq)`.
pub fn unpack_latents(packed: &Tensor, _z_dim: usize) -> Result<Tensor> {
    let (b, seq, c) = packed.dims3()?;
    let hw = (seq as f64).sqrt() as usize;
    debug_assert_eq!(hw * hw, seq, "non-square latent");
    let x = packed
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, c, hw, hw))?;
    Ok(x)
}

/// Convert a decoded `(B, 4, H, W)` image in [-1,1] to u8 RGBA bytes (row-major,
/// HWC) for the first batch item.
pub fn to_rgba_u8(img: &Tensor) -> Result<(usize, usize, Vec<u8>)> {
    let (_, c, h, w) = img.dims4()?;
    // (1,C,H,W) -> (H,W,C) first: a view of the NHWC decoder's output is
    // already HWC-dense, so no transpose copy; the ops below are elementwise.
    let hwc = img.i(0)?.permute((1, 2, 0))?;
    let hwc = ((hwc.clamp(-1.0, 1.0)? + 1.0)? * 0.5)?; // [0,1]
    let hwc = hwc.clamp(0.0, 1.0)?.to_dtype(DType::F32)?.contiguous()?;
    let data = (hwc * 255.0)?.round()?.to_dtype(DType::U8)?;
    let bytes = data.flatten_all()?.to_vec1::<u8>()?;
    debug_assert_eq!(c, 4);
    Ok((w, h, bytes))
}

use candle_core::IndexOp;

#[cfg(test)]
mod tests {

    #[test]
    fn vae_tiling_parses() {
        assert_eq!("auto".parse::<VaeTiling>(), Ok(VaeTiling::Auto));
        assert_eq!("0".parse::<VaeTiling>(), Ok(VaeTiling::Off));
        assert_eq!("32".parse::<VaeTiling>(), Ok(VaeTiling::Tile(32)));
        assert!("big".parse::<VaeTiling>().is_err());
    }

    #[test]
    fn untiled_fit_decision() {
        const MIB: usize = 1 << 20;
        let px = 1024 * 1024;
        let cudnn = decode_bytes_per_pixel(true, false);
        // Resident 1024² after the TE + DiT: ~7.4 GiB free -> whole image.
        assert!(untiled_fits(px, cudnn, 7400 * MIB));
        // Exactly at the boundary fits; one byte less does not.
        assert!(untiled_fits(px, cudnn, px * cudnn + FIT_MARGIN));
        assert!(!untiled_fits(px, cudnn, px * cudnn + FIT_MARGIN - 1));
        // 2048² resident needs ~20 GiB -> tiled.
        assert!(!untiled_fits(4 * px, cudnn, 7400 * MIB));
        // Without cuDNN (im2col) or with the f32 decoder 1024² resident tiles.
        assert!(!untiled_fits(
            px,
            decode_bytes_per_pixel(false, false),
            7400 * MIB
        ));
        assert!(!untiled_fits(
            px,
            decode_bytes_per_pixel(true, true),
            7400 * MIB
        ));
        assert!(untiled_fits(
            px,
            decode_bytes_per_pixel(true, true),
            22000 * MIB
        ));
        // Overflow saturates instead of wrapping into "fits".
        assert!(!untiled_fits(usize::MAX, cudnn, usize::MAX - 1));
    }

    use super::*;
    use candle_core::Device;

    #[test]
    fn rmsnorm_matches_reference() -> Result<()> {
        let dev = Device::Cpu;
        // x = [[3],[4]] over C=2: ||.||=5, normalized=[0.6,0.8], *sqrt(2).
        let x = Tensor::from_slice(&[3f32, 4f32], (1, 2, 1, 1), &dev)?;
        let n = RmsNorm {
            gamma: Tensor::ones((1, 2, 1, 1), DType::F32, &dev)?,
            scale: 2f64.sqrt(),
            lay: ActLayout::Nchw,
        };
        let out = n
            .forward_nchw_unfused(&x)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let s = 2f32.sqrt();
        assert!((out[0] - 0.6 * s).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 0.8 * s).abs() < 1e-5, "{out:?}");
        // The NHWC layout (candle fallback on CPU) gives the same bits.
        let nh = RmsNorm {
            gamma: Tensor::ones((1, 2, 1, 1), DType::F32, &dev)?,
            scale: 2f64.sqrt(),
            lay: ActLayout::Nhwc,
        };
        let xn = x.permute((0, 2, 3, 1))?.contiguous()?;
        let o2 = nh
            .forward_silu(&xn, None, false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(out, o2);
        Ok(())
    }

    #[test]
    fn unpack_latents_shape() -> Result<()> {
        let dev = Device::Cpu;
        // (B=1, seq=16, C=2) -> (1, 2, 4, 4)
        let packed = Tensor::zeros((1, 16, 2), DType::F32, &dev)?;
        let z = unpack_latents(&packed, 2)?;
        assert_eq!(z.dims(), &[1, 2, 4, 4]);
        Ok(())
    }

    /// The NHWC DupUp epilogue's gather formula == dup_up() (all VAE shapes'
    /// (in_c, out_c, factor_t) ratios, shrunk).
    #[test]
    fn dup_up_gather_formula_matches() -> Result<()> {
        let dev = Device::Cpu;
        for &(in_c, out_c, ft) in &[
            (16usize, 16usize, 2usize),
            (16, 8, 2),
            (8, 4, 1),
            (12, 6, 2),
        ] {
            let (h, w) = (3usize, 2usize);
            let x = Tensor::randn(0f32, 1f32, (1, in_c, h, w), &dev)?;
            let y = dup_up(&x, in_c, out_c, ft)?;
            let xv = x.flatten_all()?.to_vec1::<f32>()?;
            let yv = y.flatten_all()?.to_vec1::<f32>()?;
            for o in 0..out_c {
                for yy in 0..2 * h {
                    for xx in 0..2 * w {
                        let src = dup_up_src_channel(o, yy % 2, xx % 2, in_c, out_c, ft);
                        let want = xv[(src * h + yy / 2) * w + xx / 2];
                        let got = yv[(o * 2 * h + yy) * 2 * w + xx];
                        assert_eq!(got.to_bits(), want.to_bits(), "o{o} y{yy} x{xx}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn layout_policy() {
        assert_eq!(layout_for(true, DType::BF16, true), ActLayout::Nhwc);
        assert_eq!(layout_for(true, DType::BF16, false), ActLayout::Nchw);
        assert_eq!(layout_for(true, DType::F32, true), ActLayout::Nchw);
        assert_eq!(layout_for(false, DType::BF16, true), ActLayout::Nchw);
    }

    /// candle's fast_sum association for one pixel: N = next_pow2(min(1024,
    /// C)) threads, thread t holds 0 + v[t] (+ v[t+1024]), shared-memory tree
    /// `s[t] += s[t + s]` for s = N/2 .. 1.
    fn candle_order_sum(v: &[f32]) -> f32 {
        let c = v.len();
        let n = c.min(1024).next_power_of_two();
        let mut s: Vec<f32> = (0..n)
            .map(|t| {
                let mut a = 0f32;
                if t < c {
                    a += v[t];
                }
                if t + 1024 < c {
                    a += v[t + 1024];
                }
                a
            })
            .collect();
        let mut step = n / 2;
        while step >= 1 {
            for t in 0..step {
                s[t] += s[t + step];
            }
            step /= 2;
        }
        s[0]
    }

    /// The NHWC norm kernel's association (kernels/fusednorm/vae_nhwc.cu):
    /// lane L, element e hold residue rho = vec*L + e (mod G = 32*vec); each
    /// chunk sums its slots q = brev(j) pairwise; then lane bits 4..0
    /// (butterfly), then e bits high to low.
    fn nhwc_kernel_order_sum(v: &[f32], vec: usize) -> f32 {
        let c = v.len();
        let n = c.min(1024).next_power_of_two();
        let g = 32 * vec;
        assert!(n >= g);
        let m = n / g;
        let lm = m.trailing_zeros();
        let brev = |x: usize, bits: u32| -> usize {
            if bits == 0 {
                0
            } else {
                x.reverse_bits() >> (usize::BITS - bits)
            }
        };
        fn tree(xs: &[f32]) -> f32 {
            if xs.len() == 1 {
                xs[0]
            } else {
                let (a, b) = xs.split_at(xs.len() / 2);
                tree(a) + tree(b)
            }
        }
        // part[lane][e]
        let mut part = vec![vec![0f32; vec]; 32];
        for (lane, pl) in part.iter_mut().enumerate() {
            for (e, p) in pl.iter_mut().enumerate() {
                let leaves: Vec<f32> = (0..m)
                    .map(|j| {
                        let t = brev(j, lm) * g + vec * lane + e;
                        let mut a = 0f32;
                        if t < c {
                            a += v[t];
                        }
                        if n == 1024 && t + 1024 < c {
                            a += v[t + 1024];
                        }
                        a
                    })
                    .collect();
                *p = tree(&leaves);
            }
        }
        let mut mask = 16;
        while mask >= 1 {
            let prev = part.clone();
            for lane in 0..32 {
                for e in 0..vec {
                    part[lane][e] = prev[lane][e] + prev[lane ^ mask][e];
                }
            }
            mask /= 2;
        }
        let mut p = part[0].clone();
        let mut w = vec / 2;
        while w >= 1 {
            for e in 0..w {
                p[e] += p[e + w];
            }
            w /= 2;
        }
        p[0]
    }

    /// The NHWC norm kernel's channel-sum association == candle's, bit for
    /// bit, for every vector width it can pick.
    #[test]
    fn nhwc_norm_sum_order_matches_candle() -> Result<()> {
        let dev = Device::Cpu;
        for &c in &[144usize, 288, 576, 1152, 2048, 1030, 100, 33, 64, 18] {
            let v = (Tensor::randn(0f32, 1f32, c, &dev)?.sqr()? * 7.3)?.to_vec1::<f32>()?;
            let want = candle_order_sum(&v);
            let n = c.min(1024).next_power_of_two();
            for vec in [1usize, 2, 4, 8] {
                if c % vec != 0 || 32 * vec > n {
                    continue;
                }
                let got = nhwc_kernel_order_sum(&v, vec);
                assert_eq!(got.to_bits(), want.to_bits(), "C={c} vec={vec}");
            }
        }
        Ok(())
    }

    #[test]
    fn dup_up_doubles_spatial() -> Result<()> {
        let dev = Device::Cpu;
        // in=4, out=8, factor_t=1 -> repeats=8, output (1,8,4,4) from (1,4,2,2).
        let x = Tensor::zeros((1, 4, 2, 2), DType::F32, &dev)?;
        let y = dup_up(&x, 4, 8, 1)?;
        assert_eq!(y.dims(), &[1, 8, 4, 4]);
        Ok(())
    }
}
