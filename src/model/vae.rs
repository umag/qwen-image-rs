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

use candle_core::{DType, Tensor, D};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

/// A VAE conv (stride 1): candle's `Conv2d` (im2col + GEMM), or — with the
/// `cudnn` feature, on CUDA in bf16 — cuDNN with tensor-core math
/// ([`crate::cudnn_conv`]). `QIR_CUDNN=0` forces the candle path.
struct Conv {
    inner: Conv2d,
    #[cfg_attr(not(feature = "cudnn"), allow(dead_code))]
    padding: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cudnn")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 && cudnn_enabled() {
            match crate::cudnn_conv::conv2d_bf16(x, self.inner.weight(), self.padding) {
                Ok(y) => {
                    return match self.inner.bias() {
                        Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?),
                        None => Ok(y),
                    }
                }
                // cuDNN refused this call (unsupported shape, workspace OOM):
                // the im2col path still decodes (not bit-identical; slower).
                Err(e) => warn_cudnn_fallback(x, &e),
            }
        }
        self.inner.forward(x)
    }
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

/// `QwenImage21RMS_norm` over the channel dim of a `(B, C, H, W)` tensor.
/// Replicates `F.normalize(x, dim=1) * sqrt(C) * gamma` (RMS norm; eps 1e-12).
struct RmsNorm {
    gamma: Tensor, // (1, C, 1, 1)
    scale: f64,    // sqrt(C)
}

impl RmsNorm {
    fn load(c: usize, images: bool, vb: VarBuilder) -> Result<Self> {
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
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

fn conv2d_k3(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv> {
    let cfg = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    Ok(Conv {
        inner: conv2d(in_c, out_c, 3, cfg, vb)?,
        padding: 1,
    })
}

fn conv2d_k1(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv> {
    Ok(Conv {
        inner: conv2d(in_c, out_c, 1, Conv2dConfig::default(), vb)?,
        padding: 0,
    })
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
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let shortcut = if in_c != out_c {
            Some(conv2d_k1(in_c, out_c, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self {
            norm1: RmsNorm::load(in_c, false, vb.pp("norm1"))?,
            conv1: conv2d_k3(in_c, out_c, vb.pp("conv1"))?,
            norm2: RmsNorm::load(out_c, false, vb.pp("norm2"))?,
            conv2: conv2d_k3(out_c, out_c, vb.pp("conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        let x = self.norm1.forward(x)?;
        let x = silu(&x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x)?;
        let x = silu(&x)?;
        let x = self.conv2.forward(&x)?;
        Ok((x + h)?)
    }
}

/// `QwenImage21AttentionBlock`: single-head self-attention over HW tokens.
struct AttentionBlock {
    norm: RmsNorm,
    to_qkv: Conv,
    proj: Conv,
    dim: usize,
}

impl AttentionBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(dim, true, vb.pp("norm"))?,
            to_qkv: conv2d_k1(dim, dim * 3, vb.pp("to_qkv"))?,
            proj: conv2d_k1(dim, dim, vb.pp("proj"))?,
            dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x;
        let (b, c, h, w) = x.dims4()?;
        let xn = self.norm.forward(x)?;
        let qkv = self.to_qkv.forward(&xn)?; // (B, 3C, H, W)
                                             // (B, 3C, HW) -> (B, HW, 3C) -> chunk on last
        let qkv = qkv.reshape((b, 3 * c, h * w))?.transpose(1, 2)?; // (B, HW, 3C)
        let q = qkv.narrow(2, 0, c)?.contiguous()?;
        let k = qkv.narrow(2, c, c)?.contiguous()?;
        let v = qkv.narrow(2, 2 * c, c)?.contiguous()?;
        // scaled dot-product attention, single head, scale 1/sqrt(C)
        let scale = 1.0 / (self.dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?; // (B, HW, HW)
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // (B, HW, C)
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?; // (B, C, H, W)
        let out = self.proj.forward(&out)?;
        Ok((out + identity)?)
    }
}

/// `QwenImage21MidBlock`: resnet -> attn -> resnet.
struct MidBlock {
    resnet0: ResidualBlock,
    attn: AttentionBlock,
    resnet1: ResidualBlock,
}

impl MidBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let resnets = vb.pp("resnets");
        Ok(Self {
            resnet0: ResidualBlock::load(dim, dim, resnets.pp("0"))?,
            attn: AttentionBlock::load(dim, vb.pp("attentions").pp("0"))?,
            resnet1: ResidualBlock::load(dim, dim, resnets.pp("1"))?,
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
    fn load(dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Sequential(Upsample[0], Conv2d[1]) -> weights under "resample.1".
        Ok(Self {
            conv: conv2d_k3(dim, out_dim, vb.pp("resample").pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        Ok(self.conv.forward(&up)?)
    }
}

/// `QwenImage21DupUp3D` at T=1, `first_chunk=True`: parameter-free 2x spatial
/// upsample that maps `in_c -> out_c` via channel repeat + pixel rearrange.
fn dup_up(x: &Tensor, in_c: usize, out_c: usize, factor_t: usize) -> Result<Tensor> {
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

/// `QwenImage21ResidualUpBlock`: (num_res_blocks+1) resnets, optional learned
/// upsampler, optional param-free DupUp residual shortcut.
struct ResidualUpBlock {
    resnets: Vec<ResidualBlock>,
    upsampler: Option<Upsampler>,
    shortcut: Option<(usize, usize, usize)>, // (in_c, out_c, factor_t)
}

impl ResidualUpBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_c: usize,
        out_c: usize,
        num_res_blocks: usize,
        up_flag: bool,
        temporal_upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let mut resnets = Vec::new();
        let mut cur = in_c;
        for j in 0..num_res_blocks + 1 {
            resnets.push(ResidualBlock::load(cur, out_c, rvb.pp(j.to_string()))?);
            cur = out_c;
        }
        let (upsampler, shortcut) = if up_flag {
            let up = Upsampler::load(out_c, out_c, vb.pp("upsampler"))?;
            let factor_t = if temporal_upsample { 2 } else { 1 };
            (Some(up), Some((in_c, out_c, factor_t)))
        } else {
            (None, None)
        };
        Ok(Self {
            resnets,
            upsampler,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_copy = x.clone();
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(up) = &self.upsampler {
            x = up.forward(&x)?;
        }
        if let Some((in_c, out_c, factor_t)) = self.shortcut {
            let sc = dup_up(&x_copy, in_c, out_c, factor_t)?;
            x = (x + sc)?;
        }
        Ok(x)
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
    fn load(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
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

        let conv_in = conv2d_k3(z, dims[0], vb.pp("conv_in"))?;
        let mid_block = MidBlock::load(dims[0], vb.pp("mid_block"))?;

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
                upvb.pp(i.to_string()),
            )?);
        }
        let last = *dims.last().unwrap();
        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out: RmsNorm::load(last, false, vb.pp("norm_out"))?,
            conv_out: conv2d_k3(last, out_c, vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        x = self.mid_block.forward(&x)?;
        for up in &self.up_blocks {
            x = up.forward(&x)?;
        }
        x = self.norm_out.forward(&x)?;
        x = silu(&x)?;
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
        Ok(Self {
            dtype: vb.dtype(),
            post_quant_conv: conv2d_k1(z, z, vb.pp("post_quant_conv"))?,
            decoder: Decoder3d::load(cfg, vb.pp("decoder"))?,
            latents_mean: mean,
            latents_std: std,
        })
    }

    /// Decode an already-unpacked, normalized latent `(B, z, H, W)` into an RGBA
    /// image `(B, 4, H*16, W*16)` in [-1, 1] (pre postprocess).
    pub fn decode(&self, z_normalized: &Tensor) -> Result<Tensor> {
        // Unnormalize in f32 (per-channel std/mean; latents_std stores 1/std),
        // then cast to the conv working dtype (bf16 on CUDA → tensor-core convs).
        let z = z_normalized
            .to_dtype(DType::F32)?
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?
            .to_dtype(self.dtype)?;
        let x = self.post_quant_conv.forward(&z)?;
        let out = self.decoder.forward(&x)?;
        Ok(out.clamp(-1.0, 1.0)?)
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
    let img = ((img.clamp(-1.0, 1.0)? + 1.0)? * 0.5)?; // [0,1]
    let img = img.clamp(0.0, 1.0)?.to_dtype(DType::F32)?;
    let (_, c, h, w) = img.dims4()?;
    // (1,C,H,W) -> (H,W,C)
    let hwc = img.i(0)?.permute((1, 2, 0))?.contiguous()?;
    let data = (hwc * 255.0)?.round()?.to_dtype(DType::U8)?;
    let bytes = data.flatten_all()?.to_vec1::<u8>()?;
    debug_assert_eq!(c, 4);
    Ok((w, h, bytes))
}

use candle_core::IndexOp;

#[cfg(test)]
mod tests {
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
        };
        let out = n.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let s = 2f32.sqrt();
        assert!((out[0] - 0.6 * s).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 0.8 * s).abs() < 1e-5, "{out:?}");
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
