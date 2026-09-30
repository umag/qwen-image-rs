//! `QwenImage21Transformer2DModel` — 32-layer single-stream MMDiT. **Phase 4.**
//!
//! Joint text+image sequence, block-causal attention (text causal, the target
//! image block internally bidirectional), `causal_condition` modulation (image
//! tokens modulate from the sampled timestep, text tokens from t=0), 3-axis
//! complex RoPE. Text-to-image only: no condition images, no KV cache, no flex
//! attention — the block-causal mask is built dense and applied once per block.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::Result;

/// A no-bias linear that is full-precision, Q8_0-quantized (GGUF), or ConvRot
/// W8A8 INT8 (rotated int8 GEMM, `convrot` feature).
enum QLinear {
    Full(Linear),
    Quant(QMatMul),
    #[cfg(feature = "convrot")]
    Convrot(crate::convrot::ConvRotLinear),
}

impl QLinear {
    /// Load `(out, in)` weights. `convrot` (with a rotation `rot` and in-features
    /// a multiple of 256) takes priority over `quant`; otherwise Q8_0 when
    /// `quant`, else full-precision.
    fn load(
        in_c: usize,
        out_c: usize,
        quant: bool,
        convrot: bool,
        rot: Option<&Tensor>,
        vb: VarBuilder,
    ) -> Result<Self> {
        #[cfg(feature = "convrot")]
        if convrot {
            if let Some(r) = rot {
                if in_c % crate::model::rotation::GROUP == 0 {
                    // Pre-quantized weights on disk (from `prequantize-convrot`):
                    // load the rotated INT8 weight + col scale directly, skipping
                    // the load-time rotate+quant. Detected by `weight_i8` next to
                    // the usual `weight`.
                    if vb.contains_tensor("weight_i8") {
                        let w_i8 = vb.get_unchecked_dtype("weight_i8", DType::U8)?;
                        let col_scale = vb.get_unchecked_dtype("col_scale", DType::F32)?;
                        return Ok(QLinear::Convrot(
                            crate::convrot::ConvRotLinear::from_prequantized(w_i8, col_scale, r)?,
                        ));
                    }
                    let w = vb.get((out_c, in_c), "weight")?;
                    return Ok(QLinear::Convrot(
                        crate::convrot::ConvRotLinear::from_weight(&w, r)?,
                    ));
                }
            }
        }
        #[cfg(not(feature = "convrot"))]
        let _ = (convrot, rot);
        if quant {
            let w = vb.get((out_c, in_c), "weight")?;
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            Ok(QLinear::Quant(QMatMul::from_qtensor(qt)?))
        } else {
            Ok(QLinear::Full(linear_no_bias(in_c, out_c, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Full(l) => Ok(l.forward(x)?),
            QLinear::Quant(q) => Ok(q.forward(x)?),
            #[cfg(feature = "convrot")]
            QLinear::Convrot(c) => c.forward(x),
        }
    }
}

const INNER: usize = 4096; // num_heads * head_dim = 32 * 128
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const AXES: [usize; 3] = [16, 56, 56]; // rope dims (frame, height, width)
const ROPE_THETA: f64 = 10000.0;

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// LayerNorm with no learnable affine (elementwise_affine=False), eps 1e-6.
/// Used by `norm_mod`'s non-fused fallback.
#[cfg_attr(feature = "fusednorm", allow(dead_code))]
fn norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let mean = x32.mean_keepdim(candle_core::D::Minus1)?;
    let xc = x32.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(normed.to_dtype(x.dtype())?)
}

/// `norm_no_affine(x) * (scale + 1)` — the DiT's AdaLN pattern. Under the
/// `fusednorm` feature this is one fused CUDA kernel; otherwise the candle ops.
/// `x` and `scale` share shape `(..., INNER)`.
fn norm_mod(x: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    {
        crate::fusednorm::fused_norm_mod(x, scale, eps as f32)
    }
    #[cfg(not(feature = "fusednorm"))]
    {
        Ok(norm_no_affine(x, eps)?.broadcast_mul(&(scale + 1.0)?)?)
    }
}

/// `x * rsqrt(mean(x^2)+eps) * w` — the DiT's RMSNorm pattern with a per-channel
/// weight. `w` is the effective f32 weight vector `(N,)`: `weight+1` for
/// ZeroCenterRmsNorm, `weight` for HeadRmsNorm (both baked in f32 at load).
/// Under `fusednorm` this is one fused CUDA kernel; otherwise the candle ops.
fn rmsnorm_scale(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    {
        crate::fusednorm::fused_rmsnorm_scale(x, w, eps as f32)
    }
    #[cfg(not(feature = "fusednorm"))]
    {
        let dt = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let ms = x32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rrms = (ms + eps)?.sqrt()?.recip()?;
        let out = x32.broadcast_mul(&rrms)?.broadcast_mul(w)?;
        Ok(out.to_dtype(dt)?)
    }
}

/// `h + tanh(gate) * y` — the DiT's gated residual (2× per block). `h`, `gate`,
/// `y` share shape. Under `fusednorm` this is one fused elementwise kernel;
/// otherwise the candle ops.
fn gated_residual(h: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    {
        crate::fusednorm::fused_gated_residual(h, gate, y)
    }
    #[cfg(not(feature = "fusednorm"))]
    {
        Ok((h + gate.tanh()?.broadcast_mul(y)?)?)
    }
}

/// `QwenImage21ZeroCenterRMSNorm`: scale = weight + 1, computed in fp32. The
/// `weight+1` is baked in f32 once at load so the fused kernel and the candle
/// fallback both scale by the effective per-channel weight.
struct ZeroCenterRmsNorm {
    weight: Tensor, // (dim,) f32, = raw_weight + 1
    eps: f64,
}

impl ZeroCenterRmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: (vb.get(dim, "weight")?.to_dtype(DType::F32)? + 1.0)?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rmsnorm_scale(x, &self.weight, self.eps)
    }
}

/// `QwenImage21TextProjection`: ZeroCenterRMSNorm -> Linear -> GELU(tanh) -> Linear.
struct TextProjection {
    norm: ZeroCenterRmsNorm,
    in_layer: Linear,
    out_layer: Linear,
}

impl TextProjection {
    fn load(ctx_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: ZeroCenterRmsNorm::load(ctx_dim, 1e-6, vb.pp("text_norm"))?,
            in_layer: linear_no_bias(ctx_dim, INNER, vb.pp("in_layer"))?,
            out_layer: linear_no_bias(INNER, INNER, vb.pp("out_layer"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.in_layer.forward(&x)?;
        let x = x.gelu()?; // tanh approximation (matches GELU(approximate="tanh"))
        Ok(self.out_layer.forward(&x)?)
    }
}

/// Sinusoidal timestep embedding (256) -> MLP(256->INNER, silu, INNER->INNER).
struct TimestepEmbed {
    linear1: Linear,
    linear2: Linear,
    freqs: Tensor, // (128,)
}

impl TimestepEmbed {
    fn load(vb: VarBuilder, dev: &Device) -> Result<Self> {
        let half = 128usize;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp())
            .collect();
        Ok(Self {
            linear1: linear_no_bias(256, INNER, vb.pp("timestep_embedder").pp("linear_1"))?,
            linear2: linear_no_bias(INNER, INNER, vb.pp("timestep_embedder").pp("linear_2"))?,
            freqs: Tensor::from_vec(freqs, (1, half), dev)?,
        })
    }

    /// `timestep` (n,) in [0,1] -> temb (n, INNER), computed in the given dtype.
    fn forward(&self, timestep: &Tensor, dtype: DType) -> Result<Tensor> {
        let t = (timestep.to_dtype(DType::F32)? * 1000.0)?.reshape(((), 1))?;
        let args = t.broadcast_mul(&self.freqs)?; // (n, 128)
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], candle_core::D::Minus1)?; // (n,256)
        let emb = emb.to_dtype(dtype)?;
        let x = self.linear1.forward(&emb)?;
        let x = silu(&x)?;
        Ok(self.linear2.forward(&x)?)
    }
}

/// Per-head RMSNorm over head_dim (Qwen attention q/k norm). Weight is upcast to
/// f32 once at load (exact) so it feeds the shared `rmsnorm_scale` helper.
struct HeadRmsNorm {
    weight: Tensor, // (head_dim,) f32
    eps: f64,
}
impl HeadRmsNorm {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(HEAD_DIM, "weight")?.to_dtype(DType::F32)?,
            eps: 1e-6,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x (..., head_dim)
        rmsnorm_scale(x, &self.weight, self.eps)
    }
}

struct Attention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    norm_q: HeadRmsNorm,
    norm_k: HeadRmsNorm,
}

impl Attention {
    fn load(quant: bool, convrot: bool, rot: Option<&Tensor>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: QLinear::load(INNER, INNER, quant, convrot, rot, vb.pp("to_q"))?,
            to_k: QLinear::load(INNER, INNER, quant, convrot, rot, vb.pp("to_k"))?,
            to_v: QLinear::load(INNER, INNER, quant, convrot, rot, vb.pp("to_v"))?,
            to_out: QLinear::load(INNER, INNER, quant, convrot, rot, vb.pp("to_out").pp("0"))?,
            norm_q: HeadRmsNorm::load(vb.pp("norm_q"))?,
            norm_k: HeadRmsNorm::load(vb.pp("norm_k"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        txt_len: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let shape = (b, s, HEADS, HEAD_DIM);
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        #[cfg(feature = "sage")]
        {
            let _ = mask; // block-causal structure is expressed via narrows, not a mask
                          // BSHD-native: q/k/v stay (B,S,H,D). RoPE is fused into the
                          // sage INT8 quantizer (attend_bshd), so q/k are rotated and
                          // quantized in one pass — no separate rope kernel, no
                          // rotated-bf16 round trip, no transpose copies.
            let qh = self
                .norm_q
                .forward(&self.to_q.forward(x)?.reshape(shape)?)?; // (B,S,H,D), pre-rope
            let kh = self
                .norm_k
                .forward(&self.to_k.forward(x)?.reshape(shape)?)?;
            let vv = self.to_v.forward(x)?.reshape(shape)?; // (B,S,H,D)
            let out = self.attend_bshd(&qh, &kh, &vv, cos, sin, txt_len, scale)?; // (B,S,H,D)
            return self.to_out.forward(&out.reshape((b, s, INNER))?);
        }
        #[cfg(not(feature = "sage"))]
        {
            // (B,H,S,D) for RoPE (rope_i rotates per position across heads).
            let qh = self
                .norm_q
                .forward(&self.to_q.forward(x)?.reshape(shape)?)?
                .transpose(1, 2)?
                .contiguous()?;
            let kh = self
                .norm_k
                .forward(&self.to_k.forward(x)?.reshape(shape)?)?
                .transpose(1, 2)?
                .contiguous()?;
            let vv = self.to_v.forward(x)?.reshape(shape)?; // (B,S,H,D)
            let qh = candle_nn::rotary_emb::rope_i(&qh, cos, sin)?;
            let kh = candle_nn::rotary_emb::rope_i(&kh, cos, sin)?;
            let out = self.attend(&qh, &kh, &vv, mask, txt_len, scale)?; // (B,S,H,D)
            self.to_out.forward(&out.reshape((b, s, INNER))?)
        }
    }

    /// With `sage`: BSHD-native block-causal SageAttention (INT8-QK / FP16-PV).
    /// `qh,kh` are PRE-rope `(B,S,H,D)`; `vv` is `(B,S,H,D)`; `cos,sin` are the
    /// joint-sequence `(S, D/2)` RoPE tables. Image queries attend non-causally
    /// to the whole joint sequence; the `txt_len` text queries attend causally to
    /// the text prefix. Each attention operand is rotated + INT8-quantized by ONE
    /// fused kernel on its own S-axis narrow (with the matching cos/sin rows), so
    /// the per-block scales start at that call's first token — exactly the
    /// alignment the attention kernel's per-block scale indexing expects (k is
    /// quantized twice: text prefix and full, as before). Returns `(B,S,H,D)`.
    #[cfg(feature = "sage")]
    #[allow(clippy::too_many_arguments)]
    fn attend_bshd(
        &self,
        qh: &Tensor,
        kh: &Tensor,
        vv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        txt_len: usize,
        scale: f64,
    ) -> Result<Tensor> {
        use crate::sage::{rope_quant_bshd, sage_attention_quantized, QkRole};
        let (_b, s, _h, _d) = qh.dims4()?;
        let img = s - txt_len;
        // V is cast to f16 once (contiguous, no transpose); the narrows below are
        // zero-copy S-axis views the bridge reads via start_offset + strides.
        let vf = vv.to_dtype(DType::F16)?; // (B,S,H,D) f16
        let sc = scale as f32;
        let (ct, st) = (cos.narrow(0, 0, txt_len)?, sin.narrow(0, 0, txt_len)?);
        // text prefix: causal over [0, txt_len)
        let qt = rope_quant_bshd(&qh.narrow(1, 0, txt_len)?, &ct, &st, QkRole::Query)?;
        let kt = rope_quant_bshd(&kh.narrow(1, 0, txt_len)?, &ct, &st, QkRole::Key)?;
        let vt = vf.narrow(1, 0, txt_len)?;
        let ot = sage_attention_quantized(&qt, &kt, &vt, sc, true)?; // (B,txt,H,D)

        // image queries: full non-causal attention over the whole sequence
        let qi = rope_quant_bshd(
            &qh.narrow(1, txt_len, img)?,
            &cos.narrow(0, txt_len, img)?,
            &sin.narrow(0, txt_len, img)?,
            QkRole::Query,
        )?;
        let kf = rope_quant_bshd(kh, cos, sin, QkRole::Key)?;
        let oi = sage_attention_quantized(&qi, &kf, &vf, sc, false)?; // (B,img,H,D)
        Ok(Tensor::cat(&[ot, oi], 1)?) // (B,S,H,D)
    }

    /// With `flash-attn`: image queries take full FlashAttention (they attend to
    /// the whole joint sequence) and the `txt_len` text queries take a small
    /// causal FlashAttention — exactly the block-causal structure for
    /// text-to-image, so no dense S² matrix is materialized. Returns (B,S,H,D).
    #[cfg(all(feature = "flash-attn", not(feature = "sage")))]
    fn attend(
        &self,
        qh: &Tensor,
        kh: &Tensor,
        vv: &Tensor,
        mask: &Tensor,
        txt_len: usize,
        scale: f64,
    ) -> Result<Tensor> {
        let _ = mask;
        let (_b, _h, s, _d) = qh.dims4()?;
        let qf = qh.transpose(1, 2)?.contiguous()?; // (B,S,H,D)
        let kf = kh.transpose(1, 2)?.contiguous()?;
        let vf = vv.contiguous()?;
        let sc = scale as f32;
        let ot = candle_flash_attn::flash_attn(
            &qf.narrow(1, 0, txt_len)?.contiguous()?,
            &kf.narrow(1, 0, txt_len)?.contiguous()?,
            &vf.narrow(1, 0, txt_len)?.contiguous()?,
            sc,
            true,
        )?;
        let oi = candle_flash_attn::flash_attn(
            &qf.narrow(1, txt_len, s - txt_len)?.contiguous()?,
            &kf,
            &vf,
            sc,
            false,
        )?;
        Ok(Tensor::cat(&[ot, oi], 1)?)
    }

    /// Naive attention with the additive block-causal `mask`. Returns (B,S,H,D).
    #[cfg(all(not(feature = "flash-attn"), not(feature = "sage")))]
    fn attend(
        &self,
        qh: &Tensor,
        kh: &Tensor,
        vv: &Tensor,
        mask: &Tensor,
        txt_len: usize,
        scale: f64,
    ) -> Result<Tensor> {
        let _ = txt_len;
        let v = vv.transpose(1, 2)?.contiguous()?; // (B,H,S,D)
        let attn = (qh.matmul(&kh.transpose(2, 3)?)? * scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // (B,H,S,D)
        Ok(out.transpose(1, 2)?.contiguous()?)
    }
}

struct SwiGlu {
    proj: QLinear,
    gate: QLinear,
    out: QLinear,
}
impl SwiGlu {
    fn load(quant: bool, convrot: bool, rot: Option<&Tensor>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: QLinear::load(INNER, INNER * 3, quant, convrot, rot, vb.pp("proj"))?,
            gate: QLinear::load(INNER, INNER * 3, quant, convrot, rot, vb.pp("gate_layer"))?,
            out: QLinear::load(INNER * 3, INNER, quant, convrot, rot, vb.pp("out"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = silu(&self.gate.forward(x)?)?;
        self.out.forward(&(g * self.proj.forward(x)?)?)
    }
}

struct Block {
    attn: Attention,
    mlp: SwiGlu,
    eps: f64,
}

impl Block {
    fn load(quant: bool, convrot: bool, rot: Option<&Tensor>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(quant, convrot, rot, vb.pp("attn"))?,
            mlp: SwiGlu::load(quant, convrot, rot, vb.pp("img_mlp"))?,
            eps: 1e-6,
        })
    }

    /// `mod1`/`mod2` are the per-token (1,S,2*INNER) selected modulation slices.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        h: &Tensor,
        scale1: &Tensor,
        gate1: &Tensor,
        scale2: &Tensor,
        gate2: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        txt_len: usize,
    ) -> Result<Tensor> {
        let x = norm_mod(h, scale1, self.eps)?;
        let attn = self.attn.forward(&x, mask, cos, sin, txt_len)?;
        let h = gated_residual(h, gate1, &attn)?;
        let x = norm_mod(&h, scale2, self.eps)?;
        let m = self.mlp.forward(&x)?;
        gated_residual(&h, gate2, &m)
    }
}

/// Full DiT.
pub struct QwenImageDit {
    img_in: Linear,
    txt_in: TextProjection,
    time_embed: TimestepEmbed,
    modulation: Linear, // INNER -> 4*INNER (after silu)
    blocks: Vec<Block>,
    norm_out_linear: Linear, // INNER -> INNER
    proj_out: Linear,        // INNER -> out_channels
    device: Device,
    inv_freqs: [Vec<f32>; 3], // per-axis rope inv-freqs
}

impl QwenImageDit {
    pub fn load(
        num_layers: usize,
        out_channels: usize,
        quant: bool,
        convrot: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dev = vb.device().clone();
        // Build the 256×256 Regular Hadamard rotation once; ConvRotLinear stores
        // a cheap handle clone. Only meaningful under the `convrot` feature.
        #[cfg(feature = "convrot")]
        let rot: Option<Tensor> = if convrot {
            Some(crate::model::rotation::regular_hadamard_256(&dev)?)
        } else {
            None
        };
        #[cfg(not(feature = "convrot"))]
        let rot: Option<Tensor> = {
            let _ = convrot;
            None
        };
        let mut blocks = Vec::with_capacity(num_layers);
        let vb_b = vb.pp("transformer_blocks");
        for i in 0..num_layers {
            blocks.push(Block::load(quant, convrot, rot.as_ref(), vb_b.pp(i))?);
        }
        let inv_freqs = std::array::from_fn(|a| {
            let d = AXES[a];
            (0..d / 2)
                .map(|j| 1f32 / ROPE_THETA.powf(2.0 * j as f64 / d as f64) as f32)
                .collect()
        });
        Ok(Self {
            img_in: linear_no_bias(64, INNER, vb.pp("img_in"))?,
            txt_in: TextProjection::load(4096, vb.pp("txt_in"))?,
            time_embed: TimestepEmbed::load(vb.pp("time_text_embed"), &dev)?,
            modulation: linear_no_bias(INNER, 4 * INNER, vb.pp("modulation").pp("1"))?,
            blocks,
            norm_out_linear: linear_no_bias(INNER, INNER, vb.pp("norm_out").pp("linear"))?,
            proj_out: linear_no_bias(INNER, out_channels, vb.pp("proj_out"))?,
            device: dev,
            inv_freqs,
        })
    }

    /// Build cos/sin `(S, 32)` for interleaved RoPE, from per-token 3-axis
    /// position indices. `img_shapes` = (frame, H, W) for the target image;
    /// `txt_len` text tokens precede `H*W` image tokens.
    fn rope_cos_sin(
        &self,
        txt_len: usize,
        h: usize,
        w: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let img = h * w;
        let seq = txt_len + img;
        // frame/height/width index per token (i32 to allow negatives)
        let mut frame = vec![0i64; seq];
        let mut height = vec![0i64; seq];
        let mut width = vec![0i64; seq];
        // text: positions 0..txt_len on all axes
        for (p, item) in frame.iter_mut().enumerate().take(txt_len) {
            *item = p as i64;
            height[p] = p as i64;
            width[p] = p as i64;
        }
        // image: frame frozen at txt_len; height/width grid centered on zero
        let hs = -((h - h / 2) as i64);
        let ws = -((w - w / 2) as i64);
        for r in 0..h {
            for c in 0..w {
                let idx = txt_len + r * w + c;
                frame[idx] = txt_len as i64;
                height[idx] = hs + r as i64;
                width[idx] = ws + c as i64;
            }
        }
        // angle per token = concat over axes of pos * inv_freq
        let half: usize = AXES.iter().map(|d| d / 2).sum(); // 8+28+28 = 64
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        let idxs = [&frame, &height, &width];
        for s in 0..seq {
            let mut off = 0;
            for a in 0..3 {
                let pos = idxs[a][s] as f32;
                for (j, inv) in self.inv_freqs[a].iter().enumerate() {
                    let ang = pos * inv;
                    cos[s * half + off + j] = ang.cos();
                    sin[s * half + off + j] = ang.sin();
                }
                off += AXES[a] / 2;
            }
        }
        let cos = Tensor::from_vec(cos, (seq, half), &self.device)?.to_dtype(dtype)?;
        let sin = Tensor::from_vec(sin, (seq, half), &self.device)?.to_dtype(dtype)?;
        Ok((cos, sin))
    }

    /// Forward for text-to-image (no condition images, no mask/cache).
    /// `hidden_states` (1, H*W, 64), `encoder_hidden_states` (1, txt_len, 4096),
    /// `timestep` scalar tensor (1,), `(h,w)` target latent grid.
    /// Returns the joint output (1, txt_len + H*W, out_channels).
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let (b, txt_len, _) = encoder_hidden_states.dims3()?;
        let img_tokens = h * w;
        let seq = txt_len + img_tokens;

        let img = self.img_in.forward(hidden_states)?; // (1, img, INNER)
        let txt = self.txt_in.forward(encoder_hidden_states)?; // (1, txt, INNER)
                                                               // joint = [text, image]
        let joint = Tensor::cat(&[&txt, &img], 1)?; // (1, seq, INNER)

        // timestep embedding for [t, 0]; modulation for both rows.
        let t0 = Tensor::zeros((1,), dtype, &self.device)?;
        let ts = Tensor::cat(&[&timestep.to_dtype(dtype)?, &t0], 0)?; // (2,)
        let temb = self.time_embed.forward(&ts, dtype)?; // (2, INNER)
        let modhad = self.modulation.forward(&silu(&temb)?)?; // (2, 4*INNER)

        // Split modulation into scale1,gate1,scale2,gate2, each (2, INNER).
        let chunk = |t: &Tensor, i: usize| -> Result<Tensor> { Ok(t.narrow(1, i * INNER, INNER)?) };
        let mods: Vec<Tensor> = (0..4).map(|i| chunk(&modhad, i)).collect::<Result<_>>()?;
        // Per-token selection: image tokens use row 0 (real t), text row 1 (t=0).
        // select_rows yields (1, seq, INNER); for a batch, broadcast to (b, seq,
        // INNER) and materialize (the fused norm/gate kernels index per row and
        // read raw pointers, so a stride-0 broadcast view won't do). B=1 keeps
        // the (1, seq, INNER) tensor untouched.
        let sel = |m: &Tensor| -> Result<Tensor> {
            let r = select_rows(m, txt_len, img_tokens, &self.device)?;
            if b > 1 {
                Ok(r.broadcast_as((b, seq, INNER))?.contiguous()?)
            } else {
                Ok(r)
            }
        };
        let scale1 = sel(&mods[0])?;
        let gate1 = sel(&mods[1])?;
        let scale2 = sel(&mods[2])?;
        let gate2 = sel(&mods[3])?;

        // The dense block-causal mask is only needed by the naive path; the
        // flash and sage paths derive the same structure from the text/image split.
        #[cfg(all(not(feature = "flash-attn"), not(feature = "sage")))]
        let mask = block_causal_mask(txt_len, img_tokens, dtype, &self.device)?;
        #[cfg(any(feature = "flash-attn", feature = "sage"))]
        let mask = Tensor::zeros((1, 1, 1, 1), dtype, &self.device)?;
        let (cos, sin) = self.rope_cos_sin(txt_len, h, w, dtype)?;

        let mut x = joint;
        for block in &self.blocks {
            x = block.forward(
                &x, &scale1, &gate1, &scale2, &gate2, &mask, &cos, &sin, txt_len,
            )?;
        }

        // norm_out: AdaLayerNorm scale-only, then proj_out. Scale from temb rows.
        let scale_out = self.norm_out_linear.forward(&silu(&temb)?)?; // (2, INNER)
        let scale_out = select_rows(&scale_out, txt_len, img_tokens, &self.device)?; // (1,seq,INNER)
        let scale_out = if b > 1 {
            scale_out.broadcast_as((b, seq, INNER))?.contiguous()?
        } else {
            scale_out
        };
        let x = norm_mod(&x, &scale_out, 1e-6)?;
        let out = self.proj_out.forward(&x)?; // (1, seq, out_channels)
        let _ = seq;
        Ok(out)
    }
}

/// Select modulation rows per token: `(2, INNER)` -> `(1, seq, INNER)`, image
/// tokens take row 0 (real timestep), text tokens row 1 (t=0).
fn select_rows(m: &Tensor, txt_len: usize, img_tokens: usize, dev: &Device) -> Result<Tensor> {
    let real = m.narrow(0, 0, 1)?; // (1, INNER)
    let zero = m.narrow(0, 1, 1)?; // (1, INNER)
    let text_rows = zero.broadcast_as((txt_len, m.dim(1)?))?;
    let img_rows = real.broadcast_as((img_tokens, m.dim(1)?))?;
    let rows = Tensor::cat(&[&text_rows, &img_rows], 0)?.unsqueeze(0)?; // (1, seq, INNER)
    let _ = dev;
    Ok(rows)
}

/// Block-causal additive mask `(1,1,S,S)`: allowed[q,kv] = (q>=kv) OR both-image.
/// Used by the naive attention path only (the flash path derives the structure).
#[cfg_attr(any(feature = "flash-attn", feature = "sage"), allow(dead_code))]
fn block_causal_mask(
    txt_len: usize,
    img_tokens: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let s = txt_len + img_tokens;
    let mut data = vec![0f32; s * s];
    for q in 0..s {
        for kv in 0..s {
            let both_image = q >= txt_len && kv >= txt_len;
            let allowed = kv <= q || both_image;
            if !allowed {
                data[q * s + kv] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, s, s), dev)?.to_dtype(dtype)?)
}
