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

/// The seven linears of every transformer block, by weight prefix relative to
/// `transformer_blocks.<i>.` (the safetensors key minus `.weight`).
const BLOCK_LINEARS: [&str; 7] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.proj",
    "img_mlp.gate_layer",
    "img_mlp.out",
];

/// The DiT's non-block ("tail") linears: `(weight prefix, in K, out N, ConvRot
/// under --convrot)`. A rotated layer needs K % 256 == 0 (the 256-wide
/// Hadamard) and N % 8 == 0 (the epilogue's 128-bit 16-bit store); `img_in`
/// (K = 64) cannot rotate. N for `proj_out` is the checkpoint's out_channels.
const TAIL_LINEARS: [(&str, usize, usize, bool); 8] = [
    ("img_in", 64, INNER, false),
    ("txt_in.in_layer", 4096, INNER, true),
    ("txt_in.out_layer", INNER, INNER, true),
    (
        "time_text_embed.timestep_embedder.linear_1",
        256,
        INNER,
        true,
    ),
    (
        "time_text_embed.timestep_embedder.linear_2",
        INNER,
        INNER,
        true,
    ),
    ("modulation.1", INNER, 4 * INNER, true),
    ("norm_out.linear", INNER, INNER, true),
    ("proj_out", INNER, 64, true),
];

/// How one DiT linear computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPrecision {
    /// bf16 `x Wᵀ` (candle `Linear`).
    Full,
    /// Q8_0 GGUF weights (`--quant`; block linears only).
    Q8,
    /// ConvRot W8A8 INT8 (`--convrot`).
    ConvRot,
}

/// Block-linear suffix of a `transformer_blocks.<i>.<suffix>` prefix.
fn block_suffix(prefix: &str) -> Option<&str> {
    let rest = prefix.strip_prefix("transformer_blocks.")?;
    let (idx, suffix) = rest.split_once('.')?;
    (!idx.is_empty() && idx.bytes().all(|c| c.is_ascii_digit())).then_some(suffix)
}

/// The one precision policy for every DiT linear, keyed by its weight prefix.
/// Shared by `QwenImageDit::load` and `prequantize-convrot`, so a
/// pre-quantized file holds exactly the layers the loader rotates. Block
/// linears: ConvRot when `convrot`, else Q8_0 when `quant`, else Full. Tail
/// linears: ConvRot when `convrot` and the table enables it, else Full (Q8_0
/// never applied to them). `None` = not a DiT linear.
pub fn linear_precision(prefix: &str, quant: bool, convrot: bool) -> Option<LinearPrecision> {
    let p = if block_suffix(prefix).is_some_and(|s| BLOCK_LINEARS.contains(&s)) {
        if convrot {
            LinearPrecision::ConvRot
        } else if quant {
            LinearPrecision::Q8
        } else {
            LinearPrecision::Full
        }
    } else {
        let &(_, _, _, rot) = TAIL_LINEARS.iter().find(|t| t.0 == prefix)?;
        if convrot && rot {
            LinearPrecision::ConvRot
        } else {
            LinearPrecision::Full
        }
    };
    Some(p)
}

/// True when `prefix` is a linear that `--convrot` runs rotated INT8 (and
/// `prequantize-convrot` therefore stores as `weight_i8` + `col_scale`).
pub fn is_convrot_target(prefix: &str) -> bool {
    linear_precision(prefix, false, true) == Some(LinearPrecision::ConvRot)
}

/// Version of the stored ConvRot weight format and its quantizer math
/// (`ConvRotLinear::from_weight`: fold `W Rᵀ`, per-row amax/127, round to
/// int8). Bump it whenever that math or the tensor layout changes, so every
/// cached prequantized file (`crate::convrot_cache`) is rebuilt.
pub const PREQUANT_FORMAT: u32 = 1;

/// The precision-policy tag stored in a prequantized file: the format
/// version, the rotation group and every DiT linear's ConvRot decision. Two
/// builds agree on a cached file iff their tags are equal; flipping a
/// `TAIL_LINEARS` flag or editing `BLOCK_LINEARS` changes it.
pub fn convrot_policy_tag() -> String {
    policy_tag(&BLOCK_LINEARS, &TAIL_LINEARS)
}

fn policy_tag(block: &[&str], tail: &[(&str, usize, usize, bool)]) -> String {
    let tail: Vec<String> = tail
        .iter()
        .map(|&(p, k, n, rot)| format!("{p}:{k}x{n}:{}", u8::from(rot)))
        .collect();
    format!(
        "qir-convrot-v{PREQUANT_FORMAT};group={};block={};tail={}",
        crate::model::rotation::GROUP,
        block.join(","),
        tail.join(",")
    )
}

/// Load-time precision switches shared by every DiT linear.
#[derive(Clone, Copy)]
struct LinearOpts<'a> {
    quant: bool,
    convrot: bool,
    /// The 256×256 Hadamard; `Some` iff `convrot` (and the feature) is on.
    rot: Option<&'a Tensor>,
}

/// A no-bias linear that is full-precision, Q8_0-quantized (GGUF), or ConvRot
/// W8A8 INT8 (rotated int8 GEMM, `convrot` feature). Every DiT linear is one;
/// `linear_precision` picks the variant from the weight prefix.
enum QLinear {
    Full(Linear),
    Quant(QMatMul),
    #[cfg(feature = "convrot")]
    Convrot(crate::convrot::ConvRotLinear),
}

impl QLinear {
    /// Load the `(out, in)` weight at `vb`'s prefix with the precision
    /// `linear_precision` assigns it.
    fn load(in_c: usize, out_c: usize, opts: LinearOpts, vb: VarBuilder) -> Result<Self> {
        let prefix = vb.prefix();
        let convrot = opts.convrot && opts.rot.is_some();
        let precision = linear_precision(&prefix, opts.quant, convrot)
            .ok_or_else(|| anyhow::anyhow!("{prefix}: not a DiT linear in the precision policy"))?;
        match precision {
            LinearPrecision::ConvRot => Self::load_convrot(in_c, out_c, opts.rot, &prefix, vb),
            LinearPrecision::Q8 => {
                let w = vb.get((out_c, in_c), "weight")?;
                let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
                Ok(QLinear::Quant(QMatMul::from_qtensor(qt)?))
            }
            LinearPrecision::Full => {
                if !vb.contains_tensor("weight") && vb.contains_tensor("weight_i8") {
                    anyhow::bail!(
                        "{prefix}: the weight file holds only a ConvRot INT8 weight for this \
                         bf16 layer (run with --convrot, or regenerate it with prequantize-convrot)"
                    );
                }
                Ok(QLinear::Full(linear_no_bias(in_c, out_c, vb)?))
            }
        }
    }

    /// ConvRot load: the pre-quantized weight when the file has one (from
    /// `prequantize-convrot`, detected by `weight_i8` next to `weight`), else
    /// rotate + quantize the bf16 weight now.
    #[cfg(feature = "convrot")]
    fn load_convrot(
        in_c: usize,
        out_c: usize,
        rot: Option<&Tensor>,
        prefix: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let r = rot.ok_or_else(|| anyhow::anyhow!("{prefix}: ConvRot needs the rotation"))?;
        anyhow::ensure!(
            in_c.is_multiple_of(crate::model::rotation::GROUP) && out_c.is_multiple_of(8),
            "{prefix}: ConvRot needs K % 256 == 0 and N % 8 == 0 (K={in_c}, N={out_c})"
        );
        if vb.contains_tensor("weight_i8") {
            let w_i8 = vb.get_unchecked_dtype("weight_i8", DType::U8)?;
            let col_scale = vb.get_unchecked_dtype("col_scale", DType::F32)?;
            return Ok(QLinear::Convrot(
                crate::convrot::ConvRotLinear::from_prequantized(w_i8, col_scale, r)?,
            ));
        }
        let w = vb.get((out_c, in_c), "weight")?;
        Ok(QLinear::Convrot(
            crate::convrot::ConvRotLinear::from_weight(&w, r)?,
        ))
    }

    #[cfg(not(feature = "convrot"))]
    fn load_convrot(
        _: usize,
        _: usize,
        _: Option<&Tensor>,
        prefix: &str,
        _: VarBuilder,
    ) -> Result<Self> {
        anyhow::bail!("{prefix}: ConvRot requires the `convrot` feature")
    }

    /// True for the ConvRot INT8 variant (load-time accounting).
    fn is_convrot(&self) -> bool {
        #[cfg(feature = "convrot")]
        if let QLinear::Convrot(_) = self {
            return true;
        }
        false
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Full(l) => Ok(l.forward(x)?),
            QLinear::Quant(q) => Ok(q.forward(x)?),
            #[cfg(feature = "convrot")]
            QLinear::Convrot(c) => c.forward(x),
        }
    }

    /// Forward emitting f16 — the sage FP16 P·V operand dtype. ConvRot stores
    /// f16 straight from its f32 dequant epilogue (one rounding, no cast
    /// kernel); Full/Quant keep the bf16 forward + the same cast as before.
    #[cfg(feature = "sage")]
    fn forward_f16(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "convrot")]
        if let QLinear::Convrot(c) = self {
            return c.forward_as(x, crate::convrot::EpilogueOut::F16);
        }
        Ok(self.forward(x)?.to_dtype(DType::F16)?)
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
/// Used by `residual_norm_mod`'s non-fused fallback.
#[cfg_attr(feature = "fusednorm", allow(dead_code))]
fn norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let mean = x32.mean_keepdim(candle_core::D::Minus1)?;
    let xc = x32.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(normed.to_dtype(x.dtype())?)
}

/// One AdaLN modulation chunk (`scale` or `gate`) for the joint sequence: the
/// `(2, INNER)` rows — row 0 for image tokens (real timestep), row 1 for the
/// `txt_len` text tokens (t = 0) that open every sequence; every batch lane
/// shares the split. The fused path reads the rows by token position; the
/// candle fallback multiplies by the per-token tensor, materialized once.
struct Modulation {
    #[cfg(feature = "fusednorm")]
    rows: Tensor, // (2, INNER)
    #[cfg(feature = "fusednorm")]
    txt_len: usize,
    #[cfg(not(feature = "fusednorm"))]
    tokens: Tensor, // (B, S, INNER)
}

impl Modulation {
    #[cfg(feature = "fusednorm")]
    fn new(rows: Tensor, _b: usize, txt_len: usize, _img_tokens: usize) -> Result<Self> {
        Ok(Self { rows, txt_len })
    }

    #[cfg(not(feature = "fusednorm"))]
    fn new(rows: Tensor, b: usize, txt_len: usize, img_tokens: usize) -> Result<Self> {
        let r = select_rows(&rows, txt_len, img_tokens)?; // (1, seq, INNER)
        let tokens = if b > 1 {
            r.broadcast_as((b, txt_len + img_tokens, rows.dim(1)?))?
                .contiguous()?
        } else {
            r
        };
        Ok(Self { tokens })
    }

    #[cfg(feature = "fusednorm")]
    fn fused(&self) -> crate::fusednorm::ModRows<'_> {
        crate::fusednorm::ModRows {
            rows: &self.rows,
            txt_len: self.txt_len,
        }
    }
}

/// The DiT's AdaLN step on the residual stream: apply the pending gated
/// residual `h' = h + tanh(gate) * y` (if any), then `x = LayerNorm(h') *
/// (scale + 1)` (no-affine LayerNorm). Returns `(h', x)`: `h'` is the stream
/// the next residual adds to, `x` the sub-layer input. Every gated residual
/// in the DiT is consumed by exactly such a norm (the mid-block one by
/// `scale2`, the block-final one by the next block's `scale1` or by
/// `norm_out`), so under `fusednorm` both are one CTA-per-row kernel that never
/// re-reads `h'` (`qwen-image-rs-residual-norm-fusion`); otherwise the candle ops.
fn residual_norm_mod(
    h: &Tensor,
    residual: Option<(&Modulation, &Tensor)>,
    scale: &Modulation,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "fusednorm")]
    {
        crate::fusednorm::fused_residual_norm_mod(
            h,
            residual.map(|(g, y)| (g.fused(), y)),
            scale.fused(),
            eps as f32,
        )
    }
    #[cfg(not(feature = "fusednorm"))]
    {
        let h = match residual {
            Some((g, y)) => (h + g.tokens.tanh()?.broadcast_mul(y)?)?,
            None => h.clone(),
        };
        let x = norm_no_affine(&h, eps)?.broadcast_mul(&(&scale.tokens + 1.0)?)?;
        Ok((h, x))
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
    in_layer: QLinear,
    out_layer: QLinear,
}

impl TextProjection {
    fn load(ctx_dim: usize, opts: LinearOpts, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: ZeroCenterRmsNorm::load(ctx_dim, 1e-6, vb.pp("text_norm"))?,
            in_layer: QLinear::load(ctx_dim, INNER, opts, vb.pp("in_layer"))?,
            out_layer: QLinear::load(INNER, INNER, opts, vb.pp("out_layer"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.in_layer.forward(&x)?;
        let x = x.gelu()?; // tanh approximation (matches GELU(approximate="tanh"))
        self.out_layer.forward(&x)
    }
}

/// Sinusoidal timestep embedding (256) -> MLP(256->INNER, silu, INNER->INNER).
struct TimestepEmbed {
    linear1: QLinear,
    linear2: QLinear,
    freqs: Tensor, // (128,)
}

impl TimestepEmbed {
    fn load(opts: LinearOpts, vb: VarBuilder, dev: &Device) -> Result<Self> {
        let half = 128usize;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp())
            .collect();
        Ok(Self {
            linear1: QLinear::load(256, INNER, opts, vb.pp("timestep_embedder").pp("linear_1"))?,
            linear2: QLinear::load(
                INNER,
                INNER,
                opts,
                vb.pp("timestep_embedder").pp("linear_2"),
            )?,
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
        self.linear2.forward(&x)
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

/// How an attention block projects q/k/v from its input.
enum QkvProj {
    /// Three linears (bf16, Q8_0, or ConvRot if they could not be merged).
    Separate {
        to_q: QLinear,
        to_k: QLinear,
        to_v: QLinear,
    },
    /// ConvRot (`qwen-image-rs-gemm-merge-tune`): `x` is rotated + quantized
    /// once; q|k is ONE head-interleaved INT8 GEMM (`[q_h, k_h]` per head,
    /// N = 2 * INNER) and v a second GEMM on the same quantized `x` (its own
    /// f16 epilogue for the sage P·V). Built at load from the per-layer
    /// weights; outputs are bit-identical to the separate linears.
    #[cfg(feature = "convrot")]
    Merged {
        qk: crate::convrot::ConvRotLinear,
        v: crate::convrot::ConvRotLinear,
    },
}

struct Attention {
    qkv: QkvProj,
    to_out: QLinear,
    norm_q: HeadRmsNorm,
    norm_k: HeadRmsNorm,
}

impl Attention {
    fn load(opts: LinearOpts, vb: VarBuilder) -> Result<Self> {
        let to_q = QLinear::load(INNER, INNER, opts, vb.pp("to_q"))?;
        let to_k = QLinear::load(INNER, INNER, opts, vb.pp("to_k"))?;
        let to_v = QLinear::load(INNER, INNER, opts, vb.pp("to_v"))?;
        let qkv = match (to_q, to_k, to_v) {
            #[cfg(feature = "convrot")]
            (QLinear::Convrot(q), QLinear::Convrot(k), QLinear::Convrot(v)) => QkvProj::Merged {
                qk: crate::convrot::ConvRotLinear::interleave_heads(&q, &k, HEAD_DIM)?,
                v,
            },
            (to_q, to_k, to_v) => QkvProj::Separate { to_q, to_k, to_v },
        };
        Ok(Self {
            qkv,
            to_out: QLinear::load(INNER, INNER, opts, vb.pp("to_out").pp("0"))?,
            norm_q: HeadRmsNorm::load(vb.pp("norm_q"))?,
            norm_k: HeadRmsNorm::load(vb.pp("norm_k"))?,
        })
    }

    /// The block's ConvRot linear count (load accounting; a merged q|k counts 2).
    fn convrot_count(&self) -> usize {
        let qkv = match &self.qkv {
            QkvProj::Separate { to_q, to_k, to_v } => {
                [to_q, to_k, to_v].iter().filter(|l| l.is_convrot()).count()
            }
            #[cfg(feature = "convrot")]
            QkvProj::Merged { .. } => 3,
        };
        qkv + usize::from(self.to_out.is_convrot())
    }

    /// `x (B,S,INNER)` -> `(qh, kh, v)`: q and k per-head RMS-normalized,
    /// pre-RoPE, `(B,S,H,D)` bf16; v `(B,S,H,D)`, f16 when `v_f16` (the sage
    /// P·V operand, born f16 in the ConvRot epilogue) else bf16.
    fn project(&self, x: &Tensor, v_f16: bool) -> Result<(Tensor, Tensor, Tensor)> {
        match &self.qkv {
            QkvProj::Separate { .. } => {
                let (q, k, v) = self.project_raw(x, v_f16)?;
                Ok((self.norm_q.forward(&q)?, self.norm_k.forward(&k)?, v))
            }
            #[cfg(feature = "convrot")]
            QkvProj::Merged { qk, v } => {
                let (b, s, _) = x.dims3()?;
                let shape = (b, s, HEADS, HEAD_DIM);
                let (qk_out, v) = Self::merged_qk_v(qk, v, x, v_f16)?;
                // Rows of [q_h | k_h]: q and k are ld = 2*HEAD_DIM column views.
                let rows = qk_out.reshape((b * s * HEADS, 2 * HEAD_DIM))?;
                let qh = self
                    .norm_q
                    .forward(&rows.narrow(1, 0, HEAD_DIM)?)?
                    .reshape(shape)?;
                let kh = self
                    .norm_k
                    .forward(&rows.narrow(1, HEAD_DIM, HEAD_DIM)?)?
                    .reshape(shape)?;
                Ok((qh, kh, v))
            }
        }
    }

    /// [`Self::project`] WITHOUT the q/k RMSNorm: raw pre-RoPE q, k `(B,S,H,D)`
    /// bf16 views (merged ConvRot: zero-copy column views of the head-interleaved
    /// q|k output, strides `(S*2*INNER, 2*INNER, 2*HEAD_DIM, 1)`), v as in
    /// `project`. For the SA2 quant, which applies the norm itself.
    fn project_raw(&self, x: &Tensor, v_f16: bool) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let shape = (b, s, HEADS, HEAD_DIM);
        match &self.qkv {
            QkvProj::Separate { to_q, to_k, to_v } => {
                let q = to_q.forward(x)?.reshape(shape)?;
                let k = to_k.forward(x)?.reshape(shape)?;
                #[cfg(feature = "sage")]
                let v = if v_f16 {
                    to_v.forward_f16(x)?
                } else {
                    to_v.forward(x)?
                };
                #[cfg(not(feature = "sage"))]
                let v = {
                    let _ = v_f16;
                    to_v.forward(x)?
                };
                Ok((q, k, v.reshape(shape)?))
            }
            #[cfg(feature = "convrot")]
            QkvProj::Merged { qk, v } => {
                let (qk_out, v) = Self::merged_qk_v(qk, v, x, v_f16)?;
                let qk4 = qk_out.reshape((b, s, HEADS, 2 * HEAD_DIM))?;
                Ok((
                    qk4.narrow(3, 0, HEAD_DIM)?,
                    qk4.narrow(3, HEAD_DIM, HEAD_DIM)?,
                    v,
                ))
            }
        }
    }

    /// Merged ConvRot projection: `x` quantized once; the head-interleaved q|k
    /// GEMM `(B,S,2*INNER)` and v `(B,S,H,D)` (f16 when `v_f16`).
    #[cfg(feature = "convrot")]
    fn merged_qk_v(
        qk: &crate::convrot::ConvRotLinear,
        v: &crate::convrot::ConvRotLinear,
        x: &Tensor,
        v_f16: bool,
    ) -> Result<(Tensor, Tensor)> {
        use crate::convrot::EpilogueOut;
        let (b, s, _) = x.dims3()?;
        let qa = qk.quantize(x)?;
        let qk_out = qk.forward_quantized(&qa, EpilogueOut::Bf16)?; // (B,S,2*INNER)
        let v_ty = if v_f16 {
            EpilogueOut::F16
        } else {
            EpilogueOut::Bf16
        };
        let v = v
            .forward_quantized(&qa, v_ty)?
            .reshape((b, s, HEADS, HEAD_DIM))?;
        Ok((qk_out, v))
    }

    /// Whether the q/k RMSNorm runs inside the attention quant instead of as
    /// its own kernel (`qwen-image-rs-qk-norm-fusion`): SA2 selected, in a
    /// `sage2` + `fusednorm` build (the fused norm is bit-identical to the
    /// fusednorm kernel, not to the candle fallback). v1 / non-sage keep the
    /// standalone norm.
    #[cfg(feature = "sage")]
    fn qk_norm_fused() -> Result<bool> {
        #[cfg(all(feature = "sage2", feature = "fusednorm"))]
        {
            Ok(matches!(
                crate::sage2::attention_impl()?,
                crate::sage2::AttentionImpl::Sage2(_)
            ))
        }
        #[cfg(not(all(feature = "sage2", feature = "fusednorm")))]
        {
            Ok(false)
        }
    }

    // One of two cfg-selected bodies binds `out`; clippy sees only one and
    // would ask to inline it.
    #[allow(clippy::let_and_return)]
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        txt_len: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        #[cfg(feature = "sage")]
        let out = {
            let _ = mask; // block-causal structure is expressed via narrows, not a mask
                          // BSHD-native: q/k/v stay (B,S,H,D). RoPE is fused into the
                          // sage INT8 quantizer (attend_bshd), so q/k are rotated and
                          // quantized in one pass — no separate rope kernel, no
                          // rotated-bf16 round trip, no transpose copies.
                          // SA2 (+fusednorm): q/k stay RAW; the quant applies the per-head norm.
            let fused = Self::qk_norm_fused()?;
            let (qh, kh, vv) = if fused {
                self.project_raw(x, true)?
            } else {
                self.project(x, true)?
            }; // (B,S,H,D), pre-rope; v f16
            let out = self.attend_bshd(&qh, &kh, &vv, cos, sin, txt_len, scale, fused)?; // (B,S,H,D)
            self.to_out.forward(&out.reshape((b, s, INNER))?)
        };
        #[cfg(not(feature = "sage"))]
        let out = {
            // (B,H,S,D) for RoPE (rope_i rotates per position across heads).
            let (qh, kh, vv) = self.project(x, false)?; // (B,S,H,D)
            let qh = qh.transpose(1, 2)?.contiguous()?;
            let kh = kh.transpose(1, 2)?.contiguous()?;
            let qh = candle_nn::rotary_emb::rope_i(&qh, cos, sin)?;
            let kh = candle_nn::rotary_emb::rope_i(&kh, cos, sin)?;
            let out = self.attend(&qh, &kh, &vv, mask, txt_len, scale)?; // (B,S,H,D)
            self.to_out.forward(&out.reshape((b, s, INNER))?)
        };
        out // the cfg-selected body above (sage: BSHD-native; else BHSD)
    }

    /// With `sage`: BSHD-native block-causal SageAttention (INT8-QK / FP16-PV);
    /// with `sage2`, SageAttention2 (`crate::sage2::attend_block_causal`, same
    /// split) unless `QIR_SAGE=1`.
    /// `qh,kh` are PRE-rope `(B,S,H,D)`; `vf` is `(B,S,H,D)` f16; `cos,sin` are the
    /// joint-sequence `(S, D/2)` RoPE tables. Image queries attend non-causally
    /// to the whole joint sequence; the `txt_len` text queries attend causally to
    /// the text prefix. Each attention operand is rotated + INT8-quantized by ONE
    /// fused kernel on its own S-axis narrow (with the matching cos/sin rows), so
    /// the per-block scales start at that call's first token — exactly the
    /// alignment the attention kernel's per-block scale indexing expects (k is
    /// quantized twice: text prefix and full, as before). `norm_fused`: `qh,kh`
    /// are RAW projections and SA2's quant applies `norm_q`/`norm_k` itself
    /// (only when [`Self::qk_norm_fused`]). Returns `(B,S,H,D)`.
    #[cfg(feature = "sage")]
    #[allow(clippy::too_many_arguments)]
    fn attend_bshd(
        &self,
        qh: &Tensor,
        kh: &Tensor,
        vf: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        txt_len: usize,
        scale: f64,
        norm_fused: bool,
    ) -> Result<Tensor> {
        use crate::sage::{rope_quant_bshd, sage_attention_quantized, QkRole};
        let (_b, s, _h, _d) = qh.dims4()?;
        let img = s - txt_len;
        // V arrives f16 (to_v.forward_f16 — born f16 in the ConvRot epilogue);
        // the narrows below are zero-copy S-axis views the bridge reads via
        // start_offset + strides.
        debug_assert_eq!(vf.dtype(), DType::F16, "attend_bshd: V must be f16");
        let sc = scale as f32;
        // sage2 build: SageAttention2 unless QIR_SAGE=1 selects v1 (fallback / A/B).
        #[cfg(feature = "sage2")]
        if let crate::sage2::AttentionImpl::Sage2(accum) = crate::sage2::attention_impl()? {
            let norm = norm_fused.then_some(crate::sage2::QkNorm {
                wq: &self.norm_q.weight,
                wk: &self.norm_k.weight,
                eps: self.norm_q.eps as f32,
            });
            if norm_fused && self.norm_k.eps != self.norm_q.eps {
                anyhow::bail!("attend_bshd: fused q/k norm needs one eps for q and k");
            }
            return crate::sage2::attend_block_causal(
                qh, kh, vf, cos, sin, txt_len, sc, accum, norm,
            );
        }
        if norm_fused {
            anyhow::bail!("attend_bshd: raw q/k reached the v1 path (norm not applied)");
        }
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
        let oi = sage_attention_quantized(&qi, &kf, vf, sc, false)?; // (B,img,H,D)
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

/// How the SwiGLU MLP projects its gate and proj inputs.
enum MlpIn {
    /// Two linears (bf16, Q8_0, or unmerged ConvRot).
    Separate { proj: QLinear, gate: QLinear },
    /// ConvRot (`qwen-image-rs-gemm-merge-tune`): ONE INT8 GEMM with the
    /// gate rows then the proj rows (N = 2 * 3 * INNER), one activation quant;
    /// its two column halves feed the out linear's fused SwiGLU quantizer.
    #[cfg(feature = "convrot")]
    Merged(crate::convrot::ConvRotLinear),
}

struct SwiGlu {
    input: MlpIn,
    out: QLinear,
}
impl SwiGlu {
    fn load(opts: LinearOpts, vb: VarBuilder) -> Result<Self> {
        let proj = QLinear::load(INNER, INNER * 3, opts, vb.pp("proj"))?;
        let gate = QLinear::load(INNER, INNER * 3, opts, vb.pp("gate_layer"))?;
        let out = QLinear::load(INNER * 3, INNER, opts, vb.pp("out"))?;
        let input = match (gate, proj, &out) {
            #[cfg(feature = "convrot")]
            (QLinear::Convrot(g), QLinear::Convrot(p), QLinear::Convrot(_)) => {
                MlpIn::Merged(crate::convrot::ConvRotLinear::concat_out(&[&g, &p])?)
            }
            (gate, proj, _) => MlpIn::Separate { proj, gate },
        };
        Ok(Self { input, out })
    }

    /// The MLP's ConvRot linear count (load accounting; merged gate|proj counts 2).
    fn convrot_count(&self) -> usize {
        let input = match &self.input {
            MlpIn::Separate { proj, gate } => {
                usize::from(proj.is_convrot()) + usize::from(gate.is_convrot())
            }
            #[cfg(feature = "convrot")]
            MlpIn::Merged(_) => 2,
        };
        input + usize::from(self.out.is_convrot())
    }

    /// `out(silu(gate(x)) * proj(x))`. Under ConvRot the out linear's
    /// activation quantizer computes `silu(g) * p` itself (f32, in registers),
    /// so the product is never stored; with the merged gate|proj GEMM, g and p
    /// are its two column halves, read in place. Otherwise the candle ops.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.input {
            #[cfg(feature = "convrot")]
            MlpIn::Merged(gp) => {
                let QLinear::Convrot(out) = &self.out else {
                    anyhow::bail!("SwiGlu: merged gate|proj needs a ConvRot out linear");
                };
                let dims = x.dims().to_vec();
                let m: usize = dims[..dims.len() - 1].iter().product();
                let n = INNER * 3;
                let y = gp.forward(x)?.reshape((m, 2 * n))?; // [gate | proj]
                let h = out.forward_swiglu(
                    &y.narrow(1, 0, n)?,
                    &y.narrow(1, n, n)?,
                    crate::convrot::EpilogueOut::Bf16,
                )?; // (m, INNER)
                let mut out_dims = dims[..dims.len() - 1].to_vec();
                out_dims.push(INNER);
                Ok(h.reshape(out_dims)?)
            }
            MlpIn::Separate { proj, gate } => {
                #[cfg(feature = "convrot")]
                if let QLinear::Convrot(out) = &self.out {
                    let (g, p) = (gate.forward(x)?, proj.forward(x)?);
                    return out.forward_swiglu(&g, &p, crate::convrot::EpilogueOut::Bf16);
                }
                let g = silu(&gate.forward(x)?)?;
                self.out.forward(&(g * proj.forward(x)?)?)
            }
        }
    }
}

struct Block {
    attn: Attention,
    mlp: SwiGlu,
    eps: f64,
}

impl Block {
    fn load(opts: LinearOpts, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(opts, vb.pp("attn"))?,
            mlp: SwiGlu::load(opts, vb.pp("img_mlp"))?,
            eps: 1e-6,
        })
    }

    /// One transformer block on the residual stream `h`. `pending` is the
    /// previous block's MLP output, whose gated residual (`gate2`) is applied
    /// here, fused with this block's first norm. Returns `(h, mlp_out)`: the
    /// stream after the attention residual and this block's MLP output, whose
    /// residual the next block (or `norm_out`) applies.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        h: &Tensor,
        pending: Option<&Tensor>,
        mods: &BlockMods,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        txt_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (h, x) =
            residual_norm_mod(h, pending.map(|y| (&mods.gate2, y)), &mods.scale1, self.eps)?;
        let attn = self.attn.forward(&x, mask, cos, sin, txt_len)?;
        let (h, x) = residual_norm_mod(&h, Some((&mods.gate1, &attn)), &mods.scale2, self.eps)?;
        let m = self.mlp.forward(&x)?;
        Ok((h, m))
    }
}

/// The modulation every block shares (one `modulation` linear per forward).
struct BlockMods {
    scale1: Modulation,
    gate1: Modulation,
    scale2: Modulation,
    gate2: Modulation,
}

/// Full DiT.
pub struct QwenImageDit {
    img_in: QLinear,
    txt_in: TextProjection,
    time_embed: TimestepEmbed,
    modulation: QLinear, // INNER -> 4*INNER (after silu)
    blocks: Vec<Block>,
    norm_out_linear: QLinear, // INNER -> INNER
    proj_out: QLinear,        // INNER -> out_channels
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
        let opts = LinearOpts {
            quant,
            convrot,
            rot: rot.as_ref(),
        };
        let mut blocks = Vec::with_capacity(num_layers);
        let vb_b = vb.pp("transformer_blocks");
        for i in 0..num_layers {
            blocks.push(Block::load(opts, vb_b.pp(i))?);
        }
        let inv_freqs = std::array::from_fn(|a| {
            let d = AXES[a];
            (0..d / 2)
                .map(|j| 1f32 / ROPE_THETA.powf(2.0 * j as f64 / d as f64) as f32)
                .collect()
        });
        let dit = Self {
            img_in: QLinear::load(64, INNER, opts, vb.pp("img_in"))?,
            txt_in: TextProjection::load(4096, opts, vb.pp("txt_in"))?,
            time_embed: TimestepEmbed::load(opts, vb.pp("time_text_embed"), &dev)?,
            modulation: QLinear::load(INNER, 4 * INNER, opts, vb.pp("modulation").pp("1"))?,
            blocks,
            norm_out_linear: QLinear::load(INNER, INNER, opts, vb.pp("norm_out").pp("linear"))?,
            proj_out: QLinear::load(INNER, out_channels, opts, vb.pp("proj_out"))?,
            device: dev,
            inv_freqs,
        };
        tracing::info!(
            convrot_linears = dit.convrot_linears(),
            total_linears = BLOCK_LINEARS.len() * dit.blocks.len() + TAIL_LINEARS.len(),
            "DiT linears loaded"
        );
        Ok(dit)
    }

    /// How many linears run ConvRot INT8.
    fn convrot_linears(&self) -> usize {
        let tail = [
            &self.img_in,
            &self.txt_in.in_layer,
            &self.txt_in.out_layer,
            &self.time_embed.linear1,
            &self.time_embed.linear2,
            &self.modulation,
            &self.norm_out_linear,
            &self.proj_out,
        ];
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| b.attn.convrot_count() + b.mlp.convrot_count())
            .sum();
        tail.into_iter().filter(|l| l.is_convrot()).count() + blocks
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
        // Per-token selection (image tokens row 0 = real t, text row 1 = t=0)
        // happens inside the AdaLN step (see `Modulation`).
        let md = |m: &Tensor| Modulation::new(m.clone(), b, txt_len, img_tokens);
        let mods = BlockMods {
            scale1: md(&mods[0])?,
            gate1: md(&mods[1])?,
            scale2: md(&mods[2])?,
            gate2: md(&mods[3])?,
        };

        // The dense block-causal mask is only needed by the naive path; the
        // flash and sage paths derive the same structure from the text/image split.
        #[cfg(all(not(feature = "flash-attn"), not(feature = "sage")))]
        let mask = block_causal_mask(txt_len, img_tokens, dtype, &self.device)?;
        #[cfg(any(feature = "flash-attn", feature = "sage"))]
        let mask = Tensor::zeros((1, 1, 1, 1), dtype, &self.device)?;
        let (cos, sin) = self.rope_cos_sin(txt_len, h, w, dtype)?;

        let mut h = joint;
        let mut pending: Option<Tensor> = None;
        for block in &self.blocks {
            let (nh, y) = block.forward(&h, pending.as_ref(), &mods, &mask, &cos, &sin, txt_len)?;
            h = nh;
            pending = Some(y);
        }

        // norm_out: AdaLayerNorm scale-only (fused with the last block's MLP
        // residual), then proj_out. Scale from temb rows.
        let scale_out = self.norm_out_linear.forward(&silu(&temb)?)?; // (2, INNER)
        let scale_out = Modulation::new(scale_out, b, txt_len, img_tokens)?;
        let (_, x) = residual_norm_mod(
            &h,
            pending.as_ref().map(|y| (&mods.gate2, y)),
            &scale_out,
            1e-6,
        )?;
        let out = self.proj_out.forward(&x)?; // (1, seq, out_channels)
        let _ = seq;
        Ok(out)
    }
}

/// Select modulation rows per token: `(2, INNER)` -> `(1, seq, INNER)`, image
/// tokens take row 0 (real timestep), text tokens row 1 (t=0).
#[cfg(not(feature = "fusednorm"))]
fn select_rows(m: &Tensor, txt_len: usize, img_tokens: usize) -> Result<Tensor> {
    let real = m.narrow(0, 0, 1)?; // (1, INNER)
    let zero = m.narrow(0, 1, 1)?; // (1, INNER)
    let text_rows = zero.broadcast_as((txt_len, m.dim(1)?))?;
    let img_rows = real.broadcast_as((img_tokens, m.dim(1)?))?;
    let rows = Tensor::cat(&[&text_rows, &img_rows], 0)?.unsqueeze(0)?; // (1, seq, INNER)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The candle fallback of the AdaLN step equals the pre-fusion pair
    /// (`h + tanh(gate) * y`, then `norm_no_affine(h') * (scale + 1)`) on the
    /// per-token tensors, and the per-token selection puts row 1 on the text
    /// tokens of every lane.
    #[cfg(not(feature = "fusednorm"))]
    #[test]
    fn residual_norm_mod_fallback_matches_the_unfused_pair() {
        let dev = Device::Cpu;
        let (b, txt, img, n) = (2usize, 3usize, 5usize, 16usize);
        let rows = |seed: f32| {
            Tensor::arange(0f32, (2 * n) as f32, &dev)
                .unwrap()
                .affine(0.05, seed as f64)
                .unwrap()
                .sin()
                .unwrap()
                .reshape((2, n))
                .unwrap()
        };
        let (g, sc) = (rows(0.3), rows(1.7));
        let gm = Modulation::new(g.clone(), b, txt, img).unwrap();
        let sm = Modulation::new(sc.clone(), b, txt, img).unwrap();
        let tok = gm.tokens.to_vec3::<f32>().unwrap();
        let (g0, g1) = (
            g.get(0).unwrap().to_vec1::<f32>().unwrap(),
            g.get(1).unwrap().to_vec1::<f32>().unwrap(),
        );
        for lane in &tok {
            for (pos, row) in lane.iter().enumerate() {
                assert_eq!(row, if pos < txt { &g1 } else { &g0 }, "pos {pos}");
            }
        }
        let h = Tensor::randn(0f32, 2f32, (b, txt + img, n), &dev).unwrap();
        let y = Tensor::randn(0f32, 1f32, (b, txt + img, n), &dev).unwrap();
        let (h2, x2) = residual_norm_mod(&h, Some((&gm, &y)), &sm, 1e-6).unwrap();
        let h_ref = (&h + gm.tokens.tanh().unwrap().broadcast_mul(&y).unwrap()).unwrap();
        let x_ref = norm_no_affine(&h_ref, 1e-6)
            .unwrap()
            .broadcast_mul(&(&sm.tokens + 1.0).unwrap())
            .unwrap();
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v(&h2), v(&h_ref));
        assert_eq!(v(&x2), v(&x_ref));
        let (h3, x3) = residual_norm_mod(&h, None, &sm, 1e-6).unwrap();
        assert_eq!(v(&h3), v(&h));
        let x_ref = norm_no_affine(&h, 1e-6)
            .unwrap()
            .broadcast_mul(&(&sm.tokens + 1.0).unwrap())
            .unwrap();
        assert_eq!(v(&x3), v(&x_ref));
    }

    #[test]
    fn rotated_tail_linears_meet_the_convrot_shape_rules() {
        for (prefix, k, n, rot) in TAIL_LINEARS {
            if rot {
                assert!(
                    k.is_multiple_of(crate::model::rotation::GROUP),
                    "{prefix}: K={k}"
                );
                assert!(n.is_multiple_of(8), "{prefix}: N={n}");
            }
        }
    }

    #[test]
    fn policy_tag_tracks_every_table_entry() {
        let base = policy_tag(&BLOCK_LINEARS, &TAIL_LINEARS);
        assert_eq!(base, convrot_policy_tag());
        assert!(base.starts_with(&format!("qir-convrot-v{PREQUANT_FORMAT};")));
        // Flipping any one tail flag changes the tag.
        for i in 0..TAIL_LINEARS.len() {
            let mut t = TAIL_LINEARS;
            t[i].3 = !t[i].3;
            assert_ne!(policy_tag(&BLOCK_LINEARS, &t), base, "{}", t[i].0);
        }
        // Dropping a block linear changes it too.
        assert_ne!(policy_tag(&BLOCK_LINEARS[1..], &TAIL_LINEARS), base);
    }

    #[test]
    fn img_in_cannot_rotate() {
        assert_eq!(
            linear_precision("img_in", true, true),
            Some(LinearPrecision::Full)
        );
        assert!(!is_convrot_target("img_in"));
    }

    #[test]
    fn block_linears_follow_convrot_then_quant() {
        let p = "transformer_blocks.31.img_mlp.gate_layer";
        assert_eq!(
            linear_precision(p, true, true),
            Some(LinearPrecision::ConvRot)
        );
        assert_eq!(linear_precision(p, true, false), Some(LinearPrecision::Q8));
        assert_eq!(
            linear_precision(p, false, false),
            Some(LinearPrecision::Full)
        );
        assert!(is_convrot_target("transformer_blocks.0.attn.to_out.0"));
    }

    #[test]
    fn tail_linears_never_take_q8() {
        for (prefix, ..) in TAIL_LINEARS {
            assert_ne!(
                linear_precision(prefix, true, false),
                Some(LinearPrecision::Q8)
            );
        }
    }

    #[test]
    fn non_linears_are_outside_the_policy() {
        for p in [
            "transformer_blocks.0.attn.norm_q",
            "transformer_blocks.x.attn.to_q",
            "transformer_blocks..attn.to_q",
            "txt_in.text_norm",
            "modulation",
        ] {
            assert_eq!(linear_precision(p, true, true), None, "{p}");
            assert!(!is_convrot_target(p));
        }
    }
}
