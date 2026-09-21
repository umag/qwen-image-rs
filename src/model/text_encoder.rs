//! `Qwen3VLForConditionalGeneration` text encoder (text-only path), ported to
//! candle. **Phase 3.** For text-to-image the vision tower and DeepStack are
//! unused, so the backbone is a standard Qwen3 decoder: 36 layers, GQA (32 Q /
//! 8 KV heads, head_dim 128), per-head q/k RMSNorm, SwiGLU MLP, RMSNorm, RoPE
//! (theta 5e6). mRoPE collapses to 1D RoPE when every position id is the token
//! index (text-only), so plain RoPE is exact here.
//!
//! The pipeline extracts `hidden_states[-1]` — the last decoder layer's output
//! **before** the final RMSNorm (it hooks the norm to return its input) — for
//! all positions, then drops the leading system-prompt tokens. So this module
//! runs the layers and returns the pre-final-norm hidden states; it does NOT
//! apply the final norm or the lm_head.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, linear_b, rms_norm, Activation, Embedding, Linear, Module, RmsNorm, VarBuilder,
};

use crate::Result;

/// A no-bias linear, full-precision or Q8_0-quantized (GGUF, weight-only). Q8_0
/// halves the weight VRAM near-losslessly so the 8B text encoder fits alongside
/// the DiT and VAE, resident in 24 GB.
enum QLinear {
    Full(Linear),
    Quant(QMatMul),
}

impl QLinear {
    fn load(in_c: usize, out_c: usize, quant: bool, vb: VarBuilder) -> Result<Self> {
        if quant {
            let w = vb.get((out_c, in_c), "weight")?;
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            Ok(QLinear::Quant(QMatMul::from_qtensor(qt)?))
        } else {
            Ok(QLinear::Full(linear_b(in_c, out_c, false, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Full(l) => Ok(l.forward(x)?),
            QLinear::Quant(q) => Ok(q.forward(x)?),
        }
    }
}

/// Qwen3-VL text config (from `text_encoder/config.json` `text_config`).
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            max_position_embeddings: 8192,
        }
    }
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, dev: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        // cos/sin must match the activation dtype for candle's rope kernel.
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to `(B, H, S, D)` q and k, positions `0..S`.
    fn apply(&self, q: &Tensor, k: &Tensor, seq: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

fn repeat_kv(x: Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x);
    }
    let (b, kvh, s, d) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .broadcast_as((b, kvh, groups, s, d))?
        .reshape((b, kvh * groups, s, d))?;
    Ok(x)
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    groups: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        Ok(Self {
            q_proj: QLinear::load(h, nh * hd, quant, vb.pp("q_proj"))?,
            k_proj: QLinear::load(h, nkv * hd, quant, vb.pp("k_proj"))?,
            v_proj: QLinear::load(h, nkv * hd, quant, vb.pp("v_proj"))?,
            o_proj: QLinear::load(nh * hd, h, quant, vb.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            groups: nh / nkv,
            scale: 1.0 / (hd as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?;
        // per-head RMSNorm on q,k (applied over head_dim), then to (B,H,S,D)
        let q = q.apply(&self.q_norm)?.transpose(1, 2)?;
        let k = k.apply(&self.k_norm)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (q, k) = rope.apply(&q, &k, s)?;
        let k = repeat_kv(k, self.groups)?.contiguous()?;
        let v = repeat_kv(v, self.groups)?.contiguous()?;
        let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // (B,H,S,D)
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: QLinear::load(h, i, quant, vb.pp("gate_proj"))?,
            up: QLinear::load(h, i, quant, vb.pp("up_proj"))?,
            down: QLinear::load(i, h, quant, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.apply(&Activation::Silu)?;
        let rhs = self.up.forward(xs)?;
        Ok(self.down.forward(&(lhs * rhs)?)?)
    }
}

struct DecoderLayer {
    attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(cfg, quant, vb.pp("self_attn"))?,
            mlp: Mlp::load(cfg, quant, vb.pp("mlp"))?,
            input_ln: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.attn.forward(&xs, mask, rope)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let h = self.mlp.forward(&xs.apply(&self.post_attn_ln)?)?;
        Ok((residual + h)?)
    }
}

/// Qwen3-VL text encoder producing pre-final-norm hidden states.
pub struct QwenTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rope: RotaryEmbedding,
    device: Device,
}

impl QwenTextEncoder {
    /// Load the text encoder. When `quant` is set, the decoder linears are Q8_0
    /// (weight-only) — ~half the VRAM, near-lossless — so all three models stay
    /// resident in 24 GB. The token embedding stays full precision.
    pub fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rope = RotaryEmbedding::new(cfg, vb.device(), vb.dtype())?;
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::load(cfg, quant, vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rope,
            device: vb.device().clone(),
        })
    }

    /// Load from a pre-quantized GGUF (written by `prequantize-text`): linears
    /// come back as Q8_0 `QMatMul` directly, norms/embedding are dequantized —
    /// so NO bf16 weight tensor is ever created on the GPU (candle's CUDA pool
    /// stays small, unlike quantize-on-load). Fixes the resident-mode VRAM peak.
    pub fn load_gguf(
        cfg: &TextConfig,
        qvb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let dev = qvb.device().clone();
        let dt = DType::BF16;
        let vb_m = qvb.pp("model").pp("language_model");
        let embed_w = vb_m
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?
            .dequantize(&dev)?
            .to_dtype(dt)?;
        let embed_tokens = Embedding::new(embed_w, cfg.hidden_size);
        let rope = RotaryEmbedding::new(cfg, &dev, dt)?;
        let rms = |dim: usize, vb: &candle_transformers::quantized_var_builder::VarBuilder| -> Result<RmsNorm> {
            let w = vb.get(dim, "weight")?.dequantize(&dev)?.to_dtype(dt)?;
            Ok(RmsNorm::new(w, cfg.rms_norm_eps))
        };
        let qlin = |ic: usize, oc: usize, vb: &candle_transformers::quantized_var_builder::VarBuilder| -> Result<QLinear> {
            Ok(QLinear::Quant(candle_core::quantized::QMatMul::from_arc(
                vb.get((oc, ic), "weight")?,
            )?))
        };
        let (nh, nkv, hd, h, ii) = (
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.hidden_size,
            cfg.intermediate_size,
        );
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_layers {
            let vb = vb_l.pp(i);
            let sa = vb.pp("self_attn");
            let attn = Attention {
                q_proj: qlin(h, nh * hd, &sa.pp("q_proj"))?,
                k_proj: qlin(h, nkv * hd, &sa.pp("k_proj"))?,
                v_proj: qlin(h, nkv * hd, &sa.pp("v_proj"))?,
                o_proj: qlin(nh * hd, h, &sa.pp("o_proj"))?,
                q_norm: rms(hd, &sa.pp("q_norm"))?,
                k_norm: rms(hd, &sa.pp("k_norm"))?,
                num_heads: nh,
                num_kv_heads: nkv,
                head_dim: hd,
                groups: nh / nkv,
                scale: 1.0 / (hd as f64).sqrt(),
            };
            let mlp = Mlp {
                gate: qlin(h, ii, &vb.pp("mlp").pp("gate_proj"))?,
                up: qlin(h, ii, &vb.pp("mlp").pp("up_proj"))?,
                down: qlin(ii, h, &vb.pp("mlp").pp("down_proj"))?,
            };
            layers.push(DecoderLayer {
                attn,
                mlp,
                input_ln: rms(h, &vb.pp("input_layernorm"))?,
                post_attn_ln: rms(h, &vb.pp("post_attention_layernorm"))?,
            });
        }
        Ok(Self {
            embed_tokens,
            layers,
            rope,
            device: dev,
        })
    }

    /// Build a causal additive mask `(1, 1, S, S)` (0 on/below diagonal, -inf above).
    fn causal_mask(&self, s: usize, dtype: DType) -> Result<Tensor> {
        let mut data = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                data[i * s + j] = f32::NEG_INFINITY;
            }
        }
        Ok(Tensor::from_vec(data, (1, 1, s, s), &self.device)?.to_dtype(dtype)?)
    }

    /// Run the decoder on `input_ids` `(B, S)` and return pre-final-norm hidden
    /// states `(B, S, hidden)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mask = self.causal_mask(s, xs.dtype())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask, &self.rope)?;
        }
        Ok(xs)
    }
}

/// The prompt-encoding recipe from `QwenImage21Pipeline` (text-to-image).
pub mod prompt {
    /// System prompt prepended to every text-to-image prompt.
    pub const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

    /// Build the exact t2i chat-template string for a user prompt.
    pub fn t2i_template(prompt: &str) -> String {
        let p = if prompt.is_empty() { " " } else { prompt };
        format!(
            "<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}

/// Cosine similarity between two flattened tensors (for validation).
pub fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_has_roles() {
        let t = prompt::t2i_template("a cat");
        assert!(t.contains("<|im_start|>system\nComprehend and analyze"));
        assert!(t.contains("<|im_start|>user\na cat<|im_end|>"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn empty_prompt_becomes_space() {
        assert!(prompt::t2i_template("").contains("user\n <|im_end|>"));
    }

    #[test]
    fn repeat_kv_expands_heads() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 8, 5, 128), DType::F32, &dev)?;
        assert_eq!(repeat_kv(x, 4)?.dims(), &[1, 32, 5, 128]);
        Ok(())
    }

    #[test]
    fn cosine_identical_is_one() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev)?;
        assert!((cosine(&a, &a)? - 1.0).abs() < 1e-5);
        Ok(())
    }
}
