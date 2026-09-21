//! Architecture constants for Qwen-Image-2.1, taken from the real checkpoint
//! configs (snapshot b3179ad3) downloaded in Phase 1:
//! - `transformer/config.json` → [`DitConfig`]
//! - `vae/config.json` → [`VaeConfig`]
//! - `model_index.json` → pipeline wiring
//!
//! Latent geometry confirmed empirically: a 1024x1024 image yields a
//! 64x64x64 f32 latent (16x spatial compression, 64 channels).

use serde::Deserialize;

/// Numeric precision the weights ship in.
pub const NATIVE_DTYPE: &str = "bfloat16";

/// Diffusers pipeline class name (from `model_index.json`).
pub const PIPELINE_CLASS: &str = "QwenImage21Pipeline";

/// Transformer (DiT) shape — `QwenImage21Transformer2DModel`.
/// Single-stream, 32 layers, 32 heads x 128 = 4096 hidden.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    /// `num_layers` — single-stream DiT layers.
    pub num_layers: usize,
    /// `num_attention_heads`.
    pub num_heads: usize,
    /// `attention_head_dim`.
    pub head_dim: usize,
    /// Latent channels in/out (`in_channels` == `out_channels`).
    pub latent_channels: usize,
    /// `context_in_dim` — text-encoder conditioning width (Qwen3-VL hidden).
    pub context_in_dim: usize,
    /// `axes_dims_rope` — 3D RoPE split (temporal, height, width); sums to `head_dim`.
    pub rope_axes: [usize; 3],
    /// `mlp_ratio` — FFN expansion.
    pub mlp_ratio: usize,
    /// `patch_size`.
    pub patch_size: usize,
    /// `causal_condition` — text uses a causal mask, image is bidirectional.
    pub causal_condition: bool,
    /// LayerNorm epsilon.
    pub eps: f64,
}

impl DitConfig {
    /// Hidden size = `num_heads * head_dim` (4096).
    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

impl Default for DitConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            latent_channels: 64,
            context_in_dim: 4096,
            rope_axes: [16, 56, 56],
            mlp_ratio: 3,
            patch_size: 1,
            causal_condition: true,
            eps: 1e-6,
        }
    }
}

/// VAE shape — `AutoencoderKLQwenImage21` (Wan-style residual autoencoder).
/// RGBA in/out (4 channels), 64-channel latent, 16x compression via 4 downsamples.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    /// `in_channels` — 4 (RGBA, native transparency).
    pub in_channels: usize,
    /// Latent channels (`z_dim`).
    pub latent_channels: usize,
    /// `base_dim` — encoder base width.
    pub base_dim: usize,
    /// `decoder_base_dim` — decoder base width.
    pub decoder_base_dim: usize,
    /// `dim_mult` — per-stage width multipliers; (len - 1) downsamples.
    pub dim_mult: Vec<usize>,
    /// `is_residual`.
    pub is_residual: bool,
    /// Effective spatial compression factor (16 = 2^4).
    pub scale_factor: usize,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            latent_channels: 64,
            base_dim: 96,
            decoder_base_dim: 144,
            dim_mult: vec![1, 2, 4, 8, 8],
            is_residual: true,
            scale_factor: 16,
        }
    }
}

/// Text encoder: Qwen3-VL (~8B), encodes text and optional condition images.
/// Hidden width feeds the DiT via `context_in_dim` (4096).
#[derive(Debug, Clone, Deserialize)]
pub struct TextEncoderConfig {
    pub model_type: String,
    pub hidden_size: usize,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: "Qwen3VLForConditionalGeneration".into(),
            hidden_size: 4096,
        }
    }
}

/// Flow-matching sampler defaults (`FlowMatchEulerDiscreteScheduler`, dynamic shift).
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_inference_steps: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 40,
        }
    }
}
