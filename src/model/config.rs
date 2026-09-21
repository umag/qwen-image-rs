//! Architecture constants for Qwen-Image-2.1, sourced from the HF model card
//! and `model_index.json`. Verified against the real checkpoints in Phase 1.

use serde::Deserialize;

/// Numeric precision the weights ship in.
pub const NATIVE_DTYPE: &str = "bfloat16";

/// Diffusers pipeline class name (from `model_index.json`).
pub const PIPELINE_CLASS: &str = "QwenImage21Pipeline";

/// Transformer (DiT) shape. Single-stream, 32 layers, ~7B params.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    /// Number of single-stream DiT layers.
    pub num_layers: usize,
    /// Attention heads.
    pub num_heads: Option<usize>,
    /// Hidden size.
    pub hidden_size: Option<usize>,
}

impl Default for DitConfig {
    fn default() -> Self {
        // Placeholder until confirmed from transformer/config.json in Phase 1.
        Self { num_layers: 32, num_heads: None, hidden_size: None }
    }
}

/// VAE shape: 64-channel RGBA latent, 16x spatial compression, native alpha.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub scale_factor: usize,
    pub rgba: bool,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self { latent_channels: 64, scale_factor: 16, rgba: true }
    }
}

/// Text encoder: Qwen3-VL (~8B), encodes text and optional condition images.
#[derive(Debug, Clone, Deserialize)]
pub struct TextEncoderConfig {
    pub model_type: String,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self { model_type: "Qwen3VLForConditionalGeneration".into() }
    }
}

/// Flow-matching sampler defaults (Euler discrete, dynamic shift).
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_inference_steps: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { num_inference_steps: 40 }
    }
}
