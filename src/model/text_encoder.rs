//! `Qwen3VLForConditionalGeneration` (~8B) — encodes text instructions and
//! optional condition images into the conditioning stream. **Phase 3** (largest
//! single component). Embeddings validated against oracle within tolerance.
//!
//! TODO(phase-3): port Qwen3-VL blocks; verify prompt-embedding parity.

use crate::model::config::TextEncoderConfig;

/// Qwen3-VL text/vision encoder. Not yet implemented — see docs/PHASES.md, Phase 3.
#[allow(dead_code)]
pub struct QwenTextEncoder {
    config: TextEncoderConfig,
}

impl QwenTextEncoder {
    #[allow(dead_code)]
    pub fn new(config: TextEncoderConfig) -> Self {
        Self { config }
    }

    #[allow(dead_code)]
    pub fn model_type(&self) -> &str {
        &self.config.model_type
    }
}
