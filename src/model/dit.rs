//! `QwenImage21Transformer2DModel` — ~7B single-stream DiT, 32 layers,
//! mixed-granularity attention (token-level causal mask on text, chunk-level
//! bidirectional on image) with prefix-KV-cache reuse across denoise steps.
//! **Phase 4** (first end-to-end image). bf16 + FlashAttention-2 baseline
//! before FP8 / SageAttention are layered in Phase 5.
//!
//! TODO(phase-4): port 32 single-stream layers, prefix-KV cache, and wire the
//! flow-matching loop; match oracle at bf16.

use crate::model::config::DitConfig;

/// Single-stream DiT. Not yet implemented — see docs/PHASES.md, Phase 4.
#[allow(dead_code)]
pub struct QwenImageDit {
    config: DitConfig,
}

impl QwenImageDit {
    #[allow(dead_code)]
    pub fn new(config: DitConfig) -> Self {
        Self { config }
    }

    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }
}
