//! `AutoencoderKLQwenImage21` — 64-channel RGBA latent, 16x spatial
//! compression, native transparency. **Phase 2** (first component ported;
//! smallest, self-contained). Decode path validated against oracle latents
//! before the encoder/DiT land.
//!
//! TODO(phase-2): port decoder blocks, match oracle `latents -> RGBA image`.

use crate::model::config::VaeConfig;

/// RGBA VAE. Not yet implemented — see docs/PHASES.md, Phase 2.
#[allow(dead_code)]
pub struct QwenImageVae {
    config: VaeConfig,
}

impl QwenImageVae {
    /// Placeholder constructor; real loader lands in Phase 2.
    #[allow(dead_code)]
    pub fn new(config: VaeConfig) -> Self {
        Self { config }
    }

    /// Latent channels this VAE decodes from.
    #[allow(dead_code)]
    pub fn latent_channels(&self) -> usize {
        self.config.latent_channels
    }
}
