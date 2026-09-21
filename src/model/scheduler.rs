//! `FlowMatchEulerDiscreteScheduler` with dynamic shifting. **Phase 4** (wired
//! with the DiT). Deterministic given a fixed seed + step count so the port can
//! be diffed against oracle latents step-by-step.
//!
//! TODO(phase-4): implement sigma schedule, dynamic shift, and the Euler step.

use crate::model::config::SamplerConfig;

/// Flow-matching Euler scheduler. Not yet implemented — see docs/PHASES.md, Phase 4.
#[allow(dead_code)]
pub struct FlowMatchEuler {
    config: SamplerConfig,
}

impl FlowMatchEuler {
    #[allow(dead_code)]
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    #[allow(dead_code)]
    pub fn steps(&self) -> usize {
        self.config.num_inference_steps
    }
}
