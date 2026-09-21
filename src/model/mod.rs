//! Qwen-Image-2.1 model components. Each is ported to candle and validated
//! against the bf16 diffusers oracle before optimization (see docs/PHASES.md).

pub mod config;
pub mod dit;
pub mod rotation;
pub mod scheduler;
pub mod text_encoder;
pub mod vae;
