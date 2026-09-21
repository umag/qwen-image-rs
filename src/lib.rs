//! Rust CUDA inference for **Qwen-Image-2.1** on the RTX 4090 (Ada).
//!
//! Spine: [`candle`](https://github.com/huggingface/candle) for the CUDA
//! backend, safetensors + GGUF loading, and FlashAttention-2. Custom CUDA
//! ops (SageAttention INT8, FP8 GEMM) are FFI'd in behind feature flags and
//! validated against a bf16 diffusers oracle.
//!
//! Pipeline (`QwenImage21Pipeline`):
//! - text encoder: `Qwen3VLForConditionalGeneration` (~8B) — see [`model::text_encoder`]
//! - transformer: `QwenImage21Transformer2DModel` (~7B single-stream DiT, 32 layers) — [`model::dit`]
//! - vae: `AutoencoderKLQwenImage21` (64-ch RGBA, 16x) — [`model::vae`]
//! - scheduler: `FlowMatchEulerDiscreteScheduler` — [`model::scheduler`]

#[cfg(feature = "convrot")]
pub mod convrot;
pub mod device;
#[cfg(feature = "fusednorm")]
pub mod fusednorm;
pub mod loader;
pub mod model;
#[cfg(feature = "sage")]
pub mod sage;

/// Crate result alias.
pub type Result<T> = anyhow::Result<T>;
