//! Weight loading. Two backends, both first-class:
//!
//! - **safetensors** (bf16 / FP8 e4m3fn) via candle's `VarBuilder` — the
//!   primary path for the bf16 baseline (Phase 4) and FP8 optimization (Phase 5).
//! - **GGUF** (Q4_K / Q5_K / Q8_0, city96-style quants) via candle's built-in
//!   `candle_core::quantized::gguf_file` — the low-VRAM path for the 24 GB 4090.
//!
//! Per-phase implementations build `VarBuilder`s from these; this module only
//! resolves and validates the on-disk weight set so failures surface early.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};

use crate::Result;

/// How a component's weights are stored on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// `.safetensors` shards (bf16 or FP8).
    Safetensors,
    /// `.gguf` quantized single file.
    Gguf,
}

impl WeightFormat {
    /// Infer format from a file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => Some(Self::Safetensors),
            Some("gguf") => Some(Self::Gguf),
            _ => None,
        }
    }
}

/// A resolved set of weight files for one component (VAE / text encoder / DiT).
#[derive(Debug, Clone)]
pub struct WeightSet {
    pub format: WeightFormat,
    pub files: Vec<PathBuf>,
}

impl WeightSet {
    /// Resolve every `.safetensors` / `.gguf` file directly under `dir`.
    /// GGUF takes precedence when both are present (explicit low-VRAM opt-in).
    pub fn resolve(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            bail!("weight directory does not exist: {}", dir.display());
        }
        let mut safet = Vec::new();
        let mut gguf = Vec::new();
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("reading weight dir {}", dir.display()))?
        {
            let path = entry?.path();
            match WeightFormat::from_path(&path) {
                Some(WeightFormat::Safetensors) => safet.push(path),
                Some(WeightFormat::Gguf) => gguf.push(path),
                None => {}
            }
        }
        safet.sort();
        gguf.sort();
        if !gguf.is_empty() {
            Ok(Self {
                format: WeightFormat::Gguf,
                files: gguf,
            })
        } else if !safet.is_empty() {
            Ok(Self {
                format: WeightFormat::Safetensors,
                files: safet,
            })
        } else {
            bail!("no .safetensors or .gguf files in {}", dir.display());
        }
    }

    /// Total on-disk size of the weight set, in bytes.
    pub fn total_bytes(&self) -> Result<u64> {
        let mut total = 0;
        for f in &self.files {
            total += std::fs::metadata(f)
                .with_context(|| format!("stat {}", f.display()))?
                .len();
        }
        Ok(total)
    }
}
