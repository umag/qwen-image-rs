//! `qwen-image-rs` CLI.

use anyhow::Context;
use candle_core::{DType, Tensor};
use clap::{Parser, Subcommand};

use qwen_image_rs::{device, loader::WeightSet, Result};

#[derive(Parser)]
#[command(name = "qwen-image-rs", version, about = "Qwen-Image-2.1 inference in Rust/CUDA")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// GPU/CPU smoke test: allocate, matmul, reduce — proves the backend end to end.
    Smoke {
        /// Square matrix dimension.
        #[arg(long, default_value_t = 1024)]
        n: usize,
    },
    /// Resolve and report a component's on-disk weight set (safetensors or GGUF).
    Info {
        /// Directory holding .safetensors shards or a .gguf file.
        #[arg(long)]
        weights: std::path::PathBuf,
    },
    /// Text-to-image. Not yet implemented — see docs/PHASES.md (Phase 4).
    Generate {
        #[arg(long)]
        prompt: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Smoke { n } => smoke(n),
        Command::Info { weights } => info(&weights),
        Command::Generate { .. } => {
            anyhow::bail!("`generate` lands in Phase 4 (DiT + scheduler). See docs/PHASES.md.")
        }
    }
}

/// Allocate two NxN matrices, matmul, and reduce — the minimal end-to-end
/// exercise of the active candle backend (CPU by default, CUDA under `--features cuda`).
fn smoke(n: usize) -> Result<()> {
    let dev = device::best_device()?;
    tracing::info!(device = device::label(&dev), n, "smoke: allocating");
    let a = Tensor::randn(0f32, 1f32, (n, n), &dev)?.to_dtype(DType::F32)?;
    let b = Tensor::randn(0f32, 1f32, (n, n), &dev)?;
    let start = std::time::Instant::now();
    let c = a.matmul(&b)?;
    let sum = c.sum_all()?.to_scalar::<f32>()?;
    let elapsed = start.elapsed();
    let gflop = 2.0 * (n as f64).powi(3) / 1e9;
    println!(
        "device={} n={n} sum={sum:.3} matmul={:.3}ms ({:.1} GFLOP/s)",
        device::label(&dev),
        elapsed.as_secs_f64() * 1e3,
        gflop / elapsed.as_secs_f64(),
    );
    Ok(())
}

/// Resolve a weight directory and print format + shard count + total size.
fn info(weights: &std::path::Path) -> Result<()> {
    let set = WeightSet::resolve(weights).context("resolving weight set")?;
    let bytes = set.total_bytes()?;
    println!(
        "format={:?} files={} total={:.2} GiB",
        set.format,
        set.files.len(),
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    for f in &set.files {
        println!("  {}", f.display());
    }
    Ok(())
}
