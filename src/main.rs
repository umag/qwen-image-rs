//! `qwen-image-rs` CLI.

use anyhow::Context;
use candle_core::{DType, IndexOp, Tensor};
use clap::{Parser, Subcommand};

use qwen_image_rs::{device, loader::WeightSet, Result};

#[derive(Parser)]
#[command(
    name = "qwen-image-rs",
    version,
    about = "Qwen-Image-2.1 inference in Rust/CUDA"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Where `--convrot` finds its prequantized DiT weights (`convrot_cache`).
#[derive(clap::Args, Clone, Debug)]
struct ConvrotCacheArgs {
    /// Cache dir for the prequantized ConvRot DiT (built once per transformer,
    /// then loaded on every --convrot run). Default: $QIR_CONVROT_CACHE, else
    /// $XDG_CACHE_HOME/qwen-image-rs/convrot, else ~/.cache/qwen-image-rs/convrot.
    #[arg(long)]
    convrot_cache: Option<std::path::PathBuf>,
    /// Do not use the cache: rotate + INT8-quantize the DiT on load.
    #[arg(long, conflicts_with = "rebuild_convrot_cache")]
    no_convrot_cache: bool,
    /// Rebuild this transformer's cache entry, then load it.
    #[arg(long)]
    rebuild_convrot_cache: bool,
}

impl ConvrotCacheArgs {
    fn spec(&self) -> qwen_image_rs::convrot_cache::CacheSpec {
        qwen_image_rs::convrot_cache::CacheSpec::from_cli(
            self.convrot_cache.as_deref(),
            self.no_convrot_cache,
            self.rebuild_convrot_cache,
        )
    }
}

#[derive(Subcommand)]
enum Command {
    /// GPU/CPU smoke test: allocate, matmul, reduce — proves the backend end to end.
    Smoke {
        /// Square matrix dimension.
        #[arg(long, default_value_t = 1024)]
        n: usize,
    },
    /// Self-test the ConvRot INT8 GEMM bridge (candle -> CUTLASS kernel -> candle).
    ConvrotTest,
    /// Time the ConvRot INT8 GEMM + dequant per CUTLASS tile config at the DiT
    /// shapes (separate and merged q/k/v and gate/proj), checking every config
    /// is bit-identical.
    GemmBench {
        /// Timed launches per (shape, config).
        #[arg(long, default_value_t = 30)]
        iters: usize,
        /// Batch size B (M = B * 4117 for the image-sequence shapes).
        #[arg(long, default_value_t = 1)]
        batch: usize,
    },
    /// Self-test the SageAttention INT8-QK/FP16-PV kernel vs an f32 reference.
    SageTest,
    /// Self-test the fused LayerNorm+AdaLN kernel vs the candle reference.
    FusednormTest,
    /// Micro-benchmark the DiT's dominant ops (MLP GEMM vs attention) in bf16.
    Bench {
        #[arg(long, default_value_t = 4117)]
        seq: usize,
        #[arg(long, default_value_t = 30)]
        iters: usize,
    },
    /// Resolve and report a component's on-disk weight set (safetensors or GGUF).
    Info {
        /// Directory holding .safetensors shards or a .gguf file.
        #[arg(long)]
        weights: std::path::PathBuf,
    },
    /// Text-to-image: prompt -> PNG (tokenize -> text encode -> denoise -> VAE).
    Generate {
        /// The model snapshot directory (contains processor/, text_encoder/, transformer/, vae/).
        #[arg(long)]
        model: std::path::PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 1024)]
        size: usize,
        #[arg(long, default_value_t = 40)]
        steps: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Quantize the DiT block linears to Q8_0 (GGUF) — lower VRAM.
        #[arg(long)]
        quant: bool,
        /// ConvRot W8A8 INT8 for the DiT attn/MLP linears (`convrot` feature).
        #[arg(long)]
        convrot: bool,
        #[command(flatten)]
        cache: ConvrotCacheArgs,
        /// Q8_0-quantize the text encoder (weight-only, ~half VRAM).
        #[arg(long)]
        quant_text: bool,
        /// Load the text encoder from a pre-quantized GGUF (from prequantize-text)
        /// instead of quantizing on load — smaller VRAM pool, no ~58 s load.
        #[arg(long)]
        text_gguf: Option<std::path::PathBuf>,
        /// VAE tiled decode: latent tile size (0 = off). Caps decode peak memory
        /// (~tile²) so it fits alongside co-resident models. Try 32.
        #[arg(long, default_value_t = 0)]
        vae_tile: usize,
        /// True-CFG guidance scale. 1.0 = single forward (fast); >1 runs the DiT
        /// twice per step (cond + negative) for stronger prompt adherence.
        #[arg(long, default_value_t = 1.0)]
        guidance: f32,
        /// Negative prompt for CFG (used only when --guidance > 1).
        #[arg(long, default_value = "")]
        negative: String,
        /// Force the VAE decode in f32 (default: bf16 on CUDA).
        #[arg(long)]
        vae_f32: bool,
        /// Number of images to generate for this prompt in ONE batched denoise,
        /// each with its own seed (seed, seed+1, ...). 1 = single image (default).
        /// 4 is the efficient max on a 24GB card; B>=5 pays a per-image tax.
        #[arg(long, default_value_t = 1)]
        batch: usize,
        /// Output PNG (used when --batch 1). Written verbatim.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        /// Output directory for --batch > 1: writes 000.png..00N.png (one per
        /// seed). Required when --batch > 1.
        #[arg(long)]
        out_dir: Option<std::path::PathBuf>,
        /// Debug: also save each image's final latent as NNN.latent.safetensors
        /// (for the self-consistency check / debugging). Writes beside the PNG(s).
        #[arg(long)]
        emit_latents: bool,
    },
    /// Batch text-to-image: many prompts, each model loaded ONCE (encode all ->
    /// denoise all -> decode all), amortizing the ~24s of model loads.
    Batch {
        #[arg(long)]
        model: std::path::PathBuf,
        /// File with one prompt per line.
        #[arg(long)]
        prompts: std::path::PathBuf,
        #[arg(long, default_value_t = 1024)]
        size: usize,
        #[arg(long, default_value_t = 40)]
        steps: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        quant: bool,
        /// ConvRot W8A8 INT8 for the DiT attn/MLP linears (`convrot` feature).
        #[arg(long)]
        convrot: bool,
        #[command(flatten)]
        cache: ConvrotCacheArgs,
        /// Q8_0-quantize the text encoder (weight-only, ~half VRAM).
        #[arg(long)]
        quant_text: bool,
        /// Resident mode: load all three models once and keep them in VRAM,
        /// streaming prompts (needs --text-gguf --convrot to fit comfortably).
        #[arg(long)]
        resident: bool,
        /// Load the text encoder from a pre-quantized GGUF (from prequantize-text).
        #[arg(long)]
        text_gguf: Option<std::path::PathBuf>,
        /// VAE tiled decode: latent tile size (0 = off). Try 32 for resident.
        #[arg(long, default_value_t = 0)]
        vae_tile: usize,
        /// True-CFG guidance scale (1.0 = single forward; >1 = two forwards/step).
        #[arg(long, default_value_t = 1.0)]
        guidance: f32,
        /// Negative prompt for CFG (used only when --guidance > 1).
        #[arg(long, default_value = "")]
        negative: String,
        /// Force the VAE decode in f32 (default: bf16 on CUDA).
        #[arg(long)]
        vae_f32: bool,
        /// Output directory (writes 000.png, 001.png, ...).
        #[arg(long)]
        out_dir: std::path::PathBuf,
    },
    /// Encode input_ids through the Qwen3-VL text encoder to prompt embeddings (Phase 3).
    TextEncode {
        /// The text_encoder/ directory (config.json + *.safetensors shards).
        #[arg(long)]
        weights: std::path::PathBuf,
        /// A safetensors file with an `input_ids` tensor (from the embeds dump).
        #[arg(long)]
        input_ids: std::path::PathBuf,
        /// Number of leading system-prompt tokens to drop.
        #[arg(long, default_value_t = 0)]
        drop: usize,
        /// Q8_0-quantize the text encoder (weight-only, ~half VRAM).
        #[arg(long)]
        quant: bool,
        /// Output embeddings safetensors path.
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Run the full flow-match denoise loop (DiT x N steps) to a final latent (Phase 4).
    Denoise {
        /// transformer/ weights directory.
        #[arg(long)]
        weights: std::path::PathBuf,
        /// safetensors with initial noise under `hidden_states` (or `latent`).
        #[arg(long)]
        noise: std::path::PathBuf,
        /// safetensors with prompt embeddings under `embeds`.
        #[arg(long)]
        embeds: std::path::PathBuf,
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Quantize the DiT block linears to Q8_0 (GGUF).
        #[arg(long)]
        quant: bool,
        /// ConvRot W8A8 INT8 for the DiT attn/MLP linears (`convrot` feature).
        #[arg(long)]
        convrot: bool,
        #[command(flatten)]
        cache: ConvrotCacheArgs,
        /// Output final latent safetensors path.
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Run one DiT forward on dumped inputs and save the joint output (Phase 4).
    DitForward {
        /// The transformer/ directory (config.json + *.safetensors shards).
        #[arg(long)]
        weights: std::path::PathBuf,
        /// dit_io.safetensors with hidden_states/encoder_hidden_states/timestep.
        #[arg(long)]
        inputs: std::path::PathBuf,
        /// Quantize the DiT block linears to Q8_0 (GGUF).
        #[arg(long)]
        quant: bool,
        /// ConvRot W8A8 INT8 for the DiT attn/MLP linears (`convrot` feature).
        #[arg(long)]
        convrot: bool,
        #[command(flatten)]
        cache: ConvrotCacheArgs,
        /// Output safetensors path (joint output tensor).
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Pre-quantize the Qwen3-VL text encoder to a Q8_0 GGUF file, so
    /// `--text-gguf` loads it directly (no bf16-on-GPU transient, no ~58 s
    /// quantize-on-load) — keeps the resident-mode VRAM pool small.
    PrequantizeText {
        /// The text_encoder/ directory (bf16 config.json + *.safetensors shards).
        #[arg(long)]
        weights: std::path::PathBuf,
        /// Output .gguf file.
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Pre-quantize the DiT's ConvRot linears to a weights file (rotated INT8 +
    /// col scale), so `generate --convrot` on that file skips the load-time
    /// rotate+quant and mmaps ~half the bytes. Requires the `convrot` feature.
    PrequantizeConvrot {
        /// The transformer/ directory (bf16 config.json + *.safetensors shards).
        #[arg(long)]
        weights: std::path::PathBuf,
        /// Output .safetensors file (drop it in a dir and point --model's
        /// transformer at that dir, or pass the dir to dit-forward --weights).
        /// Omit it to (re)build this transformer's `--convrot` cache entry
        /// instead (pre-warming the default load path).
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        /// Cache dir when --out is omitted (default as for --convrot).
        #[arg(long)]
        convrot_cache: Option<std::path::PathBuf>,
    },
    /// Decode a saved latent (.pt) through the VAE to an RGBA PNG (Phase 2).
    VaeDecode {
        /// The vae/ directory (holds config.json + *.safetensors).
        #[arg(long)]
        weights: std::path::PathBuf,
        /// A packed latent saved by scripts/oracle.py (`*.latent.pt`).
        #[arg(long)]
        latent: std::path::PathBuf,
        /// Decode the VAE in bf16 (default: f32 reference).
        #[arg(long)]
        bf16: bool,
        /// Output PNG path.
        #[arg(long)]
        out: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Smoke { n } => smoke(n),
        Command::ConvrotTest => convrot_test(),
        Command::GemmBench { iters, batch } => gemm_bench(iters, batch),
        Command::SageTest => sage_test(),
        Command::FusednormTest => fusednorm_test(),
        Command::Bench { seq, iters } => bench(seq, iters),
        Command::Batch {
            model,
            prompts,
            size,
            steps,
            seed,
            quant,
            convrot,
            cache,
            quant_text,
            resident,
            text_gguf,
            vae_tile,
            guidance,
            negative,
            vae_f32,
            out_dir,
        } => batch(
            &model,
            &prompts,
            size,
            steps,
            seed,
            quant,
            convrot,
            &cache.spec(),
            quant_text,
            resident,
            text_gguf.as_deref(),
            vae_tile,
            guidance,
            &negative,
            vae_f32,
            &out_dir,
        ),
        Command::Info { weights } => info(&weights),
        Command::Generate {
            model,
            prompt,
            size,
            steps,
            seed,
            quant,
            convrot,
            cache,
            quant_text,
            text_gguf,
            vae_tile,
            guidance,
            negative,
            vae_f32,
            batch,
            out,
            out_dir,
            emit_latents,
        } => generate(
            &model,
            &prompt,
            size,
            steps,
            seed,
            quant,
            convrot,
            &cache.spec(),
            quant_text,
            text_gguf.as_deref(),
            vae_tile,
            guidance,
            &negative,
            vae_f32,
            batch,
            out.as_deref(),
            out_dir.as_deref(),
            emit_latents,
        ),
        Command::PrequantizeText { weights, out } => prequantize_text(&weights, &out),
        Command::PrequantizeConvrot {
            weights,
            out,
            convrot_cache,
        } => prequantize_convrot(&weights, out.as_deref(), convrot_cache.as_deref()),
        Command::VaeDecode {
            weights,
            latent,
            bf16,
            out,
        } => vae_decode(&weights, &latent, bf16, &out),
        Command::TextEncode {
            weights,
            input_ids,
            drop,
            quant,
            out,
        } => text_encode(&weights, &input_ids, drop, quant, &out),
        Command::DitForward {
            weights,
            inputs,
            quant,
            convrot,
            cache,
            out,
        } => dit_forward(&weights, &inputs, quant, convrot, &cache.spec(), &out),
        Command::Denoise {
            weights,
            noise,
            embeds,
            steps,
            quant,
            convrot,
            cache,
            out,
        } => denoise(
            &weights,
            &noise,
            &embeds,
            steps,
            quant,
            convrot,
            &cache.spec(),
            &out,
        ),
    }
}

/// Guided noise prediction. Runs the DiT on the conditional prompt; when
/// `guidance > 1` and a negative embedding is present, also runs it on the
/// negative and combines `v = v_uncond + guidance·(v_cond − v_uncond)` (true
/// CFG — ~2× the DiT cost). Returns the f32 noise prediction over the image
/// tokens. `guidance ≤ 1` keeps the single-forward fast path.
#[allow(clippy::too_many_arguments)]
fn guided_noise_pred(
    dit: &qwen_image_rs::model::dit::QwenImageDit,
    latents_dt: &Tensor,
    cond: &Tensor,
    neg: Option<&Tensor>,
    guidance: f32,
    tt: &Tensor,
    hw: usize,
    img_seq: usize,
) -> Result<Tensor> {
    let joint = dit.forward(latents_dt, cond, tt, hw, hw)?;
    let (_b, jl, _) = joint.dims3()?;
    let v_cond = joint
        .narrow(1, jl - img_seq, img_seq)?
        .to_dtype(DType::F32)?;
    if guidance > 1.0 {
        if let Some(neg) = neg {
            let ju = dit.forward(latents_dt, neg, tt, hw, hw)?;
            let (_b, jlu, _) = ju.dims3()?;
            let v_uncond = ju.narrow(1, jlu - img_seq, img_seq)?.to_dtype(DType::F32)?;
            return Ok((&v_uncond + ((&v_cond - &v_uncond)? * guidance as f64)?)?);
        }
    }
    Ok(v_cond)
}

/// The DiT's VarBuilder over the files `convrot_cache::resolve_dit_files`
/// picks: under `--convrot` the cached prequantized file (built on first use),
/// else the transformer dir's own safetensors.
fn dit_vb(
    dir: &std::path::Path,
    convrot: bool,
    cache: &qwen_image_rs::convrot_cache::CacheSpec,
    dtype: DType,
    dev: &candle_core::Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    let files = qwen_image_rs::convrot_cache::resolve_dit_files(dir, convrot, cache)?;
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? })
}

/// Load the text encoder, preferring a pre-quantized GGUF (`--text-gguf`) which
/// loads Q8_0 directly (no bf16-on-GPU transient); otherwise mmap the bf16
/// safetensors and (optionally) quantize on load.
fn load_text_encoder(
    model: &std::path::Path,
    dev: &candle_core::Device,
    dtype: DType,
    quant_text: bool,
    text_gguf: Option<&std::path::Path>,
) -> Result<qwen_image_rs::model::text_encoder::QwenTextEncoder> {
    use qwen_image_rs::model::text_encoder::{QwenTextEncoder, TextConfig};
    if let Some(g) = text_gguf {
        let qvb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(g, dev)?;
        QwenTextEncoder::load_gguf(&TextConfig::default(), qvb)
    } else {
        let files = WeightSet::resolve(&model.join("text_encoder"))?.files;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        QwenTextEncoder::load(&TextConfig::default(), quant_text, vb)
    }
}

/// Standalone text-to-image: prompt -> PNG. Loads the three models one at a
/// time (text encoder -> DiT -> VAE), freeing each before the next so the
/// pipeline fits in 24 GB.
#[allow(clippy::too_many_arguments)]
fn generate(
    model: &std::path::Path,
    prompt: &str,
    size: usize,
    steps: usize,
    seed: u64,
    quant: bool,
    convrot: bool,
    cache: &qwen_image_rs::convrot_cache::CacheSpec,
    quant_text: bool,
    text_gguf: Option<&std::path::Path>,
    vae_tile: usize,
    guidance: f32,
    negative: &str,
    vae_f32: bool,
    batch: usize,
    out: Option<&std::path::Path>,
    out_dir: Option<&std::path::Path>,
    emit_latents: bool,
) -> Result<()> {
    if batch < 1 {
        anyhow::bail!("--batch must be >= 1");
    }
    // Output-target validation up front (before the ~20s of model loads).
    if batch == 1 && out.is_none() {
        anyhow::bail!("--out <file> is required for --batch 1");
    }
    if batch > 1 && out_dir.is_none() {
        anyhow::bail!("--out-dir <dir> is required for --batch > 1 (writes 000.png..)");
    }
    // Warn on the non-applicable output flag rather than silently ignoring it.
    if batch == 1 && out_dir.is_some() {
        tracing::warn!("--out-dir ignored for --batch 1 (writing --out)");
    }
    if batch > 1 && out.is_some() {
        tracing::warn!("--out ignored for --batch > 1 (writing --out-dir/NNN.png)");
    }
    use qwen_image_rs::model::dit::QwenImageDit;
    use qwen_image_rs::model::scheduler::{FlowConfig, FlowMatchEuler};
    use qwen_image_rs::model::text_encoder::prompt as tmpl;
    use qwen_image_rs::model::{config::VaeConfig, vae};
    use tokenizers::Tokenizer;

    let dev = device::best_device()?;
    let dtype = if matches!(dev, candle_core::Device::Cuda(_)) {
        DType::BF16
    } else {
        DType::F32
    };
    let load_vb = |dir: std::path::PathBuf, dt: DType| -> Result<candle_nn::VarBuilder> {
        let files = WeightSet::resolve(&dir)?.files;
        Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dt, &dev)? })
    };
    // The VAE runs in bf16 on CUDA by default (tensor-core convs, ~2x decode);
    // --vae-f32 forces the f32 reference path.
    let vae_dt = if vae_f32 { DType::F32 } else { dtype };

    // Tokenize the t2i chat template; drop = system-prefix token count.
    let tok = Tokenizer::from_file(model.join("processor/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let sysp = format!("<|im_start|>system\n{}<|im_end|>\n", tmpl::SYS_PROMPT);
    let drop = tok
        .encode(sysp, false)
        .map_err(|e| anyhow::anyhow!("encode sys: {e}"))?
        .get_ids()
        .len();

    // 1. Text encoder -> prompt embeddings (+ negative, for CFG). Freed after.
    let (embeds, neg_embeds) = {
        let te = load_text_encoder(model, &dev, dtype, quant_text, text_gguf)?;
        let encode = |p: &str| -> Result<Tensor> {
            let ids: Vec<u32> = tok
                .encode(tmpl::t2i_template(p), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            let seq = ids.len();
            let hidden = te.forward(&Tensor::from_vec(ids, (1, seq), &dev)?)?;
            Ok(hidden.narrow(1, drop, seq - drop)?.contiguous()?)
        };
        let cond = encode(prompt)?;
        let neg = if guidance > 1.0 {
            Some(encode(negative)?)
        } else {
            None
        };
        (cond, neg)
    };
    tracing::info!(shape = ?embeds.dims(), guidance, "prompt embeddings");

    // Broadcast the (1, txt, 4096) embeds across the batch lanes (all lanes share
    // the prompt; only the seed differs). Materialized so the fused kernels read
    // real per-row data, not a stride-0 view. B=1 leaves the tensor untouched:
    // it is the TE output narrowed past the system prefix, which candle keeps
    // as a zero-copy view at start_offset drop*4096 (`contiguous()` is a no-op
    // for it). Every fused bridge honors that offset (crate::layout); one that
    // did not made B=1 condition on the system-prompt rows
    // (qwen-image-rs-b1-off-prompt).
    let embeds = if batch > 1 {
        let (_, t, hd) = embeds.dims3()?;
        embeds.broadcast_as((batch, t, hd))?.contiguous()?
    } else {
        embeds
    };
    let neg_embeds = match neg_embeds {
        Some(n) if batch > 1 => {
            let (_, t, hd) = n.dims3()?;
            Some(n.broadcast_as((batch, t, hd))?.contiguous()?)
        }
        other => other,
    };

    // 2. DiT + flow-match denoise -> final latent (freed after).
    let hw = size / 16; // vae spatial compression
    let img_seq = hw * hw;
    let latent = {
        let dit = QwenImageDit::load(
            32,
            64,
            quant,
            convrot,
            dit_vb(&model.join("transformer"), convrot, cache, dtype, &dev)?,
        )?;
        // One seed per lane: lane i uses (seed + i), resetting the RNG each lane
        // so lane i's NOISE is identical to a sequential `generate --seed (seed+i)`
        // run. The resulting IMAGE is NOT bit-identical to that single run — the
        // fast path is non-deterministic and diverges by batch size (different
        // GEMM tiling); that divergence is intended (see HANDOVER). Do NOT
        // collapse this into a single set_seed + N draws.
        let mut lanes = Vec::with_capacity(batch);
        for i in 0..batch {
            dev.set_seed(seed.wrapping_add(i as u64))?;
            lanes.push(Tensor::randn(0f32, 1f32, (1, img_seq, 64), &dev)?);
        }
        let mut latents = Tensor::cat(&lanes, 0)?.to_dtype(DType::F32)?; // (batch, img, 64)
        let sched = FlowMatchEuler::new(&FlowConfig::default(), steps, img_seq);
        for (i, t) in sched.timesteps().iter().enumerate() {
            let tt = Tensor::from_vec(vec![(*t / 1000.0) as f32], (1,), &dev)?;
            let np = guided_noise_pred(
                &dit,
                &latents.to_dtype(dtype)?,
                &embeds,
                neg_embeds.as_ref(),
                guidance,
                &tt,
                hw,
                img_seq,
            )?;
            latents = (latents + (np * sched.dt(i))?)?;
            if i % 10 == 0 || i + 1 == steps {
                tracing::info!(step = i, "denoising");
            }
        }
        latents
    };

    // 3. VAE decode -> RGBA PNG.
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model.join("vae/config.json"))?)?;
    let f32vec = |k: &str| -> Vec<f32> {
        cfg_json[k].as_array().map_or(vec![], |a| {
            a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
        })
    };
    let vmodel = vae::QwenImageVae::load(
        &VaeConfig::default(),
        &f32vec("latents_mean"),
        &f32vec("latents_std"),
        load_vb(model.join("vae"), vae_dt)?,
    )?;
    // One VAE (loaded above) decodes every lane's latent in a loop.
    if let Some(d) = out_dir {
        if batch > 1 {
            std::fs::create_dir_all(d)?;
        }
    }
    for i in 0..batch {
        let li = latent.narrow(0, i, 1)?; // (1, img, 64)
        let z = vae::unpack_latents(&li, 64)?;
        let img = if vae_tile > 0 {
            vmodel.decode_tiled(&z, vae_tile, (vae_tile / 4).max(1))?
        } else {
            vmodel.decode(&z)?
        };
        let (w, h, bytes) = vae::to_rgba_u8(&img)?;
        let buf: image::RgbaImage =
            image::ImageBuffer::from_raw(w as u32, h as u32, bytes).context("image buffer")?;
        let path = if batch == 1 {
            out.expect("--out validated present for batch 1")
                .to_path_buf()
        } else {
            out_dir
                .expect("--out-dir validated present for batch > 1")
                .join(format!("{i:03}.png"))
        };
        buf.save(&path)?;
        if emit_latents {
            let lp = path.with_extension("latent.safetensors");
            let mut m = std::collections::HashMap::new();
            m.insert("latent".to_string(), li.contiguous()?);
            candle_core::safetensors::save(&m, &lp)?;
        }
        println!(
            "wrote {} (seed {}, {w}x{h})",
            path.display(),
            seed.wrapping_add(i as u64)
        );
    }
    Ok(())
}

/// Batch generation: load each model once for all prompts (encode all -> free
/// -> denoise all -> free -> decode all), so the ~24s of model loads is paid
/// once instead of per image.
#[allow(clippy::too_many_arguments)]
fn batch(
    model: &std::path::Path,
    prompts_path: &std::path::Path,
    size: usize,
    steps: usize,
    seed: u64,
    quant: bool,
    convrot: bool,
    cache: &qwen_image_rs::convrot_cache::CacheSpec,
    quant_text: bool,
    resident: bool,
    text_gguf: Option<&std::path::Path>,
    vae_tile: usize,
    guidance: f32,
    negative: &str,
    vae_f32: bool,
    out_dir: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::dit::QwenImageDit;
    use qwen_image_rs::model::scheduler::{FlowConfig, FlowMatchEuler};
    use qwen_image_rs::model::text_encoder::prompt as tmpl;
    use qwen_image_rs::model::{config::VaeConfig, vae};
    use tokenizers::Tokenizer;

    let dev = device::best_device()?;
    let dtype = if matches!(dev, candle_core::Device::Cuda(_)) {
        DType::BF16
    } else {
        DType::F32
    };
    let load_vb = |dir: std::path::PathBuf, dt: DType| -> Result<candle_nn::VarBuilder> {
        let files = WeightSet::resolve(&dir)?.files;
        Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dt, &dev)? })
    };
    // The VAE runs in bf16 on CUDA by default (tensor-core convs, ~2x decode);
    // --vae-f32 forces the f32 reference path.
    let vae_dt = if vae_f32 { DType::F32 } else { dtype };
    std::fs::create_dir_all(out_dir)?;

    let prompts: Vec<String> = std::fs::read_to_string(prompts_path)?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    tracing::info!(n = prompts.len(), "batch prompts");

    let tok = Tokenizer::from_file(model.join("processor/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let sysp = format!("<|im_start|>system\n{}<|im_end|>\n", tmpl::SYS_PROMPT);
    let drop = tok
        .encode(sysp, false)
        .map_err(|e| anyhow::anyhow!("encode sys: {e}"))?
        .get_ids()
        .len();

    let hw = size / 16;
    let img_seq = hw * hw;
    let vae_cfg = |k: &str| -> Result<Vec<f32>> {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(model.join("vae/config.json"))?)?;
        Ok(j[k].as_array().map_or(vec![], |a| {
            a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
        }))
    };
    let save_png = |img: &Tensor, path: std::path::PathBuf| -> Result<()> {
        let (w, h, bytes) = vae::to_rgba_u8(img)?;
        let buf: image::RgbaImage =
            image::ImageBuffer::from_raw(w as u32, h as u32, bytes).context("image buffer")?;
        buf.save(path)?;
        Ok(())
    };

    // Resident pipeline: load all three models ONCE and keep them in VRAM,
    // running encode->denoise->decode per prompt without freeing between them.
    // Needs everything to fit at once — pass --quant-text (and --convrot) so the
    // 8B encoder + DiT + VAE stay under 24 GB. Amortizes the one-time loads
    // across every prompt (the batch/serve win).
    if resident {
        let te = load_text_encoder(model, &dev, dtype, quant_text, text_gguf)?;
        let dit = QwenImageDit::load(
            32,
            64,
            quant,
            convrot,
            dit_vb(&model.join("transformer"), convrot, cache, dtype, &dev)?,
        )?;
        let vmodel = vae::QwenImageVae::load(
            &VaeConfig::default(),
            &vae_cfg("latents_mean")?,
            &vae_cfg("latents_std")?,
            load_vb(model.join("vae"), vae_dt)?,
        )?;
        let sched = FlowMatchEuler::new(&FlowConfig::default(), steps, img_seq);
        // Negative embedding (encoded once; shared by every prompt) for CFG.
        let neg_embeds = if guidance > 1.0 {
            let ids: Vec<u32> = tok
                .encode(tmpl::t2i_template(negative), false)
                .map_err(|e| anyhow::anyhow!("encode neg: {e}"))?
                .get_ids()
                .to_vec();
            let seq = ids.len();
            Some(
                te.forward(&Tensor::from_vec(ids, (1, seq), &dev)?)?
                    .narrow(1, drop, seq - drop)?
                    .contiguous()?,
            )
        } else {
            None
        };
        tracing::info!("resident: all three models loaded, streaming prompts");
        for (i, p) in prompts.iter().enumerate() {
            let t_enc = std::time::Instant::now();
            let ids: Vec<u32> = tok
                .encode(tmpl::t2i_template(p), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            let seq = ids.len();
            let emb = te
                .forward(&Tensor::from_vec(ids, (1, seq), &dev)?)?
                .narrow(1, drop, seq - drop)?
                .contiguous()?;
            dev.synchronize()?;
            let enc_ms = t_enc.elapsed().as_millis();
            let t_dn = std::time::Instant::now();
            dev.set_seed(seed + i as u64)?;
            let mut latents =
                Tensor::randn(0f32, 1f32, (1, img_seq, 64), &dev)?.to_dtype(DType::F32)?;
            for (si, t) in sched.timesteps().iter().enumerate() {
                let tt = Tensor::from_vec(vec![(*t / 1000.0) as f32], (1,), &dev)?;
                let np = guided_noise_pred(
                    &dit,
                    &latents.to_dtype(dtype)?,
                    &emb,
                    neg_embeds.as_ref(),
                    guidance,
                    &tt,
                    hw,
                    img_seq,
                )?;
                latents = (latents + (np * sched.dt(si))?)?;
            }
            dev.synchronize()?;
            let dn_ms = t_dn.elapsed().as_millis();
            let t_dec = std::time::Instant::now();
            let z = vae::unpack_latents(&latents, 64)?;
            let img = if vae_tile > 0 {
                vmodel.decode_tiled(&z, vae_tile, (vae_tile / 4).max(1))?
            } else {
                vmodel.decode(&z)?
            };
            dev.synchronize()?;
            let dec_ms = t_dec.elapsed().as_millis();
            save_png(&img, out_dir.join(format!("{i:03}.png")))?;
            tracing::info!(
                image = i,
                enc_ms,
                denoise_ms = dn_ms,
                decode_ms = dec_ms,
                "done (resident)"
            );
        }
        println!(
            "wrote {} images to {} (resident)",
            prompts.len(),
            out_dir.display()
        );
        return Ok(());
    }

    // Phase 1: encode every prompt (+ the shared negative). Text encoder loaded
    // once, then freed.
    let mut embeds_list = Vec::with_capacity(prompts.len());
    let mut neg_embeds = None;
    {
        let te = load_text_encoder(model, &dev, dtype, quant_text, text_gguf)?;
        let encode = |p: &str| -> Result<Tensor> {
            let ids: Vec<u32> = tok
                .encode(tmpl::t2i_template(p), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            let seq = ids.len();
            let hidden = te.forward(&Tensor::from_vec(ids, (1, seq), &dev)?)?;
            Ok(hidden.narrow(1, drop, seq - drop)?.contiguous()?)
        };
        for p in &prompts {
            embeds_list.push(encode(p)?);
        }
        if guidance > 1.0 {
            neg_embeds = Some(encode(negative)?);
        }
    }
    tracing::info!("encoded {} prompts", embeds_list.len());

    // Phase 2: denoise every prompt (DiT loaded once, then freed).
    let mut latents_list = Vec::with_capacity(prompts.len());
    {
        let dit = QwenImageDit::load(
            32,
            64,
            quant,
            convrot,
            dit_vb(&model.join("transformer"), convrot, cache, dtype, &dev)?,
        )?;
        let sched = FlowMatchEuler::new(&FlowConfig::default(), steps, img_seq);
        for (i, emb) in embeds_list.iter().enumerate() {
            dev.set_seed(seed + i as u64)?;
            let mut latents =
                Tensor::randn(0f32, 1f32, (1, img_seq, 64), &dev)?.to_dtype(DType::F32)?;
            for (si, t) in sched.timesteps().iter().enumerate() {
                let tt = Tensor::from_vec(vec![(*t / 1000.0) as f32], (1,), &dev)?;
                let np = guided_noise_pred(
                    &dit,
                    &latents.to_dtype(dtype)?,
                    emb,
                    neg_embeds.as_ref(),
                    guidance,
                    &tt,
                    hw,
                    img_seq,
                )?;
                latents = (latents + (np * sched.dt(si))?)?;
            }
            latents_list.push(latents);
            tracing::info!(image = i, "denoised");
        }
    }

    // Phase 3: decode every latent (VAE loaded once).
    let vmodel = vae::QwenImageVae::load(
        &VaeConfig::default(),
        &vae_cfg("latents_mean")?,
        &vae_cfg("latents_std")?,
        load_vb(model.join("vae"), vae_dt)?,
    )?;
    for (i, lat) in latents_list.iter().enumerate() {
        let z = vae::unpack_latents(lat, 64)?;
        let img = if vae_tile > 0 {
            vmodel.decode_tiled(&z, vae_tile, (vae_tile / 4).max(1))?
        } else {
            vmodel.decode(&z)?
        };
        save_png(&img, out_dir.join(format!("{i:03}.png")))?;
    }
    println!("wrote {} images to {}", prompts.len(), out_dir.display());
    Ok(())
}

/// Phase 4: full flow-match denoise loop -> final latent.
#[allow(clippy::too_many_arguments)]
fn denoise(
    weights: &std::path::Path,
    noise: &std::path::Path,
    embeds: &std::path::Path,
    steps: usize,
    quant: bool,
    convrot: bool,
    cache: &qwen_image_rs::convrot_cache::CacheSpec,
    out: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::dit::QwenImageDit;
    use qwen_image_rs::model::scheduler::{FlowConfig, FlowMatchEuler};

    let dev = device::best_device()?;
    let dtype = if matches!(dev, candle_core::Device::Cuda(_)) {
        DType::BF16
    } else {
        DType::F32
    };
    tracing::info!(device = device::label(&dev), steps, "denoise");

    let vb = dit_vb(weights, convrot, cache, dtype, &dev)?;
    let model = QwenImageDit::load(32, 64, quant, convrot, vb)?;

    let nmap = candle_core::safetensors::load(noise, &dev)?;
    let mut latents = nmap
        .get("hidden_states")
        .or_else(|| nmap.get("latent"))
        .context("noise file needs `hidden_states` or `latent`")?
        .to_dtype(DType::F32)?;
    if latents.dims().len() == 2 {
        latents = latents.unsqueeze(0)?;
    }
    let (_b, img_seq, _) = latents.dims3()?;
    let hw = (img_seq as f64).sqrt() as usize;

    let emap = candle_core::safetensors::load(embeds, &dev)?;
    let mut enc = emap
        .get("embeds")
        .context("embeds file needs `embeds`")?
        .clone();
    if enc.dims().len() == 2 {
        enc = enc.unsqueeze(0)?;
    }
    let enc = enc.to_dtype(dtype)?;

    let sched = FlowMatchEuler::new(&FlowConfig::default(), steps, img_seq);
    let ts = sched.timesteps().to_vec();
    for (i, t) in ts.iter().enumerate() {
        let tt = Tensor::from_vec(vec![(*t / 1000.0) as f32], (1,), &dev)?;
        let out_joint = model.forward(&latents.to_dtype(dtype)?, &enc, &tt, hw, hw)?;
        // image tokens = last img_seq of the joint sequence
        let (_b, joint, _) = out_joint.dims3()?;
        let noise_pred = out_joint
            .narrow(1, joint - img_seq, img_seq)?
            .to_dtype(DType::F32)?;
        latents = (latents + (noise_pred * sched.dt(i))?)?;
        if i % 10 == 0 || i + 1 == ts.len() {
            tracing::info!(step = i, t, "denoising");
        }
    }

    let mut map = std::collections::HashMap::new();
    map.insert("latent".to_string(), latents.contiguous()?); // (1, seq, 64)
    candle_core::safetensors::save(&map, out)?;
    println!("wrote {} ({} steps)", out.display(), steps);
    Ok(())
}

/// Phase 4: run one DiT forward on dumped inputs, save the joint output.
fn dit_forward(
    weights: &std::path::Path,
    inputs: &std::path::Path,
    quant: bool,
    convrot: bool,
    cache: &qwen_image_rs::convrot_cache::CacheSpec,
    out: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::dit::QwenImageDit;

    let dev = device::best_device()?;
    let dtype = if matches!(dev, candle_core::Device::Cuda(_)) {
        DType::BF16
    } else {
        DType::F32
    };
    tracing::info!(device = device::label(&dev), "dit-forward");

    let vb = dit_vb(weights, convrot, cache, dtype, &dev)?;
    let model = QwenImageDit::load(32, 64, quant, convrot, vb)?;

    let m = candle_core::safetensors::load(inputs, &dev)?;
    let get = |k: &str| -> Result<Tensor> {
        Ok(m.get(k).context(format!("missing {k}"))?.to_dtype(dtype)?)
    };
    let hidden = get("hidden_states")?;
    let enc = get("encoder_hidden_states")?;
    let ts = get("timestep")?;
    let (_b, img, _) = hidden.dims3()?;
    let hw = (img as f64).sqrt() as usize;
    tracing::info!(img, hw, "running DiT forward");

    let output = model
        .forward(&hidden, &enc, &ts, hw, hw)?
        .to_dtype(DType::F32)?;
    tracing::info!(shape = ?output.dims(), "DiT output");

    let mut map = std::collections::HashMap::new();
    map.insert("output".to_string(), output.i(0)?.contiguous()?);
    candle_core::safetensors::save(&map, out)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// Phase 3: run the Qwen3-VL text encoder on dumped input_ids, drop the system
/// prefix, save the pre-final-norm hidden states.
fn text_encode(
    weights: &std::path::Path,
    input_ids_path: &std::path::Path,
    drop: usize,
    quant: bool,
    out: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::text_encoder::{QwenTextEncoder, TextConfig};

    let dev = device::best_device()?;
    tracing::info!(device = device::label(&dev), "text-encode");

    let set = WeightSet::resolve(weights)?;
    let files = set.files.clone();
    // bf16: the 8B encoder is ~16 GB in bf16 (fits the 24 GB 4090) and matches
    // the oracle's bf16 compute. F32 would be 32 GB and OOM.
    let dtype = if matches!(dev, candle_core::Device::Cuda(_)) {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
    let cfg = TextConfig::default();
    let model = QwenTextEncoder::load(&cfg, quant, vb)?;

    let ids_map = candle_core::safetensors::load(input_ids_path, &dev)?;
    let ids = ids_map
        .get("input_ids")
        .context("no 'input_ids' in file")?
        .to_dtype(DType::U32)?;
    let seq = ids.dims1()?;
    let ids = ids.reshape((1, seq))?;
    tracing::info!(seq, drop, "running encoder");

    let hidden = model.forward(&ids)?; // (1, seq, hidden)
    let kept = seq - drop;
    let embeds = hidden
        .narrow(1, drop, kept)?
        .i(0)?
        .to_dtype(DType::F32)?
        .contiguous()?; // (kept, hidden)
    tracing::info!(shape = ?embeds.dims(), "embeddings");

    let mut map = std::collections::HashMap::new();
    map.insert("embeds".to_string(), embeds);
    candle_core::safetensors::save(&map, out)?;
    println!("wrote {} (kept {kept} tokens)", out.display());
    Ok(())
}

/// Phase 2: load the VAE decoder, decode a saved packed latent, save RGBA PNG.
fn vae_decode(
    weights: &std::path::Path,
    latent: &std::path::Path,
    bf16: bool,
    out: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::{config::VaeConfig, vae};

    let dev = device::best_device()?;
    tracing::info!(device = device::label(&dev), "vae-decode");

    // Read vae/config.json for latents_mean/std and shape params.
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(weights.join("config.json"))?)
            .context("parsing vae config.json")?;
    let f32vec = |k: &str| -> Result<Vec<f32>> {
        Ok(cfg_json[k]
            .as_array()
            .with_context(|| format!("config.{k} not an array"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect())
    };
    let latents_mean = f32vec("latents_mean")?;
    let latents_std = f32vec("latents_std")?;
    let cfg = VaeConfig::default();

    // Resolve the safetensors shard(s) in the vae dir.
    let set = WeightSet::resolve(weights)?;
    let files: Vec<_> = set.files.clone();
    let vae_dt = if bf16 { DType::BF16 } else { DType::F32 };
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, vae_dt, &dev)? };
    let model = vae::QwenImageVae::load(&cfg, &latents_mean, &latents_std, vb)?;

    // Load the packed latent. Prefer a safetensors sidecar (candle-native);
    // fall back to a torch .pt pickle.
    let packed = if latent.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let map = candle_core::safetensors::load(latent, &dev)?;
        map.get("latent")
            .or_else(|| map.values().next())
            .context("no 'latent' tensor in safetensors")?
            .to_dtype(DType::F32)?
    } else {
        candle_core::pickle::read_all(latent)?
            .into_iter()
            .next()
            .context("no tensor in .pt file")?
            .1
            .to_device(&dev)?
            .to_dtype(DType::F32)?
    };
    tracing::info!(?packed, shape = ?packed.dims(), "loaded latent");
    let packed = if packed.dims().len() == 3 {
        packed
    } else {
        anyhow::bail!(
            "expected packed latent (B, seq, z*4), got {:?}",
            packed.dims()
        );
    };
    let z = vae::unpack_latents(&packed, cfg.latent_channels)?;
    tracing::info!(shape = ?z.dims(), "unpacked latent");

    let img = model.decode(&z)?;
    tracing::info!(shape = ?img.dims(), "decoded image");
    let (w, h, bytes) = vae::to_rgba_u8(&img)?;
    let buf: image::RgbaImage =
        image::ImageBuffer::from_raw(w as u32, h as u32, bytes).context("image buffer")?;
    buf.save(out)?;
    println!("wrote {} ({w}x{h} RGBA)", out.display());
    Ok(())
}

/// `gemm-bench`: ms per tile config for each DiT INT8 GEMM shape.
fn gemm_bench(iters: usize, batch: usize) -> Result<()> {
    #[cfg(feature = "convrot")]
    {
        use qwen_image_rs::convrot::{bench_gemm_configs, gemm_config, EpilogueOut, GEMM_CONFIGS};
        let m = 4117 * batch;
        let shapes = [
            (m, 4096, 4096),  // to_q / to_k / to_out
            (m, 8192, 4096),  // merged q|k
            (m, 12288, 4096), // gate / proj (or merged q|k|v)
            (m, 24576, 4096), // merged gate|proj
            (m, 4096, 12288), // mlp out
            (m, 64, 4096),    // proj_out
            (2, 16384, 4096), // modulation
            (2, 4096, 4096),  // norm_out / time_embed.linear_2
        ];
        for (i, c) in GEMM_CONFIGS.iter().enumerate() {
            println!(
                "cfg {i}: TB {}x{}x{} warp {}x{}x{} stages {} swizzle {}",
                c.0, c.1, c.2, c.3, c.4, c.5, c.6, c.7
            );
        }
        let mut all_same = true;
        for (out, list) in [
            (EpilogueOut::Bf16, &shapes[..]),
            (EpilogueOut::F16, &shapes[..1]),
        ] {
            for (m, n, k, times, same) in bench_gemm_configs(list, out, iters)? {
                let flop = 2.0 * (m * n * k) as f64;
                let cells: Vec<String> = times
                    .iter()
                    .enumerate()
                    .map(|(i, t)| match t {
                        Some(ms) => format!("{i}:{ms:.4}ms/{:.0}T", flop / ms / 1e9),
                        None => format!("{i}:n/a"),
                    })
                    .collect();
                println!(
                    "{out:?} M={m} N={n} K={k} [selected cfg {}] {} bit-identical={same}",
                    gemm_config(m, n, k),
                    cells.join(" ")
                );
                all_same &= same;
            }
        }
        anyhow::ensure!(all_same, "tile configs disagree (must be bit-identical)");
        Ok(())
    }
    #[cfg(not(feature = "convrot"))]
    {
        let _ = (iters, batch);
        anyhow::bail!("build with --features convrot (needs CUTLASS_DIR)")
    }
}

/// Self-test the ConvRot INT8 GEMM bridge end to end.
fn convrot_test() -> Result<()> {
    #[cfg(feature = "convrot")]
    {
        let maxdiff = qwen_image_rs::convrot::self_test()?;
        println!(
            "convrot int8-gemm bridge: max abs diff vs CPU int ref = {maxdiff} ({})",
            if maxdiff == 0 {
                "EXACT — bridge OK"
            } else {
                "MISMATCH"
            }
        );
        // (M, N, K): the odd-M alignment case + the DiT tail-linear shapes.
        let mut epi_bad = 0usize;
        for (m, n, k) in [
            (37, 256, 512),
            (2, 4096, 256),
            (2, 16384, 4096),
            (37, 64, 4096),
        ] {
            let (bad_bf, bad_hf) = qwen_image_rs::convrot::self_test_epilogue(m, n, k)?;
            println!(
                "convrot dequant epilogue vs host ref (M={m} N={n} K={k}): bf16 mismatches = {bad_bf}, f16 mismatches = {bad_hf} ({})",
                if bad_bf == 0 && bad_hf == 0 {
                    "EXACT"
                } else {
                    "MISMATCH"
                }
            );
            epi_bad += bad_bf + bad_hf;
        }
        let (d16, a16) = qwen_image_rs::convrot::self_test_f16_vs_cast()?;
        let f16_ok = d16 <= a16 / 128.0;
        println!(
            "convrot f16 epilogue vs bf16 epilogue + cast (to_v shape): max abs diff = {d16:.3e} of max |y| {a16:.3e} ({})",
            if f16_ok { "OK" } else { "MISMATCH" }
        );
        let cos = qwen_image_rs::convrot::self_test_linear()?;
        println!(
            "convrot INT8 linear vs bf16: cosine = {cos:.5} ({})",
            if cos > 0.99 { "OK" } else { "TOO LOW" }
        );
        let (cr_ms, bf_ms) = qwen_image_rs::convrot::bench_linear(50)?;
        println!(
            "MLP-shape timing: convrot {cr_ms:.3} ms vs bf16 {bf_ms:.3} ms ({:.2}x)",
            bf_ms / cr_ms
        );
        // Fused rotate+quantize vs the f64 host reference and the old path.
        let mut rq_ok = true;
        for c in qwen_image_rs::convrot::self_test_rotate_quant(&[
            (37, 256),
            (300, 4096),
            (64, 12288),
            (5, 16384),
        ])? {
            println!(
                "convrot fused rotate+quant (M={} K={}): vs f64 ref max|dq|={} ({} of {} differ), scale rel {:.1e}; vs old bf16-GEMM path max|dq|={} ({} differ, {:.3}%), scale rel {:.1e} ({})",
                c.m,
                c.k,
                c.ref_max_diff,
                c.ref_mismatches,
                c.m * c.k,
                c.ref_scale_rel,
                c.old_max_diff,
                c.old_mismatches,
                100.0 * c.old_mismatches as f64 / (c.m * c.k) as f64,
                c.old_scale_rel,
                if c.ok() { "OK" } else { "MISMATCH" }
            );
            rq_ok &= c.ok();
        }
        let rq_rej = qwen_image_rs::convrot::self_test_rotate_quant_rejects()?;
        println!(
            "convrot fused rotate+quant rejects K=128/384/16640, M=0 empty: {}",
            if rq_rej { "OK" } else { "MISMATCH" }
        );
        for (m, n, k, c) in qwen_image_rs::convrot::self_test_linear_fused_vs_unfused()? {
            println!(
                "convrot linear fused vs unfused forward (M={m} N={n} K={k}): cosine = {c:.7} ({})",
                if c > 0.9999 { "OK" } else { "TOO LOW" }
            );
            rq_ok &= c > 0.9999;
        }
        for (k, f, u) in qwen_image_rs::convrot::bench_rotate_quant(50)? {
            println!(
                "activation rotate+quant (M=4117 K={k}): fused {f:.3} ms vs bf16 GEMM + quant {u:.3} ms ({:.2}x)",
                u / f
            );
        }
        // Fused SwiGLU + rotate + quantize (silu(g)*p never stored).
        let mut swg_ok = true;
        for c in qwen_image_rs::convrot::self_test_swiglu_rotate_quant(&[
            (37, 256),
            (300, 4096),
            (64, 12288),
            (5, 16384),
        ])? {
            println!(
                "convrot fused swiglu+rotate+quant (M={} K={}): vs f64 ref max|dq|={} ({} of {} differ), scale rel {:.1e}; vs candle silu*mul + rotate+quant max|dq|={} ({} differ, {:.3}%), scale rel {:.1e} ({})",
                c.m,
                c.k,
                c.ref_max_diff,
                c.ref_mismatches,
                c.m * c.k,
                c.ref_scale_rel,
                c.old_max_diff,
                c.old_mismatches,
                100.0 * c.old_mismatches as f64 / (c.m * c.k) as f64,
                c.old_scale_rel,
                if c.ok() { "OK" } else { "MISMATCH" }
            );
            swg_ok &= c.ok();
        }
        for (what, ok) in qwen_image_rs::convrot::self_test_swiglu_views()? {
            println!(
                "convrot fused swiglu view contract: {what}: {}",
                if ok { "OK" } else { "MISMATCH" }
            );
            swg_ok &= ok;
        }
        for (what, c) in qwen_image_rs::convrot::self_test_linear_swiglu()? {
            println!(
                "convrot linear forward_swiglu vs forward(silu(g)*p) ({what}, M=257 N=4096 K=12288): cosine = {c:.7} ({})",
                if c > 0.9999 { "OK" } else { "TOO LOW" }
            );
            swg_ok &= c > 0.9999;
        }
        let (sf, su) = qwen_image_rs::convrot::bench_swiglu_quant(50)?;
        println!(
            "MLP-out activation path (M=4117 K=12288): fused swiglu+rotate+quant {sf:.3} ms vs silu + mul + rotate+quant {su:.3} ms ({:.2}x)",
            su / sf
        );
        if maxdiff != 0 || epi_bad != 0 || !f16_ok || !rq_ok || !rq_rej || !swg_ok {
            anyhow::bail!("convrot self-test FAILED (bit-exactness / rotate+quant bounds)");
        }
        report_offset_views("convrot", qwen_image_rs::convrot::self_test_offset_views()?)
    }
    #[cfg(not(feature = "convrot"))]
    anyhow::bail!("build with --features convrot (needs CUTLASS_DIR)")
}

/// Self-test SageAttention (INT8-QK / FP16-PV) vs an f32 softmax-attention
/// reference, non-causal and causal.
fn sage_test() -> Result<()> {
    #[cfg(feature = "sage")]
    {
        for causal in [false, true] {
            let cos = qwen_image_rs::sage::self_test(causal)?;
            println!(
                "sage INT8 attn (causal={causal}) vs f32 ref: cosine = {cos:.5} ({})",
                if cos > 0.99 { "OK" } else { "TOO LOW" }
            );
        }
        let rope_cos = qwen_image_rs::rope::self_test()?;
        println!(
            "rope-i BSHD vs candle rope_i: cosine = {rope_cos:.6} ({})",
            if rope_cos > 0.999 { "OK" } else { "TOO LOW" }
        );
        let (bshd_cos, maxabs) = qwen_image_rs::sage::self_test_bshd()?;
        let bshd_ok = bshd_cos > 0.9999 && maxabs.is_finite() && maxabs < 0.01;
        println!(
            "sage BSHD vs BHSD (block-causal split): cosine = {bshd_cos:.6}, maxabs = {maxabs:.4} ({})",
            if bshd_ok { "OK" } else { "MISMATCH" }
        );
        let (pt_nan, pt_diff) = qwen_image_rs::sage::self_test_partial_tile()?;
        let pt_ok = pt_nan == 0 && pt_diff == 0.0;
        println!(
            "sage partial last K/V tile under NaN-poisoned smem (txt=21, S=293): NaN = {pt_nan}, maxabs vs clean = {pt_diff:.4} ({})",
            if pt_ok { "OK" } else { "FAIL" }
        );
        let rq = qwen_image_rs::sage::self_test_rope_quant()?;
        let rq_ok = rq.int8_mismatches == 0
            && rq.scale_mismatches == 0
            && rq.attn_maxabs == 0.0
            && rq.attn_cos > 0.9999;
        println!(
            "fused rope+quant vs rope->quant (4 narrows, B=1,2): int8 mismatches = {}, scale mismatches = {}, attn cosine = {:.6}, maxabs = {:.4} ({})",
            rq.int8_mismatches,
            rq.scale_mismatches,
            rq.attn_cos,
            rq.attn_maxabs,
            if rq_ok { "OK" } else { "MISMATCH" }
        );
        #[cfg(feature = "sage2")]
        let s2_ok = {
            let r = qwen_image_rs::sage2::self_test()?;
            println!(
                "sage2 block-causal (txt=37, S=293, B=1,2, K bias, zero V/K channel) vs f32 ref: cosine fp16-accum = {:.6}, fp32-accum = {:.6} (v1 on same inputs {:.6}); non-finite = {} ({})",
                r.cos_f16,
                r.cos_f32,
                r.cos_v1,
                r.nonfinite,
                if r.cos_f16 >= 0.999 && r.cos_f32 >= 0.999 && r.nonfinite == 0 { "OK" } else { "FAIL" }
            );
            println!(
                "sage2 B=2 lanes vs B=1: mismatches = {}; offset views vs copies: mismatches = {}; run-twice: mismatches = {} ({})",
                r.lane_mismatches,
                r.offset_mismatches,
                r.nondeterministic,
                if r.lane_mismatches == 0 && r.offset_mismatches == 0 && r.nondeterministic == 0 { "OK" } else { "FAIL" }
            );
            println!(
                "sage2 partial last K/V tile under NaN-poisoned smem (txt=21, S=293): NaN = {}, maxabs vs clean = {:.4} ({})",
                r.poison_nan,
                r.poison_maxabs,
                if r.poison_nan == 0 && r.poison_maxabs == 0.0 { "OK" } else { "FAIL" }
            );
            println!(
                "sage2 fused per-layer quant (3 launches) vs per-op kernels (B=1,2, offset views, txt=37/21, fp16+fp32 accum): payload/scale/output mismatches = {} ({})",
                r.fused_mismatches,
                if r.fused_mismatches == 0 { "BIT-IDENTICAL" } else { "MISMATCH" }
            );
            r.ok()
        };
        #[cfg(not(feature = "sage2"))]
        let s2_ok = true;
        if !rq_ok {
            anyhow::bail!("fused rope+quant is not bit-exact vs rope->quant");
        }
        if !s2_ok {
            anyhow::bail!("sage2 self-test failed");
        }
        if !pt_ok {
            anyhow::bail!("sage partial-tile attention reads stale shared memory");
        }
        Ok(())
    }
    #[cfg(not(feature = "sage"))]
    anyhow::bail!("build with --features sage")
}

/// Self-test the fused LayerNorm+AdaLN kernel vs the candle reference.
/// Print the offset-view regression results and fail if any bridge read the
/// wrong rows (`qwen-image-rs-b1-off-prompt`).
#[cfg(any(feature = "fusednorm", feature = "convrot"))]
fn report_offset_views(group: &str, results: Vec<(&'static str, bool)>) -> Result<()> {
    let mut bad = Vec::new();
    for (op, ok) in results {
        println!(
            "{group} {op} on a nonzero-offset view vs fresh copy: {}",
            if ok { "BIT-IDENTICAL" } else { "MISMATCH" }
        );
        if !ok {
            bad.push(op);
        }
    }
    if !bad.is_empty() {
        anyhow::bail!("{group}: bridges ignore the view offset: {bad:?}");
    }
    Ok(())
}

fn fusednorm_test() -> Result<()> {
    #[cfg(feature = "fusednorm")]
    {
        let cos = qwen_image_rs::fusednorm::self_test()?;
        println!(
            "fused norm+AdaLN vs candle: cosine = {cos:.6} ({})",
            if cos > 0.999 { "OK" } else { "TOO LOW" }
        );
        for n in [4096usize, 128usize] {
            let cos = qwen_image_rs::fusednorm::self_test_rmsnorm(n)?;
            println!(
                "fused RMSNorm×weight (N={n}) vs candle: cosine = {cos:.6} ({})",
                if cos > 0.9999 { "OK" } else { "TOO LOW" }
            );
        }
        let cos = qwen_image_rs::fusednorm::self_test_gated()?;
        println!(
            "fused gated residual vs candle: cosine = {cos:.6} ({})",
            if cos > 0.9999 { "OK" } else { "TOO LOW" }
        );
        report_offset_views(
            "fusednorm",
            qwen_image_rs::fusednorm::self_test_offset_views()?,
        )
    }
    #[cfg(not(feature = "fusednorm"))]
    anyhow::bail!("build with --features fusednorm")
}

/// Micro-benchmark the DiT's per-layer dominant ops in bf16 on the active
/// device, to see whether the workload is GEMM-bound or attention-bound.
fn bench(seq: usize, iters: usize) -> Result<()> {
    let dev = device::best_device()?;
    let dt = if matches!(dev, candle_core::Device::Cuda(_)) {
        DType::BF16
    } else {
        DType::F32
    };
    let (h, inter, heads, hd) = (4096usize, 12288usize, 32usize, 128usize);
    let x = Tensor::randn(0f32, 1f32, (seq, h), &dev)?.to_dtype(dt)?;
    let wqkv = Tensor::randn(0f32, 1f32, (3 * h, h), &dev)?.to_dtype(dt)?;
    let wo = Tensor::randn(0f32, 1f32, (h, h), &dev)?.to_dtype(dt)?;
    let wgate = Tensor::randn(0f32, 1f32, (inter, h), &dev)?.to_dtype(dt)?;
    let wup = Tensor::randn(0f32, 1f32, (inter, h), &dev)?.to_dtype(dt)?;
    let wdown = Tensor::randn(0f32, 1f32, (h, inter), &dev)?.to_dtype(dt)?;
    let q = Tensor::randn(0f32, 1f32, (heads, seq, hd), &dev)?.to_dtype(dt)?;
    let k = Tensor::randn(0f32, 1f32, (heads, seq, hd), &dev)?.to_dtype(dt)?;
    let v = Tensor::randn(0f32, 1f32, (heads, seq, hd), &dev)?.to_dtype(dt)?;

    let time = |name: &str, f: &dyn Fn() -> Result<()>| -> Result<()> {
        f()?; // warmup
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        dev.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("  {name:20} {ms:8.3} ms/iter");
        Ok(())
    };

    println!("seq={seq} dtype={dt:?} iters={iters}");
    time("attn_proj (qkv+o)", &|| {
        let _ = x.matmul(&wqkv.t()?)?;
        let _ = x.matmul(&wo.t()?)?;
        Ok(())
    })?;
    time("mlp (gate+up+down)", &|| {
        let g = candle_nn::ops::silu(&x.matmul(&wgate.t()?)?)?;
        let u = x.matmul(&wup.t()?)?;
        let _ = (g * u)?.matmul(&wdown.t()?)?;
        Ok(())
    })?;
    time("attention (S^2)", &|| {
        let scale = 1.0 / (hd as f64).sqrt();
        let a = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let a = candle_nn::ops::softmax_last_dim(&a)?;
        let _ = a.matmul(&v)?;
        Ok(())
    })?;
    Ok(())
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

/// Pre-quantize the DiT's ConvRot linears (every weight
/// `dit::is_convrot_target` names: the 224 block linears + the rotated tail
/// linears) to one safetensors file — the same builder (and header tags) as
/// the default `--convrot` cache (`convrot_cache`), written atomically. Each
/// target `<prefix>.weight` (bf16) becomes `<prefix>.weight_i8` (rotated INT8,
/// U8 bytes) + `<prefix>.col_scale` (f32); all other tensors are copied bf16.
/// A dir holding this file is used as-is by `--convrot` (no cache). Without
/// `out`, the file goes to `weights`' cache entry (what `--convrot` loads).
fn prequantize_convrot(
    weights: &std::path::Path,
    out: Option<&std::path::Path>,
    cache_dir: Option<&std::path::Path>,
) -> Result<()> {
    use qwen_image_rs::convrot_cache as cc;
    let files = WeightSet::resolve(weights)?.files;
    anyhow::ensure!(
        !cc::files_hold_prequant(&files)?,
        "{} is already prequantized (holds *.weight_i8)",
        weights.display()
    );
    let out = match out {
        Some(o) => o.to_path_buf(),
        None => {
            let root = cc::CacheSpec::from_cli(cache_dir, false, false)
                .root
                .context("no cache dir: pass --out or --convrot-cache (or set HOME)")?;
            cc::entry_path(&root, weights)?
        }
    };
    let out = out.as_path();
    let meta = std::collections::HashMap::from([
        (
            cc::META_POLICY.to_string(),
            qwen_image_rs::model::dit::convrot_policy_tag(),
        ),
        (
            cc::META_SOURCE.to_string(),
            cc::source_fingerprint(weights, &files)?,
        ),
    ]);
    let n_quant = cc::build(&files, out, meta)?;
    println!(
        "prequantized {n_quant} convrot linears -> {}",
        out.display()
    );
    Ok(())
}

/// The Qwen3-VL text-encoder decoder linears (Q8_0-quantized in the GGUF);
/// everything else (embedding, norms) is stored F16.
fn is_text_linear(name: &str) -> bool {
    const SUFFIXES: [&str; 7] = [
        ".q_proj.weight",
        ".k_proj.weight",
        ".v_proj.weight",
        ".o_proj.weight",
        ".gate_proj.weight",
        ".up_proj.weight",
        ".down_proj.weight",
    ];
    name.contains(".layers.") && SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// Pre-quantize the text encoder to a Q8_0 GGUF (linears Q8_0, embedding+norms
/// F16), containing only the tensors the text-only encoder loads. Loaded later
/// via `--text-gguf` without ever materializing a bf16 weight on the GPU.
fn prequantize_text(weights: &std::path::Path, out: &std::path::Path) -> Result<()> {
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use std::collections::HashMap;

    let set = WeightSet::resolve(weights)?;
    tracing::info!(files = set.files.len(), "loading bf16 text encoder (CPU)");
    let mut full: HashMap<String, Tensor> = HashMap::new();
    for f in &set.files {
        for (k, v) in candle_core::safetensors::load(f, &candle_core::Device::Cpu)? {
            full.insert(k, v);
        }
    }
    // Only the tensors the text-only encoder uses: the language-model embedding
    // and decoder layers (skips the vision tower, final norm, lm_head).
    let mut names: Vec<String> = full
        .keys()
        .filter(|n| {
            n.starts_with("model.language_model.embed_tokens")
                || n.starts_with("model.language_model.layers.")
        })
        .cloned()
        .collect();
    names.sort();
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    let mut n_q8 = 0usize;
    for name in &names {
        let t = &full[name];
        let dt = if is_text_linear(name) {
            n_q8 += 1;
            GgmlDType::Q8_0
        } else {
            GgmlDType::F16
        };
        qtensors.push((name.clone(), QTensor::quantize(t, dt)?));
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let mut f = std::fs::File::create(out)?;
    gguf_file::write(&mut f, &[], &refs)?;
    println!(
        "wrote {} tensors ({n_q8} Q8_0 linears + {} F16) -> {}",
        refs.len(),
        refs.len() - n_q8,
        out.display()
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
