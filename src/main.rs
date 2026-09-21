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
        #[arg(long)]
        out: std::path::PathBuf,
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
        /// Output safetensors path (joint output tensor).
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Decode a saved latent (.pt) through the VAE to an RGBA PNG (Phase 2).
    VaeDecode {
        /// The vae/ directory (holds config.json + *.safetensors).
        #[arg(long)]
        weights: std::path::PathBuf,
        /// A packed latent saved by scripts/oracle.py (`*.latent.pt`).
        #[arg(long)]
        latent: std::path::PathBuf,
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
        Command::Bench { seq, iters } => bench(seq, iters),
        Command::Batch {
            model,
            prompts,
            size,
            steps,
            seed,
            quant,
            out_dir,
        } => batch(&model, &prompts, size, steps, seed, quant, &out_dir),
        Command::Info { weights } => info(&weights),
        Command::Generate {
            model,
            prompt,
            size,
            steps,
            seed,
            quant,
            out,
        } => generate(&model, &prompt, size, steps, seed, quant, &out),
        Command::VaeDecode {
            weights,
            latent,
            out,
        } => vae_decode(&weights, &latent, &out),
        Command::TextEncode {
            weights,
            input_ids,
            drop,
            out,
        } => text_encode(&weights, &input_ids, drop, &out),
        Command::DitForward {
            weights,
            inputs,
            quant,
            out,
        } => dit_forward(&weights, &inputs, quant, &out),
        Command::Denoise {
            weights,
            noise,
            embeds,
            steps,
            quant,
            out,
        } => denoise(&weights, &noise, &embeds, steps, quant, &out),
    }
}

/// Standalone text-to-image: prompt -> PNG. Loads the three models one at a
/// time (text encoder -> DiT -> VAE), freeing each before the next so the
/// pipeline fits in 24 GB.
fn generate(
    model: &std::path::Path,
    prompt: &str,
    size: usize,
    steps: usize,
    seed: u64,
    quant: bool,
    out: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::dit::QwenImageDit;
    use qwen_image_rs::model::scheduler::{FlowConfig, FlowMatchEuler};
    use qwen_image_rs::model::text_encoder::{prompt as tmpl, QwenTextEncoder, TextConfig};
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

    // Tokenize the t2i chat template; drop = system-prefix token count.
    let tok = Tokenizer::from_file(model.join("processor/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let ids: Vec<u32> = tok
        .encode(tmpl::t2i_template(prompt), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    let sysp = format!("<|im_start|>system\n{}<|im_end|>\n", tmpl::SYS_PROMPT);
    let drop = tok
        .encode(sysp, false)
        .map_err(|e| anyhow::anyhow!("encode sys: {e}"))?
        .get_ids()
        .len();
    let seq = ids.len();
    tracing::info!(seq, drop, "tokenized");

    // 1. Text encoder -> prompt embeddings (freed after).
    let embeds = {
        let te = QwenTextEncoder::load(
            &TextConfig::default(),
            load_vb(model.join("text_encoder"), dtype)?,
        )?;
        let ids_t = Tensor::from_vec(ids, (1, seq), &dev)?;
        let hidden = te.forward(&ids_t)?;
        hidden.narrow(1, drop, seq - drop)?.contiguous()?
    };
    tracing::info!(shape = ?embeds.dims(), "prompt embeddings");

    // 2. DiT + flow-match denoise -> final latent (freed after).
    let hw = size / 16; // vae spatial compression
    let img_seq = hw * hw;
    let latent = {
        let dit = QwenImageDit::load(32, 64, quant, load_vb(model.join("transformer"), dtype)?)?;
        dev.set_seed(seed)?;
        let mut latents =
            Tensor::randn(0f32, 1f32, (1, img_seq, 64), &dev)?.to_dtype(DType::F32)?;
        let sched = FlowMatchEuler::new(&FlowConfig::default(), steps, img_seq);
        for (i, t) in sched.timesteps().iter().enumerate() {
            let tt = Tensor::from_vec(vec![(*t / 1000.0) as f32], (1,), &dev)?;
            let joint = dit.forward(&latents.to_dtype(dtype)?, &embeds, &tt, hw, hw)?;
            let (_b, jl, _) = joint.dims3()?;
            let np = joint
                .narrow(1, jl - img_seq, img_seq)?
                .to_dtype(DType::F32)?;
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
        load_vb(model.join("vae"), DType::F32)?,
    )?;
    let z = vae::unpack_latents(&latent, 64)?;
    let img = vmodel.decode(&z)?;
    let (w, h, bytes) = vae::to_rgba_u8(&img)?;
    let buf: image::RgbaImage =
        image::ImageBuffer::from_raw(w as u32, h as u32, bytes).context("image buffer")?;
    buf.save(out)?;
    println!("wrote {} ({w}x{h})", out.display());
    Ok(())
}

/// Batch generation: load each model once for all prompts (encode all -> free
/// -> denoise all -> free -> decode all), so the ~24s of model loads is paid
/// once instead of per image.
fn batch(
    model: &std::path::Path,
    prompts_path: &std::path::Path,
    size: usize,
    steps: usize,
    seed: u64,
    quant: bool,
    out_dir: &std::path::Path,
) -> Result<()> {
    use qwen_image_rs::model::dit::QwenImageDit;
    use qwen_image_rs::model::scheduler::{FlowConfig, FlowMatchEuler};
    use qwen_image_rs::model::text_encoder::{prompt as tmpl, QwenTextEncoder, TextConfig};
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

    // Phase 1: encode every prompt (text encoder loaded once, then freed).
    let mut embeds_list = Vec::with_capacity(prompts.len());
    {
        let te = QwenTextEncoder::load(
            &TextConfig::default(),
            load_vb(model.join("text_encoder"), dtype)?,
        )?;
        for p in &prompts {
            let ids: Vec<u32> = tok
                .encode(tmpl::t2i_template(p), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            let seq = ids.len();
            let hidden = te.forward(&Tensor::from_vec(ids, (1, seq), &dev)?)?;
            embeds_list.push(hidden.narrow(1, drop, seq - drop)?.contiguous()?);
        }
    }
    tracing::info!("encoded {} prompts", embeds_list.len());

    // Phase 2: denoise every prompt (DiT loaded once, then freed).
    let hw = size / 16;
    let img_seq = hw * hw;
    let mut latents_list = Vec::with_capacity(prompts.len());
    {
        let dit = QwenImageDit::load(32, 64, quant, load_vb(model.join("transformer"), dtype)?)?;
        let sched = FlowMatchEuler::new(&FlowConfig::default(), steps, img_seq);
        for (i, emb) in embeds_list.iter().enumerate() {
            dev.set_seed(seed + i as u64)?;
            let mut latents =
                Tensor::randn(0f32, 1f32, (1, img_seq, 64), &dev)?.to_dtype(DType::F32)?;
            for (si, t) in sched.timesteps().iter().enumerate() {
                let tt = Tensor::from_vec(vec![(*t / 1000.0) as f32], (1,), &dev)?;
                let joint = dit.forward(&latents.to_dtype(dtype)?, emb, &tt, hw, hw)?;
                let (_b, jl, _) = joint.dims3()?;
                let np = joint
                    .narrow(1, jl - img_seq, img_seq)?
                    .to_dtype(DType::F32)?;
                latents = (latents + (np * sched.dt(si))?)?;
            }
            latents_list.push(latents);
            tracing::info!(image = i, "denoised");
        }
    }

    // Phase 3: decode every latent (VAE loaded once).
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
        load_vb(model.join("vae"), DType::F32)?,
    )?;
    for (i, lat) in latents_list.iter().enumerate() {
        let img = vmodel.decode(&vae::unpack_latents(lat, 64)?)?;
        let (w, h, bytes) = vae::to_rgba_u8(&img)?;
        let buf: image::RgbaImage =
            image::ImageBuffer::from_raw(w as u32, h as u32, bytes).context("image buffer")?;
        buf.save(out_dir.join(format!("{i:03}.png")))?;
    }
    println!("wrote {} images to {}", prompts.len(), out_dir.display());
    Ok(())
}

/// Phase 4: full flow-match denoise loop -> final latent.
fn denoise(
    weights: &std::path::Path,
    noise: &std::path::Path,
    embeds: &std::path::Path,
    steps: usize,
    quant: bool,
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

    let set = WeightSet::resolve(weights)?;
    let files = set.files.clone();
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
    let model = QwenImageDit::load(32, 64, quant, vb)?;

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

    let set = WeightSet::resolve(weights)?;
    let files = set.files.clone();
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
    let model = QwenImageDit::load(32, 64, quant, vb)?;

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
    let model = QwenTextEncoder::load(&cfg, vb)?;

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
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, DType::F32, &dev)? };
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

/// Self-test the ConvRot INT8 GEMM bridge end to end.
fn convrot_test() -> Result<()> {
    #[cfg(feature = "convrot")]
    {
        let maxdiff = qwen_image_rs::convrot::self_test()?;
        println!(
            "convrot int8-gemm bridge: max abs diff vs f32 = {maxdiff} ({})",
            if maxdiff == 0.0 {
                "EXACT — bridge OK"
            } else {
                "MISMATCH"
            }
        );
        Ok(())
    }
    #[cfg(not(feature = "convrot"))]
    anyhow::bail!("build with --features convrot (needs CUTLASS_DIR)")
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
