# Weights & model conversion

This repo ships **no weights**. Qwen-Image-2.1 is under the **Qwen Research
License** (research-only); download the weights yourself.

## 1. Get the weights

Model: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) on
Hugging Face (accept the license first).

```sh
# huggingface_hub CLI (pip install -U "huggingface_hub[cli]")
hf download Qwen/Qwen-Image-2.1 --local-dir weights/qwen-image-2.1
```

The snapshot must contain these subdirs (the pipeline reads them by name):

```
<snapshot>/processor/      tokenizer.json
<snapshot>/text_encoder/   Qwen3-VL 8B, bf16 safetensors
<snapshot>/transformer/    DiT 7B, bf16 safetensors + config.json
<snapshot>/vae/            AutoencoderKLQwenImage21 + config.json
```

That is all you need. Point `--model <snapshot>` at it and it runs in bf16
directly — **no conversion required**:

```sh
qwen-image-rs generate --model weights/qwen-image-2.1 \
  --prompt "a red ceramic coffee mug on a wooden table" \
  --steps 40 --seed 42 --out out.png
```

## 2. Optional: convert to the quantized formats (lower VRAM / faster)

Two optional conversions produce the files the fast/resident paths use. Both are
built into the binary; `scripts/convert.sh <snapshot> [out-dir]` runs both.
`prequantize-convrot` needs CUDA (the quantize kernel is GPU-only).

### Text encoder → Q8_0 GGUF (recommended for the resident/batch path)
```sh
qwen-image-rs prequantize-text \
  --weights <snapshot>/text_encoder --out qir-text.gguf      # ~8 GB
```
Use it with `--text-gguf qir-text.gguf`. This loads Q8_0 directly and never puts
a bf16 copy on the GPU, so the CUDA pool stays small — which is what lets all
three models stay resident (and batch) inside 24 GB.

### DiT → ConvRot INT8 (optional; cold-start / VRAM only)
```sh
qwen-image-rs prequantize-convrot \
  --weights <snapshot>/transformer \
  --out <dir>/transformer.safetensors                        # ~6.8 GB (~55% smaller)
```
Bit-exact vs on-the-fly `--convrot` (cosine 1.0). It only saves the load-time
rotate+quant, so it helps repeated loads / serve, not steady-state speed. The
common path needs no file: just pass `--convrot` and the DiT is quantized on load.

## 3. Recommended run

```sh
# fastest build
cargo build --release --features convrot,sage,fusednorm    # CUTLASS_DIR set

# resident batch: N seed-variations, ~11 s/image
qwen-image-rs batch --model <snapshot> --prompts prompts.txt \
  --resident --convrot --text-gguf qir-text.gguf --vae-tile 32 --steps 40 \
  --out-dir out/

# single image with N seeds (SDXL-style grid)
qwen-image-rs generate --model <snapshot> --prompt "..." \
  --convrot --text-gguf qir-text.gguf --vae-tile 32 --batch 4 --out-dir out/
```

## 4. Optional: regenerate the validation oracle

`scripts/oracle.py` runs the original diffusers model (fixed seed) and saves the
reference latents/images every port phase is validated against. Needs the Python
venv from `scripts/setup-host.sh`. Not required to run inference.
