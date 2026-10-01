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

Two conversions produce the files the fast/resident paths use. Both are built
into the binary; `scripts/convert.sh <snapshot> [out-dir]` runs both.
`prequantize-convrot` needs CUDA (the quantize kernel is GPU-only).

### Text encoder → Q8_0 GGUF (recommended for the resident/batch path)
```sh
qwen-image-rs prequantize-text \
  --weights <snapshot>/text_encoder --out qir-text.gguf      # ~8 GB
```
Use it with `--text-gguf qir-text.gguf`. This loads Q8_0 directly and never puts
a bf16 copy on the GPU, so the CUDA pool stays small — which is what lets all
three models stay resident (and batch) inside 24 GB.

### DiT → ConvRot INT8 (automatic: the `--convrot` cache)
`--convrot` loads a **prequantized** DiT (rotated INT8 `weight_i8` + `col_scale`
for the 231 ConvRot linears, the rest bf16; ~7.1 GB vs ~15 GB bf16). The first
`--convrot` run for a transformer dir builds it (one-time, ~26 s on the 4090)
into a cache entry; every later run loads it directly — no rotate+quantize
kernels, half the bytes read. Output is byte-identical to quantizing on load.

- **Where:** `<root>/<snapshot>-<hash>/transformer_convrot.safetensors`, root =
  `--convrot-cache DIR` > `$QIR_CONVROT_CACHE` >
  `$XDG_CACHE_HOME/qwen-image-rs/convrot` > `~/.cache/qwen-image-rs/convrot`.
  One entry per transformer location; delete the dir to reclaim the space.
- **Invalidation:** the file header records the precision policy
  (`dit::convrot_policy_tag`: format version, rotation group, which linears
  rotate) and a fingerprint of the source files (path, names, sizes, mtimes).
  A mismatch, a torn file or a missing tag rebuilds the entry in place.
- **Flags** (generate / batch / denoise / dit-forward): `--no-convrot-cache`
  quantizes on load (the old path); `--rebuild-convrot-cache` forces a rebuild.
  If the cache cannot be built or written (no CUDA, read-only dir, disk full),
  the run warns and quantizes on load — same output.
- **Pre-warm** (e.g. in `scripts/convert.sh`):
  `qwen-image-rs prequantize-convrot --weights <snapshot>/transformer`.
  With `--out FILE` it writes a standalone file instead; a `--weights` dir that
  holds such a file is used as-is (no cache).

Load time (dit-forward `--convrot`, same session): cold page cache 7.0 s → 2.7 s,
warm 2.1 s → 1.1 s.

## 3. Recommended run

```sh
# fastest build
# CUTLASS_DIR set; cudnn needs the user-space cuDNN wheel (scripts/setup-host.sh)
cargo build --release --features convrot,sage,fusednorm,sage2,cudnn

# resident batch: N seed-variations, ~11 s/image
qwen-image-rs batch --model <snapshot> --prompts prompts.txt \
  --resident --convrot --text-gguf qir-text.gguf --steps 40 \
  --out-dir out/

# single image with N seeds (SDXL-style grid)
qwen-image-rs generate --model <snapshot> --prompt "..." \
  --convrot --text-gguf qir-text.gguf --batch 4 --out-dir out/
```

## 4. Optional: regenerate the validation oracle

`scripts/oracle.py` runs the original diffusers model (fixed seed) and saves the
reference latents/images every port phase is validated against. Needs the Python
venv from `scripts/setup-host.sh`. Not required to run inference.
