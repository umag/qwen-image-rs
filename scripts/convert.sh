#!/usr/bin/env bash
# Convert Qwen-Image-2.1 weights to the quantized formats qwen-image-rs uses.
# Run on the CUDA host after building:
#   cargo build --release --features convrot,sage,fusednorm   (CUTLASS_DIR set)
#
#   scripts/convert.sh <snapshot-dir> [out-dir]
#
# Produces, in <out-dir> (default <snapshot>/qir):
#   qir-text.gguf                  Q8_0 text encoder      -> --text-gguf
#   transformer_convrot.safetensors  ConvRot INT8 DiT     -> --convrot (auto-detected)
#
# `prequantize-convrot` needs CUDA; `prequantize-text` runs on CPU too.
# Override the binary path with QIR_BIN (default: target/release/qwen-image-rs).
set -euo pipefail

SNAP="${1:?usage: convert.sh <snapshot-dir> [out-dir]}"
OUT="${2:-$SNAP/qir}"
BIN="${QIR_BIN:-target/release/qwen-image-rs}"

[ -d "$SNAP/text_encoder" ] || { echo "no $SNAP/text_encoder" >&2; exit 1; }
[ -d "$SNAP/transformer" ]  || { echo "no $SNAP/transformer"  >&2; exit 1; }
mkdir -p "$OUT"

echo "== text encoder -> Q8_0 GGUF =="
"$BIN" prequantize-text --weights "$SNAP/text_encoder" --out "$OUT/qir-text.gguf"

echo "== DiT -> ConvRot INT8 =="
"$BIN" prequantize-convrot --weights "$SNAP/transformer" --out "$OUT/transformer_convrot.safetensors"

echo "converted -> $OUT"
echo "  --text-gguf $OUT/qir-text.gguf"
echo "  --convrot   (auto-detects $OUT/transformer_convrot.safetensors)"
