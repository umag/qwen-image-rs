#!/usr/bin/env bash
# Convert Qwen-Image-2.1 weights to the quantized formats qwen-image-rs uses.
# Run on the CUDA host after building:
#   cargo build --release --features convrot,sage,fusednorm,sage2   (CUTLASS_DIR set)
#
#   scripts/convert.sh <snapshot-dir> [out-dir]
#
# Produces:
#   <out-dir>/qir-text.gguf   Q8_0 text encoder (default out-dir <snapshot>/qir) -> --text-gguf
#   the --convrot cache entry for <snapshot>/transformer (ConvRot INT8 DiT):
#     $QIR_CONVROT_CACHE or ~/.cache/qwen-image-rs/convrot/<snapshot>-<hash>/
#   --convrot builds that entry by itself on first use; this just pre-warms it.
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

echo "== DiT -> ConvRot INT8 (pre-warm the --convrot cache) =="
"$BIN" prequantize-convrot --weights "$SNAP/transformer"

echo "converted -> $OUT"
echo "  --text-gguf $OUT/qir-text.gguf"
echo "  --convrot   (loads the cached prequantized DiT)"
