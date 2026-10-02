#!/usr/bin/env bash
# Same-session re-measure of qwen-image-rs (convrot,sage,fusednorm,sage2,cudnn
# build) for the side-by-side: `batch --resident`, 3 prompts x 40 steps, so the
# DiT runs 120 consecutive forwards (sustained, power-capped). Prints the
# per-image timing lines; steady s/step = denoise_ms / 40 of images 2 and 3.
set -euo pipefail
BIN=${QIR_BIN:-$HOME/.cache/qwen-image-rs-target/release/qwen-image-rs}
SNAP=${QIR_SNAP:-$HOME/dev_tmp/weights/hf/hub/models--Qwen--Qwen-Image-2.1/snapshots/b3179ad355be050328e483a9dfdd9e60cd62adfa}
OUT=${OUT:-/tmp/qir-trt-cmp}
mkdir -p "$OUT"
cat > "$OUT/prompts.txt" <<'EOF'
a red ceramic coffee mug on a wooden table, soft morning light
a storefront window with a neon sign reading "QWEN", rainy night, reflections
an isometric low-poly island with a waterfall, pastel colors
EOF
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
"$BIN" batch --model "$SNAP" --prompts "$OUT/prompts.txt" --resident --convrot \
  --text-gguf "$HOME/dev_tmp/qir-text.gguf" --steps 40 --out-dir "$OUT" > "$OUT/run.log" 2>&1
grep -iE "denoise|step|vram|total|image" "$OUT/run.log" | tail -40
