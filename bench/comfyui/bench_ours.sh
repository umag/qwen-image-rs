#!/usr/bin/env bash
# Same-session re-measure of qwen-image-rs for the ComfyUI side-by-side.
#  1) cold single `generate` (process start -> PNG, warm file cache)
#  2) `batch --resident` over the SAME 4 prompts the ComfyUI client queues;
#     steady = images 1..3 (encode_ms + denoise_ms + decode_ms per image).
# GPU memory/power sampled with nvidia-smi exactly like run_variant.sh.
set -uo pipefail
BIN=${QIR_BIN:-$HOME/.cache/qwen-image-rs-target/release/qwen-image-rs}
SNAP=${QIR_SNAP:-$HOME/dev_tmp/weights/hf/hub/models--Qwen--Qwen-Image-2.1/snapshots/b3179ad355be050328e483a9dfdd9e60cd62adfa}
GGUF=${QIR_GGUF:-$HOME/dev_tmp/qir-text.gguf}
OUT=${OUT:-$HOME/dev_tmp/comfy-bench/results/ours}
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
rm -rf "$OUT"; mkdir -p "$OUT/img"
cat > "$OUT/prompts.txt" <<'EOF'
a red coffee mug on a wooden table
a lighthouse on a cliff at sunset
a fox in a snowy forest
a bowl of ramen, studio photo
EOF
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$OUT/baseline_mib"
nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits -lms 250 > "$OUT/gpu.csv" &
SMI=$!
T0=$(date +%s.%N)
"$BIN" generate --model "$SNAP" --prompt "a red coffee mug on a wooden table" --convrot \
  --text-gguf "$GGUF" --steps 40 --out "$OUT/img/cold.png" > "$OUT/generate.log" 2>&1
echo "cold generate wall: $(echo "$(date +%s.%N) - $T0" | bc) s"
T0=$(date +%s.%N)
"$BIN" batch --model "$SNAP" --prompts "$OUT/prompts.txt" --resident --convrot \
  --text-gguf "$GGUF" --steps 40 --out-dir "$OUT/img" > "$OUT/batch.log" 2>&1
echo "batch wall: $(echo "$(date +%s.%N) - $T0" | bc) s"
kill $SMI; wait 2>/dev/null
echo "peak_mib=$(awk -F', ' '{if($2>m)m=$2} END{print m}' "$OUT/gpu.csv") baseline_mib=$(cat "$OUT/baseline_mib") mean_load_power_W=$(awk -F', ' '$3>300{s+=$3;n++} END{if(n) printf "%.0f", s/n}' "$OUT/gpu.csv")"
grep -iE "denoise|decode_ms|encode|total|image|vram" "$OUT/generate.log" | tail -12
echo ---
grep -iE "denoise|decode_ms|encode|total|image|vram" "$OUT/batch.log" | tail -20
ls "$OUT/img"
