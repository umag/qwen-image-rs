#!/usr/bin/env bash
# One benchmark variant: fresh ComfyUI server -> bench.py -> kill server.
#   run_variant.sh <name> "<server flags>" <bench.py args...>
# Output: $RES/<name>/{server.log,gpu.csv,result.json,bench.log,startup_s}
set -uo pipefail
NAME=$1; FLAGS=$2; shift 2
ROOT=${COMFY_BENCH:-$HOME/dev_tmp/comfy-bench}
RES=${RES:-$ROOT/results}
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8188}
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
OUT=$RES/$NAME; rm -rf "$OUT"; mkdir -p "$OUT"
PY=$ROOT/venv/bin/python
cd "$ROOT/ComfyUI"
rm -rf "output/$NAME"

# idle baseline (WSL nvidia-smi reports device-wide memory incl. the desktop)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$OUT/baseline_mib"
nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits -lms 250 > "$OUT/gpu.csv" &
SMI=$!

T0=$(date +%s.%N)
# shellcheck disable=SC2086
setsid "$PY" main.py --listen 127.0.0.1 --port "$PORT" --disable-auto-launch $FLAGS > "$OUT/server.log" 2>&1 < /dev/null &
SRV=$!
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -40 "$OUT/server.log"; kill $SMI; exit 1; }
  sleep 0.5
done
echo "$(echo "$(date +%s.%N) - $T0" | bc)" > "$OUT/startup_s"
echo "[$NAME] server up in $(cat "$OUT/startup_s")s flags: $FLAGS"

"$PY" "$HERE/bench.py" --host "127.0.0.1:$PORT" --variant "$NAME" --out "$OUT/result.json" "$@" 2>&1 | tee "$OUT/bench.log"
RC=${PIPESTATUS[0]}

kill -- -$SRV 2>/dev/null; kill $SRV 2>/dev/null; sleep 3; kill -9 -- -$SRV 2>/dev/null
kill $SMI 2>/dev/null
wait 2>/dev/null
BASE=$(cat "$OUT/baseline_mib")
PEAK=$(awk -F', ' '{if($2>m)m=$2} END{print m}' "$OUT/gpu.csv")
PW=$(awk -F', ' '$3>300{s+=$3;n++} END{if(n) printf "%.0f", s/n; else print "na"}' "$OUT/gpu.csv")
echo "[$NAME] rc=$RC peak_mib=$PEAK baseline_mib=$BASE (delta $((PEAK-BASE))) mean_power_under_load_W=$PW"
echo "{\"peak_mib\":$PEAK,\"baseline_mib\":$BASE,\"mean_load_power_w\":\"$PW\",\"startup_s\":$(cat "$OUT/startup_s")}" > "$OUT/gpu_summary.json"
grep -iE "loaded (completely|partially)|offload|lowvram|Requested to load|attention|fast|kitchen|VRAM|prompt executed|Error" "$OUT/server.log" | head -60
exit $RC
