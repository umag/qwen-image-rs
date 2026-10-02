#!/usr/bin/env bash
# Export -> build -> bench (+ per-layer profile) for each named variant.
#   bash run_variants.sh bf16 fp8 "fp8 --mha" "int8sq --alpha 0.8"
# The engine name is the variant with flags folded in: fp8-mha, int8sq.
PY=${PY:-$HOME/dev_tmp/trt-bench/.venv/bin/python}
cd "$(dirname "$0")"
for v in "$@"; do
  n=$(echo "$v" | sed -e 's/ --mha/-mha/' -e 's/ --alpha [0-9.]*//')
  echo "=== $n"
  if [ "$n" = bf16 ]; then $PY export_onnx.py --quant none; else $PY export_onnx.py --quant $v; fi \
    && /usr/bin/time -f "build wall %e s, max RSS %M KB" $PY build_engine.py "$n" \
    && $PY bench_trt.py "$n" --layers | grep -E "^\[trt|engine layers" \
    && $PY bench_trt.py "$n" --n 5 --profile | grep -E "^profile|^by class"
  echo "=== END $n rc=$?"
done
echo PIPELINE_END
