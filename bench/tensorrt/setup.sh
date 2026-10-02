#!/usr/bin/env bash
# Build ~/dev_tmp/trt-bench/.venv: an exact clone of the oracle venv (same torch
# 2.14+cu130 / diffusers, installed from `uv pip freeze` so the uv cache makes it
# cheap) plus TensorRT, ONNX and NVIDIA ModelOpt. The oracle venv is read only.
# User space only, no sudo.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
ORACLE=${ORACLE_VENV:-$HOME/dev_tmp/qwen-image-oracle/.venv}
ROOT=${TRT_BENCH:-$HOME/dev_tmp/trt-bench}
V=$ROOT/.venv
mkdir -p "$ROOT"
[ -x "$V/bin/python" ] || uv venv --python 3.12 "$V"
uv pip freeze --python "$ORACLE" > "$ROOT/oracle-freeze.txt"
uv pip install --python "$V" -r "$ROOT/oracle-freeze.txt" 2>&1 | tail -3
# Pin torch/diffusers to the oracle's versions while adding the TRT stack.
grep -E '^(torch|torchvision|triton|diffusers|numpy)( @|==)' "$ROOT/oracle-freeze.txt" > "$ROOT/constraints.txt" || true
uv pip install --python "$V" -c "$ROOT/constraints.txt" \
  ${TRT_PKG:-tensorrt-cu13} onnx onnxscript onnx_graphsurgeon polygraphy \
  ${MODELOPT_PKG:-"nvidia-modelopt[torch,onnx]"} 2>&1 | tail -15
"$V/bin/python" - <<'PY'
import torch, diffusers, tensorrt, onnx
print("torch", torch.__version__, "cuda", torch.version.cuda, "diffusers", diffusers.__version__)
print("tensorrt", tensorrt.__version__, "onnx", onnx.__version__)
try:
    import modelopt; print("modelopt", modelopt.__version__)
except Exception as e:
    print("modelopt import failed:", e)
PY
