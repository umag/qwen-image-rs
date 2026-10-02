#!/usr/bin/env bash
# Install ComfyUI + its own venv + Qwen-Image-2.1 repackaged weights in user space.
# Idempotent: re-running skips what is already there.
#
#   bash bench/comfyui/setup.sh            # ComfyUI + venv + weights
#   SKIP_WEIGHTS=1 bash bench/comfyui/setup.sh
#
# Layout:  ~/dev_tmp/comfy-bench/{ComfyUI,workflow_templates,venv}
# Weights go to the shared HF cache (HF_HOME=~/dev_tmp/weights/hf) and are
# symlinked into ComfyUI/models/*, so nothing is stored twice.
set -euo pipefail
ROOT=${COMFY_BENCH:-$HOME/dev_tmp/comfy-bench}
export HF_HOME=${HF_HOME:-$HOME/dev_tmp/weights/hf}
export PATH=$HOME/.local/bin:/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
mkdir -p "$ROOT"; cd "$ROOT"

[ -d ComfyUI/.git ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI ComfyUI
[ -d workflow_templates/.git ] || git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Comfy-Org/workflow_templates workflow_templates
(cd workflow_templates && git sparse-checkout set templates >/dev/null 2>&1 || true)

if [ ! -x venv/bin/python ]; then
  uv venv --python 3.12 venv
fi
PY=$ROOT/venv/bin/python
uv pip install --python "$PY" torch torchvision torchaudio --index-url "$TORCH_INDEX"
uv pip install --python "$PY" -r ComfyUI/requirements.txt
uv pip install --python "$PY" "huggingface_hub[hf_xet]" requests
# SageAttention 2.x for sm89 (built from source against this torch; the PyPI
# `sageattention` wheel is 1.0.6 = Triton-only SA1).
if ! "$PY" -c "import sageattention, importlib.metadata as m; assert m.version('sageattention').startswith('2')" 2>/dev/null; then
  [ -d SageAttention/.git ] || git clone --depth 1 https://github.com/thu-ml/SageAttention SageAttention
  uv pip install --python "$PY" setuptools wheel ninja packaging
  # torch >= 2.13 headers require C++20; SageAttention's setup.py pins c++17.
  sed -i 's/c++17/c++20/g' SageAttention/setup.py
  (cd SageAttention && CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=8.9 EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=16 \
     uv pip install --python "$PY" --no-build-isolation -v . > "$ROOT/sage_build.log" 2>&1)
fi
(cd / && "$PY" -c "import torch, sageattention, importlib.metadata as m; print('torch', torch.__version__, torch.version.cuda, 'sageattention', m.version('sageattention'))")
uv pip install --python "$PY" websocket-client

[ "${SKIP_WEIGHTS:-0}" = 1 ] && exit 0
M=$ROOT/ComfyUI/models
dl() {  # repo file dest_dir
  local p; p=$("$PY" -c "from huggingface_hub import hf_hub_download as d; import sys; print(d(sys.argv[1], sys.argv[2]))" "$1" "$2")
  mkdir -p "$3"; ln -sfn "$p" "$3/$(basename "$2")"; echo "linked $3/$(basename "$2")"
}
R=Comfy-Org/Qwen-Image-2.1
dl $R vae/qwen_image_2.1_vae_bf16.safetensors                    "$M/vae"
dl $R text_encoders/qwen3vl_8b_bf16.safetensors                  "$M/text_encoders"
dl $R text_encoders/qwen3vl_8b_int8_convrot.safetensors          "$M/text_encoders"
dl $R diffusion_models/qwen_image_2.1_bf16.safetensors           "$M/diffusion_models"
dl $R diffusion_models/qwen_image_2.1_int8_convrot.safetensors   "$M/diffusion_models"
# G: Viggle turbo (6-step DMD distill), merged into an int8_convrot DiT, + its sigma node
V=Viggle/Qwen-Image-2.1-viggle-turbo
dl $V Qwen-Image-2.1-viggle-turbo-v0.3-6step-int8_convrot.safetensors "$M/diffusion_models"
p=$("$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('$V', 'comfyui/viggle_turbo.py'))")
cp "$p" "$ROOT/ComfyUI/custom_nodes/viggle_turbo.py"
du -shL "$M"/{vae,text_encoders,diffusion_models}
