#!/usr/bin/env bash
# Runs ON the WSL 4090 host (invoked via scripts/host.sh run, or the swamp
# `wsl-drills` exec/script method). Idempotent. Complements the existing
# `wsl-rust-cuda-setup` swamp workflow (which installs the CUDA toolkit + clang);
# this adds what qwen-image-rs needs: LATEST stable Rust, and a Python 3.12
# venv with torch + diffusers for the reference oracle.
set -euo pipefail
export PATH="$HOME/.cargo/bin:/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH"

echo "== rust: latest stable =="
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  source "$HOME/.cargo/env"
fi
rustup update stable
rustup default stable
rustc --version

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== python 3.12 oracle venv (torch has no 3.14 wheels) =="
ORACLE_DIR="$HOME/dev_tmp/qwen-image-oracle"
mkdir -p "$ORACLE_DIR"
cd "$ORACLE_DIR"
uv venv --python 3.12 .venv
# torch cu12 wheels run fine against the newer driver (forward-compatible).
uv pip install --python .venv \
  "torch" "torchvision" \
  "diffusers @ git+https://github.com/huggingface/diffusers" \
  "transformers>=5.17" accelerate safetensors pillow
.venv/bin/python -c "import torch,diffusers,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('diffusers',diffusers.__version__,'transformers',transformers.__version__)"
echo "== setup-host OK =="
