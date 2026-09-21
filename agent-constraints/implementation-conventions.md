# Implementation conventions

- **Language:** Rust, latest stable (`rust-toolchain.toml`), candle spine.
- **Feature gating:** all CUDA code behind `cuda`; FlashAttention behind
  `flash-attn`; cuDNN behind `cudnn`. The default build is CPU-only and must
  always compile on macOS (`scripts/check.sh`).
- **No in-tree `target/`:** the repo is `git archive`'d to the WSL host on every
  build; `CARGO_TARGET_DIR` is always out-of-tree.
- **Host access only via swamp:** reach the GPU through the `wsl-drills`
  `@swamp/ssh` model (`scripts/host.sh`) — never raw ssh/scp (repo anti-bypass).
- **Optimizations are additive + validated:** every Phase-5 kernel is a feature
  flag that must match the Phase-4 bf16 oracle before it is trusted. FFI
  upstream CUDA (SageAttention) rather than reimplementing it.
- **Weights never committed;** loaded from `weights/` (safetensors or GGUF).
- **PRs:** none yet — single local repo. Commit per phase; keep messages BLUF.
