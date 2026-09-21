#!/usr/bin/env bash
# Mac-side checks that need no GPU: fmt --check, cargo check, clippy, host tests.
# Uses the DEFAULT (CPU) feature set — candle's CPU backend needs no CUDA on macOS.
# The repo is scp'd to the GPU host on every build, so target/ must never live
# in-tree: an out-of-tree CARGO_TARGET_DIR is used here too.
#
#   scripts/check.sh            all stages
#   scripts/check.sh check      one stage: fmt | check | clippy | test
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="${QIR_LOCAL_TARGET_DIR:-$HOME/.cache/qwen-image-rs-target}"

stage="${1:-all}"
run() { echo "== $1 =="; shift; "$@"; }
[[ $stage == all || $stage == fmt ]]    && run fmt    cargo fmt --all --check
[[ $stage == all || $stage == check ]]  && run check  cargo check --workspace --all-targets
[[ $stage == all || $stage == clippy ]] && run clippy cargo clippy --workspace --all-targets -- -D warnings
[[ $stage == all || $stage == test ]]   && run test   cargo test --workspace
test ! -e target || { echo "ERROR: a target/ directory appeared in the repo" >&2; exit 3; }
echo "== local checks OK =="
