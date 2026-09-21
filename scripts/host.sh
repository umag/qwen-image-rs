#!/usr/bin/env bash
# Mac-side driver for the WSL 4090 host. Syncs the repo (tracked files only, via
# `git archive`) through the swamp `wsl-drills` @swamp/ssh model — NEVER raw
# scp/ssh (repo anti-bypass rule) — then runs a command on the GPU.
#
#   scripts/host.sh sync                        # push tracked files to the host
#   scripts/host.sh run '<cmd>'                 # run a shell command on the host
#   scripts/host.sh build                       # cargo build --release --features cuda
#   scripts/host.sh smoke                       # build + run the GPU smoke test
#
# Host layout: ~/dev_tmp/qwen-image-rs  (target dir is out-of-tree on the host too).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${QIR_SSH_MODEL:-wsl-drills}"
HOST_DIR="${QIR_HOST_DIR:-\$HOME/dev_tmp/qwen-image-rs}"
SWAMP_REPO="${QIR_SWAMP_REPO:-$HOME/dev_tmp/swamp}"
TAR=/tmp/qwen-image-rs.tar

# `--repo-dir` is a SUBCOMMAND option, not a global flag — it must follow the
# subcommand (`swamp model method run --repo-dir …`), not precede it.
sm() { swamp model method run --repo-dir "$SWAMP_REPO" "$MODEL" "$@"; }

sync() {
  git archive --format=tar HEAD -o "$TAR"
  sm copy --input "{\"hosts\":\"all\",\"direction\":\"to\",\"useRsync\":false,\"src\":\"$TAR\",\"dst\":\"/tmp/qwen-image-rs.tar\"}"
  run "rm -rf $HOST_DIR && mkdir -p $HOST_DIR && tar xf /tmp/qwen-image-rs.tar -C $HOST_DIR && echo extracted-to $HOST_DIR"
}

run() {
  local cmd="$1"
  local full="export PATH=\$HOME/.cargo/bin:/usr/lib/wsl/lib:/usr/local/cuda/bin:\$PATH; export CARGO_TARGET_DIR=\$HOME/.cache/qwen-image-rs-target; cd $HOST_DIR 2>/dev/null; $cmd"
  sm exec --input "$(python3 -c 'import json,sys; print(json.dumps({"hosts":"all","captureOutput":True,"timeoutSec":3600,"command":sys.argv[1]}))' "$full")" >/dev/null
  swamp data get --repo-dir "$SWAMP_REPO" "$MODEL" run-exec-wsl --json 2>/dev/null | python3 -c '
import sys,json
d=json.load(sys.stdin); a=d.get("attributes",d)
def find(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=="stdout" and isinstance(v,str): return v
            r=find(v)
            if r is not None: return r
    elif isinstance(o,list):
        for v in o:
            r=find(v)
            if r is not None: return r
    return None
print(find(a) or "[no stdout]")'
}

case "${1:-}" in
  sync)  sync ;;
  run)   run "$2" ;;
  build) sync; run "cargo build --release --features cuda 2>&1 | tail -30" ;;
  smoke) sync; run "cargo run --release --features cuda -- smoke --n 4096 2>&1 | tail -20" ;;
  *) echo "usage: host.sh {sync|run '<cmd>'|build|smoke}" >&2; exit 2 ;;
esac
