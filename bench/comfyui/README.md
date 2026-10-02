# bench/comfyui — qwen-image-rs vs ComfyUI

Reproduces [docs/COMFYUI.md](../../docs/COMFYUI.md): Qwen-Image-2.1 text-to-image,
1024², 40 steps, euler / `simple`, cfg 1, on the WSL2 RTX 4090 host.

| File | What it does |
|---|---|
| `setup.sh` | clones ComfyUI + `Comfy-Org/workflow_templates` into `~/dev_tmp/comfy-bench`, makes a uv venv (torch cu130), builds SageAttention 2.2 for sm89 (patched to C++20 for torch 2.14 headers), downloads the Comfy-Org repackaged weights into the shared HF cache and symlinks them into `ComfyUI/models/`, adds the Viggle turbo checkpoint + its sigma node |
| `bench.py` | HTTP + websocket client. Builds the API-format graph of the official `image_qwen_image_2_1_t2i` template (subgraph flattened, prompt enhancer off = template default), queues the 4 prompts × N rounds, times every prompt from websocket events. `--dump f.json` writes the graph |
| `run_variant.sh` | one variant: fresh server with the given flags → `bench.py` → kill; samples `nvidia-smi` every 250 ms |
| `run_all.sh` | the whole suite, ours first (variant names = rows of docs/COMFYUI.md); `BEST_ATTN=sage\|ck` picks the attention for the compile and turbo runs. Compile variants are expected to fail except `D2_…` (see the doc) |
| `bench_ours.sh` | same-session qwen-image-rs: one cold `generate`, then `batch --resident` over the same 4 prompts |
| `summarize.py` | prints the results table from `~/dev_tmp/comfy-bench/results/` |
| `workflows/*.json` | the API-format graphs exactly as queued (prompt 1, seed 42) |

```sh
# on the host
bash bench/comfyui/setup.sh
bash bench/comfyui/run_all.sh                 # every variant, ~40 min; or name some:
bash bench/comfyui/run_all.sh ours H6_int8_sage_fast_nodynvram
~/dev_tmp/comfy-bench/venv/bin/python bench/comfyui/summarize.py
```

Weights: ~54 GB of Comfy-Org + Viggle files land in the shared HF cache
(`~/dev_tmp/weights/hf/hub/blobs`); the install itself is ~8.3 GB.

Timing definitions (all from websocket arrival times, `time.perf_counter()`):

- **s/step**: `(t(progress=N) − t(progress=1)) / (N−1)` of the sampler node. Excludes the first step's setup.
- **per image (steady)**: `execution_start → execution_success` of prompts 2..8 (models resident; the text
  encoder re-runs because the prompt changes). Median.
- **cold first image**: the first prompt after the server is up: every model load, plus compile for the
  compile variants. Server start-up (Python imports, node registration) is reported separately.
- **VRAM**: device-wide `nvidia-smi memory.used` peak minus the idle baseline (WSL shows the desktop's
  ~1.5 GB too).
