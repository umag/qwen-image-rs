# TensorRT / PyTorch comparison bench

Measures one Qwen-Image-2.1 DiT forward (1024², 4096 image + 21 text tokens,
B=1) in PyTorch eager, `torch.compile` and TensorRT (bf16, ModelOpt FP8,
ModelOpt INT8 SmoothQuant), against the diffusers bf16 oracle `dit_io`.
Results and caveats: [docs/TENSORRT.md](../../docs/TENSORRT.md).

Everything runs on the GPU host, in `~/dev_tmp/trt-bench` (venv, ONNX, engines).
Copy this directory there (e.g. `~/dev_tmp/trt-bench/src/tensorrt`) and run from it.

```bash
bash setup.sh                                  # venv = oracle torch/diffusers + TensorRT + ModelOpt
PY=~/dev_tmp/trt-bench/.venv/bin/python
$PY bench_torch.py --modes eager,core,compile,compile-max
# export -> build -> bench (60 sustained forwards) -> per-layer profile, per variant
bash run_variants.sh bf16 fp8 "fp8 --mha" "int8sq --alpha 0.8"
$PY export_onnx.py --quant int8sq --alpha 0.8 --cos-only   # quick PTQ quality check, no export
bash bench_ours.sh                             # qwen-image-rs, same session
```

| File | Role |
|---|---|
| `common.py` | `DiTCore` (export-friendly restatement of the diffusers forward, same weights), inputs, cosine, sustained timer, calibration samples |
| `bench_torch.py` | PyTorch eager / compile baselines |
| `export_onnx.py` | optional ModelOpt PTQ, then ONNX export with external data |
| `build_engine.py` | TensorRT 11 strongly-typed engine build (timing cache in `~/dev_tmp/trt-bench`) |
| `bench_trt.py` | engine run: cosine vs oracle, s/forward (median/p90 of 60), VRAM |
| `run_variants.sh` | export → build → bench → profile for each variant |
| `bench_ours.sh` | `qwen-image-rs batch --resident`, 3 prompts × 40 steps |

Results accumulate in `~/dev_tmp/trt-bench/results_{torch,trt}.json`.
Large artifacts (ONNX, `*.plan`) stay on the host — never commit them.
