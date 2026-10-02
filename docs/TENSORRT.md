# qwen-image-rs vs TensorRT and PyTorch

Measured 2026-10-02 in one session on the same host: RTX 4090 24 GB (480 W cap),
WSL2, driver with CUDA 13.x. Workload: **one Qwen-Image-2.1 DiT forward at 1024²**
(4096 image tokens + 21 text tokens = 4117, B=1, guidance 1). This is one denoise
step. qwen-image-rs is HEAD `dd36636`, built with
`--features convrot,sage,fusednorm,sage2,cudnn`. Scripts: [`bench/tensorrt/`](../bench/tensorrt/).

## Result

| Variant | s/forward median | p90 | vs ours | cosine vs oracle | build | engine / weights | GPU memory |
|---|---|---|---|---|---|---|---|
| **qwen-image-rs** (ConvRot INT8 GEMMs + SageAttention2) | **0.1475** | – ¹ | 1.00× | **0.999911** ² | – (ConvRot cache) | ~7 GB INT8 | 21.3 GB resident with text encoder + VAE |
| diffusers eager, bf16, SDPA | 0.5058 | 0.5075 | 3.43× slower | 1.000000 (this is the oracle's code path) | – | 13.3 GB | 13.8 GiB peak |
| `torch.compile` (inductor, default) | 0.4316 | 0.4327 | 2.93× | 0.999979 | 42 s | 13.3 GB | 13.6 GiB |
| `torch.compile` (`max-autotune`, CUDA graphs) | 0.4276 | 0.4294 | 2.90× | 0.999983 | 96 s | 13.3 GB | 13.3 GiB |
| TensorRT 11.3 bf16 | 0.4275 | 0.4299 | 2.90× | 0.999988 | 52 s (+ ONNX export 339 s) | 14.2 GB | 13.9 + 0.57 GiB |
| TensorRT FP8 linears (ModelOpt W8A8), bf16 attention | 0.2652 | 0.2667 | 1.80× | 0.999699 | 28–78 s (+ export 3 min) | 7.26 GB | 7.2 + 0.44 GiB |
| TensorRT FP8 linears + FP8 attention (ModelOpt FP8 MHA) | 0.2531 | 0.2553 | 1.72× | 0.999624 | 60 s | 7.26 GB | 7.3 + 0.53 GiB |
| TensorRT INT8 SmoothQuant linears (α=0.8), bf16 attention | **0.2160** | 0.2169 | **1.46×** | 0.998214 | 96 s (+ export 2 min) | 7.29 GB | 7.3 + 0.39 GiB |
| TensorRT INT8 linears + FP8 attention (our mix) | **infeasible on Ada** | | | (fake quant 0.997624) | build error ³ | | |

¹ Our number is the mean over 80 consecutive steps (prompts 2 and 3 of a
3-prompt `batch --resident` run; `denoise_ms` 5898 and 5901 over 40 steps each,
after 40 warm-up steps). This matches the 0.1477 s/step on record. The binary
logs only per-image totals, so there is no per-step p90. The other rows are the
median and p90 of 60 back-to-back forwards after 5 warm-up forwards, timed with
CUDA events.
² From the repo's `compare_dit.py` gate (README "Quality"); not re-run in this session.
³ `IBuilder::buildSerializedNetwork: Error Code 9: API Usage Error (INT8 and FP8 mixed precision is allowed only when building network with kSTRONGLY_TYPED mode on Blackwell+ platforms.)`

"GPU memory" for TensorRT is the engine (weights) plus the execution context's
activation memory (`device_memory_size_v2`). Cosine is computed in f64 over the
full `(1, 4117, 64)` output against `~/dev_tmp/oracle_out/dit_io.safetensors`,
the same way `scripts/compare_dit.py` does it.

**Where we stand.** We are **1.46× faster than the fastest TensorRT engine**
(INT8 SmoothQuant), and that engine is also less accurate (0.9982 vs 0.9999
cosine). We are 1.72× faster than TensorRT's best FP8 engine and 2.9× faster
than any bf16 stack. bf16 TensorRT does not beat `torch.compile max-autotune`:
both run the bf16 GEMMs near the tensor-core limit, about 155 TFLOPS on about
66 TFLOP per forward.

## Where the time goes (TensorRT per-layer profiler, ms per forward)

| | GEMMs | attention | everything else |
|---|---|---|---|
| TRT bf16 | 330.7 (`sm80_xmma_gemm_bf16bf16_bf16f32_f32`) | 53.6 (fused `_gemm_mha_v2`, bf16) | 42.8 |
| TRT FP8 | 168.4 (`sm89_xmma_gemm_e4m3bf16_e4m3f32_f32`) | 54.4 (bf16 fused MHA) | 41.5 |
| TRT FP8 + FP8 MHA | ~183 ⁴ | 30.5 (FP8 fused MHA) | 40.4 |
| TRT INT8 SQ | ~120 ⁴ | 55.3 (bf16 fused MHA) | ~40 |
| **qwen-image-rs** (`nsys`) | **95.6** (CUTLASS INT8, dequant in epilogue) | **18.7** (SageAttention2: INT8 Q·K, FP8 P·V) | ~33 (fused norm/SwiGLU/quant kernels) |

⁴ Myelin's kernel names do not always separate the GEMMs from the fused
pointwise work, so these values are approximate. The profiler syncs after every
layer, so its totals are about 0–1% above the un-profiled medians.

## What TensorRT does that we don't

- **It builds from a graph automatically.** Once the model is in ONNX, Myelin
  fuses all the pointwise and norm chains: RoPE+RMSNorm, LayerNorm×(1+scale),
  and gate×residual each become one kernel. It also merges the `proj`/`gate`
  SwiGLU GEMMs into one GEMM. We wrote each of these fusions by hand.
- **It recognizes the attention pattern** (MatMul→Softmax→MatMul) and replaces
  it with a fused flash-style MHA kernel in bf16 or FP8. Its FP8 MHA is 1.8× its
  bf16 MHA.
- **Its quantization is generic.** ModelOpt quantizes any `nn.Linear` with
  calibration in about 13 s. INT8, FP8 and Q/DQ placement are configuration,
  not kernel work.
- **It tunes tactics per shape.** The build profiles kernels for this exact GPU
  and shape, in about 1 minute.

## What we do that TensorRT can't (on this GPU)

- **We mix INT8 GEMMs with FP8 attention.** TensorRT 11 on Ada refuses this mix
  (see ³). Our recommended build uses exactly this mix.
- **We get quality out of INT8.** TensorRT INT8 supports only static per-tensor
  activation scales. Even with SmoothQuant that reaches cosine 0.9969 in
  PyTorch fake-quant and 0.9982 in the engine. Plain W8A8 (no smoothing)
  collapses to 0.59. Our ConvRot (online rotation) INT8 keeps 0.99991. To match
  our quality, TensorRT has to use FP8, which is 1.7–1.8× slower than us.
- **Our GEMMs and attention are faster.** Our CUTLASS INT8 GEMMs take 95.6 ms
  against about 120 ms for TensorRT INT8 and 168 ms for FP8. SageAttention2
  takes 18.7 ms against 30.5 ms for TensorRT FP8 MHA and 54 ms for bf16 MHA.
- **We have no per-shape engine.** A TensorRT engine is fixed to one GPU, one
  TensorRT version and (here) one text length. A different prompt length needs
  a dynamic-shape profile or a new build. We have no export step and no 14 GB
  ONNX file. Getting each quantized TensorRT variant took ONNX surgery (below).

## Method

1. **PyTorch baseline.** `bench_torch.py` runs the stock
   `QwenImage21Transformer2DModel` in bf16 (the venv has the same torch
   2.14.0+cu130 and diffusers 0.41.0.dev0 as the oracle).
2. **Exportable graph.** `DiTCore` (`common.py`) restates the diffusers forward
   for text-to-image using the same submodules and weights:
   - real-valued RoPE in place of complex RoPE;
   - the block-causal attention as two SDPA calls (text queries causal over the
     text keys, image queries over all keys);
   - static slicing in place of token metadata.

   Like qwen-image-rs, it runs the full joint sequence every step (no prefix KV
   cache). It matches diffusers at cosine 0.999988, the bf16 noise level.
3. **bf16 ONNX.** `torch.onnx.export(dynamo=True)` at opset 20, with 14.2 GB of
   external data. The TensorRT 11 network is strongly typed (TensorRT 11 has no
   FP16/INT8 builder flags; precision comes from the graph). Shapes are fixed.
   Builder optimization level 3, 6 GiB workspace. No `trtexec` (the pip wheels
   do not ship it); timing uses the Python runtime with `execute_async_v3`.
4. **FP8.** ModelOpt 0.47 `FP8_DEFAULT_CFG` (per-tensor FP8 E4M3 weights and
   activations) on the 224 per-block linears. `img_in`, `txt_in`, timestep,
   modulation, `norm_out`, `proj_out` and the LayerNorms stay bf16, as in
   qwen-image-rs. Calibration uses 16 real DiT inputs: the 3 oracle prompts × 5
   flow-matching noise levels of the oracle latents, plus the `dit_io` sample.
   Export goes through ModelOpt's TorchScript Q/DQ symbolics, then
   `quantize_weights` (real FP8 weights). `--mha` also attaches ModelOpt's FP8
   q/k/v/softmax quantizers to the image attention (`ImgAttn`).
5. **INT8 SmoothQuant.** `INT8_SMOOTHQUANT_CFG` (per-channel INT8 weights,
   per-tensor INT8 activations).

   | α | fake-quant cosine |
   |---|---|
   | 0.5 | 0.99516 |
   | 0.8 | **0.99659** |
   | 1.0 (the default) | 0.98527 |

   We used α=0.8. ModelOpt's own INT8 TorchScript export segfaults inside
   torch 2.14's JIT tracer (`torch/jit/_trace.py:140`). The calibrated layers
   are therefore re-expressed as plain `QuantizeLinear→DequantizeLinear`
   (`QDQLinear`), with the same smoothed weights, SmoothQuant pre-scale and
   scales. Its cosine (0.99688) matches the fake-quant model.

## Fixes the quantized graphs needed

Each fix is in `export_onnx.py` and found by a failed build.

- **Mixed-type attention.** The TorchScript exporter's SDPA symbolic mixes f32
  and bf16, and the parser fails with: `IMatrixMultiplyLayer must have same
  input types. A is of type Float and B is of type BFloat16`. Fix: attention is
  written out in bf16 (`common.sdpa`), with the 1/√d scale as a Python float.
  The traced `int ** -0.5` turns into a `Cast` to COMPLEX128, which TensorRT
  rejects.
- **Lone FP8 Softmax.** ModelOpt's FP8 exporter always adds an FP8 Q/DQ after
  every Softmax (for TensorRT's FP8 MHA fusion). Without FP8 q/k/v this blocks
  any MHA fusion: attention runs as raw MatMuls and the FP8 engine takes
  0.525 s, slower than bf16. Fix: `strip_softmax_qdq` removes the pair when
  `--mha` is off. The inserted scales are FLOAT, so `qdq_scales_to_bf16` recasts
  them.
- **FP8 MHA dtype.** FP8 MHA needs `trt_high_precision_dtype="BFloat16"` on the
  attention quantizers, or ModelOpt raises: `the qdqs must be in 16 bits`.
- **LayerNorm inputs.** ModelOpt also wraps `nn.LayerNorm` with an input
  quantizer. That quantizes the residual stream (amax about 8.5e3) and
  destroyed the INT8 quality. `img_norm1`/`img_norm2` are excluded.

## Fairness notes

- **Shapes and batch.** All rows use the same tokens (21 text + 4096 image),
  B=1, the same oracle input and the same full-sequence work. The TensorRT
  engines are fixed to L=21. Ours handles any prompt length without a rebuild.
- **Sustained load.** The 4090 power-caps at 480 W. Every number is a sustained
  run: 60 forwards for PyTorch and TensorRT (13–30 s of continuous load), 120
  forwards for ours. p90 is within 1% of the median in every PyTorch/TensorRT row.
- **Attention implementations.**
  - PyTorch: SDPA (flash).
  - torch.compile: inductor's SDPA lowering.
  - TensorRT: its fused `_gemm_mha_v2` in bf16, or FP8 with `--mha`.
  - Ours: SageAttention2 (INT8 Q·K, FP8 P·V, fp16 accumulate).
- **Numerics.** Ours is 0.999911. TensorRT FP8 is 0.99962–0.99970, TensorRT
  INT8 0.99821. The bf16 paths are 0.99998+.
- **Untried or partial.**
  - Torch-TensorRT: not tried. ModelOpt's Q/DQ path is ONNX-based, so ONNX was
    the common route for every TensorRT variant.
  - TensorRT FP16 (not bf16): not tried.
  - B=2: not measured for TensorRT.
  - Builder optimization level 5: not tried.
  - INT8 + FP8 MHA: refused on Ada (³).

## Reproduce and clean up

Run the commands in [`bench/tensorrt/README.md`](../bench/tensorrt/README.md).
The artifacts live on the GPU host in `~/dev_tmp/trt-bench`, about 98 GB in all:

- the venv;
- the ONNX exports: bf16 14.2 GB, fp8 and fp8-mha 7.3 GB each, int8sq and
  int8sq-mha 14.2 GB each (their weights are bf16 with Q/DQ in front);
- the engines: `bf16.plan` 14.2 GB, `fp8`, `fp8-mha` and `int8sq` 7.3 GB each.

`rm -rf ~/dev_tmp/trt-bench` removes all of it.
