#!/usr/bin/env python3
"""Run a TensorRT engine on the oracle DiT input: cosine vs the oracle output,
sustained s/forward (CUDA events around execute_async_v3), device memory.

  bench_trt.py <name> [--n 60] [--layers]   (engines/<name>.plan)

--layers prints the engine's per-layer summary (kernel names, precisions) via
the engine inspector, to see which attention/GEMM kernels TensorRT picked.
"""
import argparse
import json

import tensorrt as trt
import torch

from common import WORK, core_inputs, cosine, fmt, load_dit_io, sustained

TRT2TORCH = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
             trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
             trt.DataType.INT64: torch.int64}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--profile", action="store_true", help="per-layer times (IProfiler), top 25 + by kernel class")
    args = ap.parse_args()
    plan = WORK / "engines" / f"{args.name}.plan"
    free0, total = torch.cuda.mem_get_info()
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    eng = rt.deserialize_cuda_engine(plan.read_bytes())
    ctx = eng.create_execution_context()
    free1, _ = torch.cuda.mem_get_info()

    io = load_dit_io()
    xin = dict(zip(["hidden_states", "encoder_hidden_states", "timestep", "cos", "sin"], core_inputs(io)))

    bufs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = tuple(eng.get_tensor_shape(n))
        dt = TRT2TORCH[eng.get_tensor_dtype(n)]
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            bufs[n] = xin[n].to(dt).reshape(shape).contiguous()
        else:
            bufs[n] = torch.empty(shape, dtype=dt, device="cuda")
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    stream = torch.cuda.Stream()

    def run():
        ctx.execute_async_v3(stream.cuda_stream)

    with torch.cuda.stream(stream):
        run()
        stream.synchronize()
        cos = cosine(bufs["output"].float(), io["output"])
        r = sustained(run, n=args.n)
    free2, _ = torch.cuda.mem_get_info()
    meta_p = WORK / "engines" / f"{args.name}.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    r.update(cos_overall=cos[0], cos_img_tok=cos[1],
             engine_load_gib=(free0 - free1) / 2**30, vram_used_gib=(total - free2) / 2**30,
             device_memory_gib=eng.device_memory_size_v2 / 2**30 if hasattr(eng, "device_memory_size_v2") else None,
             **meta)
    print(f"[trt-{args.name}] {fmt(r)} cos={cos[0]:.6f} tok_cos={cos[1]:.6f} "
          f"vram_used={r['vram_used_gib']:.2f}GiB activ={r['device_memory_gib']}", flush=True)
    out = WORK / "results_trt.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    key = f"trt-{args.name}" + (f"-n{args.n}" if args.n < 60 else "")  # short profiling runs never overwrite the sustained one
    prev[key] = r
    out.write_text(json.dumps(prev, indent=1))
    if args.profile:
        class Prof(trt.IProfiler):
            def __init__(self):
                super().__init__()
                self.t = {}

            def report_layer_time(self, name, ms):
                self.t[name] = self.t.get(name, 0.0) + ms
        prof = Prof()
        ctx.profiler = prof
        reps = 5
        for _ in range(reps):
            ctx.execute_v2([bufs[eng.get_tensor_name(i)].data_ptr() for i in range(eng.num_io_tensors)])
        tot = sum(prof.t.values()) / reps
        print(f"profile: {tot:.1f} ms/forward over {len(prof.t)} layers (profiled, includes per-layer sync)")
        cls = {}
        for n, ms in prof.t.items():
            k = ("mha" if "mha" in n.lower() else "gemm/linear" if ("linear" in n or "gemm" in n.lower()) else
                 "matmul" if "MatMul" in n else "other")
            cls[k] = cls.get(k, 0.0) + ms / reps
        print("by class (ms):", {k: round(v, 1) for k, v in sorted(cls.items(), key=lambda x: -x[1])})
        for n, ms in sorted(prof.t.items(), key=lambda x: -x[1])[:25]:
            print(f"  {ms / reps:7.2f} ms  {n[:150]}")
        r["profile_ms_by_class"] = cls
        prev[key] = r
        out.write_text(json.dumps(prev, indent=1))
    if args.layers:
        insp = eng.create_engine_inspector()
        info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
        layers = info.get("Layers", [])
        (WORK / "engines" / f"{args.name}.layers.json").write_text(json.dumps(info, indent=1))
        print(f"{len(layers)} engine layers; first 40:")
        for L in layers[:40]:
            if isinstance(L, dict):
                print(" ", L.get("LayerType"), "|", L.get("TacticName", "")[:90], "|",
                      [o.get("Format/Datatype", "") for o in L.get("Outputs", [])][:1], "|", L.get("Name", "")[:80])
            else:
                print(" ", str(L)[:160])


if __name__ == "__main__":
    main()
