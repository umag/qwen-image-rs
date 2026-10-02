#!/usr/bin/env python3
"""Build a TensorRT engine from onnx/<name>/model.onnx -> engines/<name>.plan.

  build_engine.py <name> [--opt-level 3] [--workspace-gib 6]

TensorRT 11 networks are strongly typed (no FP16/INT8/FP8 builder flags any
more): precision comes from the ONNX tensor types and Q/DQ nodes. Fixed shapes
(the ONNX has no dynamic dims), so no optimization profile is needed.
"""
import argparse
import json
import time

import tensorrt as trt

from common import WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--workspace-gib", type=float, default=6)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    onnx_path = WORK / "onnx" / args.name / "model.onnx"
    eng_dir = WORK / "engines"
    eng_dir.mkdir(parents=True, exist_ok=True)
    plan = eng_dir / f"{args.name}.plan"

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    t0 = time.perf_counter()
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i))
        raise SystemExit(1)
    t_parse = time.perf_counter() - t0
    print(f"parsed {onnx_path} in {t_parse:.1f}s: {net.num_layers} layers, "
          f"inputs={[(net.get_input(i).name, tuple(net.get_input(i).shape), str(net.get_input(i).dtype)) for i in range(net.num_inputs)]}",
          flush=True)
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 2**30))
    cfg.builder_optimization_level = args.opt_level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache_path = WORK / "timing.cache"
    cache = cfg.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    t1 = time.perf_counter()
    ser = builder.build_serialized_network(net, cfg)
    t_build = time.perf_counter() - t1
    if ser is None:
        raise SystemExit("build failed")
    plan.write_bytes(bytes(ser))
    cache_path.write_bytes(bytes(cfg.get_timing_cache().serialize()))
    meta = {"name": args.name, "trt": trt.__version__, "parse_s": round(t_parse, 1),
            "build_s": round(t_build, 1), "engine_bytes": plan.stat().st_size,
            "opt_level": args.opt_level, "workspace_gib": args.workspace_gib}
    (eng_dir / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
