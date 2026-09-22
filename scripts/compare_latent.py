#!/usr/bin/env python3
"""Compare two latent safetensors (key `latent`): cosine, maxabs, MSE.

Batched multi-seed generation is INTENDED to diverge from single-image runs:
the fast-path pipeline is not bit-reproducible (INT8/CUTLASS split-K atomic
reductions vary per run, amplified over the chaotic flow-match trajectory), and
the GEMM tiling differs by batch size, so a batched lane and a `--seed i` single
run legitimately land in different basins. Bit-exactness is NOT the goal.

What this tool checks is SELF-CONSISTENCY: the same seed's lane should match
across batch sizes (e.g. B=2 lane i vs B=4 lane i) within the pipeline's
reproducibility floor. Default threshold 0.99 (self-consistency runs ~0.9999);
pass --threshold to override.

    compare_latent.py <a.latent.safetensors> <b.latent.safetensors> [--threshold 0.99]
"""
import sys

import numpy as np
from safetensors.numpy import load_file

thr = 0.99
args = []
it = iter(sys.argv[1:])
for a in it:
    if a == "--threshold":
        thr = float(next(it))
    elif not a.startswith("--"):
        args.append(a)

a = load_file(args[0])["latent"].astype(np.float64).reshape(-1)
b = load_file(args[1])["latent"].astype(np.float64).reshape(-1)
if a.shape != b.shape:
    print(f"SHAPE MISMATCH {a.shape} vs {b.shape}")
    sys.exit(1)
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
maxabs = float(np.abs(a - b).max())
mse = float(((a - b) ** 2).mean())
ok = cos > thr
print(f"cosine={cos:.7f} maxabs={maxabs:.6f} mse={mse:.3e} (thr={thr}) -> {'OK' if ok else 'DIVERGED'}")
sys.exit(0 if ok else 1)
