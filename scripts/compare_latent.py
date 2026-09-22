#!/usr/bin/env python3
"""Compare two latent safetensors (key `latent`): cosine, maxabs, MSE.

Used by the batched-generation per-lane correctness gate — a batched lane's
final latent must match the sequential single-seed run near-bit-exactly
(cosine > 0.99999, maxabs ~ 0), since every DiT op is per-lane.

    compare_latent.py <a.latent.safetensors> <b.latent.safetensors>
"""
import sys

import numpy as np
from safetensors.numpy import load_file

a = load_file(sys.argv[1])["latent"].astype(np.float64).reshape(-1)
b = load_file(sys.argv[2])["latent"].astype(np.float64).reshape(-1)
if a.shape != b.shape:
    print(f"SHAPE MISMATCH {a.shape} vs {b.shape}")
    sys.exit(1)
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
maxabs = float(np.abs(a - b).max())
mse = float(((a - b) ** 2).mean())
ok = cos > 0.99999 and maxabs < 1e-2
print(f"cosine={cos:.7f} maxabs={maxabs:.6f} mse={mse:.3e} -> {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
