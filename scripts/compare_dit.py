#!/usr/bin/env python3
"""dit-forward oracle gate: compare our `output` tensor vs the diffusers oracle.

usage: compare_dit.py <ours.safetensors> <oracle dit_io.safetensors>
Prints overall cosine, MSE, and the mean per-token cosine over the last 4096
rows (the image tokens). Run with the oracle venv python (numpy + safetensors).
"""
import sys

import numpy as np
from safetensors.numpy import load_file


def load_output(path):
    t = load_file(path)["output"]
    return t.astype(np.float64)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ours = load_output(sys.argv[1])
    ref = load_output(sys.argv[2])
    if ref.ndim == 3:
        ref = ref[0]
    if ours.ndim == 3:
        ours = ours[0]
    if ours.shape != ref.shape:
        sys.exit(f"shape mismatch: ours {ours.shape} vs ref {ref.shape}")
    a, b = ours.ravel(), ref.ravel()
    overall = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    mse = float(np.mean((a - b) ** 2))
    oa, rb = ours[-4096:], ref[-4096:]
    per_tok = (oa * rb).sum(1) / (np.linalg.norm(oa, axis=1) * np.linalg.norm(rb, axis=1))
    print(f"overall_cos={overall:.6f} mse={mse:.6e} per_token_cos_mean={float(per_tok.mean()):.6f}")


if __name__ == "__main__":
    main()
