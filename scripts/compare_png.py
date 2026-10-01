#!/usr/bin/env python3
"""PSNR / max-abs between two same-size PNGs (RGB(A), 8-bit).

    python scripts/compare_png.py ours.png ref.png
Prints `psnr_db=<x> maxabs=<n> identical=<bool>`; use the oracle venv python.
"""
import sys

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGBA"), dtype=np.float64)


a, b = load(sys.argv[1]), load(sys.argv[2])
if a.shape != b.shape:
    sys.exit(f"shape mismatch {a.shape} vs {b.shape}")
# Compare RGB only when the reference alpha is opaque everywhere.
if (b[..., 3] == 255).all() and (a[..., 3] == 255).all():
    a, b = a[..., :3], b[..., :3]
mse = ((a - b) ** 2).mean()
psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
print(f"psnr_db={psnr:.2f} maxabs={int(np.abs(a - b).max())} identical={bool(mse == 0)}")
