#!/usr/bin/env python3
"""Print the results table from ~/dev_tmp/comfy-bench/results/*/ (run on the host).

Per ComfyUI variant: steady s/step (median over images 1..N), steady per-image
wall (median), cold first image (first prompt after server start: model loads +
any compile), peak GPU memory (nvidia-smi device total minus idle baseline),
mean power under load, and whether the server log shows partial loading /
offload. "ours" is parsed from bench_ours.sh logs.
"""
import json, os, re, statistics, sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dev_tmp/comfy-bench/results")
OURS_STEP = None


def ours():
    d = os.path.join(RES, "ours")
    log = re.sub(r"\x1b\[[0-9;]*m", "", open(os.path.join(d, "batch.log")).read())
    rows = [tuple(map(int, m)) for m in re.findall(r"image=(\d+) enc_ms=(\d+) denoise_ms=(\d+) decode_ms=(\d+)", log)]
    ts = re.findall(r"(\d\d:\d\d:\d\d\.\d+)Z.*done \(resident\)", log)
    def sec(t):
        h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)
    walls = [sec(b) - sec(a) for a, b in zip(ts, ts[1:])]
    steady = rows[1:]
    gpu = open(os.path.join(d, "gpu.csv")).read().splitlines()
    peak = max(int(l.split(", ")[1]) for l in gpu)
    base = int(open(os.path.join(d, "baseline_mib")).read())
    return {"s_step": statistics.median(r[2] for r in steady) / 40000,
            "wall": statistics.median(walls) if walls else None,
            "enc": statistics.median(r[1] for r in steady) / 1000, "dec": statistics.median(r[3] for r in steady) / 1000,
            "peak_gb": (peak - base) / 1024}


print(f"{'variant':34} {'s/step':>8} {'img s':>7} {'enc s':>6} {'dec s':>6} {'cold s':>7} {'start s':>7} {'VRAM GB':>8} {'W':>5}  offload  vs-ours(step,img)")
try:
    o = ours()
    print(f"{'ours (qwen-image-rs)':34} {o['s_step']:8.4f} {o['wall']:7.2f} {o['enc']:6.3f} {o['dec']:6.3f} {'':>7} {'':>7} {o['peak_gb']:8.1f}")
except Exception as e:  # noqa: BLE001
    o = None
    print("ours: n/a", e)

for v in sorted(os.listdir(RES)):
    p = os.path.join(RES, v, "result.json")
    if v == "ours" or not os.path.exists(p):
        continue
    r = json.load(open(p))
    g = json.load(open(os.path.join(RES, v, "gpu_summary.json")))
    log = open(os.path.join(RES, v, "server.log"), errors="replace").read()
    off = []
    if re.search(r"loaded partially", log): off.append("partial")
    if re.search(r"lowvram|offloaded", log, re.I): off.append("offload")
    m = re.findall(r"(\d+)MB Staged", log)
    steps = r["rows"][1:]
    sps = statistics.median(x["s_per_step"] for x in steps)
    wall = statistics.median(x["wall_s"] for x in steps)
    rel = f"{sps / o['s_step']:.2f}x,{wall / o['wall']:.2f}x" if o else ""
    print(f"{v:34} {sps:8.4f} {wall:7.2f} {statistics.median(x['encode_s'] for x in steps):6.3f} "
          f"{statistics.median(x['decode_s'] for x in steps):6.3f} {r['cold_first_image_s']:7.2f} {g['startup_s']:7.1f} "
          f"{(g['peak_mib'] - g['baseline_mib']) / 1024:8.1f} {g['mean_load_power_w']:>5}  {'/'.join(off) or 'no':8} {rel}")
