#!/usr/bin/env python3
"""Drive a running ComfyUI over its HTTP + websocket API and time Qwen-Image-2.1 t2i.

The API-format graph is the official template `image_qwen_image_2_1_t2i.json`
(Comfy-Org/workflow_templates) flattened out of its subgraph, prompt enhancer
off (the template default): UNETLoader -> QwenImage21Cache -> KSampler,
CLIPLoader -> TextEncodeQwenImage21, EmptyLatentImage, VAELoader -> VAEDecode
-> SaveImage. Optional TorchCompileModel / LoraLoaderModelOnly are inserted on
the MODEL edge. `--dump` writes the graph JSON and exits.

Timing comes from websocket events stamped with time.perf_counter() on arrival:
  wall      execution_start -> execution_success of each prompt
  encode    executing(TextEncode) -> executing(next node)
  sampler   executing(KSampler) -> executing(VAEDecode)
  s/step    (t(progress=N) - t(progress=1)) / (N-1)   (excludes step-1 setup)
  decode    executing(VAEDecode) -> executing(SaveImage)
History timestamps (ms) are recorded too as a cross-check.
"""
import argparse, json, statistics, sys, time, urllib.request, uuid

import websocket  # websocket-client

PROMPTS = [
    "a red coffee mug on a wooden table",
    "a lighthouse on a cliff at sunset",
    "a fox in a snowy forest",
    "a bowl of ramen, studio photo",
]


def build_graph(a, prompt, seed, prefix):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": a.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.clip, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
    }
    model = ["unet", 0]
    if a.lora:
        name, strength = a.lora.rsplit(":", 1)
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = ["lora", 0]
    if a.compile:
        g["compile"] = {"class_type": "TorchCompileModel", "inputs": {"model": model, "backend": "inductor"}}
        model = ["compile", 0]
    g["cache"] = {"class_type": "QwenImage21Cache", "inputs": {"model": model, "device": "auto", "dtype": "default"}}
    model = ["cache", 0]
    g["enc"] = {"class_type": "TextEncodeQwenImage21",
                "inputs": {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}}
    g["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": a.size, "height": a.size, "batch_size": 1}}
    if a.sigmas:
        # few-step distill: explicit sigma list via SamplerCustomAdvanced (as the LoRA's own workflow does)
        sig_in = {k: (["latent", 0] if v == "@latent" else v) for k, v in json.loads(a.sigmas).items()}
        g["sig"] = {"class_type": a.sigmas_node, "inputs": sig_in}
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        g["guider"] = {"class_type": "CFGGuider",
                       "inputs": {"model": model, "positive": ["enc", 0], "negative": ["enc", 1], "cfg": a.cfg}}
        g["ksel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}}
        g["ks"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0],
                              "sigmas": ["sig", 0], "latent_image": ["latent", 0]}}
    else:
        g["ks"] = {"class_type": "KSampler", "inputs": {
            "model": model, "seed": seed, "steps": a.steps, "cfg": a.cfg, "sampler_name": a.sampler,
            "scheduler": a.scheduler, "positive": ["enc", 0], "negative": ["enc", 1],
            "latent_image": ["latent", 0], "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}}
    return g


def http(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1:8188")
    p.add_argument("--variant", required=True)
    p.add_argument("--unet", default="qwen_image_2.1_bf16.safetensors")
    p.add_argument("--weight-dtype", default="default")
    p.add_argument("--clip", default="qwen3vl_8b_bf16.safetensors")
    p.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=2, help="passes over the 4 prompts (seed+round)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--lora", default="", help="name.safetensors:strength")
    p.add_argument("--sigmas", default="", help='JSON inputs for --sigmas-node (few-step); "@latent" links the empty latent')
    p.add_argument("--sigmas-node", default="")
    p.add_argument("--out", default="")
    p.add_argument("--dump", default="")
    a = p.parse_args()

    if a.dump:
        json.dump(build_graph(a, PROMPTS[0], a.seed, a.variant), open(a.dump, "w"), indent=1)
        return

    cid = uuid.uuid4().hex
    ws = websocket.create_connection(f"ws://{a.host}/ws?clientId={cid}", timeout=3600)
    jobs = []
    for r in range(a.rounds):
        for i, pr in enumerate(PROMPTS):
            g = build_graph(a, pr, a.seed + r, f"{a.variant}/r{r}_p{i}")
            res = http(f"http://{a.host}/prompt", {"prompt": g, "client_id": cid})
            jobs.append({"id": res["prompt_id"], "prompt": pr, "round": r, "ev": [], "t_queue": time.perf_counter()})
    by_id = {j["id"]: j for j in jobs}
    done = 0
    while done < len(jobs):
        msg = ws.recv()
        if not isinstance(msg, str):
            continue  # binary preview frames
        t = time.perf_counter()
        m = json.loads(msg)
        d = m.get("data", {})
        j = by_id.get(d.get("prompt_id"))
        if j is None:
            continue
        if m["type"] == "progress":
            j["ev"].append((t, "progress", d.get("node"), d.get("value")))
        elif m["type"] in ("executing", "execution_start", "execution_cached", "execution_success", "execution_error"):
            j["ev"].append((t, m["type"], d.get("node"), None))
            if m["type"] == "execution_error":
                print("ERROR", json.dumps(d)[:2000], file=sys.stderr)
                done += 1
            if m["type"] == "execution_success":
                done += 1
                print(f"[{a.variant}] done {done}/{len(jobs)}", flush=True)

    rows = []
    for k, j in enumerate(jobs):
        ev = j["ev"]
        ts = {e[1]: e[0] for e in ev if e[1] in ("execution_start", "execution_success")}
        ex = [(e[0], e[2]) for e in ev if e[1] == "executing" and e[2] is not None]
        ex.append((ts.get("execution_success", float("nan")), "_end"))

        def span(node):
            for idx, (tt, n) in enumerate(ex[:-1]):
                if n == node:
                    return ex[idx + 1][0] - tt
            return float("nan")
        prog = [(e[0], e[3]) for e in ev if e[1] == "progress" and e[2] == "ks"]
        sps = (prog[-1][0] - prog[0][0]) / (prog[-1][1] - prog[0][1]) if len(prog) >= 2 and prog[-1][1] > prog[0][1] else float("nan")
        h = http(f"http://{a.host}/history/{j['id']}")[j["id"]]
        hm = {mm[0]: mm[1].get("timestamp") for mm in h["status"]["messages"]}
        hist_wall = (hm.get("execution_success", 0) - hm.get("execution_start", 0)) / 1000.0
        imgs = [im["subfolder"] + "/" + im["filename"] for o in h["outputs"].values() for im in o.get("images", [])]
        rows.append({"idx": k, "round": j["round"], "prompt": j["prompt"],
                     "wall_s": ts.get("execution_success", float("nan")) - ts.get("execution_start", float("nan")),
                     "hist_wall_s": hist_wall, "encode_s": span("enc"), "sampler_s": span("ks"),
                     "s_per_step": sps, "steps_seen": len(prog), "decode_s": span("dec"),
                     "load_unet_s": span("unet"), "load_clip_s": span("clip"), "compile_node_s": span("compile"),
                     "images": imgs})
    for r in rows:
        print(f"  #{r['idx']} wall {r['wall_s']:.3f}s (hist {r['hist_wall_s']:.3f}) enc {r['encode_s']:.3f} "
              f"sampler {r['sampler_s']:.3f} ({r['s_per_step']:.4f} s/step, {r['steps_seen']} ev) dec {r['decode_s']:.3f} "
              f"{r['images']}")
    steady = rows[1:]
    summ = {"variant": a.variant, "args": vars(a),
            "cold_first_image_s": rows[0]["wall_s"],
            "steady_wall_median_s": statistics.median(r["wall_s"] for r in steady),
            "steady_wall_mean_s": statistics.mean(r["wall_s"] for r in steady),
            "steady_s_per_step_median": statistics.median(r["s_per_step"] for r in steady),
            "steady_encode_median_s": statistics.median(r["encode_s"] for r in steady),
            "steady_decode_median_s": statistics.median(r["decode_s"] for r in steady),
            "rows": rows}
    print("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k not in ("rows", "args")}))
    if a.out:
        json.dump(summ, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
