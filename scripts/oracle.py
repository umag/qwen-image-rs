#!/usr/bin/env python3
"""Generate the bf16 reference ("oracle") set for qwen-image-rs.

Every Rust port phase is validated against these deterministic outputs
(fixed seed + step count). Saves, per prompt: the final PNG, the pre-VAE
latent, and the prompt embeddings — so the VAE (Phase 2), text encoder
(Phase 3) and DiT (Phase 4) can each be diffed in isolation.

Runs on the WSL 4090 in the Python 3.12 venv from setup-host.sh. The model
(~15B params) does not fit in 24 GB at bf16, so CPU offload is on by default.

  python oracle.py --out ~/dev_tmp/qwen-image-rs/oracle_out
"""
import argparse
import json
import pathlib

import torch

MODEL = "Qwen/Qwen-Image-2.1"
PROMPTS = [
    "a red ceramic coffee mug on a wooden table, soft morning light",
    'a storefront window with a neon sign reading "QWEN", rainy night, reflections',
    "an isometric low-poly island with a waterfall, pastel colors",
]
SEED = 42
STEPS = 40
SIZE = (1024, 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--no-offload", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    if args.no_offload:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    captured: dict = {}

    def grab_latents(_pipe, i, _t, kwargs):
        # Snapshot the final latent just before VAE decode (Phase 2 oracle).
        captured["latents"] = kwargs["latents"].detach().to(torch.float32).cpu()
        return kwargs

    manifest = []
    for idx, prompt in enumerate(PROMPTS):
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        out = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            width=SIZE[0],
            height=SIZE[1],
            generator=gen,
            callback_on_step_end=grab_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        stem = f"{idx:02d}"
        out.images[0].save(args.out / f"{stem}.png")
        if "latents" in captured:
            torch.save(captured["latents"], args.out / f"{stem}.latent.pt")
        manifest.append({"idx": idx, "prompt": prompt, "seed": SEED,
                         "steps": args.steps, "size": SIZE})
        print(f"[oracle] {stem} done: {prompt[:48]}")

    (args.out / "manifest.json").write_text(json.dumps(
        {"model": MODEL, "items": manifest}, indent=2))
    print(f"[oracle] wrote {len(manifest)} references to {args.out}")


if __name__ == "__main__":
    main()
