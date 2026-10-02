# Iteration limits

Autonomous review-resolution loops (Phases 3, 4b, 5) cap at **5 iterations**
before pausing for human input. The approval gate (Phase 3 `approve_plan`,
Phase 5 `resolve_findings`) is always human — never auto-approved.

Standing authorization for this project (2026-09-21): the human pre-approved
plans for overnight autonomous execution ("approve and go ahead for plans").
Record every plan in the model regardless, for morning review.

Standing authorization (2026-09-30), denoise-optimization series — issues
`qwen-image-rs-rope-quant-fusion`, `qwen-image-rs-convrot-tail-linears`,
`qwen-image-rs-bf16-v-pv`, `qwen-image-rs-sageattention2`: the human
pre-approved plan approval AND `resolve_findings` + `attest` + `complete` when a
lifecycle exits clean (0 CRITICAL / 0 HIGH, dit-forward oracle cosine >= 0.99993,
no denoise regression vs the prior step). Any safeguard exit, parity drop,
regression, or pivot-required finding stops and reports to the human instead.
Run these lifecycles strictly sequentially (shared host build dir + main branch).
A measured negative result (lever not worth it) is a valid clean outcome.

Gate update (2026-09-30, human: "don't bother much with gates"): the dit-forward
--convrot cosine USED to be run-to-run noisy (0.99942-0.99994) — that was the sage
stale-smem NaN bug, fixed in `-bf16-v-pv`; it is now deterministic (0.999944 every
run) and B=1 generate is bit-reproducible, so a --convrot reading below 0.99993 is
a real regression. Clean = no-convrot dit-forward bit-identical to the
prior build (or cosine >= 0.99999 when the math legitimately changes) + a same-
session A/B showing no denoise regression + images look right. Keep it light.

Standing authorization (2026-10-01): bug `qwen-image-rs-b1-off-prompt` — same
terms as the denoise series (approve_plan, resolve_findings, attest, complete on
a clean exit).

Gate update (2026-10-01, human): SageAttention2 (FP8 P·V) ACCEPTED as the default
fast path. Its dit-forward --convrot vs oracle baseline is 0.999894 (no-convrot
0.999945); that is the reference for later work on the SA2 path. A change that
does not alter the math must stay bit-identical to the prior build; one that does
must stay within ~1e-5 of the prior build's oracle cosine and keep images clean.
Standing auth also covers `qwen-image-rs-tail-linears-unify` and
`qwen-image-rs-sage2-quant-fusion` (same clean-exit rules).

Standing auth (2026-10-01, human: "make 5 default, and continue with 1 2 3"):
`qwen-image-rs-prequant-default`, `qwen-image-rs-fused-rotate-quant`,
`qwen-image-rs-fused-swiglu`, `qwen-image-rs-gemm-merge-tune` — same clean-exit
rules as the "Gate update (2026-10-01)" paragraph above (current oracle baseline
dit-forward --convrot 0.999898 with sage2 default). Run strictly sequentially.

Reference update (2026-10-01): fused-rotate-quant accepted (human gate stance:
"don't bother much with gates"). New dit-forward --convrot vs oracle reference on
the default SA2 config: 0.999879 (QIR_SAGE=1: 0.999900). The oracle cosine reacts
chaotically to tiny upstream changes — judge the whole config (all QIR_SAGE modes,
no-convrot parity, self-tests vs f64/old path, clean images), not one number.

Reference update (2026-10-01): fused-swiglu accepted under the same stance. New
dit-forward --convrot vs oracle references: default SA2 0.999911, QIR_SAGE=1
0.999881, QIR_SAGE=2f32 0.999890 (no-convrot 0.999945 unchanged).

Standing auth (2026-10-01, human: "lets optimise per-head q/k RMSNorm"):
`qwen-image-rs-qk-norm-fusion` — same clean-exit rules; oracle references after
gemm-merge-tune: SA2 0.999911, QIR_SAGE=1 0.999881, 2f32 0.999890.

Standing auth (2026-10-01, human: "fuse gated residual into norm_mod"):
`qwen-image-rs-residual-norm-fusion` — same clean-exit rules; references
unchanged (SA2 0.999911, QIR_SAGE=1 0.999881, 2f32 0.999890, no-convrot 0.999945).

Standing auth (2026-10-01, human: "run 1 to 4 as lifecycles" — VAE decode series):
`qwen-image-rs-vae-cudnn` (spike: install cuDNN on the GPU host in user space or
via the system package manager, build candle `cudnn`), `qwen-image-rs-vae-fused-norm`,
`qwen-image-rs-vae-implicit-gemm-conv` (only if cuDNN falls short),
`qwen-image-rs-vae-untiled`. Run sequentially. VAE gate: decode PSNR vs the diffusers
oracle stays >= ~50 dB (bf16 baseline 55.2 dB) and vs the prior build >= ~50 dB when
conv/accumulation order changes (byte-identical where the math is unchanged);
images clean; DiT path untouched (dit-forward cmp-identical).

Standing auth (2026-10-02, human: "push for better tiling, try optimisators"):
`qwen-image-rs-gemm-tiling-push` — INT8 accumulation is exact, so every GEMM
config change must be byte-identical (cmp dit-forward in all QIR_SAGE modes, PNG
md5). Determinism required (no atomic / non-deterministic split-K reductions).

Standing auth (2026-10-02, human: "lets do nhwc"): `qwen-image-rs-vae-nhwc` —
VAE decode series gate applies (PSNR vs oracle >= ~50 dB, vs prior build >= ~50 dB
when conv algorithms/accumulation change, byte-identical where math is unchanged,
images clean, DiT path cmp-identical). Target: beat ComfyUI's 0.20 s decode.
