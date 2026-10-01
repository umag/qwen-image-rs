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
