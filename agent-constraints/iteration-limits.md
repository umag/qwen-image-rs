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
