# Iteration limits

Autonomous review-resolution loops (Phases 3, 4b, 5) cap at **5 iterations**
before pausing for human input. The approval gate (Phase 3 `approve_plan`,
Phase 5 `resolve_findings`) is always human — never auto-approved.

Standing authorization for this project (2026-09-21): the human pre-approved
plans for overnight autonomous execution ("approve and go ahead for plans").
Record every plan in the model regardless, for morning review.
