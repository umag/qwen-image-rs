# Verification controls (Phase 4b)

Mechanical controls the lifecycle runs before code review. CPU-buildable stages
run on the Mac via `scripts/check.sh`; GPU stages run on the WSL 4090 via
`scripts/host.sh`.

| Control | Command | Tier | Where |
|---------|---------|------|-------|
| fmt | `scripts/check.sh fmt` | blocking | Mac |
| lint | `scripts/check.sh clippy` (`-D warnings`) | blocking | Mac |
| typecheck/build (CPU) | `scripts/check.sh check` | blocking | Mac |
| unit tests (CPU) | `scripts/check.sh test` | blocking | Mac |
| GPU build | `scripts/host.sh build` (`--features cuda`) | blocking for GPU code | WSL 4090 |
| oracle parity | port output vs `oracle_out/` within tolerance | blocking for a ported component | WSL 4090 |

Notes:
- CPU-only phases (scaffold, loader) need not hit the GPU stages.
- "oracle parity" is the numerical gate: a ported component must match the
  bf16 diffusers reference before its phase is `verified`.
