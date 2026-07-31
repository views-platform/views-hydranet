# 03 — Harness & Invariants

**Date:** 2026-06-10 · Guardrails for safe, one-variable experimentation on the ZINB head.

## Standing invariants (must always hold)
- **No `expm1` of a free prediction.** The only `expm1` is on the bounded real target (count-target bridge). This is the C-113 firewall.
- **Softplus μ** — the count-space mean stays sub-exponential; the fed-back `log1p(E[y])` cannot run away the way the log1p point head did.
- **Flag-off byte-identical.** `output_distribution` default-off ⇒ the emitted stack is byte-identical to the current head.

## New harness (build behind these tests)
1. **Count-target provider** (#98): exact round-trip `expm1(log1p y_true) == y_true`; NaN/Inf guards; an assertion it touches **targets only, never predictions**. Reuses FeatureScaler's inverse.
2. **`ZINBLoss`** (#99): known-value NLL tests (hand-computed cases); finite gradients on zero and positive cells; registry-selectable.
3. **Head parity** (#100): flag-off byte-identical (clone `test_feedback_clamp.py`); flag-on wires μ (softplus) / π (`1−sigmoid(cls)`) / θ (scalar).
4. **Explosion-check gate** (#102, read-only — NOT a clamp): `scripts/diagnose_io_gain.py` on `E[y]` over 36 steps. Bounded → eval; explodes → STOP, escalate. This is a go/no-go, never a model modification.

## Pre-flight checklist (before any GPU run)
- Full `pytest` suite + `ruff` green — **including #95 fixed** (otherwise collection aborts and masks regressions).
- Config dry-run validates (`ConfigInitializer(cfg).get_config()`), so we fail at config time, not 24 min in.
- **Checkpoint before the GPU run.** CUDA pre-flight healthy.
- One variable vs the recorded baseline; the comparator artifact preserved.
