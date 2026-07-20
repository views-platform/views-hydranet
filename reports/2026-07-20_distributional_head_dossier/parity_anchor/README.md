# Legacy emit-math parity anchor (ADR-067 / A-S2)

**Purpose.** Freeze the byte-exact output of the existing `HydraNetInference._emit_magnitude` for
every `output_distribution` value, on one fixed synthetic input, **before** any strangler-fig wiring
touches those switch-sites. The wiring stories (A-S6..A-S8) delegate each branch to a distribution
family; this anchor is the reference they must reproduce byte-for-byte.

## Files

- `build_emit_parity_anchor.py` — regenerates the golden. Calls the **real** unbound
  `_emit_magnitude` through a minimal stand-in carrying only the three attributes it reads
  (`output_distribution`, `hurdle_theta`, `lognormal_sigma`) — no heavy `__init__`, no re-derivation
  of the math.
- `emit_parity_golden.json` — the golden fixture: the fixed `(reg, prob, theta, sigma)` input and the
  recorded `log1p(E[y])` output for each of `standard`, `hurdle_nb`, `hurdle_lognormal`,
  `hurdle_shrinkage`, `dense_nb`, `quantile`. Machine-generated; do not edit by hand.

## The test

`tests/distributions/test_parity_anchor.py` reloads the fixed input from the golden, re-runs the real
`_emit_magnitude`, and asserts `torch.equal` against the recorded output per distribution. Green today
= self-consistency. If a future wiring change alters any emit path, the mismatch fails loud.

## Scope

This is the **emit-level** proxy (deterministic, CPU, GPU-free). Full end-to-end forecast parity
(model + data, K=1 byte-identity) is A-S8's integration test. Regenerate the golden only on an
**intended** emit-math change, and review the diff when you do.
