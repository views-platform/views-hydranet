# Class Intent Contract: ForecastComposer (`views_hydranet/distributions/composition.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-07-24
**Related ADRs:** ADR-069 (forecast_composition axis), ADR-068 (arm naming), ADR-067 (distribution
families), ADR-008 (Error Propagation), ADR-009 (Boundary Contracts & Configuration Validation)

---

## 1. Purpose

> Combine a trained **body** (a distribution family's samples / mean) with the **gate** (`P(y>0)`) into
> the single emitted forecast, per the `forecast_composition` arm. The single "compose once" authority
> the emit path calls instead of an inline `if composition == …` ladder.

Two entry points, because the sample cube and the point / AR-feedback path compose differently:
- `compose_samples(samples, gate, composition, threshold, generator)` — mask-based (the sample cube).
- `compose_mean(mean, gate, composition, threshold)` — scale-based, the expectation (the point path).

The keyword and τ are validated at the config boundary (`HydraNetConfig.validate_forecast_composition`);
this module assumes a valid arm and fails loud only defensively.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** validate the config cross-field rules (self-zeroed-may-not-be-gated,
  nb-requires-a-gate) — that is the config boundary (ADR-069). It validates only its own arguments
  defensively (unknown arm; τ ∉ (0,1) for `threshold_gate`).
- Does **not** produce the gate — the classification head does; the composer only consumes `P(y>0)`.
- Does **not** own the count↔log1p boundary: `compose_samples` operates in **log1p** space (0/1 masks
  are exact there); `compose_mean` operates in **count** space and the caller (`_emit_magnitude`)
  wraps the result in `log1p`. It never `expm1`'s a free prediction (C-140).
- Does **not** apply to legacy heads — those compose in their own `_emit_magnitude` branches; the
  composer runs only on the family (nb/zinb) path.

---

## 3. Responsibilities and Guarantees

- **`self_zeroed`** — passthrough (identity) in both functions. A caller that does not opt in is
  **byte-identical** to the pre-ADR-069 path (proven by the S2 characterization goldens for legacy +
  zinb self-zeroed). The body already carries its zeros (zinb's structural π; nb is rejected here at
  the config boundary).
- **`soft_gate`** — the glossary's per-draw composition:
  - `compose_samples`: a per-draw `Bernoulli(gate)` 0/1 mask on the log1p draws. **Deterministic**
    under the caller's seeded `torch.Generator` (S2 #121 gate).
  - `compose_mean`: `gate × mean` (= `E[Bernoulli(gate)·Y]`, the arm's expectation).
- **`threshold_gate(τ)`** — a hard, a-priori threshold:
  - `compose_samples`: keep the whole cell's k-vector where `gate ≥ τ`, else zero it.
  - `compose_mean`: `(gate ≥ τ) × mean`.
  - Boundary is **inclusive** (`gate ≥ τ` keeps). τ is a-priori, never fit on scored months (Goodhart).
- Both preserve input shape and dtype; masks are 0/1 so the log1p multiply is exact (`log1p(0) == 0`).

---

## 4. Inputs and Assumptions

- `composition` is one of `self_zeroed` / `soft_gate` / `threshold_gate` (guaranteed valid by the
  config boundary; re-checked defensively).
- `samples` is `[..., k]` **log1p-space** body draws; `mean` is `[...]` **count-space** `E[Y|body]`.
- `gate` is `[...]` per-cell `P(y>0)` in `[0,1]`, broadcastable over the sample cube's `k` axis.
- `threshold` (τ) is a float in the open `(0,1)` for `threshold_gate`; `None` otherwise.
- `generator` is the caller's seeded `torch.Generator` (used only by `soft_gate`'s Bernoulli).

---

## 5. Consumers

- `to_cube_samples` (`views_hydranet/distributions/sampling.py`) — the D×K sample cube, per MC-dropout
  pass, using the pass's gate.
- `HydraNetInference._emit_magnitude` — the point / AR-feedback forecast the T>0 rollout consumes.
- Both read `forecast_composition` / `gate_threshold` from `self.config`; a legacy head never reaches
  the family branch, so the composer is inert for it.

---

## 6. Failure Modes

- **Unknown arm:** raises `ValueError` (defensive; the config boundary should have caught it).
- **`threshold_gate` without τ ∈ (0,1):** raises `ValueError`.
- **`soft_gate`/`threshold_gate` with `gate=None`** at `to_cube_samples`: raises `ValueError` (a gated
  arm needs a gate).

---

## 7. Test Anchors

- `tests/distributions/test_composition.py` — the composer contract (both functions × three arms +
  determinism + validation).
- `tests/test_composition_wiring.py` — the emit-time wiring (`_emit_magnitude` composes per arm).
- `tests/test_composition_characterization.py` — the byte-identity parity anchor (legacy + zinb
  self-zeroed unchanged; nb passthrough).
- `tests/distributions/test_sampler_dxk.py` — the D×K sampler + determinism.
