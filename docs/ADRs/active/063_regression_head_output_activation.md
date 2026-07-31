# ADR-063: Regression-head output activation — softplus for hurdle bodies, ReLU for standard

**Status:** Accepted
**Date:** 2026-06-26
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## 1. Context
**Why are we doing this now?**
- **Problem:** The regression heads (`HydraBNUNet06_LSTM4`, one head per target — sb/ns/os) must emit a
  **non-negative, continuous, unbounded** value: the target is `log1p(count) ∈ [0, ∞)`. The head's *output*
  activation (the link function applied to each head's final conv `dec_conv4_head{1,2,3}_reg`) historically
  defaulted to **ReLU** for everything except `hurdle_nb`.
- **Assumptions (now false):** that ReLU is a safe positive-output activation. **C-178** falsified this: under
  the hurdle/active_window mask (heavy zero-supervision) on a *rare* target, the pre-activation `H_reg` drifts
  **negative on 100% of cells (including event cells)**, so `ReLU≡0` and — because `ReLU'(<0)=0` — **no gradient
  flows back. The head is unrecoverably DEAD** (verified: ns/os emit identically 0 while the gate fires
  normally). This silently produced zero forecasts for 2/3 targets (#66 flatline, #73 "shrinkage puzzle").
- **Urgency:** the dead head was masquerading as a modeling result; it is a silent Tier-1 defect that must be
  fixed at the architecture default, not patched per-config.

---

## 2. Decision
**The new Law of the Land.**
- **Statement:** "The regression-head output activation defaults to **softplus** for all hurdle output
  distributions (`hurdle_nb`, `hurdle_shrinkage`, `hurdle_lognormal`) and **ReLU** for `standard`. An explicit
  `reg_activation` config key still overrides the default."
- **In-Scope:** the *output/link* activation of the 3 regression heads (`_reg_activation`, applied to
  `H{1,2,3}_reg → out_reg{1,2,3}`); the default-selection logic keyed off `output_distribution`.
- **Out-of-Scope:** the network's **internal** activations (encoder/decoder/bottleneck `F.relu`, the ConvLSTM
  gate sigmoids) — unchanged; the **classification** head activation (see ADR-064); the body **loss/likelihood**
  (orthogonal — losses that need the pre-activation read `reg_latent`).

---

## 3. Rationale & Integrity Impact
- **Logic (Correctness > Convenience):** softplus, `log(1 + eˣ)`, is a **smooth ReLU** with range `(0, ∞)` —
  positive, continuous, **unbounded above** (softplus(20) ≈ 20, no clamp), and `softplus(x) ≈ x` for any
  meaningful positive value. Crucially its derivative is `sigmoid(x) ∈ (0,1)` — **always non-zero**, so a unit
  can never get permanently stuck at 0. This eliminates the ReLU dead-zone (C-178) by construction. It is NOT
  softmax and does not bound or normalise the output. `standard` keeps ReLU so non-hurdle models stay
  byte-identical to the pre-#100 baseline.
- **Fortress State (Numerical Stability):** softplus is sub-exponential, keeping the emitted/fed-back magnitude
  in the `log1p` training range (avoids the C-113 expm1 runaway that an exponential link would invite).
- **Fail-Loud:** the failure was *silent* (identically-zero forecast, no error) — exactly why this is promoted
  to an architecture default plus a regression test, rather than left to per-config discipline.

---

## 4. Consequences

### ✅ Positive
- [x] Eliminates the C-178 dead-ReLU failure mode for every hurdle body (no silent identically-zero forecast).
- [x] Output remains a non-negative, continuous, **unbounded** magnitude — matches the `log1p(count)` target.
- [x] `standard` path unchanged → non-hurdle models byte-identical (regression-safe).

### ⚠️ Negative
- [x] softplus has a soft floor `softplus(0) ≈ 0.69` (never exactly 0). Benign here: the **gate** owns the
  zeros (`E[y] = π · body`), so the body's small positive floor on quiet cells is masked by `π≈0`.
- [x] Artifacts trained with the *old* ReLU default for a hurdle point/shrinkage/lognormal body are activation-
  incompatible on reload (weights were fit for ReLU) → retrain. The shipped production floor (`hurdle_nb`) was
  **already softplus** and is unaffected.

---

## 5. Validation
- **Invariants:** `output_distribution='standard' ⇒ ReLU` (byte-identical to pre-#100); every hurdle body ⇒
  softplus unless `reg_activation` overrides.
- **Tests (ADR-005 taxonomy):** `tests/test_falsify_reg_head_dead_relu.py` —
  `test_hurdle_shrinkage_body_uses_nondying_activation` (Red→Green: asserts the softplus contract) +
  `test_relu_body_can_emit_identically_zero_with_zero_gradient` (characterizes the trap);
  `tests/test_reg_activation.py`; `tests/test_output_distribution_head.py`. Empirical confirmation: the #74
  fix-test (active_window+softplus) revived the ns/os bodies from 100%-zero to alive.
- **Failure Mode:** a hurdle body emitting identically-0 output on cells with events ⇒ reopen this ADR.

---

## 6. Implementation Notes
- **Location:** `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` — `_reg_activation` selection
  (`F.softplus if output_distribution.startswith("hurdle") else F.relu`, with explicit `reg_activation`
  override); applied at `out_reg{1,2,3} = self._reg_activation(H{1,2,3}_reg)`; threaded via
  `choose_model(... reg_activation=config.get("reg_activation"))`.
- **References:** risk register C-178; dossier `reports/2026-06-23_body_sweep_dossier/17_dead_relu_rootcause_and_softplus_fix.md`;
  issue #100 (softplus mu for hurdle_nb). Related: ADR-064 (classification-head activation).
