# ADR-066: Decompose `output_distribution` into orthogonal body_family × zero_handling axes

**Status:** Proposed
**Date:** 2026-07-19
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## 1. Context
**Why now?**
- **Problem:** `output_distribution` (a single enum string in `config_initializer.py`, consumed by
  `hydranet_inference.py:_emit_magnitude`) **welds three orthogonal decisions into one value**:
  (a) the body *family* (point / NB / lognormal / quantile), (b) *zero handling* (does the body own
  its zeros, or is there a separate occurrence gate), and (c) — via the `hurdle_` prefix — historically
  the training mask (now `body_mask`'s job, ADR-065). The 8-expert config-landscape review flagged this
  as the load-bearing mess (Hickey: complecting; Martin: SRP violation; Ousterhout: shallow-wide
  interface leaking a decode-`hurdle_shrinkage` burden onto every reader).
- **Naming collision:** `output_distribution='hurdle_shrinkage'` collides with `loss_reg='shrinkage'`;
  "shrinkage" denotes two unrelated things.
- **A concrete capability gap (the urgent driver):** the sibling **views-baseline** horse-race found
  **ZINB** (NB + explicit structural zero-inflation) is the one parametric family that robustly beat
  plain NB *and* the climatology baseline on the low-volume targets, across seeds and partitions — via
  its escalation ladder **all-cell NB → ZINB → hurdle**. ZINB corresponds to *body_family=nb,
  zero_handling=zero_inflation* — a combination the current enum **cannot express**. So the tangle does
  not merely read badly; it blocks the empirically-indicated best next model. (Baseline caveat: a
  different setup — climatology resamplers on the raw target, not this covariate-conditioned ML model —
  so this *indicates*, it does not *prove*. See [[research_baseline_distributional_findings]].)
- **Urgency:** the distributional-head work starts now; building it on the overloaded enum would sprout
  combinatorial values (`gated_quantile`, `dense_nb_zi`, …). Decompose before we add, not after.

---

## 2. Decision
**The new Law of the Land.**
- **Statement:** "We replace `output_distribution` with **three orthogonal, validated config axes** and
  retire the enum behind a translation shim + fail-loud rejection."
  | axis | values | controls |
  |---|---|---|
  | `body_family` | `point` · `nb` · `lognormal` · `quantile` | the magnitude/count distribution |
  | `zero_handling` | `none` · `zero_inflation` · `hurdle` | the escalation ladder: body owns zeros / structural ZI spike / separate gate × positives-body |
  | `body_mask` | `none` (default; ADR-065) | training mask (proven-dead beyond `none`) |
- **The NB-vs-ZINB comparison is a one-variable A/B by construction:** both hold `body_family=nb`,
  `body_mask=none`; only `zero_handling` differs (`none` → all-cell NB; `zero_inflation` → ZINB). This
  is a hard requirement — the config must let a single sweep run both arms with one variable changed.
- **Legal-combo validation** (`model_validator`): `zero_handling='zero_inflation'` requires
  `body_family='nb'` (ZINB is NB-specific); `zero_handling='hurdle'` + a latent all-cell body is
  rejected (mirrors the ADR-065 `pos_*`+latent rule); unknown values fail loud (ADR-009).
- **In-Scope:** the config surface + validators + the `_emit_magnitude`/`choose_loss` wiring + the new
  `zinb` emit path. **Out-of-Scope:** the training loop's `body_mask` (unchanged), the gate/classification
  head (unchanged), the loss registries' internals.

### 2.1 Old → new mapping (byte-identity target)
| retired `output_distribution` | `body_family` | `zero_handling` |
|---|---|---|
| `standard` | point | none |
| `hurdle_shrinkage` | point | hurdle |
| `hurdle_nb` | nb | hurdle |
| `hurdle_lognormal` | lognormal | hurdle |
| `dense_nb` | nb | none |
| `quantile` | quantile | hurdle |
| **NEW** | **nb** | **zero_inflation** (ZINB — previously unexpressible) |

---

## 3. Rationale & Integrity Impact
- **Decomplecting (Hickey) = the core win:** each axis becomes independently settable. "Do I gate?" is
  now `zero_handling`, not a buried enum prefix — the exact question that repeatedly confused this
  program. The `shrinkage` naming collision dissolves (there is no `hurdle_shrinkage` string).
- **Enables the empirically-indicated model:** ZINB becomes a first-class, sweepable arm, and NB-vs-ZINB
  is a clean one-variable test — directly actionable on the frozen ruler.
- **Fail-Loud:** unknown axis values, illegal combos, and the retired `output_distribution` key all raise
  at config time (ADR-008/009), before a GPU is touched.
- **Fortress State:** the six existing enum values map to exact axis combinations proven byte-identical by
  a characterization net, so no current model regresses.

---

## 4. Consequences
### ✅ Positive
- [ ] "Gate vs no-gate" is a legible first-class axis; the `hurdle_shrinkage` collision is gone.
- [ ] ZINB (and future NB×{none,ZI,hurdle}) expressible without new enum churn.
- [ ] Config reads as three obvious knobs, not one decode-me string.

### ⚠️ Negative
- [ ] Migration: every config setting `output_distribution` (views-models floor, sweep drivers, tests)
      must move to the two new keys — a translation shim covers one release, then fail-loud.
- [ ] Two new validated fields + a legal-combo validator (small, principled).

---

## 5. Validation
- **Invariants:** each of the 6 legacy values ⇒ its mapped `(body_family, zero_handling)` pair produces
  a **byte-identical** emitted forecast (characterization net over `_emit_magnitude`). Illegal combos and
  the retired key raise.
- **Tests (ADR-005):** Green — per-combo emit contract incl. the new `zinb` path; Beige — illegal-combo +
  retired-key rejection; Red — a characterization net pins the 6 legacy emits before the refactor.
- **Science bar (higher than calibration-only, per views-baseline meta-finding):** any head graduating off
  this axis is judged on the **validation partition**, **≥3 seeds**, **PIT conditioned on active cells**
  (pooled PIT is a trap — 99% zeros mask it), and **twCRPS** (tail-weighted), not just calibration CRPS.
- **Failure Mode (reopen):** a legacy value ceasing to be byte-identical; or a 4th zero-handling scheme
  arriving (⇒ consider a registry per axis, ADR-049 style).

---

## 6. Implementation Notes
- **Location:** `config_initializer.py` (new `body_family`/`zero_handling` fields + `validate_*` +
  `mode="before"` retirement of `output_distribution`); `hydranet_inference.py:_emit_magnitude` (dispatch
  off the axes; add the `zinb` branch = NB mean with data-matched structural zero-inflation π);
  `utils.py:choose_loss` (ZINB loss / `dense_nb` reuse). Reuse the ADR-065 translation-shim + characterization
  pattern verbatim.
- **First experiment this unblocks:** distributional-head **M1 = all-cell NB vs ZINB** (one variable =
  `zero_handling`), 3 seeds, on the frozen lodestar ruler + the science bar above — the covariate-
  conditioned analogue of the views-baseline ladder, testing whether HydraNet's features beat the
  baselines' shared active-cell under-dispersion.
- **References:** 8-expert config-landscape review (2026-07-19, risks C-a…C-e / D-1…D-3);
  [[research_baseline_distributional_findings]]; ADR-065 (body_mask; the pattern to reuse), ADR-049
  (registry seam), ADR-008/009 (fail-loud), ADR-054/055/059 (latent losses). Glossary `reports/GLOSSARY.md`
  (the §1/§2/§3 axes this ADR makes the config mirror).
