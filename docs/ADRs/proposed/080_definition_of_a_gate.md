# ADR-080: The Definition of a Gate

**Status:** Draft — scaffold only. Sections 1–6 are empty on purpose. What is known so far is in
*Temporary notes* at the end, which is deleted when this ADR is done. ⚠️ Not the current focus:
ADR-079 (the definition of a body) comes first.
**Date:** 2026-11-15
**Deciders:** Simon Polichinel von der Maase
**Informed:** views-hydranet contributors; views-models roster owners

**Related:** ADR-079 (the definition of a body) · ADR-064 (classification-head activation) ·
ADR-069 (`forecast_composition`)

---

## Summary (read this first — self-contained)

*To be written last.*

---

## 1. Context

*To be written.*

---

## 2. Decision

*To be written.*

### In-scope

### Out-of-scope

---

## 3. Rationale & Integrity Impact

*To be written.*

---

## 4. Consequences

### ✅ Positive

### ⚠️ Negative

---

## 5. Validation

*To be written.*

---

## 6. Implementation Notes

*To be written.*

---

## Temporary notes — ⛔ DELETE THIS SECTION WHEN THE ADR IS DONE

What is known about the gate as of 2026-11-15. Nothing here has been discussed as a definition yet.
**Checked** means read in the code or the configs on 2026-11-15. **Read** means taken from a
document and not re-checked.

- **Checked.** Each HydraNet target has two heads. The gate is the classification head: one per
  target, predicting whether the next value is above zero.
- **Checked (ADR-064, Accepted).** The gate heads emit raw logits with no activation at the head.
  The sigmoid is applied inside the loss (`binary_cross_entropy_with_logits`) and explicitly at
  inference. ⭐ This is the same pattern Simon proposed for the body on 2026-11-15 (ADR-079 notes,
  D8): the head emits raw numbers and the loss applies the map itself.
- **Checked.** The gate loss is built per target: one weighted BCE-with-logits loss per
  classification target, each with its own `pos_weight` (`utils/utils.py:261`).
- **Checked.** The gate heads' width is `output_channels` and does not depend on which body is
  chosen (`HydraBNrecurrentUnet_06_LSTM4.py:251`, `:289`, `:339`).
- **Checked.** The gate does not read the body: nothing in the package calls the body's
  `prob_positive`.
- **Checked.** The gate and the body meet in one place, `forecast_composition` (ADR-069). Live
  values across the 37 configs: 10 `self_zeroed`, 12 `soft_gate`, 15 `threshold_gate`.
  `gate_threshold` is set to 0.20 in 13 configs, 0.14 in 1, 0.16 in 1, and `None` in 10.
- **Read (ranking dossier `00_README.md`).** In the ranking programme the gate has exactly three
  licensed roles: it composes the forecast, it is the ordering benchmark, and it defines the firing
  set. It is "not the target".
- **Read (ranking dossier `03` §1c).** The gate is calibrated: median `|p − observed|` 0.0054 over
  842,634 cell-months.
- **Open.** Whether the six-part shape of the body definition (width, meaning, objective, forecast,
  own zeros, required numbers) carries over to the gate, and which gate settings (`loss_class`,
  `pos_weight`, `onset_bias_init`) are part of "which gate" versus numbers a gate requires.
