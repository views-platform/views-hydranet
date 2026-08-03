# Class Intent Contract: IntegrityGuardian

**Status:** Active
**Owner:** Sentinel
**Last reviewed:** 13.03.2026
**Related ADRs:** ADR-028

---

## 1. Purpose

The `IntegrityGuardian` is the **Numerical Sentinel** of the HydraNet pipeline. Its primary purpose is to detect and halt training immediately upon detection of numerical instability (NaN, Inf, or magnitude explosions) in losses, predictions, gradients, or weights.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** fix or heal numerical instabilities (no clamping, no substitution).
- This class does **not** perform model training, inference, or data processing.
- This class does **not** implement the architectural guards described in ADR-028 Sections 1-4 (damping, cell-state clamping, output clamping, layer scaling). See ADR-028 Status Notes.

---

## 3. Responsibilities and Guarantees

- **Loss Monitoring:** `monitor()` guarantees immediate `RuntimeError` if the loss tensor contains NaN or Inf.
- **Prediction Magnitude Ceiling:** `monitor()` guarantees immediate `RuntimeError` if any prediction exceeds ±1,000 in absolute value (`PREDICTION_MAGNITUDE_CEILING`; C-51: lowered from 10,000 for log1p-transformed conflict data).
- **Gradient Scan:** `monitor()` scans all parameter gradients (where they exist) for NaN/Inf.
- **Weight Corruption Scan:** `monitor()` scans all model parameters for NaN/Inf **weights** (at the source, before a corrupt parameter reaches the next forward), in the same `named_parameters()` pass that checks gradients.
- **Numpy Array Monitoring:** `monitor_numpy()` guarantees immediate `RuntimeError` if a numpy array contains any NaN or Inf values.
- **Static Methods:** All public methods are `@staticmethod` — no instance state, no side effects beyond logging and raising.

---

## 4. Inputs and Assumptions

- **Model:** Requires a `torch.nn.Module` with named parameters.
- **Prediction:** Requires a `torch.Tensor` of model predictions.
- **Loss:** Requires a scalar `torch.Tensor` of the computed loss.
- **Context:** Accepts an optional string for error message traceability.

---

## 5. Outputs and Side Effects

- **Success Signal:** Returns `None` silently if all checks pass.
- **Fatal Exceptions:** Raises `RuntimeError` with a descriptive error message on any detection.
- **Logging:** Logs the error via `logger.error()` before raising.

---

## 6. Failure Modes and Loudness

- **NaN/Inf Loss:** Immediate `RuntimeError` — "FATAL NUMERICAL EXPLOSION".
- **Prediction Magnitude >1,000:** Immediate `RuntimeError` — includes the actual max absolute value. Three-tier escalation in inference: WARNING at 100, ERROR at 500, halt at 1,000.
- **Non-finite Numpy Array:** `monitor_numpy()` raises `RuntimeError` with count of non-finite values.
- **Gradient NaN/Inf:** Immediate `RuntimeError` — identifies the specific parameter name.
- **Weight NaN/Inf:** Immediate `RuntimeError` — identifies the specific parameter name.

---

## 7. Boundaries and Interactions

- **Training Loop:** Called by the training loop after each backward pass (`monitor()`, which scans loss, predictions, gradients, and weights in one call).
- **ADR-028:** Implements only the detection/halting portion. The architectural prevention guards (Sections 1-4) are deferred.

---

## 8. Examples of Correct Usage

```python
# After each training step — scans loss, predictions, gradients, AND weights
IntegrityGuardian.monitor(model, prediction, loss, context="Epoch 5, Lesson 42")
```

---

## 9. Examples of Incorrect Usage

- **Wrapping in try-except:** Catching the `RuntimeError` and continuing training defeats the purpose of the halt.
- **Calling before backward:** Gradient scan is meaningless if `loss.backward()` has not been called.

---

## 10. Test Alignment

- **🟩 Green Team:** Clean-pass tests with finite values in `tests/test_integrity_guardian.py`.
- **🟫 Beige Team:** Boundary tests at magnitude 999 and 1,001 in `tests/test_integrity_guardian.py`.
- **🟥 Red Team:** NaN/Inf injection into loss, predictions, gradients, and weights in `tests/test_integrity_guardian.py`.

---

## End of Contract

This document defines the **intended meaning** of `IntegrityGuardian`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
