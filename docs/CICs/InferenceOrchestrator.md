# Class Intent Contract: InferenceOrchestrator

**Status:** Active  
**Owner:** Actor  
**Last reviewed:** 19.02.2026  
**Related ADRs:** ADR-038, ADR-039, ADR-044

---

## 1. Purpose

The `InferenceOrchestrator` is the **Unified Symmetry Engine** of the HydraNet pipeline. Its primary purpose is to execute the mandatory, bit-perfect sequence of operations required to generate spatiotemporal predictions, ensuring that "Forecast-is-Backtest."

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform model training.
- This class does **not** handle raw data fetching from disk (must receive a `VolumeHandler`).
- This class does **not** implement neural network layers (must receive a `nn.Module`).
- This class does **not** decide the scientific scaling of targets (delegates to `FeatureScaler`).

---

## 3. Responsibilities and Guarantees

- **Unification Guarantee:** Ensures that the exact same code path is used for both rolling-origin backtests and single-origin operational forecasts.
- **Law of Sequence (ADR-039):** Guarantees the immutable order: Extrapolate -> Predict -> Wrap -> Invert -> Collapse -> Reconstruct.
- **Identity Integrity:** Ensures that every prediction is anchored to the correct geographic and temporal scaffold provided by the history.
- **Stochastic Awareness:** Correctly handles the 5th dimension (`S`) across the inference sequence without silent collapse.

---

## 4. Inputs and Assumptions

- **History:** Assumes a valid `VolumeHandler` containing the observed historical data.
- **Model:** Requires a trained PyTorch model compatible with the feature set in the history.
- **Origin(s):** Receives either a single integer month or a list of months to initiate forecasting.

---

## 5. Outputs and Side Effects

- **Prediction List:** Produces a list of "Pure State" DataFrames (ADR-032) for every origin provided.
- **Diagnostic Summaries:** Triggers automated summaries of the prediction statistics (ADR-034).

---

## 6. Failure Modes and Loudness

- **Numerical Instability:** Fails immediately if the model produces non-finite values (`NaN`, `Inf`) during the autoregressive loop.
- **Temporal Discontinuity:** Fails if the requested origin does not exist within the provided history.
- **Sequence Violation:** Raises a `ContractViolation` (ADR-008) if the order of operations is bypassed.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Receives instructions from `HydranetManager`.
- **Custodian:** Consumes and produces `VolumeHandler` instances.
- **Actors:** Interacts with `FeatureScaler` for vectorized inversion and `VolumeHandler` for bridge reconstruction.

---

## 8. Examples of Correct Usage

```python
orchestrator = InferenceOrchestrator(config, model, device)

# Operational Forecast
predictions = orchestrator.predict(history_vh, origins=[latest_month])

# Backtest
backtest_results = orchestrator.predict(history_vh, origins=[400, 412, 424])
```

---

## 9. Examples of Incorrect Usage

- **Manual Loop Implementation:** Bypassing the orchestrator to run manual autoregressive loops in the manager.
- **In-Place Inversion:** Attempting to invert predictions before they are wrapped in a `VolumeHandler`.

---

## 10. Test Alignment

- **🟩 Green Team:** Bit-perfect parity tests between backtest and forecast modes in `tests/test_inference_orchestrator.py`.
- **🟫 Beige Team:** Tests for invalid origin months and mismatched feature sets.
- **🟥 Red Team:** Survival tests against model explosion during long-horizon (36 month) roll-forwards.

---

## End of Contract

This document defines the **intended meaning** of `InferenceOrchestrator`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
