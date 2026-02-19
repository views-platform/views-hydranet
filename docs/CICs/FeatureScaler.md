# Class Intent Contract: FeatureScaler

**Status:** Active  
**Owner:** Actor  
**Last reviewed:** 19.02.2026  
**Related ADRs:** ADR-003, ADR-009, ADR-019, ADR-032

---

## 1. Purpose

The `FeatureScaler` is the **Normalizer** of the HydraNet pipeline. Its primary purpose is to serve as the authoritative, stateful gateway for all mathematical transformations (e.g., log1p, asinh) and their bit-perfect inverses.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** determine which features to scale (must follow the `config`).
- This class does **not** perform data cleaning or imputation.
- This class does **not** handle file persistence.
- This class does **not** rename columns to reflect scale (Prefix-Purity Law, ADR-003).

---

## 3. Responsibilities and Guarantees

- **Bit-Perfect Reversibility:** Guarantees that every forward transform has a mathematically exact inverse.
- **Stateful Lock:** Guarantees that once the scaler is fitted (`fit_transform`), its parameters are immutable for the duration of the run.
- **Immediate Raw Principle:** Provides high-performance, vectorized inversion for whole volumes (`VolumeHandler`) to ensure downstream math happens in Raw Space.
- **Scientific Accuracy:** Ensures that point-estimates (Averages) are calculated *after* inversion to avoid violating Jensen's Inequality.

---

## 4. Inputs and Assumptions

- **Configuration:** Requires a `transform` dictionary in the config defining the method for each feature/target.
- **Fitting:** Assumes it must be "Fitted" on training data before being used for inference inversion.
- **Data Shape:** Assumes inputs are either `pd.DataFrame` or `VolumeHandler` objects.

---

## 5. Outputs and Side Effects

- **Transformed Data:** Returns scaled DataFrames or Volumes.
- **Internal State:** Stores scaling parameters (e.g., means, scales) if the method is stateful.

---

## 6. Failure Modes and Loudness

- **State Error:** Fails loud if `inverse_transform` is called before the scaler has been initialized/fitted.
- **Missing Config:** Raises an exception if a feature column is encountered that is not covered by the `transform` registry.
- **In-place Mutation:** Strictly prohibited from mutating inbound objects; must return new instances.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Initialized and invoked by `HydranetManager`.
- **Custodian:** Performs direct math on the underlying NumPy arrays of `VolumeHandler` instances.
- **Boundary:** Sits between `DataFetcher` and the model input.

---

## 8. Examples of Correct Usage

```python
scaler = FeatureScaler(config)

# Training Inbound
df_scaled = scaler.fit_transform(df_raw)

# Inference Outbound (Vectorized)
vh_raw = scaler.inverse_transform_volume(vh_semantic)
```

---

## 9. Examples of Incorrect Usage

- **Prefix Tampering:** Renaming `lr_sb_best` to `ln_sb_best` after logging.
- **Scalar Inversion:** Attempting to invert model outputs one sample at a time instead of using the vectorized volume method.

---

## 10. Test Alignment

- **🟩 Green Team:** Round-trip accuracy tests in `legacy_tests/test_scaling_parity.py`.
- **🟫 Beige Team:** Tests for missing transformation keys in the config.
- **🟥 Red Team:** Verification that NaNs or Infs are preserved through transformation and not "hidden" by the scaler.

---

## End of Contract

This document defines the **intended meaning** of `FeatureScaler`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
