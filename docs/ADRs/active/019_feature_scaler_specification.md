# ADR 019: Specification for FeatureScaler (The Normalizer)

**Status:** Accepted  
**Context:** Spatiotemporal signals often require non-linear scaling (e.g., log-transformation) to stabilize training. To maintain the "Producer Contract," we must be able to reverse these transformations with bit-perfect precision. This ADR defines the `FeatureScaler` as the authoritative, stateful gateway for all mathematical transformations.

---

## 1. Ontological Categorization (The "Roles")

### Role 1: Configuration (The Handshake)
... (existing content) ...

### Role 4: Volume Support (Vectorized Inversion)
*   **Responsibility:** Providing high-performance mathematical inversion for contiguous spatiotemporal tensors (VolumeHandlers).
*   **The "Immediate Raw" Principle:** Predictions should be inverse-transformed as soon as they leave the inference engine. This ensures that any subsequent operations (like Point-Collapse in ADR 021) happen in scientifically accurate Raw Count Space.
*   **Prefix-Awareness:** The inversion logic MUST be aware of the `pred_lr_` and `pred_by_` prefixes (ADR 032). It must correctly identify the base target name to resolve the appropriate inverse function while strictly ignoring binary probability heads (`pred_by_`).
*   **Mechanism:** `inverse_transform_volume(VolumeHandler) -> VolumeHandler`. This method applies math directly to the underlying NumPy data using the volume's internal Ledger to map channels to their specific inverse functions.

---

## 2. Structural Invariants (The "Spirit")

1.  **Bit-Perfect Reversibility:** Every transformation method must define a mathematically exact inverse (e.g., `log1p` ↔ `expm1`).
2.  **Stateful Gate Law:** The Scaler is a "one-way gate." Once `fit_transform` is called, the configuration is locked. `inverse_transform` must fail if called before the scaler has been fitted.
3.  **No Content Discovery:** The Scaler never "guesses" which columns need scaling based on their values. It only follows the explicit instructions in the `config`.
4.  **Fail Loud and Proud:** If a requested column is missing from the DataFrame during either forward or inverse passes, the scaler must raise an immediate exception.
5.  **Direct Tensor Math:** Volume transformations must use native NumPy vectorized operations to avoid the "Object Tax" bottleneck.

---

## 3. Data Flow Topology (The Professional Path)
`Model Outputs` → **`FeatureScaler (Role 4)`** → `Raw Volume` → `VolumeHandler (Collapse)` → `Point Volume` → `DataFrame Reconstruction`.

---

## 4. Contractual Precision (The "Constraints")

### `inverse_transform_volume(vh: VolumeHandler) -> VolumeHandler`
*   **Pre-condition:** Scaler state is `LOCKED`.
*   **Post-condition:** Returns a NEW `VolumeHandler` where the `data` array has been mathematically inverted according to the internal channel names in the Ledger.

## 2. Verification Protocol (Team Audit)

### Green Team (Accuracy)
- Prove that `inverse_transform(fit_transform(X)) == X` within float32 precision for all supported methods.
- Verify that `inverse_transform_volume` correctly targets only feature channels while ignoring IDs.

### Beige Team (Robustness)
- Verify that calling `inverse_transform` before `fit_transform` raises a `StateError`.
- Verify that missing configuration for a feature column triggers a `ConfigurationError`.

### Red Team (Invincibility)
- Verify that scaling does not mutate the input DataFrame in-place.
- Verify that NaNs or Infs in non-scaled columns are preserved exactly as they were (no silent healing by the scaler).

---

## 5. Semantic Naming
*   `Raw Space`: Data as it exists on disk or in the final output (e.g., event counts).
*   `Semantic Space`: Data as the model perceives it (e.g., logged intensities).
