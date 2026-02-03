# ADR 019: Specification for FeatureScaler (The Normalizer)

**Status:** Proposed  
**Context:** Spatiotemporal signals often require non-linear scaling (e.g., log-transformation) to stabilize training. To maintain the "Producer Contract," we must be able to reverse these transformations with bit-perfect precision. This ADR defines the `FeatureScaler` as the authoritative, stateful gateway for all mathematical transformations.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Configuration (The Handshake)
*   **Responsibility:** Initializing the scaler with a declarative mapping of columns to methods.
*   **The Law:** The Scaler only transforms columns explicitly listed in the `config`. If a column is missing from the mapping, it remains in its raw state.

### Zone 2: Forward Transformation (Raw → Semantic)
*   **Responsibility:** Applying non-linear math to stabilize input distributions.
*   **Mechanism:** `fit_transform(df)`. This method "locks" the state of the scaler.

### Zone 3: Inverse Transformation (Semantic → Raw)
*   **Responsibility:** Reversing all forward math to restore the original count space for the evaluation package.
*   **Mechanism:** `inverse_transform(df)`.

---

## 2. Structural Invariants (The "Spirit")

1.  **Bit-Perfect Reversibility:** Every transformation method must define a mathematically exact inverse (e.g., `log1p` ↔ `expm1`).
2.  **Stateful Gate Law:** The Scaler is a "one-way gate." Once `fit_transform` is called, the configuration is locked. `inverse_transform` must fail if called before the scaler has been fitted.
3.  **No Content Discovery:** The Scaler never "guesses" which columns need scaling based on their values. It only follows the explicit instructions in the `config`.
4.  **Fail Loud and Proud:** If a requested column is missing from the DataFrame during either forward or inverse passes, the scaler must raise an immediate exception.

---

## 3. Data Flow Topology
`DataFetcher` → `Raw DF` → **`FeatureScaler`** → `Semantic DF` → `VolumeHandler` → `Model` → `VolumeHandler` → `Semantic DF` → **`FeatureScaler`** → `Raw Prediction DF`.

---

## 4. Contractual Precision (The "Constraints")

### `fit_transform(df: pd.DataFrame) -> pd.DataFrame`
*   **Pre-condition:** `df` contains all columns listed in the scaling config.
*   **Post-condition:** Returns a new `df` where values are transformed. Scaler state is `LOCKED`.

### `inverse_transform(df: pd.DataFrame) -> pd.DataFrame`
*   **Pre-condition:** Scaler state is `LOCKED`. `df` contains all columns that were previously transformed.
*   **Post-condition:** Returns a `df` where values are restored to their original scale.

---

## 5. Semantic Naming
*   `Raw Space`: Data as it exists on disk or in the final output (e.g., event counts).
*   `Semantic Space`: Data as the model perceives it (e.g., logged intensities).
