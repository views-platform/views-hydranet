# Specification Card: <Component Name> v<Version>

## 1. Intended Statistical Model & Purpose

- **Name:** <Official name>
- **Version:** <Version>
- **Purpose:** Describe the high-level intent (e.g., "Quantile Regression", "Time-series forecasting with N-BEATS").
- **Scientific Context:** Why this specific model/algorithm? What problem does it solve in this research context?

---

## 2. Canonical Formula & Code Mapping

Explain the math and how it maps to code.

`Formula: <The Math Expression>`

### Code Mapping:

| Symbol / Term | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `y` | `y` | `targets` | The target value. |
| `y_hat`| `y_hat` | `preds` | The predicted value. |
| `tau` | `tau` | `self.tau` | Description of the hyperparameter. |

### Mandatory Genome (DNA):

List all parameters required for reproducibility.

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `...` | `...` | `...` | `...` |

---

## 3. Domain Constraints & Invariants

- **Input Domain:** (e.g., `torch.Tensor`, non-negative, `float32`).
- **Parameter Bounds:** (e.g., `0 < tau < 1`).
- **Invariants:** What must *always* be true? (e.g., "The loss is always non-negative").

---

## 4. Edge Case Policy

- **Zeros/Nans/Infs:** How are they handled? (e.g., "Fail-loud via NumericalSanityError").
- **Extreme Spikes:** How does the component behave under stress?

---

## 5. Known Equivalences & Invariants

- **Redundancies:** (e.g., "When `tau = 0.5`, this is equivalent to MAE").
- **Invariants:** (e.g., "This loss is scale-invariant if the input is log-transformed").

---

## 6. Practical Guidance & Tuning

- **Hyperparameter Sensitivity:** How do `delta`, `c`, or `a` affect behavior?
- **Scaling Recommendations:** Which data pipeline (log, minmax, etc.) works best?
- **Related Guides:** Link to `reports/guides/` for detailed tuning instructions.

---

## 7. Known Failure Modes

- **"Cowardice" Profile:** Does it tend to collapse to the mean/median?
- **Numerical Instability:** Are there regions where gradients explode?
- **Entropy Sensitivity:** How sensitive is this to the random seed?
