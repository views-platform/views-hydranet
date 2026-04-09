
# Spatio-Temporal Panel DataFrame Specification

## 1. Overview

The data produced for EVALUATION are represented as a **pandas DataFrame encoding multivariate spatio-temporal panel data**.
The table serves as the **canonical evaluation and analysis representation** for observed outcomes and model predictions at the **grid-cell–month level**.

The structure supports both **deterministic (point) predictions** and **stochastic predictions** represented as **posterior samples**, while preserving a stable schema and index.

---

## 2. Index Structure

The DataFrame uses a **two-level hierarchical index (MultiIndex)**:

| Index level | Name       | Description                            |
| ----------- | ---------- | -------------------------------------- |
| Level 0     | `month_id` | Discrete monthly time identifier       |
| Level 1     | `grid_id`  | Stable geographic grid-cell identifier |

### Index semantics

* Each `(month_id, grid_id)` pair uniquely identifies **one spatial unit at one monthly time step**.
* Time is **discrete, ordered, and regular (monthly)**.
* Spatial units are **stationary over time** (no drifting or redefined grid cells).

---

## 3. Column Structure

Columns are organized into **three aligned column families**, defined **per target variable**.

Let `{target}` denote the name of an observed count variable.

### 3.1 Observed actuals (counts)

* One column per target variable.
* Scalar realized count values.
* Always deterministic.

**Column name:**

```
{target}
```

---

### 3.2 Predicted counts

* One column per target variable.
* Point predictions or posterior samples of the corresponding count variable.

**Column name pattern:**

```
pred_{target}
```

> **Note:** The `pred_` prefix is prepended to the full target name. For regression targets
> starting with `lr_`, this produces columns like `pred_lr_sb_best`. For classification
> targets starting with `by_`, this produces columns like `pred_by_sb_best`.
> See ADR-032 and ADR-047 for the authoritative naming contract.

**Cell semantics:**

| Evaluation mode       | Cell contents                                             |
| --------------------- | --------------------------------------------------------- |
| Point prediction      | Single scalar value                                       |
| Stochastic prediction | List / np.array of `s` samples from the predictive posterior |

---

### 3.3 Predicted occurrence probabilities

* One column per target variable.
* Represents the probability of observing **any non-zero count** (`count > 0`).
* Not an expected count.

**Column name pattern:**

```
pred_{target}
```

> **Note:** Classification targets use the `by_` prefix (e.g., `by_sb_best`),
> so the predicted probability column becomes `pred_by_sb_best`.

**Cell semantics:**

| Evaluation mode       | Cell contents                                     |
| --------------------- | ------------------------------------------------- |
| Point prediction      | Single scalar probability                         |
| Stochastic prediction | List / np.array of `s` posterior probability samples |

---

## 4. Naming and Alignment Invariants

For each target variable `{target}`, the DataFrame contains an aligned column pair:

```
{target}          (observed value)
pred_{target}     (predicted value)
```

Regression and classification targets are separate variables with distinct names:

```
lr_sb_best        (observed count)
pred_lr_sb_best   (predicted count — regression target)

by_sb_best        (observed binary indicator)
pred_by_sb_best   (predicted probability — classification target)
```

Naming rules:

* All predicted columns are prefixed with `pred_`.
* The target name itself encodes the variable type:
  * `lr_*` targets → predicted counts (regression)
  * `by_*` targets → predicted occurrence probabilities (classification)
* `{target}` must be identical across the observed and predicted columns.
* Each target produces exactly one `pred_{target}` column — no target appears twice.

---

## 5. Stochastic Prediction Semantics

The DataFrame supports **two mutually exclusive evaluation modes**:

### 5.1 Deterministic (point) prediction mode

* All predicted columns contain **scalar values**.
* Exactly one value per `(month_id, grid_id)`.

### 5.2 Stochastic (posterior-sample) prediction mode

* Predicted columns contain **list-valued cells**.
* Each list contains `s` samples drawn from the predictive posterior distribution.
* In the current configuration:
  **`s = 128` samples per cell**.
* All list-valued cells:

  * Have identical length `s`
  * Are aligned across prediction types and targets
* Observed actual columns remain scalar-valued.

This results in a **ragged DataFrame** (value-ragged, not index-ragged) that preserves the long-format panel structure.

---

## 6. Structural Properties

* The DataFrame is a **multivariate spatio-temporal panel**.
* The representation is **long format**, not wide or tensorized.
* Observed outcomes and model outputs are **co-located and aligned** on a shared spatio-temporal index.
* The structure is:

  * Suitable for **model evaluation and diagnostics**
  * Losslessly convertible to:

    * Tensor representations (time × space × channel)
    * Sample-expanded long tables for metrics requiring flat inputs

---

## 7. Compact Technical Summary

> A long-format pandas DataFrame indexed by `(month_id, grid_id)`, containing scalar observed count variables and paired predicted count (`pred_lr_{target}`) and occurrence-probability (`pred_by_{target}`) columns for each target, where predicted cells store either point estimates or fixed-length posterior sample lists (`s = 128`) depending on evaluation mode. The `pred_` prefix is prepended to the full target name; the `lr_` and `by_` prefixes are part of the target name itself, not separate suffixes.


