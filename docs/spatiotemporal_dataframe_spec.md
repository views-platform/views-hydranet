
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
pred_{target}_raw
```

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
pred_{target}_prob
```

**Cell semantics:**

| Evaluation mode       | Cell contents                                     |
| --------------------- | ------------------------------------------------- |
| Point prediction      | Single scalar probability                         |
| Stochastic prediction | List / np.array of `s` posterior probability samples |

---

## 4. Naming and Alignment Invariants

For each target variable `{target}`, the DataFrame contains the following aligned column triplet:

```
{target}
pred_{target}_raw
pred_{target}_prob
```

Naming rules:

* All predicted columns are prefixed with `pred_`.
* Suffixes disambiguate prediction type:

  * `_raw` → predicted count
  * `_prob` → predicted probability of any occurrence
* `{target}` must be identical across the observed and predicted columns.

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

> A long-format pandas DataFrame indexed by `(month_id, grid_id)`, containing scalar observed count variables and paired predicted count (`pred_{target}_raw`) and occurrence-probability (`pred_{target}_prob`) columns for each target, where predicted cells store either point estimates or fixed-length posterior sample lists (`s = 128`) depending on evaluation mode.


