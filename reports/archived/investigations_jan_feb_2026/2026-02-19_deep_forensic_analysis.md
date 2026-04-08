# Deep Forensic Analysis: HydraNet Repository
**Date:** 2026-02-19
**Status:** Completed
**Scope:** Full repository — every source file, test, ADR, CIC, and spec
**Test Suite:** 73/73 tests PASS (45.66s)

---

## 1. System Architecture: What This Thing Actually Does

HydraNet is a spatiotemporal conflict-forecasting pipeline. It ingests a **sparse VIEWS DataFrame** (indexed by `[month_id, priogrid_gid]`), reshapes it into a **3D spatiotemporal volume** (`[T, H, W, C]`), trains a **recurrent U-Net** (LSTM4 variant) on spatial windows, and produces autoregressive predictions reconstructed back into a DataFrame.

### Component Inventory

| Component | File | Role |
|---|---|---|
| `HydranetManager` | `hydranet_manager.py` | Lifecycle orchestrator: train / evaluate / forecast |
| `DataFetcher` | `utils/data_fetcher.py` | Loads raw DF from disk; no sorting |
| `DataSniffer` | `utils/data_sniffer.py` | Validates structure, bounds, continuity; does not check sort order |
| `FeatureScaler` | `utils/feature_scaler.py` | log1p / asinh / identity transforms; one-shot fit |
| `VolumeHandler` | `utils/volume_handler.py` | DF → 3D volume via numpy fancy indexing; the spatial heart |
| `VolumeSampler` | `utils/volume_sampler.py` | Extracts 32×32 windows; importance sampling by activity |
| `CurriculumLearner` | `utils/curriculum_learner.py` | Schedules training lessons |
| `train_model.py` | `train/train_model.py` | Training loop; gradient accumulation; loss computation |
| `HydraBNUNet06_LSTM4` | `architectures/HydraBNrecurrentUnet_06_LSTM4.py` | 3 reg + 3 class heads, hardcoded |
| `choose_loss` | `utils/utils.py:60` | Creates MultiTaskLoss with **hardcoded** `[T,T,T,F,F,F]` mask |
| `MultiTaskLoss` | `utils/mtloss.py` | Learnable task weighting via log-variance |
| `InferenceOrchestrator` | `utils/inference_orchestrator.py` | ADR 038/039: Predict→Wrap→Invert→Collapse→Reconstruct |
| `HydraNetInference` | `utils/hydranet_inference.py` | MC Dropout posterior; autoregressive loop |
| `PureStateAdapter` | `utils/pure_state_adapter.py` | Enforces ADR 032 output schema; reads config targets |
| `HydraNetConfig` | `utils/utils_config.py` | Pydantic validation; `extra="allow"` |
| `ConfigInitializer` | `utils/config_initializer.py` | Raw dict → validated HydraNetConfig → dict |

### Data Flow (Text Diagram)

```
[Disk: raw VIEWS parquet/csv]
        │
        ▼
   DataFetcher.fetch_df()          ← No sort. Returns MultiIndex DF.
        │
        ▼
   DataSniffer.sniff_ingestion()   ← Checks uniqueness, finiteness, bounds.
        │                             Does NOT check sort order.
        ▼
   FeatureScaler.fit_transform()   ← Applies log1p/asinh per transform dict.
        │                             One-shot: fits once, immutable after.
        ▼
   VolumeHandler.from_df()         ← Sparse DF → dense [T, H, W, C] array.
        │                             Uses fancy indexing: r = y_col - row_offset
        │                             c = x_col - col_offset
        │
   ┌────┴─────────────────────────────────┐
   │                                      │
TRAINING                             INFERENCE (Evaluation / Forecast)
   │                                      │
   ▼                                      ▼
VolumeSampler                    InferenceOrchestrator
   │                              (ADR 038/039 sequence)
   ▼                                      │
training_loop                             ▼
   ├─ model.forward(t0)           HydraNetInference.generate_posterior_samples()
   │    → t1_pred [B,3,H,W]          autoregressive: t0 = t1_pred.detach()
   │    → t1_pred_class [B,3,H,W]         │
   │                                      ▼
   └─ choose_loss()               wrap_predictions()
       HARDCODED                         → inverse_transform_volume()
       [T,T,T,F,F,F]                     → collapse_to_point()
       MultiTaskLoss(6)                  → to_evaluation_df()
                                              │
                                              ▼
                                        PureStateAdapter
                                        (reads config targets)
                                              │
                                              ▼
                                       [Final DataFrame]
```

---

## 2. Fault Lines: Complete Inventory

### FAULT LINE 1 — Hardcoded Loss Mask (CRITICAL, ~85% probable)

**Location:** `views_hydranet/utils/utils.py:60`

```python
is_regression = torch.Tensor([True, True, True, False, False, False])
multitaskloss_instance = MultiTaskLoss(is_regression, reduction='sum')
```

This is **completely disconnected from config**. It assumes exactly 3 regression + 3 classification targets, always. It does not read `config["regression_targets"]` or `config["classification_targets"]`.

The training loop in `train_model.py` then does:
```python
for j in range(t1_pred.shape[1]):           # iterates 3 reg heads
    losses_list.append(criterion_reg(...))
for j in range(t1_pred_class.shape[1]):     # iterates 3 class heads
    losses_list.append(criterion_class(...))
losses = torch.stack(losses_list)           # always a tensor of 6
loss = multitaskloss_instance(losses)       # MultiTaskLoss expects 6
```

If the actual config specifies a different number of targets (even just testing with 1 target), the model still trains against all 6 heads, but only 1 has valid ground truth. The other heads optimize against wrong or garbage labels. Semantic disconnect = random output.

**The Smoking Gun:** Every other component correctly reads config:
- `PureStateAdapter`: reads `config["regression_targets"]` and `config["classification_targets"]`
- `InferenceOrchestrator`: reads `config["classification_targets"]`
- Only `choose_loss()` is hardcoded.

---

### FAULT LINE 2 — Spatial Coordinate Scrambling (CRITICAL, ~60% probable)

**Location:** `views_hydranet/utils/volume_handler.py` (from_df method)

```python
r_idx = (df[y_col] - row_offset).astype(int).values
c_idx = (df[x_col] - col_offset).astype(int).values
volume[r_idx, c_idx, t_idx, :] = df[feature_cols].values
```

`row_offset` and `col_offset` come from the external config. There is no validation that the resulting indices are non-negative and within bounds.

**How it breaks:**
- If `row_offset` is set to `100` but the data contains local coordinates starting at 0 → `r_idx` becomes negative → numpy fancy indexing **wraps** to the end of the array → data is written in reversed/scrambled positions.
- A CNN fed spatially scrambled data loses all spatial autocorrelation. Output is indistinguishable from RNG.

**Current safeguard gap:** `DataSniffer._check_spatial_bounds()` checks that `max(y_col) - min(y_col) < height`. This passes even if the offset is wrong — it only verifies the *span*, not the *position*.

---

### FAULT LINE 3 — Scaling Handshake Failure (HIGH, ~40% probable)

**Location:** `views_hydranet/utils/utils_config.py`

`HydraNetConfig` uses `extra = "allow"` (Pydantic class-based config, line 18). This means the config dict is permissive — extra keys are accepted silently.

The `FeatureScaler` applies only the transforms listed in `config["transform"]`. If a target column is absent from the transform dict (e.g., due to a refactor that changed key names), the scaler silently skips it. That column arrives at the model in raw-count space (0 to 5000+) while the model was trained expecting ~log-scale values (0 to ~10).

**What the validation is supposed to catch:**
```python
# utils_config.py lines 120-139
all_required_cols = set(self.features) | set(self.regression_targets) | set(self.classification_targets)
missing = all_required_cols - mapped_set
if missing:
    raise ValueError("Scaling Law Violation...")
```

This is the right check. But it fires only if the Pydantic model is instantiated via `ConfigInitializer`. If the raw config dict is passed directly to any component, this validator never runs.

**Additional note:** `HydraNetConfig` uses the deprecated class-based Pydantic v1 `class Config:` syntax (raised as a warning in the test run). This creates a silent upgrade risk.

---

### FAULT LINE 4 — Unsorted DataFrame (MEDIUM, ~30% probable)

**Location:** `views_hydranet/utils/data_fetcher.py`

`fetch_df()` returns the DataFrame as loaded from disk. No sort by `[time_col, id_col]` is applied.

`VolumeHandler.from_df()` uses fancy indexing which is position-agnostic (writes each row to its computed `r, c, t` position). So spatial placement is correct *if* offsets are right. However:
- `DataSniffer` continuity checks inspect `df[time_col].min()` and `.max()` to infer time range. An unsorted frame with all months present will pass this check even if rows are in random order.
- More subtly, if any component ever relies on positional iteration (e.g., iterating rows to build sequences), unsorted data breaks it silently.

---

### FAULT LINE 5 — Autoregressive Shape Mismatch (MEDIUM, ~15% probable)

**Location:** `views_hydranet/utils/hydranet_inference.py`

```python
t0 = t1_pred.detach()   # shape: [B, 3, H, W]
t1_pred, t1_pred_class, h_tt = self.execute_freeze_h_option(t0, h_tt)
```

The model's first conv layer expects `[B, input_channels, H, W]`. If `config["input_channels"] != 3` (the number of regression output channels), this creates a silent shape mismatch or broadcasting failure during the autoregressive loop.

`HydraNetConfig` does enforce `input_channels == len(features)`, so this is caught at initialization. But if config is malformed and bypasses `ConfigInitializer`, there is no runtime guard inside the inference loop.

This fault line would most likely raise a hard exception rather than produce random output, making it lower priority.

---

### FAULT LINE 6 — Head Count vs Config Target Count (MEDIUM, ~10% probable)

**Location:** `architectures/HydraBNrecurrentUnet_06_LSTM4.py`

The architecture unconditionally creates 3 regression decoder heads and 3 classification decoder heads. There is no `n_regression_heads` parameter driven from config.

If the config specifies fewer than 3 regression targets (e.g., a test run with 1 target), the training loop still computes losses for all 3 heads but only 1 has valid ground truth. The model receives gradient signal from a corrupt loss landscape.

**Gap:** No `assert len(config["regression_targets"]) == 3` exists anywhere. The mismatch would be silent.

---

### FAULT LINE 7 — Pydantic v1 / v2 Deprecation Cliff (LOW, future risk)

**Location:** `views_hydranet/utils/utils_config.py:18`

```python
class HydraNetConfig(BaseModel):
    class Config:        # ← Pydantic v1 syntax, deprecated in v2
        extra = "allow"
```

The test run produced a deprecation warning:
> `Support for class-based config is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0.`

When Pydantic v3 removes this, `HydraNetConfig` silently stops enforcing `extra = "allow"`, and **all config validation breaks** simultaneously without a test failure.

---

## 3. Inconsistencies: Docs vs Code

| Claim | Source | Reality | Status |
|---|---|---|---|
| "All features and targets must be in transform dict" | HydraNetConfig validator | True, but only if ConfigInitializer is used | **Conditional gap** |
| "InferenceOrchestrator implements ADR 039 sequence" | ADR 038 | True — Predict→Wrap→Invert→Collapse→Reconstruct is implemented | ✓ Consistent |
| "VolumeHandler must not collapse S dimension silently" | ADR 012 | True — collapse_to_point is explicit, not silent | ✓ Consistent |
| "Loss function must be config-driven" | (implied by ADR 009 boundary contracts) | False — choose_loss() is completely hardcoded | **Gap** |
| "PureStateAdapter reads regression_targets from config" | ADR 032 | True | ✓ Consistent |
| "input_channels == len(features)" | HydraNetConfig | Validated | ✓ Consistent |
| "n_regression_heads == len(regression_targets)" | No ADR covers this | Not validated anywhere | **Gap** |

---

## 4. Test Suite Analysis

**Result: 73/73 PASS** in 45.66s under `views_pipeline` conda env.

**Coverage observation:** The test suite tests the reconstruction pipeline, volume handler geometry, scaler symmetry, and inference orchestrator contracts thoroughly. What is **not covered** by tests:

- The `choose_loss()` function is not tested for config-driven mask alignment
- There is no test asserting that `len(regression_targets) == 3` matches the architecture
- No test verifies that wrong `row_offset` is caught before volume creation
- The `DataFetcher` sort-order behaviour is not tested

**Legacy tests** exist but are in `legacy_tests/` and are not run by the current pytest invocation. They cover things like `test_manager_smoke`, `test_train_smoke`, and `test_scaling_parity` — these would be valuable to run.

---

## 5. Questions for the Developer

These are the information gaps that cannot be resolved by reading code alone. Answers to these will close the diagnosis.

### Q1 — What is the current `regression_targets` and `classification_targets` in the active config?
The architecture hardcodes 3+3 heads. If the config drifted from this, the loss function is computing against wrong labels.
*Expected healthy answer:* Both lists have exactly 3 members each.

### Q2 — What are the exact values of `row_offset` and `col_offset` in the active config, and what are the actual min/max values of `y_col` and `x_col` in the raw DataFrame?
If `row_offset` doesn't satisfy `df[y_col].min() - row_offset >= 0`, the volume is spatially scrambled.
*Expected healthy answer:* `df[y_col].min() - row_offset == 0` (or close to it), and same for columns.

### Q3 — Is `ConfigInitializer` being called at the entry point, or is the raw config dict being passed directly to components?
If `ConfigInitializer` is bypassed, no config validation (including the scaling law check) ever runs.
*Expected healthy answer:* `ConfigInitializer` is the sole entry point for config.

### Q4 — What changed between the last known-good run and the current broken run?
- Did the config structure change (new keys, renamed keys)?
- Was the `transform` dict modified?
- Were `row_offset` or `col_offset` recalculated?
- Was a different data source used (different coordinate system)?
- Was any dependency updated (numpy, pydantic, pandas)?

### Q5 — Does the raw DataFrame arrive sorted by `[time_col, priogrid_gid]`?
Run `df[time_col].is_monotonic_increasing` immediately after `DataFetcher.fetch_df()`.
*Expected healthy answer:* `True`.

### Q6 — What does `transform` dict look like in the active config?
Specifically: are all 6 targets (3 regression + 3 classification) present as keys or values?
*Expected healthy answer:* All 6 target column names appear somewhere in the transform dict.

### Q7 — Has the model been retrained from scratch since the regression, or is this a failure at inference time on a previously trained model?
If inference is failing: the model weights may be fine but the input preprocessing is broken.
If training is failing: the loss function mismatch (Fault Line 1) is the primary suspect.

### Q8 — What is `input_channels` in the active config, and does it match the number of endogenous conflict features fed back in the autoregressive loop?
If `input_channels != 3` and the autoregressive loop feeds back 3-channel predictions, the loop will silently produce garbage.

### Q9 — Is the `DataSniffer.sniff_volume()` call present and being executed after `VolumeHandler.from_df()`?
If this step is skipped, no structural validation of the volume occurs at all.

### Q10 — Are there any NaN values in the training loss, or is the loss finite but large and non-decreasing?
- NaN loss → Fault Line 3 (scaling failure / gradient explosion)
- Finite but non-converging loss → Fault Line 1 (loss mask mismatch) or Fault Line 2 (spatial scrambling)
- Loss decreasing but predictions still random → Fault Line 1 (optimizing wrong heads against wrong targets)

---

## 6. Priority Remediation Map

| Priority | Fault Line | Action | Cost |
|---|---|---|---|
| 1 | Hardcoded loss mask | Add pre-flight: print `config["regression_targets"]`, print `is_regression` mask, assert lengths match | 5 min |
| 2 | Spatial scrambling | Add `assert (df[y_col] - row_offset).min() >= 0` in VolumeHandler or DataSniffer | 5 min |
| 3 | Scaling handshake | Add CLI log: print transform dict coverage at startup | 5 min |
| 4 | Unsorted data | Add `df.sort_values([time_col, id_col], inplace=True)` in DataFetcher | 2 min |
| 5 | Pydantic deprecation | Migrate `class Config` → `model_config = ConfigDict(extra="allow")` | 15 min |
| 6 | Head count mismatch | Add `assert len(config["regression_targets"]) == 3` at model init | 2 min |
