# Config Cross-Reference: What the Active Config Tells Us
**Date:** 2026-02-19
**Config Source:** `get_hp_config()` (active production config)

This document updates the deep forensic analysis by cross-referencing the active config against the code. Six of the ten original questions are now answered. Three fault lines are cleared. One new bug was found.

---

## Fault Lines Cleared by Config + Code Reading

### ✅ Fault Line 1 — Hardcoded Loss Mask: CLEARED
`regression_targets` = 3, `classification_targets` = 3. The hardcoded `[T,T,T,F,F,F]` mask in `utils.py:60` matches exactly. There is no semantic disconnect in this config.

Additionally, the training loop creates binary targets inline:
```python
# train_model.py:94
t1_binary = (t1.clone().detach() > 0) * 1.0
```
This is correct: `log1p(count) > 0` ⟺ `count > 0`. The classification head receives valid 0/1 labels. No `by_` columns are needed in the volume for training.

### ✅ Fault Line 3 — Scaling Handshake: CLEARED
The `transform` dict covers all 3 features (`lr_sb_best`, `lr_ns_best`, `lr_os_best`) under `log1p`. Regression AND classification targets are all 3 of the same features. Nothing is missing.

### ✅ Fault Line 5 (old numbering) — Autoregressive Shape: CLEARED
`input_channels=3`, model outputs 3 regression channels, autoregressive loop feeds 3 channels back. Shapes match.

### ✅ ConfigInitializer IS called: CONFIRMED
`hydranet_manager.py:84`: `self.configs = ConfigInitializer(self.configs).get_config()` — called at the start of every lifecycle method. Config validation always runs.

### ✅ `evalution_mode` typo: NOT A BUG
`utils_config.py:97-98` has a backward-compat bridge. The Pydantic model uses `evalution_mode` as its canonical field name. The inference orchestrator reads `config.get("evalution_mode")`. Internally consistent throughout.

### ✅ Head count preflight check: EXISTS
`hydranet_manager.py:72`: explicitly checks `if n_reg != 3 or n_class != 3` and warns. This is already wired.

---

## Fault Lines Still Open

### ⚠️ Fault Line 2 — Spatial Scrambling: STILL UNKNOWN (HIGHEST PRIORITY)

**The core uncertainty:** The config has `spatial_cols: ['row', 'col']` and `row_offset: 87`, `col_offset: 310`. The volume is built by `r_idx = df['row'] - 87`.

The config also has `identity_cols: ['month_id', 'priogrid_gid', 'c_id', 'row_id', 'col_id']`. There are **two sets of spatial coordinates** in the DataFrame: `row`/`col` and `row_id`/`col_id`.

The unresolvable question from code alone: **are the values in the `row` column global priogrid coordinates (range ≈ 87–267) or local coordinates (range ≈ 0–179)?**

- If **global** (87–267): `r_idx = row - 87 = 0..179` ✅ correct
- If **local** (0–179): `r_idx = row - 87 = -87..92` ❌ numpy wraps negatives → spatial scrambling

The existence of both `row`/`col` AND `row_id`/`col_id` strongly implies these are different things. The pattern suggests `row` = global and `row_id` = local (already pre-subtracted). If a pipeline refactor replaced `row` with `row_id` in the raw DataFrame (or vice versa), the offset subtraction would silently scramble the map.

**To close this immediately:** `print(df['row'].min(), df['row'].max())` after `DataFetcher.fetch_df()`. If min is near 87 and max near 267 → correct. If min is near 0 → scrambled.

### ⚠️ Fault Line 4 — Unsorted DataFrame: CONFIRMED OPEN

`DataFetcher.fetch_df()` loads without sorting. `DataFetcher.standardize_raw_df()` only calls `df.reset_index()` — no sort. Confirmed: the DataFrame reaches `VolumeHandler.from_df()` in whatever order the parquet file stores it.

`VolumeHandler.from_df()` uses fancy indexing so data placement is position-agnostic — this doesn't affect spatial correctness. However, it can affect the DataSniffer's temporal continuity checks and any code that iterates rows positionally.

Note: `VisualDiagnostics.biopsy_dataframe()` **does sort** at line 58 before plotting, so visual biopsies are immune to unsorted input.

---

## NEW FAULT LINE FOUND: `xavier_norm` Silently Ignored (CONFIRMED BUG)

**Location:** `views_hydranet/utils/utils.py:79-86`

The active config has:
```python
'weight_init': 'xavier_norm'
```

The `init_weights()` function handles only two cases:
```python
if config['weight_init'] == 'xavier_uni':
    nn.init.xavier_uniform_(m.weight)
elif config['weight_init'] == 'kaiming_uni':
    nn.init.kaiming_uniform_(m.weight)
# 'xavier_norm' has NO handler. Falls through. No error. No warning.
```

`'xavier_norm'` is silently ignored. `model.apply(init_fn)` iterates every module but does nothing. The model retains PyTorch's **default initialization** (Kaiming Uniform for Conv2d, which is the PyTorch default). `nn.init.xavier_normal_` is never called.

**Severity assessment:** Unlikely to cause "random noise" on its own — Kaiming Uniform is a reasonable initialization for U-Nets and the model likely converges from it. However, it means the model was never trained with the intended initialization. If the original SOTA result used a different init scheme that happened to work well with this architecture, this silent mismatch could explain a quality regression.

**How to confirm:** The model has been training but not initialized as intended for its entire existence. No previous run has used `xavier_norm` as specified.

---

## Critical Operational Finding: VisualDiagnostics Already Implemented and Wired

**The Visual Diagnostics plan is already implemented and fully wired.** `VisualDiagnostics` is imported and instantiated in all three manager lifecycle methods (train, evaluate, forecast) with biopsies at Stages 1, 2, 3, and 6.

However, it is **completely inactive** because:
```python
# visual_diagnostics.py:28
self.active = config.get("diagnostic_visualizations", False)
```
The active config has **no `diagnostic_visualizations` key**. The default is `False`. Every `viz.biopsy_*()` call is a no-op.

**Single action to activate the entire diagnostic engine:**
Add to the config:
```python
'diagnostic_visualizations': True
```

On the next run, this will generate PNG biopsies to `reports/plots/diagnostics/{timestamp}/` at every pipeline stage. Stage 3 will show the `row`/`col` gradient maps that directly answer the spatial scrambling question.

---

## Remaining Questions (Reduced from 10 to 3)

After cross-referencing the config, the original 10 questions have been narrowed to 3 that cannot be answered from code alone:

### Q1 (Critical) — What are the actual `row` and `col` values in the raw DataFrame?
Run this immediately after `DataFetcher.fetch_df()`:
```python
print("row range:", df['row'].min(), "to", df['row'].max())
print("col range:", df['col'].min(), "to", df['col'].max())
```
Expected if correct: row ≈ 87–267, col ≈ 310–490.
Expected if scrambled: row ≈ 0–179, col ≈ 0–179.

### Q2 (Important) — Is this a training failure or an inference failure?
- If inference on a **previously trained** (pre-regression) model is failing → points to data pipeline issue (offset, scaler)
- If a **freshly trained** model is failing → could be the `xavier_norm` init bug, the offset issue, or both

### Q3 (Important) — What was the `weight_init` value in the last known-good run?
If it was `'xavier_uni'` or `'kaiming_uni'` before the regression and was changed to `'xavier_norm'`, then initialization was working correctly before and is now silently broken.

---

## Summary: Updated Fault Line Ranking

| Fault Line | Status | Likelihood | Severity |
|---|---|---|---|
| Spatial Scrambling (`row_offset`) | **OPEN — cannot close without data** | 70% | Critical |
| `xavier_norm` silently ignored | **CONFIRMED BUG** | 50% | Medium-High |
| Unsorted DataFrame | **CONFIRMED OPEN** | 25% | Medium |
| Loss Mask Mismatch | ✅ Cleared | — | — |
| Scaling Handshake | ✅ Cleared | — | — |
| ConfigInitializer bypassed | ✅ Cleared | — | — |
| evalution_mode typo | ✅ Cleared | — | — |
| Head Count Mismatch | ✅ Cleared | — | — |

## Immediate Action Plan (Priority Order)

1. **Add `'diagnostic_visualizations': True` to config** and run any pipeline stage. The Stage 3 biopsy will show the `row`/`col` gradient maps and answer Q1 definitively within minutes.
2. **Print `df['row'].min()` after DataFetcher** to manually verify offsets before even running the full pipeline.
3. **Add `'xavier_norm'` handler to `init_weights()`** — add `elif config['weight_init'] == 'xavier_norm': nn.init.xavier_normal_(m.weight)` — this is a 1-line fix that closes a confirmed bug.
