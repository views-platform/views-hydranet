# 02 — Design: the input-side `FeatureFrame` boundary swap

## The one-sentence design
Replace *"parquet → `pd.DataFrame` (MultiIndex) → `VolumeHandler.from_df`"* with
*"`feature_frame_path.get_feature_frame()` → `FeatureFrame` (numpy) → `VolumeHandler.from_feature_frame`"*,
moving the two real transforms (derivations, scaling) onto their already-existing numpy forms, and letting
`SpatioTemporalIndex` be the input contract the pandas MultiIndex used to be.

## What actually changes (and what does not)

### Inbound — `DataFetcher`
- **Today:** `fetch_df()` reads a pre-baked `<run_type>_viewser_df.parquet` via pipeline-core
  `read_dataframe` (pandas), then `standardize_raw_df` enforces the `(month_id, priogrid_id)` MultiIndex.
- **Frame-native:** call pipeline-core `feature_frame_path.get_feature_frame(...)` (or the datafactory
  `load_dataset(output_format="feature_frame")` it wraps), which returns a `FeatureFrame` whose
  `SpatioTemporalIndex` **is** the completeness/uniqueness contract — so `standardize_raw_df`'s MultiIndex
  gymnastics are replaced by a construction-time invariant, not re-implemented.
- The `apply_blueprint` derivations (ADR-046 binary `by_*` channels) move to
  `VolumeHandler._execute_derivations`, which `tests/test_derivation_parity.py` **already** proves
  equivalent. No new derivation code — reuse the proven numpy twin.

### The Handshake — `VolumeHandler.from_feature_frame(ff)` (new)
The crux, and the good news: `from_df` **already** builds the tensor with numpy, not pandas:
```
r_idx, c_idx, m_idx = (coords derived by arithmetic from columns).astype(int)
vol = np.zeros([H, W, T, C], np.float32)
vol[r_idx, c_idx, m_idx, i] = df[col].values     # scatter-write — the DataFrame dies here
vol = np.flip(vol, 0); vol = np.transpose(vol, (2,0,1,3))   # pure numpy
```
`from_feature_frame` is the *same* body with two sourcing changes:
1. coordinates come from `ff.index` (integer `time`/`unit` arrays) instead of DataFrame columns;
2. channel values come from `ff.values[:, feat_idx, :]` (with `ff.feature_names` giving `feat_idx`)
   instead of `df[col].values`.
The North-Up flip, transpose, static-channel geometry fills, and `to_pytorch` are **unchanged** (already
numpy). `from_df` is kept as a deprecated diagnostic shim during the parity window, then retired.

The single pandas idiom left inside `from_df` — the `groupby([y,x])` time-invariance guard for static
channels (`volume_handler.py:312`) — has a direct numpy replacement: with rows already keyed by the
`SpatioTemporalIndex`, per-cell max−min is a segmented reduction over the (unit)-sorted axis
(`np.maximum.reduceat` / a reshape once the grid is dense), asserted equal in a red test.

### Transformation — `FeatureScaler`
- `fit_transform`/`inverse_transform` are elementwise `log1p`/`asinh` per feature column — rewrite to
  operate on `FeatureFrame.values` (a `(N,F,S)` numpy array) column-wise. The `TRANSFORMS` functions are
  already numpy-compatible; only the container changes.
- `inverse_transform_volume` (the one used on the hot path post-prediction) is **already numpy** — no change.

### Gate — `DataSniffer`
- Hot-path gates (`sniff_ingestion`, `sniff_forecast_alignment`): finiteness, positive-ID, bounds,
  uniqueness — all become numpy checks over `ff.values` + `ff.index`. Uniqueness/completeness are largely
  **free** (the `SpatioTemporalIndex` guarantees them at construction; the gate becomes an assertion, not a
  computation).
- `sniff_pure_state_parity` / `sniff_pure_state_schema` (the `.duplicated()`/`.equals()`/dtype-coercion
  differ) are **audit-only**, not called from `_run_data_pipeline`. Leave on pandas behind the diagnostic
  extra, or gate off — out of scope for a pandas-free *run*.

### Peripheral
- `utils_logging`: reporting only — rewrite the banner to read `ff.feature_names` / `len(ff.index)` (trivial).
- `visual_diagnostics.biopsy_dataframe`: viz-only, null-object-gated, under the `viz` optional extra —
  leave as-is; it never runs headless.
- `hydranet_manager.prepare_actuals_df`: becomes a thin `TargetFrame`/numpy passthrough once `apply_blueprint`
  is numpy; keep a pandas-typed shim only if the pipeline-core actuals handshake still requires one
  (verify against the consumer contract — may itself have a frame-native form).

## Semantic naming (ADR-000 §1.6)
`from_feature_frame` (intent: "ingest a validated feature contract"), not `from_numpy`/`from_array`
(mechanical). The Ledger stays `VolumeHandler`; the external contract is `FeatureFrame`.

## Risks & how the design contains them
- **Float-order divergence** (a pandas reduction vs a numpy one differing in the last bits): the parity gate
  uses a pre-registered `atol`; any channel exceeding it is reconciled (match reduction order) before the
  gate passes — never widened silently.
- **Static-channel `groupby` replacement correctness**: covered by the existing
  `test_static_channel_seam.py` / `test_data_backed_static_channel.py` as the parity oracle.
- **Actuals seam** (`prepare_actuals_df`) may be a pipeline-core contract, not ours to unilaterally change —
  flagged as a verify-before-touch item; keep a pandas shim there if the consumer still demands it, without
  it being on the model *input* path.
- **Cross-repo coupling**: `feature_frame_path` lives in pipeline-core; if its API shifts, our inbound
  breaks. Pin the behaviour with a floor test (as views-datafactory does with `test_views_frames_floor.py`).

## What this design explicitly refuses to do
- Touch the viewser→datafactory provider swap (ADR-071 owns it).
- Fix column-name-as-role coupling (C-173/C-174).
- Change any output-path or scoring code (ADR-047 already done; frozen).
