# I/O Format Landscape — Data Flow Reference

**Date:** 2026-06-02
**Status:** Current snapshot — will evolve as FeatureFrame integration proceeds
**Triggered by:** Discovery of hardcoded `set_dataframe_format(".parquet")` in manager

This is a map, not a decision document. It describes the current state of data
formats, ownership boundaries, and known gaps across the pipeline.

---

## Full Data Flow

```
                    INPUT SIDE                                    OUTPUT SIDE

  ┌─────────────┐                                    ┌──────────────────────┐
  │ VIEWSER /    │  .parquet                          │  Track A (numpy)     │
  │ datafactory  │────────┐                           │  y_pred.npy          │
  └─────────────┘         │                           │  identifiers.npz     │
                          ▼                           └──────────┬───────────┘
                 ┌─────────────────┐                             │
                 │ ViewsDataLoader │  (pipeline-core)            │
                 │ reads parquet   │                             │
                 └────────┬────────┘                 ┌───────────┴──────────┐
                          │                          │ PredictionFrame      │
                          ▼                          │ (pipeline-core)      │
                 ┌─────────────────┐                 │ y_pred: np.ndarray   │
                 │ DataFetcher     │  (hydranet)     │ identifiers: dict    │
                 │ fetch_df()      │                 └───────────┬──────────┘
                 │ blueprint       │                             │
                 └────────┬────────┘                             │
                          │ pd.DataFrame              ┌──────────┴──────────┐
                          ▼                           │ InferenceOrchestrator│
                 ┌─────────────────┐                  │ generate_prediction_ │
                 │ VolumeHandler   │  (hydranet)      │ frames()             │
                 │ from_df()       │                  └──────────┬──────────┘
                 │ → np.ndarray    │                             │
                 │ (T, H, W, C)   │                             │
                 └────────┬────────┘                  ┌──────────┴──────────┐
                          │                           │ Model forward()      │
                          ▼                           │ [B, C, H, W] →      │
                 ┌─────────────────┐                  │ reg, cls, h_next     │
                 │ _process_seq()  │──────────────────┘
                 │ training_engine │
                 └─────────────────┘
```

---

## Format Summary

| Stage | Format | Owner | Repo | Serialization | Status |
|-------|--------|-------|------|---------------|--------|
| Raw data (disk) | Parquet | ViewsDataLoader | pipeline-core | `.parquet` | Working |
| Raw data (memory) | pd.DataFrame | DataFetcher | hydranet | MultiIndex | Working |
| **Future input** | **FeatureFrame** | **views-datafactory** | **datafactory** | **`.npy`/`.npz`** | **Designed, NOT connected** |
| Features (memory) | np.ndarray | VolumeHandler | hydranet | (T,H,W,C) | Working |
| Predictions (internal) | PredictionFrame | PredictionFrame | pipeline-core | `.npy`/`.npz` (Track A) | **Authoritative** |
| Predictions (legacy) | Parquet | PredictionFrameConverter | pipeline-core | `.parquet` (Track B) | Optional, gated |

---

## Ownership Boundaries

| Concern | Owner | Location |
|---------|-------|----------|
| Data source selection | ViewsDataLoader | pipeline-core |
| Parquet caching | ViewsDataLoader | pipeline-core (data_raw/) |
| DataFrame validation | CoreDataSniffer (pipeline-core) + DataSniffer (hydranet) | **duplicated** |
| Feature engineering | DataFetcher (hydranet) | apply_blueprint() |
| DataFrame→numpy | VolumeHandler (hydranet) | from_df() |
| numpy→PredictionFrame | PredictionFrameAssembler (hydranet) | assemble_evaluation() |
| PredictionFrame persistence | PredictionFrame.save() | pipeline-core |
| Parquet delivery (Track B) | PredictionFrameConverter | pipeline-core (optional) |
| Format config | PipelineConfig.dataframe_format | pipeline-core (global singleton) |

---

## Known Gaps

### 1. FeatureFrame not integrated (pipeline-core #136)

FeatureFrame exists in views-datafactory with full save/load support but is not
consumed by pipeline-core or any engine. The input path still goes through
parquet → DataFrame → numpy, adding an unnecessary intermediate.

### 2. Global mutable format singleton (pipeline-core #137)

`PipelineConfig.dataframe_format` is set by side-effect in manager constructors,
not driven by model configs. Researchers cannot change it from views-models.

### 3. No unified I/O contract (pipeline-core #138)

ADRs 042, 047, 048 cover pieces. No single document covers the full flow.
This document is the first attempt.

### 4. Ghost prediction_format config (hydranet #52)

`prediction_format: "prediction_frame"` appears in test configs but is not in
HydraNetConfig schema. CoreConfigSniffer requires it at runtime. Pydantic
silently passes it through via `extra="allow"`.

### 5. Track B retirement unclear (pipeline-core #139)

Track B (parquet delivery) still exists but `skip_predictions_delivery=True`
in all HydraNet configs. `PredictionIOManager._upload_to_prediction_store()`
raises `NotImplementedError` for Arrow Tables. Track B may be dead code for
PF models.

### 6. Dual data sniffers

`CoreDataSniffer` (pipeline-core) and `DataSniffer` (hydranet) both validate
input data. No clear boundary between framework-level and engine-level checks.

---

## Target State (strategic direction)

```
FeatureFrame (numpy)  ──→  VolumeHandler  ──→  Model  ──→  PredictionFrame (numpy)
     │                                                            │
     └── .npy/.npz (disk)                                        └── .npy/.npz (disk)
```

No pandas. No parquet in the hot path. FeatureFrame for input, PredictionFrame
for output. Parquet only for legacy consumers and external delivery.

---

## Related Issues

| Issue | Repo | Title | Covers |
|-------|------|-------|--------|
| #136 | pipeline-core | FeatureFrame not integrated | Smell #1 (consuming side) |
| #93 | datafactory | FeatureFrame: expose for consumption | Smell #1 (producing side) |
| #137 | pipeline-core | dataframe_format singleton | Smell #3 |
| #138 | pipeline-core | No unified I/O contract | Smell #4, documentation |
| #139 | pipeline-core | Track B retirement | Smell #2 |
| #140 | pipeline-core | Ownership duplication: dual sniffers/loaders | Smell #5 |
| #141 | pipeline-core | Strategic: full numpy pipeline | Actions #7-9 |
| #52 | hydranet | Ghost prediction_format config | Smell #6 |
| #118 | pipeline-core | Sweep ghost run | Related (format-aware fetch) |
