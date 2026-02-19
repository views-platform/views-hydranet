# HydraNet Data Pipeline: The Authoritative Flow
**Version:** 1.0 (Restoration Complete)  
**Philosophy:** Geometric Invariance and Immutable Reference Points

---

## 1. Introduction: From "Clever" to "Boring"

The HydraNet Data Pipeline is a high-hygiene spatiotemporal engine designed to transform raw tabular conflict data into 4D/5D geometric volumes for deep learning. 

Historically, this transformation was handled by "Smart" logic that attempted to dynamically discover features and shift coordinates based on the data's immediate context. This "Cleverness" led to structural fragility, including silent indexing shifts and memory explosions.

The modernized pipeline operates on the principle of **Boring Infrastructure**. It treats the data universe as a fixed geographic fixture. Every transformation is invertible, every coordinate is absolute, and every tensor is "Born with a Name." The pipeline serves as a reliable custodian of the raster, ensuring that the model receives exactly what the configuration dictates—no more, no less.

### The Linear Flow:
1. **Ingest:** Fetch and Standardize (DataFetcher).
2. **Validate:** Hard Contract Enforcement (DataSniffer).
3. **Scale:** Raw-to-Semantic Transition (FeatureScaler).
4. **Rasterize:** Building the Immutable Universe (VolumeHandler).
5. **Sample:** Stochastic Importance Windowing (VolumeSampler).
6. **Train:** Guarded Sequence Optimization (Trainer + IntegrityGuardian).

---

## 2. The Ingestion Gate: DataFetcher & DataSniffer

### DataFetcher (The Retrieval)
The flow begins with the `DataFetcher`. Its sole responsibility is the physical retrieval of `.parquet` files from disk and their structural normalization. 
* **The Contract:** It consumes the `index_names` from the "Sacred Source" (Config) and enforces a strict MultiIndex structure (typically `month_id`, `priogrid_gid`). 
* **Result:** A standardized DataFrame where indices are moved into columns, making them accessible for the next stages of the pipeline.

### DataSniffer (The Sentinel)
Immediately after ingestion, the `DataSniffer` performs an exhaustive "Fail-Fast" audit. It is a strictly read-only observer that refuses to "heal" or "fix" data.
* **Obligatory Check:** Ensures all `identity_cols` and `features` requested in the config actually exist.
* **Spatiotemporal Uniqueness:** Verifies that no `(month_id, priogrid_gid)` combination is duplicated, preventing corruption of the 4D raster.
* **Numerical Sanity:** Performs range checks on `month_id` and ensures all feature data is finite (no NaNs or Infs).
* **Spatial Capacity:** Calculates if the geographic **span** of the data fits within the 180x180 fixture.

---

## 3. The Semantic Gateway: FeatureScaler

The `FeatureScaler` manages the boundary between **Raw Space** (real-world counts) and **Semantic Space** (log-scaled features ready for gradient descent).

* **One-Shot State:** Each instance is fitted exactly once and refuses re-fitting to prevent accidental double-scaling.
* **Additive Purity:** The scaler is strictly transformative. It creates a copy of the DataFrame and modifies only the configured feature columns, preserving all Identity and Bookkeeping columns (`c_id`, `month_id`) for the volume builder.
* **Declarative Math:** It uses a registry of invertible functions (e.g., `log1p`, `asinh`, `identity`). 
* **The Audit Report:** Upon transformation, the scaler outputs a detailed "Data State Report" showing the [min, max] range of every column, providing instant visibility into data magnitude before it reaches the GPU.

---

## 4. The Custodian of Geometry: VolumeHandler

The `VolumeHandler` is the immovable reference point of the pipeline. It transforms the tabular DataFrame into a 4D NumPy Volume (`[T, H, W, C]`) and maintains its geographic pedigree.

* **Absolute Anchoring:** It uses `row` and `col` directly as indices relative to a fixed geographic datum (Anchor). This ensures that index `(0,0)` always has a global meaning.
* **Lossless Storage:** Internally, the volume is stored as `float64`. This prevents "Float32 drift" from corrupting integer identities like `priogrid_gid`.
* **The Immutable Ledger:** Every `VolumeHandler` carries a `VolumeMetadata` object (the Ledger) that stores:
    * **Axes Labels:** e.g., `("T", "H", "W", "C")`
    * **Channel Map:** A permanent mapping of indices to names (e.g., Index 5 = `lr_sb_best`).
    * **Geographic Datum:** The `row_offset` and `col_offset` used for anchoring.
    * **Transformation History:** A "Flight Recorder" of every flip or permutation.
* **Symmetry:** The `to_df()` method can reverse the entire process, returning a prediction tensor to a bit-perfect coordinate-aware DataFrame.
* **Visual Audit:** Includes a built-in `visual_audit()` method to render 5x8 geographic heatmaps for manual verification of orientation and feature alignment.

---

## 5. Stochastic Importance Sampling: VolumeSampler

The `VolumeSampler` acts as a "Lens" that looks at the `VolumeHandler`. It is responsible for the destructive act of extracting $32 \times 32$ windows for model training.

* **Gradient Starvation Prevention:** Instead of random sampling, it uses an **Activity Heatmap** to identify "Busy" spatiotemporal tubes. This ensures the model learns from meaningful conflict signals rather than empty ocean/background cells.
* **Translation Invariance:** By randomly jittering the window around an anchor, the sampler forces the model to learn localized feature detectors rather than global map patterns.
* **Batch Orchestration:** It serves windows in batches (e.g., size 3), ensuring each task (SB, NS, OS) is represented in every optimizer step.
* **The Carrier Pattern:** Crucially, the sampler returns **Small Universe** `VolumeHandler` objects. These samples are not naked arrays; they inherit the Ledger and Axis knowledge of their parent, allowing the Trainer to remain layout-agnostic.

---

## 6. The Training Sentinel: IntegrityGuardian

The final gate of the pipeline is the `IntegrityGuardian`. It performs numerical forensic audits during every sequence loop.

* **Explosion Stop:** It monitors loss, activations, and gradients for NaNs, Infs, or magnitude explosions (>10,000). 
* **Hard Stop:** Unlike standard logging, the Guardian raises a `RuntimeError` to immediately halt the run. This prevents "Garbage In, Garbage Out" cycles from wasting GPU hours or polluting W&B logs with corrupt weights.

---

## 7. Conclusion: Reliability via Precision

The HydraNet Data Pipeline has been rebuilt to prioritize **Precision over Cleverness**. By establishing authoritative reference points and immutable ledgers, we have eliminated the "hidden magic" that caused systemic fragility. The architecture is now "Boring"—it is predictable, traceable, and geometrically sound. 🖖