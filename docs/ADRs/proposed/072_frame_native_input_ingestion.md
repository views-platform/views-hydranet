# ADR-072: Frame-Native Input Ingestion (pandas-free `FeatureFrame` input path)

| ADR Info    | Details |
|-------------|---------|
| Subject     | How raw feature data enters the model — the input-side data container |
| ADR Number  | 072 |
| Status      | **Proposed** |
| Author      | Simon Polichinel von der Maase (decider) · drafted by Claude for maintainer review |
| Date        | 2026-08-13 |
| Builds on   | ADR-047 (pandas-free **output** path), ADR-046 (feature lifecycle), ADR-010/012 (spatiotemporal representation / the volume), ADR-019 (FeatureScaler), ADR-032 (target naming), ADR-060 (static channels), ADR-071 (violet datafactory provider — which schedules this work as its step "S9") |
| Amends-scope | (1) ADR-047 §2's "Input path is out-of-scope" clause; (2) **ADR-019**'s DataFrame-based *forward* FeatureScaler contract (`fit_transform`/`inverse_transform`), which this moves to `FeatureFrame` (ADR-019 Role 4 `inverse_transform_volume` is unchanged — already numpy). Both ADRs (and the FeatureScaler CIC) must be amended in lockstep when this ships. |

> **This is a proposal, not an accepted decision.** It was drafted by an AI agent. Every factual claim
> below is pinned to a `file:line` so you can check it; please verify before adopting. Companion execution
> plan: `reports/2026-08-13_frame_native_input_dossier/`.

---

## 0. TL;DR

Today, feature data enters the model as a **pandas `DataFrame`** and leaves as a numpy container. The
output half was already de-pandased (ADR-047); the input half was deliberately left for later. This ADR is
that later. It proposes that input data arrive as a **`FeatureFrame`** — a small numpy-only container from
the `views-frames` library — so pandas leaves the pipeline entirely. The change is a *refactor*, not a new
behaviour: the guarantee is that the model receives a **bit-for-bit identical** input array before and
after, proven against the current path, and pandas is only deleted once that proof passes.

## 1. Primer — the data pipeline, in plain terms

A HydraNet run moves data through four stages:

```
raw VIEWS data on disk  →  a TABLE of features, one row per (grid-cell, month)
                        →  a dense 4-D ARRAY the CNN consumes  →  predictions
```

- The **input path** is the first arrow: read raw data, assemble the per-(cell, month) table, then pack it
  into the dense array. Today the "table" is a `pandas.DataFrame`.
- The **dense array** — called the **volume** in this codebase — has shape `[T, H, W, C]`: `T` months,
  an `H×W` spatial grid, `C` channels (features + derived targets + static maps). It is what the
  convolutional model actually eats.
- The **output path** is the last arrow: model predictions → scored forecasts. ADR-047 already removed
  pandas from it (predictions are numpy `PredictionFrame`s).

So pandas today lives **only on the input half** — the table-assembly step between "raw data" and "volume".
This ADR removes it from there too, making the whole pipeline numpy/`views-frames`-native end to end.

*"v2"* below means the current data foundation: targets and covariates sourced from **views-datafactory**
(as opposed to the retired *v1* viewser-sourced foundation). See ADR-071.

## 2. Vocabulary (defined once, used throughout)

| Term | One-line meaning | Anchor |
|---|---|---|
| **`VolumeHandler`** | The class that builds and owns the dense `[T,H,W,C]` volume. It carries a self-describing **Ledger** — its coordinate system + channel map (ADR-012) — so that *after* assembly the volume (not the source table) is authoritative. | `views_hydranet/utils/volume_handler.py` |
| **the volume** | The dense `[T,H,W,C]` numpy array fed to the model. `T`=months, `H×W`=grid, `C`=channels. | `volume_handler.py:236` |
| **`FeatureFrame`** | A numpy-only input container from `views-frames`: a `(N, F, S)` float32 array + row labels + feature names. `N`=one row per (cell, month), `F`=features, `S`=a trailing sample axis (`S=1` when not sampled). | `views-frames/src/views_frames/feature_frame.py:26` |
| **`SpatioTemporalIndex`** | The row labels of a frame: two read-only **integer** arrays `time` and `unit` (plus a spatial-level tag), one entry per row. It is the numpy analogue of a pandas `(month, cell)` index. | `views-frames/src/views_frames/index.py:39` |
| **`lr_` / `by_` targets** | `lr_` = a regression/count target (e.g. `lr_sb_best`); `by_` = the binary "did any event occur" version of it (1 if count > 0). | ADR-032:23–33; `volume_handler.py:20-23` |
| **derivation** | Creating a `by_*` column from an `lr_*` one by thresholding (`> 0 → 1.0`). | `data_fetcher.py:199` |
| **static channel** | A feature that is constant over time for a given cell (e.g. a geography map). Must not vary month-to-month. | `volume_handler.py:308-321` |

## 3. Context & problem

**Why pandas is here at all.** The input table — "features per (cell, month)" — is naturally tabular, and
pandas has historically been the container for it. `DataFetcher` reads a pre-baked file into a
`pd.DataFrame` and enforces a `(month_id, priogrid_gid)` MultiIndex on it
(`data_fetcher.py:79-85` rejects any non-`MultiIndex` input outright). That DataFrame is then scaled,
validated, and finally packed into the volume by `VolumeHandler.from_df`.

**Why it is worth removing.**
- **Asymmetry.** The output path is already numpy (ADR-047); only the input path still needs pandas. This
  ADR closes the asymmetry.
- **Undeclared, load-bearing dependency.** `pyproject.toml` declares only `views-pipeline-core`,
  `views-frames`, `torch` (`pyproject.toml:8-13`) — **pandas is not listed**, yet 7 of the 57 package
  modules import it. It works today only because an upstream drags pandas in transitively. If an upstream
  ever drops it, every run breaks at ingestion with nothing in our own manifest explaining why.

**Three facts (verified 2026-08-13) that make this removable now, not "someday".**

1. **The input container already exists and is numpy-only.** `views-frames`
   (`dependencies = ["numpy"]`) ships **`FeatureFrame`** — a `(N,F,S)` float32 array + a
   `SpatioTemporalIndex` + `feature_names` (`feature_frame.py:26-56`) — the input-side sibling of the
   `PredictionFrame` we already use on output. No pandas.
2. **The data provider hands one back for free.** `views_datafactory.load_dataset(...,
   output_format="feature_frame")` returns a `FeatureFrame` and never touches pandas (its storage is
   zarr→xarray→numpy). views-pipeline-core already wraps this in a fetch path whose module docstring is
   literally *"pandas never touched"* (`feature_frame_path.py:1`); the loader entry point is
   `ViewsDataLoader.get_feature_frame` (`dataloaders.py:1112`), which delegates to
   `fetch_feature_frame` (`feature_frame_path.py:125`). *(Correction to an earlier draft: `get_feature_frame`
   is the loader method in `dataloaders.py`; the worker it calls, `fetch_feature_frame`, lives in
   `feature_frame_path.py`.)*
3. **The one non-trivial transform already has a numpy twin, and a test cross-checks it.** The `by_*`
   derivations run today in pandas (`DataFetcher.apply_blueprint`) *and* in numpy
   (`VolumeHandler._execute_derivations`). `tests/test_derivation_parity.py` asserts the two produce
   element-for-element identical results (`np.testing.assert_array_equal`, lines 116-124). *Precise scope:*
   the test cannot import `DataFetcher` (it pulls in `views_pipeline_core`), so it compares
   `_execute_derivations` against a **faithful inline replica** of `apply_blueprint`'s thresholding
   (`test_derivation_parity.py:59-70`), citing `data_fetcher.py:apply_blueprint` as the source of truth
   (lines 6-7). So it proves the *logic* is equivalent, not that the two Python functions are the same
   object.

## 4. Decision

**Input data enters as a `views-frames` `FeatureFrame`, not a `pandas.DataFrame`. No `pd.DataFrame` is
built before `VolumeHandler` on a training or evaluation run.** The `FeatureFrame` + its
`SpatioTemporalIndex` become the authoritative input container; pandas leaves the input path.

### 4.1 Worked example (one feature, one cell, one month)

Suppose the africa grid config has `row_offset=87, col_offset=310, height=180, width=180`, the panel
starts at `month_min=121`, and we have a covariate `X = 3.7` for the grid cell at row `150`, column `400`,
in month `200`.

| Stage | How that datum is represented |
|---|---|
| **Today — pandas row** | one DataFrame row, MultiIndex `(month_id=200, priogrid_gid=…)`, column `X = 3.7` |
| **Proposed — FeatureFrame** | `ff.index.time[r] = 200`, `ff.index.unit[r] = <cell id>`, `ff.feature_names = [..., "X", ...]`, `ff.values[r, feat_idx("X"), 0] = 3.7` |
| **In the volume (unchanged)** | coords `r_idx = 150−87 = 63`, `c_idx = 400−310 = 90`, `m_idx = 200−121 = 79`; written to `vol[63, 90, 79, channel_idx("X")] = 3.7` |

The **volume is identical either way** — only the container the `3.7` travels in before the write changes.
That equivalence is exactly what the parity gate (§7) proves.

### 4.2 The four roles (which code does what)

The input path already separates cleanly into four jobs; the change swaps each job's *container*, not its
*logic*:

| Role | Job | Frame-native form |
|---|---|---|
| **Inbound** (fetch) | get raw features from the provider | `ViewsDataLoader.get_feature_frame()` → `FeatureFrame` (`dataloaders.py:1112`), replacing `DataFetcher.fetch_df()`'s parquet→pandas read |
| **Transform** (math) | derive `by_*` channels; scale features | `VolumeHandler._execute_derivations` (numpy; cross-checked by `test_derivation_parity`) + a `FeatureScaler` that scales `FeatureFrame.values` (a `(N,F,S)` numpy array). **This amends ADR-019's forward contract** from DataFrame-keyed to feature-name-keyed: its Invariant 4 ("fail loud on a missing *column*") becomes "fail loud on a missing `feature_name`"; Invariant 3 (config-driven only, no content discovery) is preserved; the numpy `inverse_transform_volume` (ADR-019 Role 4) is unchanged. |
| **Validate** (gate) | finiteness, bounds, no missing rows, no duplicate rows | numpy checks over `ff.values` + `ff.index` (see §7 for the completeness-vs-uniqueness distinction) |
| **Assemble** (build the Ledger) | pack the table into the dense volume | new `VolumeHandler.from_feature_frame(ff)` — the volume-write loop is unchanged numpy; the *source* of the coordinates and values changes, **and** the one pandas idiom in that method (the static-channel `groupby` guard) is swapped for its numpy equivalent (§5) |

## 5. How it actually works (the assembly step, with the real code)

**Object flow** (which object hands what to which — ADR-000 §1.4):
```
get_feature_frame → _execute_derivations → FeatureScaler → DataSniffer → from_feature_frame → to_pytorch → Model
   (FeatureFrame) →     (FeatureFrame)    →  (FeatureFrame) →  (gate)    →    (volume/Ledger)  →  (tensor)  → …
```

The load-bearing claim is "building the volume is already numpy; only the source container changes." Here
is the actual `from_df` body so that claim is checkable, not asserted.

**1 — coordinates.** Turn each row's (row, col, month) into zero-based array indices
(`volume_handler.py:186-188`):
```python
r_idx = (df[y_col] - row_offset).astype(int).values
c_idx = (df[x_col] - col_offset).astype(int).values
m_idx = (df[time_col] - month_min).astype(int).values
```

**2 — allocate** the empty dense volume (`volume_handler.py:236`):
```python
vol = np.zeros([height, width, month_range, len(channel_map)], dtype=np.float32)
```

**3 — scatter-write** each feature column into the volume at those indices — a numpy fancy-indexing
assignment, no pandas reshape/pivot (`volume_handler.py:322-325`):
```python
for i, col_name in enumerate(channel_map):
    if col_name in geom_static:
        continue  # geometry-derived, not from a column
    vol[r_idx, c_idx, m_idx, i] = df[col_name].values
```

**4 — orient** to image + time-major layout (`volume_handler.py:339-340`):
```python
vol = np.flip(vol, axis=0)          # North-Up (row 0 = northernmost)
vol = np.transpose(vol, (2, 0, 1, 3))  # [H,W,T,C] → [T,H,W,C]
```

**What `from_feature_frame` changes — and only this.** Steps 2–4 are untouched. In step 1 the coordinates
come from `ff.index` (its integer `time`/`unit` arrays) instead of DataFrame columns; in step 3 the values
come from `ff.values[:, feat_idx, :]` (using `ff.feature_names` to find `feat_idx`) instead of
`df[col_name].values`. The DataFrame is never constructed.

**The one genuinely pandas-idiomatic line** is the static-channel guard
(`volume_handler.py:312`), which asserts a "static" feature really is constant over time within each cell:
```python
per_cell_spread = df.groupby([y_col, x_col])[name].agg(lambda s: s.max() - s.min())
# raises if any per-cell spread > 1e-9  (volume_handler.py:314-321)
```
This is a per-cell (grouped) max−min. Its numpy replacement is a segmented reduction over the
`(row, col)` key: sort rows by cell, then `np.maximum.reduceat` / `np.minimum.reduceat` per cell and check
the spread — no pandas index needed. The existing `test_static_channel_seam.py` /
`test_data_backed_static_channel.py` are the parity oracle for that swap.

**Static channels keep their ADR-060 contract, unchanged.** Geometry-derived statics still come from
`GridGeometry` (`volume_handler.py:331-336`), data-backed statics from the frame (ADR-060 I7: role is
authoritative, not column-presence); both are filled **before** the North-Up flip so they flip in sync with
the dynamic channels (ADR-060 **I6**; `:331-336` precedes `:339`). ADR-060 I1/I2 still hold — a static is
input-only, never a target and never in an output frame. The parity gate (§7) therefore covers static
channels too (ADR-060 **I5**: with statics disabled the pipeline stays bit-identical), so this ADR changes
*how statics are sourced*, not the ADR-060 invariants that govern them.

## 6. Scope

**In scope (must be pandas-free for a run):** the inbound fetch (`DataFetcher` →
`get_feature_frame`), `VolumeHandler.from_feature_frame`, `FeatureScaler.fit_transform`/`inverse_transform`,
and the hot-path `DataSniffer` gates (`sniff_ingestion`, `sniff_forecast_alignment`).

**Out of scope (do not entangle):**
- **The provider swap** (viewser→datafactory) — that is ADR-071; this ADR assumes v2 data already lands.
- **Column-name-as-role coupling** (register C-173/C-174): the fact that a channel's *role* is inferred
  from its column *name*. Swapping containers neither fixes nor worsens it; it is a separate refactor.
- **Diagnostic/audit-only pandas:** `DataSniffer.sniff_pure_state_parity` (a DataFrame differ using
  `.duplicated()`/`.equals()`), `visual_diagnostics.biopsy_dataframe` (plots, behind the optional `viz`
  extra), and `utils_logging` banners. None run on a headless production path or return anything the model
  consumes; they may keep pandas behind the optional extra or be gated off.

## 7. Guarantees, and how we prove them

**The governing gate (plain terms):** for the same v2 inputs, the frame-native path must produce a volume
tensor — and therefore predictions — **bit-for-bit identical** to today's pandas path (or within a
*pre-registered* float tolerance, target zero). **Pandas is not deleted until that proof passes.** Same
discipline as ADR-071's clean-cut and the datafactory-dossier P4 gate.

**Structural invariants.**
- **Zero-Magic:** grid coordinates and channel order come from config + the `SpatioTemporalIndex`, never
  from a DataFrame column's *position*.
- **Fail-Fast — completeness:** identifiers in a `FeatureFrame` are required to be **integer** dtype, and
  integers cannot be NaN — so there is no such thing as a silently-missing `(time, unit)` label
  (`views-frames/src/views_frames/_validation.py:45-49`). This replaces one half of what the old pandas
  MultiIndex contract gave us.
- **Fail-Fast — uniqueness (note the subtlety):** a `SpatioTemporalIndex` **deliberately allows duplicate
  `(time, unit)` rows** (register C-21 — cross-level aggregation produces them on purpose;
  `index.py:28-37`). Uniqueness is **not** enforced at construction; it is an opt-in check,
  `has_unique_rows()` (`index.py:231-240`). The old MultiIndex path assumed one row per (month, cell), so
  **the frame-native input path must assert `ff.index.has_unique_rows()`** (or build via the `cartesian`
  constructor, which rejects duplicate inputs at `index.py:104-109`) to preserve that guarantee. This is a
  real design obligation, not a free lunch — call it out in code.
- **Explicit-over-shared:** `from_feature_frame` is a distinct method, not `from_df` with a type flag.

**Verification protocol (ADR-000 §1.7 — Green / Beige / Red).**
- **Green (it does what we claim):** on identical v2 inputs, `from_feature_frame(ff)` volume == `from_df(df)`
  volume (bit-for-bit, or ≤ pre-registered atol); one frozen origin's end-to-end predictions match the
  pandas baseline; `test_derivation_parity` stays green.
- **Beige (failures are loud):** a `FeatureFrame` with a missing month, a **duplicate** `(time,unit)`, an
  off-grid cell, a NaN value, or the wrong `feature_names` must raise a **named** exception at the assembly
  step — never a silent zero-fill or truncation. New tests assert each.
- **Red (it cannot be silently corrupted):** shuffling the frame's rows, corrupting the index integers, or
  dropping a feature cannot change the produced volume without tripping a gate.

**Pre/Post-conditions of `from_feature_frame` (ADR-000 §1.5).**
- **Pre:** `ff.index.has_unique_rows()` is true and the index covers the model's `(time, unit)` grid;
  `ff.feature_names` contains every channel the config requires; `ff.values` is finite.
- **Post:** returns a `VolumeHandler` whose `[T,H,W,C]` volume is fully populated (zeros only where a cell
  is genuinely absent or a static channel is undefined), North-Up, with channel order equal to config order.

## 8. Consequences

**Positive.** Pandas leaves the hot path entirely, matching ADR-047 on the output side. The
undeclared-dependency fragility is closed (we can declare numpy/views-frames honestly and stop relying on a
transitive pandas). One intermediate disappears (parquet→DataFrame→numpy becomes FeatureFrame→numpy). The
global scale-up inherits the lighter path.

**Negative / risks.**
- Two input paths (`from_df`, `from_feature_frame`) coexist until `from_df` is retired.
- The CICs for `DataFetcher` / `DataSniffer` / `FeatureScaler` / `VolumeHandler` must be updated in the same
  PR (ADR-000/007) — a semantic change without a contract update is disallowed here.
- The parity gate is exacting: a pandas reduction and a numpy reduction can differ in the last floating-point
  bit. Any such channel must be reconciled (match the reduction order) — **never** by widening the tolerance
  to make the gate pass.
- The uniqueness obligation (above) is easy to forget precisely because pandas used to give it implicitly.

## 9. Fit with the design principles (SOLID + component principles)

This ADR is a *dependency-and-boundary* change, so it is judged mostly by these principles. Below: what it
advances, and — explicitly — what it **defers** to a separate structural step (§9.3), so nothing is
smuggled into the parity-gated scope.

### 9.1 What this ADR advances
- **DIP (Dependency Inversion).** Ingestion stops depending on the concrete `pd.DataFrame` + parquet reader
  and depends on the minimal, versioned `FeatureFrame` **data contract** instead. The pre-existing
  *undeclared* pandas dependency was itself a DIP violation (depending on a concretion you do not even name);
  declaring `views-frames` fixes it.
- **SDP / SAP (Stable Dependencies / Abstractions).** `views-frames` is the most stable component (numpy-only,
  floor-tested, "root of the dependency DAG"). Depending toward it is depending toward stability. *(Precise:
  `FeatureFrame` is a stable **value-contract**, not a polymorphic abstraction — the right kind of stability
  for a data layer.)*
- **OCP (Open/Closed).** `from_feature_frame` is **added**; `from_df` is untouched under the byte-identical
  gate. Extension, not modification.
- **ADP (Acyclic Dependencies).** One-directional `hydranet → views-frames`; no cycle (the datafactory
  adapters also point *into* views-frames).
- **CRP (Common Reuse).** Removing pandas from the hot path stops every downstream consumer transitively
  pulling it; the viz/audit pandas stays behind the optional `viz` extra.
- **LSP / composition.** The frame trio (`FeatureFrame`/`TargetFrame`/`PredictionFrame`) has **no shared base
  class** (ADR-011 "Option C"): composition over inheritance, so there is no substitution contract to break.
  Both builders yield a substitutable `VolumeHandler`, and the parity gate makes that substitutability *provable*.
- **ISP (Interface Segregation).** `FeatureFrame` is a focused input container; input clients do not depend
  on prediction/actuals methods.

### 9.2 What this ADR does NOT fix (deliberately deferred)
- **SRP (Single Responsibility).** This ADR *adds* `from_feature_frame` to `VolumeHandler`, which is already
  an oversized class slated for a split (register **C-156/C-160**; ADR-062 channel-role refactor). The real
  SRP win — a dedicated frame→volume assembler — is the **separate step** in §9.3, kept out of scope so the
  byte-identical parity proof stays tractable.
- **CCP + Screaming Architecture.** The ingestion modules
  (`data_fetcher`/`feature_scaler`/`data_sniffer`/`volume_handler`) live in `views_hydranet/utils/` — a
  ~40-module grab-bag spanning losses, inference, training, config, and diagnostics. That package **screams
  nothing**. This ADR touches the ingestion modules *together* (a CCP instinct) but does not move them; the
  reorg is §9.3.
- **REP (Reuse/Release Equivalence).** The reused pieces (`views-frames`, `views-datafactory` adapters,
  pipeline-core `feature_frame_path`) are released and *pinned* separately; the parity gate + ADR-071's
  sequencing lock are what keep them released together, and views-models must bump its pipeline-core pin
  (already seen at `==3.0.0`).

**Why not do it all here:** bundling the §9.3 reorg + god-class split into this ADR would give the change
*many* reasons to change and make the parity proof harder — a violation of SRP/CCP applied to the change
itself. Hence the split.

### 9.3 The deferred structural step — a proposed `ingestion/` package (its own ADR, *after* the swap lands)

**Goal:** a new developer reads the package layout and *sees* "this is how raw data becomes a model input"
without opening a file (Screaming Architecture); modules that change together live together (CCP); and the
frame→volume assembly is its own unit (SRP).

**Today** — one grab-bag that names nothing:
```
views_hydranet/utils/            # ~40 modules, 5+ domains, screams nothing
  data_fetcher.py  data_sniffer.py  feature_scaler.py  volume_handler.py
  volume_sampler.py  static_channels.py  grid_naming.py  utils_logging.py
  + ~12 loss modules + inference_orchestrator + curriculum + config_initializer + …
```

**Proposed** — an `ingestion/` package that names the responsibility, one concept per file:
```
views_hydranet/ingestion/
  __init__.py     # public surface: provider → FeatureFrame → volume (the 4 roles of §4.2)
  fetch.py        # DataFetcher   — inbound: provider → FeatureFrame        (was utils/data_fetcher.py)
  scale.py        # FeatureScaler — forward transform on FeatureFrame        (was utils/feature_scaler.py)
  derive.py       # by_* derivations + static-channel logic (ADR-046/060)    (from volume_handler + static_channels)
  validate.py     # DataSniffer   — the hot-path gates                       (was utils/data_sniffer.py)
  assemble.py     # NEW FrameToVolume — the from_feature_frame handshake (§5) (extracted from VolumeHandler)
  volume.py       # VolumeHandler — owns the [T,H,W,C] Ledger + to_pytorch   (slimmed volume_handler.py)
  report.py       # ingestion-only logging                                   (the ingestion slice of utils_logging.py)
```
- **`assemble.py` (`FrameToVolume`) is the SRP extraction:** it holds the coordinate-arithmetic + scatter-write
  (§5) and nothing else. `VolumeHandler` (`volume.py`) then *owns* the volume + its Ledger and no longer
  *builds* it — and this is where `from_df` finally retires (only the assembler knows how to build).
- **CICs move with their classes** (`DataFetcher.md`, `FeatureScaler.md`, `DataSniffer.md`, `VolumeHandler.md`
  + a new `FrameToVolume.md`).
- **Borderline members** decided during the reorg, not now: `volume_sampler.py` (ingestion vs training),
  `grid_naming.py` (inbound boundary vs shared util).
- **Not in this ADR.** A distinct, separately-reviewed step *after* the container swap is byte-identical and
  merged. It carries a *structural* parity discipline (pure moves + one extraction, imports rewired, behaviour
  unchanged). The other `utils/` domains (losses, inference, training, config) get the same treatment in their
  own steps; this ADR only stakes the **ingestion** slice.

## 10. References

**Local ADRs** (clickable, per ADR-000 §1.8):
- [ADR-000](../active/000_standard_for_adrs.md) — the ADR quality standard this doc is written against.
- [ADR-047](../active/047_pandas_free_prediction_output.md) — the **output**-side precedent this mirrors.
- [ADR-019](../active/019_feature_scaler_specification.md) — FeatureScaler spec; this ADR **amends** its
  forward (DataFrame) contract (see header "Amends-scope" + §4.2); its Role-4 `inverse_transform_volume` is
  unchanged.
- [ADR-060](../active/060_static_exogenous_input_channels.md) — static-channel invariants I1–I7 that still
  govern statics in the frame-native path (§5).
- [ADR-012](../active/012_volume_ledger_and_topology.md) — the VolumeHandler **Ledger** (self-describing state).
- [ADR-032](../active/032_authoritative_output_schema.md) — `lr_`/`by_` target naming.
- [ADR-046](../active/046_symmetric_feature_lifecycle.md) — the `transformations`/`derivations` config lifecycle.
- [ADR-071](071_violet_visitor_datafactory_provider.md) §S9 — sequencing lock: this runs *after* the
  provider swap, as a separate byte-identical step.

**Cross-repo** (paths given for discoverability; not relative-linkable from this repo):
- **views-frames:** `src/views_frames/feature_frame.py:26` (`FeatureFrame`), `index.py:39` (`SpatioTemporalIndex`;
  uniqueness stance `:28-37`, `has_unique_rows` `:231-240`), `_validation.py:45-49` (completeness rule),
  `PredictionFrame`/`TargetFrame` (output/actuals siblings).
- **views-datafactory:** `load_dataset(output_format="feature_frame")`; adapters `grid_to_feature_frame` /
  `feature_frame_to_grid`; issue `#381` (drop the transitive xarray→pandas install).
- **views-pipeline-core:** `ViewsDataLoader.get_feature_frame` (`dataloaders.py:1112`) → `fetch_feature_frame`
  (`feature_frame_path.py:125`, docstring "pandas never touched"); issue `#136` (FeatureFrame ingestion).

**Local dossiers / register:** `reports/2026-07-28_datafactory_migration_dossier` (P4), `reports/io_format_landscape.md`
(target-state), register **C-173/C-174** (out-of-scope column-name-as-role coupling), **C-21** (why index rows
are not unique by default).
