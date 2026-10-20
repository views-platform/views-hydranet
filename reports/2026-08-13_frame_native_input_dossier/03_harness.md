# 03 — Harness: guardrail audit, handshake specification, pre-flight checklist

**Written 2026-10-19.** The dossier was scaffolded 2026-08-13 without this document, which is why its own
README said *"Nothing here is implemented"* and why it was not executable: a dossier with no harness cannot
be run (rnd-dossier `status` refuses to call such a program ready).

This document does three jobs:
1. **re-verifies every load-bearing claim** in 00/02/04/05 against the code as it stands today — four of
   them were wrong, and two of those change the implementation;
2. **specifies the handshake** (`from_feature_frame`) line by line against the real `from_df` body, so S3
   is transcription rather than design;
3. **states the pre-flight checklist** that gates S1.

Every claim below carries a `file:line`. Nothing here is from memory.

---

## 1 — Verification pass (2026-10-19)

> ⛔ **Attribution, checked before this document shipped.** The corrections below are corrections to **this
> dossier (00/02/04/05)**, not to **ADR-072** — which is a much stronger, code-verified document and already
> carries two of them. ADR-072 `:89-91` *already* corrects the `get_feature_frame` naming ("Correction to an
> earlier draft"), and `:228-232` *already* states the uniqueness obligation in full, citing views-frames
> C-21, `index.py:28-37` and `has_unique_rows`, and calling it *"a real design obligation, not a free
> lunch"*. Its §5 line numbers (`:236`, `:322-325`, `:339-340`) are the same ones this audit re-derived.
> **Four findings are new against ADR-072 as well**, and are listed in §1.9; they are amended into the ADR
> rather than only recorded here. I nearly filed §1.2 and §1.3 as discoveries — the same re-derivation
> failure the programme has recorded three times (`RESULTS.md §12`).

### 1.1 Claims that HOLD

| Claim (source) | Verdict | Evidence |
|---|---|---|
| pandas imported by 7 production modules | **HOLDS** | `data_sniffer.py:11`, `feature_scaler.py:10`, `data_fetcher.py:10`, `hydranet_manager.py:14`, `utils_logging.py:10`, `visual_diagnostics.py:11`, `volume_handler.py:9` |
| ~~pandas undeclared in `pyproject.toml`~~ | ⛔ **NO LONGER HOLDS — fixed 2026-10-20** | `pandas>=1.5.3,<3.0` is now declared, range matched to views-pipeline-core's. It held when this was written, and 0.1.0–0.1.3 shipped without it |
| `FeatureFrame` is `(N,F,S)` float32 + index + names | **HOLDS** | `views-frames/src/views_frames/feature_frame.py:26`, `:88-104` |
| `SpatioTemporalIndex` exists, integer `time`/`unit` | **HOLDS** | `views-frames/src/views_frames/index.py:25`, `:44-60` |
| datafactory's `output_format` default is `feature_frame` | **HOLDS** | `views-datafactory/src/datafactory_query/dataset.py:124` |
| the frame path never touches pandas | **HOLDS** | `views-pipeline-core/.../feature_frame_path.py:1-24` module docstring, falsify P4 |
| the four named parity oracles exist | **HOLDS** | `tests/test_derivation_parity.py` (124 ln), `test_volume_handler_hard_gates.py` (459 ln), `test_static_channel_seam.py` (140 ln), `test_data_backed_static_channel.py` (193 ln) |
| the tensor build is already numpy; the DataFrame dies at the scatter-write | **HOLDS** | `volume_handler.py:236-340` — `np.zeros`, scatter, `np.flip`, `np.transpose` |

### 1.2 ⛔ Correction 1 (dossier only — ADR-072 `:227-234` already has this right) — `SpatioTemporalIndex` does **not** give uniqueness or completeness

02_design says the index *"**is** the completeness/uniqueness contract — so `standardize_raw_df`'s MultiIndex
gymnastics are replaced by a construction-time invariant, not re-implemented"*, and 05's P5 leans on it for
the duplicate-row case.

**That is false.** `index.py:25-38` states the opposite in its own class docstring: *"a frame **may** contain
duplicate `(time, unit)` rows … uniqueness is **not** a global invariant and is **not** validated at
construction"* (views-frames register C-21). The check is opt-in: `has_unique_rows()` at `index.py:245-254`,
`O(n log n)`, *"not run by default."* Completeness is likewise only a property of the `cartesian`
constructor (`index.py:46`), not of the type.

`_validation.validate_values` (`views-frames/src/views_frames/_validation.py:79-106`) rejects non-float32, object dtype
and `ndim < 2` — and **does not reject NaN or Inf.**

**Consequence (changes the build):** `DataSniffer._check_spatiotemporal_uniqueness` and
`_check_non_finite` (`data_sniffer.py:54,57`) must be **ported to numpy, not deleted**. The handshake calls
`ff.index.has_unique_rows()` explicitly and fails loud — exactly as ADR-072 `:227-234` already requires.
02's claim that uniqueness becomes "free" is struck. This is the C-303 class (prose asserting a guard the
code does not implement) **in the dossier**, contradicting its own ADR.

### 1.3 ⛔ Correction 2 (naming: ADR-072 `:89-91` already corrected it. ⚠️ The *who calls* half is new to both) — hydranet should call **neither** API

02/04 name `feature_frame_path.get_feature_frame(...)`. The module-level function is **`fetch_feature_frame`**
(`feature_frame_path.py:125`); `get_feature_frame` is the loader method
(`dataloaders.py:960-985`, signature `partition, use_saved, level, validate=True, override_month=None`).

But the deeper point is that **hydranet never fetches.** `DataFetcher.fetch_df` reads a pre-baked parquet
off disk (`data_fetcher.py:60`, `read_dataframe(path_raw_file)`); the fetch happens upstream in
views-pipeline-core's `ModelManager`.

And **that upstream half is already built and shipped** (pipeline-core #290):

| piece | location |
|---|---|
| the config declaration `data_format: feature_frame` | `datafactory_contract.py:33-49` (`declared_data_format`; unknown value fails loud) |
| the dispatch to the frame fetch | `managers/model/model.py:1269-1277` |
| the cached frame directory handed to engines | `model.py:1010-1024` `_get_cached_frame_path()` — *"Engine subclasses call this instead of hardcoding the directory convention"* |
| the loader for it | `frame_cache.load_frame_cache(cache_dir) -> FeatureFrame` (`frame_cache.py:141-160`) |
| the migration message pointing here | `model.py:64-69` — *"migrate the model to data_format='feature_frame'"* |

**Consequence (shrinks the build substantially):** hydranet's inbound change is a **like-for-like disk read**
— `read_dataframe(parquet)` → `load_frame_cache(self._get_cached_frame_path())` — not a new fetch path, not
a cross-repo API integration. The cross-repo work was done two months ago in pipeline-core and nobody here
noticed. S3's inbound stage collapses to a few lines.

### 1.4 ⛔ Correction 3 (new against ADR-072 too) — the derivations do not need to move; they already moved

02/00 "Why now" item 4 makes the `by_*` derivation parity the pivot of the migration (*"the one non-trivial
transform already has a proven numpy twin"*). Re-checked: **the numpy twin is already the one on the hot
path.**

- `volume_handler.py:103` calls `self._execute_derivations(config["derivations"])` during construction.
- `channel_map` (`volume_handler.py:165-168`) = `[time_col, id_col] + identity_cols + features +
  static_channels`. For amber_kestrel that is
  `['month_id','priogrid_id','c_id','row','col','lr_sb_best','lr_ns_best','lr_os_best']`
  (`config_hyperparameters.py:78-84`). **No `by_*`.**
- So the `by_*` columns `apply_blueprint` writes into the DataFrame (`data_fetcher.py:151`) are **never read
  into the volume.** The channels the model trains against come from `_execute_derivations`, in numpy,
  already.

**Consequence:** `apply_blueprint` is dead weight on the *input* path and is **deleted**, not ported. Its one
live consumer is `HydranetManager.prepare_actuals_df` (`hydranet_manager.py:199`) — the actuals seam already
pre-registered as F4. `test_derivation_parity.py` stays green throughout and becomes a regression test for a
path that is already the only one in use.

### 1.5 ⛔ Correction 4 (new against ADR-072 too — its §4.1 worked example assumes `row`/`col` exist) — the only genuinely new code is a priogrid→(row,col) decoder, and it is needed twice

`from_df` requires `config['spatial_cols'] = ['row','col']` **as DataFrame columns** and fails loud without
them (`volume_handler.py:142-160`), then uses them for the scatter indices (`:186-187`).

Neither datafactory adapter supplies them: `grid_to_dataframe` returns a `(month_id, priogrid_id)` MultiIndex
plus feature columns only (`grid_to_dataframe.py:84-90`). They are added **by pipeline-core, on the pandas
path only**:

```python
# views-pipeline-core/.../dataloaders.py:537-542
if loa == "priogrid_month" and _grid_level is not None:
    pgids = df.index.get_level_values(_grid_level)
    if "row" not in df.columns:
        df["row"] = ((pgids - 1) // _PRIOGRID_NCOL + 1).astype(float)   # _PRIOGRID_NCOL = 720 (:66)
    if "col" not in df.columns:
        df["col"] = ((pgids - 1) %  _PRIOGRID_NCOL + 1).astype(float)
```

The frame path deliberately refuses to: *"no `row`/`col` derivation — grid geometry is engine domain;
engines derive from the canonical `priogrid_id` unit ids when they need it"*
(`feature_frame_path.py:17-19`).

**We are that engine.** And `row`/`col` are needed **twice**, which 02 misses entirely:
1. as the **scatter indices** (`r_idx`, `c_idx`);
2. as **channel values**, because `identity_cols` lists them (`config_hyperparameters.py:81`) so they occupy
   channels 3 and 4 of the volume and reach the model's identity channels.

**Consequence:** one new function, `views_hydranet/utils/grid_geometry.py::row_col_from_priogrid(unit)`, with
the pipeline-core line above as its parity oracle, and `row_offset: 1 / col_offset: 1` already consistent
with it (`config_hyperparameters.py:88-91` documents exactly this: *"pipeline-core numbers row/col from 1"*).
This is the single highest-risk piece of the migration and it is 2 lines of arithmetic with an exact oracle.

### 1.6 ⛔ Correction 5 (new against ADR-072 too — it flags last-bit *reduction* differences at `:264-265`, a different mechanism, and prescribes no policy) — the precision asymmetry will break the atol=0 gate

05's P1 pre-registers `atol` **exactly 0**. Derivation says it cannot hold as the design stands:

| step | pandas path | frame-native path |
|---|---|---|
| delivered dtype | `ensure_float64(df)` — **float64** (`dataloaders.py:577`) | **float32**, by `FeatureFrame` contract (`_validation.py:101`) |
| scaling | `np.log1p` on a float64 Series (`feature_scaler.py:121`, `config_initializer.py:19`) → **float64 log1p** | `np.log1p` on `ff.values` → **float32 log1p** |
| volume write | downcast to float32 at the scatter (`volume_handler.py:236`) | already float32 |

`float32(log1p_f64(x))` and `log1p_f32(float32(x))` differ in the last bit for some `x`. So a frame-native
scaler written the obvious way fails P1, **F1 fires, and 04's rule says STOP** — for a reason that has
nothing to do with the container. That is a false negative built into the pre-registration.

**Consequence (pre-registered design constraint, not a tolerance widening):** the frame-native scaler
**upcasts to float64 before the transform and lets the volume scatter-write do the only downcast**, mirroring
the pandas path exactly. Then P1 at `atol=0` is reachable and remains a real test. This constraint goes into
05 as part of the intervention definition, so "match the reduction order" is specified in advance rather than
improvised at the gate.

**Still unmeasured (pre-flight item PF-6):** the dtype of the values as they leave zarr. If the source is
float64 and the frame coerces to float32 *before* we see it, information is lost upstream of us and P1 is
structurally unreachable — a STOP that must be known before S1, not discovered at S4.

### 1.7 ⛔ Correction 6 (new against ADR-072 too) — the NaN-fill asymmetry is a parity hazard with a known counter

The pandas path fills NaN with 0.0 and **logs the count** (`dataloaders.py:561-574`). The frame path does
not fill: *"no silent `fillna(0.0)` — NaN policy belongs to the engine boundary"* (`feature_frame_path.py:20-22`).

`FeatureScaler.fit_transform` already rejects NaN and Inf loudly (`feature_scaler.py:49-75`), so the frame
path does not go silent — it goes **loud**, where the pandas path went quiet. That is a behaviour change,
and the right one, but it is a change.

**Consequence:** PF-5 reads that logged fill count on the anchor. **Zero** ⇒ the two paths agree and parity is
clean. **Non-zero** ⇒ the frame path raises where the pandas path trained, and the migration surfaces a
pre-existing coverage hole. Record it as a finding; do not fill it to make the gate pass.

### 1.8 Claim NOT yet verified, and it is the hard precondition

04 requires *"the datafactory provider swap (ADR-071) has landed for the target model(s)."* **ADR-071 is
still in `docs/ADRs/proposed/`.** In practice the swap *has* landed for amber_kestrel —
`models/amber_kestrel/configs/config_queryset.py` returns `{"source": "views-datafactory", ...}` — and that
is what matters, because `fetch_feature_frame` refuses outright for a non-frame-capable source
(`feature_frame_path.py:165-171`). So:

- the **data** precondition is met for amber_kestrel (and any other datafactory model);
- the **governance** precondition is not: two ADRs in `proposed/` and the ADR-071 ordering lock is recorded
  in a document that was never accepted. PF-1 covers this.
- the three viewser models (`pink_pirate`, `blue_stranger`, `purple_alien`) **cannot** be migrated by this
  program at all; they are excluded by the source gate, not by scope choice.

### 1.9 What is new against **ADR-072** (and therefore amends it)

| id | finding | where the ADR is affected |
|---|---|---|
| **N1** | pipeline-core #290 ships the `data_format: feature_frame` dispatch (`model.py:1269-1277`), `_get_cached_frame_path()` (`:1010-1024`) and `load_frame_cache` (`frame_cache.py:141`). hydranet reads an artifact off disk and never holds a loader | §4.2's **Inbound** row names `ViewsDataLoader.get_feature_frame()` as hydranet's call. It is the *manager's*, upstream. The engine's call is `load_frame_cache(self._get_cached_frame_path())` |
| **N2** | `apply_blueprint`'s `by_*` columns are absent from `channel_map` and never enter the tensor; `_execute_derivations` is already the only live derivation path | §3 fact 3 and §4.2's **Transform** row treat derivation parity as load-bearing for the migration. It is a regression test, and `apply_blueprint` is **deleted** from the input path |
| **N3** | `row`/`col` are supplied by neither datafactory adapter; the frame path refuses by contract; `identity_cols` needs them as channel values too | §4.1's worked example assumes "the grid cell at row 150, column 400" and §5 step 1 says coordinates "come from `ff.index`". A decoder is required, used twice |
| **N4** | the pandas path scales in float64 and downcasts at the write; a float32 frame-native scaler differs in the last bit | §8 flags last-bit *reduction* differences and forbids widening the tolerance, but names no mechanism and no policy. The policy is float64 transform, float32 only at the scatter |
| **N5** | the pandas path NaN-fills and logs the count (`dataloaders.py:561-574`); the frame path refuses to, by contract | §7's Beige rule already demands a loud NaN failure — so the ADR is *correct* and the **behaviour change relative to today** is simply unstated. Measure the fill count before S1 |
| **N6** | `FRAME_CAPABLE_SOURCES` excludes the three viewser models outright; `static_channels: []` fleet-wide means the `groupby` is dead code today | §6 scope should say the viewser models are excluded by a source gate, not deferred by choice, and §5's "one genuinely pandas-idiomatic line" does not execute on any shipped config |
| **N7** | **three of ADR-072's citations have drifted** in two months: `dataloaders.py:1112` (cited 3×) points **past the end** of a 1010-line file — the method is now at `:960`; `has_unique_rows` moved `index.py:231-240` → `:245-254`; the uniqueness stance `:28-37` → `:25-38`. `_validation.py:45-49` is still correct. | §3 fact 2, §4.2, §7 and §10's reference list. A plan whose line numbers no longer resolve is a plan a reader cannot check — re-pin before review |

---

## 2 — The handshake specification

Transcribed from `from_df` (`volume_handler.py:106-345`). `from_feature_frame(ff, config)` performs the same
steps; only the sourcing lines differ. ✅ = unchanged code, ⚙️ = re-sourced, 🆕 = new.

| # | step | `from_df` | `from_feature_frame` |
|---|---|---|---|
| 1 | ledger roles from config | `:117-150` | ✅ identical (still `time_col`, `id_col`, `spatial_cols`, offsets) |
| 2 | handshake / missing-input check | `:152-160` missing **columns** | ⚙️ missing **feature_names**, plus `ff.values.shape[2] == 1` (S axis; `from_2d` convention, views-frames ADR-012) |
| 3 | row uniqueness | pandas MultiIndex + `_check_spatiotemporal_uniqueness` | 🆕 `ff.index.has_unique_rows()` — **not free** (§1.2) |
| 4 | finiteness | `DataSniffer._check_non_finite` + `FeatureScaler` guards | 🆕 explicit `np.isfinite(ff.values).all()` — the frame does not check (§1.2) |
| 5 | channel map | `:165-171` | ✅ identical ordering rule |
| 6 | time anchoring | `df[time_col].min()/.max()` `:181-183` | ⚙️ `ff.index.time.min()/.max()` |
| 7 | **row/col** | `df[y_col]`, `df[x_col]` | 🆕 `row_col_from_priogrid(ff.index.unit)` (§1.5) |
| 8 | scatter indices | `:186-188` | ⚙️ same arithmetic on the arrays from 6 and 7 |
| 9 | spatial integrity guards | `:190-234` | ✅ identical (operate on `r_idx`/`c_idx`, not on the container) |
| 10 | allocation | `np.zeros([H,W,T,C], float32)` `:236` | ✅ identical |
| 11 | dense time channel | `:238-241` | ✅ identical |
| 12 | identity channels | `df[col].values` | ⚙️ `month_id`/`priogrid_id` from `ff.index`; `row`/`col` from 7; `c_id` from `ff.values[:, idx, 0]` |
| 13 | static-channel classification | `:242-277` | ✅ identical (registry-driven, not column-driven — C-237) |
| 14 | data-backed static guards incl. the `groupby` | `:279-320` | ⚙️ the `groupby` at **`:312`** → segmented max−min. **Dead on the shipped fleet**: every model config declares `static_channels: []`, so this branch does not execute today. Port it for correctness; it is not on the parity critical path |
| 15 | feature scatter | `vol[r,c,m,i] = df[col].values` `:322-325` | ⚙️ `= ff_f64[:, feat_idx, 0]` — **float64 input, float32 destination** (§1.6) |
| 16 | geometry statics | `:331-337` | ✅ identical |
| 17 | flip + transpose | `:339-340` | ✅ identical |
| 18 | derivations | `:103` `_execute_derivations` | ✅ identical — already the only live path (§1.4) |

**Net new code:** one 2-line decoder, one uniqueness call, one finiteness call, one segmented reduction (for
a branch nothing currently runs), and a dtype policy. Everything else is re-sourcing.

---

## 3 — Guardrails that already exist (reuse, do not rebuild)

| guardrail | location | role in this program |
|---|---|---|
| derivation parity oracle | `tests/test_derivation_parity.py` | regression: the numpy twin stays equal to the retired pandas one |
| volume hard gates | `tests/test_volume_handler_hard_gates.py` (459 ln) | the loud-failure suite `from_feature_frame` must satisfy identically |
| static-channel seams | `tests/test_static_channel_seam.py`, `tests/test_data_backed_static_channel.py` | oracle for step 14 |
| reproducibility gate | `views_hydranet/infrastructure/reproducibility_gate.py:47` | P3's determinism precondition; `lock_entropy` on the inference path (`hydranet_inference.py`) |
| frame-native audit on every delivery | `CoreFrameSniffer` via `fetch_feature_frame(..., validate=True)` | upstream gate; **does not** cover uniqueness/NaN for us (§1.2) |
| cache integrity | `frame_cache.load_frame_cache` raises on torn write / empty frame / layout mismatch (`frame_cache.py:144-160`) | replaces nothing in hydranet; strictly better than `read_dataframe` |
| suite | 189 test files | the standing green bar |

## 4 — Guardrails that do NOT exist (the gaps this program must close)

| gap | why it matters | precedent to copy |
|---|---|---|
| **no import-purity test in views-hydranet** | F3 (hidden pandas) is unfalsifiable without one, and this is exactly how pandas stayed undeclared through release 0.1.2 | `views-pipeline-core/tests/test_import_purity.py`, `views-datafactory/tests/test_import_purity.py` |
| **no dependency-floor test** | `views-frames` / pipeline-core API drift breaks us silently | `views-datafactory/tests/test_views_frames_floor.py`, `views-pipeline-core/tests/test_dependency_floors.py` |
| ~~**pandas undeclared in `pyproject.toml`**~~ ✅ **CLOSED 2026-10-20** | was: a published package with a load-bearing undeclared import; same class as #381 (wandb, undeclared until 2026-09-19) | the `#381` comment already in `pyproject.toml` |
| **no golden-reference capture tool** | S1 has nothing to capture with | S1 builds it; `scripts/arm_identity_check.py` is the house idiom |
| **no priogrid→(row,col) decoder, and no test of one** | §1.5 | `dataloaders.py:537-542` is the oracle |

⚠️ **The `pyproject.toml` line is separable and worth doing first.** Declaring the dependency that is
already load-bearing is a one-line honesty fix that needs none of this program and removes the live risk
from the published package. It is listed as PF-0 for that reason.

## 5 — Invariant classification

**Hard (violating one is a defect, not a design choice):**
- channel order == `[time, id] + identity_cols + features + static_channels` — the autoregressive feedback
  indexes it positionally (`volume_handler.py:162-171`).
- North-Up flip before the transpose, and geometry statics filled before the flip (I6, `:326-340`).
- `row_offset`/`col_offset` numbering from 1, matching pipeline-core (`config_hyperparameters.py:88-91`).
- a static channel is time-invariant (ADR-060 I3, `:312-320`).
- `_execute_derivations` ≡ `apply_blueprint` (`test_derivation_parity.py`).

**Intentionally changed by this program:**
- the input container: `pd.DataFrame` → `FeatureFrame`.
- the NaN policy on the input path: silent 0.0 fill upstream → loud rejection (§1.7).
- `apply_blueprint` on the input path: deleted, not ported (§1.4).

**Respect while changing:**
- the scaling precision (§1.6) — float64 transform, float32 only at the scatter.
- the uniqueness and finiteness checks — ported, never dropped (§1.2).
- the actuals seam (`prepare_actuals_df`) — F4; may keep a pandas shim, off the input path.
- `visual_diagnostics.biopsy_dataframe` — viz extra, never headless.

---

## 6 — Pre-flight checklist (**every box green before S1 captures the anchor**)

| id | gate | state 2026-10-19 |
|---|---|---|
| **PF-0** | pandas declared in `pyproject.toml` (the standalone honesty fix) | ✅ **DONE 2026-10-20** — `pandas>=1.5.3,<3.0`, on branch `fix/declare-pandas` |
| **PF-1** | ADR-072 reviewed by a human; ADR-071's ordering lock resolved (accept, or restate the lock in an accepted ADR) | ☐ **open — the only human gate**, both still in `proposed/` |
| **PF-2** | target model named and its source confirmed frame-capable | ✅ amber_kestrel, `source: "views-datafactory"` (`config_queryset.py:48-56`) |
| **PF-3** | `data_format: feature_frame` added to the target's descriptor | ☐ **open — views-models change ⇒ GH ISSUE, not a commit from here** |
| **PF-4** | an actual `FeatureFrame` for the target exists on disk, loadable by `load_frame_cache` | ☐ open — requires PF-3, then one fetch run |
| **PF-5** | the pandas path's NaN-fill count on the anchor partition, read from the log (§1.7) | ☐ open — zero expected; non-zero is a finding |
| **PF-6** | the source dtype at the zarr boundary (§1.6) — decides whether `atol=0` is reachable at all | ☐ open — **measure before S1**, a STOP if float64 |
| **PF-7** | import-purity + dependency-floor tests added (red is fine; they must exist) | ☐ open — §4 |
| **PF-8** | `row_col_from_priogrid` written with `dataloaders.py:537-542` as its oracle, test green | ☐ open — §1.5 |
| **PF-9** | 05 re-locked with the §1.6 precision constraint inside the intervention definition | ☐ open — else F1 fires falsely |
| **PF-10** | suite green, `ruff check` clean, on `development` at the anchor commit | ☐ open — run at S1 |

**PF-1 and PF-3 are the only two items that are not mine to close.** Everything else is mechanical.

---

## 7 — What this harness deliberately does not cover

- **Performance.** The migration is justified by correctness and honesty, not speed. No throughput claim is
  made or tested. (Standing rule: no cost or power figure without measuring one — and none is measured here.)
- **The three viewser models.** Excluded by the frame-capable-source gate (§1.8), not by preference.
- **`prepare_actuals_df`.** F4 territory; a pandas shim there is an acceptable outcome.
- **The column-name-as-role coupling** (C-173/C-174). Orthogonal, and 02's refusal stands.
- **Any scientific claim.** This is a refactor: success is *no behaviour change*.
