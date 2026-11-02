# ADR claim verification — ADRs 072–078

**Date:** 2026-11-02 · **Method:** three independent falsification audits, each run in a clean
context against the working tree, checking every factual claim against the code.

## What this document is for

ADRs state claims about the current system. Those claims must be true, but the `file:line`
citations that prove them **rot** — eight of ADR-072's fifty-three had already drifted when audited.

So the division of labour is:

| | holds | may contain line numbers? |
|---|---|---|
| **the ADR** | the claim | no — they rot, and the ADR outlives them |
| **this dossier** | the proof | **yes, by design** |

⇒ An ADR's Evidence pointer names this file. **This file must actually contain what the ADR says
it does** — a corollary violated four times before this dossier existed, with ADRs 074 and 075
citing a blast-radius report for a field inventory, two origin numberings, a recoverability
fraction and a memory sizing that it does not contain.

⚠️ Line numbers below were correct on 2026-11-02 and will drift. The **claim** is the durable part;
a drifted line here is a dossier maintenance task, not an ADR defect.

---

# 1. ADR-074 — an emitted artifact records what it is

| claim | verdict | source |
|---|---|---|
| prediction output carries only a month identifier and a unit identifier per row | ✅ CONFIRMED | `views-pipeline-core/.../prediction/prediction_frame_io.py:18-27` writes `y_pred.npy` + `np.savez(identifiers.npz, **frame.identifiers)`; `views_frames/index.py:81-84` returns `{time, unit}` only |
| the spatial level is not persisted | ✅ CONFIRMED | `prediction_frame_io.py:35-36` — *"the layout does not store it, so it is supplied here"*. Built with the level at `views_hydranet/utils/prediction_frame_assembler.py:177`, discarded on write |
| the trained-model sidecar does not record emit-time settings | ✅ CONFIRMED on the load-bearing half | `views_hydranet/train/train_model.py:75-140`. No `forecast_composition`, `gate_threshold`, `emit_family_core`, `emit_distribution`, `n_posterior_samples` |
| …it records "architecture and seeds" | ⚠️ **OVERSTATED** | it also records `n_head_samples`, which is consumed **at inference** — comment at `train_model.py:122-126`: *"K changes the sampled D×K cube at inference, so it must ride in the manifest"*. A manifest recording sample widths would duplicate this one value |
| the origin in a directory name is a sequence number, not the origin | ✅ CONFIRMED exactly | `views_hydranet/utils/inference_orchestrator.py:308` `for i, origin in enumerate(origins)` → `:332` `origin_sink(i, …)`; the real `origin` is never passed. Consumed as `f"origin_{origin_idx}"` at `model.py:1457,1463` |
| a separate diagnostic records the origin directly, so two artifacts of one run number origins differently | ✅ CONFIRMED | `hydranet_inference.py:1015-1019` writes `bodymean_origin{origin}.npz` with `origin=np.int64(origin)`. Corroborated at `adapter.py:170-171`, comment *"# Origin is the list index"* |
| the prediction-frame layer treats a month as an opaque integer and validates only structure | ✅ CONFIRMED | `views-frames/docs/CICs/SpatioTemporalIndex.md:30-31` — *"`time` is an opaque integer; epoch, range, and monotonicity are a producer concern (register C-11)"*. `views_frames/_validation.py:25-61` |
| ADR-003 forbids inferring semantics and requires declaration | ✅ CONFIRMED, and understated | ADR-003 Law 1 prohibits *"Magic default values (e.g. assuming a spatial resolution if not provided)"* — the exact case; Law 2 prohibits positional assumptions — the exact origin case; Law 3 *"Data never travels alone"* |
| views-frames anticipates an optional `origin` identifier as a MINOR addition | ✅ CONFIRMED to the sentence | `views-frames/docs/ADRs/013_metadata_and_identifier_model.md:29-30` — *"Identifiers are a fixed, required `{time, unit}` for v1. Any future per-row axis (`step`, `origin`, `scenario`) is added as an **optional** identifier (MINOR) — **never** as a required…"* |
| that layer's serialisation already writes a header with level and producer metadata, unused by the pipeline | ✅ CONFIRMED | `views_frames/io/npz.py:20-39` writes `header.json` with `{"level", "metadata"}`; `save_pf` does not call it. `views_frames/metadata.py` `FrameMetadata(model, run_type, timestamp, seed)`, frozen, all-optional |

## ⛔ Claims found FALSE — do not reintroduce

| claim as written | why it is false |
|---|---|
| *"ADR-032 fixes output to the scale of the input — counts in, counts out"* | ADR-032 contains **no scale rule**. It is a naming invariant that explicitly survives transformation: §2.2 *"This name remains constant even if the underlying data is logged or transformed."* The scale rule is **ADR-019** (FeatureScaler) and **ADR-039** (`inverse_transform_volume`, "Predict → Align → Wrap → **Invert** → Collapse"). Under ADR-003 Law 6 the configuration is the authority on a column's scale |
| *"that rule is unaffected by this one"* (of ADR-032) | ADR-032 **§3.2 mandates** that `row`, `col` and `c_id` *"MUST be carried losslessly to ensure geographic traceability."* The prediction bytes carry none of the three. ADR-032's identity requirement is **already unmet** |
| *"nothing asserts either premise"* (horizon contiguity) | the production evaluation path declares a step mapping from a declared origin — `views-pipeline-core/.../evaluation/stage.py:442-470`, *"the orchestrator is the sole authority on lead-times" (ADR-040)* — and `modules/validation/adapter.py:89-107` **raises** on any month outside it. The offset-from-earliest rule is the documented **legacy fallback** (`adapter.py:53,186-189`). The claim holds only for readers of the artifact on disk and for tools |

---

# 2. ADR-075 — emitting the body

## ⛔⛔ The premise that was false: the gate is already emitted

| | source |
|---|---|
| the gate cube is allocated | `hydranet_inference.py:1656` |
| filled with the raw, **uncomposed** classifier probability | `:1765-1767` |
| returned and concatenated onto the forecast cube | `:1805`; `inference_orchestrator.py:123-127` `np.concatenate([post_reg, post_cls], axis=-2)` |
| carried through Wrap → Invert → Collapse → Assemble with it | same path |
| written to disk per classification target | `prediction_frame_assembler.py:180-189` |
| **budgeted** by the memory preflight | `hydranet_inference.py:1645-1652` sizes both cubes |
| a declared classification target, scored against `by_*` ground truth | `hydranet_manager.py:350-351`; `views-evaluation/.../native_evaluator.py:251-256` |
| **sanctioned by ADR-032 §2.3** as one of four legitimate target states | `pred_by_sb_best` — *Binary Prediction* |
| already stated in our own report | `reports/2026-11-01_body_output_blast_radius/00_REPORT.md:17` |

⇒ **Only the body is unavailable.** The gate needs no decision: it is emitted, governed, and
correct. Its one live problem is **transport**, which is views-models' (issue #658 item M-2).

## The asymmetry that justifies the ADR

| | forecasts an observable? | ground truth? | can be a target? | recoverable from the composed output? |
|---|---|---|---|---|
| **gate** | yes — did violence occur | yes, `by_*` | ✅ already is | n/a |
| **body** | **no** | **none** | **never** | **only above threshold** |

⇒ The body is the only quantity with all of: unobservable, unscoreable, unrecoverable.

| claim | verdict | source |
|---|---|---|
| composition is applied before the array is written | ✅ CONFIRMED | `hydranet_inference.py:1752` `to_cube_samples(..., gate=…, composition=…)` |
| under `threshold_gate` the body is zeroed below threshold for every draw, untouched above | ✅ CONFIRMED | `distributions/composition.py:58-59` — a deterministic per-cell 0/1 mask |
| the composition field is read by the autoregressive feedback as well as emission | ✅ CONFIRMED | `_emit_magnitude` `:1056`, `_sample_feedback` `:1142`; emit at `:1752`; a check at `:519` |
| the existing body-mean dump is reachable only from a driver script | ✅ CONFIRMED | `inference_orchestrator.py:102,222,299`; `hydranet_manager.py:353` sets no diagnostic attribute |
| the measured above-threshold fraction | ✅ MEASURED HERE | **0.46%** of cells at or above τ=0.20 — 3,839 of 842,634 cell-months, 13 origins, h=1, from `views-models/models/amber_kestrel/.../gate_h1/`. ⚠️ This was **not** in the blast-radius report; it is recorded here |

## ⛔ Claims found FALSE

| claim as written | why it is false |
|---|---|
| *"the gate and the body [are not] declared as regression or classification targets"* | the gate **is** a declared classification target — `hydranet_manager.py:350-351` — and is scored. True of the body only |
| *"a second inference pass yields a different set of dropout draws"* | dropout is **seeded**. `hydranet_manager.py:319-321` locks all RNG once per run; `hydranet_inference.py:1719-1721` seeds the family generator from `torch_seed`; `distributions/sampling.py:99-101` derives a deterministic per-(pass, step) sub-seed. Two runs at one seed give **identical body draws at h=1**. They diverge at h≥2 because composition changes the fed-back field — the conclusion holds, the reason did not |
| *"several independent guards refuse it, correctly"* — as the whole story | two guards do refuse (`model.py:474-486`; `native_evaluator.py:251-256`), but the report's own conclusion is the opposite emphasis: the serving API *"discovers targets by prefix with no allow-list"* and its documented fail-loud gate *"does not discriminate a body from a forecast"* — measured, `series_of()` returns the same series for every candidate body name **and for the gate** |
| the green test as stated unconditionally | body × gate-at-threshold reproduces the forecast under **`threshold_gate` only**. `soft_gate` masks per draw (`composition.py:56`); `self_zeroed` makes the body *be* the forecast (`:53-54`) |

---

# 3. ADR-076 — the provider and the parity standard

## ⛔⛔ The migration is complete. There is no gap.

**17 HydraNet models, identified by declared algorithm in `config_meta.py`. 17/17 declare
`"source": "views-datafactory"`. Zero declare viewser.** Last three migrated in `9302a0c4`
(2026-08-08). No HydraNet config imports `viewser` or builds a `Queryset`.

⚠️ **The error this corrects:** an earlier census counted every config with a `REGION` key — that is
every datafactory *consumer*, not the HydraNet fleet — and reported 35 datafactory / 9 viewser. The
nine were not HydraNets. The same error wrongly flagged `light_strider`, a `ConflictologyModel`.
⛔ **views-models#658 carries the wrong numbers and must be corrected.**

| claim | verdict | source |
|---|---|---|
| the loader dispatches on what a model's configuration declares | ✅ CONFIRMED | `views_pipeline_core/data/model_path.py:851-868` → `modules/dataloaders/dataloaders.py:119-155` |
| the point of change is one function in the model's own configuration | ✅ CONFIRMED | `config_queryset.generate()`; supported by **views-pipeline-core ADR-015** (model-specific input querysets) |
| the providers differ by deliberate preprocessing | ✅ CONFIRMED | KGI handling and GED vintage revisions — `reports/2026-07-28_datafactory_migration_dossier/02_design.md:31-32` |
| …by "filtering" imprecisely-located events | ⛔ **FALSE** | the datafactory **reassigns** them: *"imprecisely-located events placed on a different (usually neighbour) gid … per-month totals largely conserved by the reassignment"* (`02_design.md:31`). viewser's `_nokgi` *"left KGI unhandled (legacy stub)"* (`07_experiment_log.md:60`). **Neither provider filters** |
| the platform has two providers | ⛔ **FALSE — there are three** | `views_pipeline_core/data/constants.py:10` `CACHE_SOURCES = frozenset({"viewser", "datafactory", "synthetic"})`; `modules/dataloaders/synthetic.py:5` calls synthetic *"a first-class data source alongside VIEWSER and datafactory"* |
| viewser's non-conflict covariates are not trusted | ⚠️ **a recorded judgement, not a measurement** | `docs/ADRs/archive/071_…:36-38` — *"the maintainer does not trust viewser beyond the conflict targets."* Nothing in `reports/` measures it |
| a provider is readable from configuration alone | ⚠️ OVERSTATED | reading it **executes** `generate()` and type-sniffs the result, and can raise `ImportError` for a missing client (`model_path.py:859-865`). Not a static read |

---

# 4. ADR-077 — global land coverage

| claim | verdict | source |
|---|---|---|
| HydraNet coverage is global land | ✅ CONFIRMED | **17/17 declare the exact string `REGION = "land"`.** Zero `land_gaul`, zero `africa_me_legacy`, zero `global` |
| the change was fleet-wide on 2026-09-19 | ✅ CONFIRMED for HydraNet | `455b9674` (2026-09-19 23:08) flipped exactly the eight HydraNets that existed then; the other nine were created later already at `land` |
| the deliverable's coverage is a subset of global land | ✅ CONFIRMED | `candid_condor/configs/config_queryset.py:22-24` — *"64,742 land cells that GAUL gives a country to — `land_gaul`, NOT `land` (64,818) … The 76-cell difference is small islands."* Every HydraNet config carries *"The FAO delivery cuts land_gaul from this"* |
| ADR-076 is a genuine prerequisite, not circular | ✅ CONFIRMED | `REGION` is a key of the **datafactory descriptor only**; a viewser `Queryset` has no such declaration, so coverage cannot be stated at all on the viewser path |
| `land_gaul` is a HydraNet coverage needing accommodation | ⛔ **FALSE** | `land_gaul` is declared by **16 non-HydraNet** models (condor/sparrow pairs) plus two postprocessors. No HydraNet declares it |

## ⛔ Claims found FALSE

| claim as written | why |
|---|---|
| *"removes the only architectural text asserting the superseded coverage"* | 071 was already moved to `archive/` and bannered in `94466f3` (2026-11-01), **before** 077 was written, and that banner already states the coverage was abandoned. 077 removes nothing |
| *"the only"* | `proposed/072_frame_native_input_ingestion.md:111` still carries a worked example on the Africa crop, in a live document |

⚠️ **Open, for views-models:** the roster test pinning `REGION == "land"`
(`views-models/tests/test_roster_conformance.py:73-82`) covers only the original **eight**. **Nine
of seventeen** HydraNet coverage declarations are unpinned.

---

# 5. ADR-078 — origin and horizon selection

## ⛔⛔ Origin selection is not available

| finding | source |
|---|---|
| the pipeline requires **exactly 13 prediction sequences** and raises otherwise | `model.py:2034-2043` — *"Contract enforcement: evaluation must return exactly MAX_SHIFT_COUNT + 1 sequences"* → `ValueError("Pre-flight sequence count check FAILED…")`. `MAX_SHIFT_COUNT = 12` at `core_config_sniffer.py:243` |
| ⇒ an origin subset is refused on evaluation runs — the expensive case | |
| `fixed_last_origin` is **dead**, not test-only | `utils_orchestration.py:8`. A repo-wide grep returns **four hits, all inside that one function** (`:8,20,25,26`). The production caller `hydranet_manager.py:346` omits it; no test passes it |
| …and it cannot express an emitted subset | it fixes the **last** origin and rolls backwards `num_windows` times (`:25-42`), so it changes what is **computed** |

| claim | verdict | source |
|---|---|---|
| nothing subsets targets, origins or horizons | ✅ CONFIRMED | both orchestrator paths iterate the full lists |
| the only per-origin gate is a visualisation one, first origin only | ✅ CONFIRMED | `inference_orchestrator.py:169-172` |
| only the **length** of the step list is read **inside views-hydranet** | ✅ CONFIRMED | `volume_sampler.py:60-66`; `config_initializer.py:639-642`; `reproducibility_gate.py:35`. No contiguity check, no value check |
| every current configuration declares a contiguous range from one | ✅ CONFIRMED | no live configuration is affected by a contiguity rule |

## ⛔ Claims found FALSE

| claim as written | why |
|---|---|
| *"the values are read nowhere"* | true inside views-hydranet, **false across the boundary**. `model.py:1953-1958` and `evaluation/stage.py:466-470` build the month→step mapping **from the values**: `{base_origin + i + s: s for s in steps}`. `model.py:2012-2109` raises on a rogue month, noting *"configs['steps'] is the sole source of truth in views-pipeline-core"* |
| *"the mismatch is reported nowhere … the run completes"* | a sparse list **raises** on any evaluation run. The trap survives on the **forecasting path only** (`model.py:2030-2032` — the check *"is only called from _execute_model_evaluation (never the forecasting path)"*). The decision stands; this justification does not |
| *"an origin's forecast depends on the state carried forward to it"* | **no state is carried between origins.** `hydranet_inference.py:1243-1248` re-initialises the hidden state on every `predict()`; each origin re-digests its own history. The real constraint is the **global seeded dropout stream** — `hydranet_manager.py:319-321` locks entropy once per run, so masks for origin *n* depend on how many forwards preceded it |
| *"labels them as the horizons named"* | nothing attaches a step label to a row in either repository. Months come from `slice_time(origin+1, …)` (`inference_orchestrator.py:150`); the index carries `month_id`/`unit` only. The declared values are **ignored** downstream of the length check |

---

# 6. ADR-072 — frame-native input ingestion

**Citations: ~49 of 53 checked. 41 OK, 8 drifted or gone.**

| claim | verdict | source |
|---|---|---|
| the eleven `volume_handler` citations, the four `feature_frame_path` body citations, the views-frames validation and float32 contracts, the pipeline dispatch seam, ADR-019 Invariants 3/4 and Role 4, ADR-060 I1/I2/I5/I6/I7 | ✅ all CONFIRMED | the technical core holds under checking |
| the four verbatim production code blocks in §5 | ✅ all four OK | `volume_handler.py:186-188, :236, :322-325, :339-340` |
| one genuinely pandas-idiomatic line | ✅ CONFIRMED | `volume_handler.py:312` — the `groupby`. And every config declaring `static_channels` declares `[]` (10/10), so that guard is dead code today |

## ⛔ Claims found FALSE

| claim as written | why |
|---|---|
| *"pandas is not listed [in pyproject.toml]"* | **declared since 2026-10-20** — `pandas>=1.5.3,<3.0` at `pyproject.toml:67`, under a comment at `:64`: *"⚠️ This DECLARES the dependency; it does not remove it. Removal is proposed ADR-072."* The cited anchor `:8-13` is license/readme; `dependencies` begins at `:33`. ⇒ **One of the ADR's two stated motives no longer exists**, and §8 still claims closing that gap as a benefit |
| *"`apply_blueprint` … its one live consumer is `HydranetManager.prepare_actuals_df`"* | `DataFetcher.standardize_raw_df` ends `return DataFetcher.apply_blueprint(df_out, config)` — `data_fetcher.py:151` — and is called on the **live input path** at `hydranet_manager.py:130`. It runs on every training, eval and forecast run |
| *"**completeness:** identifiers are required to be integer dtype … so there is no such thing as a silently-missing `(time, unit)` label"* | integer dtype rules out a **null label on a row that exists**; it says nothing about a **row that is absent**. The ADR's own §7 Beige rule requires missing-month coverage be tested separately. The invariant labelled "completeness" does not guarantee completeness |
| §6's *"diagnostic/audit-only pandas … none run on a headless production path"*, applied to `.duplicated()` | `.duplicated()` is in `_check_spatiotemporal_uniqueness` (`data_sniffer.py:285-286`), called from `sniff_ingestion` (`:54`) — which §6 lists as **in scope** and which runs on every run (`hydranet_manager.py:133`) |
| §4.2/§6's Inbound role | contradicted by the ADR's own Amendment 1: *"hydranet calls neither API … `get_feature_frame` is the **manager's** call, not the engine's."* A1.1 is correct (`model.py:1269-1277`, `:1010-1024`, `frame_cache.py:141-160`) but §6 was never amended, so the scope statement a reviewer reads is the superseded one |
| the §4.1 worked example | built on `row_offset=87, col_offset=310, 180×180` — the purged Africa crop. The live grid is `1, 1, 360, 720`, and 071's archive banner names that exact crop as data not to act on |

## Drifted anchors

| cited | actual |
|---|---|
| `pyproject.toml:8-13` | GONE — `dependencies` at `:33` |
| `dataloaders.py:1112` (×4) | GONE — file is 1010 lines; `get_feature_frame` at `:960` |
| `index.py:231-240` | `has_unique_rows` at `:245-254` |
| `index.py:28-37` | `:25-38` |
| `index.py:104-109` | `:104-105` is the integer-dtype message; the duplicate rejection is `:108-113` |
| `dataloaders.py:577` | `ensure_float64` is `:576`; `:577` is blank |
| `feature_frame_path.py:20-22` | the fillna refusal is `:19-21` |
| `[ADR-071](071_…)` | file moved to `archive/` — **fixed 2026-11-02** |

---

# 7. ADR-073 — emitted vs trained distribution family

| claim | verdict | source |
|---|---|---|
| the capability exists and is default-off | ✅ CONFIRMED | `config_initializer.py:250` `emit_distribution: str | None = Field(default=None)`; seam at `hydranet_inference.py:278-280` |
| three validation rules as described | ✅ CONFIRMED, plus a fourth unmentioned | `config_initializer.py:1261, 1268, 1286-1296`, and `:1277-1285` (trained family unregistered). Re-checked at `hydranet_inference.py:561-586` |
| only one family pair is admissible | ✅ CONFIRMED | `FAMILY_N_PARAMS = {"nb": 2, "zinb": 3, "mixture_nb": 5, "truncated_nb": 2}` (`registry.py:67-72`) |
| the fed-back draw resolves from the **emitted** family | ✅ CONFIRMED | `hydranet_inference.py:1109` `fam = self._family`, under the decision comment at `:1099-1108`. ⇒ §5's "open question" was already answered by a default |

## ⛔ Claims found FALSE

| claim as written | why |
|---|---|
| *"Bin count swept … so the optimum is out-of-sample and not an overfit"* | **withdrawn 2026-10-27.** `reports/RESULTS_LEDGER.md:425` carries it struck through: the turnover is the plug-in estimator's bias, which grows as equal-count bins shrink. Under the fair estimator the curve never turns over. This was the only argument offered that the decisive measurement is not an overfit |
| *"−17.6% against the best parametric cell"* | arithmetic: 0.039327 / 0.046023 − 1 = **−14.55%**. −17.6% holds only against a cell that is not the best. The independent reproduction records −14.9% plug-in / −15.3% fair (`RESULTS_LEDGER.md:426`) |
| *"scores 0.039327 … with the same m=16 plug-in estimator"* | `reports/PREDICTION_LEDGER.md:307` (W-98): *"the number I anchored on has never been produced by any implementation."* `:386` (R-43): the implied *m* is **155**, not 16. `:385` (R-42): the published trio fails its own identity |
| *"ADR-068 (`emit_family_core`) is not subsumed and must not be folded in"* | two faults. **ADR-069 (Active) §79 decided to retire it** — *"Retire `emit_family_core` — the inference flag for the KILLED gated_ZINBcore arm (dead code)"*. And `emit_family_core` appears **nowhere in ADR-068**, which is a naming-convention ADR |

⚠️ **Code/ADR drift, independent of 073:** ADR-069 decided to retire `emit_family_core`; the flag is
still live at `config_initializer.py:240`, `hydranet_inference.py:1046`, `:1117`. **Owed an issue.**

## Drifted anchors

`hydranet_inference.py:270` (the seam) → `:278-280`. `:1052` (the fed-back draw) → `:1109`.
`:1054-1058` (the C-234 quotation) → `:1115-1116`. `OUTCOME.md:77-84` → `:85-90`.

---

# 8. Cross-ADR consistency

| pair | verdict |
|---|---|
| **074 vs 078** | ⛔ **CONTRADICTION.** 074 §2.3 fails a run on non-contiguous months; 078's headline case is emitting the first and last horizon. In the order 078 specifies, its use case is unimplementable |
| **075 vs 069** | ✅ consistent, and protective. 069 makes composition an in-model emit-time axis applied to both the cube and the feedback path; 075's demand for a separate field is what keeps 069's semantics intact |
| **075 vs 073** | ⚠️ orthogonal but uncited. 073 splits *which family* is emitted, 075 *which quantities*. No contradiction — but 073's own verdict records an emit-time axis silently reaching the feedback path, which is 075's named hazard class, already registered. 075 does not reference 073 |
| **073 vs 069** | ⛔ **CONTRADICTION** on `emit_family_core`, and a one-directional amendment 069 does not record |
| **076 vs 077** | ✅ a genuine prerequisite, not circular — `REGION` exists only on the datafactory descriptor |
| **072 vs 077** | ⛔ 072's worked example asserts the coverage 077 retires |
| **074 vs 032** | ⛔ 074 declared 032 intact; 032's §3.2 identity columns are already absent from the bytes |
| **all vs 032's scale rule** | ✅ none contradicts it. The 032 problems are the three-prefix law and the identity columns |

## Dependency graph

074 ← 075, 078 (reciprocal ✅) · 076 ← 077 (reciprocal ✅ after correction) · acyclic ✅.
072 and 073 declare no relations; 073 is adjacent to 075 and uncited by it.

---

# 9. Findings index

**19 critical.** Each resolves to a correction, an issue, or a recorded decision — none is dropped.

| # | ADR | finding | disposition |
|---|---|---|---|
| 1 | 075 | the gate is already emitted | 075 → body only |
| 2 | 075 | the gate is a declared target | claim deleted |
| 3 | 075 | dropout is seeded | reason corrected |
| 4 | 076 | migration gap is zero | 076 → the parity standard; **#658 corrected** |
| 5 | 074 | ADR-032 misread, twice | corrected |
| 6 | 074 | horizon claim overstated | corrected |
| 7 | 074/078 | contiguity contradiction | 074 §2.3 dropped |
| 8 | 078 | step values read across the boundary | justification corrected |
| 9 | 078 | origin selection refused by the 13-sequence contract | origin selection dropped |
| 10 | 078 | no state carried between origins | reason corrected |
| 11 | 078 | `fixed_last_origin` is dead | claim corrected, seam not reused |
| 12 | 072 | pandas now declared — a motive is gone | motive removed |
| 13 | 072 | `apply_blueprint` runs on the input path | corrected |
| 14 | 072 | "completeness" guarantees something else | corrected |
| 15 | 072 | §4.2/§6 incompatible scope | §6 amended |
| 16 | 072 | `.duplicated()` is in-scope | corrected |
| 17 | 073 | anti-overfit argument withdrawn | sentence deleted |
| 18 | 073 | −17.6% is wrong | corrected to −14.55% |
| 19 | 073 | asserts a retired flag is live | corrected; **069 drift → issue** |

**Owed as issues:** #658 correction · the `emit_family_core` code/ADR drift · nine unpinned
coverage declarations · the views-frames `origin` field · the pipeline bypassing the frames header.
