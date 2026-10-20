# Frame-Native Input Ingestion — migration dossier (2026-08-13, completed 2026-10-19)

## Purpose
Adopt `views-frames` `FeatureFrame` as the **input** data contract for HydraNet, removing pandas from the
data-ingestion path so views-frames is the native substrate **end-to-end** (input *and* output). This is
the execution plan for **proposed ADR-072**; it is the deferred "P4 / S9" pandas-removal step named in the
datafactory-migration dossier and ADR-071.

> **Status 2026-10-19: READY TO EXECUTE, pending two gates that are not the agent's to close.**
> Every claim in this dossier has been re-verified against the code (`03_harness.md §1`), which produced
> **six corrections to these documents** — two shrink the work, two grow it, one changes the
> pre-registration, one is a naming fix — plus **one finding about ADR-072 itself** (three of its citations
> no longer resolve). ⚠️ **ADR-072 was already right about two of the six**: it corrects the API naming at
> `:89-91` and states the uniqueness obligation in full at `:227-234`. The dossier was the stale half, and in
> one place it contradicted its own ADR. Five findings are new against the ADR as well and are amended into
> it (**ADR-072 Amendment 1**). The harness (03) and the experiment log (07) now exist, and 05 is amended
> with the precision policy the original plan omitted. **Still open: PF-1** (a human must review ADR-072 and resolve the
> ADR-071 ordering lock) and **PF-3** (one key added to the target model's descriptor in views-models — a
> GitHub issue, not a commit from here). Nothing is implemented; no production code is changed by this
> dossier.

## Why now (verified 2026-08-13, re-verified 2026-10-19)
1. **The output side is already pandas-free** (ADR-047 / `PredictionFrame` / `InferenceOrchestrator`). Only
   the input half remains — this closes the asymmetry. ✅ holds.
2. **The input-side substrate already exists**: `FeatureFrame` (numpy `(N,F,S)`), the datafactory grid
   adapters, and a built+tested frame path in views-pipeline-core. ✅ holds — and **more of it is built than
   this dossier knew**: pipeline-core #290 ships the `data_format: feature_frame` dispatch, the fetch, and
   `_get_cached_frame_path()` for engine subclasses (`03_harness.md §1.3`).
3. **The datafactory is not the bottleneck**: `output_format="feature_frame"` is the default and never
   touches pandas. ✅ holds (`dataset.py:124`).
4. ~~**The one non-trivial transform already has a proven numpy twin**~~ — ⛔ **superseded by something
   better**: the numpy twin is *already the only path feeding the volume*. Nothing has to move; the pandas
   half is deleted (`03_harness.md §1.4`).
5. ~~**Latent fragility to close**: pandas is an *undeclared but load-bearing* dependency (7 modules,
   absent from `pyproject.toml`).~~ ✅ **CLOSED 2026-10-20 — `PF-0` is DONE.** `pandas>=1.5.3,<3.0` is
   now declared, with the seven import sites and the range's provenance in the comment. It held when
   written and it shipped undeclared in **0.1.0 through 0.1.3**; it does not hold any more. ⚠️ **This
   declares the dependency, it does not remove it** — removal is still the whole point of this program.

## Governing gate (non-negotiable)
**Byte-identical parity**: for the same v2 inputs, the frame-native path must produce a model input volume
tensor — and therefore predictions — byte-identical to the current pandas path at `atol=0`. ⛔ Reachable
**only** under the precision policy pre-registered in `05` Amendment A1.1 (float64 transform, float32 only
at the scatter-write); without it the gate fails on a last-bit artefact and F1 fires with the wrong
diagnosis. If parity fails *with* that policy in force, **STOP** — the migration is not free. A1.4
pre-excludes every other "reconciliation".

## Sequencing lock (do not entangle)
Runs **after** the viewser→datafactory provider swap (ADR-071), as a **separate** step. ⚠️ ADR-071 is
itself still `proposed/`; the swap has landed *in practice* for the datafactory models, and
`fetch_feature_frame` refuses outright for a viewser source — so the data precondition is met and the
governance one is not (`03_harness.md §1.8`). Not to be bundled with the column-name-as-role refactor
(C-173/C-174, orthogonal).

## Document index
| # | Doc | Status |
|---|-----|--------|
| 00 | README (this spine) | living |
| 01 | literature | **N/A** — this is a refactor; its hypothesis is *equivalence*, not a scientific claim. The prior art is ADR-047, ADR-071 and the datafactory dossier, all cited inline. No stub file. |
| 02 | design (the boundary swap + roles/handshake) | draft **+ errata 2026-10-19** (read the errata first) |
| 03 | **harness** (verification pass, handshake spec, pre-flight checklist) | **written 2026-10-19** |
| 04 | roadmap (staged, parity-gated) | draft **+ revision 1 2026-10-19** |
| 05 | analysis_plan (pre-registration) | draft **+ amendment 1 2026-10-19**; locks when PF-1 and PF-3 close |
| 06 | glossary | **N/A by standing rule** — there is exactly one glossary, `reports/GLOSSARY.md`. A missing term is edited into it, never forked here. |
| 07 | **experiment log** | **opened 2026-10-19** with E0 (the verification pass) and ten named slots |

## The pandas surface being removed (verified map, 2026-10-19)
| Module | Pandas role | Disposition |
|---|---|---|
| `utils/data_fetcher.py:10` | parquet→DataFrame (`read_dataframe`, `:60`), MultiIndex contract, `apply_blueprint` | **replace** — `load_frame_cache(self._get_cached_frame_path())`; MultiIndex contract → ported numpy gates; **`apply_blueprint` deleted from the input path** |
| `utils/volume_handler.py:9` | `from_df` column access + one `groupby` (`:312`) | **add `from_feature_frame`** (18-step spec in `03_harness.md §2`; 13 steps unchanged). The `groupby` is dead on the shipped fleet (`static_channels: []` everywhere) — port for correctness |
| `utils/feature_scaler.py:10` | `fit_transform`/`inverse_transform` on a DataFrame | **rewrite** over `FeatureFrame.values` — ⛔ **with the float64 upcast** (A1.1). `inverse_transform_volume` is already numpy |
| `utils/data_sniffer.py:11` | hot-path gates + audit-only `sniff_pure_state_parity` | **rewrite gates numpy-native** — ⛔ uniqueness and finiteness are **ported, not inherited** (the frame type supplies neither). Audit method may stay pandas behind the diagnostic extra |
| `manager/hydranet_manager.py:14` | `prepare_actuals_df` (`:192-199`) + a boolean filter | **thin** — the actuals seam is F4; a pandas shim there is an acceptable outcome, off the input path |
| `utils/utils_logging.py:10` | print/report over df shape | **trivial rewrite** (peripheral) |
| `utils/visual_diagnostics.py:11` | `biopsy_dataframe` only (viz, null-object-gated) | **leave** under the `viz` extra — never on the hot path |

## The only genuinely new code
`views_hydranet/utils/grid_geometry.py::row_col_from_priogrid` — `row = (pgid-1)//720 + 1`,
`col = (pgid-1)%720 + 1`, oracle `views-pipeline-core/.../dataloaders.py:537-542`. Needed **twice**: as
scatter indices and as channel values (`identity_cols` lists `row`/`col`). Everything else is re-sourcing.
(`03_harness.md §1.5`.)

## Current state & next actions
- [x] Cross-repo feasibility verified (views-frames / datafactory / pipeline-core), 2026-08-13.
- [x] Proposed **ADR-072** drafted (`docs/ADRs/proposed/072_frame_native_input_ingestion.md`).
- [x] Dossier scaffolded (00/02/04/05), 2026-08-13.
- [x] **Every claim re-verified against the code; six corrections + one ADR finding recorded** (`03_harness.md §1`, `07` E0), 2026-10-19.
- [x] **ADR-072 amended** with the five findings that are new against it, incl. the precision policy and the row/col decoder, 2026-10-19.
- [x] **Harness written**, with the 18-step handshake spec and an 11-item pre-flight checklist (`03`), 2026-10-19.
- [x] **Experiment log opened** (`07`), 2026-10-19.
- [x] **05 amended** with the precision policy, so the gate tests the container and not the dtype, 2026-10-19.
- [x] ✅ **PF-0 — pandas declared in `pyproject.toml`**, 2026-10-20: `pandas>=1.5.3,<3.0`, range matched to views-pipeline-core's documented floor and ceiling. Done alone, as planned.
- [ ] **PF-1** — ⛔ maintainer review of ADR-072 (now + Amendment 1) and the ADR-071 ordering lock. **The human gate.** ⚠️ Re-pin the three drifted citations first (`03_harness.md §1.9` N7) — an ADR a reviewer cannot check by following its own line numbers is not reviewable.
- [ ] **PF-3** — ⛔ `"data_format": "feature_frame"` on the target descriptor. **views-models ⇒ GitHub issue.**
- [ ] **PF-6 / PF-5** — the two cheap pre-S1 measurements that can stop the program (source dtype; NaN-fill count).
- [ ] Then S1 onward per `04` revision 1; every stage writes its entry in `07` before being called done.

## Conventions
Numbered dated docs; `00_README` living. git-tracked via `git add -f` (reports/ gitignored). Vocabulary from
`reports/GLOSSARY.md`, singular. Cross-repo IDs as `repo#id`. On close → `reports/archived/`.
