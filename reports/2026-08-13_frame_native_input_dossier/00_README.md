# Frame-Native Input Ingestion — migration dossier (2026-08-13)

## Purpose
Adopt `views-frames` `FeatureFrame` as the **input** data contract for HydraNet, removing pandas from the
data-ingestion path so views-frames is the native substrate **end-to-end** (input *and* output). This is
the execution plan for **proposed ADR-072**; it is the deferred "P4 / S9" pandas-removal step named in the
datafactory-migration dossier and ADR-071.

> **Status: DRAFT proposal for maintainer review.** Drafted by an AI agent. Nothing here is implemented;
> no code is changed by this dossier. Read ADR-072 first, then this plan, and verify the claims.

## Why now (verified 2026-08-13)
1. **The output side is already pandas-free** (ADR-047 / `PredictionFrame` / `InferenceOrchestrator`). Only
   the input half remains — this closes the asymmetry.
2. **The input-side substrate already exists**: `FeatureFrame` (numpy `(N,F,S)`) in views-frames, plus the
   grid adapters in views-datafactory and a built+tested `feature_frame_path` in views-pipeline-core
   ("pandas never touched"). This is *connect-what-exists*, not build-from-scratch.
3. **The datafactory is not the bottleneck**: `load_dataset(output_format="feature_frame")` is the default
   and never touches pandas (zarr→xarray→numpy).
4. **The one non-trivial transform already has a proven numpy twin**: `tests/test_derivation_parity.py`
   pins `VolumeHandler._execute_derivations` (numpy) ≡ `DataFetcher.apply_blueprint` (pandas).
5. **Latent fragility to close**: pandas is an *undeclared but load-bearing* dependency here (imported by
   7 modules; not in `pyproject.toml`).

## Governing gate (non-negotiable)
**Byte-identical parity**: for the same v2 inputs, the frame-native path must produce a model input volume
tensor — and therefore predictions — byte-identical (or within a pre-registered float tolerance) to the
current pandas path. This mirrors ADR-071's clean-cut discipline and P4's "GATE: byte-identical vs
baseline". If parity fails and cannot be reconciled, **STOP** — the migration is not free.

## Sequencing lock (do not entangle)
This runs **after** the viewser→datafactory provider swap (ADR-071), as a **separate** step — exactly the
lock the datafactory dossier records ("data-swap + parity FIRST; pandas-removal LAST"). It also must not be
bundled with the column-name-as-role refactor (C-173/C-174, orthogonal).

## Document index
| # | Doc | Status |
|---|-----|--------|
| 00 | README (this spine) | living |
| 02 | design (the boundary swap + roles/handshake) | draft |
| 04 | roadmap (staged plan S0–S6, parity-gated) | draft |
| 05 | analysis_plan (pre-registration: the byte-identical gate + falsifiers) | draft |
| 01/03/06/07 | literature / harness / glossary / experiment-log | to fill at `init` if the program is greenlit |

## The pandas surface being removed (verified map)
| Module | Pandas role | Disposition |
|---|---|---|
| `utils/data_fetcher.py` | parquet→DataFrame, MultiIndex contract, `apply_blueprint` derivations | **replace** — inbound via `feature_frame_path`; derivations via numpy `_execute_derivations`; contract via `SpatioTemporalIndex` |
| `utils/volume_handler.py` | `from_df` column access + 1 `groupby` (static-channel time-invariance) | **replace** — add `from_feature_frame`; the tensor build is *already* numpy scatter-write |
| `utils/feature_scaler.py` | `fit_transform`/`inverse_transform` on DataFrame | **rewrite** to scale `FeatureFrame.values` (numpy); `inverse_transform_volume` is *already* numpy |
| `utils/data_sniffer.py` | hot-path gates + audit-only `sniff_pure_state_parity` | **rewrite gates** numpy-native; audit method may stay pandas (diagnostic) or gate off |
| `manager/hydranet_manager.py` | `prepare_actuals_df` passthrough + 1 boolean filter | **thin** — follows from the above |
| `utils/utils_logging.py` | print/report over df shape | **gate/trivial-rewrite** (peripheral) |
| `utils/visual_diagnostics.py` | `biopsy_dataframe` only (viz, null-object-gated) | **optional-extra** — leave under `viz`, never on the hot path |

## Current state & next actions
- [x] Cross-repo feasibility verified (views-frames / datafactory / pipeline-core), 2026-08-13.
- [x] Proposed **ADR-072** drafted (`docs/ADRs/proposed/072_frame_native_input_ingestion.md`).
- [x] This dossier scaffolded (00/02/04/05).
- [ ] **Maintainer review of ADR-072 + this plan** (the immediate gate — human, not agent).
- [ ] Greenlight → `rnd-dossier init` to fill 01/03/06/07 + pre-register 05 formally before any code.
- [ ] Dependency: the datafactory provider swap (ADR-071) must have landed for the target model(s).

## Conventions
Numbered dated docs; `00_README` living. git-tracked via `git add -f` (reports/ gitignored). Follows
`reports/GLOSSARY.md`. On close → `reports/archived/`.
