# 04 — Roadmap (staged, parity-gated)

Each stage is small, reviewable, and TDD-first (ADR-005). The **byte-identical parity gate** (05) sits
between "build the frame-native path" and "retire the pandas path" — nothing pandas is deleted until parity
is proven. Precondition for the whole program: the datafactory provider swap (ADR-071) has landed for the
target model(s), so a v2 `FeatureFrame` input actually exists.

## S0 — Gate on governance (human)
- Maintainer reviews **ADR-072** + this dossier against ADR-000's eight criteria.
- Decide: accept ADR-072 (Proposed→Accepted), or send back. **No code until accepted.**
- `rnd-dossier init` to fill 01/03/06/07 and formally lock 05.

## S1 — Parity anchor BEFORE (characterization)
- Freeze one v2 origin's inputs. Capture the **current** pandas-path model input volume tensor
  (`from_df(...).volume`) and one origin's predictions as the golden reference (`.npy`, checksummed).
- Reuse the existing parity oracles: `test_derivation_parity`, `test_volume_handler_hard_gates`,
  `test_static_channel_seam`, `test_data_backed_static_channel`.

## S2 — Red tests for the frame-native path (TDD, fail first)
- `test_volume_handler_from_feature_frame.py`: `from_feature_frame(ff).volume == from_df(df).volume`
  (byte-identical / pre-registered atol) on the S1 anchor.
- Beige/Red: missing month, duplicated `(time,unit)`, off-grid cell, NaN feature, wrong `feature_names`
  → named exception at the handshake (not silent).
- `FeatureScaler` numpy path parity vs the pandas `fit_transform` on the same features.

## S3 — Implement the frame-native path (behind the gate)
- Add `VolumeHandler.from_feature_frame` (mirror `from_df`; source coords from `SpatioTemporalIndex`,
  values from `ff.values`); replace the static-channel `groupby` with the numpy segmented reduction.
- Wire `DataFetcher` inbound to `feature_frame_path.get_feature_frame`; route derivations through
  `_execute_derivations`.
- `FeatureScaler` scales `FeatureFrame.values`; numpy-native `DataSniffer` hot-path gates.
- `from_df` kept as a deprecated shim. Both paths live; default still selectable by a flag for the anchor.

## S4 — Parity anchor AFTER (the GATE)
- Run the frame-native path on the S1 origin → assert the volume + predictions match the golden reference
  byte-for-byte (or ≤ atol). Full suite green; determinism gate; ruff.
- **If parity fails and cannot be reconciled → STOP and report.** Do not widen tolerance to pass.

## S5 — Flip default + retire pandas on the hot path
- Make frame-native the default input path. Remove pandas imports from the load-bearing modules
  (`data_fetcher`, `volume_handler`, `feature_scaler`, `data_sniffer` hot path).
- Keep pandas only behind the `viz` optional extra (`visual_diagnostics.biopsy_dataframe`) and any
  audit-only method deliberately retained.
- Update `pyproject.toml`: drop the transitive-pandas reliance from the hot path; declare what remains
  honestly. Update the CICs (`DataFetcher`, `DataSniffer`, `FeatureScaler`, `VolumeHandler`) in the same PR
  (ADR-000/007). Amend ADR-047's "input out-of-scope" clause to point at ADR-072.

## S6 — Verify + close
- `import views_hydranet` pulls no pandas on the training/eval path (an import-purity test, like
  pipeline-core's / datafactory's); a smoke train+eval on the target model is green and byte-identical to
  the pre-flip baseline.
- Promote ADR-072 (Accepted); archive this dossier; register any residual (e.g. the deferred C-173/C-174,
  the audit-only pandas island) with `repo#id` discipline.

## Dependency & ordering summary
```
ADR-071 provider swap (landed) ──► S0 review ──► S1 anchor ──► S2 red ──► S3 build
                                                                             │
                                              (byte-identical GATE) ◄── S4 anchor-AFTER
                                                                             │
                                                              S5 flip+retire ──► S6 verify+close
```
Out-of-scope throughout: the provider swap, column-name-as-role coupling, any output/scoring code.

---

# Revision 1 — 2026-10-19 — the staging after verification

`03_harness.md §1` changed the shape of the work: the inbound integration is already shipped upstream and
the derivation migration is already done, while two gates have to be ported that the design thought were
free, and one new decoder has to be written. The S0–S6 spine above stands; this revision replaces its
contents.

## S0 — governance + the separable honesty fix
- ✅ **PF-0 — DONE 2026-10-20.** `pandas>=1.5.3,<3.0` declared. **Was independent of everything else** — it removed a live
  risk from a published package (the #381 class) and needs no greenlight. Do it first, alone.
- **PF-1 (human):** review ADR-072; resolve ADR-071's ordering lock, which is recorded only in a document
  that is itself still `proposed/`.
- **PF-3 (views-models, GH ISSUE — not a commit from here):** add `"data_format": "feature_frame"` to the
  target's queryset descriptor. One key. `declared_data_format` fails loud on a typo
  (`datafactory_contract.py:43-48`).

## S0.5 — the two measurements that can stop the program (do before S1)
- **PF-6 — source dtype at the zarr boundary.** If float64, the frame coerces to float32 upstream of us,
  `atol=0` is structurally unreachable, and that is the finding. **Cheap; a STOP if it fires.**
- **PF-5 — the pandas path's NaN-fill count on the anchor partition**, read from
  `dataloaders.py:561-574`'s log line. Zero ⇒ the paths agree. Non-zero ⇒ the frame path raises where the
  pandas path trained, and the migration has surfaced a coverage hole. Record it; do not fill it.

## S1 — build the two pieces that have exact oracles, then capture the anchor
- **PF-8:** `views_hydranet/utils/grid_geometry.py::row_col_from_priogrid`, oracle
  `dataloaders.py:537-542`, `_PRIOGRID_NCOL = 720`, numbering from 1 to match `row_offset/col_offset: 1`.
- **PF-7:** the import-purity and dependency-floor tests
  (templates: `views-pipeline-core/tests/test_import_purity.py`,
  `views-datafactory/tests/test_views_frames_floor.py`). Red is fine; F3 is unfalsifiable without them.
- Freeze one v2 origin. Capture the **current** `from_df(...)` volume and one origin's predictions as the
  golden reference, checksummed.

## S2 — red tests (TDD, must fail for the right reason)
- `tests/test_volume_handler_from_feature_frame.py`: `from_feature_frame(ff).volume == from_df(df).volume`
  at `atol=0` on the S1 anchor.
- The five malformed-frame cases of P5, each raising a **named** exception at the handshake — our gates,
  since the frame type supplies neither (§1.2).
- `FeatureScaler` numpy-path parity **with the float64 policy in place**.

## S3 — implement (smaller than originally planned)
- `VolumeHandler.from_feature_frame(ff, config)` — the 18-step transcription table in `03_harness.md §2`;
  13 of the 18 steps are unchanged code.
- `DataFetcher`: `load_frame_cache(self._get_cached_frame_path())` in place of `read_dataframe`.
  **Delete `apply_blueprint` from the input path** (§1.4); leave it for `prepare_actuals_df`.
- `FeatureScaler` over `ff.values` with the float64 upcast; numpy-native `DataSniffer` hot-path gates.
- `from_df` kept as a deprecated shim for the parity window. Both paths live behind a flag.

## S4 — THE GATE (unchanged, and deliberately so)
Frame-native path on the S1 origin → volume and predictions match the golden reference at `atol=0`. Suite
green, determinism gate, ruff. **If parity fails and the A1.1 precision policy is already in force, F1
fires and the program STOPS.** A1.4 pre-excludes every other "reconciliation".

## S5 — flip + retire
Default to frame-native. Remove pandas from `data_fetcher`, `volume_handler`, `feature_scaler` and the
`data_sniffer` hot path. Keep it behind the `viz` extra and on the audit-only island. Update the CICs
(`DataFetcher`, `DataSniffer`, `FeatureScaler`, `VolumeHandler`) in the same PR; amend ADR-047's
"input out-of-scope" clause to point at ADR-072.

## S6 — verify + close
PF-7's import-purity test goes green; smoke train+eval byte-identical to the pre-flip baseline; promote
ADR-072; archive the dossier; register the residuals (the audit-only pandas island, C-173/C-174, the three
viewser models excluded by the frame-capable-source gate).

## What changed from the original staging
| | original | revised |
|---|---|---|
| inbound | integrate a cross-repo fetch API | read a different artifact off disk — pipeline-core #290 shipped the rest |
| derivations | port pandas → numpy, parity-gated | already numpy; **delete** the pandas half |
| uniqueness / NaN | free from `SpatioTemporalIndex` | **port both gates ourselves** |
| row/col | "from `ff.index`" | **a new decoder, used twice**, with an exact oracle |
| precision | unmentioned | **pre-registered policy**, or the gate fails for the wrong reason |
| STOP risks before S1 | none named | **two** (PF-6 dtype, PF-5 fill count) |
