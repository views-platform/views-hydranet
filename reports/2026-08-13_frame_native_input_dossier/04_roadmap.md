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
