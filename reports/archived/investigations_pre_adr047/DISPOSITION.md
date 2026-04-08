# Disposition: Pre-ADR-047 Investigation Archive

| Field         | Value                                                        |
|---------------|--------------------------------------------------------------|
| Archived      | 2026-04-08                                                   |
| Reason        | ADR-047 (pandas-free output path) rendered these artifacts obsolete |
| Archived by   | Simon / Claude                                               |
| Decision      | Preserve as historical record; do not reference from active docs |

## Context

These files document the exploration, planning, and prototyping work that led to
ADR-047 (Pandas-Free Prediction Output Path). The investigation spanned January
to February 2026 and evaluated multiple approaches: Polars reconstruction, Vader
bridge, PureStateAdapter, stochastic DataFrame scalability, and manifest-driven
evaluation — before converging on the numpy-only PredictionFrame path.

The code in these files references classes, methods, and API signatures that no
longer exist in the codebase:

- `VolumeHandler.to_evaluation_df()` — removed
- `VolumeHandler.to_forecast_df()` — removed
- `PureStateAdapter` — removed
- `InferenceOrchestrator.generate_forecasts()` — removed
- `wrap_predictions(base_names=...)` — parameter renamed to `target_names`

## Contents

| Directory / File | Type | Description |
|---|---|---|
| `archive_stochastic_bridge/` | Python scripts | 11 sandbox probes: Polars nesting, Vader bridge, hostile consumer audit, stochastic handshake |
| `archive_prompts/` | Markdown | 4 prompt transcripts from planning sessions |
| `2026-02-04_df_reconstruction_bottleneck.md` | Investigation | Identified the DataFrame bottleneck that motivated ADR-047 |
| `2026-02-04_stochastic_dataframe_scalability_plan.md` | Plan | Scalability options for stochastic DataFrame output |
| `2026-02-04_stochastic_reconstruction_options.md` | Analysis | Comparative analysis of reconstruction strategies |
| `2026-02-04_evaluator_implementation_plan.md` | Plan | Evaluator design that was superseded by ADR-047 |
| `2026-02-04_manifest_driven_evaluation_proposal.md` | Proposal | Manifest approach — not adopted |
| `2026-02-05_evaluation_adapter_plan.md` | Plan | Adapter pattern — superseded by direct PF construction |
| `2026-02-05_pure_state_implementation_plan.md` | Plan | PureStateAdapter design — class later removed |
| `2026-02-21_evaluation_ontology_liberation_plan.md` | Plan | Evaluation decoupling — resolved by ADR-047 |

## Policy

These files are retained for historical context only. They must not be:
- Referenced from active ADRs, CICs, or standards
- Used as implementation guidance (APIs have changed)
- Treated as current architectural documentation
