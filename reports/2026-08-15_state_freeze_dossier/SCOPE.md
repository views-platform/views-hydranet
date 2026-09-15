# SCOPE — what this dossier does and does not do

## In scope
Measure whether holding the ConvLSTM recurrent state during free-running preserves **gate** skill, on a
saved artifact, emit-only. Four arms (`none` / `hidden` / `cell` / `all`), one vehicle, activation-aware
metrics.

## Explicitly out of scope

| # | Excluded | Why |
|---|---|---|
| 1 | **Reinstating `freeze_h`** | ADR-027 retired it and that stands. A hard freeze is a train/inference mismatch. `freeze_recurrent` is a diagnostic argument with **no config key**, so no production run can enable it. |
| 2 | Building a soft / decayed state prior | That is the *next* question and needs its own pre-registration — and only if the verdict warrants it. |
| 3 | Re-running the C-113 bloom question | The 2026-06 ablation answered that (freezing is inert against the runaway) and this dossier does not revisit it. `crps_all` at h36 is indeed unmoved across all four arms. |
| 4 | The body / `truncated_nb` | The T=0 composition defect is fixed and separate. |
| 5 | The three #262 training approaches | This is a diagnostic, not a fix. |
| 6 | `violet_visitor` confirmation | 160 lessons, 13 origins — the confirmation vehicle, **not run**. Every claim here is INDICATIVE. |
| 7 | Multi-seed | One seed. The direction and the hidden-vs-cell asymmetry are the result; magnitudes are not rankable. |

## The railguard that held
**Fix nothing the probe finds.** Everything changed here is infrastructure *for* the measurement — the
diagnostic argument, the driver, the guards. No model was retrained and no remedy was implemented.
