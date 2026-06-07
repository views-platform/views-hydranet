# wandb training-logging bug — falsification claim ledger

**Date:** 2026-06-07 · **Branch:** `fix/wandb-training-run-logging` (views-hydranet)
**Mode:** information-gathering loop — `/falsify` each claim → `/register-risk` the findings.
**Scope:** READ-ONLY investigation. No source fixes (that comes later). `/falsify` may write
failing-test stubs + reports; `/register-risk` writes the register. Nothing else.

**Context:** A `python ./main.py -r calibration -t -e` run produced an empty/early-finished
wandb run; training metrics did not appear. An initial diagnosis (hydranet manager's
`_execute_model_training` override dropping the base `initialize_run("train")` wrapper) was
flagged by the user as **WRONG / incomplete** → these claims must NOT presuppose it.

## Loop procedure (per claim)
```
for claim in CLAIMS:
    investigate enough to state the claim concretely
    /falsify claim          # design probes + execute (autonomous, no approval stop) + classify
    /register-risk findings  # dedupe + record to reports/technical_risk_register.md
    update PROGRESS.md
# converge when every claim has a recorded verdict; then stop. NO fixes.
```

## Claims

| # | Name | Claim | Status |
|---|------|-------|--------|
| 1 | Locus | The root cause is in **views-hydranet**, not views-pipeline-core, the wandb SDK, or wandb account/entity/project config. | pending |
| 2 | Location | The defect is fully localized to one identified spot (file+function+lines); it is the true cause (not a symptom), with **no second contributing site**. | pending (unverified) |
| 3 | Mechanism | The stated causal chain — how training metrics fail to reach wandb — is correct and complete. | pending (unverified) |
| 4 | Harm | The harm is correctly characterized: what is lost (which metrics/runs), under which run types (train/eval/sweep/forecast), and that nothing beyond observability (artifacts, training correctness, eval numbers) is affected. | pending |
| 5 | Fix | The proposed fix actually makes training metrics reach wandb and addresses the cause (not a symptom patch). | pending (unverified) |
| 6 | Splash zone | We understand the full blast radius (every run type, manager, ensemble, other model sharing the affected code) and the fix introduces no regressions there. | pending (unverified) |
| 7 | Why not caught | We correctly understand the specific test/CI/process gap that let this through. | pending |
| 8 | Prevention | We have a concrete, adequate guard (test/check) that would catch this class of regression in future. | pending |

**Note:** claims 2, 3, 5, 6 rest on the diagnosis the user flagged as wrong → expected to fire.
