# wandb falsification loop — PROGRESS / verdicts

**Date:** 2026-06-07 · **Branch:** `fix/wandb-training-run-logging` · See [CLAIMS.md](CLAIMS.md).
**Method:** 3 parallel investigators (lifecycle trace · entity/project · git-history) + direct
code read + run-dir data survey, then reconciled against observable facts.

## The reconciled truth (path-specific bug)

`_train_model_artifact()` is reached by THREE phase methods in the base `ModelManager`
(views-pipeline-core `managers/model/model.py`), each wrapping it in its own wandb run:

| Path | Route | hydranet override? | wandb run open during training? | logs? |
|------|-------|--------------------|---------------------------------|-------|
| Single train (`-t`) | `_execute_model_training` (base ~L1186 wraps `initialize_run(job_type="train")`) | **YES** — `hydranet_manager.py:185-187`, bare `self._train_model_artifact()`, no wrapper | **NO** | ❌ training metrics lost |
| Sweep (`-s`) | `_execute_model_sweeping` (base ~L1537 wraps `initialize_run(job_type="sweep")`, calls `_train_model_artifact()` at ~L1551) | no | yes | ✅ training curves |
| Eval (`-e`) | `_execute_model_evaluation` (base ~L1253 wraps `initialize_run(job_type="evaluate")`) | no | yes | ✅ CRPS/MCR/QS |

**Root cause:** the hydranet `_execute_model_training` override replaces the base method
(which wraps the call in `initialize_run("train")` AND runs `TrainingStage.finalize_training`
+ `ModelTrainingException` handling) with a bare `self._train_model_artifact()`. So on the
single-run `-t` path only, `wandb.run is None` throughout training and every guarded
`wandb.log` in `training_engine.py` (L640/651/664) no-ops. Sweeps and eval are NOT overridden → unaffected.

**Run-dir evidence:** pink_pirate has TRAIN-type runs (6 keys, May 28/31) from sweeps + many
EVAL runs (81-91 keys) through June 5. violet/blue have EVAL runs (81 keys). Today's violet
`-t -e` produced only a 23s fetch run (training logged nothing; killed before eval). → confirms
training logs on sweep/eval paths, never on single-`-t`.

## Falsification verdicts

| # | Claim | Verdict | Note |
|---|-------|---------|------|
| 1 | Locus = views-hydranet | ✅ SURVIVED | the override is in views-hydranet; entity/project + pipeline-core ruled out |
| 2 | Single location | ✅ SURVIVED (refined) | `hydranet_manager.py:185-187`; PATH-SPECIFIC (`-t` only), not all training |
| 3 | Mechanism | ✅ SURVIVED (scope-corrected) | override bypasses `initialize_run("train")`; **NOT universal** — sweeps/eval log fine. Alt "C-111 crash→finish_run" 🔴 FALSIFIED (run reached lesson 60, no crash) |
| 4 | Harm | ✅ CHARACTERIZED | single-run `-t` training metrics (loss, mtl_log_var, sigma, ss_epsilon) silently lost; sweeps + eval + correctness + artifacts + eval numbers UNAFFECTED. Observability-only |
| 5 | Fix correct | ⏳ PENDING | direction: wrap override body in `initialize_run("train")`; NOT "delete override" (it deliberately skips `finalize_training`). → after /expert-code-review |
| 6 | Splash zone | ⏳ PENDING | needs code review: interaction with `finalize_training`, other hydranet phases, other models sharing base |
| 7 | Why not caught | ✅ ESTABLISHED | team lives in sweeps (`-s`, log fine) + eval metrics (`-e`, log fine); single-run training curves rarely relied on → gap invisible. No test asserts a train run is open |
| 8 | Prevention | ⏳ PENDING | candidate: a test asserting `wandb.run is not None` during training / that phase overrides preserve the base wandb lifecycle. → after fix |

## Correction to earlier diagnosis
Earlier I claimed (a) "we broke it a couple days ago" and (b) "training never logs to wandb."
Both wrong. (a) The override is from ~March (longstanding). (b) Training logs fine via sweeps;
only the single-`-t` path is broken. The user's memory of correct results (pink_pirate, June 5)
is accurate. Bug registered as C-132.

## RESOLUTION — finalize_training investigation (3 parallel agents, 2026-06-07)

**Verdict: the override is a STALE DIVERGENCE; the fix is to DELETE it.**

- **Why it skips finalize_training:** it doesn't on purpose. Override last edited 2026-03-03; `finalize_training` first existed 2026-04-06. The override NEVER contained a wandb wrapper or bookkeeping — born behind the base (which had the `job_type="train"` wrapper since 2025-04-09) and never reconciled. Not a deliberate opt-out.
- **finalize_training does** (TrainingStage.finalize_training, stage.py:49-71): (1) `handle_single_log_creation` — writes a training-exec log (needs only fields hydranet has); (2) `send_alert` — wandb completion alert (no-op if notifications off, never raises). NO artifact double-save, NO crash, NO return-shape assumption. Both are ADDITIVE — hydranet currently lacks them.
- **hydranet is the lone offender:** ADR-045/050 + canonical template say subclasses override only the HOOKS (`_train_model_artifact`), never the `_execute_*` phase templates. Baseline/Stepshifter/Darts/Example all override only hooks → all get wandb training runs. The two sanctioned phase-overrides (Ensemble, Darts-sweep) re-implement the lifecycle. Hydranet's bare override is anomalous.

**Claim verdicts finalized:**
- Claim 5 (Fix) ✅ — DELETE `hydranet_manager.py:185-187` (NOT wrap; NOT "skip finalize on purpose" — earlier note reversed by evidence). Restores wandb train run + per-lesson logging + training-log + alert; contract-correct.
- Claim 6 (Splash zone) ✅ — affects only hydranet's training phase; gains wandb run + training-log + completion alert + live per-lesson logging. No other subclass affected (they don't override). EnsembleManager unaffected (own lifecycle). Minor verify: wandb-init config keys vs hydranet `self.configs`.
- Claim 8 (Prevention) ✅ (planned) — (a) fail-loud: one-time WARNING when a training loop runs with `wandb.run is None` (C-134 — catches the class without making the template `@final`, which would break the legitimate Ensemble override); (b) pinning test: assert HydranetManager training phase opens a `job_type="train"` run / per-lesson `wandb.log` fires.

**D-07 resolved:** third option — delete (contract-correct, minimal) + cheap fail-loud. Not Side-A wrap, not Side-B `@final`.

## Next: TDD the fix — red test (train phase opens a run) → delete override → green + fail-loud + verify config keys.
