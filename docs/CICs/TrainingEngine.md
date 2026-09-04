# Class Intent Contract: TrainingEngine

**Status:** Active
**Owner:** Training Strategy
**Related ADRs:** ADR-011, ADR-056, ADR-058 (parked), ADR-060, ADR-070; C-246 / C-259 (feedback parity), C-184 (BN recalibration)
**Last reviewed:** 2026-08-14

---

## 1. Purpose

`views_hydranet/train/training_engine.py` owns the **recurrent training loop** that threads LSTM
hidden state through a temporal window and feeds predictions back across the sequence. The model's
`forward` is per-timestep; this module is where the T-loop, teacher forcing, and scheduled sampling
(ADR-056) live (`_process_sequence`, `training_engine.py:253`). It also drives **curriculum lesson
iteration** (`training_loop`, `training_engine.py:803`) and **post-training BatchNorm
recalibration** (`_recalibrate_bn`, `training_engine.py:777`). It is pure training logic with no
framework dependencies — Entity-layer gradient/sequence/curriculum math kept off Framework-layer
types (module docstring `training_engine.py:1`).

---

## 2. Non-Goals (Explicit Exclusions)

- This module does **not** define the model topology — it consumes `choose_model(config, device)`
  and calls `model(...)` (`training_engine.py:89`, `:346`); architecture lives in
  `views_hydranet/architectures/`.
- This module does **not** define the loss families — it consumes `choose_loss` / `FamilyLoss` /
  `resolve_family` (`training_engine.py:98`, `:434`, `:688`); distributions live in
  `views_hydranet/distributions/`.
- This module does **not** perform the free-running inference rollout — that is
  `hydranet_inference` (mirrored, not owned; see §7).
- This module does **not** validate the scheduled-sampling parameters — that is
  `HydraNetConfig.validate_scheduled_sampling_params` and the `ScheduledSamplingMixer` constructor
  (see the ScheduledSamplingMixer CIC).
- This module does **not** fetch data or manage artifact I/O — it receives a `VolumeHandler`
  (`training_engine.py:591`, `:809`).

---

## 3. Responsibilities and Guarantees

- **Recurrent T-loop authority:** `_process_sequence` steps `range(seq_len - 1)`
  (`training_engine.py:325`), carrying hidden state `h` forward (`output.h_next`,
  `training_engine.py:347`) and returning it (`:539`). Guarantees each step forwards
  `model(t0_input, h)` and accumulates a multi-task loss per step (`:346`, `:516`, `:523`).
- **Ground-truth loss target (invariant):** the loss is **always** computed against the true next
  step `y_reg = t1[:, idx.reg]` / `y_cls = t1[:, idx.cls]` (`training_engine.py:330`). Scheduled
  sampling substitutes only the **INPUT**, never the target (ADR-056; see §4).
- **Input-only scheduled substitution (invariant):** when `ss_epsilon > 0` and a previous
  prediction exists, `dyn_input = torch.where(mask, prev_pred, t0_gt)` (`training_engine.py:337-339`),
  with `prev_pred` **detached** (`:362`, `:364`) so no gradient flows through the feedback edge.
- **Train/inference feedback parity (invariant):** the fed-back copy is built by
  `_family_feedback_log1p` (`training_engine.py:218`), which for `ss_feedback="sample"` produces a
  composition-aware family draw that **mirrors** `hydranet_inference._sample_feedback`
  (`hydranet_inference.py:292`). The premise of scheduled sampling ("train exposure == deploy
  exposure") holds only if the two construct the same object; a mismatch silently invalidates any
  SS verdict (C-246 / C-259; test `tests/train/test_feedback_parity.py`).
- **Input noise touches DYNAMIC channels only (invariant):** `input_noise_dropout` (#311) is applied to `dyn_input` after the scheduled-sampling resolution and **before** the static re-attach, and `_noisable_channels` excludes any feature also declared static. Geometry is *"always the true values, never sampled"*. The Stage-5 diagnostic biopsy is never noised — it is a clean-performance probe. ⚠️ No arm in this fleet declares statics, so the exclusion branch is covered by a synthetic-config test (C-309).
- **Static-channel re-attach (invariant):** every forward re-attaches input-only static channels
  as `[dynamic ⧺ static]` via `_attach_static_channels` (`training_engine.py:160`, called at `:344`
  in the main loop and `:710` in the diagnostic biopsy) — statics are geometry-constant, always the
  true value, never sampled or fed back (ADR-060 I3).
- **Curriculum orchestration:** `training_loop` iterates `total_lessons × windows_per_lesson`,
  handshaking `CurriculumLearner` (planner) and `VolumeSampler` (lens) per window
  (`training_engine.py:921-959`), accumulating loss and stepping the optimizer once per lesson at
  the optimization gate (`:974`, `:1032`).
- **BN recalibration (default ON, C-184):** `training_loop` calls `_recalibrate_bn` unless disabled
  (`config.get("bn_recalibrate", True)`, `training_engine.py:1083`). It resets every BatchNorm's
  running stats and sets `momentum=None` (`_reset_bn_stats`, `:765`), then re-accumulates them
  forward-only over real curriculum windows (`:794-800`) — fixing the seed-bimodal eval collapse
  (~40% of trained seeds saturate the gate at eval).
- **Determinism guarantee:** `make` re-seeds immediately before construction
  (`ReproducibilityGate.lock_entropy`, `training_engine.py:85-88`, C-119) so weight init depends
  only on the seed, not on RNG consumed earlier in the pipeline; `training_loop` re-locks entropy at
  start (`:819`).
- **Fail-loud on explosion:** each lesson passes through `IntegrityGuardian.monitor`
  (`training_engine.py:976`).

---

## 4. Inputs and Assumptions

- **`make(config, device)`** (`training_engine.py:80`): builds `(model, criterion, optimizer,
  scheduler)`. Assumes `config` carries seeds for production/determinism runs (`np_seed` /
  `torch_seed`); seedless callers (unit tests) are guarded (`:85`). Wires ADR-055 learnable sigmas
  and the C-111 MultiTaskLoss balancer into the optimizer with `weight_decay=0.0` (`:106-134`).
- **`TrainingContext`** (`training_engine.py:544`): bundles the "wired once" components
  (`model`, `optimizer`, `scheduler`, `criterion_reg`, `criterion_class`,
  `multitaskloss_instance`, `config`, `device`, `viz`, `forensics`). Reduces `train()` from 13
  parameters to 5 (docstring `:545-548`, C-17). Constructed once in `training_loop`
  (`:842-853`).
- **`train(ctx, sample_handler, pbar, stage_label="", ss_epsilon=0.0)`** (`training_engine.py:589`):
  assumes `sample_handler` is a `VolumeHandler` convertible to a `[B, T, C, H, W]` tensor
  (`:613`), and that `config` supplies `regression_targets` / `classification_targets` / `features`
  / `static_channels` for `_SequenceIndices` (`:144-157`).
- **`training_loop(config, model, criterion, optimizer, scheduler, handler, device,
  run_timestamp=None)`** (`training_engine.py:803`): assumes `config` carries `np_seed`,
  `torch_seed`, `total_lessons`, `windows_per_lesson` (`:819`, `:876`, `:921`).
- **`ss_epsilon` convention:** the probability of using the **model's prediction** in place of
  ground truth (complement of Bengio et al.; see ScheduledSamplingMixer CIC). Assumed in `[0, 1]`;
  the guard `ss_epsilon > 0.0` gates the whole feedback branch (`:337`, `:353`), so `ss_epsilon=0`
  is byte-identical to pure teacher forcing.
- **`ss_feedback ∈ {mean, sample}`** (`training_engine.py:277`): `"mean"` feeds
  `log1p(E[y])`; `"sample"` feeds one composition-aware family draw per target
  (`:230-250`). Assumes it matches inference's feedback mode (parity invariant, §3).

---

## 5. Outputs and Side Effects

- **`make`** returns `(model, criterion, optimizer, scheduler)` (`training_engine.py:136`); mutates
  the model in place via `init_weights` and optional onset-bias init (`:91-96`).
- **`train`** returns `{"total", "reg", "cls"}` loss tensors (`training_engine.py:758-762`); the
  `"total"` retains the graph for `.backward()` in `training_loop` (`:964`). Side effects: sets
  `model.train()` / `multitaskloss.train()` (`:596-597`), optional random flips
  (`:606-610`), forensic recording (`:371-414`), and Stage-5 diagnostic biopsy plots when
  `viz and stage_label` (`:696-752`). An empty `stage_label` ⇒ forward-only, no biopsy (used by
  BN-recal, `:798`).
- **`_process_sequence`** returns `{"total", "reg", "cls", "h", "per_step_losses"}`
  (`training_engine.py:535-541`). The scheduled-sampling draw is **stochastic** (family sample +
  composition Bernoulli), seedable via the `generator` argument (`:218`, `:228`, C-261).
- **`training_loop`** returns a diagnostic summary dict (`final_loss`, `min_loss`, `max_loss`,
  `max_raw_grad_norm`, `loss_history`, `weight_norms`, `learning_rate`) (`training_engine.py:1109-1117`).
  Side effects: mutates `model` weights (optimizer steps, `:1032`) and BN running stats
  (`_recalibrate_bn`, `:1090`); emits wandb per-lesson metrics when a run is active
  (`:1040-1067`); optional trajectory CSV (`:900-915`, `:1011-1024`); optional `bn_recal_from`
  artifact load (`:859-864`).

---

## 6. Failure Modes and Loudness

- **Explosion (hard stop):** `IntegrityGuardian.monitor` runs per lesson and hard-stops on a
  non-finite / exploded loss (`training_engine.py:976`). This is the primary loud guard.
- **The SS parameter guards live upstream, not here:** invalid `ss_schedule` / `ss_k` /
  `ss_epsilon_max` are rejected by `HydraNetConfig.validate_scheduled_sampling_params` and the
  `ScheduledSamplingMixer` constructor before the mixer reaches `training_loop` (`:885-893`); the
  engine itself does not re-validate them.
- **The engine raises no explicit `ValueError`s of its own** in the hot loop — correctness is
  enforced by invariants (detached input-only substitution, static re-attach, ground-truth targets)
  rather than by raises. A silent violation of the feedback-parity invariant (§3) is the sharpest
  hazard: it does not raise, it degrades the scientific validity of scheduled sampling, and is
  therefore pinned by a byte-equality test (`tests/train/test_feedback_parity.py`).
- **BN-recal is fail-safe, not fail-loud:** a recalibration failure must never lose a completed
  run — `training_loop` snapshots the BN buffers first and restores them on any exception, then
  saves the model as-is (`training_engine.py:1084-1098`).
- **Missing wandb run is warned, not fatal:** logged at `WARNING` so a missing train run is visible
  rather than an empty dashboard (`training_engine.py:828-833`, C-134).
- **Diagnostic biopsy is defensive:** a failed time-index extraction is logged and the biopsy is
  skipped, never crashing training (`training_engine.py:644-648`).

---

## 7. Boundaries and Interactions

- **Model:** consumes `model(t0_input, h)` returning `output.reg` / `output.cls` / `output.h_next`
  (`training_engine.py:346-347`) and `model.init_hTtime(...)` (`:632`). Treats the architecture as
  opaque.
- **Losses:** consumes `choose_loss` (`:98`), per-target `FamilyLoss` / `QuantileLoss` dispatch
  (`:434-441`), the `MultiTaskLoss` balancer (`:516`), and `resolve_family` for the feedback /
  biopsy (`:688`, `:727`, `:743`).
- **Curriculum / sampling:** handshakes `CurriculumLearner` (planner) and `VolumeSampler` (lens)
  per window (`:934-937`).
- **Forensics:** consumes `TrainingForensics.record` / `record_params` / `finalize_lesson` /
  `get_dossier` (`:392`, `:398`, `:990`, `:996`), tagging each with `step_idx=i` for horizon
  splitting (`:393`).
- **Inference mirror (must stay behaviorally identical):** the recurrent loop here is implemented
  **independently** of the free-running rollout in `hydranet_inference`, and the two MUST stay
  behaviorally identical (C-246). Concretely, `_family_feedback_log1p` (`:218`) mirrors
  `hydranet_inference._sample_feedback` (`hydranet_inference.py:292`), and
  `_family_target_log1p_mean` (`:203`) mirrors `_emit_magnitude`'s family branch
  (`hydranet_inference.py:234`).
- **Reproducibility:** consumes `ReproducibilityGate.lock_entropy` (`:86`, `:819`).
- **Dependency Rule:** by design this module does NOT import `views_pipeline_core` /
  `ModelPathManager` at module level (docstring `:4-9`; enforced by
  `tests/test_training_engine.py`).

---

## 8. Examples of Correct Usage

```python
from views_hydranet.train.training_engine import make, training_loop

# Build the wired components, then run the curriculum loop.
model, criterion, optimizer, scheduler = make(config, device)
summary = training_loop(
    config, model, criterion, optimizer, scheduler, handler, device,
    run_timestamp=run_ts,
)
# summary -> {"final_loss", "min_loss", "max_loss", "max_raw_grad_norm", ...}
# Post-loop, `model` has C-184-corrected BN running stats (bn_recalibrate default ON).
```

```python
# Enabling scheduled sampling: set ss_schedule + ss_feedback in config. training_loop constructs
# the mixer, computes ss_epsilon per lesson, and threads it through train() -> _process_sequence.
config["ss_schedule"] = "linear"
config["ss_epsilon_max"] = 0.5
config["ss_feedback"] = "sample"   # MUST match inference feedback for parity (C-246/C-259)
```

---

## 9. Examples of Incorrect Usage

- **Substituting the loss target under scheduled sampling.** Scheduled sampling replaces only the
  INPUT (`torch.where(mask, prev_pred, t0_gt)`, `:339`); computing loss against `prev_pred` instead
  of the true `y_reg` / `y_cls` (`:330`) breaks the ground-truth-target invariant (ADR-056).
- **Feeding back a non-detached prediction.** `prev_pred` must be `.detach()`ed (`:362`, `:364`);
  leaving the graph attached lets gradients flow through the exposure edge — not the trained model.
- **Setting `ss_feedback="sample"` in training while inference feeds the mean (or vice versa).**
  This silently violates train/inference parity (C-246/C-259) — the two feedback objects diverge and
  any scheduled-sampling result is invalid.
- **Skipping static re-attach in a hand-rolled forward.** The model was widened for statics
  (ADR-060 I3); passing `N` dynamic channels into an `N+static`-channel model crashes or mis-feeds
  — always go through `_attach_static_channels` (`:160`, `:710`).
- **Re-implementing the AR loop in the manager instead of mirroring inference.** The train and
  inference loops are the two halves of one behavior (C-246); a third copy drifts.

---

## 10. Test Alignment

- **🟩 Green — structure & smoke:** `tests/test_training_engine.py` (module exists, no
  `views_pipeline_core` import, exports `make`/`_SequenceIndices`/`_process_sequence`/`train`/
  `training_loop`, C-18 full-loop smoke on tiny synthetic data).
  `tests/test_train_loop.py` (train() finite loss + flowing gradients; multi-lesson dynamics).
- **🟩 Green — feedback parity (the load-bearing anchor):**
  `tests/train/test_feedback_parity.py` — byte-equality of `_family_feedback_log1p` vs
  `hydranet_inference._sample_feedback` under a shared seeded generator (C-246/C-259). Protects the
  train/inference exposure invariant against regression.
  `tests/train/test_gtf_sample_feedback.py` — EXP-4/GTF: `"mean"` collapses to `n_reg` channels,
  `"sample"` is a composition-aware draw, `ss_epsilon=0` stays byte-identical.
- **🟩 Green — scheduled sampling:** `tests/test_scheduled_sampling.py` (schedule computation,
  config acceptance/rejection, integration; see ScheduledSamplingMixer CIC).
- **🟩 Green — BN recalibration (C-184):** `tests/test_bn_recalibrate.py` — `_recalibrate_bn`
  resets BN, re-accumulates forward-only (`num_batches_tracked > 0`), leaves the model in eval mode
  without touching weights; `training_loop` calls it by default and skips it only for a
  `bn_recal_from` run.
- **🟩 Green — forensics horizon:** `tests/test_forensics_horizon.py` +
  `tests/test_training_forensics.py` — `step_idx` populates lesson-aligned horizon slices.
- **🟩 Green — static seam (ADR-060):** `tests/test_static_channel_seam.py`,
  `tests/test_data_backed_static_channel.py`, `tests/test_static_top_skip.py`,
  `tests/test_active_window_mask.py` (re-exported `_active_window_mask`).
- **🟥 Red — determinism / reproducibility:** `tests/test_training_engine.py::
  test_init_deterministic_regardless_of_prior_rng_state` (C-119) and
  `::test_training_run_is_reproducible` (C-79) — same-seed weight identity independent of prior RNG
  consumption. `tests/test_reproducibility_gate.py`.
- **🟥 Red — failure modes:** `tests/test_training_engine.py::TestRed` (NaN weights ⇒ non-finite
  loss, not silent continuation); `tests/test_optimization_gate.py`;
  `tests/test_balancer_freeze.py` (C-113 bisect); `tests/test_feedback_clamp.py`.

---

## 11. Evolution Notes

- **Stable:** the ground-truth-target and detached-input-only-substitution invariants (ADR-056); the
  static re-attach seam (ADR-060 I3); BN-recal default ON (C-184).
- **Coupled to inference:** any change to `_family_feedback_log1p` or `_family_target_log1p_mean`
  MUST be mirrored in `hydranet_inference` (`_sample_feedback` / `_emit_magnitude`) or the parity
  test breaks — this coupling is deliberate (C-246) and must be preserved, not refactored away.
- **Expected to change:** the additive regularizers threaded through `_process_sequence` (qs99
  decay-gate penalty, `pi_penalty` family ridge — C-200/C-205) are opt-in hooks that may grow; new
  ADR-070 sample-feedback compositions extend the `ss_feedback` / `forecast_composition` axes.
- **Revisit this contract if:** the AR T-loop moves out of `_process_sequence`, the feedback object
  stops mirroring inference, or BN-recal stops being the default.

---

## End of Contract

This document defines the **intended meaning** of the training-engine surface
(`TrainingContext` + `make` / `train` / `training_loop` / `_process_sequence` /
`_family_feedback_log1p` / BN recalibration).
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
