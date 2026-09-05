# 03 — Harness and invariants

Audited 2026-09-04 for Epic #311 / S0 (#312). **The honest finding: the harness largely exists. The
gaps here are wiring and one new instrument, not invention.** Several good gates are CI-tested and
invoked by no recent dossier.

## A. Invariants

### A.1 Hard — violating one invalidates the programme

| # | Invariant | Where it lives |
|---|---|---|
| H1 | **Statics are never noised.** CoordConv channels are geometry — *"always the true values, never sampled and never fed from the output"* | `_attach_static_channels`, `training_engine.py:187-198` |
| H2 | **The default path is byte-identical.** Noise off ⇒ the pre-flag model, bit for bit | the repo's standing rule for every opt-in flag |
| H3 | **The Stage-5 diagnostic biopsy is never noised.** It is a *clean-performance* probe; corrupting it destroys the thing it measures | `training_engine.py:920-926` |
| H4 | **One variable per arm.** | epic #311, `05_analysis_plan.md` |
| H5 | **Fail loud, no silent clamp.** A noise scale that silently degrades to a no-op is the C-324 signature | `IntegrityGuardian`; config validators |
| H6 | **Full suite + lint green; CIC field count in sync.** | CI: `ruff check .`, `ruff format --check .`, `pytest tests/` |

### A.2 Deliberately changed by this programme

The training input is currently **exactly the ground truth** (or, under ε>0, exactly the model's own
prediction). This programme replaces that with a **deliberately corrupted** input on the noise arm
only, behind a default-off flag. Reviewers should not defend the clean-input behaviour on the treated
arm — replacing it *is* the experiment.

### A.3 Respect while changing

- **BatchNorm.** Recurrent BN is already a documented source of seed-bimodality (C-184). Input noise
  changes BN's batch statistics; the effect is real and must not be mistaken for the treatment.
- **The `ss_epsilon` seam.** Noise is injected *after* the scheduled-sampling `torch.where` so both
  branches are augmented identically. Injecting into the `else` only would leave the ε>0 arm
  unaugmented and make the arms incomparable.
- **The pushforward self-fed forward** (`training_engine.py:695`). Pushforward is *itself* a
  noise-injection alternative; whether its input is also noised is a **deliberate fork**, to be
  recorded in `02_design.md`, not decided by accident.

## B. The standing harness — what already exists

| Mechanism | Status | Where |
|---|---|---|
| Default-off feature flags | **exists, mature** | `config_initializer.py` — `pushforward_weight`, `ss_backprop_through_feedback`, `ss_feedback_grad_clip`; every one defaults to the no-op |
| No-shadow-default discipline | **exists, AST-enforced** | `tests/test_falsification_magic_numbers.py:31` |
| Parity / regression gates | **exists** | CI: `ruff` + `pytest tests/`; CIC field-count test at `tests/test_falsification_loss_param_validation.py:36` |
| Reproducibility | **exists, with a known limit** | `ReproducibilityGate.lock_entropy`; `scripts/compare_run_determinism.py`. ⚠️ C-119: init-time RNG drift was real, and **the `.pt` sha256 is not a valid weight-identity check** — judge by weight-tensor hash |
| Fast, retrain-free readout | **exists** | emit-only arms via `run_realism_arms.py`; ~10 min/arm vs ~110 min to retrain |
| Evaluation comparability | **exists** | `AP@h18`, `sb`, free-running, 13-origin support; `score_arms.sh` / `read_screen.py` |
| Hardware gates | **exists** | launcher `guard()`: CUDA present (C-163 — no silent CPU grind), `<3000 MiB` already on the GPU, `≥25G` free, HEAD unchanged mid-run |
| Run discipline | **exists** | `flock`, heartbeat/`PHASE`, per-arm `timeout`, stall watchdog, `ANOMALIES.txt` |
| Negative-result discipline | **exists** | `08_postmortem.md` convention; six post-mortems on record |
| **Potency gate (C-324)** | **exists as a pattern, NOT as a hook** | `scripts/potency_check.py` (96 lines, CI-tested). **Every dossier must write its own `preflight_*.py`** |
| **`scripts/floor_gate.py` (C-299)** | **exists, CI-tested, invoked by NO recent dossier** | 222 lines. Last used by `2026-08-17_ss_retention` and `2026-08-18_lesson_curve` |
| **`scripts/arm_postflight.py`** | exists, CI-tested, wired per-dossier | 161 lines |
| **`scripts/arm_identity_check.py`** | exists, CI-tested, wired per-dossier | 138 lines |

> **The structural finding: no gate in this repo is repo-wide.** Each is opted into by a dossier
> launcher, and a dossier that forgets one gets **no warning**. That is why §D enumerates the gate set
> rather than assuming it.

## C. New harness this programme needs (build before the first run)

1. **The free-running error instrument** (S1 #313) — the only genuinely new measurement. Must be
   truth-referenced: **C-319** is Tier 2 precisely because occurrence, magnitude and alignment are all
   *internal* statistics that survive a roll which destroys the forecast.
2. **A statics-untouched test using a SYNTHETIC config.** ⚠️ Measured at S0: `static_channels` is
   **empty** on `fullzero_fortytwo` and `screendetached_fortytwo`, and `features ==
   regression_targets` exactly. So **H1 cannot be exercised by any arm in this fleet** — the guard
   would be real code that no test ever runs. **C-309: a guard whose passing case has never been
   observed is not a guard.** The test must build a config *with* statics.
3. **`preflight_input_noise.py`** — the potency gate on the arm's own config, and at a **trained**
   checkpoint (C-325, one day old: two #308 mechanism tests measured an untrained network, returned
   correct numbers, and were recorded as ruling a mechanism out).
4. **Launcher wiring** for `floor_gate`, `arm_identity_check`, `arm_postflight`, the weight-hash
   post-condition, and the recursive **`kill_tree`** (C-326 — `kill -TERM $APID` signals the subshell
   and leaves the training process alive; measured running 7 lessons past a declared cap).

## D. Pre-flight checklist — must be green before S5 spends GPU

- [ ] Noise implemented + unit-tested, mutation-tested to exhaustion — **blocker** (S2/S3)
- [ ] Behind a default-off flag; **byte-identity with it off proven by a test**, not asserted
- [ ] **H1 tested against a synthetic config carrying statics** (see C.2)
- [ ] Config field count 95 → 96; CIC annotated; `TrainingEngine.md` invariant added
- [ ] `05_analysis_plan.md` locked **before** S1 produces numbers, with a non-numeric decision branch
- [ ] Potency gate passes on the arm's own config **and at a trained checkpoint**, and is shown able to
      **refuse** a deliberately inert configuration
- [ ] `floor_gate` wired and **passing on the control arm before any treatment arm starts**
- [ ] `arm_identity_check` confirms the arms differ in exactly the intended keys
- [ ] Weight-hash post-condition wired ahead of any score read
- [ ] `kill_tree` in the launcher; `bash -n` clean
- [ ] Full suite + lint green; tree clean

## E. Rules of engagement

One variable per arm. Pre-register, then run. Cheap readout before expensive. **A pre-registered
falsifier that fires kills the hypothesis — document it, never rescue it ad hoc.** And improvements
must come from the representation, not from masking: a clamp that hides a symptom is not a result.

**Read the control arm's score the moment it lands, not after the queue drains.** The floor-limited
post-mortem lost three days to a condition visible in the control's own CSV **5 h 47 min** before the
last arm finished.
