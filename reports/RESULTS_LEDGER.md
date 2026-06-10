# Results Ledger — HydraNet experiments (C-113 program & beyond)

**Purpose:** a durable, human-curated record of *what we ran, what we got, and what we now
know* — keyed to the **parameters/architecture** of each run, including things wandb does
**not** track (architecture variant, loss family, config knobs, qualitative read). This is our
sanity record; wandb holds the curves, this holds the *conclusions*.

**Living doc.** Append rows; never rewrite history. git-tracked via `git add -f` (reports/ is gitignored).

---

## Evaluation & selection criteria — adopted from FAO Pre-Release Note 05 (Topics C & D)

Source: `~/brain/2_projects/fao02/_dev_materials/prerelease_notes/fao_02_pre_release_note_05/`
(Topic C = model validity/selection; Topic D = ensemble construction). **We use these going forward.**

**Metrics** (cell–month level, temporal backtest protocol):
| Metric | Role | Better |
|--------|------|--------|
| **CRPS** | **primary ranking metric** | lower |
| **QS99** (99th-pct quantile score) | guardrail — tail sanity (catches *timid* models) | lower |
| **Brier** | guardrail — onset/hurdle probability calibration (`y>0`) | lower |
| **MCR** (mean pred ÷ mean obs) | guardrail — magnitude calibration | **closest to 1** |
| *Bounded?* (ours, C-113) | sanity gate — does the 36-step rollout stay in range or `expm1`-explode? | bounded |

**Eligibility (strict conjunction — Topic C.5):** a model is **Eligible** iff
1. **Superiority:** CRPS ≥ **5%** better than the baseline (C.2), *and*
2. **All guardrails non-inferior:** QS99 & Brier within **1%** of baseline; MCR calibration-distance `|MCR−1|` at least 1% closer-to-1 than baseline's (C.3/C.4).

Fail either → **Ineligible** (regardless of ranking). Ranking (CRPS order) ≠ admissibility (guardrails).

**Ensemble (Topic D):** eligible models only → **equal-weight mixture** of predictive samples →
**greedy forward selection** (seed = lowest-CRPS eligible; add the model giving the greatest
*strict* ensemble-CRPS reduction — no 5% margin intra-ensemble; stop when none strictly improves).

---

## Baseline reference

> The baseline that superiority/guardrails are measured against. (FAO uses an empirical heuristic
> baseline, Topic B; for the C-113 program our working reference is the SS-off clean run below until
> a formal baseline is set.) **TBD — set once the first clean eval lands.**

---

## Run ledger

Config fingerprint columns capture the *variation axes* (what we change between runs).
Metrics filled from `--evaluate`; *Bounded?* from the 36-step rollout / `diagnose_io_gain`.

| # | Date | Model · arch | Key params (variation axes) | wandb run | CRPS | QS99 | Brier | MCR | Bounded? | Eligibility | Notes (incl. non-wandb) |
|---|------|--------------|------------------------------|-----------|------|------|-------|-----|----------|-------------|--------------------------|
| R1 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | loss=tobit; **ss_epsilon_max=0.0 (SS OFF, pure TF)**; balancer=active(unfrozen); seeds 42/42; dropout 0.15; sigma{1.0,0.75,0.5}; onset_bias −7.0; log1p; total_lessons=80 | train `serene-plant-49` · eval `ro537u46` | os 0.04 ✓ / sb 0.7 ✓ / **ns 5.9e8 💥** | logged ✓ | logged ✓ | os 0.13 / sb 64⚠ / **ns 1.1e11 💥** | **NO — `lr_ns_best` explodes** | **Baseline (PATHOLOGICAL zero-point)** | C-113 reproduces on the clean SS-off baseline → explosion is NOT caused by scheduled sampling. **Head-specific**: `os` healthy, `sb` bounded but over-predicts ~64×, `ns` explodes (worst in posterior-sample-mean; robust CRPS for ns ~55–129 already ~1000× `os`). Active balancer + log1p. All 4 PRN-05 metrics emit (CRPS/QS99/Brier/MCR). This is the reference every fix is measured against. |
| R2 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | **sweep trial: dropout=0.10, lr=0.0005**; else = R1 baseline (tobit, SS off, sigma{1,.75,.5}, seeds 42, 80 lessons, active balancer, log1p) | sweep `ih3fc9u9` | os 0.04 ✓ / **sb 1.4e8 💥** / ns 50 ⚠ | logged ✓ | logged ✓ | os 2.0 / **sb 2.3e10 💥** / ns 3.0e4 ⚠ | **NO — sb explodes** | Diagnostic (sweep 1/9) | Explosion persists at lower dropout+lr. **Worst head SHIFTED to `lr_sb_best`** (R1 was `ns`) → runaway is not tied to one head; both sb & ns unstable, os consistently healthy. |
| R3 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | **sweep trial: dropout=0.10, lr=0.001**; else = R1 baseline | sweep `esa59shz` | os 0.04 ✓ / sb 0.16 ✓ / **ns 8.2e4 💥** | logged ✓ | logged ✓ | os 0.43 / sb 0.09 ✓ / **ns 3.8e7 💥** | **NO — ns explodes** | Diagnostic (sweep 2/9) | Explodes on `ns` (sb bounded & well-calibrated here). max\|metric\|≈1.8e9 — **less extreme than R1/R2 (~1e11)**. Pattern holding: *something* always explodes (sb or ns), magnitude varies, `os` always healthy. |

**Status legend:** `Eligible` · `Ineligible` · `Baseline` (the reference) · `Diagnostic` (not a candidate, e.g. a probe) · `Failed` (run errored/exploded).

---

## What we know / wins (running narrative)

- **2026-06-07 — wandb training logging restored (C-132).** Single-run `-t` training now opens a
  `job_type="train"` run and logs per-lesson curves (was silently dropped by a stale phase-template
  override). Fix = delete the override; pinning test + fail-loud guard added. → all training results
  from here are observable. See `reports/2026-06-07_wandb_falsification/`.
- **2026-06-07 — clean baseline established (R1).** violet retrained with scheduled sampling OFF
  (`ss_epsilon_max=0.0`) = pure teacher forcing — the unfixed-model zero-point for C-113. Training
  healthy (loss↓).
- **2026-06-07 — C-113 REPRODUCED on the clean baseline (R1 eval).** Explosion is **head-specific**:
  `lr_ns_best` explodes (CRPS_mean ~6e8, MCR ~1e11), `lr_sb_best` bounded but over-predicts ~64×,
  `lr_os_best` healthy (CRPS ~0.04, MCR ~0.13). **Key inference:** the runaway is NOT caused by
  scheduled sampling (it was off) — it persists with pure teacher forcing + active balancer + log1p.
  The explosion concentrates in the **posterior-sample-mean** (a few runaway draws dominate), and is
  worst on the **non-state head**. Leads to chase: why `ns` specifically; the active-balancer (C-111)
  interaction; sample-mean vs robust-aggregate divergence. All 4 PRN-05 metrics confirmed emitted.

- **2026-06-08 ~00:40 — sweep progress + a data-availability flag.** 4/9 trials trained. R2 (0.1/0.0005, sb-explode) & R3 (0.1/0.001, ns-explode) confirmed via each run's *own* config — mappings correct. **BUT trials 3 (0.1/0.002) & 4 (0.15/0.0005) wrote train-only wandb summaries (13 keys, NO CRPS/MCR/eval)** — unlike trials 1–2 (88-key with eval). So their explosion status is **unreadable from wandb so far**. Open question for the morning: *does the sweep reliably evaluate every trial, or only some?* If sweep trials are train-only, the sweep tells us about training stability but NOT the rollout explosion (which is an eval-phase phenomenon) — meaning a sweep may be the wrong tool for the explosion question, and we'd need per-config `-e` runs. Will recheck at completion (eval may flush late) and do a careful full pass then. **Confirmed pattern from R1–R3 (the trials that DID eval): something always explodes (sb or ns), `os` always healthy.**

- **2026-06-08 ~01:09 — overnight sweep OOM-KILLED at trial ~5/9 (NOT restarted — would re-die).**
  Kernel log: `Out of memory: Killed process 2097902 (python) total-vm:33.7GB anon-rss:13.2GB global_oom`.
  **Mechanism — CORRECTED 2026-06-08:** my first claim ("RAM accumulates ~2.6 GB/trial across the
  sweep") is **RETRACTED — it's wrong.** Counter-evidence: `pink_pirate` (a *healthy* model) ran dozens
  of sweep trials over hours without OOM, so sweep trials DO free memory between them; a sweep does not
  use more RAM than the single run it's made of. The real distinguishing factor: **this model EXPLODES
  (C-113)**, and it died **mid-eval** (posterior sampling) where preds hit ~1e11 (`expm1`→inf). Leading
  (UNVERIFIED) hypothesis: the explosion balloons a single trial's eval memory past the limit — an OOM
  that is a **symptom of C-113**, not a sweep property. (R1, a single exploding run, survived its
  explosion, so it's not deterministic per trial — depends on blow-up severity.) NOT a CUDA wedge, NOT
  an in-process crash (external SIGKILL) — a true RAM OOM, cause not yet measured.
  **Salvaged (clean evals): R1, R2, R3 only** — trials 4–5 trained but their eval didn't complete/flush
  before the kill (they evaluate *after* training; sweep trials DO eval — earlier "train-only" was just
  mid-eval). **Sweep produced NO new eval rows beyond R1–R3.**
  **DECISION: not auto-restarted** — a fresh `-s` re-runs 1→9 and re-OOMs at ~trial 5 (infinite re-fail
  loop unattended). → registered as a risk (C-135). **Morning options:** (a) run trials individually
  (`-t -e` per config) so RAM frees between trials; (b) fix the cross-trial RAM accumulation (cf. ADR-047
  streaming `del`+`gc.collect`); (c) lower `n_posterior_samples`; (d) cap the sweep grid to ≤3–4 combos
  per process. The OOM is itself a finding: **the explosion isn't just a metric artifact — it inflates
  eval memory enough to kill the process.**

**Confirmed findings (3 clean evals, R1–R3): C-113 reproduces, SS-independent; the exploding head varies
(ns/sb), `os` always healthy; magnitude ~1e8–1e11; and the explosion is severe enough to OOM eval.**

- **2026-06-08 (pm) — collapse baseline + sweep-crash correction + magnitude-calibration dossier opened.**
  Today's **40-lesson** calibration run (`calibration_model_20260608_165326.pt`, eval `ffldgbxf`)
  **COLLAPSED**: MCR ≈ 0.002–0.03 (predicts ~0 everywhere); CRPS small (sb 0.13 / ns 0.05 / os 0.04) **only
  because ~95%+ cells are zero** — CRPS rewards the collapse. This is the **opposite** mode to R1–R3
  (80-lesson, explode) → framed as **two distinct failure modes**: collapse = zero-inflation reward;
  explosion = no rollout training. Verified (git): **no probability-coupled hurdle was ever implemented**
  (only C-45 ground-truth masking `aba45bc` + Tobit censoring `56194d2`; neither couples magnitude to
  predicted probability). New program → `reports/2026-06-08_magnitude_calibration_dossier/` (candidate #1 =
  minimal hurdle; judge on twCRPS+Coverage, MCR diagnostic only).
  **Sweep-crash correction (distinct from the ~01:09 RAM OOM above):** the 2026-06-08 *afternoon* sweep
  failures (`ikvegy30`/`3vidg2tr`/`4atlhfw0`/`x9zk3ujt`/`ksubmpw3`, ~14:3x) were a **CUDA `unspecified launch
  failure` — kernel `Xid 62` → `Xid 45`** at `model.to(device)` (`make()`), i.e. a **GPU-context fault**
  (one long-lived sweep process; trial 1 ran, then the context was poisoned — likely by a concurrent
  memory-intensive GPU job — so every later trial died identically). **Not** the RAM OOM logged at ~01:09;
  two different mechanisms. Fixed by `rmmod/modprobe nvidia_uvm`; a fresh single run trained + evaluated
  cleanly afterward. (This corrects/extends the ~01:09 entry: there were *two* distinct sweep-failure
  events on 2026-06-08.)

- **2026-06-09 — Arm-1 (hurdle, lognormal_nll, 40 lessons): magnitude UN-COLLAPSED one-step (directional); the rollout exploded.** *(Originally filed "FAIL → ZITD"; reframed 2026-06-09 — see the ⟳ note below.)*
  One variable vs the 40-lesson Tobit baseline (`loss_reg` tobit→lognormal_nll, sigma dict→0.9 scalar, +`hurdle_threshold=0`).
  Training completed HEALTHY; **eval FAILED — `views-evaluation` rejected the predictions ("Input contains infinity").**
  Verified from saved preds (`predictions_calibration_20260609_051916`, origin_0): **NOT a collapse — an autoregressive
  runaway.** Step-1 magnitudes are non-zero (sb 61 / ns 580 / os 91 — the hurdle *un-collapsed* the head vs baseline ~0.02),
  then the 36-step free-running rollout grows exponentially per step → expm1 → **INF by step ~13–15**. **Key finding:
  un-collapsing the magnitude head directly TRIGGERS the C-113 explosion — magnitude calibration and rollout stability are
  coupled; the head can't be fixed in isolation.** Per the binding stopping rule (dossier `2026-06-08_magnitude_calibration`)
  → **commit to ZITD/structural, no more loss tweaks.** ZITD is doubly motivated: its sub-exponential softplus link makes
  drift linear, dissolving exactly this expm1 explosion. Artifact `calibration_model_20260609_051916`.
  **⟳ Reframe 2026-06-09 (step-1 read; C-136 / M-R1 / M-R2):** the "FAIL → ZITD" verdict conflated two axes. A
  teacher-forced **step-1 `MCR_pos`** (rollout not engaged) shows the magnitude head moved **off ~0**: sb 0.11→**0.19**,
  ns 0.02→**1.29**, os 0.03→**0.73** vs Tobit. The explosion is purely the **untrained rollout** (Axis B). ⚠ **Direction,
  not values** — MCR is a diagnostic ratio (not a proper score); single-draw, 1 origin/seed, small n (131/59/50) → "un-collapsed"
  holds, "calibrated" does not. Proper-subset score + CI + 2nd seed = R4's readout (#93). **Revised next step:** hurdle +
  **rollout training** (cheap SS-middle probe → GTF), NOT a count-likelihood rebuild. Stopping rule intact (rollout ≠ a new loss).
  *(Numbers refined 2026-06-10 by the durable `scripts/mcr_readout.py` — see dossier `07`: the all-origins aggregate MCR is
  mean-dominated, the median positive cell still under-predicts, and sb barely moved. The step-1 figures above are the
  superseded origin-0 `/tmp` throwaway.)*

- **2026-06-10 — R4 (hurdle + scheduled sampling `ss_epsilon_max=0.5`, 40 lessons) → EXPLODED at eval (same as Arm-1).**
  The cheap rollout-training probe — SS-middle on the hurdle config, one variable vs Arm-1. Trained clean; eval failed
  "Input contains infinity". Durable readout (`scripts/mcr_readout.py`, `predictions_calibration_20260610_010843`):
  FULL-rollout MCR ≈ **3.4e33 / 3.1e33 / 7.5e33** (sb/ns/os) — indistinguishable from Arm-1; step-1 magnitude no better
  (sb 0.21→0.088). **Completes the scheduled-sampling bracket: 0.25→explode, 0.5→explode, 1.0→collapse — plain per-step
  SS is EXHAUSTED (proven, not assumed), even on the un-collapsed hurdle head.** → the real fix is **GTF / B1 rollout
  training (#78, cross-step gradients)** or the **count-likelihood head** (ZINB/hurdle-NB). Rollout dossier `07` EXP-02.

*(Append wins/lessons here as runs land — especially negatives and "looked-right-but-wasn't".)*
