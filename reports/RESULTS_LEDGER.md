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

*(Append wins/lessons here as runs land — especially negatives and "looked-right-but-wasn't".)*
