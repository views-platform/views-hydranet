# 05 — Pre-analysis plan: EXP-1, the current-behavior rollout-skill curve (GPU-free)

**Pre-registered 2026-07-25, BEFORE looking at any h>1 number.** Locks predictions + falsifiers for the
first scored read. Follows the method-review (`02b`) + chair rulings (§6b). Metrics honor FAO-02 (no
twCRPS/PIT). This experiment is a **diagnostic of today's mean-feedback rollout**, NOT the deployed-skill
verdict (that is EXP-2, the ancestral arm — separately pre-registered later). Per C-218, no "deployed
skill" claim may be drawn from EXP-1.

## Hypothesis (H-EXP1)

The frozen lodestar scorer, applied per horizon `h=1..36` to the **already-persisted** (mean-feedback)
rollouts, will show the crps split degrading with horizon and crossing above the baselines at a **small**
horizon for the dense-feedback arms (the bloom), while the τ≥0.8 (th_gated) arm stays numerically bounded
but is **not** thereby skillful at long h (STABILITY ≠ SKILL).

## Intervention (the ONE variable)

**Horizon.** We change nothing about the model, the feedback, or the emitted cubes — we score the existing
`origin_*` rollout data at **all** horizons instead of only h=1 (the lodestar's `sel = t == m0`). No GPU,
no re-inference. (Feedback content is EXP-2's variable, not this one.)

## Scope / arms / support

- **Arms (existing on-disk calibration rollouts):** the nb foundation + the composition arms
  (self_zeroed/ZINB, soft_gate/gated_NB, threshold_gate/th_gated_NB) from the lodestar + S8 evals. *Loader
  step 0 enumerates the exact persisted (arm, seed) set; any arm without a persisted 36-horizon dir is
  listed as absent (no silent omission).*
- **Baselines (same support, per horizon):** climatology (white_ranger, horizon-flat), persistence
  (`truth[o]` held), and the **mixture baseline** (red/green/yellow_ranger).
- **Support:** 13 rolling origins (T=0 = 457–469), full 36 horizons each, all inside the held-out
  calibration test window 457–504 (C-217 cleared). Identical (origin, cell) support across horizons.
- **Metrics per (arm, target, h):** **crps_all / crps_events / crps_none** split + **Brier / MCR / QS99**
  (locked guardrails). Size-ratio per h. **NO twCRPS, NO PIT, NO LogScore.** CRPSS (vs climatology) computed
  only for the crossover plot, never a decision metric.
- **Seeds:** 42/43/44 where persisted; any KEEP/ranking claim requires ≥3 seeds (single-seed = INDICATIVE).
- **CIs:** **block bootstrap over origins** (never iid over cells) — |O|=13 with overlapping futures.

## Skepticism ledger (what could make this misleading)

1. The scored rollout is **mean-feedback = broken by construction** (C-218): its curve is a *lower bound* on
   achievable skill, NOT the deployed rollout. No "the model can/can't roll out" conclusion here.
2. **crps_all is zero-dominated** (C-219): read the split; a low crps_all can be pure crps_none (timid zeros).
3. **STABILITY ≠ SKILL:** a bounded τ≥0.8 curve is not evidence of accuracy.
4. **|O|=13, autocorrelated** (C-221): long-horizon CIs will be wide; the crossover may be imprecise.
5. **Scored object must be the D×K cube, not `E[y]`** (C-220) — guarded by a loader test.

## Pre-registered predictions (before looking)

- **P1 (faithfulness):** for every arm, the **h=1** crps split == the frozen lodestar T=0 numbers to ≥4 dp
  (e.g. nb foundation sb ≈ 0.137; gated_NB ≈ 0.138; ZINB ≈ 0.141).
- **P2 (bloom as skill collapse):** the dense-feedback arms (nb soft_gate, ZINB) degrade **monotonically**
  in crps_all/events with h and cross above climatology at a **small** h_x (single digits).
- **P3 (τ bounded but not skillful):** the th_gated (τ≥0.8) arm's crps stays bounded across h (per
  `bloom_investigation`) BUT does **not** beat climatology at large h (its long-h skill ≈ climatology or
  worse) — bounded ≠ skillful.
- **P4 (short-h dominance):** every arm beats persistence AND climatology at h=1 (they beat white_ranger at
  T=0 per the lodestar), then decays.
- **P5 (mixture ≥ climatology):** the mixture baseline is ≥ climatology at every h (stronger reference).

## Pre-committed falsifiers

- **F1 (loader bug — HARD STOP):** any arm's h=1 crps split ≠ the lodestar T=0 number ⇒ the per-horizon
  loader is wrong. STOP and fix before any other reading. (No skill claim on a broken instrument.)
- **F2 (bloom absent):** if a dense-feedback arm does **not** degrade with h (stays flat/skillful to h=36),
  that contradicts `bloom_investigation` — flag as a surprising result needing a mechanism, not a win.
- **F3 (τ is genuinely skillful):** if th_gated (τ≥0.8) **beats climatology at long h** (not just bounded),
  that is genuine long-horizon skill from the conservative-zero rollout — a major, decision-changing
  positive that would reframe the epic (the "timid but stable" read would be wrong). Record loudly.
- **F4 (baseline inversion):** if persistence or the rollout beats the mixture baseline at long h on
  crps_events, re-check the baseline construction (a too-weak reference would inflate apparent skill).
- **F5 (determinism):** if re-running the scorer changes any number, the S2 #121 gate is violated — fix
  before trusting.

## Method

1. **G1 loader (TDD):** `gather_all_horizons` (index by h = month − origin) + `rollout_skill_score.py`
   wrapping the frozen `crps_ensemble`/AP/Brier verbatim. Red tests: h=1 == `gather_t0` byte-exact (F1);
   scored object is the cube not the mean (C-220); a synthetic fixture scores as hand-computed.
2. **Guard:** assert each arm's artifact `config.json` partition is **calibration** (train 121–456), not
   validation (C-217 residual guard).
3. **Score** all persisted arms + the 3 baselines at all h on identical support; bank per-(arm,target,h)
   crps split + guardrails + size-ratio, with block-bootstrap CIs over origins.
4. **Read-out (only after the above run):** skill-vs-horizon per target; locate crossover h_x per arm;
   plot CRPSS vs climatology for communication. Compare against P1–P5 / F1–F5.

## Decision rules

- **F1 fires ⇒ STOP** (fix loader), re-run, then proceed.
- The result **informs but does not decide** deployed skill (C-218). It quantifies *how bad today's
  mean-feedback rollout is* and sets the reference for EXP-2 (ancestral) and EXP-3 (oracle gap).
- **F3 fires ⇒** escalate immediately: the conservative-zero rollout may already be a usable long-horizon
  product; re-scope before building fixes.
- Otherwise: proceed to EXP-2 (pre-register the ancestral sample-feedback rollout = the deployed-skill
  verdict) and EXP-3 (the teacher-forced one-step-conditioned ceiling = the bug-vs-ceiling gap).
