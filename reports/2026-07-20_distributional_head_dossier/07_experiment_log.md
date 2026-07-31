# 07 — Experiment log (append-only; negatives first-class)

Each entry links its pre-registration (`05_analysis_plan`), names the ONE variable, records the readout,
states the verdict against the pre-registered falsifiers (which fired / none), and the decision.

---

## (seed) 2026-07-20 — M0 scaffold
- Pre-reg: `05_analysis_plan` locked.
- Variable: none (setup).
- Readout: dossier created; red TDD tests pending; ground truth verified (`02_design`).
- Verdict: n/a. Decision: proceed to red tests, then M1 smoke on user go-ahead.

## (seed) 2026-07-20 — M0 red tests
- Pre-reg: n/a (TDD scaffolding).
- Variable: none.
- Readout: `tests/test_nb_dist_head.py` — 12 tests, all RED for the right reasons (config rejects
  nb/zinb; `n_head_samples` field missing; head emits 1 ch/target not 2/3; `nb_dist_loss` module
  absent). Contract pinned: per-cell (mu,theta[,pi]) activated head channels; NBDistLoss/ZINBDistLoss
  gradient reaches per-cell theta/pi; nb_to_samples/zinb_to_samples → (…,K) counts; D×K = S.
- Verdict: n/a. Decision: implement M1 (config → head → loss → sampler), turning tests green.

## (seed 42/43/44) 2026-07-24 — ZINB 3×300 (self-zeroed body) — M2 result
- Pre-reg: `05_analysis_plan` §Forecast-composition arms, arm 1 (ZINB self-zeroed = `(1−π)μ`, no ×gate).
- Variable: body family **foundation (global-θ point body) → per-cell ZINB head (D×K sampled)**; the C-212
  `log_prob_zero` NaN fix (`b5a87d5`) was in place, all 3 seeds trained clean (was 2/3 crashing pre-fix).
- Readout (frozen ruler, T=0, count-only, N=170430):

  | target | crps-all ZINB | foundation | white_ranger | size-ratio |
  |--------|--------------:|-----------:|-------------:|-----------:|
  | sb | 0.141 | 0.137 | 0.191 | 0.02 → **0.25** |
  | ns | 0.084 | 0.083 | 0.088 | (timid → moved) |
  | os | 0.040 | 0.028 | 0.030 | — |

  Seed-stable across 42/43/44. The **timid body is fixed** (size-ratio 0.02→0.25). crps-all is at
  parity with the foundation on sb/ns and beats white_ranger on sb/ns; **os is the localized weak
  spot** (0.040 > both foundation 0.028 and WR 0.030).
- Verdict vs falsifiers: **none fired.** F1 no θ/π collapse, no seed-instability, sampler non-degenerate.
  F2 not triggered (crps-all not worse on ≥2 targets vs foundation — sb/ns parity). F3 deferred (M3
  validation). **NOT a KILL.** Decision: keep ZINB as the crps-all/magnitude front-runner; os under-fit
  noted (localized, not global). Proceed to the composition-arm head-to-head.

## (seed 42/43/44) 2026-07-24 — gated_NB (soft cls-gate re-score) — composition arm 2
- Pre-reg: `05_analysis_plan` §arm 2 (forecast = NB body × cls occurrence gate; pure re-score of the
  preserved nb 3×300 cubes, ZERO extra GPU).
- Variable: composition **ZINB self-zeroed → gated_NB (nb body × gate)** — same underlying head family
  question, different zero mechanism (external gate vs structural π).
- Readout (frozen ruler, T=0):

  | target | crps-all | AP |
  |--------|---------:|---:|
  | sb | 0.159 | **0.447** |
  | ns | 0.091 | **0.385** |
  | os | 0.046 | **0.259** |

  ZINB wins **crps-all** on all 3 targets; gated_NB wins **AP** (locality/occurrence) on all 3. A clean
  **magnitude (ZINB) vs locality (gated_NB) tradeoff** — the sharp cls gate localizes better; the
  structural-π body scores the count magnitude better.
- Verdict: no falsifier; both arms viable on different axes. Decision: the tradeoff motivates testing
  whether the two axes **fuse** → build + score gated_ZINBcore (arm 4).

## (seed 44) 2026-07-24 — gated_ZINBcore — composition arm 4 — **NEGATIVE (hypothesis falsified)**
- Pre-reg: ADR-068 + `05_analysis_plan` composition arms. **Pre-committed prediction:** gated_ZINBcore
  (ZINB's π-stripped NB core × external cls gate) fuses **ZINB's crps-all** (better body magnitude) with
  **gated_NB's AP** (sharp gate) → best-of-both. Code: `sample_core` (drops π) + `emit_family_core` flag.
- Variable: zero mechanism for the ZINB **core** — **structural π → external cls gate** (π dropped, not
  stacked; avoids the `(1−π)μ × gate` double-count). Single-seed proof (seed 44), same basis as the
  ZINB/gated_NB comparison.
- Readout (frozen ruler, T=0):

  | target | crps-all | (ZINB) | (WR) | AP | crps-none |
  |--------|---------:|-------:|-----:|---:|----------:|
  | sb | **0.981** | 0.141 | 0.191 | 0.438 | 0.870 |
  | ns | **0.488** | 0.084 | 0.088 | 0.389 | 0.415 |
  | os | **0.462** | 0.040 | 0.030 | 0.256 | 0.440 |

- Verdict vs falsifiers: **F2 FIRED** (crps-all worse than everything on all 3 targets — 5–15× ZINB, and
  worse than the white_ranger baseline). It DID take gated_NB's AP (0.438 ≈ 0.447, same gate, expected)
  but crps-all **exploded**. **Worst-of-both, not best-of-both.** (Precision: the frozen ruler's crps-all
  is **gate-independent** — computed on the raw body samples; the gate feeds only AP/Brier. So 0.981 is the
  *ungated* ZINB-core body, crps-none-dominated; a soft external gate cannot touch crps-all by construction.
  See `postmortem_gated_zinbcore`.) Decision: **KILL the arm.** Do NOT spend
  3-seed compute — the failure is structural (see postmortem), so seed variation cannot rescue it. → see
  `postmortem_gated_zinbcore.md`.

## (seed 42/43/44) 2026-07-24 — EMIT-TIME composition re-eval — Epic #183 S8 (#191) — **PASS**
- Pre-reg: `05_analysis_plan` §"Pre-registration — EMIT-TIME composition re-eval" (ADR-069). The
  composition-arm numbers above were a *score-time* re-score that never actually composed `gate × body`
  (the ruler's count-CRPS is gate-independent). Epic #183 made composition a real config axis applied
  **inside the model at emit time**; this re-scores the three arms from the MODEL's composed output.
- Variable: composition moved from **score-time re-score → emit-time (in-model)**. Eval-only re-inference
  of the existing nb ×3 + zinb artifacts (NO retrain); scored count-only on the frozen ruler (composition
  baked into the cube). Stealth trap-restore, floor md5 verified before+after.
- Readout (crps-all, mean of seeds 42/43/44):

  | arm | sb | ns | os | vs banked |
  |-----|---:|---:|---:|-----------|
  | **gated_NB (real soft_gate)** | **0.138** | **0.080** | **0.031** | banked ungated 0.159/0.091/0.046 — MOVED DOWN |
  | **th_gated_NB (real τ=0.5)** | **0.141** | **0.079** | **0.031** | banked score-time 0.139/0.080/0.031 — REPRODUCED |
  | ZINB (self-zeroed, unchanged) | 0.141 | 0.084 | 0.040 | passthrough (unit-parity proven) |
  | foundation / white_ranger | 0.137/0.191 | 0.083/0.088 | 0.028/0.030 | — |

- Verdict vs the pre-registered F-EMIT falsifiers: **none fired.**
  - F-EMIT-1 (th_gated differs >5% from score-time): NO — within ~1.5% on all targets (implementation
    faithful; the emit-time hard threshold = the score-time hard threshold on the same gate/body).
  - F-EMIT-3 (gated_NB crps-none *rises*): NO — it DROPPED (~0.041 → ~0.015), exactly as predicted:
    the real per-draw `Bernoulli(gate)` zeros the body on low-gate cells; the score-time re-score never
    gated the body at all.
  - Prediction 1 (gated_NB moves down): CONFIRMED. Prediction 2 (th_gated reproduces): CONFIRMED.
    Prediction 3 (ZINB byte-identical): held by construction (self_zeroed passthrough + S2 unit parity).
- **Decision / headline:** the three arms are now REAL, honestly-composed model outputs (T=0 calibration,
  seed-stable). **"th_gated_NB is the uniquely strongest arm" is FALSIFIED** — properly composed,
  **gated_NB ≈ th_gated_NB** (gated_NB a hair better on sb: 0.138 vs 0.141), because the score-time
  re-score badly undersold gated_NB (it never applied the per-draw gate). Both beat ZINB on ns+os, tie
  on sb; all three still lose the foundation on sb/os and win ns (the arms are occurrence plays, not a
  magnitude fix). AP is gate-head (composition-independent) ~0.447/0.385/0.259 for both.
- Housekeeping: the nb calibration cubes now hold the last-composed (threshold_gate) samples — the banked
  numbers are preserved here; the `.pt` artifacts are intact. Scope respected: T=0 calibration, eval-only,
  no retrain, no M3 validation, no bloom. Epic #183 implementation+T=0 validation COMPLETE.

## (seed 42/43/44) 2026-07-24 — th_gated_NB — composition arm 3 — **POSITIVE (clears; strongest all-round)**
- Pre-reg: `05_analysis_plan` §arm 3 (hard cls-gate threshold: full nb body where `gate ≥ τ`, zeroed
  where `gate < τ`; two FIXED a-priori τ = 0.5 and per-target base rate). Gate to run was met: the ZINB
  vs gated_NB split confirmed a real magnitude-vs-locality tradeoff (arm not moot).
- Enabling work: the frozen lodestar scorer computes crps-all on the raw body samples (gate-independent),
  so a hard-threshold body composition had to be **added** to the ruler. Done TDD, **byte-identical** for
  all existing arms (HEAD vs extended scorer max|Δ| = 0.00e+00), `--selftest` re-frozen (`apply_threshold_gate`).
- Variable: composition **soft gated_NB → hard th_gated_NB** (τ zeros the body); pure re-score of the
  preserved nb 3×300 cubes, ZERO GPU. τ pre-registered a-priori (no Goodhart).
- Readout (frozen ruler, T=0, mean of seeds 42/43/44; per-seed rock-stable):

  | target | arm | crps-all | AP | crps-events | crps-none |
  |--------|-----|---------:|---:|------------:|----------:|
  | sb | gated_NB soft | 0.159 | 0.447 | 15.62 | 0.038 |
  |    | **th_gated_NB@0.5** | **0.139** | 0.447 | 15.73 | **0.017** |
  |    | ZINB (banked) | 0.141 | — | — | 0.042 |
  | ns | gated_NB soft | 0.091 | 0.385 | 22.20 | 0.014 |
  |    | **th_gated_NB@0.5** | **0.080** | 0.385 | 22.35 | **0.003** |
  |    | ZINB (banked) | 0.084 | — | — | — |
  | os | gated_NB soft | 0.046 | 0.259 | 6.24 | 0.020 |
  |    | **th_gated_NB@0.5** | **0.031** | 0.259 | 6.30 | **0.005** |
  |    | ZINB (banked) | 0.040 / WR 0.030 | — | — | — |

  th50 beats soft gated_NB on crps-all on **all 3 seeds × 3 targets**; **≥ ZINB** (ties sb, wins ns+os —
  ZINB's weak targets); keeps gated_NB's **AP** (0.447/0.385/0.259 ≫ WR 0.334/0.223/0.158). The fusion
  gated_ZINBcore failed at. τ=**baserate** ≈ no-op (base rates ~0.4–0.8% too low to zero anything).
- Verdict vs falsifiers: **none fired.** The crps-all win is entirely **crps-none** (−55…−78%, confident
  zeros on true-zero cells); **crps-events flat-to-+1%** (few real events below τ=0.5). Honest limits:
  th50 is better at aggregate score + occurrence, **not** at sizing — **size-ratio DROPS** (sb 0.29→0.13),
  so the prereg's "th_gated wins size-ratio" expectation is **falsified** (the body was already unshrunk
  in this gate-independent ruler; thresholding only adds zeros, incl. on false-negative events). **os still
  loses crps-all to white_ranger** by a hair (0.0305 vs 0.0299) and on crps-events. Decision: th_gated_NB
  @ τ=0.5 is the **strongest all-round arm** (best crps-all + AP, seed-stable) — candidate for the M3
  validation-partition graduation. Its edge is decisive occurrence, not magnitude.


## (seed 42/43/44) 2026-07-25 — BLOOM investigation (T>0) — first pass — **see `bloom_investigation.md`**
- Pre-reg: none formal (exploratory diagnostic; the sample-feedback probe IS pre-registered in
  `plan_bloom_fix_sparse_feedback.md` §NEXT). Eval-only re-inference, calibration partition, stealth.
- Variable: the AR-feedback composition (soft_gate / threshold_gate τ ∈ {0.5,0.8,0.9} / ZINB self_zeroed),
  measuring the per-step (T=0..35) rollout magnitude + gate trajectory (13-origin mean).
- Readout (count/cell @T=35): soft **29e9** · τ0.5 6.5e3 · **τ0.8 0.3** · **τ0.9 0.1** · ZINB **2.8e9**.
  T=0 crps-all (3 seeds): th_gated @ τ≥0.8 **beats the foundation** on sb+ns, ties os (via crps-none
  collapse — a decisive-zero win; crps-events slightly WORSE, AP lower).
- Verdict: **sparsity is the lever** (confirmed 3 seeds for stability). **ZINB blooms too** — its learned
  π decalibrates in rollout like the classifier gate; only a HARD rule (τ) held. `feedback_clamp_log1p`
  was **inert** (byte-identical with/without — cause unknown, registered).
- ⚠️ **Load-bearing caveats (do not drop):** (1) **STABILITY ≠ SKILL** — we do NOT score the T>0 rollout
  vs truth; a τ=0.9 rollout is bounded partly because it predicts ~nothing. (2) bloom cases are s44-only.
  (3) calibration partition, T=0-scored, not M3-validated. (4) "bloom is a symptom / sample-feedback is the
  fix" is an INTERPRETATION + an UNTESTED hypothesis.
- Decision: τ logged as a **tool, not a solution**. Next = the pre-registered **sample-feedback rollout**
  probe (feed back a draw, not the mean). Full detail + epistemic table in `bloom_investigation.md`.

---

## EXP — body_mask magnitude sweep (NB, seed 42, 40 lessons) — pre-reg `08_magnitude_bodymask_prereg.md`
**Date:** 2026-07-27 · **One variable:** `body_mask ∈ {none, pos_cells, pos_timelines}` (NB head) ·
**Driver:** `mag_sweep.sh` (3 trains + 9 emit-and-score) · **Results:** `results/mag_bodymask/` ·
**Ruler:** frozen lodestar + `sharpness_scorecard` (C-167). Floor md5 intact; no F-DEGEN.

**Grid — composition `threshold_gate τ=0.5` (the deliverable arm), sb / ns / os:**

| arm | crps_all | crps_events | crps_none | size_ratio | STEP-1 FSS@1 | STEP-1 MCR | STEP-1 area_ratio |
|-----|----------|-------------|-----------|------------|--------------|------------|-------------------|
| none (baseline) | 0.140 / 0.082 / 0.030 | 18.10 / 23.72 / 6.64 | 0.0002 / 0.0007 / 0.003 | 0 / 0 / 0 | 0.01 / 0.00 / 0.00 | 0.006 / 0.002 / 0.012 | 0.0× / 0.0× / 0.1× |
| **pos_cells** | 0.267 / 0.132 / 0.097 | **16.79 / 23.59 / 7.05** | **0.138 / 0.051 / 0.068** | 0.167 / 0 / 0 | **0.21 / 0.19 / 0.11** | **0.69 / 0.65 / 0.85** | 4.4× / 2.7× / 2.9× |
| pos_timelines | 0.146 / 0.090 / 0.030 | 18.29 / 23.73 / 6.87 | 0.005 / 0.008 / 0.001 | 0 / 0 / 0 | 0.01 / 0.00 / 0.00 | 0.26 / 0.09 / 0.30 | 0.3× / 0.3× / 0.1× |

(ungated τ=1e-6 control: `pos_cells` crps_all **0.480 / 0.577 / 0.413** — 2.8–5× the baseline; the old
pre-#183 negative reproduced exactly.)

**Verdict against the pre-registered falsifiers:**
- **Magnitude is RECOVERABLE — the timid body is a supervision artifact, NOT a wall.** `pos_cells`
  un-collapses the body: STEP-1 MCR 0.006→**0.69/0.65/0.85** (near 1), FSS@1 0.00→**0.21/0.19/0.11**,
  size_ratio 0→0.167 (sb). (Prediction 1 ✅.)
- **NOT smearing (F-MAG-1 does NOT fire):** FSS *improved* (0→0.2–0.8) — the C-167 guard says `pos_cells`
  is genuinely better-*localized*, not a diffuse blob. The problem is the opposite of smearing.
- **F-MAG-2 FIRES:** gated (τ=0.5) `pos_cells` still blows composed crps-all vs `none` (0.267 vs 0.140
  sb; +60–220%) because it **over-fires on gate-retained cells** — pos_mcr 2.9–5×, area_ratio 3–4×
  (STEP-1) → 43–50× (FULL), crps_none 0.0002→0.138 (~550×). The gate halves the ungated blowup
  (0.480→0.267 sb) but cannot rescue over-cooked magnitude on cells it *keeps*. (Prediction 2 ✅, 3 partial.)
- **`pos_timelines` is timid** (size_ratio 0, FSS ~0, MCR 0.09–0.30) — no lift; wrong-direction control
  reproduces the banked negative. (Prediction 4 ✅.)

**Decision (1-seed screen — advance, no kill):** the mask direction is SOUND (un-collapses magnitude +
improves localization, no smearing) but **over-shoots** (magnitude + area too big). This is exactly the
setup the robust-trendline targets: keep the un-collapsing, tame the overshoot by training the masked
body on a robust central level (running mean/median) instead of the raw spike magnitudes → drive
pos_mcr/area_ratio toward 1 while holding FSS. **NEXT LEVER = robust-trendline target on `pos_cells`.**
Caveats: 1 seed (BN-recal mitigates, not eliminates); FULL-subset `max` EXPLODED flag fires on ALL arms
incl. the well-behaved `none` baseline ⇒ a scorecard max-column quirk, not discriminating (STEP-1 is the
clean read). twCRPS/PIT not used (FAO-02).

---

## EXP — body_supervision window sweep (NB × {all,0/0,0/2,2/2,0/6} × {42,43,44}, 40 lessons) — pre-reg `09`
**Date:** 2026-07-28 · **Driver:** `mag_sup_sweep.sh` (18 trains + 33 emit-and-score, watched, resumable) ·
**Results:** `results/sup_window/` · **Ruler:** frozen lodestar + tail_scorecard + sharpness. Floor md5 intact;
0 F-DEGEN; ran clean once the machine's swap-exhaustion was cleared (peak trainRSS 3.46 GB, no balloon).

**Grid — composition `threshold_gate τ=0.5`, mean over 3 seeds, sb / ns / os:**

| arm (onset,cess) | crps_all | crps_events | crps_none | size_ratio |
|---|---|---|---|---|
| `all` (foundation) | **0.143 / 0.084 / 0.030** | 18.15 / 23.69 / 6.67 | 0.0025 / 0.0020 / 0.0022 | 0 / 0 / 0 |
| `active 0,0` (=pos_cells) | 0.220 / 0.139 / 0.089 | **16.22** / 22.56 / 6.84 | 0.095 / 0.062 / 0.061 | 0.098 / 0 / 0 |
| `active 0,2` (decay) | 0.195 / 0.111 / 0.062 | 17.67 / 23.06 / 6.85 | 0.059 / 0.032 / 0.034 | 0 / 0 / 0 |
| `active 2,2` (sym) | 0.183 / 0.112 / 0.056 | 17.92 / 23.26 / 6.65 | 0.044 / 0.032 / 0.029 | 0 / 0 / 0 |
| `active 0,6` (long decay) | 0.177 / 0.103 / 0.047 | 18.11 / 23.96 / 6.64 | 0.037 / 0.020 / 0.019 | 0 / 0 / 0 |

**Verdict against the pre-registered falsifiers:**
- **F-SUP-1 FIRES — no active window beats `all` on composed crps-all.** crps-all is *monotone* in window
  size: `all 0.143 < a06 0.177 < a22 0.183 < a02 0.195 < a00 0.220` (identical ordering on ns, os). The timid
  foundation wins outright. (Prediction 1 FALSIFIED.)
- **Prediction 2 CONFIRMED — the window cuts the true-zero over-cook, monotonically:** crps_none `a00 0.095 →
  a02 0.059 → a22 0.044 → a06 0.037`. The boundary supervision *does* anchor the near-activity zeros.
- **…but only by RE-TIMIDIFYING.** As the window grows, crps_events climbs back to the foundation
  (`a00 16.22 → a06 18.11 ≈ all 18.15`) and size_ratio collapses to 0. So `body_supervision=active` is a single
  **dial between "un-collapsed-but-over-cooked" (a00) and "timid-but-clean" (all)** — every interior point sits
  on that line; there is **no sweet spot** that keeps the magnitude while killing the over-cook. It is the
  magnitude-XOR-calibration wall, now isolated on a clean sweepable axis.
- **F-SEED does NOT fire — seed-stable:** rankings hold across 42/43/44 (`all` always ≈0.14; `a00` worst;
  interior between). A robust negative, not a basin artifact.
- **Prediction 4 CONFIRMED — tail dead across ALL radii:** top-bin (truth 549) reach=0%, q90=0 for every arm.
  The window is orthogonal to the ξ≈0.8 tail, as scoped.
- ZINB×all reference: crps-all ≈0.16 (sb/ns), between NB-all and NB-pos_cells — no advantage here.

**Decision (per prereg 09):** F-SUP-1 fires ⇒ **the body-supervision region is NOT the lever for the
bulk-magnitude downward bias.** The knob is built + productionized (a clean negative, not wasted — `all` and the
endpoints are now one validated axis). **Pivot to the family/tail axis** — the persistence-anchored monotone
quantile-Δ head (D-13) and/or the heavy tail (C-149/C-224), exactly where the 6-seat method-review panel pointed.
The magnitude ceiling is the family, not where we supervise it.
