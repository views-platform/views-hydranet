# 07 — Experiment log (append-only; negatives first-class)

Every run links its pre-registration (`05` / a `preanalysis_*`) and its verdict vs the pre-committed
falsifiers. No success-only drift. Empty until Phase 2's first scored read.

<!-- entries appended below, newest last -->

## EXP-1 — current-behavior rollout-skill curve (GPU-free) — 2026-07-25

**Pre-registration:** `05_analysis_plan.md` (P1–P5, F1–F5). **Instrument:** `tools/rollout_skill_score.py`
(imports the frozen lodestar primitives verbatim). **One variable:** horizon (re-score existing on-disk
rollouts at all h). **Support:** 13 origins × N=170,430, calibration test 457–504 (held out, C-217).
**Arms (single-seed INDICATIVE):** nb (`…102130`), th_gated_nb@baserate, zinb (`…063927`, clean), vs
climatology (white_ranger) + persistence. **Metrics:** crps_all/events/none split + AP + Brier + size_ratio
(frozen set; NO twCRPS/PIT/QS99). **Result:** `results/exp1_skill.csv` (540 rows).

### Faithfulness (F1) — PASS
`gather_all_horizons`@h=1 is **byte-identical** to the frozen `gather_t0`; crps delta **0.00e+00** on
sb/ns/os (h=1 nb = 0.134/0.079/0.028 == lodestar T=0). The hard-stop did not fire; the instrument is
faithful.

### Headline (sb, indicative) — STABILITY ≠ SKILL, demonstrated with numbers
| model | crps_all h1/h12/h36 | crps_events h1/h12/h36 | crps_none h1/h12/h36 | size_ratio | AP h1→h36 |
|---|---|---|---|---|---|
| **nb** | 0.134 / 0.112 / 0.883 | 16.9 / 13.2 / 84.5 | 0.003 / 0.007 / 0.010 | **0.0** (all h) | 0.44 → 0.16 |
| climatology | 0.191 / 0.174 / 0.960 | 19.2 / 15.2 / 86.4 | 0.042 / 0.053 / 0.068 | ~0.1–0.4 | 0.33 → 0.19 |
| zinb | 0.139 / **3.93** / **5.41** | 15.5 / 13.6 / 83.9 | 0.020 / **3.85** / **4.59** | 1.5 → 3.9 | 0.31 → 0.02 |
| persistence | 0.187 / 0.207 / 0.984 | 19.1 / 17.4 / 87.3 | 0.040 / 0.069 / 0.083 | 0.0 | 0.14 → 0.08 |
(ns/os same qualitative story; ns nb crps_all *decreases* 0.079→0.029 with h — pure timid-zero, crps_events ~12–23.)

**Reading (honest):**
1. **nb "beats climatology on crps_all" is the C-219 Goodhart trap, not skill.** crps_all ≈ crps_none (99.7%
   zeros); nb wins purely by being *more confidently zero* (crps_none 0.003 vs 0.042). On events it has
   **no magnitude skill at any horizon** — crps_events ~13–17, and **size_ratio = 0.0** (the median forecast
   on event cells is literally zero). The bounded nb rollout is **timid-zero**, not accurate. Had we
   headlined crps_all (as the pre-reg forbade), we would have falsely crowned it. The split caught it.
2. **nb has REAL occurrence skill (AP) only at short horizons** — AP 0.44 (h1) beats climatology (0.33),
   decaying to parity by ~h15 (nb 0.23 vs clim 0.26 @ h18) and below after. AP is rank-based (zero-immune),
   so this is a genuine, non-Goodhart signal: the rollout locates conflict better than climatology for
   ~1 year, then loses it.
3. **zinb blooms** — crps_none explodes 0.02→4.6 (decalibrated π smears positive mass onto true-zero
   cells); size_ratio grows to ~3.9 (over-fires). Its crps_events (~14) is no better than nb's, so ZINB has
   no event skill AND destroys its zeros. Confirms the `bloom_investigation` ZINB runaway on the frozen
   ruler.
4. **h=36 spike is REAL data, not an artifact:** crps_events jumps to ~84–87 for ALL models at h=36 (a
   large true event in months 492–504); it hits climatology too, so it is a truth feature, not a bug.

### Verdict vs pre-committed falsifiers
- **F1 (loader HARD STOP):** did NOT fire — h=1 byte-exact to lodestar. ✅
- **F2 (bloom absent for a dense arm):** **FIRED for nb** — the nb mean-feedback rollout does NOT run away
  (crps_all bounded). *Explained, not a win:* nb is timid (size_ratio 0), so its fed-back mean stays small
  and in-distribution → no runaway. **Reconciles the `bloom_investigation` "soft_gate → 29 billion" (s44):**
  the bloom depends on the feedback *composition* — the timid plain-nb mean stays bounded; the dense
  soft_gate×mean (and zinb's decalibrated mean) bloom. So "does it bloom?" is feedback-content-dependent.
- **F3 (τ genuinely skillful):** NOT tested — I used th_gated@**baserate** (τ≈0.008), too low to zero
  anything, so th_gated_nb == nb byte-identical. The bloom-bounding τ≥0.8 is a *feedback* threshold, not a
  score-time one; testing it needs the τ-feedback rollout (EXP follow-up), NOT this score-time composition.
- **F4 (baseline inversion):** none — persistence < climatology on crps_all, as expected.
- **F5 (determinism):** pure re-score, deterministic.

### Decision (per the pre-reg — diagnostic only, C-218)
EXP-1 **validates the ruler** (the split unmasked the timid-zero Goodhart trap that crps_all alone would
have hidden) and gives the first honest per-horizon picture: **the current mean-feedback rollouts have no
magnitude/event skill at any horizon** (nb timid-zero; zinb over-fires + blooms); **nb has real occurrence
skill only to ~h15**. This is the BROKEN mean-feedback object (C-218) — **not** the deployed-skill verdict.
**Next:** EXP-2 = the ancestral (sample-feedback) rollout (the true deployed object; needs the H-SAMPLE
re-run) + EXP-3 = the teacher-forced one-step ceiling (the bug-vs-ceiling gap). **Follow-ups noted (not
silently dropped):** mixture baseline (per-target `lr_ged_*` template); 3-seed + block-bootstrap CIs; the
τ≥0.8 *feedback* rollout (F3 proper test).

---

## EXP-2 — the ancestral (sample-feedback) rollout — 2026-07-26

**Pre-registration:** `05b_preanalysis_exp2_ancestral.md` (P1–P4, F-S1..F-S4). **One variable:** feedback
content (`rollout_feedback` = mean vs sample; ADR-flag, parity-proven). **Arm run:** zinb (`…063927`), the
bloom arm, single-seed s44 INDICATIVE. **A/B:** zinb sample-feedback (this run) vs zinb mean-feedback
(EXP-1, same artifact). **Result:** `results/exp2_zinb_sample.csv`. **nb NOT run** — the ADR-069 validator
rejects `nb + self_zeroed` (the old nb dir predates that validator; nb needs a declared gate + a
composition-aware feedback — deferred, see below).

### Headline: the bloom is a FIXABLE BUG (exposure bias), not intrinsic instability — but STABILITY ≠ SKILL
| metric (sb) | zinb **mean** (EXP-1) h6/12/24/36 | zinb **SAMPLE** (EXP-2) h6/12/24/36 |
|---|---|---|
| crps_all | 1.51 / 3.93 / 4.60 / 5.41 | **0.14 / 0.15 / 0.24 / 1.10** |
| crps_none | 1.42 / 3.85 / 4.52 / 4.59 | **0.04 / 0.06 / 0.12 / 0.24** |
| crps_events | 13.9 / 13.6 / 14.1 / 83.9 | 13.4 / 12.2 / 12.9 / 83.5 |
| size_ratio | 3.3 / 3.9 / 3.7 / 3.1 | 0.19 / 0.23 / 0.19 / 0.13 |
| AP | 0.05 / 0.02 / 0.02 / 0.02 | 0.19 / 0.14 / 0.08 / 0.06 |
(ns/os same shape: zinb-mean crps_none blooms to 4.06/2.05 @h36; zinb-SAMPLE stays 0.008/0.064.)

**Reading:**
1. **P1 CONFIRMED / F-S1 does NOT fire.** Feeding back a *sample* instead of the mean **eliminates the zinb
   bloom** — crps_none stays bounded (sb 0.02→0.24 vs mean's 0.02→4.59), a ~20× crps_all improvement at
   h24. The learned-π decalibration runaway is pure **exposure bias** from feeding back the diffuse mean;
   the sparse draw stays in-distribution. *The bloom is a fixable bug, not an intrinsic map instability.*
2. **STABILITY ≠ SKILL, again.** The now-bounded rollout is **not a better forecaster**: crps_events is
   tied with climatology/nb (~14 all h) — **no magnitude skill**; size_ratio ~0.2 (sb) / 0.0 (ns/os long-h)
   — modestly less timid than nb but no real event magnitude; AP falls below climatology by ~h6 (0.19 vs
   0.30). Sample-feedback fixes the *stability*; it does **not** buy *magnitude predictability*.
3. **The bug-vs-ceiling separation (the epic's core question), answered for zinb:** the runaway was a BUG
   (mean-feedback exposure bias — fixed); the residual "can't predict event sizes past a few horizons" is
   the **CEILING** (the amount-ceiling wall), and it is *independent of feedback method*. sample-feedback
   moves zinb from "catastrophic bloom" to "bounded ≈ climatology-magnitude, short-horizon occurrence".

### Verdict vs pre-committed falsifiers
- **F-S1 (also runs away):** did NOT fire — bounded. H-SAMPLE's core claim holds. ✅
- **F-S2 (still timid ⇒ body defect):** **effectively FIRES for magnitude** — the bounded rollout has no
  event-size skill (crps_events tied, size_ratio small). Consistent with F-S2's spirit: the residual is a
  **body/ceiling** matter, not a feedback one. The feedback fix is necessary (stops the bloom) but not
  sufficient for magnitude.
- **P2/P3 (nb less timid / event skill):** NOT tested (nb validator block).
- **F-S3 (determinism):** the flag is seeded + parity-proven (test suite); the eval rc=0.

### Decision
- **Sample-feedback is validated as the rollout-stability fix** (for the learned-π family). The bloom is
  demoted from "mysterious runaway" to "solved exposure-bias bug." This retires the τ-as-solution framing:
  a *sample* is the principled bounding mechanism (τ was its frugal proxy).
- **The long-horizon magnitude ceiling stands** — no feedback trick beats it; this is the predictability
  wall, not a bug. EXP-3 (teacher-forced oracle) would quantify exactly how much headroom remains, but
  EXP-2 already shows the bounded ancestral rollout sits ≈ climatology on magnitude.
- **Data-hygiene scar:** the eval OVERWRITES the cube dir in place (name = artifact ts), so the driver's
  new-dir capture returned empty AND the `…063927` dir now holds the zinb SAMPLE cube (the mean/original
  self-zeroed cube was overwritten; EXP-1's zinb scores are banked + the `.pt` is intact → regenerable via
  `rollout_feedback=mean`). Future drivers must write to a fresh `--output`/dir, not re-eval in place.

### Follow-ups (noted, not dropped)
nb ancestral (needs a declared gate composition + a composition-aware `_sample_feedback` so the mean/sample
A/B isn't confounded by gated-vs-ungated feedback); 3-seed + block-bootstrap; EXP-3 oracle for the exact
bug-vs-ceiling gap; a T>0 *skill* graduation on the validation partition.

---

## EXP-3 — the teacher-forced oracle — 2026-07-26 — bug-vs-ceiling, FULLY DECOMPOSED

**Pre-registration:** `05c`. **One variable:** `rollout_feedback='teacher_forced'` (feed the REAL month-t
input each step; zero exposure bias = the one-step-conditioned ceiling). Same zinb `…063927`, s44
INDICATIVE. **gap(h) = sample − oracle** = the exposure-bias cost; whatever the oracle can't do = the
ceiling. **Result:** `results/exp3_zinb_oracle.csv`. Floor md5 restored.

### The three-way verdict (sb, indicative; ns/os same shape)
| metric | oracle h6/12/24/36 | sample (deployed) | climatology | reading |
|---|---|---|---|---|
| **AP** | 0.31 / 0.30 / 0.31 / **0.31** | 0.19 / 0.14 / 0.08 / **0.06** | 0.30 / 0.28 / 0.25 / 0.19 | **BUG (exposure bias)** — huge, recoverable |
| **crps_none** | 0.019 flat | 0.04 → 0.24 | 0.05 → 0.07 | BUG (small residual smear) |
| **crps_events** | 12.5 / 11.3 / 12.2 / 82 | 13.4 / 12.2 / 12.9 / 83 | 17 / 15 / 15 / 86 | **CEILING** — oracle ≈ sample ≈ clim |
| **size_ratio** | 0.27 / 0.31 / 0.29 / 0.25 | 0.19 / 0.23 / 0.19 / 0.13 | 0.31 / 0.30 / 0.22 / 0.07 | **CEILING** — oracle ~0.3, NOT →1 |

### Reading — the epic's core question, answered
1. **OCCURRENCE (where) = a BUG, with LARGE headroom.** The oracle holds **AP flat ~0.30** across all 36
   horizons and beats climatology at long h; the deployed sample rollout **collapses 0.29→0.056**. The
   *entire* occurrence decay is exposure bias — with perfect inputs the one-step map keeps locating conflict
   as well at h36 as h1. **P2 CONFIRMED strongly.** A training fix that teaches recovery from own feedback
   (scheduled sampling / GTF, ADR-056) has ~0.06→0.30 AP headroom — a big, motivated next lever.
2. **MAGNITUDE (how big) = a CEILING.** Even with perfect inputs the oracle undershoots event sizes
   (size_ratio ~0.3, crps_events ~12–15, ≈ climatology) — perfect feedback gives only a modest lift, NOT
   size_ratio→1. **P1 CONFIRMED, F-O3 does NOT fire.** The amount-ceiling wall
   ([[project_amount_ceiling_wall]]) is intrinsic; no rollout/feedback trick recovers event magnitude.
3. **STABILITY = a BUG, already fixed** (EXP-2 sample-feedback). crps_none oracle flat 0.02.

**So the bloom decomposes cleanly:** numerical runaway = bug (fixed by sample-feedback); occurrence-skill
decay = bug (exposure bias, recoverable by GTF/scheduled-sampling training — the oracle proves the
headroom); event-magnitude = ceiling (irreducible). The deployed recursive rollout is bounded with ~1yr
occurrence skill today; a rollout-training fix could push occurrence skill much further; magnitude stays
capped regardless.

### Verdict vs falsifiers
- **F-O1 (oracle not better than sample):** did NOT fire — massive AP gap (0.31 vs 0.06 @h36) + crps_none
  gap. Exposure-bias headroom is real and large. ✅
- **F-O2 (oracle blooms):** did NOT fire — oracle bounded (crps_none flat 0.02, crps_all ≤ sample all h).
- **F-O3 (magnitude recoverable, size_ratio→1):** did NOT fire — oracle size_ratio ~0.3. Magnitude ceiling
  stands.

### Decision
- **Next lever is a rollout-training retrain (scheduled sampling / GTF, ADR-056)** — the oracle proves large,
  measurable OCCURRENCE headroom (AP 0.06→0.30). This is rung-3 of the old fix ladder, now *motivated by a
  measured gap*, not a guess. It stacks on the sample-feedback fix (feed back samples during training).
- **Magnitude is parked as the ceiling** — do NOT spend rollout effort chasing event sizes; that is the
  amount-ceiling wall (a features/DGP limit, not a rollout bug).
- Caveats: zinb only, s44, single-seed INDICATIVE, one-step-conditioned ceiling (C-222: gap ⊇ induced
  state-drift, not pure input-exposure-bias). Harden with 3 seeds + the nb arm before committing the retrain.
- **Data:** the `…063927` dir now holds the ORACLE cube (sample was there after EXP-2; both regenerable, all
  scores banked in `results/`). Same in-place-overwrite scar — future drivers must use fresh dirs.

---

## HARDENING — 3-model zinb + the nb arm — 2026-07-26

Composition-aware `_sample_feedback` (`6e089d3`) unblocked the nb arm (mean/sample now both soft_gate-composed
= clean one-variable A/B). 9 evals, score-after-each (`results/harden_*.csv`): 2 more zinb models
(`000250`, `032256`) + nb gated_NB (`102130`), each × {mean, sample, oracle}. Floor restored. NOTE: the
artifacts store no training seed → these are **3 independently-trained models**, not verified seeds 42/43/44.

### The three-way verdict — HELD, with two honest nuances (sb crps_all @ h24 / AP @ h24)
| model | mean crps_all | sample | oracle | mean AP | sample AP | oracle AP |
|---|---|---|---|---|---|---|
| zinb 000250 | 0.31 (no bloom) | 0.23 | 0.15 | 0.009 | 0.009 | 0.018 |
| zinb 032256 | **15.0 bloom** | 0.39 | 0.14 | 0.010 | 0.059 | 0.293 |
| zinb 063927 (s44) | **4.60 bloom** | 0.24 | 0.13 | 0.018 | 0.081 | 0.309 |
| **nb gated_NB** | **5.03 bloom** | 0.13 | 0.14 | 0.011 | **0.238** | **0.467** |

1. **STABILITY (bloom) — sample-feedback bounds ALL 4 models robustly**; mean-feedback blooms **3/4** but NOT
   `000250`. ⇒ the bloom is **model-dependent** (seed-bimodality, as `bloom_investigation` warned), but the
   sample FIX is universal. The fix is more reliable than the bug.
2. **OCCURRENCE — a recoverable BUG, GENERALIZES (esp. to nb).** oracle AP holds high (nb **0.46**, zinb
   032256/s44 ~**0.30**) while mean collapses to ~0.01; sample recovers much of it (nb **0.24**, ≈ climatology;
   zinb ~0.06–0.08). **Nuance:** the weak model `000250` has ~0 AP even at the oracle → no occurrence skill to
   recover (model heterogeneity). Headroom is real for the *good* models.
3. **MAGNITUDE — a CEILING, ROBUST.** Even the oracle stays timid (nb size_ratio 0.125, crps_events ~13; all
   models tied ~14). No feedback/oracle recovers event magnitude on any model.

### nb generalization (the strongest case)
gated_NB (soft_gate) under sample-feedback is a genuinely good bounded rollout on OCCURRENCE: crps_all 5.03→
0.13 (bloom fixed), **AP 0.01→0.24** (≈ climatology, held to h24), and the oracle shows **0.46** headroom —
the largest of any arm. Magnitude stays capped (size_ratio 0→0, oracle 0.125). So the three-way split is
not a zinb quirk — it holds across families, and gated_NB is the most promising rollout arm on occurrence.

### Verdict / decision (hardened)
- The **sample-feedback fix + the bug(occurrence)-vs-ceiling(magnitude) decomposition HOLD** across 3 zinb
  models + nb. Two honest caveats now on record: the *bloom* is model-dependent (fix still universal); the
  *occurrence headroom* is model-dependent (present for good models, absent for the weak `000250`).
- **Next lever unchanged and reinforced:** a rollout-training retrain (scheduled sampling / GTF, ADR-056) to
  cash the occurrence headroom — nb gated_NB is the strongest candidate (oracle AP 0.46; sample already 0.24).
- Still single sampling-seed per model, T=0-calibration, one-step-conditioned oracle (C-222). A true
  training-seed 3× + validation-partition graduation remain for a production claim.



---

## CORRECTION — "magnitude = a robust ceiling" was OVERCLAIMED — 2026-07-27

**Trigger:** the user challenged the claim that magnitude is "a ceiling — the model cannot predict how big",
noting a true skill-ceiling would produce a WIDE interval (misses both ways), not a SYSTEMATIC downward bias.
Correct. size_ratio ≈ 0.13 (predicted mean ~8× *below* the typical event) is not the signature of honest
uncertainty — it is a systematic under-prediction. My earlier framing collapsed two distinct things.

**Bulk-vs-tail biopsy** (`tmp/bulk_tail.py`, oracle cubes, ~54k event-obs, size_ratio by realized deaths):

| deaths | ZINB oracle | NB-gated oracle |
|--------|-------------|-----------------|
| 1–2    | **0.69**    | 0.125 |
| 3–9    | 0.38        | 0.179 |
| 10–29  | 0.22        | 0.151 |
| 30–99  | 0.11        | 0.080 |
| 100+   | **0.035**   | 0.021 |

**Corrected reading — magnitude is NOT one wall; it is TWO stacked effects:**
1. **Timid-body shrinkage (a fixable head/loss lever)** — a systematic downward bias in EVERY bin (even
   1–2 death cells come in < 1.0), the zero-dominated-loss artifact (the long-standing "timid body",
   size-ratio ~0.02–0.29). ZINB's self-zeroing lets its body fire far larger on the bulk (0.69) than the
   double-suppressed NB-gated (0.125) — so this component is real, family-dependent, and movable. The
   oracle inherits the trained weights, so it CANNOT rule this out (my error was reading "oracle still timid"
   as "irreducible").
2. **Tail ranking-ceiling (genuinely irreducible)** — the monotonic collapse to 0.035 at 100+ deaths: the
   model puts ~1/30th of the real mass on the biggest events. Corroborated by the separate amount-ceiling
   result (size-rank spearman 0.30 < persistence 0.37, confound-clean) — WHICH cell becomes a bloodbath is
   close to unpredictable. This IS a ceiling, but only for the TAIL, not for magnitude wholesale.

**This CONFIRMS (not challenges) the prior bulk-vs-tail finding** (S3 conditional quantiles: bulk sharp+
calibrated, tail risk-only, ξ≈0.8): magnitude predictability degrades smoothly with event size — bulk
substantially recoverable, tail a risk-only ceiling. **Finance analogy (ARCH/GARCH, "predict volatility not
level") HOLDS**, sharpened: tail *size* ≈ unpredictable (ceiling); *risk/occurrence* predictable+recoverable
(oracle AP 0.46). Conflict adds the timid-body shrinkage on top (finance models don't train on 99.7% zeros).

**Correct one-liner going forward** (retire "magnitude is a wall"): *occurrence = a recoverable exposure-bias
bug (GTF target); bulk-magnitude = a recoverable timid-body/head lever (ZINB already ~0.69); tail-magnitude
= an irreducible ranking ceiling.* Does NOT change the GTF plan (GTF targets occurrence). Flags a SEPARATE,
independent lever: the head/loss for bulk magnitude (out of scope for the rollout epic; noted for later).

---

## EXP-4 BUILD + SMOKE — 2026-07-27 (pre-reg `05d`)

**Build (committed `f345984`):** `_family_feedback_log1p` (mean = log1p E[y] / sample = composition-aware
draw); threaded into `_process_sequence` + `train()`, computed only when ε>0 so ε=0 is byte-identical
(parity by construction — 189 tests green incl. all training-engine + train-loop + distributions + CIC-86);
`ss_feedback` config field + validator; 4 TDD unit tests; ruff clean. The load-bearing fix: scheduled
sampling now feeds a family a composition-aware SAMPLE (was raw n_params, shape-mismatched → untested).

**Smoke (2-lesson nb+soft_gate train, ss_epsilon_max=0.25, ss_feedback=sample):** rc=0, both lessons × 3
windows trained, forensic dossiers generated, model saved, **finite loss (min 2345 / max 2945), NO real
NaN/Inf**, floor md5 restored, smoke artifact cleaned. Build integrates end-to-end on GPU. *Honest limit:*
the offline log printed no ε scalar, so this run doesn't independently prove ε>0 fired in lesson 2 (the
sample-feedback FUNCTION is unit-tested; the real EXP-4 driver will log ε/lesson). This is a BUILD gate, NOT
a skill result — 2 lessons can't show occurrence recovery.

**STATUS: at the launch gate.** Per `05d` §Cost (ask-before-long-batches, BINDING): the real EXP-4 =
retrain gated_NB with ss_feedback=sample (full 40-lesson recipe), then deploy-sample-rollout + T=0 eval,
scored on the frozen ruler, A/B vs the baseline gated_NB (deployed-sample AP 0.24 → oracle ceiling 0.46).
Cost ≈ 30–60 min train + ~3 min eval **per seed**; a claim needs 3 seeds (~2–3 GPU-hr). **The retrain does
NOT fire without explicit user go.**

---

## EXP-4 RESULT (1-seed indicative) — GTF REGRESSED occurrence — 2026-07-27

**Run:** gated_NB retrained with `ss_feedback=sample` (ε_max=0.5, warmup 10/40, seed 44) → artifact
`012051` → deployed sample-feedback rollout → `results/exp4_gtf_nb_sample.csv`. A/B vs baseline gated_NB
(`102130`, `harden_nb_..._sample`). All rc=0, floor restored.

### Occurrence AP (sb) — GTF WORSE at every horizon (P1 FAILED, F-G1 fired)
| h | GTF | baseline | oracle ceiling |
|---|-----|----------|-------|
| 1 | 0.276 | 0.440 | — |
| 12 | 0.063 | 0.280 | ~0.30 |
| 24 | 0.010 | 0.238 | 0.46 |
(ns/os same: h12 GTF 0.019/0.016 vs baseline 0.133/0.132.) GTF did NOT climb toward 0.46 — it fell below
baseline, including at T=0.

- **F-G2 (T=0 crps guardrail): HELD** — sb h1 0.147 vs 0.143 (within tol); ns 0.083 vs 0.082; os 0.045 vs
  0.032. Magnitude/crps fine; it is specifically the GATE/occurrence that degraded (AP down, crps_events
  tied ~14, crps_none GTF 0.0004 vs 0.0027 — GTF is even MORE timid-zero).
- **Stability held** — GTF crps_all bounded across h (0.15→0.87), no bloom.

### Mechanism (hypothesis) + the CONFOUND (why this is NOT yet a GTF verdict)
Pattern (gate down, body tied) is consistent with **ε_max=0.5 too aggressive**: feeding back noisy sparse
samples 50% of the time corrupts the GATE's input → the occurrence classifier trains on noise → worse gate.
BUT the A/B is **confounded (C-112)**: (1) SEED — GTF-seed44 vs baseline-102130 (unknown seed; occurrence
is model-dependent — `000250` had ~0 AP even at oracle), so part of the gap may be seed, not GTF; (2) ε —
0.5 was an open pre-reg choice. **Cannot cleanly attribute the regression to GTF-per-se.** Declaring GTF
dead here = the corrupted-probe trap.

### Verdict + decision (pre-reg F-G1 → "escalate or accept the gap")
The indicative DIRECTION is negative (GTF-ε0.5 regressed occupancy), but ambiguous. Disambiguating options
(each ~1 train, ~30–60 min): (a) **matched no-GTF seed-44 baseline** — isolates GTF from seed (the clean
attribution the C-112 guardrail wants); (b) **gentler ε** (e.g. 0.1–0.15) — tests the too-aggressive
hypothesis; (c) **accept the indicative negative** and stop — the bloom epic's core findings (bloom=fixed
bug via sample-feedback; occurrence=recoverable-in-principle per the oracle; magnitude=bulk-lever+tail-
ceiling) stand regardless; GTF-as-implemented does not (yet) cash the occurrence headroom. Caveats
throughout: single seed, T=0-cal, one-step oracle, ε_max unpinned. Artifact `012051` retained.
