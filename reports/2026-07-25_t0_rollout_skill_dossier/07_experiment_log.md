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


