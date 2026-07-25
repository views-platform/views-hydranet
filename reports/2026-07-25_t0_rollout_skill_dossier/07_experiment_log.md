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

