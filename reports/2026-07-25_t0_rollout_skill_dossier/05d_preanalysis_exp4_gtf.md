# 05d — Pre-analysis plan: EXP-4, the scheduled-sampling + sample-feedback retrain (GTF)

**Pre-registered 2026-07-27, BEFORE running.** The rung-3 fix: teach the model, DURING TRAINING, to recover
from its own **sampled** feedback → reclaim the occurrence headroom EXP-3 measured (deployed sample AP 0.24 →
oracle ceiling 0.46). Variant chosen: **scheduled sampling (Bengio 2015) with sample-feedback** (not
GTF-proper α-state-weighting — deferred; its chaos-premise doesn't transfer to conflict, C-125c). This is a
**training-side change** — expensive; the actual retrain launch is a SEPARATE gated decision (§Cost).

## Hypothesis (H-GTF)

The occurrence decay under free-running rollout is exposure bias (EXP-3: the oracle keeps AP flat ~0.30–0.46;
the deployed rollout collapses to ~0.01 under mean-feedback, ~0.24 recovered by sample-feedback). If we
**train** the model on its own sampled feedback (curriculum: start teacher-forced, ramp in fed-back samples),
it learns to stay in-distribution under rollout → the DEPLOYED sample-feedback rollout's AP climbs above 0.24
toward the 0.46 ceiling — WITHOUT regressing the T=0 forecast (the current product).

## The ONE change (behind the existing ss flag)

Scheduled sampling (ADR-056) is already wired (`ScheduledSamplingMixer` ε-ramp; `_process_sequence` mixes
GT vs fed-back input per cell). **The load-bearing fix:** what it feeds back today is `prev_pred =
t1_pred.detach()` — the **mean/raw params** (the diffuse object EXP-2 proved is the disease), and it predates
the family head (feeds raw n_params channels, shape-mismatched to the n_reg dynamic inputs — so ss is
currently untested/broken for nb/zinb). **Change:** the training feedback must be the **composition-aware
SAMPLE** (the same emit-or-sample path as inference `_sample_feedback`), so **training exposure = deployment
exposure**. That is the entire hypothesis in one line: match train to deploy.

## Arm & design

- **Arm:** gated_NB (soft_gate) — strongest occurrence (oracle AP 0.46; deployed sample 0.24) and the
  deployable family. Retrain from scratch on the current foundation recipe + ss-sample curriculum.
- **A/B (one variable = the ss-sample curriculum):** GTF-retrained gated_NB vs the baseline gated_NB
  (`…102130`), BOTH scored the same way — deployed **sample-feedback** rollout on the frozen per-horizon
  ruler (`rollout_skill_score.py`), + the T=0 lodestar.
- **Curriculum:** ε ramps 0→ε_max over lessons (ScheduledSamplingMixer); teacher-forced early (don't train
  on garbage), sampled feedback late.

## Pre-registered predictions

- **P1 (occurrence reclaimed):** GTF-retrained deployed-sample AP(sb, h12–24) **> 0.24** (baseline sample),
  moving toward the 0.46 oracle. The core win.
- **P2 (T=0 preserved):** GTF-retrained T=0 crps-all within the guardrail of the baseline foundation
  (≤ +0.005 sb, i.e. no material regression) — the retrain must not trade the product for the rollout.
- **P3 (stability kept):** the GTF model's sample-feedback rollout stays bounded (no bloom) — sample-feedback
  already bounds it; training on it should not break that.
- **P4 (magnitude unchanged):** size_ratio ≈ baseline (this retrain targets occurrence, NOT magnitude — the
  timid-body/tail lever is separate, out of scope).

## Pre-committed falsifiers

- **F-G1 (no occurrence gain):** GTF-retrained AP ≈ baseline sample (0.24), not climbing → curriculum
  training does NOT reclaim the headroom → escalate to GTF-proper (α-state) or accept the gap. Kills H-GTF.
- **F-G2 (T=0 regression — HARD STOP):** T=0 crps-all regresses beyond the guardrail → the retrain sacrifices
  the current product; unacceptable unless the T>0 win is large AND the ensemble can carry a T>0-specialist
  arm. Escalate to the user, do not ship.
- **F-G3 (stability lost):** the GTF model reintroduces the bloom → training on sampled feedback destabilized
  it (surprising; would need diagnosis).
- **F-G4 (calibration gamed, C-126):** AP up but crps-none / crps-events degraded → point-occurrence improved
  at calibration's expense (the C-126 point-stability≠calibration trap). Read the full split, not AP alone.
- **F-G5 (truncated-horizon, C-125b):** the gain holds at short h (within the training window seq_len) but
  vanishes by h24–36 → short-window SS training doesn't transfer to the long deploy rollout. Report the gain
  vs h honestly; a short-h-only gain is a partial result, not the win.

## Guardrails (from C-125/C-126 — pre-committed)

- **No proper-score contamination (C-125a):** scheduled sampling changes EXPOSURE, not the loss — it adds NO
  stability regularizer, so the headline T=0 proper score stays clean. (This is why SS-sample is cleaner than
  GTF-proper, which DOES add an α-term.) The training loss remains the family NLL.
- **Truncated-horizon honesty (C-125b):** train-window seq_len < deploy-36; F-G5 checks transfer to long h.
- **Calibration co-primary (C-126):** success = AP up AND crps split not degraded, on the frozen ruler.
- **Attribution (C-112):** GTF vs baseline must be device-matched, ideally ≥3 seeds; single-seed = INDICATIVE.
- **Stealth:** floor trap-restore; never commit views-models.

## Build steps (TDD, before any GPU)

1. **Composition-aware training sample-feedback** — factor the `_sample_feedback` emit-or-sample logic so
   `_process_sequence` can feed back a composed SAMPLE (n_reg channels) instead of `t1_pred` raw params.
   Red tests: ss with a family head shape-matches (currently doesn't); ε=0 ⇒ byte-identical to today
   (teacher-forced parity); ε=1 ⇒ feedback differs; deterministic (seeded).
2. **Config**: reuse `ss_schedule` / `ss_epsilon_max`; add a `ss_feedback ∈ {mean (legacy), sample}` if
   needed to keep the old point-head path byte-identical. CIC bump.
3. **Smoke** (2-lesson) + full suite + ruff + determinism green. **Then STOP for the launch gate.**

## Cost & the launch gate (ask-before-long-batches — BINDING)

A real retrain: foundation recipe = 40 lessons × windows, GPU-hours per seed; a claim needs ≥3 seeds. This is
**not** eval-only. After the build + smoke pass, I **STOP and present the smoke result + the exact launch
cost**, and the retrain runs only on explicit go. No 3-seed retrain fires without that.

## Decision rules

- **Build first (this scope), smoke, gate.** Retrain only on user go.
- **P1 holds + P2/F-G2 clean ⇒** SS-sample recovers occupancy without costing T=0 → a genuine rollout fix;
  gated_NB-GTF becomes the deployable long-horizon arm. Then 3-seed + validation graduation.
- **F-G1 ⇒** curriculum insufficient → GTF-proper (α-state) is the next escalation, or accept the gap.
- **F-G2 ⇒** HARD STOP + user decision (T=0 vs T>0 trade).
- **F-G5 ⇒** partial (short-h) win; consider a longer training window before scaling.
