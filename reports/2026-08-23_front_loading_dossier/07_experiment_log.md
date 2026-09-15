# Experiment log — the step-1 occurrence shortfall

Pre-registration `05_analysis_plan.md` **LOCKED `9dc28df`** with `tools/` empty; **AMENDMENT 1**
committed before §4 was read.

---

### EXP-01 · 2026-08-23 · **DOUBLE-COUNT CONFIRMED — 4/4 seeds**

`G = mean(gate)/truth` · `C = mean(gate × P(NB>0))/truth` · observed = the emitted step-1 ratio from the
existing `fedfield` CSVs.

| seed | `G` | `C` | observed | \|C−obs\| | §4 |
|---|--:|--:|--:|--:|---|
| 42 | 1.202 | 0.406 | 0.371 | 0.035 | DOUBLE-COUNT |
| 43 | 1.237 | 0.435 | 0.398 | 0.037 | DOUBLE-COUNT |
| 44 | 1.297 | 0.457 | 0.427 | 0.030 | DOUBLE-COUNT |
| 45 | 1.382 | 0.400 | 0.371 | 0.029 | DOUBLE-COUNT |

**Mean `G` = 1.280 — the gate OVER-predicts occurrence by 28%.** Multiplying by the NB body's
`P(draw > 0)` attenuates it by **×0.332**, leaving **`C` = 0.424**. §4's first branch requires
`G ≥ 0.80` and `C` within ±0.05 of observed: **both hold on every seed**, and `G` clears its bar by a
margin the rule did not anticipate — the hypothesis was "the gate is approximately right", and the gate
is in fact *hot*.

**The occurrence process is modelled once by the classifier and then silently applied a second time by
the body's zero mass.** `soft_gate` emits `body_sample × bernoulli(gate)`, so a cell survives only if the
gate fires **and** the NB draw is non-zero — but the gate was already trained to answer exactly the
question the NB's zero mass re-answers.

## Falsifiers

| | result |
|---|---|
| **F1** origin | **PASS** — `origin = 335`, period 371, both passed as required arguments; `seq_len - 1 = 383` is never reachable in this tool (C-308 has fired twice) |
| **F2** decomposition matches the sampler | **PASS** 4/4 after AMENDMENT 1 |
| **F3** same truth | **PASS** — `truth = 0.00340` on every seed, matching the `use_real` step-1 active fraction |
| **F4** finite | **PASS** |

## F2 was broken as registered, and the fix was to the estimator

F2 compared the analytic decomposition against **one** Bernoulli realisation with a **2%** tolerance. At
this prevalence that realisation carries a Monte-Carlo standard error of ~**15%** (~45 active cells in
32,400) — **seven times its own tolerance**. The first run returned FAIL at 21.6%, which is ~1.8σ of pure
sampling noise. **As registered, F2 could essentially never pass; it was measuring the sampler's
variance.** AMENDMENT 1 averaged the sampled emission over **k = 200** draws (standard error ≈ 1.1%) and
**kept the 2% tolerance unchanged**. Widening the threshold would have made F2 pass by weakening it.

## What this does and does not establish

**Does:** at step 1 — where the input is entirely real data and no feedback has occurred — the
composition emits 42% of true occurrence while the gate alone would emit 128%. The shortfall is
**located in the composition layer**, not in the gate and not in training.

**Does not:** this is **occurrence at one step**. No AP was measured. **The claim that fixing the
composition would improve AP is NOT established by this experiment** — it is the obvious next question,
and it needs its own pre-registration and a scored run.

⚠️ `C` sits systematically ~0.03 **above** observed on all four seeds. Within §4's ±0.05, but the sign is
consistent, so it is a real small bias rather than noise: this tool runs **dropout OFF** while the
`fedfield` CSVs come from the production path with locked dropout on. Recorded rather than absorbed.

## Consequence

**This is upstream of every rollout result in the ledger.** The compounding shown in `05` §0 — ×0.78 per
step — starts from a step-1 value that is 2.4× too low for a reason that has nothing to do with the
rollout. It also reframes M42/M30–M33: every scheduled-sampling arm was training the model to feed back
a field whose occurrence was already halved by composition.

**Free cross-check, per §6:** `self_zeroed` (ZINB) does **not** multiply by the gate, so it should **not**
show this shortfall. Existing ZINB artifacts can test that without a new run — **not done here**, and it
is the first thing the follow-up should do.
