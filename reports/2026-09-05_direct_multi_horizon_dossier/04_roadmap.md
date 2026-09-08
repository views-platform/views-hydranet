# 04 — Roadmap

**Written:** 2026-09-05. Phased and gated; no phase starts until the prior gate is green.

```
S0 dossier ──DONE
   │
S1 library gap: fetch Shi2017 ──────► GATE: the backbone's own paper is in the evidence base
   │
S2 design review (F1–F6) ───────────► GATE: a seated panel has ruled
   │
S3 pre-registration ────────────────► GATE: paired 2-seed, CLAMPED control, h1 floor, VOID branch
   │
S4 implement (registry subclass) ───► GATE: incumbent byte-identical; contract tests green
   │
S5 the ruler ───────────────────────► GATE: proven to score a direct cube identically to a rollout cube
   │
S6 adversarial audit ───────────────► GATE: mutation testing to exhaustion, clean-context non-author
   │
S7 smoke + potency ─────────────────► GATE: potent on the arm's own config AND at a trained checkpoint
   │
S8 run: clamped control vs direct ──► GATE: floor gate on the CONTROL before the treatment arm runs
   │
S9 score + locked rule
   │
S10 disposition
```

| S | title | substance |
|---|---|---|
| **S0** | Dossier | **done** |
| **S1** | Close the library gap | Fetch **Shi2017** (encoder-forecaster + B-MSE/B-MAE rare-event weighting) and extract claims. 567 papers and no primary ConvLSTM source is not an acceptable base for redesigning the ConvLSTM's forecast path. Extract `Wen2017_MQRNN` key passages while there. |
| **S2** | `expert-method-review` on `02_design` | Rule on F1–F6. Seat the ConvLSTM and forecasting-evaluation chairs at minimum; the PF panel's F1 split (cell vs hidden) recurs here and needs settling with a measurement, not a vote. |
| **S3** | Pre-registration | **Paired 2-seed** (M65: paired MDE 0.0131 vs the +0.0367 that matters), the **clamped** arm as control, a **non-inferiority floor at h1**, the `act_ratio` band with a stated abandonment condition, and an explicit VOID branch. |
| **S4** | Implement | A subclass registered via `registry.py`, following `FiLMSkip`. Horizon covariate, forking-sequence sampler, per-horizon loss. Incumbent byte-identical; registry contract tests green. |
| **S5** | The ruler | **The story this dossier could most easily skip and must not.** The scoring path assumes a rollout-produced cube. Prove it treats a directly-produced cube identically — same months, same cells, same truth, same `S`. The M-record is full of rulers that flattered their own verdict (the h36 CRPSS artifact; the unequal-`m` bias). |
| **S6** | Adversarial audit | Mutation testing to exhaustion, committed before mutating; `/falsify guard`; clean-context non-author. |
| **S7** | Smoke + potency | 2-lesson smoke; potency on the arm's own config **and at a trained checkpoint** (C-324/C-325). |
| **S8** | Run | Clamped control vs direct, 300 lessons, **2 seeds paired**. ~4 arms. |
| **S9** | Score | Weight-hash post-condition read first; `AP@h18` primary; fair-CRPS co-primary with its reliability–resolution decomposition; h1 floor checked before the headline. |
| **S10** | Disposition | Ledger M-entry, register, and either a proposed ADR or an honest close. |

## Decision points

* **After S2** — the panel may rule the marginal-horizon cost fatal. One review beats ~16 GPU-hours.
* **After S5** — if the ruler cannot be shown to score both cube types identically, **stop**. An
  unfair ruler makes every downstream number meaningless, and this repo has been bitten by exactly
  that.
* **After S7** — if the knob is inert at a trained checkpoint, stop (C-325).

## Budget

Two arms × two seeds × 300 lessons ≈ **16 GPU-hours**, plus implementation. Double the usual screen,
and **the reason is M65**: at n=1 this design could not detect an effect the size of the cell freeze.
Paying for the second seed is what makes the result readable.
