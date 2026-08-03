# 02b — Expert Method Review of `02_design` (Axis B / rollout training)

**Date:** 2026-06-05 · **Skill:** `expert-method-review` · **Target:** `02_design.md`
**Gate:** this review **precedes** any training-loop change (per `00_README` §5).
**Chair:** simon (not seated).

---

## 1. Target & decisions under review

The Axis-B design fixes C-113 (autoregressive runaway in `HydraBNUNet06_LSTM4`, a
ConvLSTM U-Net) at the **training-algorithm** level. Decisions:

1. **Keep the recurrence; fix the training** (vs N-BEATS-style direct multi-horizon).
2. **Put gradient back into the prediction→input feedback loop** that is currently
   detached (`training_engine.py:200`) — the operator the io-gain diagnostic blamed.
3. **Choose among** B1 pushforward (Brandstetter), B2 GTF (Hess), B3 Professor
   Forcing (Lamb).
4. **`rollout_horizon` K** config HP (proposed default K=12) + GPU cost (B1 flat
   memory vs B2 O(K) BPTT; checkpointing; temporal bundling).
5. **Kill `freeze_h`** (inference-time state-freeze hack).

**DGP the design must respect.** Monthly PRIO-GRID conflict-count fields:
spatiotemporal, **heavy-tailed/zero-inflated**, strongly **persistent**, with
escalation bursts. *Crucially unestablished:* whether this DGP is **chaotic**
(`λ_max > 0`) — the premise GTF's theory rests on. The observed explosion is an
`expm1` out-of-range amplification of a log-space drift, which may be a *distribution-
shift* artifact (Brandstetter) rather than *Lyapunov divergence* (Hess).

## 2. Panel

Chair-requested core + two adds to complete the fault lines (6 seats, within 4–7):

| Seat | Why seated | Fault lines |
|---|---|---|
| **Sepp Hochreiter** | authored the LSTM; owns BPTT exploding/vanishing dynamics — the runaway's lineage | BPTT cost/stability; is-it-chaotic; B1-vs-B2 |
| **DL-engineer archetype** | BPTT memory/throughput, optimization stability, HP surface | B1-vs-B2; cost |
| **Rich Sutton** | bitter lesson; anti-hand-engineered-prior contrarian | is-it-chaotic; keep-vs-drop recurrence; over-engineering |
| **Tilmann Gneiting** | proper scoring; sharpness s.t. calibration; CRPS purist | proper-score-vs-regularizer; point-stability≠calibration |
| **Xingjian Shi** *(add)* | **authored ConvLSTM for spatiotemporal nowcasting** — our exact backbone and nearly our exact problem (multi-step rollout) | keep-vs-drop recurrence; blurring/calibration |
| **Operational archetype** *(add)* | production GPU budget, freeze_h-removal blast radius, HP/reproducibility surface | feasibility; sequencing vs the open C-111 sweep |

Adds justified: without **Shi** the "keep recurrence" claim has no authoritative
anchor opposing Sutton's "drop it"; without the **operational** seat the GPU-cost and
freeze_h-removal decisions have no production-reality check.

## 3. Library grounding

**Held & load-bearing (cited below):**
- `recurrent_stability/Brandstetter2022_*` (B1), `deep_consored/hess23a.pdf` (B2),
  `rollout_training/1610.09038v1.pdf` (B3) — all read in full this session (`01`).
- `recurrent_stability/Pascanu2013_DifficultyTrainingRNNs.pdf` — exploding-gradient +
  clipping (Hochreiter, DL-engineer).
- `recurrent_stability/MillerHardt2019_StableRecurrentModels.pdf` — stability costs
  expressiveness (the counter-pressure to "just make it contractive").
- `recurrent_stability/{Erichson2021_LipschitzRNN, Chang2019_AntisymmetricRNN,
  Arjovsky2016_UnitaryRNN, Miyato2018_SpectralNormalization}.pdf` — the architectural
  stability family (Hochreiter's belt-and-suspenders; Hess's "insufficient alone").
- `papers/Oreshkin2019_NBEATS.pdf` — **held** — the direct-multi-horizon alternative
  (Sutton's baseline; Gneiting's "it's a point forecaster" objection).
- `papers/Gneiting2007_StrictlyProperScoringRules.pdf`,
  `papers/Gneiting2014_ProbabilisticForecasting.pdf`,
  `papers/Gneiting2011_ComparingDensityForecasts.pdf`,
  `papers/Dawid2007_GeometryProperScoringRules.pdf` — **held** — the proper-scoring /
  energy-score grounding for Gneiting's seat.
- `deep_consored/NIPS-2015-scheduled-sampling-*.pdf` — ADR-056's basis (the biased
  estimator).

**Gaps to fetch:**
1. **Mikhaeil, Monfared & Durstewitz (2022)** — the chaotic-DS ill-posedness proof
   GTF rests on. *Load-bearing for D1 (is-it-chaotic).*
2. **Huszár (2015)** — the scheduled-sampling bias argument.
3. **Sanchez-Gonzalez et al. (2020)** — learned simulators / noise injection (the
   baseline pushforward beats; needed to state the noise alternative fairly).
4. *(have, but cite)* **Gneiting & Raftery (2007)** energy score for multivariate /
   trajectory forecasts — already covered by the held Gneiting set.

## 4. Independent critiques

### 4.1 Sepp Hochreiter — *the gradient-flow realist*
- **Endorse the layer, with a warning that B2 reopens the original wound.** Fixing
  this in training (not architecture) is right — but B2/GTF runs **BPTT through K
  steps**, which is precisely the exploding/vanishing-gradient regime the LSTM was
  invented to survive (Hochreiter & Schmidhuber 1997; Pascanu 2013). GTF's `(1−α)`
  Jacobian-scaling (Hess Eq. 7–8) is *recognisably* a gradient-magnitude control —
  kin to the constant-error-carousel intuition — so it is the right tool, but it must
  be paired with **gradient clipping** (Pascanu) as standard hygiene; the design omits
  clipping.
- **Why is the LSTM gating not already protecting us?** The cell-state gating bounds
  the *recurrent* path — but the runaway rides the **output→input feedback** path
  (the diagnostic), which the gating does **not** guard. This is the real diagnosis:
  we have a stable cell but an unguarded prediction-feedback operator with gain > 1.
  The design should *say* this — it reframes B1/B2 as "train the one path the
  architecture doesn't protect."
- **Challenge K=12 truncation.** Truncated BPTT gives **biased gradients** for
  dependencies longer than K (the long-term-dependency problem I spent a career on).
  If the runaway compounds over 24–36 steps, training to 12 may certify the wrong
  horizon. Either K=36 (with checkpointing) or a pre-registered falsifier that checks
  steps 13–36.
- **What to build alongside:** consider a **bounded-Lipschitz** complement
  (Erichson 2021; Miyato 2018 spectral norm on the feedback operator). Hess says
  architecture alone is insufficient *for chaotic systems* — but combined with
  rollout training it is belt-and-suspenders, and it directly attacks the measured
  `‖J‖₂ > 1`.

### 4.2 DL-engineer archetype — *the throughput/memory pragmatist*
- **B1-first ordering is correct and under-sold.** We already detach
  (`prev_pred.detach()`), so B1 is a *small* diff: feed the model's own prediction
  always (not Bernoulli@`ss_epsilon`), add the weighted stability term, loop to K.
  ~2× forward, **flat activation memory** — essentially no OOM risk. Prototype it in
  an afternoon; measure before paying for anything bigger.
- **"Very likely feasible at K=12" (§6) is hand-waved — measure it.** A ConvLSTM
  U-Net stores `[B, C, H, W]` feature maps *per step per skip-level*. Report the real
  peak memory at K=12, current batch, 32×32 *before* committing to B2. If it OOMs,
  `torch.utils.checkpoint` on the per-step forward trades ~30% compute for O(√K)
  memory — standard, mention it as the plan not a footnote.
- **The new HPs are a hidden cost.** B1 adds a stability-loss **weight**; B2 adds **α**
  and its schedule. Tuning these on top of the *already-unstable* C-111 balancer is
  two-unstable-knobs-at-once. Demand a **cheap dose-response** (sweep K∈{1,5,12} ×
  weight on the `diagnose_io_gain` proxy, retrain-free where possible) before any full
  golden_hour retrain.
- **Bias of truncation is also a feature:** TBPTT at K=12 is exactly what the
  LSTM-TBPTT baselines do (Hess Table 1) — defensible, just declare it.

### 4.3 Rich Sutton — *the bitter-lesson contrarian*
- **Kill `freeze_h` hardest — it is the anti-pattern.** A hand-coded inference-time
  prior that freezes state is exactly the kind of human-engineered cleverness the
  bitter lesson says loses. Unanimous, but I want it on record as *principle*, not
  just "the ablation says it's inert."
- **B1 is the *general* fix; be suspicious of B2/B3's added machinery.** "Train the
  model the way you use it" is a general method (match train to test) — that scales.
  GTF's α-bounding and Professor Forcing's discriminator are *added structure*. Reach
  for them only if the general method demonstrably fails.
- **Is the system even chaotic?** The design imports Hess's chaotic-DS apparatus
  (Lyapunov exponents, `α* = 1−1/σ̃_max`) for a conflict-count forecaster that is
  very likely **not** chaotic — it's persistent + heavy-tailed. The explosion looks
  like `expm1`-out-of-range drift, not exponential trajectory divergence. **Don't
  import a theory whose premise you haven't checked** (→ fetch Mikhaeil 2022; the
  design's own §9-Q1 admits this — promote it from "open question" to "blocking check
  before B2").
- **The cheapest rollout training is a longer training window.** If the training
  window is currently *shorter* than the 36-step inference rollout, the most general,
  least-clever fix is to **train on 36-step windows** (more data/compute through the
  rollout), not a bespoke loss term. Verify `seq_len` vs 36 first — it may dominate
  the fancy options.

### 4.4 Tilmann Gneiting — *the proper-scoring purist*
- **The stability term is not a proper scoring rule — quarantine it.** Pushforward
  `L_stability` and GTF's interpolation are **regularisers**, not proper scores
  (Gneiting & Raftery 2007). If they are weighted into the objective with a fixed
  nonzero coefficient, the minimiser is **no longer the true predictive
  distribution** — you buy stability by biasing the forecast. Keep the coefficient
  **small and/or annealed→0**, and **report CRPS uncontaminated** as the headline.
  The design's §8 *states* this principle but the §7 path doesn't operationalise the
  annealing — make it concrete.
- **Point-stability ≠ calibration — your success metric is wrong.** `diagnose_io_gain`
  measures whether the *point* trajectory stays in-range. The runaway is a *mean*
  pathology; the chronic problem (MCR, ADR-057) is a *calibration* pathology. You can
  fix the former and leave — or worsen — the latter. **The readout must include PIT /
  coverage and sharpness**, not attractor magnitude alone. Declaring C-113 "solved"
  on in-range-attractor would be a category error.
- **Multi-step scoring: per-step CRPS misses path dependence.** Summed per-step CRPS
  is proper *marginally* but ignores the joint distribution over the 36-step path. If
  the trajectory shape matters (it does for escalation), the proper object is the
  **energy score / variogram score** (Gneiting 2008/2011, held). At minimum, decide
  explicitly whether you're scoring marginals or the path.
- **On dropping recurrence for N-BEATS (held):** N-BEATS is a **point** forecaster
  with no native predictive distribution. Switching to it would *regress* the
  probabilistic objective we evaluate on. A reason to **keep** recurrence + a proper
  distributional head — agrees with Shi, against Sutton's "drop it."

### 4.5 Xingjian Shi — *the ConvLSTM nowcasting author*
- **Rollout training is native to this backbone — endorse, this is solved territory.**
  ConvLSTM nowcasting (Shi 2015; encoder-forecaster, Shi 2017) *already* unrolls the
  forecaster over many steps and trains through it. "Keep the recurrence and train the
  rollout" is the standard nowcasting recipe; the design is rediscovering a known-good
  pattern, which is reassuring. The spatial+temporal inductive bias is exactly why
  ConvLSTM fits — conflict has both. **Against Sutton's "drop it."**
- **Beware the blurring failure mode — it couples to Gneiting's point.** Multi-step
  ConvLSTM rollouts are notorious for **regression-to-the-mean blurring**: optimising
  multi-step error rewards hedging toward the spatial mean. For zero-inflated conflict
  fields this would *deflate* the predicted intensities and *worsen* the zero-rate/MCR
  problem. **Monitor sharpness and the zero-rate**, not just stability. This is the
  one place rollout training could actively hurt the chronic problem.
- **The feedback is a *field*, not a scalar — what exactly is fed back matters.**
  Specify whether the fed-back quantity is the post-`expm1` count field, the log-space
  field, or the latent — the operator gain the diagnostic measured is in log1p space.
  GTF interpolation `(1−α)·pred + α·GT` must happen in the **same space** the model
  consumes; get this wrong and α means nothing.

### 4.6 Operational archetype — *the production realist*
- **Sequence this AFTER the C-111 balancer question closes.** The balancer×seed sweep
  is in flight and F2 has already fired (seed-4-frozen → inf). Enabling rollout
  training now adds knobs to a system whose acute instability is still unattributed —
  you won't be able to tell which fix did what. Finish the balancer verdict first.
- **`freeze_h` removal has blast radius.** It's a live config option across
  golden_hour inference. Removing it changes the inference path for **every** model —
  gate it behind the **parity guard** (`rollout_horizon=1` byte-identical) *and* a
  re-eval of every golden_hour model with `freeze_h="none"` before merge. Don't flip
  it globally blind.
- **`rollout_horizon` as a per-model config HP fits the pattern** (views-models
  configs) — but it's another reproducibility surface (manifest audit, C-43) and
  another GPU-budget multiplier. Default **K=1 (off)** is the right safe default;
  B2 at K=12+checkpoint roughly doubles train time × 3 models × seeds — a real monthly
  cost. Favors B1.

## 5. Key disagreements (the product)

**D1 — Is the DGP chaotic, and does GTF's theory therefore apply?**
*Sutton + Gneiting:* No — conflict counts aren't chaotic; the blow-up is `expm1`
out-of-range drift, not Lyapunov divergence. Don't import GTF's α* bound as *theory*.
*Hochreiter:* The **BPTT mechanism** (Jacobian-product growth) bites for *any*
feedback operator with gain > 1 — which the diagnostic measured — independent of
whether the DGP is formally chaotic. GTF's α-scaling is a general gradient-control
tool, valid as a *heuristic* even if the chaos premise fails.
*Merit/resolution:* Both right at different levels (theory vs mechanism). **B1
sidesteps it** (no deep BPTT). If escalating to B2, use α as pragmatic gradient
control (clip-like), **not** as a correctness guarantee — and fetch Mikhaeil 2022 to
settle whether the bound is even meaningful here.

**D2 — Is B1 sufficient, or is B2 necessary?**
*DL-engineer + Sutton:* B1 first — cheap, general, flat memory, likely enough.
*Hochreiter:* B1 trains only *one-step-back* stability; the runaway **compounds** over
many steps and truncated K=12 gives biased gradients for the 24–36 tail — B1 may
certify the wrong horizon.
*Merit/resolution:* Pre-register the falsifier — B1 wins iff the io-gain attractor is
in-range **and** step-wise CRPS is bounded through **all 36** steps (not just ≤K). If
bounded at K but divergent past K → escalate to B2.

**D3 — Proper score vs stability regulariser.**
*Gneiting:* The stability term corrupts the proper score; anneal it →0, report CRPS
clean; and point-stability is the wrong target — calibration is.
*Pragmatic camp (DL-engineer, implicitly Brandstetter):* A small fixed weight
empirically works (Brandstetter's results).
*Merit/resolution:* Gneiting is theoretically correct and cheap to honor — **anneal
the weight, keep CRPS + calibration as the uncontaminated headline.** No real tension
once annealed.

**D4 — Keep recurrence vs drop it (N-BEATS direct multi-horizon).**
*Shi + Hochreiter + Gneiting:* Keep — ConvLSTM encodes the right spatiotemporal bias,
nowcasting proves rollout training works on it, and N-BEATS (held) is a *point*
forecaster that would regress the probabilistic objective.
*Sutton:* At least *know* the direct-multi-horizon baseline number; a big direct model
+ compute might beat the engineered recurrence fix.
*Merit/resolution:* Keep recurrence (3-to-1, and the probabilistic-objective argument
is decisive). Sutton's ask is cheap and fair: record a direct baseline for expectation
calibration, but it is **not** the path.

**D5 — Does rollout training help or hurt the *chronic* problem? (shared caution)**
*Shi + Gneiting (aligned, against the design's implicit optimism):* Multi-step
optimisation encourages mean-hedging/blurring → could **worsen** sharpness and the
zero-rate/MCR (ADR-057). Fixing point-stability is not obviously progress on
calibration.
*Resolution:* This is the **strongest dissent to keep live** — the readout must track
MCR/zero-rate/coverage, and "C-113 solved" must not be declared on attractor magnitude.

## 6. Synthesis & recommendation

**Build, in this order:**

1. **Kill `freeze_h` now** (unanimous; Sutton on principle, Operational on process):
   set `freeze_h="none"`, remove the inference-time state-freeze — **gated** by the
   parity guard + a golden_hour re-eval. Independent of everything below.
2. **Verify `seq_len` vs 36 first** (Sutton's cheap lever): if the training window is
   shorter than the inference rollout, *lengthening the window* is the most general,
   least-clever rollout training and may dominate the fancy options. Cheap to check;
   do it before implementing a loss term.
3. **Implement B1 pushforward behind `rollout_horizon` (K=1 default)** — smallest diff
   (we already detach), flat memory, beats noise injection (Brandstetter). With:
   - **annealed/small stability-loss weight; CRPS reported uncontaminated** (Gneiting);
   - **a readout that includes PIT/coverage + MCR/zero-rate + per-step CRPS through all
     36 steps**, not just the io-gain attractor magnitude (Gneiting + Shi);
   - **a cheap dose-response** (K∈{1,5,12} × weight on the proxy) before a full retrain
     (DL-engineer).
4. **Escalate to B2 GTF only on a pre-registered B1 failure** (bounded at K, divergent
   past K). Then: α as *gradient control* not chaos theory (Sutton/Gneiting), **add
   gradient clipping** (Hochreiter), **measure peak memory at K=12 + checkpoint**
   (DL-engineer), interpolate **in the model's input space** (Shi), and fetch Mikhaeil
   2022 to check the premise.
5. **B3 Professor Forcing stays catalogued** — revisit only if calibration stays
   broken after point-stability is fixed (i.e. if D5's caution materialises and B1/B2
   don't address the *distributional* mismatch).
6. **Sequence the whole program after the C-111 balancer verdict closes**
   (Operational) — don't tune rollout HPs while the acute trigger is unattributed.

**What's methodologically missing in `02_design` (fold in before pre-registration):**
gradient clipping (Hochreiter); the calibration/sharpness readout + the
proper-score-quarantine made concrete (Gneiting); the explicit feedback-space
specification (Shi); measured (not asserted) K=12 memory + checkpoint plan
(DL-engineer); the `seq_len`-vs-36 check + a direct-multi-horizon baseline number
(Sutton); the chaos-premise check promoted from open-question to B2-blocker.

**Strongest dissent to keep live (D5):** *fixing the point-trajectory runaway may not
fix — and could worsen — the calibration/zero-rate problem the model also has.*
Carry it into the pre-analysis plan as a falsifier, not a footnote.

**Verdict:** the design is **sound and graduates toward ADR-058** — the layer is
right, the ordering (B1→B2→B3) is right, freeze_h-kill is right. It is **not yet
ready for an experiment** until the readout is upgraded to measure calibration (not
just magnitude), the proper-score quarantine is operationalised, and the
chaos-premise + seq_len checks are done.

## 7. Methodological risks (register-compatible — for `register-risk`, not appended here)

| ID | Tier | Title | Trigger | Location | Narrative |
|----|------|-------|---------|----------|-----------|
| **M-RT1** | 2 | Stability term corrupts the proper score | Implementing B1/B2 with a fixed nonzero `L_stability` weight and reporting CRPS as the headline | `02_design` §7–8; `training_engine` loss assembly | Pushforward/GTF terms are regularisers, not proper scoring rules (Gneiting & Raftery 2007). A fixed weight moves the optimiser off the true predictive distribution — stability bought by bias. Must anneal→0 / keep small and report CRPS uncontaminated. |
| **M-RT2** | 2 | Truncated-horizon blindness (K<36) | Choosing K=12 and declaring stability from a ≤K readout | `02_design` §5 | TBPTT to K gives biased gradients and no stability certification for steps K+1…36; the runaway compounds in the tail. Either K=36 (checkpointed) or a falsifier that explicitly checks steps 13–36. |
| **M-RT3** | 3 | Chaotic-DS theory imported without establishing λ_max>0 | Adopting B2 and setting α via the σ̃_max bound as a correctness guarantee | `02_design` §4.2, §9-Q1 | GTF's `α*=1−1/σ̃_max` bound is justified only for chaotic systems (Hess/Mikhaeil 2022, not held). The conflict DGP is likely non-chaotic; the explosion is `expm1` out-of-range drift. Use α as heuristic gradient control, not theory; fetch Mikhaeil 2022. |
| **M-RT4** | 2 | Point-stability ≠ calibration (success-metric conflation) | First rollout-training experiment using only `diagnose_io_gain` as the readout | `02_design` §8–9; links ADR-057 | The runaway is a point/mean pathology; MCR is a calibration pathology. Fixing the attractor magnitude can leave — or worsen (mean-hedging/blurring) — calibration. Readout must include PIT/coverage + MCR/zero-rate, else "C-113 solved" is a category error. |
| **M-RT5** | 3 | `freeze_h` removal changes production inference path unguarded | Setting `freeze_h="none"` globally before re-evaluating golden_hour models | `02_design` §3 (Element 1); `hydranet_inference.py` | Removing the inference-time state-freeze alters the path for every model. Gate behind the `rollout_horizon=1` parity guard + a per-model golden_hour re-eval before merge. |
| **M-RT6** | 3 | New HPs interact with the open C-111 balancer instability | Enabling rollout training while the balancer freeze/active sweep is unresolved | `02_design` §9-Q4; ties C-111/C-124 | Stability-weight + K are tuned atop an already-unstable balancer; simultaneous unstable knobs confound attribution. Sequence after the balancer verdict. |

*(Tiers are proposals; `register-risk` owns final tiering, dedup, and linking — M-RT4
likely links the chronic-MCR cluster; M-RT5/6 link C-113/C-111/C-124.)*
