# 02b — Expert method-review of the T>0 Rollout Skill ruler (design `02`)

**Date:** 2026-07-25 · **Read-only critique.** Precedes pre-registration (`05`). Panel is task-selected on
the **evaluation/scoring** axis + **rollout-dynamics** dissent. The user is the chair, not a seat.

## 1. Target & decisions under review

An **instrument** (not a model): a frozen ruler scoring a 36-step AR rollout against `truth[o+h]` per
horizon, reusing the lodestar CRPS/AP/Brier. Decisions: **(D-a)** the skill scalar (crps-all vs CRPSS);
**(D-b)** the two-rollout decomposition (free-running vs teacher-forced-oracle) as an exposure-bias/ceiling
separator; **(D-c)** the baselines (climatology + persistence); **(D-d)** the origin-set / support; **(D-e)**
the deeper validity — does CRPS on the per-horizon marginal honestly credit a *diffuse-but-calibrated*
rollout, or does the 99.7%-zero DGP make crps-all a Goodhart trap. **DGP:** monthly conflict deaths,
~99.7% structural zeros, heavy positive tail, strong spatial+temporal autocorrelation; the rollout is a
recurrent ConvLSTM feeding its own output back (io-gain>1 on OOD dense input = the bloom).

## 2. Panel (6 seats; fault lines covered)

| Seat | Why seated | Fault line |
|---|---|---|
| **Tilmann Gneiting** | proper-scoring purist; CRPS author-adjacent | D-a, D-e (properness) |
| **Rob Hyndman** | forecasting practice; skill-scores; recursive-vs-direct | D-a (opposes Gneiting), D-d |
| **David Salinas (DeepAR)** | ancestral-sampling rollout = our literal blueprint | D-b, feedback-content confound |
| **Sepp Hochreiter** | LSTM author; recurrent state dynamics | D-b (state-drift confound) |
| **Yann LeCun (JEPA)** | diffuse-marginal-is-correct (steelman, not the chair) | D-e (opposes Gneiting's emphasis) |
| **Operational VIEWS archetype** | production/partition/baseline reality | D-c, D-d, partition discipline |

Live opposition: **D-a** Gneiting⟂Hyndman; **D-b** Salinas+Lamb (valid) ⟂ Hochreiter (confounded);
**D-e** Gneiting (properness handles it) ⟂ LeCun (crps-all scalar still penalizes honesty on this DGP).

## 3. Library grounding (held ✓)

- Proper scoring: **Gneiting2007**, **Matheson1976**, **Brocker2007**, **Machete2012**, **Thorarinsdottir2013**. ✓
- Zero/extreme domination + the fix: **Lerch2017** (Forecaster's Dilemma), **Taillardat2023** (twCRPS). ✓
- Exposure bias / teacher-forcing / ancestral rollout: **Salinas2020/2019** (DeepAR), **Lamb2016** (Professor Forcing), **Gasthaus2019**. ✓
- Horizon-indexed forecast-hub eval: **Bracher2021** (WIS, epidemic hub), **Jordan2019** (scoringRules). ✓
- Ensemble AR rollout scored per lead-time: **Price2023** (GenCast). ✓  · Climatology-as-reference: **Candille2005**, **Bessac2021**. ✓
- Conflict-forecasting system / persistence / partitions: **Hegre2019 (ViEWS)**, **vonDerMaase2025**. ✓
- Recursive-vs-direct + skill scores: **Makridakis2020 (M4)**. ✓  · Prob U-Net (diffuse dense output): **Kohl2018**. ✓

**Gaps to fetch:** a spatial-count CRPSS-vs-lead-time precedent (metric shape); a clean "teacher-forced =
predictability-ceiling vs one-step-error-ceiling" statement (we argue it below from Lamb2016 + first
principles, but a citation would harden `03.F.2`).

## 4. Independent critiques

### Gneiting — score the ensemble, properly; crps-all alone is Goodhart-prone here
1. **Score the ENSEMBLE, never the mean.** CRPS on the emitted D×K sample cube is *strictly proper*
   (Gneiting2007): it rewards a calibrated-diffuse forecast and punishes an overconfident-sharp one — so the
   "diffuse mean" worry is a category error *provided the scored object is the sample cube*. The bloom is a
   *mean-feedback* artifact; the *scoring* must consume samples. **Make this explicit + add a guard test**
   that the scored object is the cube, not `E[y]`.
2. **crps-all is proper but zero-dominated ⇒ a bad headline scalar on a 99.7%-zero DGP.** A conservative-zero
   rollout scores well simply because almost every cell is zero (the T=0 crps-none domination is already on
   record). This is the **Forecaster's Dilemma** in reverse (Lerch2017). **Fix = add threshold-weighted CRPS
   (twCRPS, Taillardat2023)** up-weighting the event region, and keep crps-events separate. Headline should
   be twCRPS + the crps-none/events split, not crps-all.
3. **CRPSS is not itself proper** — it's a reference-dependent monotone transform, fine for *ranking at a
   fixed horizon* but not decomposable and not to be optimized. Report raw CRPS for decisions. **Add the
   CRPS decomposition (reliability/resolution/uncertainty, Hersbach) + a per-horizon PIT/coverage** — that,
   not the scalar, tells you *timid-but-stable vs honestly-diffuse*.

### Hyndman — is the recursive rollout even the product? and your N is tiny
1. **Recursive vs direct (the reframe).** Recursive multi-step *accumulates error by construction* — the
   bloom is that pathology. In forecasting practice (M4, Makridakis2020) **direct h-step** models routinely
   beat recursive at long horizons. If the deployable product could be a *direct* h-step forecast, the whole
   rollout-skill ruler measures a strawman. **Seat a direct-h forecast as a baseline** — "does the recursive
   rollout beat a direct-h forecast?" is the question that decides whether to fix the rollout at all.
2. **Pro-CRPSS (against Gneiting) for the crossover.** Operationally you need a horizon-comparable skill
   score vs a naive benchmark (the MASE ethos). Report **CRPSS = 1 − CRPS/CRPS_clim** as the *communication*
   axis for the crossover plot — raw CRPS values aren't legible across horizons.
3. **|O| ≈ 12 origins with overlapping 36-month futures = severe temporal autocorrelation.** Per-cell CRPS
   means at h=36 have an *effective* sample size far below 12×N_cells. **iid-over-cells bootstrap CIs will be
   wildly overconfident — use a block bootstrap over origins.** No significance claim survives without this.

### Salinas (DeepAR) — you're scoring the WRONG rollout as "deployed"
1. **Mean-feedback is not how you roll out a probabilistic recursive model** — DeepAR (Salinas2020) rolls out
   by feeding back **ancestral samples**, not the mean. The on-disk "free-running" rollout feeds back the
   *emit-mean* → it is a **broken rollout by construction**, and its bloom is partly an artifact of the wrong
   method. **Scoring it and calling it "the deployed rollout's skill" measures a strawman.** The honest
   deployed object is the **ancestral (sample-feedback) rollout** — which is exactly the H-SAMPLE probe. ⇒
   the "GPU-free first result" scores the *current-broken* rollout, not deployed skill; the true
   free-running arm needs the sample-feedback re-run. **This collides with the roadmap's Phase-2 claim.**
2. **The oracle IS a valid exposure-bias reference** (teacher-forced vs free is the canonical exposure-bias
   gap, Bengio2015 / Lamb2016) — keep it. **But relabel it:** it's a *one-step-conditioned ceiling*, not a
   "predictability ceiling." If the one-step model is itself biased, the oracle inherits that bias; it upper-
   bounds *rollout* skill given the trained one-step map, not intrinsic predictability.

### Hochreiter — the free−oracle gap is not *pure* exposure bias
1. **State-drift confound.** Free and oracle differ in the fed-back *input*, but the ConvLSTM hidden state
   `h_t` evolves from that input *every step*. So `gap(h)` = input-exposure-bias **⊕ the divergent
   hidden-state trajectory it induces** — not cleanly "the fed-back value." To isolate you'd need an arm that
   feeds back the prediction but teacher-forces the *state* (or vice versa) — the retired `freeze_h` ablation
   was exactly that and was found *inert vs the runaway* (the bloom rides the input path, not the state),
   which is reassuring but should be **cited as the reason the gap is interpretable**, not assumed.
2. The oracle's boundedness partly reflects *in-distribution inputs* (io-gain ≤ 1 on realistic inputs) —
   which is precisely the quantity we want the ceiling to capture. Consistent; state it.

### LeCun (JEPA) — don't let a zero-dominated scalar penalize honesty
1. **At long h the true marginal IS wide** (many futures consistent with the past). A *correctly wide*
   ensemble is right even if its mean looks blobby; the failure of mean-feedback is that it **collapses to
   the mean and feeds that back**, destroying multimodality. Agrees with Gneiting that a proper score on the
   *ensemble* handles this — but **emphasizes the opposite risk to Gneiting's**: crps-all as the scalar, on a
   99.7%-zero DGP, will rank a timid τ≥0.8 rollout *above* an honestly-diffuse ensemble, i.e. **penalize
   honest uncertainty**. So the calibration read is not optional garnish — it is the headline.
2. **Add a per-horizon calibration/spread diagnostic** (PIT histogram / coverage of the ensemble; does spread
   grow correctly with h?). The scientifically interesting quantity is *marginal calibration vs horizon*, not
   point error. (Kohl2018: a distribution over dense outputs, scored for calibration, is the right frame.)

### Operational VIEWS archetype — partition discipline + the strongest baseline
1. **BLOCKER — partition leakage.** Origins with a full 36-month future sit at the *early* edge of
   calibration (≈457–468). FAO-02 locks eval to the **validation** partition. If these origins/months were
   seen in training, the skill is optimistic and the ruler is not trustworthy. **Verify the origin set
   respects the train/validation boundary (Hegre2019 partition discipline) before pre-reg.**
2. **DQ4 — add the mixture baseline.** white_ranger (climatology) is the right *floor* reference (VIEWS
   uses it), but the **mixture baseline (red/green/yellow_ranger)** is stronger and already exists. A rollout
   that beats white_ranger but loses the mixture isn't operationally useful — score against the *strongest*
   reference (Bracher2021 hub ethos), not a strawman.
3. **The crossover horizon is a genuinely useful, policy-legible deliverable** ("to what lead-time does the
   model beat climatology?"). Endorse the framing; note the production version needs the rolling-origin
   protocol, not a 12-origin research read.

## 5. Key disagreements (the product)

- **D-a — crps-all vs CRPSS.** *Gneiting:* raw proper CRPS + decomposition; CRPSS not proper, comms-only.
  *Hyndman:* CRPSS as the horizon-comparable headline. **Merit both** → report raw CRPS/twCRPS for
  *decisions*, CRPSS for the *crossover visualization*. Neither is optimized on.
- **D-b — is the oracle the ceiling?** *Salinas/Lamb:* valid exposure-bias reference — but a *one-step*
  ceiling, not a predictability ceiling. *Hochreiter:* the gap is confounded by state-drift. **Both right:**
  keep the oracle, **relabel** it, and **hedge** the gap as "input-exposure-bias ⊕ induced state-drift"
  (cite the inert `freeze_h` result as evidence the input path dominates).
- **D-e — does CRPS honestly credit diffuseness?** *Gneiting:* yes, if you score the ensemble.
  *LeCun:* yes on the ensemble, but **crps-all as the scalar still penalizes honest diffuseness on a
  99.7%-zero DGP** → the calibration read + twCRPS are the headline, not crps-all. **LeCun's caveat wins
  the emphasis:** a stable-but-timid rollout must not be allowed to look like skill.
- **THE BIG ONE — Salinas's feedback-content confound.** The on-disk free-running arm is *mean-feedback* =
  the wrong rollout method. **The GPU-free "first result" scores the current-broken rollout, not deployed
  skill.** This is the review's sharpest finding and it revises the roadmap.

## 6. Synthesis & recommendation

**The ruler is sound in skeleton (per-horizon, identical-support, frozen-scorer reuse, faithfulness h=1) —
but four revisions are load-bearing before pre-registration:**

1. **Fix the scored scalar (D-a/D-e).** Headline = **twCRPS (event-weighted) + crps-none/events split +
   per-horizon PIT/coverage**; crps-all demoted to context; CRPSS only for the crossover plot. This directly
   defuses the Goodhart/timidity trap. *(Gneiting + LeCun + Lerch2017/Taillardat2023.)*
2. **Reframe the "free-running" arm (Salinas).** Score **three** rollout arms, honestly labeled: (a) *current
   mean-feedback* (on disk, GPU-free) — labeled **"current deployed (mean-feedback) — a broken rollout"**,
   NOT "free-running skill"; (b) *ancestral sample-feedback* (the H-SAMPLE re-run) = the **true deployed
   object**; (c) *teacher-forced one-step-conditioned ceiling*. The GPU-free read is still worth doing as a
   *diagnostic of today's behavior*, but the **skill verdict waits for the ancestral arm.** This merges the
   old "Phase 2" and the H-SAMPLE probe into one coherent step.
3. **Verify partition discipline (Operational — BLOCKER).** Confirm the 36-future origin set is on the
   validation side of the train boundary before any skill number is trusted.
4. **Baselines + stats.** Add the **mixture baseline** and a **direct-h forecast** baseline (Hyndman's
   recursive-vs-direct question); compute CIs with a **block bootstrap over origins** (never iid over cells).

**Order:** (Phase 1 loader unchanged) → verify partition (#3) → build the scored scalars (#1) → run the
GPU-free *current mean-feedback* diagnostic (honest label) + baselines → **then** the ancestral + oracle
arms (#2) for the skill verdict and the bloom-cost decomposition.

**Strongest dissent to keep live:** *Hyndman's recursive-vs-direct.* If a direct-h forecast beats the
recursive rollout at every h>1, the entire bloom-fix program is misdirected — we should ship direct-h and
stop trying to tame the rollout. The direct-h baseline is cheap insurance against spending the epic on the
wrong object.

## 6b. Chair's resolution (binding — 2026-07-25)

The panel's advice is faithful; the chair rules on two points, overriding the panel where the locked
framework or the cost accounting says so:

- **twCRPS + PIT are OUT (FAO-02 locked, previously tested negative).** The panel (Gneiting/Lerch) would add
  twCRPS for the zero-domination guard and PIT/coverage for calibration; both are in FAO-02's **rejected**
  set and the lab already found them unhelpful. **Ruling:** the Goodhart guard is the **crps_all /
  crps_events / crps_none split** (affirmed) + the locked **Brier / MCR / QS99** guardrails — NO twCRPS, NO
  PIT, NO LogScore. Per-horizon calibration is read via **MCR** (a locked guardrail), not PIT. twCRPS may
  return ONLY after a fresh test re-establishes its usefulness. This supersedes the twCRPS/PIT wording in §4
  (Gneiting/LeCun) and revises **C-c** and **C-d** below.
- **Direct-h is NOT a cheap Phase-2 baseline — it's a parked architectural alternative.** Cost accounting:
  recursive = 1 model, 36 sequential inference passes, accumulates error; "direct = 36 models" = **36×
  training** (the chair's original, correct reason to choose recursive); "direct = 1 multi-horizon decoder"
  = ~1× training, 1 inference pass, no accumulation — but a real HydraNet architecture change, not a
  baseline. **Ruling:** the recursive rollout was the right pragmatic start. The scientific question Hyndman
  raises (is error-accumulation the problem?) is **already answered by the free−oracle exposure-bias gap** —
  a large gap *is* the accumulation cost. So we do NOT build a direct baseline now; direct-multi-horizon is
  **parked as an alternative that a large oracle gap would motivate.** This revises **C-g** to Tier 4 (a
  deferred architectural option, not a missing baseline). Hyndman's dissent stays *live* but *deferred*.

## 7. Methodological risks (register-compatible — for /register-risk) — as revised by §6b

- **C-a (Tier 2)** — *Partition leakage in the rollout origin set.* Trigger: pre-registering/​scoring the
  free-running curve before confirming origins are validation-side. Location: `02_design §2`, `03 §C-G4`.
  Narrative: 36-future origins sit at the calibration edge; if in-sample, all skill numbers are optimistic
  and the ruler is untrustworthy. Verify vs the train/validation boundary (Hegre2019).
- **C-b (Tier 2)** — *Scoring mean-feedback as "deployed rollout skill."* Trigger: reporting the GPU-free
  re-score as the model's rollout skill. Location: `04_roadmap Phase 2`. Narrative: mean-feedback is not the
  correct ancestral rollout (Salinas2020); its bloom is partly method artifact. Relabel; gate the skill
  verdict on the ancestral arm.
- **C-c (Tier 2)** — *crps-all Goodhart on the 99.7%-zero DGP.* Trigger: using crps-all as the headline
  skill scalar / reporting a single crps-all number as "skill." Location: `02_design §2 metrics`. Narrative:
  zero-domination lets a timid conservative-zero rollout outscore an honestly-diffuse ensemble (Lerch2017).
  Guard = the **crps_all/events/none split** + locked **Brier/MCR/QS99** guardrails read per horizon (NOT
  twCRPS/PIT — FAO-02 rejected, chair-ruled §6b). Never headline crps-all alone.
- **C-d (Tier 3)** — *Ensemble-vs-mean scoring guard missing.* Trigger: the scorer consuming `E[y]` instead
  of the D×K cube. Location: G1 loader. Narrative: proper scoring requires the sample object; add a guard
  test (Gneiting2007). (Calibration read via MCR, not PIT — §6b.)
- **C-e (Tier 3)** — *Small, temporally-autocorrelated origin set → overconfident CIs.* Trigger: any
  significance/KEEP claim with iid-cell bootstrap. Location: `02_design DQ2`. Narrative: |O|≈12 overlapping
  futures; use block bootstrap over origins.
- **C-f (Tier 3)** — *free−oracle gap mislabeled as pure exposure bias.* Trigger: attributing the whole gap
  to the fed-back value. Location: `02_design §3`. Narrative: gap = input-exposure-bias ⊕ state-drift
  (Hochreiter); relabel oracle "one-step-conditioned ceiling"; cite the inert freeze_h result.
- **C-g (Tier 4, deferred)** — *Recursive rollout may not be the optimal product (direct-multi-horizon
  alternative).* Trigger: a LARGE, growing free−oracle exposure-bias gap that persists after the
  sample-feedback fix — i.e. accumulation is intrinsic to recursion. Location: `02_design §6 / 02b §6b`.
  Narrative: recursive was the right pragmatic start (1 model, 36× cheaper training than 36-model direct,
  horizon-flexible). A single multi-horizon decoder avoids accumulation at ~1× training / 1 inference pass
  but is an architecture change, not a baseline. The oracle gap already diagnoses whether accumulation is
  the problem; this option is parked until that gap motivates it. (Makridakis2020.)
