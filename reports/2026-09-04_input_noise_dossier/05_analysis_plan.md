# 05 — Pre-analysis plan: input-noise SCREEN (LOCKED)

**Date:** 2026-09-04 · **Epic:** #311 · **Story:** S0 (#312) · **Builds on:** #308 (closed),
`SanchezGonzalez2020_GraphNetworkSimulators`, `Aceituno2025_TemporalHorizons`, M45, M50, M62/M63.

**Written before S1 (#313) has produced a single number.** Amendments are appended as dated
`## AMENDMENT An` blocks, never edited in place.

> **PROVENANCE — checkable, not asserted.** "Pre-registered" is a claim about *ordering*, so it is
> made verifiable rather than stated. This file entered the repository in its own commit, before any
> file exists under `results/` and before S1 (#313) ran. Verify:
>
> ```
> git log --diff-filter=A --format='%h %ad %s' -- reports/2026-09-04_input_noise_dossier/05_analysis_plan.md
> git log --diff-filter=A --format='%h %ad' -- reports/2026-09-04_input_noise_dossier/results/ | tail -1
> ```
>
> The first must precede the second. **C-303's fourth occurrence was a provenance document
> overstating its own provenance** — a plan claiming its falsifiers were "pre-committed … enforced in
> code before any result was read", when the enforcing tool entered git *in the same commit as the
> score CSVs*, four hours after the run. The check above is what makes this document different from
> that one.

## THIS IS A SCREEN, NOT A RESULT

n=1 per arm against **~20% training variance** (C-119/C-184). On a control `AP@h18` of ≈0.31 that is a
noise band of roughly **±0.06** — so **only an effect larger than that is visible at all**. A null here
is **INCONCLUSIVE, not negative**. This is stated first because **C-307** is already on the register
for cheap screens repeatedly recorded as closures, and because the floor-limited post-mortem's own
self-audit reads: *"I wrote 'INDICATIVE, one vehicle' in every scope section and then reasoned as
though it were not — the caveat was ritual, not load-bearing."*

## 1. Hypothesis

**H:** Corrupting the training inputs with a perturbation *matched to the model's own measured
free-running error* improves free-running skill (`AP@h18`) relative to a clean-input control, because
the model stops depending on inputs being perfect and its own errors stop taking it off-distribution.

**H₀:** It makes no difference outside the noise band, or it hurts.

## 2. Intervention — exactly one variable per arm

| arm | variable vs control |
|---|---|
| **control** | — (ε=0, no noise, no pushforward) |
| **noise** | `+ input noise` (design from S2, scale from S1) |
| **pushforward** | `+ pushforward_weight > 0` — built, merged, audited 20/20, **never run** |

The pushforward arm reuses its own pre-registration
(`reports/2026-08-26_pushforward_dossier/05_analysis_plan.md`) and is **not** a second test of H. It is
a same-family comparator: it answers "is the noise result better or worse than the thing we already
own", which a lone noise-vs-control contrast cannot.

## 3. Skepticism ledger — what would make this fail, written now

1. **The firing lever.** M45: AP loss scales with how much the model fires, and four other
   interventions have died here. Any noise that *manufactures occurrence* is expected to cost AP. This
   is the single most likely failure mode.
2. **The sparsity floor.** The field is ~99.94% zero and `log1p` has a floor at 0. Additive noise
   creates off-manifold negative log-counts and dense spurious signal (M50 already records fed
   magnitude going negative).
3. **The design rests on one measurement, one vehicle.** If S1's distribution is unstable, the
   parameterisation is a guess wearing a number.
4. **BatchNorm.** Noise changes BN batch statistics; C-184 already records BN as a source of
   seed-bimodality. A difference at the BN layer is not a difference from the hypothesis.
5. **Scheduled sampling is measured harmful** (M30–M33, and #308). Both arms therefore run **ε=0**, so
   this is a clean-input vs noised-input contrast, not a noise-vs-SS one.

## 4. The measure, and the noise floor — named in advance (C-320)

**Primary:** `AP@h18`, target `sb`, free-running, standard 13-origin support.
**Noise floor:** ~20% training variance ⇒ **±0.06** on a control near 0.31. Emit-only seed spread is
sd ≈ 0.0075 (M56) and is *not* the relevant floor here, because these arms are retrained.
**Secondary, reported always, never used to override the primary:** `AP@h1`, `AP@h36`, `act_ratio`,
`size_ratio`, `crps_events`.

## 5. The S1 → S2 design-selection rule — committed BEFORE S1 runs

S1 measures, per horizon and per origin: the fraction of truly-active cells the model **silences**
(FN), the fraction of truly-zero cells it **fires on** (FP), and the magnitude error on cells active in
both.

**Selection:**

| S1 finding | S2 builds |
|---|---|
| FN dominates (model goes silent) | occurrence **dropout** — silence true events in the input |
| FP dominates (model over-fires) | occurrence **injection** |
| FN ≈ FP and magnitude error dominates | **magnitude-only** jitter on already-active cells |

"Dominates" = the larger rate exceeds the smaller by **≥ 2×** at h18. If neither does, the
magnitude-only arm is built, because it is the only option that cannot manufacture occurrence — and
manufacturing occurrence is skepticism-ledger item 1.

**⛔ STOP-gate (a) — proceed only if the design has an evidential basis.** The dominant rate's
coefficient of variation across the 13 origins must be **≤ 0.5**. Above that, the distribution is not a
stable target and S2 does **not** proceed on a plausible-looking parameterisation. *This threshold is a
judgement call, made in advance and recorded as such rather than derived.*

**Accumulation:** random-walk across steps, not i.i.d. — the paper's own ablation found random-walk
best *"as an attempt to simulate accumulation of error in a rollout."* Per-step σ is scaled so the
**terminal** variance is the controlled quantity, as the paper does.

**No target adjustment.** The paper adjusts targets only because it predicts an integrated quantity;
it states this *"happens implicitly when the loss is defined directly on next-step ground-truth"* —
which is our case.

## 6. Decision rule — committed now

Let `Δ = AP@h18(arm) − AP@h18(control)`, evaluated separately for the noise and pushforward arms.

| outcome | reading | next |
|---|---|---|
| **BRANCH 0** — an arm produced no scoreable artifact | **VOID, not negative.** No Δ is quoted, estimated or implied | fix and re-run; a crash is never evidence about H |
| **Δ ≥ +0.02** | survives the screen | buy the 4-seed run |
| **Δ ≤ 0** | did not help at n=1 | **INCONCLUSIVE, not "noise does not work".** State the band; do not close H on one seed |
| **0 < Δ < +0.02** | inside what this design can resolve | **INCONCLUSIVE.** Not "promising" |

BRANCH 0 is listed **first and checked first**. #308's rule enumerated three numeric outcomes and had
no branch for the arm crashing — which is exactly what happened. That is C-320's fourth instance and
it is fixed here in advance rather than recorded afterwards.

## 7. Falsifiers — pre-committed

| | fires when | consequence |
|---|---|---|
| **F1** | the noise-off path is not byte-identical to the pre-flag model | implementation invalid; **stop**, do not run |
| **F2** | the potency gate shows the knob inert **at a trained checkpoint** | **no GPU is spent** (C-324 Tier 1, C-325) |
| **F3** | `floor_gate` fails on the **control** arm | the vehicle cannot resolve the effect we came to measure; screen **VOID** before treatments run |
| **F4** | any two arms' trained weight hashes are identical | **VOID** — the treatment was inert (C-324) |
| **F5** | the noise arm's `act_ratio` rises while `AP@h18` falls | not a noise-specific failure — **M45 again**, the fifth-plus confirmation that firing is not the lever. Record it as such, not as "noise is bad" |
| **F6** | S1's dominant error rate has CV > 0.5 across origins | STOP-gate (a); S2 does not proceed |

**A falsifier that fires kills the branch it guards. It is documented, not rescued.**

## 8. Pre-flight, all blocking

See `03_harness_and_invariants.md` §D. In short: byte-identity proven by test; H1 tested against a
**synthetic** config carrying statics (no arm in this fleet has any — measured at S0); potency gate
passing on the arm's own config *and* at a trained checkpoint, and demonstrated able to **refuse**;
`floor_gate`, `arm_identity_check`, `arm_postflight`, weight-hash and `kill_tree` wired into the
launcher; full suite and lint green.

## 9. What this screen cannot answer

Whether noise helps **at 4 seeds**. Whether **direct multi-horizon** (#310) — the road
`Aceituno2025_TemporalHorizons` actually argues for — is better; that is the next epic. Whether a
*different* noise family would work, if the one S1 selects does not. And nothing here revisits
BPTT-SA, which is closed.
