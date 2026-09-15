# 05 — Pre-analysis plan: the architecture bake-off (2026-08-25)

**Status: LOCKED.**

## ⚠️ Provenance caveat, stated first

Earlier dossiers locked their plan with `tools/` **empty**, so `git log` proved the analysis could not
have been written around a result. **That is not true here.** The harness, the six architectures, the
builder, the preflight and the smoke were all built first, and the smoke has already run. What `git
log` *can* prove is that **no scored 300-lesson arm existed when this was written** — the only arms on
disk are 2-lesson smokes, explicitly never scored, and `results/` contains no `score_*.csv`.

That is a weaker guarantee than the previous dossiers and it is recorded rather than dressed up.

## 1. Question

Our own measurements say the oracle gap is **placement**: on a converged vehicle, deleting 25% of a
real fed field costs **3%** of the oracle (`thin_0.75` 0.4807 vs 0.4974) while scrambling locations
costs **81%** (0.0925), and real occurrence with the model's own magnitudes reaches 0.4888 — so
magnitude is worth ~2%. Every lever pulled so far acted on *how much* the model emits, and **M45**
showed those losses scale with the dose. **The architecture is the untouched surface.**

**Does any of six architectural changes improve free-running gate skill over the incumbent?**

## 2. The one variable

`model` — the architecture. Verified single-variable: each arm differs from its **own-seed control**
in exactly `{"model"}`, asserted by the builder, by `preflight.py`, and again by `verify_bakeoff.py`
at run time.

## 3. Design, and its power — stated honestly

Six candidates × **2 seeds** (42, 43) at L=300, against the **existing** `fullzero_*` controls.

**This is a SCREEN, not a significance test.** At 2v2 the exact one-sided permutation floor is
`1/C(4,2) = 0.167`; no candidate can reach conventional significance here, and running six comparisons
makes that worse, not better. **No p-value will be reported.** The ITF pilot made exactly this mistake
readable by saying so up front, and M42 stands as a screen because of it.

The screen's job is to decide **which candidate earns a 4-seed test**, not to declare a winner.

## 4. Decision rule — registered before any arm runs

σ = **0.0134**, the measured control seed sd of AP@h18 (n=4, `fullzero_*`).

| outcome | condition |
|---|---|
| **PROMOTE** | both seeds ≥ control **+1σ** on AP@h18 **and** no body-guardrail regression (§5) |
| **REJECT** | both seeds ≤ control **−1σ** |
| **INCONCLUSIVE** | anything else — most candidates are expected to land here, and that is not a failure of the screen |
| **VOID** | any falsifier in §6 fires for that candidate |

Only **PROMOTE** buys a 4-seed follow-up. Ranking six candidates by point estimate and promoting "the
best" is explicitly forbidden — at 2 seeds and σ=0.0134 the ordering among INCONCLUSIVE candidates is
noise, and picking a winner from it is the garden of forking paths.

## 5. Guardrails — an AP gain alone does not promote

Reported at **h1/6/12/18/24/30/36**, both sides: gate (`AP`, `Brier`, `precision_at_k`, `act_ratio`,
`n_false_pos`) and body (`crps_all`, `crps_events`, `crps_none`, `size_ratio`, `mcr_*`,
`mag_on_false_pos`). Plus the **oracle** per arm, which separates *"the model got worse"* from *"the
rollout got worse"* — the distinction that made M45 interpretable.

**A candidate with an AP gain and a `crps_all` regression is a TRADE, not a win**, and does not
promote on the AP alone.

**Parameter counts are reported beside every result**: AntiAliasedPool +0.0%, FiLMSkip +0.2%,
DynamicTopSkip +0.6%, DualStream +1.9%, WideMemory +6.0%, **ShallowPool −16.3%**. ShallowPool is
asymmetric — a **win** there cannot be capacity, a **loss** might be.

## 6. Falsifiers — pre-committed, all mechanised

* **F1 identity** — every arm's `model` matches, re-asserted by `verify_bakeoff.py` independently of
  the builder's declaration. *(A `/falsify` audit showed the declaration alone is not enough.)*
* **F2 floor gate** — PASS on every arm; a FAIL means the vehicle cannot show an effect (C-299).
* **F3 setup integrity** — `arm_postflight.audit_arm` per arm: artifacts present, no NaN, `N` and
  `n_event` identical to the control (a differing support makes every paired comparison invalid).
* **F4 seed-matched controls** — each control must reproduce its published AP@h18 (0.3298 / 0.3318).
* **F5 h1 sanity** — no arm may lose h1 AP by more than σ. h1 is nearly teacher-forced; a large loss
  there means the architecture is broken, not that the rollout is.
* **F6 mechanism** — each candidate's distinguishing mechanism is pinned by
  `tests/architectures/test_candidate_mechanisms.py`, verified to fire against seven neutering
  mutations. A candidate cannot silently run as a clone of its control.

Any of F1–F5 firing makes that candidate **VOID**, never "worse".

## 7. Predictions — stated up front so this cannot be read as blind

* **AntiAliasedPool** — the mechanism is real and the remedy is free, but Zhang's reported gains are
  on classification consistency, not a rare-event rollout. *Small positive or null.*
* **WideMemory** — the memory is 0.5% of the model and the state freeze is our only win. *The most
  likely PROMOTE, and the one I would bet on.*
* **DynamicTopSkip / FiLMSkip** — the pair exists to separate primitive from content. *Null for both
  is the modal outcome; FiLM > raw concat if either moves.*
* **ShallowPool** — genuinely two-sided: preserves resolution, but `Islam2020` says position
  information accumulates with depth, so removing a stage may cost what it buys. *Null or negative.*
* **DualStream** — HRNet-lite is the most compressed of the six. *Null; a positive would argue for the
  faithful version, not for this one.*

**If five of six land INCONCLUSIVE, the screen worked.**

## 8. False-negative mode (C-307)

**2 seeds cannot distinguish a real small effect from noise.** An INCONCLUSIVE verdict closes nothing
— it says *"not worth 4 seeds yet"*, not *"this architecture does not help"*. **Reopen** any candidate
if a mechanism-level probe later shows it moves placement, or if a different vehicle/lesson count
changes the baseline. This is registered because the programme's habit is to drop things on a cheap
screen and rediscover them later.

## 9. Scope

L=300, `sb`, calibration, 2 seeds, one grid, one queryset. No training-loop change. The PRIO-GRID
equal-area distortion (`Radford2022`) is **out of scope** and recorded in `01_literature` as the
strongest candidate for the next programme.
