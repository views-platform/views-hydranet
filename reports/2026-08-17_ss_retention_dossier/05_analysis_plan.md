# Pre-Analysis Plan — does correctly-configured scheduled sampling help or hurt rollout retention?

**Date:** 2026-08-17 · pre-registered **before** any treatment arm runs
**Companion to:** `reports/2026-08-14_scheduled_sampling_dossier/` (*take 1, VOID as run*),
`reports/postmortem_floor_limited_vehicle.md`, risk register **C-299** (floor), **C-300** (the
mismatch), **C-259** (the rule), views-models#404
**Status:** LOCKED. Stage A has run and **failed the gate**; Stage A′ is in flight. No treatment arm
has been trained.

---

## 1. Hypothesis

**H:** Scheduled sampling, configured as the codebase now requires (`ss_feedback='sample'`), changes
free-running rollout retention on a vehicle that has measurable retention.

**Direction, pre-registered:** SS **lowers** AP@h18. This is a one-sided test. The direction comes from
the roster observation — which, per §7, cannot itself settle the question.

## 2. Intervention (the ONE variable)

`ss_epsilon_max` ∈ {0.0, 0.5}, with `ss_feedback='sample'` in every arm. Everything else — architecture,
`output_distribution`, `forecast_composition`, `body_supervision`, data, partition, horizon set — is
held identical by construction: each arm is a **clone of `violet_visitor`** whose config is asserted to
differ from the floor's in *exactly* the intended key set (`make_ss_arm.py`, symmetric-difference check
on the resolved dicts).

## 3. Skepticism ledger

1. **This cannot settle the observation that motivated it.** The four SS-on roster models were trained
   before C-259 landed, with `ss_feedback` defaulting to `'mean'` — an **ungated** mean field in
   training against a gated sample at inference. That configuration is now forbidden and cannot be
   reproduced without disabling a guard. **A null here does not falsify the roster pattern**; it
   answers the forward-looking question only. Stated first because it is the easiest thing to forget
   once numbers arrive.
2. **The RNG stream is not held constant.** `training_engine.py:342` guards the SS branch with
   `if ss_epsilon > 0.0`, so at ε=0 the family sampler and its mask never execute. Every later draw is
   displaced in the ε>0 arms. ε=0 vs ε>0 is therefore not a *pure* single-variable contrast. With ≥3
   seeds the between-seed SD upper-bounds this nuisance; a placebo arm (`ε=1e-7`, sampler runs, mask
   effectively never fires) is pre-registered as a **contingency** if the observed effect is smaller
   than 2× the between-seed SD.
3. **One vehicle.** `violet_visitor` is a config outlier on six axes relative to the rest of the roster.
   A result here is about this vehicle.
4. **40 lessons is already excluded** (Stage A, §6) — so this runs at 160L, and 160L is *not* the
   300L the production board uses.
5. **Retention is a ratio of two noisy quantities.** Hence AP@h18 absolute is primary and retention
   co-primary, and both must agree in sign (§4).
6. **Goodhart.** `crps_all` is not an endpoint here; Epic #263 and M9 both show it is blind to the
   occurrence behaviour this experiment is about.

## 4. Endpoints and the test

| | |
|---|---|
| **Primary** | `AP_sb(h=18)`, free-running, 13 origins, identical support, pinned v2 truth |
| **Co-primary** | retention = `AP(h18)/AP(h1)`; **must agree in sign** with the primary |
| **Guard** | \|ΔAP(h1)\| ≤ 3 × MDE_AP(h1). If violated, SS damaged the anchor and "retention" is the wrong frame — report as a traded failure, not a retention result |
| **Test** | exact one-sided permutation on the **seed-level** values, direction pre-registered as *SS lowers AP@h18*, α = 0.05 |
| **Secondary** | paired sign test across matched seeds |

**Design: 2 ε × 4 seeds, not 4 ε × 1 seed.** The SS training multiplier is 3.32× and **flat in ε**
(measured: 3.27 / 3.22 / 3.47), so an ε point and a seed cost exactly the same. 1 seed × 4 ε yields no
error bar and admits no test; 4 seeds × 2 ε yields an exact permutation test with min one-sided
p = 1/C(8,4) = **0.014**. All four SS-on roster models sit at ε=0.5, so there is no observational signal
at 0.1/0.25 to justify estimating a curve for an effect whose existence is unestablished. **Dose-shape
only after an effect is demonstrated.**

## 5. The floor gate — binding, and it has already rejected one vehicle

Evaluated on the **control arm only**, after its cube is scored, **before any treatment arm launches**.
`scripts/floor_gate.py`, 19 tests, regression-checked against both archived score CSVs.

| clause | rule | binding |
|---|---|---|
| **FG-A** | `AP_ctrl(h18) ≥ 5 × prevalence(h18)` | yes |
| **FG-B** | `AP_ctrl(h1) ≥ 1.2 × AP_clim(h1)` | advisory |
| **FG-C** | `0.70 × AP_ctrl(h18) ≥ 3 × MDE_AP(h18)` | yes |

**θ = 0.30, h\* = 18, target `sb`. Threshold block md5 = `6d5714d5ceda147ed16f53143abe7e37`.**
A driver must refuse to launch treatment arms unless a `FLOORGATE_*_PASS` exists whose md5 equals that
string — **relaxing a threshold after seeing the control invalidates the token.**

The gate is **re-evaluated after Stage B on the sweep's own ε=0 controls**. If they no longer pass, the
sweep is `VOID` regardless of what the treatment arms did.

## 6. Stage A has already run — VEHICLE REJECTED, and it answered an open question

40L `nb`, ε=0, seed 42, a clone of `violet_visitor` differing in **exactly one key** (`total_lessons`):

```
FG-A [FAIL] AP 0.01962 / prevalence 0.009077 = 2.16x (need >= 5.0x)
FG-B [FAIL] AP(h1) 0.28887 / clim 0.29798 = 0.969x (want >= 1.2)
FG-C [PASS] a 30% effect is 0.01374 AP; 3x MDE is 0.00742
```

Retention **0.068**. There is no cheap vehicle, so the sweep runs at 160L (~26 h).

**Byproduct, and it is a real finding: training length is the dominant cause of the floor.** Cutting
160 → 40 lessons on an otherwise identical config collapses retention 0.54 → 0.068. The replication
dossier could not attribute this because `truncated_smoke` differs on three axes; this isolates it. The
residual 0.068 → 0.02 is what `truncated_nb` and `body_supervision` contribute.

**Note FG-C passed while FG-A failed** — the resolution was fine (MDE 0.0025), the *vehicle* was not.
The two clauses measure different things, which is the argument for keeping both.

## 7. Decision states — four, not two

| state | condition |
|---|---|
| **EFFECT** | p ≤ 0.05 **and** mean drop ≥ 3·MDE_AP(h18) **and** both endpoints agree in sign |
| **NULL** | p > 0.05 **and** the CI on the mean difference **excludes** θ = 0.30 |
| **UNDERPOWERED** | p > 0.05 **and** the CI **includes** θ |
| **VOID** | the post-hoc floor gate on this sweep's own controls fails, or any harness invariant fails |

This is what makes "no effect" distinguishable from "couldn't tell" — the distinction take 1 lacked, and
the reason its null was uninterpretable rather than merely negative.

**Censoring clause:** if any treated arm lands below `2 × prevalence(h18)`, the effect *magnitude* is
censored at the floor; report it as "≥ X", never as a point estimate.

## 8. Falsifiers (harness-level — any one voids the affected arm, not the hypothesis)

- **F1** an arm's config differs from the floor outside the intended key set ⇒ arm not built (asserted
  pre-GPU by `make_ss_arm.py`).
- **F2** `diagnostic_visualizations` is not `False` ⇒ refuse. Violet's is `True` and the per-origin
  biopsy costs ~28 min/origin ≈ **6 h per emit**.
- **F3** `N` ≠ 170430 or `n_origins` ≠ 13 in any scored row ⇒ arms compared on different supports.
- **F4** two arms share an identical AP at any horizon to 1e-12 ⇒ one cube was scored twice.
- **F5** two arms share a weight-tensor sha256 ⇒ the same model was evaluated twice. (Weight hash, never
  the `.pt` file sha — the nondeterminism postmortem records file shas as an invalid identity check.)
- **F6** repo HEAD differs across sweep arms ⇒ the arms are not comparable.

## 9. Scope

160 lessons (not the board's 300), one vehicle, one target (`sb`), 13 origins, S=16, calibration
partition, ε ∈ {0, 0.5} only. Per the standing rule of 2026-08-17, a positive is an **escalation
trigger** — second vehicle, dose shape — not a conclusion. And per §3.1: **whatever this finds, it does
not settle what the roster showed.**
