# Pre-analysis plan — the control vehicle

**Written 2026-10-24, BEFORE the run.** views-models#593. One `nb` control training run, ~$2.25,
300 lessons. Not an arm and not a hypothesis test: this run exists to be a **vehicle** — an
artifact later work can be measured *on*.

⛔ **Nothing launches on my say.** Simon approves the config edit and the spend.

---

## 0. Why this document exists

#593's Acceptance checklist was audited on 2026-10-24 (`/falsify`, probe P5) and **two of its
items are not satisfiable as written**:

- It requires "Standing gates: floor gate". `scripts/floor_gate.py` documents **FG-C as binding**
  and as *"the clause that names the real failure"*, and FG-C needs `theta` — a **pre-registered
  effect** — plus `mde_ap`. A control run pre-registers no effect, so FG-C cannot be evaluated for
  it. Omitting `mde_ap` yields `PROVISIONAL`, which is *"the state 2026-08-14 was actually in, had
  anyone asked"* (the module's own docstring).
- It names `kill_tree`, which is **C-326** — a technique, not a script. `tools/podrun/pod_run_model.sh`
  contains no process-group handling. ⚠️ Probably N/A for a foreground pod run; recorded as
  **unverified**, not as a finding.

Thirteen pre-registrations sit in `reports/`; none was for #593. So the single artifact that #433,
#437 and the body scoreboard's re-baseline all depend on had **no VOID rule** — while **C-184**
(recurrent-BatchNorm seed bimodality, ~40% bad basins) is a registered, live way for this exact run
to come out unusable with every log looking clean.

This plan supplies the readout and the VOID rules. It does **not** invent a `theta`.

## 1. What the run is for

A `nb` vehicle whose provenance is established, so that measurements taken on it can be placed.

⛔ **Why not an existing artifact.** Five `nb` checkpoints are on disk (2026-08-12/13) and all five
are **VOID** under `reports/2026-10-12_underfire_diagnosis/vehicle.py` F6: no `PROVENANCE` beside
any of them, so lessons, region and REHEARSAL-status are unknowable; and their `dropout_rate` is
0.125 against the delivered 0.15, with no `forecast_composition` at all (they predate ADR-069).

## 2. The configuration — ONE model, TWO instrumentation changes

Vehicle: **`heavy_freighter`** (the proven global/datafactory template; `bright_starship` is the
admissible alternative and would change nothing in this plan but the name).

| | delivered config | this run |
|---|---|---|
| `h_init` | `'abs_rand_exp-100'` | **removed** |
| `trajectory_log_path` | unset | **`/workspace/deliver/heavy_freighter/trajectory.csv`** |
| `output_distribution` | `nb` | `nb` — unchanged |
| `forecast_composition` | `soft_gate` | `soft_gate` — unchanged |
| `loss_class_pos_weight` | 2.0 | 2.0 — unchanged |
| everything else, incl. seed and 300 lessons | — | identical |

⛔ **`h_init` is a deletion, not a change of behaviour.** `init_hTtime` returns `torch.zeros(...)`
unconditionally; the historic implementation was `abs(randn(...)*exp(-100))` and `exp(-100) =
3.72e-44`, below float32's smallest normal, so the two were never distinguishable. The key is read
nowhere in views-hydranet (grep, zero hits) and is swallowed by `extra="allow"`.

⛔ **`trajectory_log_path` is ONE switch, not two.** #593 says "two things must be ON". There is no
separate per-task logging flag: `training_engine.py` reads `config.get("trajectory_log_path")` as
the only gate, and the six per-task columns follow from it automatically. #593 overstates it.

**No hyperparameter is tuned.** That is what makes this a control.

## 3. ⛔ Skepticism ledger

| # | The worry | Why it is pre-committed, not waved away |
|---|---|---|
| S1 | A bad BatchNorm basin (C-184, ~40%) produces a vehicle that cannot rank, and nothing says so | **V7** below: FG-A is binding and self-contained in the control's own score row |
| S2 | The trajectory file is written but its per-task cells are blank — the instrumentation silently off | **V5**: a named VOID rule, checked on the file, not on the config |
| S3 | The pre-flight passes and the field is still inert (C-324) | **A4**: `scripts/potency_check.py` at the trained checkpoint. The pre-flight explicitly does not cover this and says so |
| S4 | I score it with whichever CRPS estimator flatters it (C-358, FAO-02 is silent) | **§4** pre-commits the estimator *and* requires both be reported |
| S5 | The vehicle is compared against a reference of different ensemble width | **§4**: C-330's equal-S rule, which is necessary but *not sufficient* |
| S6 | This run becomes "the control" for arms it was never matched to | **§8**: what this does not establish |

## 4. Pre-committed readout

**Ruler:** the FROZEN T=0 lodestar, `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`.
Identical months and cells on both sides, by construction.

**Targets:** all three — `lr_sb_best`, `lr_ns_best`, `lr_os_best` (FAO-02 requires all three).

**Primary readout horizon: h = 18.** ⚠️ **The one value here chosen on precedent rather than
derived** — M42, M45 and `floor_gate`'s own validated examples all read out at h18. **Simon to
confirm or change before the run**; changing it changes V7's threshold and nothing else.

**Statistics, per the locked glossary:** `crps-all`, `crps-events`, `crps-none`, `size-ratio`,
`pos-mcr`, and gate AP.

⛔ **CRPS estimator, pre-committed (C-358 / #442).** The **fair** estimator, `1/(m(m−1))`, is
primary. The plug-in `1/m²` must **also** be reported beside it. Reason: the plug-in bias is
`≈E|X−X′|/(2m)` **per arm** and differed **5.5×** across two arms on 2026-10-22, moving a headline
from −5.94% to −16.27% and inverting its sign-of-interest. Reporting one alone is not admissible
here. ⚠️ This pre-commitment binds **this run only** — FAO-02 is LOCKED and Simon's alone to amend.

⛔ **Ensemble width.** Any reference must be scored at the **same S** as this vehicle's cube
(C-330). Equal S is **necessary and not sufficient**, so the width is recorded in the readout rather
than assumed.

⛔ **Population, transform, estimator named with every number** (C-351). A statistic over the grid
median and a statistic over gate-weighted cells are different numbers — on 2026-10-22 they differed
**24,000×** on θ — and may not be mixed.

## 5. ⛔ VOID rules — pre-committed

A VOID run is **not a negative result**. It is re-run or discarded; it never enters a ledger as
evidence. Each rule names its source.

| ID | VOID if | Source |
|---|---|---|
| **V1** | no `artifact/PROVENANCE` beside the checkpoint, or its `mode` is REHEARSAL, or its `lessons` ≠ 300 | `vehicle.py` F6, verbatim |
| **V2** | no config sidecar beside the artifact, or the sidecar's config is not identical to the run config | `vehicle.py` F6 (amended) |
| **V3** | the file's sha256 does not match PROVENANCE's | `vehicle.py` |
| **V4** | the scored support is not the global land grid (360×720). ⛔ A 13,110-cell support is `africa_me_legacy` and a different experiment | `vehicle.py`; #499 Step 1 |
| **V5** | `trajectory.csv` absent, **or** its six per-task columns are blank for every lesson | the capability #593 exists to turn on; the failure the 0.1.7 pre-flight fix was written for |
| **V6** | any non-finite loss at any lesson | C-212 (float32 `log_prob_zero` NaN-backward) |
| **V7** | **FG-A FAILS**: gate AP at h18 < `5.0 × prevalence(h18)`, with `prevalence = n_event / n_cells` read from the control's **own** score row | `floor_gate.py` FG-A, binding, `r = 5.0` (the module default, and the multiplier its validated PASS/FAIL pair uses) |
| **V8** | the BatchNorm recalibration pass ran under any augmentation | C-328, five recorded instances |

⛔ **V7 is the C-184 detector.** It is the only rule here that can catch a bad basin, and it is
self-contained in the control's own scored row — which is what makes it checkable before any
reference exists.

## 6. The floor gate for a control, stated honestly

| clause | status for THIS run | why |
|---|---|---|
| **FG-A** (range) | **BINDING** — V7 | `AP_ctrl(h*) ≥ R·prevalence(h*)` needs nothing but the control's own row |
| **FG-B** (reference) | advisory, run if a climatology reference at equal S exists | the module already calls it advisory |
| **FG-C** (power) | ⛔ **NOT EVALUABLE** | it requires a pre-registered `theta` and an `mde_ap`. A control pre-registers no effect. The gate will return `PROVISIONAL`, and that is the **correct** verdict, not a failure to be worked around |

⛔ **Therefore: every future arm that uses this vehicle must supply its own `theta` and `mde_ap` and
evaluate FG-C itself, before spending.** This plan does not discharge that obligation and cannot.
The live noise floor to size against is the **≤0.019 seed sd** (M34–M37), and the demonstrated
paired-by-month resolution is **±0.010**. ⛔ **Not** C-119's "~20% training variance" — C-119 is
RESOLVED and that figure is a PRE-FIX measurement (C-357).

## 7. Acceptance — what makes this a usable vehicle

- [ ] **A1** Pre-flight, **on the pod, after the install, before training**, exit 0, output captured:
      `python -m views_hydranet.infrastructure.pod_preflight --config models/heavy_freighter/configs/config_hyperparameters.py --require trajectory_log_path`
      ⚠️ Run it from **outside any checkout** — from inside the repo, the source tree shadows the
      installed wheel and the gate reports on the wrong tree (reproduced accidentally 2026-10-24).
      ⛔ Requires views-hydranet **≥ 0.1.7**; `0.1.6` checks only that the schema *declares*
      `trajectory_log_path`, not that the config *sets* it. `~=0.1.1` resolves to 0.1.7 (verified).
- [ ] **A2** No VOID rule fired (§5), each checked and recorded, including the ones that passed
- [ ] **A3** `scripts/arm_postflight.py` and the weight-hash post-condition
- [ ] **A4** `scripts/potency_check.py` at the trained checkpoint — the C-324 gap the pre-flight
      declines to cover
- [ ] **A5** The readout of §4 computed and written, with both CRPS estimators
- [ ] **A6** `PROVENANCE` beside the artifact — the absence that made five `nb` checkpoints unusable

## 8. ⛔ What this run does NOT establish

- **It is not a matched control for any arm.** It is matched to its own config. An arm that differs
  from it in two keys is not a one-variable comparison, and the composition programme has already
  paid for that mistake once.
- **It proves no field is wired.** A2's pre-flight is a declaration-and-setting check; A4 is the
  potency check; neither proves a knob changes a forecast (C-324).
- **It settles nothing about the body.** The body's zero mass, θ's lack of an interior optimum and
  the truncation-inverts-placement finding (M67) are untouched by having a vehicle.
- **A null on anything measured here is INCONCLUSIVE, not negative** (C-307): n=1 at 300 lessons.

## 9. Open before launch — Simon's two

1. **Confirm h\* = 18**, or name another. It sets V7's threshold.
2. **Confirm the vehicle** is `heavy_freighter` rather than `bright_starship`.

Everything else in this plan is pre-committed as written.
