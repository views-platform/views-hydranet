# Pre-analysis plan — does removing the gate's improper tilt buy anything?

**Written 2026-10-16, BEFORE the arm has been trained.** One arm, one variable, one readout.

⛔ **Nothing launches on my say.** Simon approves the config change and the spend.

⛔ **A NEW pre-registration.** Not an amendment to `reports/2026-10-15_loss_balance_prereg.md`
(the denominator arm), which stands unrun and unchanged. Simon set this arm first on 2026-10-16;
the denominator arm follows.

**Arm naming:** views-models owns model codenames. This document calls it **the `pos_weight` arm**
and leaves the directory name to the `models` session.

---

## 1. Hypothesis

**H1.** The gate's training objective is **improper**, and that impropriety costs something.

`weighted_bce` at `pos_weight = w` has its optimum at

```
g* = w·p / (1 + (w−1)·p)        ≈ 2p  for small p at w = 2
```

so at `w = 2` the loss **instructs the gate to overstate the conflict probability by ~2×**. This is
Gneiting & Ranjan (2011) Thm 1: weighting a proper scoring rule by a function of the outcome makes
it improper, and the minimiser becomes a tilted law rather than the truth.

**But the gate does not obey.** P2 measured, over 842,634 cell-months on all 64,818 land cells at
13 out-of-sample h=1 months:

| | |
|---|---|
| median \|gate − p\| | **0.0054** ← the honest target |
| median \|gate − 2p/(1+p)\| | **0.0573** ← its own loss's target |
| median implied `w` | **1.0365** |
| verdict | **D2 — already calibrated** |

So the gate sits **at the truth and ~10× away from its own objective's optimum.** H1 is that
holding it there against an improper objective consumes something — shared trunk capacity, or
resolution — and that setting `w = 1` releases it.

⛔ **H1's own premise is also the strongest argument that H1 is false.** If the gate already emits
`p`, then at `w = 1` it is *already at the optimum* and nothing need change. §3.1.

---

## 2. The intervention — ONE variable

| | contrast (as-run `amber_kestrel`) | this arm |
|---|---|---|
| `loss_class_pos_weight` | **2.0** | **1.0** |
| `output_distribution` | `truncated_nb` | `truncated_nb` — unchanged |
| `forecast_composition` | `soft_gate` | `soft_gate` — unchanged |
| `body_nll_denominator` | `positives` (default) | `positives` — **unchanged** ⛔ |
| `freeze_multitask_balancer` | `True` | `True` — unchanged |
| `body_supervision` | `all` | `all` — unchanged |
| seed 42, 300 lessons, `window_dim` 32, `windows_per_lesson` 3, `sigmoid`, `min_events` 5 | — | identical |

⛔ **`body_nll_denominator` stays at its default.** The denominator arm is a *separate*
pre-registration and must not ride along, or neither arm is one-variable.

**Structure (decided by Simon, 2026-10-16):** a **second arm directory** whose declared control is
`amber_kestrel`, leaving the as-run arm intact. The alternative — editing `amber_kestrel` in place —
would permanently spend the `amber_kestrel ↔ violet_visitor` composition contrast, because that
control artefact cannot be retrained (**C-317**). Chain: `pos_weight arm → amber_kestrel →
violet_visitor`. ⚠️ The guard is **non-transitive** by `models`' design, so the chain's cumulative
distance from `violet_visitor` is a fact for the arm's docstring, not something a test asserts.

**No release needed.** `loss_class_pos_weight` is an existing key, live in 11 views-models configs.
None of 0.1.3's ordering machinery applies.

---

## 3. ⛔ Skepticism ledger

### 3.1 The honest prior is a NULL, and I am saying so with a number

If the gate already emits `p` under `w = 2`, then `w = 1` makes `p` the loss optimum too — and the
gate is already there. **The most likely outcome is that nothing moves.**

**My prior: ~30% that any endpoint moves beyond noise.** Stated so it can be scored.

That is a *lower* prior than I held for the denominator arm (~60%), and the reason is asymmetric
evidence: r4 **measured** the denominator's mechanism reaching the weights (shared-trunk gradient
ratio 0.312 → 10.497, `cos(∇body,∇gate)` +0.72 → −0.20, 75% of windows in conflict). **This arm has
no comparable measurement.** Its case is derivational — the objective is provably improper — and
theory about an objective is weaker evidence than a measured gradient.

### 3.2 ⛔ THE STRUCTURAL POINT: AP cannot move by the mechanism this arm is motivated by

**Average precision depends on the gate only through RANKS.**

The tilt `g* = wp/(1+(w−1)p)` is **strictly monotone in `p`** for any `w > 0`. So a *pure*
recalibration — the exact thing H1 says is wrong — **leaves every cell's ordering unchanged and
therefore leaves AP exactly unchanged.**

It follows that:

- **AP is the wrong primary endpoint for this arm.** If the only effect is the one the arm is
  designed to remove, AP is structurally incapable of showing it.
- **An AP change would be evidence of something else** — the altered loss changing what the trunk
  *learns*, i.e. a shared-capacity effect. That is a real and interesting outcome, but it is not
  H1's motivated mechanism, and it must not be reported as confirming H1.

⚠️ So the primary endpoint is **calibration**, which is where the motivated mechanism lands, and
which **FAO-02 already contains**: CRPS decomposes into Brier scores across thresholds, and the
**threshold-0 term is exactly the occurrence Brier score at weight 1** (Gneiting & Raftery 2007
§4.2; Hersbach 2000). No new metric is proposed — this is a decomposition of the locked one.

### 3.3 And the primary endpoint is nearly saturated

Calibration is already **0.0054**, against a base rate of 0.003448. There is very little room to
improve and the floor is not zero. **This arm is weakly powered on its own primary endpoint**, which
is the single biggest weakness in its design and the reason §7 treats a null as *informative*
rather than as a failure.

### 3.4 Seventh/eighth intervention on the same lever

Six interventions have failed and four failed on firing (M45 `truncated_nb` ×1170 → −0.238 AP; M62
×3.0; M64 ×56). ⚠️ This arm **lowers** the gate's eagerness, so if anything it should fire *less*,
which is the opposite direction from M45's failure mode. That is a genuine difference in kind. But
`act_ratio` is still a pre-registered guardrail (F4), because "different in kind" is what every one
of the six said.

### 3.5 The contrast has a measured handicap

The as-run arm's gate AP is **0.5324** against the control's **0.5631** — the composition arm lost
0.0307 of gate skill, which the denominator arm's evidence attributes to the loss imbalance. **That
imbalance is still present in this arm** (`body_nll_denominator` stays `positives`). So this arm is
measured against a damaged gate, and a recovery toward 0.5631 could be *either* effect. ⛔ F2 and
P-3 are therefore sized against **0.5324**, not 0.5631, and §7 pre-commits that an AP recovery here
does **not** license a claim about the denominator.

---

## 4. Pre-registered predictions

**Contrast: the as-run `amber_kestrel`** — `calibration_model_20261013_010739.pt`, sha
`2a5efc2a67c08eac…`, `truncated_nb`, git `73c3f42`. One variable against it.

| | endpoint (primary first) | prediction | threshold |
|---|---|---|---|
| **P-1** | **median \|gate − p\|**, all land cells, 13 h=1 months — the occurrence-Brier/calibration axis | **does not worsen**; stays **≤ 0.010** | > 0.010 ⇒ the gate left the truth, which H1 says is impossible |
| **P-2** | **median implied `w`** on the same bins | moves toward 1.0 or stays; **in [0.90, 1.15]** | outside ⇒ the gate is not tracking its new optimum |
| **P-3** | gate AP, **paired by month**, against the as-run arm's **0.5324** | **unchanged — within ±0.010** | ⛔ a *prediction of no change*, see §3.2 |
| **P-4** | `crps_all` | **does not worsen**; improves by ≤ 2% if at all | > 5% worse fails FAO-02's bar |
| **P-5** | emitted positive fraction, 100-cell set, h=1 | stays in **[0.55, 0.85]** | the composition is unchanged |
| **P-6** | `act_ratio` at h18 | **≤** the as-run arm's | a *decrease* is the expected direction |

⚠️ **P-3 is the unusual one and it is deliberate: I am pre-registering that the headline occurrence
metric does NOT move.** If it does, §7 says that is a different finding, not a louder version of
this one.

---

## 5. Falsifiers — pre-committed

| | fires if | then |
|---|---|---|
| **F1** | median \|gate − p\| **> 0.010** | **KILL.** The gate left the truth under an objective whose optimum *is* the truth. H1's model of the gate is wrong. |
| **F2** | paired-by-month gate AP is **below −0.010** against the as-run arm | **KILL.** Removing an improper tilt cost occurrence skill — which would mean the impropriety was load-bearing. |
| **F3** | `crps_all` worsens by **> 5%** | fails FAO-02's own bar in the wrong direction. |
| **F4** | `act_ratio` at h18 **>** the as-run arm's | ⛔ unexpected direction. This arm lowers eagerness; firing more would mean the knob does not do what its own loss says. A **finding**, not something to tune away. |
| **F5** | emitted positive fraction leaves **[0.55, 0.85]** | the composition moved when only the gate loss was supposed to — a confound. **VOID.** |
| **F6** | the arm's config differs from `amber_kestrel` anywhere but `loss_class_pos_weight` | **VOID** |
| **F7** | artifact arrives without all three files + `PROVENANCE` with `lessons ≥ 300` and `mode: production` | **VOID** — reuse `2026-10-12_underfire_diagnosis/vehicle.py`'s gate verbatim |
| **F8** | training diverges, NaNs, or the stability gate trips (C-184 BatchNorm seed-bimodality) | **VOID, not negative** (C-307) |

⛔ **THE KILL CRITERION, stated once and plainly:** **F1 or F2.** Either means the improper tilt was
doing work the model needed, and `pos_weight` is not a free correctness fix. On either, the arm is
reported as **KILLED** and `pos_weight` returns to 2.0 for all subsequent arms.

⛔ **Pre-committed consequence.** n=1 SCREEN at 300 lessons. Against the measured AP seed sd ≤0.019
and the design's demonstrated paired-by-month half-width of ~0.010, **an outcome where every
endpoint holds its prediction is a NULL, and a null here is INFORMATIVE, not a failure** — see §7.
It is *not* a pass that licenses shipping `pos_weight = 1.0` fleet-wide without a seed replication.

---

## 6. Method

- **Contrast:** the as-run `amber_kestrel` artefact (§4). Not `violet_visitor` — that would be two
  variables.
- **Primary slice h=1**, the only matchable one (teacher-forced in both input and state,
  `hydranet_inference.py:1219-1221`). h18/h36 read **only** for F4/P-6.
- **Statistic:** paired by month for AP (half-width ~0.010, demonstrated in `OUTCOME.md:67-68`);
  pooled mean for the positive fraction. ⛔ **Never the median over cells** — quantised at 1/16.
- ⛔ **Never mix two statistics.** Calibration figures are medians over bins with n ≥ 1000;
  AP figures are paired-by-month means. The two are not comparable and must not be put in one row.
- **Reuse P2's harness verbatim** for P-1/P-2 — it is the instrument that produced 0.0054 and
  1.0365, and re-deriving it would make the comparison incommensurable.
- **Standing gates** (#311 precedent): `floor_gate.py` FG-A/FG-C on the control first; the potency
  pre-flight on the arm's own config **and** at a trained checkpoint (C-324/C-325);
  `arm_postflight.py`; `arm_identity_check.py`; the weight-hash post-condition; `kill_tree` (C-326).
- ⛔ **Set `trajectory_log_path`** to an absolute path under the pod's deliver directory. 0.1.3
  writes one `loss/<task>` column per task and creates the parent; a relative path dies with the
  pod (views-models#558 / C-161).
- **Budget: one pod, ~3 h, ~$2.25.**

### 6a. ⚠️ The pre-flight for this arm is NOT the denominator arm's

The denominator arm's F1 was a loss-magnitude assertion. This arm changes no reduction, so that
gate is silent here and would pass vacuously. The free pre-flight for *this* arm is instead:
**confirm from the built loss that `pos_weight` is 1.0 and that nothing else moved** — i.e.
`arm_identity_check.py` plus reading `crit_cls.pos_weight_value` off the constructed criterion.
Cheap, and it is the only check that can catch the config key not reaching the loss.

---

## 7. Decision rules

| outcome | decision |
|---|---|
| **every prediction holds (the expected case)** | **NULL, and informative.** It closes the "the gate's objective is miscalibrated" explanation for the gate's under-reach, leaving the **trunk/representation** as the remaining candidate — which is where the denominator arm's measured evidence already points. Record it; `pos_weight = 1.0` becomes the defensible default on correctness grounds alone (the objective is proper), **not** on skill grounds. |
| **F1 or F2 fires** | ⛔ **KILL** (§5). `pos_weight` returns to 2.0 everywhere. |
| **P-1/P-2 improve AND P-3 holds** | the cleanest possible result: the tilt is gone, occurrence ranking is untouched, the delivered probabilities are now honest. Ship `pos_weight = 1.0` as a correctness change after a seed replication. |
| **P-3 moves (AP changes beyond ±0.010)** | ⛔ **A DIFFERENT FINDING, and it must be reported as one.** Per §3.2 the tilt is monotone and cannot move a rank statistic, so an AP change means the altered loss changed what the trunk *learned*. That is a shared-capacity result, it does **not** confirm H1, and it must **not** be read as evidence about the denominator arm — whose imbalance is still present here (§3.5). |
| **F3/F4** | recorded as the pre-registered trade; F4 in particular is a finding about the knob. |
| **F5/F6/F7/F8** | VOID. Re-run or abandon; never reported as negative. |

---

## 8. Out of scope

- **The denominator arm.** Separate pre-registration, runs after, `body_nll_denominator` stays at
  its default here.
- **Per-target `pos_weight`.** The glossary names it (§3) and it is a real axis; one variable at a
  time.
- **M-1** (the balancer's LR never reaching its param group). Latent — this arm keeps the balancer
  frozen — and it needs its own change.
- **The joint-vs-marginal question.** Our loss is a sum of per-target NLLs, i.e. a product of
  marginals, and conditional independence across violence types is assumed and untested. Simon
  raised it; it is larger than this arm and it is recorded, not addressed.

## 9. Definition of done, and the stop rules

**Done when:** a number for P-1 through P-6; every falsifier marked fired or not; one sentence
saying whether removing the improper tilt bought anything; a `found, not chased` list.

- ⛔ **One arm. No sweep.** `pos_weight` is not to be scanned.
- ⛔ **Nothing added mid-run.** Discoveries go on `found, not chased`.
- ⛔ **F1/F2 are hard stops** on the readout; **F8** is a hard stop on interpretation.
- ⛔ **A null is INFORMATIVE here** (§7) — which is *not* the same as C-307's "inconclusive". The
  distinction is deliberate: a null confirms a *derived* prediction (§3.1), rather than failing to
  detect a hoped-for effect.

## 10. Glossary

Checked against `reports/GLOSSARY.md`. **`pos_weight`** is locked (§3, "the gate's eagerness knob
(higher = fires more)"), as is **weighted_bce**. No new term introduced; no edit needed.
