# Outcome — the composition arm, h=1

**Pre-registration:** `reports/2026-10-12_composition_arm_prereg.md` (+ Amendments 1–3)
**Arm:** `amber_kestrel`, `calibration_model_20261013_010739.pt`, sha `2a5efc2a67c08eac…`
**Control:** `violet_visitor`, `calibration_model_20261012_033112.pt`, sha `4acbf4d3c2b457b0…`
**Readout:** `r1_readout.py` → `r1_result.npz`. Cost: one pod run, ~$2.25.

## Verdict in one line

**No falsifier fired. The composition mechanism works exactly as designed, the primary
prediction missed, and the reason it missed is the finding: retraining moved the gate down.**

## Gates

| | result | |
|---|---|---|
| **F1** gate AP < 0.4131 | arm **0.5324** | does not fire |
| **F2** emitted positive fraction > 0.85 | **0.4999** | does not fire |
| **F5** config identity | byte-identical bar the triple | does not fire |
| **F6 / F6c** artefact + sha | digest MATCH; `73c3f42` resolves and declares the triple | does not fire |
| **F8** data vintage | `2026-09-21T03:11:06.788855+00:00 / 24471bd70ae98adb`, unmoved | does not fire |
| **F3 / F4** act_ratio, crps_all | **not measured** — h≥2 gate posterior deliberately not retrieved (#558) | open |

## Predictions

| | predicted | measured | |
|---|---|---|---|
| **P-1 (primary)** emitted positive fraction, 100 cells, h=1 | **[0.62, 0.76]** | **0.4999** | **MISSED** |
| **P-2** composed `P(Y=0)` | ≈ 0.31 | 0.5001 | missed, same quantity |
| **P-3** gate AP within ±0.05 of 0.5631 | ±0.05 | **0.5324** (−0.0307) | **HELD** |

## The mechanism works. This is the part that is not in doubt.

`truncated_nb` has `P(body = 0) = 0` by construction, so under `soft_gate` the emitted
`P(y>0)` should **equal the gate**. Measured on the emitted draws, independently of any config
file:

```
emitted positive fraction   0.4999
mean gate, same cell-months 0.5023      |difference| 0.0024
self_zeroed would have given 1.0000
```

The composition did precisely what #396 said it would. **The emitted map is now the gate's
opinion, and nothing else.**

## ⛔ Why P-1 missed, and why it is the result rather than a disappointment

P-1 was derived from the control gate's median **0.6945** on those cells. **That figure came
from P0, which swept months 421–455 — IN-SAMPLE.** The prediction was therefore derived
in-sample and tested out-of-sample. That is my error in the derivation, not the arm's failure,
and it is recorded as such.

The matched comparison — same 100 cells, same months 457–469, same statistic:

| | mean gate | median gate |
|---|---|---|
| control `violet_visitor` | **0.6050** | 0.6642 |
| arm `amber_kestrel` | **0.5023** | 0.5298 |
| delta | **−0.1027** | −0.1345 |
| observed | **0.6438** | — |

**The control's gate was already very nearly right out-of-sample (0.6050 against 0.6438).
Retraining under the new composition moved it down by 0.10 on exactly the cells that matter.**

The AP figures say the same thing in the other currency: pooled gate AP 0.5631 → **0.5324**,
and paired by month the loss is **systematic, not noise** — mean −0.0299, 95% CI
**[−0.0401, −0.0197]**, **0 of 13 months favour the arm**. Inside P-3's ±0.05 tolerance, so
P-3 held; but a tolerance holding is not the same as nothing happening.

So the map did fill, about half way:

```
control 0.3791  ->  arm 0.4999  ->  observed 0.6438      45.6% of the gap
```

against M1's analytic 76.3%. **The missing 30 points are the gate the retrain spent.**

## What this implies for the next move

**The gate was the asset, and retraining is what cost us.** Composing the *control's own*
gate — which is better than the arm's — would have emitted ≈ 0.6050 against an observed
0.6438, with no training at all.

**That is not expressible as a config today, and the reason is worth recording.**
`output_distribution` selects one family that serves **both** the training loss
(`FamilyLoss`, `utils/utils.py:186-187`) **and** the emit-time draw
(`DISTRIBUTION_REGISTRY`, `distributions/registry.py:36-44`). There is no axis that says "train
the body as `nb`, emit it zero-truncated". M1 computed that combination *analytically* from the
control's parameters and got 76.3%; the engine cannot currently *run* it.

So the candidate next step is **a new emit-time axis, no GPU** — separate the trained family
from the emitted one — and then recompose the existing 2026-10-12 control. That is a code
change to pre-register on its own, not an extension of this arm, and it is cheap in money and
not cheap in care: it widens a config surface that ADR-069 deliberately kept narrow.

## Not measured, by design

- **F3 / F4** need the gate posterior at h≥2. `pod_run_model.sh` archived only `lr_*`
  (views-models#558, C-161), and the h≥2 gate was 1.3 GB at 48 KB/s ≈ 7.7 h. The arm's readout
  was pre-registered h=1 only. Retrieving the horizon profile is a new decision, not an
  assumption.
- **The control's gate posterior is gone** — its pod is destroyed. Everything above is
  computable only because P2 saved the control's gate field to `p2_result.npz` on 2026-10-12.
  That file is now the sole copy of the control's gate at h=1 and should be treated as data.

## Correction carried in from the readout

The artefact's `.pt.config.json` sidecar holds **12 summary keys and does not record
`forecast_composition`**. My first readout treated its absence as evidence the pod had not
trained the arm, and aborted. Absence there is a property of the sidecar. The composition is
established two ways instead: the config at the recorded sha, and the measured
emitted-equals-gate identity above. **A guard that reads the wrong file fails exactly like a
defect.**

---

# ⛔ ADDENDUM — the arm was not a one-variable experiment, and F5 could not see it

Found while answering "why did the gate move?". **This supersedes the explanation above: the
gate degradation is not a general fact about retraining, it is a mechanism.**

## The mechanism

`distributions/truncated_negative_binomial.py:83-101` reduces the body NLL **over positive
cells only**, and says so:

> *"The reduction is a mean over `y>0` (the truncated law's support), NOT over all cells"* …
> *"`torch.where` keeps zero cells OUT of the backward graph (their term is never
> differentiated)"*

The control's dense `nb` body (`negative_binomial.py:54-61`) reduces over **all** cells.

The observed positive rate on land is **0.003448**. So the body task's gradient support went
from **100% of cells to 0.34%** — a factor of **290** — purely as a consequence of selecting
`truncated_nb`.

The two heads share the ConvLSTM/U-Net trunk. `freeze_multitask_balancer: True` on both sides
(verified at sha `73c3f42`) rules out adaptive reweighting, so the task weights are fixed; what
changed is what the body task *has to say* to the trunk. **The gate head's own objective was
untouched — BCE at `pos_weight` 2.0 on both sides — but its inputs were not.** That is a
complete and sufficient explanation for gate AP 0.5631 → 0.5324 and the −0.1027 level drop,
and it predicts the direction correctly.

## The silent override

Both configs declare `'body_supervision': 'all'`. Under `truncated_nb` that value is a
**no-op** — the family's own docstring (`:21-22`) states the mask is applied *inside* the
family regardless of what the caller passes. **The config asks for all-cell supervision and
gets 0.34%, with nothing raised and nothing logged.**

## What this does to F5

F5 ("differs anywhere but the composition triple → VOID") **passed**, correctly: the configs
*are* byte-identical bar the triple. But selecting `truncated_nb` changed a **second** thing —
the body's supervision support — and it changed it in the engine rather than in the config, so
**no config-identity check of any scope could have seen it.** F5 was not too narrow this time;
it was looking in the wrong layer.

**Fourth instance today, and the sharpest.** P1: was the reconstruction instrument trustworthy.
#552: a construction check scoped to one model list. F5/REGION: an identity check scoped to one
file. Here: an identity check that reads configs at all, when the variable lives in how the
engine treats a declared value. **In each the pre-registered question was right and the thing
reading it was looking at the wrong object.**

## What it does to the conclusion and the recommendation

**Strengthens the recommendation and changes its reason.**

The earlier reading — "retraining spent the gate" — is too general. The honest statement is:
**a retrain under `truncated_nb` necessarily starves the body's supervision by 290×, and the
gate pays for it through the shared trunk.** Nothing here says a retrain must cost gate skill;
it says *this* retrain had to.

Which makes the emit-time axis the right next move for a second, independent reason: emitting
an `nb`-trained body zero-truncated keeps **all-cell body supervision** *and* gets the
truncated emission. It is not merely cheaper than this arm — it is the only one of the two that
holds the body's supervision fixed, and therefore the only one that is actually the
one-variable experiment #396 intended.

**Registered consequence for the pre-registration of that change:** its F5 must be a
*supervision-support* check — assert the count of cells entering the body's backward graph is
unchanged between control and arm — not a config diff. A config diff is now known to be blind
to exactly this.
