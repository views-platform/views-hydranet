# Loss arm — design notes

**2026-10-21.** ⛔ **These are NOTES, not a pre-registration.** Nothing here is committed, no
prediction is registered, no falsifier is binding. The prereg comes after the loss seam is read and
after Simon rules on the open questions in §6. Written because the whole discussion existed only in
one conversation, which is the failure mode `W-41` is logged for.

---

## 1. The claim this arm would test

> **The fields are over-smooth because the objective is minimised by a smooth field, not because the
> network cannot represent a sharp one.**

A loss evaluated **one cell at a time** is minimised by the **conditional mean** of that cell. The
conditional mean of *"a sharp spot somewhere near here"* **is** a smooth blob. ⇒ On that reading the
model is not failing; it is succeeding at the wrong objective.

## 2. The evidence pointing at the loss rather than the architecture

| source | what it establishes | strength |
|---|---|---|
| `Mathieu2016` C-3633/C-3634 | Enlarging the receptive field left multi-scale ℓ2 sharpness **identical at 24.7** while PSNR rose 26.5 → 27.6 ⇒ *"the loss function, not the architecture"* | ⭐ a controlled experiment, not an argument |
| `Ayzel2020_RainNet` | Trains with **`logcosh`** (≈ MAE / median) and smooths anyway; C-3654 reports PSD loss at 16 km and below | ⇒ it is **not** an MSE quirk; any per-cell loss does it |
| `Ravuri2021` DGMR C-3657/C-3659 | U-Net effective resolution 8 km at T+30, 32 km at T+90; 56 forecasters preferred the sharp generative model **88–89%**, and *"CSI does not penalise loss of small-scale structure"* | the sharp alternative exists and humans can tell |
| **our own D12** | the deficit is **monotone in scale** — body carries 0.391 of its due structure at 1 cell, 0.870 at 32; gate **0.112** at 1 cell | ⭐ the signature that mechanism predicts, measured on our own model |

⚠️ **What the evidence does NOT establish:** that *our* objective is the binding constraint. Three
other mechanisms remain live (BN recal fitted on 32×32 windows and applied to 360×720, global
`weight_decay` 0.1, recurrent **state** accumulation). D12 cannot separate them, and nor can this arm
— it can only show whether changing the objective moves the profile.

## 3. Three candidate forms

### A — a neighbour-gradient term (`Mathieu2016`'s own fix)
Penalise the mismatch between **neighbour differences** in the prediction and in the truth.
⭐ Targets exactly the band D12 measured — `d = 1` *is* a neighbour difference. Smallest change.
⛔ Our output is a **distribution** (gate × NB), not a picture, so something must be chosen to
differentiate: the conditional mean (risks de-calibrating — the *"rescales ≠ calibrates"* wall from
`project_body_knob_quest`) or a reparameterised sample (NB is Gamma–Poisson; `rsample` exists for
Gamma, implicit-gradient path unverified).

### B — a block-aggregate proper scoring rule ⭐ (the one to design)
Score the **aggregate over k×k blocks** at several `k` (1, 2, 4, 8), not only the single cell. Getting
the block distribution right at every `k` **is** getting the scale profile right.
⇒ Stays a **proper** scoring rule, so there is no sharpness-for-calibration trade by construction,
and it is closer to how forecasters verify (neighbourhood / fractions-skill thinking).
⛔ A bigger change to the loss plumbing than A, and the aggregate's distribution is only analytic in
special cases — a sum of NBs shares `p` only if `theta` is shared, which ours is not per cell.

### C — adversarial / DGMR-style
A discriminator on spatial patches. **This is what actually produced the sharp fields in the
literature.** ⛔ Not first: GAN instability on a 99.7%-zero field, a large engineering surface, and no
cheap kill gate.

## 4. ⭐ Why this is not the eighth failed firing intervention

**Four of the seven failures moved the same lever: how much the model fires** (M45 `truncated_nb`
×1170 → −0.238 AP; M62 act_ratio ×3.0; M64 ×56; the ×1.6/×5 pair). **This arm does not touch firing
volume at all** — it changes **where the loss looks**, which is the property D12 measured to be wrong
and which no intervention has touched.

⚠️ That is an argument, not a guarantee. It could be the eighth.

## 5. Gate or body? — ⛔ BODY. Reversed 2026-10-21, and the reason is about me, not the data

**An earlier version of this section read *"Lean: the gate first"*, on the strength of the gate being
*"3.5× worse than the body at one cell."* ⛔ That ratio is withdrawn** (`W-51`): `logit(gate)` and
`log mu` are different transforms of different quantities, each normalised by its own variance ratio,
so the comparison between them was never like-for-like. **The number that pointed this arm back at the
gate does not survive C-351 — the rule registered the same day I wrote it.**

⚠️ **And Simon named the pattern before I checked it, which is why this section is rewritten rather
than defended.** The record supports him, and it is countable:

| evidence | count |
|---|---|
| preregistrations named for the **gate** in the last seven (`pos_weight`, `gate_phase2`, `gate_underdispersion`) | **3** |
| preregistrations named for the **body** | **0** |
| §8.2 of `2026-10-20_d3_static_channel_ceiling.md`, whose own heading is *"The scale discriminator ran, and tested **the wrong head**"* — written one section after §8.1 established the body, not the gate, carries the slope defect | 1 documented instance |
| the reason D4 had to exist at all: **no saved artifact held `mu` on the full grid**, while the gate already had `p2_gate_calibration.py`, `t0_gate_vs_body.py` and `gate_reliability.py` | instrumentation asymmetric **before** any result was |

⇒ **A measurement bias and a research bias are indistinguishable from inside**: more instruments on
the gate produce more findings on the gate, which justify more instruments on the gate. ⛔ **So the
arm is BODY-first by decision, not by argument**, and the body half of the readout is the one that
decides.

**What still has to be true about the body half, pre-committed here:**
- the readout is the **body's** scale profile, `γ_model(d)/γ_truth(d)` over the same-shape null, at `k ∈ {1,2,4,8}`;
- ⛔ **the gate's profile is not read at all unless the body's moves.** If the body half is null, a gate improvement is not a result, it is a consolation;
- ⛔ a gate-vs-body comparison may only be quoted if both sides share a transform and a null. Today's do not.

⚪ **Kept from the old section, because they are still true and still arguments for the gate:** a block
version of the gate's loss is a natural proper object (*"did this k×k block contain any event?"*), and
`pos_weight` is closed on algebra so nothing competes there. ⇒ They are reasons the gate is *tractable*,
not evidence it is *where the defect is*. ⚠️ That distinction is exactly what the last three preregs lost.

## 6. ⛔ Open questions — these are what the prereg has to answer, and three need Simon

1. ⚪ **Unverified, and the first thing to read:** can `choose_loss` / `FamilyLoss` take an additional
   spatial term without touching the balancer? (`views_hydranet/utils/utils.py`,
   `views_hydranet/distributions/family_loss.py`.) If not, the arm is a training-engine change, which
   is a different size of job.
2. ⛔ **The term's weight is a free parameter** the frozen balancer will not tune. The standing rule is
   to **sweep** a free parameter before reporting a scalar — which multiplies the arm's cost by the
   number of sweep points. **Simon's call** whether to sweep or to fix it by a derivation.
3. ⛔ **Gate-only, body-only, or composed?** (§5.) **Simon's call.**
4. **Which `k`?** D12 says the deficit is ~closed by 32 cells and worst at 1, so `k ∈ {1, 2, 4, 8}`
   covers it. Needs no decision, but it is the one place D12's measurement sets the design.
5. ⚪ **Readout:** D12's scale profile recomputed on the new artifact — free, 391 s, **no fitted
   parameter**, and ⛔ **independent of AP**, which matters because C-352 put one of the platform's
   three AP constructions in doubt. FAO-02's CRPS + guardrails stay primary and unchanged.
6. ⛔ **FAO-02 is NOT at issue** and must not be presented as if it were: it locks **evaluation**
   metrics. A training objective is a different object, and the record already draws that line
   (`2026-10-15_loss_balance_prereg.md`). ⚠️ If anyone proposes scoring the arm *by* its own new loss,
   that **is** an FAO-02 change and is Simon's alone.

## 7. Cost, estimated and marked as such

⚠️ **Estimate, not a measurement** — the standing rule is that a cost is measured before it is quoted,
and nothing has run. By analogy with prior arms: **~$2.25 per arm at 300 lessons** on datafactory, one
retrain, control already in hand. ⛔ Multiply by the sweep in §6.2 if that is the ruling.

## 8. What would kill it

1. **Sharpness bought with calibration** — the profile improves while CRPS degrades. This project has
   hit that wall twice (`project_body_knob_quest`, `project_body_mask_sweep_negative`): *"a global
   point-dial cannot calibrate a heterogeneous body"*.
2. **The profile does not move at all** ⇒ the objective is not the binding constraint, and the three
   remaining mechanisms (BN recal, weight decay, recurrent state) become the live list.
3. **It moves the body and not the gate** ⇒ the worse half is untouched and the composed forecast
   probably does not improve.
