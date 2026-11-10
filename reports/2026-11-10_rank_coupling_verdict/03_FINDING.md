# FINDING — is B worth building? ⚠️ ANSWER 3: not yet, and the missing piece is not the one we expected

**Epic #579, story S9 #588.** Pre-registration: `PREREG.md` + amendments **A-1**, **A-2**, all
committed before the number they bear on existed. Verdict rule: `PREREG.md` §6, fixed in advance.

⛔ **Zero GPU, zero training, no production code touched.** ~6 minutes of CPU across five
instruments, all of which re-run from committed artifacts.

---

## 1 THE VERDICT, IN THE PRE-REGISTERED FORM

> ### ⚠️ **ANSWER 3 — "Not yet, and here is what's missing."**

**What we would say to a colleague:** replacing the body's magnitudes with a rank-coupled empirical
draw **does carry real information** — the mechanism is measurable at the ordering quality the body
already has, at **8.1 σ̂**, in **42 of 42** arm-fold combinations. But it **does not beat the
incumbent**, and the reason is now a number rather than a guess: **what the gain rides on is
event-versus-zero separation, which is the gate's job, not magnitude ordering among events, which is
the body's.** Two targets, both measured:

| target | we are at | needed | gap |
|---|---|---|---|
| ordering among event cells, `ρ_event` | **+0.3127** ± 0.0713 | **+0.4441** = λ\* | **+0.1314** |
| separation on the open set, AUC | **0.6764** | ≈ **1.0** to beat A on CRPS | large |

⭐ **And the pointed detail:** the **gate** already orders event magnitudes at **+0.4418** ± 0.0583
— *at λ\**. The body sits a tenth below it. The ordering the design needs is already in the model;
it is just not in the half of the model the design proposed to change.

⛔ **Answer 3 is a result, not a hedge.** It was declared in advance (`PREREG.md` §5 V6, §6) exactly
so that a null could not be read as "B is dead". Four previous definitions of done lacked it.

---

## 2 WHAT EACH READOUT SAID

| story | readout | result |
|---|---|---|
| **S2 #581** | reproduce the chain | ✅ `ap_ctl` **0.5667056384496834**, `\|Δ\|` **0.000e+00** at tol 1e-12; both committed `.out`s bit-identical ⇒ **V1 cleared** |
| **S5 #584** | within-B, 7 folds | ✅ REAL ordering **−3.5763** fair-CRPS vs the noise ordering, CI **[−4.24, −3.06]**, **7/7** folds. Controls: no-op **+0.1257** (spans 0) ⇒ **V3 PASS**; oracle **−13.27** ⇒ **V4 PASS** |
| **S5 #584** | B against A | ⛔ B **loses** by **+8.52** CRPS, 7/7 folds. The oracle only **ties** A (−1.16, spans 0) |
| **S6 #585** | the promotion criterion | ⛔⛔ **VOID.** Exact no-op reads **0.000e+00** ⇒ machinery sound; oracle moves the **wrong way** (−2.47) ⇒ **V4 fires**, the statistic is blind |
| **S7 #586** | σ̂, in-run, `ddof=1` | **σ̂ = 0.4441**, **margin = 0.8881**, measured across **6** control seeds before any comparison. Effect **4.0155 = 9.04 σ̂** |
| **S8 #587** | λ\* | **λ\* = ρ_event +0.4441**, bracketed **(+0.3965, +0.4441]** ⇒ **V6 FIRES** ⇒ answer 3 |

---

## 3 ⭐⭐ THE MECHANISM, AND WHY IT IS THE EPIC'S REAL OUTPUT

The two dials separate cleanly, and they say opposite things:

| dial | holds fixed | moves | B beats A on CRPS? |
|---|---|---|---|
| **λ** (`PREREG` §1) | nothing — it is a **separation** dial (A-2) | AUC 0.507 → **1.000** | ✅ **yes, at λ=1.00**: −13.27 vs A's −12.11 |
| **ν** (A-2) | AUC **exactly** 0.6764 (`G-L3`, 0.000e+00) | `ρ_event` 0.313 → **0.998** | ⛔ **no, not even at ν=1.00**: −11.53 vs A's −12.11 |

⇒ **A perfect ordering of event magnitudes, at the body's own separation, still loses to A. A
perfect separation wins.** The rank-coupled draw's value is almost entirely in *knowing which cells
are events*, and only secondarily in *ordering them*.

⚠️ This is consistent with what the platform already measured and is the third independent arrival
at it: `thin_0.75` keeps 97% of the oracle after deleting 25% of events while scramble costs 81%
⇒ **recall is free, placement is everything**; and `M99`/`M100` found the **gate** out-ranks the
delivered draws on 10/10 arms. ⭐ The gate out-ranking the body is not a curiosity — on this
measurement it is the whole design.

### Why B loses to A at all

A's open set is **2,256** cell-months over 7 folds, of which only **45.83%** are events. B assigns a
strictly positive empirical draw to **every** open cell, so it pays for the 54% that are not events.
A predicts a literal point mass at zero on **128** of them and wins those outright. ⇒ B's cost is
structural, not a tuning failure.

---

## 4 ⛔ WHAT IS NOT CLAIMED

- ⛔ **Not** that B should replace A. On the only criterion with demonstrated power B loses.
- ⛔ **Not** that B is dead. The mechanism is real and measurable; the constraint is located.
- ⛔ **Not** that the promotion criterion said no. It said **nothing** — it is VOID, and a void
  instrument is not evidence in either direction.
- ⛔ **Not** anything about h36, teacher-forced inference, or `mu` (unavailable — `PREREG.md` §9).
- ⚠️ **n = 1 vehicle for the headline** (`ctlfullzero_fortytwo`); σ̂ spans 6 control seeds, and the
  within-B sign held in **42/42** arm-fold combinations.
- ⚠️ λ\* is bracketed at the **0.05** grid spacing of ν, i.e. to ±0.024 in `ρ_event`. Not finer.

---

## 5 ⛔ PREDICTIONS — LOGGED, AND ALL THREE WERE WRONG

Registered in `PREREG.md` §7 before the run. ⚠️ Base rate: **WRONG 137 / RIGHT 89**.

| # | predicted | measured | verdict |
|---|---|---|---|
| **P-1** | B does **not** clear 2 σ̂ at the real head's ordering | clears at **8.1 σ̂** on the criterion with power. ⚠️ On the amended §2 criterion it does *not* clear (0.592 < 0.888) — but that criterion is **VOID**, so reading P-1 as RIGHT from it would be crediting a prediction to a broken instrument | ⛔ **WRONG** |
| **P-2** | λ\* at `ρ_event` **0.80–0.95**, above the real ordering of **0.5125** ⇒ answer 3 | λ\* = **0.4441**; the real ordering is **0.3127**, not 0.5125. The **consequence** (answer 3) was right; **both numbers** were wrong, and the stated range does not contain the answer | ⛔ **WRONG** |
| **P-3** | the recovered-regression ranker and the gate ranker are **not separated** | they separate: CRPS gaps **−3.60 [−4.26, −3.06]** vs **−5.60 [−6.42, −4.66]**, non-overlapping | ⛔ **WRONG** |

⭐ **0 of 3.** The one thing that held was the *structure* that made a null informative — the two-ended
dial and the declared third outcome — not any quantity I predicted.

---

## 6 ⚠️ FIVE DEFECTS THE INVESTIGATION MEASURED (registered in S10 #589)

1. ⛔ **The λ dial cannot move the quantity λ\* was stated in.** Its oracle term is a global rank
   over a 99.63%-zero field, so all events sit in the **top 0.37%** of it (sd 0.0021 against noise
   sd 1.0). `ρ_event` is flat from λ=0.00 to λ=0.95 and jumps at λ=1.00. ⇒ 2026-11-04's
   *"Spearman 0.68 at lam=0.75"* was a **separation** figure, not an ordering one.
2. ⛔ **`argsort(argsort(x))` is not a rank** on tied data — it is a function of **cell order**. It
   made `G-L3` fail at 1.571e-03 on an invariant that cannot be violated, and read the oracle's
   Spearman as 0.9963 instead of 1.0. A's prediction is quantised at 1/16, so ties are the norm.
   Fixed with midranks; `G-L3` now **0.000e+00**.
3. ⛔ **The 2026-11-04 simulation's gate was Bernoulli(π) while A ships `threshold_gate`** — so
   *"same gate, one varying thing"* was false, and its REAL gap of **−0.0137 [−0.1138, +0.0826]**
   is a Bernoulli, one-month number. Under A's own deterministic gate over 7 folds the same contrast
   is **−3.5763 [−4.24, −3.06]**. ⇒ **The published null was an instrument artefact.**
4. ⛔ **The `ranks FIXED` ablation held nothing** — `key = np.sort(key)` makes `argsort` the
   identity, assigning magnitudes in **cell-index order**. Its column was the largest spread in the
   output and was an artefact (flat 3–6 across deciles; corrected it rises 0.116 → 155.459).
5. ⚠️ **τ = 0.20 was an unverified transcription** in two docstrings. Now **read** from the arm
   configs: all **11** arms agree, and the transcription was right. ⚠️ τ is an **unswept prior**,
   deliberately set below this model's measured calibrated 0.26.

---

## 7 ⭐ WHAT THIS MAKES THE NEXT QUESTION

⛔ **Not** "build B". ⛔ **Not** "B is dead".

1. ⭐ **The gate already clears λ\*** (+0.4418 vs +0.4441) **and the body does not** (+0.3127).
   ⇒ The cheapest live question is whether a rank-coupled draw ordered **by the gate** is worth
   anything — it needs no new training, and `M99`/`M100` already said the gate out-ranks the draws
   on 10/10 arms. ⚠️ It is also the arm this epic did **not** pre-register as a promotion candidate,
   so it needs its own pre-registration, not a paragraph in this one.
2. ⚠️ **The promotion criterion is still missing.** `PREREG.md` §2's statistic is VOID and this epic
   deliberately did **not** replace it a second time (that is the loop that produced four withdrawn
   definitions of done). A criterion that can see **placement within a bin** must be designed and
   pre-registered **before** any arm is scored against A. ⇒ `C-3572`'s `MCB − DSC + UNC` split and
   `DSC_ORD`/`R*` are the candidates; both are report-only today.
3. ⛔ **Separation, not ordering, is the binding constraint** — and separation is the gate's. That
   relocates the whole programme's next move away from the body's magnitude head.

⚠️ **This epic answers the question it asked and does not authorise an arm.** Promotion needs a
criterion with demonstrated power, and there is not one yet.
