# CONCLUSION — the gate under-dispersion axis

**Date:** 2026-10-19 · **Status:** ⛔ **CLOSED for measurement.** One result outstanding (H2, below) and
**the decision does not wait on it.**
**Evidence:** `RESULTS.md` (complete measurement record, §0–§12) · `PREREG.md` (14 amendments) ·
`RND_PLAN.md` · `../2026-10-16_pos_weight_arm_outcome/OUTCOME.md` rev 3
**Written by:** the views-hydranet seat. Every row below points at the section of `RESULTS.md` that
carries its evidence. Nothing new is measured here.

---

## The decision, in one line

**The defect is real, one fix exists and is banked unapplied, the cause is not ours to find yet — so the
axis closes and the body starts now.**

---

## ⛔ The decision table

Every row has a verdict **and** a decision. A row without a decision is bookkeeping, which is the
objection that produced this document.

| # | Question | Finding | Evidence | ⛔ DECISION |
|---|---|---|---|---|
| 1 | Is the gate under-dispersed? | **YES.** `b ≈ 1.1875` on amber (shipped), `1.3596` on copper. Six fits, ten rounds of two-seat review. `b` is a **lower bound** | §1 | **Recorded. Stop measuring it.** The fact is established to more precision than any use we have for it |
| 2 | Is it an artefact of MC-dropout averaging? | **NO.** The reporting path explains **~8%**; the compression is in the network | §3 | **Closed.** Do not revisit the averaging path |
| 3 | Can `pos_weight` address it? | **NO, at any value.** `logit(s*) = log(w) + logit(p)` exactly — a pure intercept shift, **provably zero slope effect**, per-target or shared | §10, OUTCOME §2.4/§8.1 | **CLOSED ANALYTICALLY. `pos_weight` stays at 2.0** by the pre-registered rule — not because 2.0 is right. ⛔ Do not go looking for an honest value; there isn't one |
| 4 | Does re-estimating BatchNorm fix it? | ⛔ **NO — it overshoots and inverts.** `b` 1.1875 → **0.7793**; pooled mean gate 0.00417 → 0.01981 (**5.75× truth**) | §6 | ⛔ **DEAD, and a GOODHART BAN stands** on tuning the recalibration distribution. The axis has no correct point; stop looking for one |
| 5 | Is there a fix that works? | **R1, temperature scaling.** One parameter. amber `T = 0.8708` (stable to 0.8%), **71%** calibration-error reduction, isotonic cannot beat it. ⛔ **Fails on copper** | §7 | 🟡 **BANK IT. DO NOT APPLY IT.** It is honest on one of two models one config key apart. Applying a fix that fails its own sibling is not a fix |
| 6 | Does R1 change *where* the model fires? | **NO, by construction.** A monotone map is rank-preserving; measured `ΔAP = 0.00000` exactly | §10 | **Recorded as a limit, not a failure.** R1 moves firing **volume**, never **placement**. Never cite it as a skill improvement |
| 7 | Does R1 fix the firing rate? | **Partly. F11, run 2026-10-19, no GPU.** amber `act_ratio` 1.2970 → **0.9825** (passes); copper 0.7078 → **0.5703** (⛔ **fires**) | §7b | **Recorded.** Consistent with row 5: one model fixed, the sibling made worse |
| 8 | Do the threshold-gate models need fixing? | ⛔ **NO — they are structurally immune**, and their empirically tuned τ **already absorbs** the distortion. Verified exactly: 200,000/200,000 identical firing decisions | §8 | ⛔ **DO NOT TOUCH THEM.** Applying R1 while keeping τ would change their firing set for no reason. Simon's ruling 2026-10-18: not changing them, *"if anything the opposite"* |
| 9 | Would measuring the training window separate the causes? | **NO — killed on POWER, not on the idea.** 2,421 land cells / **314 positives** in the interior; `sd(b)` fails the 0.027 gate at every count and is **non-monotone**, so no sufficient count can be extrapolated | §5 | **Closed. Not worth running.** ⛔ The *measurement* is retired; `e6c_power.py` and the 142× boundary confound are **instruments and stay live** |
| 10 | Does training `window_dim` move the BN moments at all? (E7 screen) | **UNKNOWN — never ran.** Its only justification was being free; it OOM-killed this machine twice and the honest cost was never measured | `results/e7_bn_window_screen.py` | 🟡 **PARKED, instrument banked** with five VOID rules and a pre-committed reading. ⛔ Its answer cannot change any decision in this table — which is why parking it costs nothing |
| 11 | **H2 — is `weight_decay: 0.1` on the class heads the cause?** | ⛔ **OPEN.** Running on pods: views-models **#571** (launch, OPEN), **#572** (measure and read out, OPEN), under epic **#565** | §10 | **THE ONE BLANK.** Fill this row when #572 reports. ⛔ **Nothing else in this table waits on it** — see below |
| 12 | H3 — is there a resolution limit? | ⛔ **NOT ESTABLISHED and not testable as posed.** F7 firing is *consistent with* it, but **AP cannot move under a monotone map: a test that cannot fail cannot confirm** | §10 | **Recorded as NOT PURSUED.** Live and untested. Do not cite F7 as support for it |
| 13 | CRPS on the composed forecast under a corrected gate | ⛔ **NOT ESTABLISHED.** The body sample is absent wherever the gate is small, which is nearly everywhere; no substitute is admissible | §10 | **Needs a GPU re-run and is NOT recoverable from the archives.** Not scheduled. State it as unknown, never as neutral |
| 14 | The root cause | ⛔ **Never scoped by this programme.** Train/deploy geometry, training-side | §10 | 🔵 **After the body.** Not a gate-axis question |
| 15 | Is the shipping gate safe to validate a fix against? | ⛔ **NO.** There is **no occurrence guardrail**, and halving the firing rate buys **37–83% CRPS** — so the artefact this programme found **satisfies the eligibility criteria outright** | §9, RND_PLAN §7 | 🔴 **BLOCKS VALIDATION, NOT MEASUREMENT.** 3 issues + 1 ledger correction, below. Governance is Simon's call, not this seat's |

---

## ⛔ Why the decision does not depend on the pod result

Row 11 is the only open question, and the next action is the same under every outcome it can have:

| if #572 reports | what changes | what we do next |
|---|---|---|
| `weight_decay` **is** the dominant cause | we learn the mechanism | **the body** — the fix is a training-side change, which is body-programme work |
| `weight_decay` is **not** the cause | H2 joins H3 as unestablished | **the body** — no remaining gate hypothesis is both live and testable |
| the arms are **inconclusive** (run-to-run variance) | n=1 at 300 lessons cannot resolve the ~0.019 seed sd | **the body** — and that is a C-307 inconclusive, not a negative |

⛔ **There is no outcome under which the gate axis reopens.** The defect is recorded, the one fix is banked,
`pos_weight` is closed by algebra, the BN axis has no correct point, and the only untested hypotheses are
either untestable as posed (H3) or training-side (the root cause). **That is what closes the axis, and it
is why the conclusion could be written before the result arrived.**

---

## What ships, what is banked, what is forbidden

- **SHIPS: nothing.** No model is changed by this programme. ⛔ No gate-calibration number exists for any
  served model.
- **BANKED, unapplied:** R1 (`T = 0.8708` amber, `0.9155` copper); the E7 screen; `e6c_power.py`; the 142×
  patch-boundary confound measurement; `scripts/gate_reliability.py` (which predates this programme and was
  reinvented because a "superseded" marker hid it).
- **FORBIDDEN:** tuning the BN recalibration distribution (Goodhart ban, row 4); searching for an honest
  `pos_weight` (row 3); applying R1 to a threshold-gate model (row 8); citing `ΔAP` or `precision_at_k` as
  evidence for or against R1 (row 6).

---

## Open items, with owners

| | item | owner | decision |
|---|---|---|---|
| 🔴 | 3 governance issues + 1 ledger correction (RND_PLAN §7) | **Simon** — eligibility is governance | **Open them.** Until then no fix can be called an improvement |
| 🟡 | Phase 2 survey | this seat | ⛔ **Recommend NOT running it.** File the unidentifiable-serving-weights issue instead |
| 🟡 | 13 issues in `OUTCOME.md §9`, 3 in `RND_PLAN §7` | this seat | **Unopened.** Open on Simon's go; several are in production code |
| ⚠️ | `act_ratio` has **no pre-registered numeric band** | this seat | **Recorded as a gap.** Row 7's verdict is directional, not against a committed threshold |
| ⚠️ | Validated at **h=1 only**; the delivery is 36 months | — | **#558 / C-161.** Nothing here extends past h=1 |
| ⚠️ | **n=2**, same architecture, one config key apart — and the fix that works on one fails on the other | — | ⛔ **No generalisation claim.** Not to the roster, not to a served model |
| 🔵 | The post-mortem (`../postmortem_two_seat_design_churn_2026-10.md`) | both seats | **Stays DRAFT until epic #565 reports** |

---

## Cost of this axis, measured

17–19 October. **44 commits, 31 of them pre-registration amendments, against 4 measurements** (E6c, E6f,
Tier 2/R1, F11). **$0 spent on the gate axis itself** — every measurement ran on CPU. The broader
"gate or body" thread opened 12 October, so 8 days.

⛔ **31 amendments to 4 measurements is the number worth keeping.** `RESULTS.md §11` records twelve
withdrawn claims across both seats and **nine cost or power estimates, nine of them optimistic**. The
standing rule it produced — *neither seat states a cost or a power figure again without measuring one
first* — was then broken once more on 19 October by measuring the E7 screen's **time** and not its
**memory**, which OOM-killed Simon's machine twice. ⛔ **The rule now reads: time AND memory.**
