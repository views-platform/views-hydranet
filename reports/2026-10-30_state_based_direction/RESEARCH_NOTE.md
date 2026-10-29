# Research note — the latent-state direction, and why Epic #230's null may not be clean

**2026-10-30.** ⛔ **A research note, NOT a work item.** It files no issue and adds no
`RESEARCH_PROGRAMME.md` queue entry, because a re-rank is Simon's call and `D-3` forbids a new
emitting family now. It exists because the argument arrived in chat and `D-6` says log as insights
arise, not at the end.

---

## 1 · Simon's framing, in his words

> *"this could be like basically a state-based situation… there is of course a difference between
> being in peace, low-level conflict and high-level conflict, and maybe the state of being in a
> conflict that can even experience extreme events is a state of itself. So there might be two
> states that look the same 99% of the time because they are like active conflict, for instance.
> But one might latently be in a state that can experience extreme events. And it might also be
> that there are places that are in states which is, you know, peace, but also no potential
> conflict — this is a place in the middle of nowhere, there's no people, there's no cities."*

⇒ Two directions named: **state-based models** (HMM / Markov-switching) and **mixture of experts**.

⛔ **And a correction of mine that belongs at the top.** In `M90` §4 I called the three dispersion
strata *"three different regimes"*. **They are bins of the model's own predicted spread** — a
diagnostic stratification of a predicted statistic. They are **not** latent states of the world.
"Regime" imports a Markov-switching connotation the measurement does not carry: **drift on my part**
against a FIXED glossary (`feedback_use_fixed_glossary`). Simon's reading was the natural one.

---

## 2 · ⭐⭐ Simon's own objection, and it is the strongest point in this note

> *"I don't know if this is retracted but as I understood it we agree that the untruncated NB right
> now is not fitted correctly. so likely a 2-NB mixture would also not have been fitted correctly."*

**Verified against the record, and it holds.**

- `RESEARCH_PROGRAMME.md:455` (**RP-7**, rank 6, *"the most interesting unexplained fact in the
  record"*): the body **sits below its own NLL optimum** — M6's `θ̂` **0.154** against a
  conditionally-honest **0.432**.
- **ADR-073 §0** gives the derived mechanism: the NB NLL of a zero is `θ·log(1+μ/θ)`, **monotone
  increasing in `θ`**, so on an empty cell the optimum is `θ → 0` at **every** `μ`, with **no
  interior optimum**. **99.66% of cells are empty.**

⇒ ⭐⭐ **The body's shape parameter is set by the zeros, not by the events.** A 2-NB mixture trained
under the same objective inherits exactly that: **both** components' `θ` are pulled by the same
99.66% zero mass. ⇒ **Epic #230's mixture null is confounded by the thing RP-7 names, and is
therefore not a clean test of whether a mixture can reach the tail.**

⚠️ This does **not** reinstate the mixture as a candidate — it **demotes the strength of the null**
from *"a mixture cannot crack the wall"* to *"a mixture trained under an objective whose shape
parameter is captured by the zeros did not crack the wall."* Those are different claims, and only
the second is supported.

⚠️ **Provenance, flagged because the claims audit flagged this class.** Epic #230's numbers
(*"component 2 alive, median w|act 0.71–0.92, μ₂:μ₁ 14×–690×, magnitude still capped ⇒ the gap is
tail SHAPE (ξ>0)"*) exist in this repo **only as prose** at `RESEARCH_PROGRAMME.md:501`. ⛔ **No
committed `.out` in this repository contains them** (`grep -rln "mixture" --include="*.out"
reports/` returns nothing). They are **relayed, not measured here.**

---

## 3 · What supports the direction

| evidence | where | what it says |
|---|---|---|
| ⭐ the **shape** mismatch | `M90`(2), `spread_skill.out` §2 | OVER-dispersed in 9 of 10 bins (1.15–2.14×), UNDER-dispersed only in the top bin (0.48 at B=10, 0.39 at B=50). ⇒ **Non-monotone. No single multiplicative rescaling of a two-parameter family fixes both ends.** |
| ⭐ a heavy tail, measured | `project_volatility_ceiling_predictable` | jump **risk/spread IS predictable** (S2 0.79 vs 0.39; quantiles 48% sharper and calibrated; **tail ξ ≈ 0.8**) |
| ⚠️ spread information beyond the mean | `M91`, `spread_partial.out` §4 | **+0.18** conditional on the mean, events only — small, stable, correctly signed |

⇒ ⭐ A fixed two-parameter NB has a **fixed shape family**; a state-dependent emission does not. The
non-monotone correction the data wants is a structural argument for heterogeneity. ⚠️ `ξ ≈ 0.8` is
also, in extreme-value language, exactly Simon's *"state that can experience extreme events"*.

---

## 4 · ⭐⭐ The distinction that decides whether this is new

- A **mixture weight** `w(x_t)` is computed fresh from current features. **No memory.**
- A **latent state** `s_t` with transitions `P(s_t | s_{t-1})` carries forward. **Memory.**

*"Latently in a state that can experience extreme events"* is a **persistent** property — two cells
look identical today and differ in what they can do tomorrow. ⛔ **An emission mixture cannot
represent that; an HMM can.** Epic #230 tested the **memoryless** version. ⇒ **The persistent
version has never been built here, and is the content of Simon's proposal.**

⚠️ **A mixture of experts over NB heads is the memoryless version** and inherits NB's tail
regardless of how far apart the component means are — #230 measured 14×–690× separation with
magnitude still capped. ⇒ If the experts emit NB, the gap stays `ξ`. **A state-based model only
helps if its high-risk state has a different TAIL FAMILY, not just a different mean.**

---

## 5 · ⚠️ What gates it

1. ⚠️ **Simon's "peace with no potential" state already exists in the admitted set** — that is
   ZINB's structural `π`. **ZINB was FALSIFIED in the V2 rollout** (blooms, loses). ⇒ Evidence
   against **one parameterisation** — per-cell `π` from current features, no persistence — not
   against the idea.
2. ⛔ **RP-10 gates any state model.** A latent high-risk state and a geocoding/reporting artifact
   produce the **same residual signature at one cell**. Without the Bayes floor, a state model will
   fit the reporting process.
3. ⛔ **`D-3`: no new emitting family.** A heavy-tailed state needs one.
4. ⛔ **RP-7 should come first**, by §2's argument: fitting a mixture under an objective whose shape
   parameter is captured by the zeros repeats #230's confound at higher cost.

---

## 6 · 🔵 The one measurable this opens, stated so it can be scored

Simon connected `M91` to **ARCH** — conditional heteroskedasticity.

⚠️ `M91` establishes that **~+0.18 of spread information exists beyond the level.** It does **not**
establish that the information is **autoregressive** — i.e. predictable from a cell's **own past
shocks** rather than from its current covariates.

⇒ 🔵 **UNTESTED and cheap:** does `|residual_{t-1}|` at a cell predict `|residual_t|` after the
model's own predicted mean and spread are held fixed? If yes, there is volatility persistence the
emission does not carry, and that is the **measurable** form of Simon's latent-state hypothesis.
⛔ Not queued — RP-7 and RP-10 sit in front of it.

---

## 7 · Status

⛔ **A DIRECTION, not a next step.** Open, not parked, not closed. Nothing here is a result.
