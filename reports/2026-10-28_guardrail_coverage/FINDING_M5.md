# M5 (#482) — of FAO-02's three locked guardrails, the frozen ruler computes exactly one as defined. QS99 is a coverage gap; MCR is a definitional divergence, and the glossary was part of it.

**Date** 2026-10-28 · **Issue** #482 (epic #477, RP-1) · **Method** reading, not measurement — the
frozen ruler, three dossier scorers, the training engine, and **the governing document itself**
(`fao_02_pre_release_note_05/main.pdf`, Appendix A.1, extracted with `pdftotext -layout`) ·
**No compute, no cubes touched, nothing in `lodestar_score.py` or FAO-02 changed**

⛔ **Verdict, stated plainly as #482 requires:**
- **QS99 — CLOSED as a coverage gap.** It is absent from the frozen ruler and computed under
  **one** definition in **three** places, all of which match FAO-02's Appendix A.1 formula exactly.
- ⛔⛔ **MCR — OPEN, and it is the bigger finding.** FAO-02 locks a **ratio of means over all
  cell-months**; the repo holds **three** mutually non-equal statistics called MCR, the frozen
  ruler computes **none** of them under the locked definition, and `reports/GLOSSARY.md` listed
  `MCR` as an alias of `pos-mcr`.
- ⚠️ **Decisions — FLAGGED, not open.** No FAO-02 eligibility decision has been taken at all. But
  FAO-02's **5% margin has been borrowed** for an intra-programme bound without the conjunction it
  belongs to.

---

## 1. QS99 is absent from the frozen ruler — confirmed by reading, not by grep

⚠️ #482 says so explicitly, and the reason is `W-27`: *a grep that finds nothing and a grep that
never ran look identical.* All 277 lines of
`reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` were read. Its output row is a
dict with fourteen keys:

`target · model · N · n_event · n_zero · AP · Brier · crps_all · crps_events · crps_none ·
size_ratio · pos_mcr · gate_source · thresh`

⇒ **No QS99.** A locked guardrail is not computed by the instrument every current result is scored
on.

## 2. ⭐ But it is not an absence: one definition, three implementations, and they match the locked one

**FAO-02, Appendix A.1** (read, not recalled):

> `QS_0.99(q, y) = (1{y < q} − 0.99) · (q − y)`

Expanded, with `d = y − q`: `max(0.99·d, 0.01·(−d))` = the pinball loss at τ = 0.99 against the
forecast's **predicted 99th percentile**, at **every** cell–month ("All metrics are computed at the
level of individual PRIO-GRID cell–month observations").

| site | code | matches A.1 |
|---|---|---|
| `t0_score.py:29-31,59-60` | `pinball(truth, q99, 0.99)` with `d = truth − q`, `max(τd, (τ−1)d)`; `q99 = np.quantile(mag, 0.99, axis=1)` | **yes** |
| `bulk_score.py:111-113` | `d = truth − q99`; `np.maximum(0.99*d, -0.01*d).mean()` | **yes** |
| red-team `score()` (M2/M3, #479/#480) | `d = TRUTH − q99`; `np.maximum(TAU*d, (TAU-1)*d).mean()` | **yes** |

⇒ ⭐ **#482 predicted *"a coverage gap, probably not an absence"* — correct.** Three independent
implementations, one formula, agreeing with the locked document.

⛔ **One naming collision, and it is not a definitional conflict.** `training_engine.py:958-965`'s
`qs99_weight` / `qs99_tau` is a **training regulariser**: the same pinball formula applied to
**`mu`** — the body's mean — rather than to a predicted 99th percentile, and on
**positive-masked cells only** (`mask.any()`, `target_j[mask] - pred_j[mask]`). Its own comment is
accurate (*"distribution-free pinball on mu"*). ⇒ The **name** invites the confusion; the code does
not make the error. `scripts/tail_scorecard.py`'s `pin90` is τ = 0.9 and says so, and says *"NEVER
select on it"*.

## 3. ⛔⛔ The finding #482 did not anticipate: MCR, a second locked guardrail, diverges by definition

**FAO-02, Appendix A.1:**

> `MCR = ȳ̂ / ȳ`, where `ȳ̂` is the mean predicted fatalities and `ȳ` the mean observed fatalities
> **"computed across all PRIO-GRID cell–month observations in the validation evaluation period"**.
> Judged as **distance from 1**, non-inferiority margin **1%**.

**A ratio of means, over every cell.** The repo holds three different statistics under that name:

| name | formula | population | equals FAO-02's MCR? |
|---|---|---|---|
| `lodestar_score.pos_mcr` | **mean of** (`ey`/truth) | truth > 0 | ⛔ **no** — mean of ratios, conditioned |
| `lodestar_score.size_ratio` | **median of** (`ey`/truth) | truth > 0 | ⛔ no |
| `sharpness_scorecard` "MCR" | mean(pred) / mean(truth) | truth > 0 | ⛔ no — right order of operations, wrong population |
| red-team `MCR_field` (M2) | mean(`ey`) / mean(truth) | **all cells** | ⭐ **yes** |

⇒ ⛔⛔ **Of FAO-02's three locked guardrails, the frozen ruler computes exactly one — Brier — as
defined.** QS99 is absent; MCR is present only as two statistics that are not it.

⭐ **M3 already measured how far apart the two forms are.** `pos_mcr`'s per-cell weight is exactly
`1/truth`, confirmed to **7.6e-17** against a closed form, so the **same 100 units of mass move it
115.3× more when added to the smallest truths**. ⇒ `pos_mcr` is a statement about one-event cells;
FAO-02's MCR is a statement about the total. **Neither bounds the other**, and the delivered model
is known to be mass-deficient (total 16,612 against 50,881) precisely where the aggregate form
would see it.

⚠️ **And `bulk_score.py` bans MCR by name** — *"the pooled-mean/MCR cancellation cannot occur; MCR
is BANNED"* — which is **right for its own question** (a per-cell bulk-calibration read, where a
pooled mean lets over- and under-prediction cancel). ⇒ Both positions are correct about different
questions. The error available here is not choosing wrongly; it is letting one stand in for the
other.

## 4. ⛔ The glossary was part of the problem

`reports/GLOSSARY.md` §4 listed `MCR` among the **banned aliases** of `pos-mcr`, and gave MCR no
locked row of its own. ⚠️ Banning the alias is correct — *don't say MCR when you mean pos-mcr* —
but with no MCR row anywhere, the practical reading was *"MCR is pos-mcr, just spelled wrong"*, and
FAO-02's guardrail had no entry in the fixed vocabulary at all.

**Fixed in this pass** (the glossary is to be edited, never worked around):
- a locked **`MCR`** row with FAO-02's formula, population, margin and the three non-equal repo
  statistics named;
- a locked **`QS99`** row with the A.1 formula, the three agreeing implementations, and the
  training-regulariser collision called out;
- a continuation line on **`pos-mcr`** carrying M3's 115.3× measurement and the sentence that
  matters: ⛔ **a `pos_mcr` number does not discharge the MCR guardrail.**

## 5. Was any locked decision taken without its guardrails?

⛔ **No — and the honest answer is more useful than a yes.** FAO-02 governs the **validation
partition**, over a **36-month rolling-origin backtest**, against the **empirical conflictology
baseline**, with eligibility a **strict conjunction**: superiority (CRPS ≥ 5% better) **AND** all
three guardrails (≥ 1% better each). Every decision in this record was taken on the **T=0** frozen
ruler, which FAO-02 does not govern — a step-1 result is development, not governance.

⇒ So this is **not** a retrospective invalidation. It is a **readiness gap**: the instrument the
programme develops on computes one of the three guardrails the governance decision will need.

⚠️ **One flag, and it is C-306's shape.**
`reports/2026-10-21_structure_penalty_arm_prereg.md` P-5 and F3 use **FAO-02's 5% margin** as a
non-inferiority bound on `crps_all` for an **intra-programme** comparison (arm vs control). That
margin was derived in the locked document for a **different contrast** — candidate vs the empirical
baseline — and it travels there with three conjunctive guardrails that this use does not carry.
⇒ The margin is a borrowed yardstick. Declared, not hidden, but it should say what it is.

## ⇒ Consequences

1. ⛔ **Nothing is added to the frozen ruler.** It is frozen; #482 forbids it; and the fix is not
   needed for development work. ⇒ **Recommendation only:** before any FAO-02 eligibility claim, a
   separate guardrail pass must compute **QS99** and **FAO-02's MCR** on the validation partition,
   over the 36-month protocol, against the conflictology baseline. ⚠️ That is a job, not a column.
2. ⛔ **Register candidate:** MCR names three non-equal statistics, one of which is a locked FAO
   guardrail, and the fixed glossary aliased the locked name to one of the others. **C-351's shape
   at the level of a governance term.**
3. ⚠️ **Never write "the guardrails pass" from a `lodestar_score` row.** It carries Brier and
   nothing else from the locked set.
