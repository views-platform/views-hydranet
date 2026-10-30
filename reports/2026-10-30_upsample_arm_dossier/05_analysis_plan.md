# 05 — Analysis plan

**2026-10-30.** ⛔ **The pre-registration itself is `PREREG.md` in this directory** (committed alone
and before any architecture code existed, plus amendment 1). `structure.md` permits the
pre-analysis plan to live as its own artifact linked from here; it is **not restated**, because
restating a committed pre-registration is how two documents come to disagree about the same number
(`C-363`).

## What `PREREG.md` contains

| § | content |
|---|---|
| §1 | ⛔⛔ **the prior**: EXP-01's 12 arms, nothing promoted, `ShallowPool` REJECT, **12/12 negative at h30 and h36**, damage ordered by intervention aggressiveness, every candidate matching or beating the control's oracle ceiling. ⇒ The repo's own prior art **predicts this arm loses**, which is why it runs as the **first prospective test** of that post-hoc finding |
| §2 | the three remedies, the **+33.9% disqualification**, and the rule that `G-F`'s numbers choose the arm |
| §3 | C-351 slots — population, truth, estimator, **inference mode (AUTOREGRESSIVE, the opposite of `M92`'s)**, free parameters, and the one varying config key |
| §4 | `G-F`, and why it is a real test rather than a tautology |
| §5a | ⭐ **`S-1` and `S-2` — Simon's predictions, quoted verbatim.** `S-1` is about **my behaviour**; `S-2` (a fail lowers a probability, it does not kill) was conceded and verified against the record |
| §5b | `P-20`…`P-26`, anchored to EXP-01's **measured** gradient rather than intuition |
| §6 | ⛔⛔ **the verdict rule and the five anti-`S-1` devices**, committed before any number existed |
| §7 | the gates |
| §8 | limits — n=2, **no σ̂, no significance claimed**, `sb` only |
| §10 | amendment 1: the control is **re-run**, three sibling tools, the `_ARCH_TAG` addition, queue order, and **`P-27`** |

## Endpoints, restated because they are the one thing most likely to be got wrong

⛔ **`h1` and `h36`. Never `h18`.** EXP-01's own stated design error was choosing it: *"h18 was the
worst available primary endpoint… it sits almost exactly at the sign change and reports 'nothing is
happening' for a pattern that is unmistakable at h1 and h36."*

## Outcomes so far

| prediction | status |
|---|---|
| `P-20` at least one remedy removes the signature | ✅ **HELD, 2 of 2** — `R-57` |
| `P-21` `UpsampleConv1`'s gap below `UpsampleConv2`'s | ⚠️ **weak hold on a 0.0094 margin; not claimed as a win** — `R-58` |
| `P-22`…`P-27` | 🔵 OPEN — `O-3` |
| `S-1`, `S-2` | 🔵 `S-1` open; `S-2` conceded |

## ⚠️ OWED BEFORE THE QUEUE: the refetch declaration

The cached datafactory pull is from **2026-09-19** and its queryset digest no longer matches, so
`C-61` refuses it. A refetch is required, and it **changes the data vintage relative to EXP-01's
August arms**.

⇒ ⛔ **This must be declared here, as an amendment, BEFORE the queue runs**, with:
- the pull date and row count recorded;
- the consequence that **`P-27` then has two possible causes if it fails** — code drift *or* data
  vintage — which this design **cannot separate**;
- ⇒ therefore every cross-reference to EXP-01's **absolute** numbers is **indicative**, while
  arm-vs-control stays valid because both run on the same pull, the same commit and the same box.

⛔ **It is a logged decision, never a flag on a gate.** An earlier version of `smoke_exp02.sh`
carried a `SMOKE_REFETCH=1` escape; it is gone (`07`'s postmortem).
