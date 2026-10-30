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
| §11 | amendment 2: two confounds an independent review found — one fixed, one declared |
| §12 | amendment 3: the `k=2` remedy padded with **zeros**, measured as an edge defect (interior 2.0000, last row 1.0000, corner 0.5000); `padding_mode="replicate"`, and **`P-28`** / **`P-29`** |
| §13 | ⭐ amendment 4: **the refetch, declared before the pull** — the digest mismatch measured, the vintage consequence conceded, and **`P-30`** |

## Endpoints, restated because they are the one thing most likely to be got wrong

⛔ **`h1` and `h36`. Never `h18`.** EXP-01's own stated design error was choosing it: *"h18 was the
worst available primary endpoint… it sits almost exactly at the sign change and reports 'nothing is
happening' for a pattern that is unmistakable at h1 and h36."*

## Outcomes so far

| prediction | status |
|---|---|
| `P-20` at least one remedy removes the signature | ✅ **HELD, 2 of 2** — `R-57` |
| `P-21` `UpsampleConv1`'s gap below `UpsampleConv2`'s | ⚠️ **weak hold on a 0.0094 margin; not claimed as a win** — `R-58` |
| `P-28` the padding fix moves `UpsampleConv2`'s gap less than the 0.0094 `P-21` margin | ✅ **HELD, Δ = −0.000107**, declared sign correct; point estimate 10× pessimistic — `R-59` |
| `P-29` `UpsampleConv1`'s gap unchanged to within 1e-6 | ✅ **HELD exactly, Δ = 0.000000** — `R-60` |
| `P-22`…`P-27` | 🔵 OPEN — `O-3` |
| `P-30` the refetched pull's shape is unchanged (`pgm`, 121, 504, 5,034,240 rows) | 🔵 OPEN — settles in `07` with the first arm's entry |
| `S-1`, `S-2` | 🔵 `S-1` open; `S-2` conceded |

## ✅ DECLARED BEFORE THE QUEUE: the refetch — `PREREG.md` §13 (amendment 4)

⇒ **This is no longer owed.** It was written on 2026-10-30, before any pull, and it is measured
rather than asserted:

| | |
|---|---|
| the cache | `violet_visitor/data/raw/calibration_datafactory_df.parquet`, **2026-09-19 14:56:20**, **5,034,240 rows × 8 cols** = 13,110 cells × 384 months, `pgm`, months **121–504** |
| why `C-61` refuses it | cached digest `b595ff4d…` vs the digest recomputed today from `config_queryset.py` `b32a221b…` — ⇒ **mismatch**, measured, not repeated from an earlier note |
| the decision | **one deliberate, logged refetch.** `-sa` stays unconditional; the reverted `SMOKE_REFETCH=1` is not coming back |
| ⛔ the conceded cost | **`P-27` gains a second possible cause** — code drift *or* data vintage — and this design **cannot separate them**. Every cross-reference to EXP-01's **absolute** numbers drops to *indicative* |
| ✅ what survives intact | **arm-vs-control**, because all four runs take the same pull, the same commit and the same box. `P-23` is untouched |
| the new prediction | **`P-30`** — the refetch changes *values*, not *shape* |

⚠️ **What the digest mismatch does not establish:** whether the queryset changed, one of its inputs
changed, or `views-pipeline-core`'s digest function changed between the pull and the installed
**3.3.4**. All three produce the same refusal and the same decision; §13a says so rather than
picking one.

⛔ **The recording requirement is the part that can still fail.** The new pull's fetch timestamp,
row count, `month_first`/`month_last` and `queryset_digest` must land in `07_experiment_log.md` **in
the same entry as the first arm**. A queue that runs without those five numbers has undeclared data
provenance — the `C-347` family — and the declaration above would then be a promise rather than a
record.
