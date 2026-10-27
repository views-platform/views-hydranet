# The truth is not ground truth — and the sharpness we score against is partly an artifact

**Date** 2026-10-27 · **Raised by Simon**, in response to M1's observation that the delivered model
*"fires 3.5× too often while carrying one third the mass"* · **Register entry RP-10** ·
⛔ **Nothing here is measured yet.** This document states the argument, its consequences, and the
one place it cuts the other way, so that none of it is lost before it is tested.

## The argument, as Simon put it

> *"The input data is not ground truth. It's month and cell aggregates of points which have known
> biases and unobserved measurement error. … The sharpness of these points does not represent
> certainty in when and where conflict was. It is to some extent an artifact of the measurement,
> reporting, and aggregation choices made. … Even if the time and space of historical conflict
> events perfectly predicted future conflict events, we would have aleatoric uncertainty which would
> constitute a minimum 'Bayes error' due to measurement and reporting error/bias. That might mean
> that the best we can do is less sharp patterns with a solid ranking and a correct magnitude when
> summed over some area."*

## Why it holds

`ged_sb_best` at `(priogrid_id, month_id)` is a **sum of geocoded point events**. Each contributing
event carries, in GED itself, a coded location precision and a coded date precision — many events are
geocoded to an administrative centroid or a named town rather than to the place the deaths occurred,
and many carry month-or-coarser dates.

⇒ Aggregating those points into a 0.5° × 1-month bin produces a **spiky field by construction**.
Some of that spikiness is the conflict. Some of it is **our ignorance of where and when**, recorded
as though it were knowledge.

⇒ ⛔ **CRPS against this field is a proper scoring rule for predicting the OBSERVATION, not the
process.** A model that correctly predicted the latent conflict intensity would score **worse** on
our ruler than one that had additionally learned to mimic the measurement noise. And there is an
**irreducible CRPS** — a Bayes error floor — that no model can beat.

⇒ ⛔⛔ **We have never estimated that floor.** Which means **"the model scores 0.039" is
uninterpretable**: we cannot say whether it is near-optimal or terrible.

## ⚠️ A correction I owe, made before anything is built on it

In conversation I said *"GED carries `where_prec` and `date_prec` — the data states its own location
and date uncertainty,"* and proposed perturbing the truth within its own declared precision.

⛔ **That is true of GED as a source and false of the data we hold.** Measured:
`reports/2026-10-12_underfire_diagnosis/pull_land_all3_121_504.parquet` has exactly three columns —
`['ged_sb_best', 'ged_ns_best', 'ged_os_best']` — and **no precision fields of any kind**. The
precision codes are **event-level**; we hold cell-month aggregates, where they have already been
summed away.

⇒ **Consequence for the design.** The floor cannot be estimated from what is on disk. Two routes,
and they are not equivalent:

| route | what it would give | cost |
|---|---|---|
| **A. Declared, swept kernel** — blur the truth over a stated radius, total preserved, sweep the radius | a floor **conditional on an assumed** precision. ⚠️ Not the data's own precision, so it bounds the *shape* of the answer, not its level. | **CPU, minutes.** Available now. |
| **B. Event-level GED** — recover `where_prec` / `date_prec` and perturb each event within its own coded precision, then re-aggregate | the **actual** floor the data declares for itself | needs an event-level pull that does not currently exist here |

⇒ ⭐ **Start with A because it is free and it tests the shape of the claim. Do not report A's number
as "the floor"** — report it as *"the floor under an assumed blur of radius r"*, with r swept (D-4).

## ⭐⭐⭐ Where it reframes our most-quoted spatial result

D12 measured: the body carries **0.391** of its due spatial structure at **1 cell** and **0.870** at
**32 cells**.

We have read that one way only — *"the model fails at fine scale."* **There is a second reading:
the fine scale is partly unlearnable, because at 1 cell we are scoring measurement noise.**

⇒ ⛔ **The two readings have never been distinguished, and they imply opposite actions** — the first
says fix the model, the second says change the target.

⇒ **The measurement floor PER SPATIAL SCALE settles it.** If the floor at 1 cell sits near our
measured performance while the floor at 32 cells sits far below it, the fine-scale deficit is the
**task** and the coarse-scale deficit is the **model**.

⇒ ⭐ This upgrades **RP-7** from *"nobody scores aggregates, and that is the audit-facing number"* to
*"aggregates may be the only well-posed target."* RP-7 moves up the queue, behind RP-10.

## ⚠️ The bound on it, from our own record

`project_firing_is_not_the_lever`, measured: `thin_0.75` keeps **97%** of the oracle after deleting
25% of events, while **scrambling placement costs 81%**.

⇒ **If the truth's locations were mostly artifact, scrambling them could not cost 81%.** The
placement carries real signal, and this argument cannot be used to excuse placement error in general.

⭐ But the two are compatible **at different scales**: real signal at 32 cells, substantial noise at
1 cell — which is precisely D12's profile. ⇒ The argument is about the **sharpness**, not about
whether location matters.

## ⛔ And the part that cuts against the argument — which is the useful half

The argument defends **spreading** the mass. It does **not** defend **losing** it.

- Location error **moves** a death between cells. **Total preserved.**
- Date imprecision **moves** it between months. **Total preserved** within the window.
- Under-reporting means the observed total is an **undercount** ⇒ the latent total is **≥ 50,881**.

Our delivered model emits **16,612** against an observed **50,881**.

⇒ ⛔⛔ **Measurement error excuses the smoothing and makes the mass deficit WORSE, not better.** If
the latent total exceeds the observed one, a model at 0.33× of the observed is further from the
latent than it looks.

⇒ ⭐ **So the diagnosis splits in two, and the split is the contribution:**

| question | status under this argument |
|---|---|
| *Is the field too smooth?* | ⚠️ **possibly defensible** — and possibly even correct |
| *Is the total one third of truth?* | ⛔ **not defensible by any measurement-error story** |

⚠️ Those two have been carried as one defect ("under-firing"). They are not one defect.

## ⛔ What this argument does NOT touch

None of RP-1's instrument findings depend on what the truth means:

- `size_ratio` saturating at 0.0000 for four of seven forecasts is **arithmetic** — a median over
  event cells is 0 whenever more than half of them get a zero prediction.
- The plug-in CRPS bias is **arithmetic** — `E|X−X′|/(2m)`.
- AP ranking on the gate is a **code path** — `average_precision(ybin, p)`.
- **C-387** is a **derivation** about a parameterisation.

⇒ These are properties of the **instruments**. The measurement floor is a property of the
**target**. Both are real; neither excuses the other.

## What gets measured, and in what order

1. ⭐ **An eighth red-team forecast, now, inside M1 (#478):** the truth itself, blurred over a
   declared radius, **total exactly preserved**, swept over r ∈ {1, 2, 4}. Deterministic, CPU,
   minutes.
   ⇒ ⛔ **If blurred truth scores WORSE than raw truth on our columns, the battery actively
   penalises the forecast this argument says may be right** — which would be the strongest blindness
   result in epic #477, and it costs a few lines.
2. **RP-10**, as a register entry: estimate the floor properly, per spatial scale, via route A and
   then route B if route A says the question is live.
3. **RP-7 moves up**, behind RP-10, on the reasoning above.

## Status

🔵 **OPEN. Argument recorded, nothing measured.** ⛔ No number in this document is a result; the only
measurements in it are the parquet's three column names and figures quoted from elsewhere with their
sources.
