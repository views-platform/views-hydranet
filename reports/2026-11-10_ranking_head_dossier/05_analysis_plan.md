# 05 — Analysis plan · ⛔ **STUB — DELIBERATELY NOT WRITTEN**

**Serves:** the decision half of Simon's question — *how we would know* whether the ranking body
ranks better. ⛔ **It cannot be written yet, and writing it anyway would be the specific error this
programme has already committed four times.**

Opened 2026-11-10 as a stub. ⚠️ This file is a **STUB** on purpose, not an oversight and not an
unfinished draft.

---

## Why this is a STUB

A pre-analysis plan must declare its **primary superiority quantity** and its **margin** before the
numbers exist. `C-3531` (`Gneiting2011_MakingEvaluatingPointForecasts`): post-hoc scoring-rule
selection is inadmissible. The handover records what happens when that is ignored:

> *"⛔ **The scoring rule is declared EX ANTE.** … ⚠️ I violated this four times in one day by
> picking the metric after the fact."*

and

> *"⛔ Four successive definitions of done. All were success criteria, or compared the wrong two
> things, or used CRPS, or were satisfiable by a no-op."*

⇒ Two things are missing, **neither of them a decision this session may take.** They are listed
below with an owner each. When both land, this file is replaced by a real pre-registration in the
template of `reports/2026-11-04_delivered_gate_lookup/PREREG.md` — the best one in the repo.

---

## Blocker 1 · the primary quantity  ·  **owner: Simon**

The handover is explicit that this is his and not mine:

> *"`FAO-02` already has this shape (primary at 5%, guardrails at 1%); only the primary **quantity**
> needs substituting, **which is a request to Simon, not an interpretation.**"*

### ⛔⛔ And my own candidate is contaminated — stated against myself

I proposed **`MCB`** — each arm's distance from its own isotonic recalibration. It satisfies three
of the handover's five constraints cleanly: it is conditional by construction (`§2.2`), it compares
each arm to *its own* best-calibrated self rather than to the other arm (so it survives the
`Brocker2009` blocker, `§4`), and it is not a rank metric, so it does not score B on its own
training objective (`§2.1`).

**But it was measured before it was proposed.** `../2026-11-10_rank_coupling_verdict/results/s11_tau_sweep.out`
already prints `MCB_A = 0.138` against `MCB_B = 2.952` at τ = 0.20, across all nine rows. I knew the
sign and the scale of that comparison when I named the quantity. ⇒ **Declaring it primary now is
exactly what `C-3531` forbids**, and it would be the fifth instance in one programme.

⚠️ **The one honest escape, and it must be stated in the prereg rather than assumed:** what S11
measured was `MCB` for a rank-coupled emission over a **frozen cube** — A's own point forecast
reshuffled. `MCB` for a **trained** ranking head is a different arm and its value is genuinely
unknown. That distinction may rescue `MCB` as a primary quantity, but it is a claim that has to be
written down and defended **before** the number exists, not after.

⇒ **What is needed from Simon:** a primary quantity, or approval of `MCB` *with the frozen-cube
contamination recorded in the prereg as a stated limit.*

## Blocker 2 · the margin  ·  ⛔ **owner REASSIGNED 2026-11-10 — it cannot be the `baseline` session**

Handover `§2` constraint 4:

> *"⛔ **The margin is in units of MEASURED seed dispersion, never percent.** `C-392`: seed noise on
> `MCR_field` is **84×** FAO-02's 1% margin, and the magnitude columns carry CV **0.237–0.356**,
> 10× worse than the placement columns. **Measure the dispersion, then choose `k` in `δ > k·SD`.**"*

**What exists:** `σ̂_paired = 0.4441` at `ddof=1`, declared — the dispersion of the paired gap across
six control-form arms (`../2026-11-10_rank_coupling_verdict/results/s7_sigma_hat.out:66`).
⛔ **It is in `fair-CRPS` units.** It is the dispersion of a *different quantity* and cannot serve as
a margin for whatever blocker 1 names.

**What is needed:** the seed-to-seed dispersion of the quantity blocker 1 names, measured at
`ddof=1` **declared**. ⛔ **Not a go/no-go** — it sets the margin and the replicate count, nothing
else.

### ⛔⛔ Why the `baseline` session cannot supply it — answered 2026-11-10, and the answer is structural

The `baseline` session was asked (handover `§5` item 1, commissioned 2026-11-09) and **replied with
an inventory and a refusal, both correct.** Recorded here in full because it changes who owns this
blocker:

**What `baseline` holds, all of it:** average CRPS per target — plain CRPS, *not* fair-CRPS — over
all PGM cells on the validation partition at `S = 64`, for 4 models (conflictology + 3 parametric
dwarfs), seeds `{42, 123, 456}` so **`n = 3`**, estimator `np.std` i.e. ⛔ **`ddof = 0`**. On `lr_sb`,
SD ≈ 0.002–0.003, range ≈ 0.006; tiny on `lr_ns`/`lr_os`. Plus a same-model distribution-distance
null (Wasserstein / energy / C2ST) on a 5,197-entity subsample, which is not a scalar seed SD.

⭐⭐ **And the argument that settles it, which this dossier had not made:** *a baseline model is
deterministic given the history.* Changing its seed changes only the **Monte-Carlo draws**, not the
model. So a baseline's seed SD measures **sampling noise alone**. A trained ranking head's seed SD
also contains **training noise** — initialisation and data order. ⇒ **A baseline-supplied margin
would systematically UNDERSTATE the dispersion of the thing being judged**, which makes a difference
easier to call significant than it is. That is the wrong direction for a margin to err in.

⇒ **The margin must come from retrained arms** — the dispersion of the quantity across seeds of a
*trained* head. ⚠️ Which collides with `00 §6`'s separation rule: the session that supplies the
yardstick must not also produce the number judged against it. **That conflict is unresolved and is
Simon's to resolve**, and it is now the whole content of this blocker. Two shapes exist, neither
chosen here: measure the dispersion on the **control** arm's existing seeds (available, same
training machinery, no new arm) — or commission a third session that owns neither.

⚠️ **One thing `baseline` flagged that was my error, not theirs:** my request implied `σ̂ = 0.4441`
was a `baseline` number and they spent a search failing to find it in `views-baseline`. It is
**ours** — `../2026-11-10_rank_coupling_verdict/results/s7_sigma_hat.out:66`, epic #579 S7, measured
in-run across six control arms. Corrected to them directly.

⛔⛔ **And a trap this uncovered, recorded before it bites: `0.4441` names TWO different quantities in
this programme.** `σ̂_paired = 0.4441` is in **fair-CRPS** units (S7); `λ* = +0.4441` is in
**`ρ_event`** units (S8). Same four digits, unrelated quantities, both load-bearing, both in
committed documents. ⇒ Never write `0.4441` in this programme without its unit attached. This is
`C-351` in its purest form — a number is its population, its transform and its estimator, and here
two numbers collided on the digits alone.

---

## What this stub already commits to, so it cannot be negotiated later

These are not placeholders. They are pre-committed now, while no number exists, precisely so that
the eventual plan cannot be shaped around a result:

1. ⛔ **The superiority quantity must not be in B's training objective.** B trains on rank ⇒ **every
   rank metric is report-only**, including `ρ_event` and the finish line of `00 §⭐⭐`. The finish
   line is the programme's *destination*; it is **not** the promotion criterion. These are different
   things and conflating them would score B on its own objective.
2. ⛔ **Every magnitude criterion and guardrail is CONDITIONAL — per-cell or stratified by the index
   — never marginal.** B draws from the empirical marginal, so it is marginally calibrated **by
   construction**; a pooled magnitude check is a no-op wearing a guardrail's clothes.
3. ⛔ **A declared no-op control** that must read zero on the primary criterion. If it does not, the
   criterion is broken, not the arm.
4. ⛔ **The decision rule's shape:** `PROMOTE ⟺ (≥1 strict superiority criterion cleared by the
   margin) ∧ (all non-inferiority guardrails held)`. ⭐ A B that changes nothing scores zero on the
   superiority criterion and passes every guardrail, so it **fails automatically**. That property is
   the point.
5. ⛔ **Three outcomes, not two** — including **not resolvable at this `n`** as a *result*.
6. ⛔ **The sweep is reported whole.** Quoting the best cell of a grid without the grid beside it is
   not a result (`C-393`).
7. ⛔ **If the result is negative and I then wish to argue the test was inadequate, that argument is
   logged as a `W-` row against me in `reports/PREDICTION_LEDGER.md` first** — before it is made.
   This is the anti-fishing device from `2026-11-04_delivered_gate_lookup/PREREG.md`, and it applies
   here by adoption.

---

## ⛔ Not in scope for this file, ever

Pre-registering **after** phase 4 has run. If phases 1–4 of `04_roadmap` complete while blockers 1
and 2 are still open, the correct outcome is **a trained arm with no verdict** — reported as
*"the criterion was never declared"*, which is a process finding against me, not a result about B.

⚠️ `04` phases 1–4 deliberately do not wait on this file. Phase 5's verdict does.
