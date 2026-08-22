# Analysis plan — state-freeze at L=300

> ## ⚠️ PROVENANCE: this document is PART pre-registration and PART retrospective. Read the stamps.
>
> | section | status |
> |---|---|
> | §1–§4 (EXP-01/02: the 8-arm run and the paired CI) | **RETROSPECTIVE — written 2026-08-22 after the results.** Not a pre-registration and must not be cited as one. |
> | §5 (EXP-03: the decay dial) | **PRE-REGISTERED.** Committed in `DIAL_PAUSED.md` at `da3156d`, 2026-08-22 10:58:07 +0200; the first dial arm scored 19:29:15 — an 8.5-hour gap verifiable in git history. |
> | §6 falsifiers | **PRE-COMMITTED for all experiments**, because they were enforced *in code* (`tools/freeze_table.py`) before any result was read. |
>
> **Why §1–§4 are retrospective, plainly:** the overnight run was launched at 00:49 on request to have
> something on the GPU before the user slept, straight from a driver, with no plan written. That is a
> departure from the programme's prereg→log→postmortem loop and it is recorded here rather than
> papered over. The results stand on their falsifiers and their paired interval, not on a plan that
> did not exist.

---

## §1 Question *(retrospective)*

**M8** claims freezing recurrent state recovers gate AP@h18 `0.0070 → 0.0912`. It was measured on
`truncated_smoke` — **40 lessons, one seed** — which **M28** now classifies as having no skill at any
horizon. The pre-registered `violet_visitor` confirmation was never run, making M8 the primary suspect
in **#280**.

**The number that forces the question:** M8's *recovered* value (0.0912) is **3.6× BELOW** what an
L=300 model scores free-running with no intervention at all (0.3298, M34).

## §2 Intervention — the one variable *(retrospective)*

`freeze_recurrent ∈ {None, hidden, cell, all}` at inference. Explicit argument, never a config key, so
ADR-027's retirement of `freeze_h` is untouched and no production run can reach it. Emit-only on
existing L=300 ε=0 weights — **no retraining**, so the model is held fixed and only the rollout changes.

## §3 Endpoints *(retrospective)*

* **Primary:** gate AP at h18, free-running, target `sb`, calibration partition.
* **Secondary:** AP at h6 and h36; the per-horizon decay shape.
* **Reference:** the published free-running value for each seed (M34).

## §4 Decision rule as applied *(retrospective — this is the honest label)*

No numeric threshold was set in advance for EXP-01/02. What was applied after the fact:

* direction and consistency across two seeds;
* a **paired origin-block CI** excluding zero (`ap_diff_origin_block_ci`, added for this purpose);
* the effect judged against its **own** paired MDE rather than the SS sweep's between-seed MDE.

**This ordering is the weakness of EXP-01/02 and the reason §5 exists.** A rule chosen after seeing the
data is not a rule. It is mitigated — not repaired — by the falsifiers in §6 and by the interval being
4.5× its own MDE rather than marginal.

## §5 Decay dial — PRE-REGISTERED

**Question.** EXP-01/02 measured only the endpoints (`weight=0`, `weight=1`). Freezing recovers 23% of
the oracle gap and leaves 77% open, so a hard clamp is the most extreme setting of a dial nobody had
turned.

**Intervention.** Convex weight on the cell half: `w·anchor + (1−w)·new`, `w=1.0` byte-identical to the
hard clamp (original branch taken verbatim).

**Registered decision table** (verbatim from `DIAL_PAUSED.md`, committed before any arm ran):

| `cell@0.5` at h18 | reading |
|---|---|
| **> 0.3709** | a **dial** with an interior optimum — sweep 0.25 and 0.75 next |
| between **0.3318** and **0.3709** | **monotone ⇒ a switch**; the hard freeze is the answer, ship it |
| **< 0.3318** | non-monotone in the other direction; the mechanism is not what we think |

**Registered scope caveat**, also verbatim: *"Confirm any interior win on seed 42 before believing it —
one seed has decided nothing in this programme."*

**Outcome (EXP-03):** switch, and sharply saturating — a shape **none of the three branches predicted**.
Recorded as such rather than forced into the nearest branch.

## §6 Falsifiers — PRE-COMMITTED (enforced in code)

Both are implemented in `tools/freeze_table.py` and printed **above** the results table, so a failure
cannot be read past:

1. **h1 identical across all arms.** There is no feedback at step 1, so freezing cannot reach it. Any
   movement means the arms are not what they claim. *(Held: 0.47737082595880015 across all six arms.)*
2. **Every `none` arm reproduces its published free-running value** (M34: seed 43 0.3318, seed 42
   0.3298, tol 5e-4). A drift means the vehicle is not what we think and the table is unreadable.
   *(Held.)*

Both are sabotage-verified in `tests/test_freeze_table.py`.

## §7 Scope — what this cannot settle

* **One vehicle.** All arms are the same architecture and data.
* **AP only.** No CRPS claim; a frozen state is scored against the same ruler and the `crps_all`
  **ARTIFACT** verdict (#263) is untouched.
* **The paired CI and the dial curve are one seed.** EXP-01's endpoint effect replicates across seeds
  42/43; the interval and the dose-response do not.
* **This is a rollout-time intervention, not a fix.** +0.039 at h18 against an oracle ceiling near 0.50.
* **C-293's static-map worry is only weakened, not answered.** A frozen state is a static risk map by
  construction. The effect *growing* with horizon is consistent with both "the state carries real
  information" and "a static map beats a degrading gate", and no arm here separates them.

## §8 Postmortem — the process failure

**The prereg→log→postmortem loop was skipped for EXP-01/02.** The cause was a request to have work
running before the user slept; the effect was a decision rule chosen after seeing data. Two things
limited the damage — falsifiers were in code beforehand, and the effect is far outside its interval —
but neither is a substitute.

**The dial was done correctly**, and the difference is visible: §5's registered table did *not* predict
the observed shape, and that mismatch is itself informative. A retrospective rule would have quietly
accommodated the result and taught us nothing.

**Rule for the next run in this dossier:** no arm launches without a `05_analysis_plan.md` section
carrying a numeric decision rule and a commit timestamp that precedes it.
