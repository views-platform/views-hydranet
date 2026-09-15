# 05 — Pre-analysis plan, Wave 1 (LOCKED before the run)

**Locked 2026-09-03 02:49**, at commit `0a76a6a`, before the first arm started. The questions are
`06_question_battery.md` unchanged; this file records which of them Wave 1 actually answers, the
decision rule, and what was pre-committed.

## Intervention

**One variable:** `freeze_recurrent` ∈ {*none*, `hidden`, `cell`, `all`}, on the `identity`
(free-running) arm. Four seeds — 42/43/44/45 — same artifact family, 13 origins, emit-only.
Everything else held: model, composition, S, origins, code (a git-HEAD guard aborts an arm if the
repo moves mid-run).

## Decision rule — M48's, reused rather than reinvented

* **SUPPORTED** — 4/4 seeds agree in sign **and** |mean effect| exceeds the seed spread (sd)
* **CONTESTED** — 3/4, or the effect sits inside the seed spread. Reported as such, never resolved
  by choosing.
* **No p-values are claimed.** A paired sign-flip at n=4 floors at 1/16 = 0.0625 and **cannot** reach
  p ≤ 0.05. Saying so here prevents a later reader mistaking 4/4 for significance.

Per-seed values are printed beside every verdict: a mean that hides a 3/1 split is exactly the
failure `aggregate_seeds.py` refuses by design.

## Questions Wave 1 answers, with their pre-committed falsifiers

| ID | question | refuted if |
|---|---|---|
| **B.1** | does the cell's advantage hold for the **body**, or is it gate-only? | any freeze arm moves `size_ratio` off 0 at h18 or h36 |
| **B.2** | is `all` better than `cell`, or redundant? | `all` exceeds `cell` beyond the seed spread ⇒ the halves carry complementary information and M39's decomposition is incomplete |
| **B.3** | does freezing ever **hurt**? | no arm is worse than *none* on any head at any horizon beyond seed spread ⇒ the clamp is strictly dominant, which is itself a finding |
| **Q0.1** | is there any escalation skill to preserve? | \|rho\| < 0.05 at every horizon ⇒ C.1–C.3 are **UNANSWERABLE**, not null |
| **C.1/C.2** | does freezing flatten the dynamics? | dispersion under freezing ≥ dispersion under *none* |
| **C.3** | does the flattening cost direction skill? | — the trade-off readout; no falsifier, it is a measurement |
| **C.4** | does freezing buy **continuation** at the cost of **onset**? | the gain on new cells is at least as large as on continuing cells |
| **D.1** | does any of it replicate? | the decision rule above |

## Verification, pre-committed

* **Reproduction falsifier.** The dump change did not touch the cube path, so the re-run seed-42
  `none` and `cell` arms must reproduce the archived `AP@h18` values
  (`0.3298395823400329`, `0.3621885544392029`) **exactly**. If they move, something other than the
  dump changed and the whole wave is suspect.
* **h1 identity.** All freeze arms must be identical at h1 within a seed — the clamp acts only for
  `t > origin`.
* **Instrument provenance.** Every dump must record `n_passes = 4`; the retired pass-0 instrument
  reproduced the scorer's AP only to 10.7% and must not be mixed in silently.
* **Completeness.** 13 origins per arm; a partial dump directory is a failed arm, not a short one.

## Known defect in C.4, recorded when it was found (2026-09-03, seed-42 interim)

C.4's falsifier says "at least as large" **without naming the measure**, and absolute gain,
relative gain and skill-above-base do not agree. All three are reported; none is chosen post hoc.
This is the **third** mis-specified gate in the programme (after F3's unsatisfiable band and FR-4's
untested assumption), and the root cause is the same each time: a threshold written without first
checking what the statistic can deliver.
