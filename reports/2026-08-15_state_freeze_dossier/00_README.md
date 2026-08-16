# Recurrent-state freeze probe — does the rollout gate collapse live in the model's memory?

**Status:** **RUN — verdict STATE-IMPLICATED, but NO ARM CLEARS A PERSISTENCE BASELINE.** `truncated_smoke`
(40 lessons, one seed) — **INDICATIVE**; `violet_visitor` confirmation not run.

> **Read this before the numbers.** Holding the recurrent state recovers ~23% of the oracle gap relative to
> the collapsed control — and every arm still scores **below persistence** (repeat the last observed map)
> from h6 onward. It is not a skill claim. Two further corrections are recorded in `07_experiment_log.md`:
> the damage is **not** localised to the cell state (C-292 — `hs` is a readout of `hl`, so `cell ≈ all` is
> architecturally predetermined), and the earlier "CONFIRMED AND QUANTIFIED" phrasing overclaimed against
> this dossier's own pre-registration.
**Parent:** #258 (rollout collapse) / #262 (the training-lever handoff). **Register:** C-222.

## The question in one line

The gate finds conflict well at month 1 (AP 0.30) and not at all by month 18 (AP 0.01). **Is that because
the model's memory gets poisoned by feeding on its own predictions?**

## Why this is open, when #262 said it was closed

`#262` recorded the collapse as *"NOT hidden-state / recurrent drift (overturns the prior C-222-based bet)"*,
on the strength of the oracle probe: feed the model real data each month and gate AP holds 0.30 → 0.27 to
month 36.

**That does not follow.** The oracle changes the *input* and lets memory update normally, so it shows memory
is fine when it is never polluted. The free-running case is the one where memory *is* polluted, and nothing
tested it. C-222 names this exact confound.

Separately, the maintainer recalls that freezing the state used to preserve classification skill across the
horizon. `reports/results_freezeh_ablation.md` (2026-06-04) ran the same four arms and found freezing inert —
but its endpoint was **regression CRPS**, on a pre-ADR-070 artifact that exploded at ~1e17 in every arm.
It never measured classification, and activation-aware metrics did not exist until 2026-08. So the
recollection is **untested, not contradicted**.

## What is run

Four arms on one saved artifact, emit-only, nothing retrained: the state's short-term half held, its
long-term half held, both held, and neither (the control). Hold starts after the seed step, so month 1 is
identical in every arm by construction — which is the probe's own self-test.

Pre-registration and decision rule: **`05_analysis_plan.md` (LOCKED)**. Results: `07_experiment_log.md`.

## Result (2026-08-15, `truncated_smoke`)

**STATE-IMPLICATED** by the pre-registered rule, threshold untouched. Gate AP at h36:
`none` 0.008 → `hidden` 0.025 → `cell` 0.067 → `all` 0.069, against an oracle ceiling of ~0.271.

* **It is the long-term (cell) memory** — `cell` alone carries 89% of the combined effect.
* **23% of the oracle gap** is recovered by a total freeze, so state corruption is *a* mechanism, not
  *the* mechanism. #262's distributional-gap thesis still holds most of the remainder.
* **Magnitude is untouched** (`size_ratio` 0.0000 at h≥18 in every arm) — P3 as pre-registered.
* **`crps_all` is blind to all of it** — the four arms agree to 3 decimals at h18 while gate AP spans 13×.

Full numbers and the falsifier verdicts: `07_experiment_log.md` EXP-02.

## What a result means

- **State implicated** → the interesting object is a *soft* prior: let memory update, but less, or less
  confidently, while the model is feeding on itself. Not a hard on/off switch.
- **State inert** → cross it off, settle C-222 negative, and the search returns to #262's three training
  approaches with one fewer live hypothesis.

**Neither outcome reinstates `freeze_h`.** ADR-027 retired it and that stands; `freeze_recurrent` is a
diagnostic argument with no config key, so no production run can enable it.

## Layout

| path | what |
|---|---|
| `05_analysis_plan.md` | the LOCKED pre-registration — predictions P1–P3, falsifiers F1–F3, decision rule |
| `07_experiment_log.md` | append-only record, one entry per arm set, verdict against each falsifier |
| `tools/run_freeze_arms.py` | the batch driver — one subprocess per arm, score-then-delete, disk preflight |
| `tools/freeze_arm_entry.py` | one arm: `HydranetManager` subclass that sets the arm on the orchestrator |
| `results/` | per-arm score CSVs, logs, and the run manifest |

The mechanism itself lives in the package, not here: `views_hydranet/utils/hydranet_inference.py`
(`blend_recurrent_state`, `HydraNetInference.freeze_recurrent`), tested by
`tests/test_recurrent_state_freeze.py` — twelve tests, each checked against a deliberate sabotage.
