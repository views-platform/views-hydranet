# Measuring the feedback realism gap — which statistic does rollout skill depend on?

**Status:** **COMPLETE** (2026-08-16) — 13 arms + a gate-structure probe.

## Result

**The rollout collapse is a spatial-structure failure.** Scrambling only the *locations* of an otherwise
perfect field reproduces it almost exactly (gate AP 0.3008 → 0.0097, against free-running 0.0070) — with
identical active count and identical magnitudes.

| what carries the damage | share |
|---|--:|
| occurrence being **correct** (right places) | 89% |
| of which: occurrence merely being **plausible** | 43% |
| magnitude | 8% |

Sparsity is survivable on its own: `thin:0.75` matches the collapse's activation rate (0.33 vs 0.27) with
**32× the AP**. Under-firing is not what kills the gate; misplacement is.

**A follow-up probe falsified the obvious mechanism.** The gate does not merely have its structure discarded
by the independent-Bernoulli sampler — the gate's own probability field smears out (Moran's I 0.50 → 0.16 by
step 6). But independent sampling is **~10× more destructive on a diffuse gate than a sharp one** (25× vs
2.6×), so the two compound in a loop. A coherent sampler would move the fed field from 0.010 to ~0.25
clustering (target 0.449) with no retraining; whether that breaks the loop has to be run.

This also explains why coordinate channels never helped (C-152): coords improve *which cells are likely* —
a marginal property — while what fails is the **joint** spatial coherence plus an independence-assuming
sampler.

**Scope:** 40 lessons, seed 42, one origin set, one target, one vehicle. Indicative; the ordering is the
result, not the magnitudes. **Parent:** #258 / #262. **Glossary:** `the feedback realism gap`.

## The question in one line

The model's emitted field is not distributed like real conflict history, so feeding it back poisons the
rollout. **Which property of the field does the damage?**

## Why this, and why now

We have only ever measured the two endpoints: `teacher_forced` (a real field — gate AP 0.30 → 0.27 at h36)
and fully generated (0.30 → 0.01). Everything between is unmeasured, and the mechanism lives there.

Every candidate fix needs the answer:

| fix | what it needs to know |
|---|---|
| distribution matching | **which statistics to penalise** — it only fixes what you name |
| Professor Forcing | where to attach the critic |
| K-step rollout training | how many steps carry the damage |

Today we know one drift number (`P(on|on)` 0.418 real vs 0.090 free-running) and **zero** sensitivities.

## The decomposition

```
skill loss  =  Σᵢ  [ how far statistic i has drifted ]  ×  [ how much skill depends on statistic i ]
                    ← E1 measures this                     ← E2 measures this
```

**E1** fingerprints the generated field per horizon. **E2** degrades the *real* field one axis at a time and
watches skill fall. **E3** feeds a real field from the wrong month — realistic but uninformative — which can
invalidate the whole distribution-matching family in one run. **E4** splices the model's *where* with the
real *how much*, and the mirror.

E2 degrades from the good end, E4 repairs from the bad end; together they bracket a response curve that is
known to be sharply nonlinear (a cliff to h6, then saturation).

## What makes the numbers trustworthy

- **`use_real` byte-reproduces `teacher_forced`.** If the transform read the wrong month or the wrong
  channels, every arm would be confidently wrong while looking healthy. Checked as a CI test *and* against
  the archived real numbers.
- **Every arm self-reports the field it actually fed** (`fedfield_*.csv`, per origin × sample × step ×
  target). The fixture tests prove a transform moves its axis on a hand-built field; this proves it moved on
  the real one. An arm whose statistics did not shift is a **silent no-op**, and its score is void rather
  than being read as "this axis does not matter".
- **Every guard was checked against a deliberate sabotage** before any arm ran.
- The arms are scored by the unchanged `score_v2_horizons.py` + `activation_metrics.py`, so every number is
  comparable to EXP-SS-2, the state-freeze dossier and the v2 board.

## Two design decisions worth knowing

**The persistence axis is `shuffle_months`, not a spatial roll.** A torus roll was built first and rejected:
it preserves the statistics, but the grid is a map of Africa, not a torus — rolling a blob off the east edge
lands it in another country while the coordinate/static channels stay fixed, confounding "persistence
broken" with "field decoupled from geography". Caught by a failing orthogonality test.

**`spatial_scramble` carries an irreducible confound.** Destroying clustering necessarily breaks the field's
alignment with the statics, because plausible locations *are* the clustering. Read that arm as "spatial
structure **and its geographic grounding**". Stated before the run, not after.

## Layout

| path | what |
|---|---|
| `05_analysis_plan.md` | the LOCKED pre-registration — P1–P4, F1–F6, decision rule |
| `07_experiment_log.md` | append-only, verdict against every falsifier before any prediction is read |
| `tools/run_realism_arms.py` | batch driver — one subprocess per arm, score-then-delete, disk preflight |
| `tools/realism_arm_entry.py` | one arm; also writes that arm's fed-field record |
| `results/score_*.csv` | the ruler's output per arm |
| `results/fedfield_*.csv` | what each arm actually fed, per origin × sample × step × target |

The mechanism lives in the package: `views_hydranet/utils/feedback_field_transforms.py` and
`HydraNetInference.feedback_transform`, tested by `tests/test_feedback_field_transforms.py` and
`tests/test_feedback_transform_seam.py`. It is a diagnostic argument with **no config key**, so no
production run can enable it.
