# ADR-075: Emission selection — a config axis for *which* artifacts a run writes, distinct from composition

**Status:** Proposed
**Date:** 2026-11-01
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Sibling of:** ADR-069 (`forecast_composition` as a config axis) · **Requires:** ADR-074 (emission provenance) · **Builds on:** ADR-067 (distribution family subsystem), ADR-009 (boundary contracts and configuration validation) · **Respects:** D-3 (no new emitting family)

## Summary (read this first — self-contained)

A HydraNet cell forecast has two parts: a **body** (how many, if any) and a **gate** (the chance of
any violence). ADR-069 made *how they combine* a config setting — the **composition**. What that
ADR did not make configurable is *which of the three resulting quantities a run actually writes to
disk*.

Today it is an **either/or**. `forecast_composition: threshold_gate` writes the gated forecast;
`self_zeroed` writes the plain body. One run cannot produce both, because composition is applied
*before* the array is written (`utils/hydranet_inference.py:1752`). The gate is written alongside
either, so in practice you get two of the three quantities and can never choose which two.

That is a problem because the body is not recoverable from the gated forecast. Measured on a real
global gate posterior: under `threshold_gate` at τ=0.20, **99.54%** of cell-months have their body
zeroed outright. The information is gone, not noisy.

This ADR adds three config fields — `emit_quantities`, `emit_origins`, `emit_horizons` — that say
**what a run writes**, additively. It is deliberately a *separate* setting from
`forecast_composition`, because that key is also read by the autoregressive rollout: driving
emission from it would silently change every forecast horizon beyond the first.

It changes no default. A config that sets none of these fields emits exactly what it emits today.

## 1. Context

**Problem 1 — emission is an either/or, and the body is the casualty.**

`compose_samples` (`distributions/composition.py`) is applied inside
`to_cube_samples(..., gate=prob_thwc, composition=...)` at `utils/hydranet_inference.py:1752`,
before `posterior_magnitudes_zstack` is populated. So the magnitude cube on disk is **already
composed**. Under `threshold_gate` the mask is a deterministic per-cell 0/1, so cells at or above τ
pass through untouched and cells below τ are zeroed for every draw.

Measured on `views-models/models/amber_kestrel/.../gate_h1/` (13 origins, 842,634 cell-months, h=1,
global land):

| | cells | share |
|---|---|---|
| gate < 0.20 — body **destroyed** | 838,795 | **99.54%** |
| gate ≥ 0.20 — body recoverable exactly | 3,839 | 0.46% |

⇒ Reconstruction covers 0.46% of the grid. ⚠️ Note that `_dump_body_mean`'s docstring
(`hydranet_inference.py:987-991`) argues this case from `soft_gate`'s `1/gate` variance inflation.
Under the composition we actually run the loss is **total**, not noisy — the conclusion is right and
the stated reason does not apply. That docstring should be corrected when the file is next touched.

**Problem 2 — the existing diagnostic seam cannot be reached in production.**

Every diagnostic on the orchestrator is a plain attribute defaulting to `None`:
`body_mean_dump_dir`, `record_gate_probe`, `feedback_length_scale`, `freeze_anchor_roll`,
`per_step_roll` (`utils/inference_orchestrator.py:93-109`).

`manager/hydranet_manager.py:353` constructs the orchestrator and **sets none of them** before
calling `generate_prediction_frames`. A production or pod run is `main.py` → `ModelManager` →
`HydranetManager` → orchestrator.

⇒ On any run that is not driven by a bespoke local script, the seam is unreachable and writes
nothing. A config field is therefore **forced**, not merely preferred.

**Problem 3 — horizons and origins are not selectable, and one of the seams is already built.**

Nothing in views-hydranet or pipeline-core selects targets, origins or horizons for emission. Both
orchestrator paths loop the full origin list and the full target list; the whole `(T,H,W,·,S)` cube
becomes rows. The only existing per-origin gate is a visualisation one
(`inference_orchestrator.py:169-172`, `biopsy_volume` fires only on `origin_idx == 0`).

⭐ `utils/utils_orchestration.py:8,25` already accepts `fixed_last_origin` — an origin-selection
parameter that **only tests pass**. `hydranet_manager.py` never wires it.

**Assumption no longer true.** ADR-069 assumed one composition per run is what a run needs, because
the composed forecast was the object under study. With the body itself now the object of study
(silence-vs-fade, gate/body decomposition, the composition programme), a run must be able to emit
more than one view of the same inference pass — and paying for a second full inference pass to get
the second view is both wasteful and **not comparable**, because MC-dropout draws differ between
passes.

**Urgency.** The next experiment needs the body. Without this, the only routes are a bespoke local
driver (which cannot run on a pod — Problem 2) or a second inference pass (not comparable).

---

## 2. Decision

**Statement.** *We will make **which quantities, origins and horizons a run emits** a first-class
config axis, additive and separate from `forecast_composition`, so that one inference pass can
yield every view of itself that an experiment needs.*

### 2.1 The three fields

| field | declares | default |
|---|---|---|
| `emit_quantities` | which of the **gated forecast**, the **gate** and the **body** are written, and therefore which directory prefixes appear | today's set — existing configs byte-identical |
| `emit_origins` | all origins, or a named subset. Wires the existing `fixed_last_origin` seam | all |
| `emit_horizons` | all horizons, or a subset | all |

**Additive, not exclusive.** `emit_quantities` is a *set*. Asking for the gated forecast and the
body yields both, from one pass, from the same draws — which is the entire point and is impossible
today.

### 2.2 ⛔ Emission is a separate setting from composition, and must never touch the feedback path

`forecast_composition` is read at **four** sites, and two of them are not emission:

| site | role |
|---|---|
| `hydranet_inference.py:1752` | the scored sample cube — **emission** |
| `hydranet_inference.py:1056` | `_emit_magnitude` — the point forecast **fed back** into the rollout |
| `hydranet_inference.py:1127` | `_sample_feedback` — the sampled feedback draw |
| `hydranet_inference.py:519` | a config consistency check |

⇒ Expressing "also emit the body" by varying `forecast_composition` would change **what the
autoregressive rollout feeds back**, and therefore every horizon h≥2. The run would complete, write
correct shapes and correct row counts, and not be the same model. Nothing would report it.

**Decision: `emit_quantities` is its own field. It selects what is written. It does not reach the
feedback path, and a test asserts that h≥2 output is bit-identical with it on and off.**

This is the single rule in this ADR whose violation corrupts a result rather than failing a run.

### 2.3 Compute-all, emit-some

An autoregressive rollout cannot skip months: horizon 36 exists only because 1…35 were computed.

⇒ `emit_horizons` and `emit_origins` select what is **written**, never what is **computed**. They
are not a speed-up, and their documentation must say so — otherwise someone will set a single
horizon expecting a short run and get a full one.

#### 2.3.1 `steps` is a horizon COUNT, not a horizon SELECTOR

`emit_horizons` is the *only* way to select horizons. The `steps` configuration field is not, and
today it looks as though it is.

Only the **length** of `steps` is ever read — by the training-window slice, by the checksum against
the declared step count, and by the reproducibility record. The step **values** are read nowhere on
the inference path, and there is no contiguity check and no value check.

⇒ A configuration declaring a sparse step list passes every gate, rolls out that many
**consecutive** months, and labels them as though they were the horizons named. Nothing reports it.

**Decision: `steps` must be contiguous ascending from one, or the configuration raises.** A sparse
set of horizons is expressed with `emit_horizons`, never by relabelling `steps`.

This belongs here rather than with provenance (ADR-074) because it is the same decision as §2.3: a
field that *appears* to select horizons but only sets their number is the inverse of
compute-all-emit-some, and the two rules are only coherent stated together.

✅ Latent, not firing: every live configuration declares a contiguous range from one, so **no past
result is affected.**

### 2.4 The body is a diagnostic quantity, never a pipeline target

The body is emitted **outside** the target system — it is not added to `regression_targets` or
`classification_targets`, and it does not enter `combined_targets()`.

This is a correctness decision, not a convenience one. `views-pipeline-core`'s
`managers/model/model.py:474-486` requires every name in `combined_targets` to exist as a
ground-truth column, because evaluation slices actuals by target name. **There is no observable
called "the ungated body."** It is a component of a forecast, not a forecast of anything in the
world. Declaring it a target is a category error, and four independent guards already refuse it:
`config_initializer.py:612-621` (the 3-regression-head invariant),
`views-models/tests/test_roster_conformance.py:531,540`,
`views-evaluation/.../native_evaluator.py:256`, and `model.py:474-486` itself.

⇒ ✅ **D-3 is respected: this adds no emitting family.** The body is an existing quantity of the
existing family (`fam.mean` / `fam.sample`, already computed on the inference path and already
dumped as a mean by `_body_mean_field`). Nothing new is fitted, trained or scored.

### 2.5 In-scope / out-of-scope

**In scope:** the config fields, their validation, and HydraNet honouring them.

**Out of scope:**
- **What comes home.** Transport, archive selection and the pod runner belong to views-models and
  are filed as issues there. ⭐ views-models already has the right vocabulary for it — ADR-019's
  delivery declaration, with its `DELIVERY` decides / `REQUIRE` refuses split. This ADR does not
  fork a second selection language; it gives that one something honest to select.
- **Editing model configs.** Which models set these fields is a views-models decision.
- **The on-disk format of the body artifact.** Implementation, constrained by ADR-074.
- **Composition semantics.** ADR-069 stands unchanged. This ADR adds no composition value and
  changes no existing one.

---

## 3. Rationale & Integrity Impact

| alternative | why rejected |
|---|---|
| overload `forecast_composition` (ADR-069's key) | ⛔⛔ §2.2 — it is read by the rollout feedback. Would silently alter every h≥2 while every shape and row count stayed correct. The one option that corrupts rather than fails. |
| the orchestrator-attribute route, as the five existing diagnostics do | **measured unreachable in production** — `hydranet_manager.py:353` sets no attribute, so a pod run writes nothing. Not a preference; a dead end. |
| declare the body a pipeline target | category error — §2.4. Correctly refused by four guards. |
| run inference twice, once per composition | wasteful, and **not comparable**: MC-dropout draws differ between passes, so the two views would not be of the same posterior. |
| reconstruct the body from the gated forecast and the gate | covers **0.46%** of cells — §1. |
| a general "artifact selection DSL" inside views-hydranet | machinery that does not earn itself here, and it would fork views-models' ADR-019 vocabulary. Three config fields are the least that solves the actual problem. |

**Fortress state.** Reproducibility improves: an experiment that needs the body and the gated
forecast gets both from **one** pass over **one** set of draws, so a body-vs-composed comparison is
of the same posterior rather than of two samples of it. Combined with ADR-074, what was emitted is
also recorded.

**Fail-loud.** Validation refuses rather than degrades (§5): an `emit_quantities` entry the
configured family cannot produce raises at **config construction**, and the memory preflight raises
**before** allocation rather than letting the host OOM-kill the run — a failure already observed on
this work.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The body becomes obtainable at all, and obtainable **alongside** the gated forecast from one
      pass — the thing that is impossible today.
- [ ] Emission becomes declarable from an ordinary model config, so it works on a pod. No driver
      script, no attribute, no never-modify-list file touched.
- [ ] Horizons and origins become selectable, which (with ADR-074 naming them) makes a
      horizon-indexed readout expressible for the first time.
- [ ] An already-built but unwired seam (`fixed_last_origin`) is put to use rather than duplicated.
- [ ] The emission/composition confusion is closed by construction rather than by care.

### ⚠️ Negative (Costs)

- [ ] **Memory and disk.** Emitting the body adds a third `(T,H,W,n_reg,S)` cube: **+50% peak RAM**
      (two cubes become three — `hydranet_inference.py:1655-1656`) and, at production shape,
      **2.09 GiB per origin at S=20 — ~27 GiB across 13 origins.** Mitigated by a hard preflight,
      not by hope.
- [ ] Three more config fields on a schema that already has ~104, each then on the path of every
      later change. Accepted: they are plain declarative fields with defaults, not guards.
- [ ] `emit_horizons` will be misread as a speed-up by someone, despite §2.3. Its docstring is the
      only defence.
- [ ] A request that cannot be satisfied now fails at config time rather than producing something
      adjacent. Intended, and louder than today.

---

## 5. Validation

**Invariants that must remain bit-perfect.**
- ⭐ **With `emit_quantities` on and off, forecast horizons h≥2 are bit-identical.** This is the
  test that guards §2.2 and the most important assertion in either ADR.
- A config setting none of the three fields produces output byte-identical to a pre-ADR baseline at
  the same seed.

**Tests.**
- *Red:* an `emit_quantities` value the configured family cannot produce raises at config
  construction.
- *Red:* the cube-fit preflight (`utils/disk_guard.py:64`, already variadic over `*shapes`) raises
  **before** `np.zeros` when the extra cube does not fit. Assert the raise, not an OOM.
- *Green:* with the body emitted, `body × (gate ≥ τ)` reproduces the gated cube **exactly** on the
  cells at or above τ. This proves the emitted body is the same body that was composed, not a
  second draw.
- *Green:* `emit_origins` / `emit_horizons` subsets change only *which* artifacts exist; the arrays
  that do exist are bit-identical to the corresponding slices of a full run at the same seed —
  i.e. §2.3's compute-all guarantee holds.
- *Green:* the ADR-074 manifest records the quantities actually emitted, for every combination
  tested above.

**Failure mode that would trigger reconsideration.** If a future emitted quantity cannot be
expressed as a set member — for instance one requiring its own inference pass or its own loss —
then `emit_quantities` has become a dispatch mechanism rather than a declaration, and the axis
needs rethinking rather than extending.

---

## 6. Implementation Notes

- **Location.** Fields and validators in `utils/config_initializer.py` (ADR-009's boundary-contract
  layer). Honouring them is in the inference path and the orchestrator; the writer is the module
  ADR-074 introduces.
- **Ordering.** ADR-074 is a prerequisite: emitting a third quantity without recording which
  prefix carries which quantity would make the ambiguity it fixes strictly worse.
- **Note, not fixed (out of scope).** Forecasting runs write no `origin_N` directory level at all
  (`views-pipeline-core/.../model.py:1637-1643`) while calibration/validation/test do
  (`:1459-1465`), so the two run types have different directory shapes. Real, recorded here,
  **not addressed by this ADR.**
- **References.** ADR-067, ADR-068, ADR-069 (the composition axis this is the sibling of), ADR-074;
  views-models ADR-019 (delivery declaration) and ADR-025 (a check that cannot see must refuse).
- **Evidence.** `reports/2026-11-01_body_output_blast_radius/00_REPORT.md`; the 99.54% measurement
  is reproducible from `views-models/models/amber_kestrel/data/generated/calibration_delivery_20261013/gate_h1/`.
