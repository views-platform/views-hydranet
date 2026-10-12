# Pre-analysis plan — does moving occurrence to the gate fill the map?

**Written 2026-10-12, BEFORE the arm has been trained.** Issue #396's diagnosis is closed; this is
the single intervention it recommends. One arm, ~$2.25, one readout.

⛔ **Nothing launches on my say.** Simon approves the two merges and the spend.

---

## 0. The established facts this starts from

All measured on the 2026-10-12 global-land vehicle (`4acbf4d3c2b457b0…`, 300 lessons, the delivered
`violet_visitor` config), against its own emitted draws archive.

| | | source |
|---|---|---|
| observed positive rate, 100 most-active land cells, honest out-of-sample | **0.6944** | M5 |
| as-shipped emitted positive fraction, same cells | **0.4290** | P0 rung C |
| the body supplies this share of the emitted zeros | **93–99%** | T0 / P0 (convention-dependent) |
| cell-steps the `threshold_gate` actually kills | **3.9%** | P0 |
| draws that are zero *because of* the gate | **0.52%** | P0 |
| **the gate is calibrated**: median \|gate − observed p\| over 842,634 cell-months | **0.0054** | P2 |
| the gate's implied `pos_weight` | **1.0365** | P2 |
| recomposition of the model's own output, nothing retrained | P(Y=0) **0.53 → 0.3055** | M1 |
| reconstruction reproduces the written draws (residue, 95% CI) | **+0.0034 [−0.0051, +0.0118]** | P1 |

**The mechanism, stated plainly.** The gate knows where conflict is and says so correctly. The body
also decides occurrence — it draws zero in ~57% of draws — and `threshold_gate` keeps the body's
answer wherever the gate clears τ, which is 96.1% of the time. So the calibrated half is discarded
and the miscalibrated half decides. Under `soft_gate` with a zero-truncated body, `P(body>0) = 1` by
construction, so `emitted P(y>0) = gate` exactly, and occurrence moves to the half that is right.

## 1. Hypothesis

**H1.** The emitted zero mass on active cells is a composition artefact, not a limit of the model's
learned representation. Replacing the composition triple will raise the emitted positive fraction
from ~0.43 to ~0.69 **without** a loss of gate skill, because nothing about the gate's weights or
training changes — only what its output is used for.

## 2. The intervention — ONE variable

The composition triple, and nothing else:

| | control (`violet_visitor`) | arm (`amber_kestrel`) |
|---|---|---|
| `output_distribution` | `nb` | **`truncated_nb`** |
| `forecast_composition` | `threshold_gate` | **`soft_gate`** |
| `gate_threshold` | `0.20` | **omitted** (ADR-069 refuses it under `soft_gate`) |
| `loss_class_pos_weight` | 2.0 | 2.0 — **unchanged, per P2** |
| everything else, incl. seed 42, 300 lessons | — | identical |

Config identity verified by diff on both sides; the symmetric difference is exactly those three
keys. The arm is pinned in **`EXPERIMENT_ARMS`**, *not* `ROSTER` — `ROSTER` is read as
`rusty_bucket`'s ensemble membership, and a ninth member would be pooled into the ensemble.

**No hyperparameter is tuned.** P2's verdict D2 is what makes this a pure composition change, and it
is the single most important difference from M45.

## 3. ⛔ Skepticism ledger

**Six interventions have failed, and four failed on the same lever** — AP loss scales with how much
the model fires (M45 `truncated_nb` ×1170 → **−0.238 AP**; M62 act_ratio ×3.0; M64 ×56, −0.196 AP).
**This is the seventh intervention aimed at firing, and it uses the same family M45 used.**

*Why I think it differs:* M45 truncated the body and left `threshold_gate` and a gate calibrated for
a body that was still absorbing ~99% of the zeros. The two halves then both owned occurrence and
fought, which is exactly the compounding M45 recorded. Here occurrence is moved to one half, and that
half is **measured** calibrated (P2) rather than assumed.

*Why it might not:* every failed intervention also had a mechanism story, and M45's was not obviously
worse than this one. If AP loss scales with firing *irrespective* of which component produces it,
this fails identically and the diagnosis in #396 is incomplete.

*My honest prior:* **~60%** that P1 holds and F1 does not fire. Not high. Stated so it can be scored.

## 4. Pre-registered predictions

- **P-1 (primary).** Emitted positive fraction, 100-cell set, h=1, pooled over the 13 origins:
  **[0.62, 0.76]**, centred ~0.69. Derived, not hoped: emitted `P(y>0) = gate`, the gate's median on
  these cells is 0.6945, and the honest observed rate is 0.6944.
- **P-2.** Composed `P(Y=0)` on the same set ≈ **0.31** (M1's recomposition of the control's own
  parameters), against the as-shipped 0.53.
- **P-3.** Pooled gate AP, all land cells, 13 h=1 months: within **±0.05** of the control's
  **0.5631**. The gate's weights are trained by the same loss on the same targets; only the
  downstream use changes. *(A change beyond that is informative in either direction.)*
- **P-4.** `act_ratio` rises at h=1 but does **not** compound with horizon as M45's did (×8–22 by
  h18). Prediction: **< 3×** the control's at h18.
- **P-5.** `crps_all` does **not** worsen. M6: CRPS-optimal θ is 0.4298 against the model's 0.1544;
  M4: the model is *already* worse on standard CRPS. The metric should reward this.

## 5. Falsifiers — pre-committed

| | fires if | then |
|---|---|---|
| **F1** | pooled gate AP **< 0.4131** (control 0.5631 − 0.15) | **KILL.** The composition cost real occurrence skill. |
| **F2** | emitted positive fraction **> 0.85** | the gate did not stay calibrated once it owned occurrence alone. A **finding**, not something to tune away. |
| **F3** | `act_ratio` at h18 **> 5×** the control's | M45's mechanism reproduced; the composition is not the fix and #396's diagnosis is incomplete. |
| **F4** | `crps_all` worsens by **> 5%** | fails FAO-02's own bar, in the wrong direction. |
| **F5** | the arm's config differs from the control anywhere but the composition triple | **VOID** |
| **F6** | artifact arrives without all three files + `PROVENANCE` with `lessons ≥ 300` and `mode: production` | **VOID** — reuse `2026-10-12_underfire_diagnosis/vehicle.py`'s gate verbatim |
| **F7** | training diverges, NaNs, or the stability gate trips | **VOID, not negative** (C-307: a null from an unstable run is uninterpretable) |

### ⚠️ How F1 is sized, and the tension in the sizing

Two figures on record disagree about AP noise by 6×:

- the persistence re-reference measured **AP seed sd ≤ 0.019** directly, n=4, at L=300;
- **C-119** records **~20% training variance**, which on 0.5631 would be **0.113**.

Month-to-month sd measured here is **0.0413**. **F1 is set at −0.15 so it is defensible under the
pessimistic reading too**: 7.9× the optimistic sd, 1.3× the pessimistic one, and still below both
collapses actually on record (C-228's gate AP 0.31→0.13 = −0.18; M45's −0.238).

⛔ **Consequence, pre-committed:** an AP change between **−0.15 and 0** is recorded as
**UNRESOLVED at n=1** — explicitly *not* a pass. The arm could not ship on that basis without a seed
replication. F1 is a collapse detector, not a measurement of small AP differences, and it must not
be read as one.

## 6. Method

- **One arm.** The control is the existing 2026-10-12 vehicle: same config, same 300 lessons, same
  global land, with its 13 origins of draws already on disk. `models` verified that moving artifact
  collection between `-t` and `-e` is behaviour-neutral — evaluation reloads from disk
  (`hydranet_manager.py:367`) and re-seeds **absolutely** via `lock_entropy`, so two processes sample
  as one. **If anything else in that PR touches training, the vehicle stops being a valid control and
  this becomes two arms.**
- **Readout** by the reconstruction instrument validated in P1, plus the arm's own draws archive.
- **Primary slice: h=1**, the only matchable one (P1: teacher-forced in both input and state;
  `hydranet_inference.py:1219-1221`). h18/h36 are read **only** for F3.
- **Statistic:** pooled mean positive fraction. ⛔ **Not** the median over cells — P1 established it
  is quantised at 1/16 = 0.0625, coarser than the effects being compared.

## 7. ⛔ Definition of done, and the stop rules

**Done when:** a number for P-1 through P-5; each falsifier marked fired or not; one sentence saying
whether the composition fix works; a `found, not chased` list.

**Then stop.** Specifically:

- ⛔ **One arm. No sweep.** If the arm fails, the next step is a decision, not another arm.
- ⛔ **Nothing is added mid-run.** Discoveries go on `found, not chased`.
- ⛔ **F1 or F7 is a hard stop** — no further readout is interpreted.
- Budget: **~$2.25, one pod, ~3 h.**

## 8. Out of scope

- `ns` / `os`. **`sb` only.**
- The free-running roll and its contribution to the emptiness (C-113 territory).
- The magnitude ceiling (#241). I-A: magnitude carries ~0% of the skill.
- **Why both halves fall short of their own objectives** — P2 found the gate sits below its BCE
  optimum, M6 found the body below its NLL optimum, and the "objective rewards emptiness" story is
  retracted. Two independent halves under-reaching in the same direction is the most interesting
  open question in the record and it is **not** this experiment's question.
- The gate cube not travelling home. P1 made reconstruction trustworthy, so it is no longer blocking;
  an issue, filed after this run.

## 9. Glossary

`reports/GLOSSARY.md` locks `gated_NB` (soft-gated NB) and `th_gated_NB` (threshold-gated NB) but has
**no locked name for a soft-gated zero-truncated body**. Per the standing rule the glossary is edited
rather than a synonym invented: this arm is to be named there before the run, and this document uses
the config triple explicitly until it is.
