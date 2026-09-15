# Pre-analysis plan — do the feedback-realism findings survive a vehicle that has skill?

**LOCKED 2026-08-17 01:55 CEST, before any arm ran.** Amendments append with a timestamp; nothing above
this line is edited after locking.

## Why

Every arm of the state-freeze probe (#277) and the feedback-realism probes (#278) ran on
`truncated_smoke` — a **40-lesson** vehicle whose own config comment reads `# SMOKE (not a scored
result)`. Reading `reports/2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv` (Epic #263, audited,
origin-block CI on every row) after those probes shipped shows the two vehicles are not the same
phenomenon:

| sb gate AP, free-running, `soft_gate` / `sample` / 36 steps / S=16 | h1 | h6 | h18 | h36 |
|---|--|--|--|--|
| `violet_visitor` (160 lessons, `nb`) | 0.4745 | 0.3924 | 0.2569 | 0.1370 |
| `climatology` (the FAO-02 reference) | 0.2980 | 0.2620 | 0.2251 | 0.1667 |
| `truncated_smoke` (40 lessons, `truncated_nb`) | 0.2979 | 0.0284 | 0.0070 | 0.0083 |

`truncated_smoke`'s h1 AP **equals climatology's** (0.29792 vs 0.29798), so it never had occurrence skill
at any horizon, and it collapses 42× by h18. `violet_visitor` degrades gracefully and is **REAL** against
climatology through h18 by the pre-registered `verdict_token`.

So the question is not whether the model has skill — it does. It is whether the *mechanism* findings
("occurrence placement carries 89% of the damage", "clustering is not sufficient", "sparsity is
survivable") describe **the rollout**, or only describe **an undertrained model failing**.

## Vehicle and scope

`violet_visitor` · artifact `calibration_model_20260812_191742.pt` · sha
`909f44c0096ee6b5c675f65539fcc31977bbe3ea6bb52438f2ed7b165237ddcb` (matches `partition_audit.json`'s pin)
· target `sb` · h = 1,6,12,18,24,30,36 · 13 origins (train 121–456, test 457–504, `leak: false`) ·
**S = 16** (`n_posterior_samples` 4 × `n_head_samples` 4) · truth pinned to
`620f4aa3…` · calibration partition · **seed 42, one seed** · eval-only, **no training, no config
mutation**.

`truncated_smoke` is also S=16, so the two vehicles are S-matched and `reference_sample_width`'s rule
holds between them.

## Arms

`identity` is **not re-run** — the production cubes for this exact artifact survived on disk and were
quarantined to `_quarantine_predictions_calibration_20260812_191742`. They *are* the control, because
`feedback_transform='identity'` is byte-identical to the production path (F3, tested in #278).

| arm | establishes | truncated_smoke reference |
|---|---|---|
| `identity` (control, on disk) | the free-running trajectory | AP h18 0.0070 |
| `use_real` | the oracle ceiling; denominator of every share | AP h18 0.3008 |
| `spatial_scramble` | the placement claim | 0.9% of the gap survived |
| `occurrence_real_magnitude_model` | E4a, occurrence share | 88.6% |
| `occurrence_model_magnitude_real` | E4b, magnitude share | 7.9% |
| `thin:0.75` | "sparsity is survivable" | 32× the control's AP |

## Predictions

| # | Prediction |
|---|---|
| **P1** | Re-scoring the preserved cubes reproduces `rescore.csv` **exactly** (tolerance 1e-9 — same cubes, same scorer, same pinned truth, so any difference is a defect, not noise). |
| **P2** | The oracle→control gap at h18 is **far smaller** on `violet_visitor` than the 0.2938 measured on smoke, because the control does not collapse. |
| **P3** | `spatial_scramble` destroys **materially less** of the gap than the ~99% it destroyed on smoke. |
| **P4** | The E4 occurrence/magnitude split differs from smoke's 88.6 / 7.9 by more than 10 points. |

## Falsifiers — checked and recorded BEFORE any prediction is read

| # | Falsifier | Consequence |
|---|---|---|
| **F1** | the preserved cubes do not reproduce `rescore.csv` exactly | the scorer, truth pin or cubes are not what produced the shipped board ⇒ **VOID, stop**. Would also mean the shipped board is unreproducible — a bigger finding than this experiment. |
| **F2** | `use_real` ≢ the real field on this vehicle (fed-field `active_fraction` / `mean_magnitude` do not match the real field's) | the transform reads the wrong month or channels here ⇒ VOID |
| **F3** | oracle − control gap at h18 < 0.05 AP | nothing to decompose on this vehicle; E4 shares are noise ⇒ report undecidable, **do not quote shares** |
| **F4** | h=1 `AP`/`Brier`/`crps_all` are **not identical across all arms** to 1e-6 | step 1 has no feedback, so every arm must agree there; a spread means something other than the feedback path moved ⇒ VOID |
| **F5** | `N` differs across arms or from the reference's 170430 at any (target, h) | the arms were scored on different supports and are not comparable ⇒ VOID |
| **F6** | any arm's fed-field statistics do not move its own axis on the real field (relations in §Arm separation) | that arm is a silent no-op; its score is void, **not** evidence that the axis does not matter |

### Arm separation — the relations that prove a transform bit on THIS vehicle

Fixture tests prove each transform moves its axis on a hand-built field. These prove it moved on
`violet_visitor`. Relational, so they transfer across vehicles:

* `active_fraction(spatial_scramble) ≡ active_fraction(use_real)` to 1e-6 — permutation preserves the multiset
* `neighbour_pairs_per_active(spatial_scramble) < 0.5 ×` that of `use_real` — clustering destroyed
* `active_fraction(occurrence_real_magnitude_model) ≡ active_fraction(use_real)` to 1e-6
* `mean_magnitude_on_active(occurrence_real_magnitude_model)` differs from `use_real`'s by > 5%
* `active_fraction(thin:0.75) = 0.25 × active_fraction(use_real) ± 5%`
* `active_fraction(identity) ≠ active_fraction(use_real)`

## Decision rule — pre-committed

Report `violet_visitor`'s occurrence/magnitude/placement shares beside `truncated_smoke`'s 88.6 / 42.6 /
7.9.

* **If they differ materially** — today's mechanism conclusions are **vehicle-specific**, and every
  inference row in `reports/RESULTS_LEDGER.md` §Claims Ledger is re-scoped to "measured on an undertrained
  vehicle with no occurrence skill".
* **If they replicate** — the mechanism survives a genuine generalisation test. Per the standing rule
  adopted 2026-08-17, that is an **escalation trigger** (second seed, third vehicle), **not** a conclusion.

## Stated confounds — before the run, not after

1. **Two axes at once.** `truncated_smoke` differs from `violet_visitor` in *both* training length (40 vs
   160 lessons) *and* body family (`truncated_nb` vs `nb`). This experiment cannot separate them. No
   existing run disentangles them either — all four models in `rescore.csv` are 160-lesson and none is
   `truncated_nb`. Any difference found is "smoke vehicle vs production vehicle", never "training length".
2. **One seed.** Seed 42 only. `bright_starship` and `violet_visitor` are both 160-lesson `nb` `soft_gate`
   models yet land on opposite sides of the verdict, so between-run spread on this board is large. Nothing
   here is a multi-seed claim.
3. **`spatial_scramble` carries C-291's irreducible confound** (structure vs geographic grounding); it is
   inherited unchanged from #278.
4. **Code has moved since the artifact was trained.** Seven commits touched `views_hydranet/` after
   2026-08-12. F1 passing exactly shows the *scoring* path is unchanged; F4 is what guards the *inference*
   path.

---

## AMENDMENT 1 — 2026-08-17 02:58 CEST, after the batch launched, before any arm completed

**The control changes from the preserved cubes to an `identity` arm run tonight.**

**Why.** The preserved cubes were written **2026-08-12 19:18**. Three commits touching the inference
path landed *after* them:

| commit | when | what |
|---|---|---|
| `d3a2626` | 2026-08-13 21:42 | `truncated_nb` body |
| `c07a352` | 2026-08-14 04:19 | SS piping + `truncated_nb` sampler |
| **`a2eabeb`** | **2026-08-14 19:28** | **per-site LockedDropout — independent MC-dropout masks per layer (C-128 S2)** |

`violet_visitor` evaluates with `evaluation_mode: 'stochastic'` and `dropout_rate: 0.15`, so MC-dropout
is **live at inference** and per-site masks change the posterior draws. Comparing today's treatment arms
against a control generated before that change would confound **each transform's effect** with **a
dropout change** — the exact class of defect this dossier exists to avoid, and one that would have been
invisible in every output except F4.

**What changes.** `identity` is queued as a sixth arm (chained after the batch via
`tools/chain_identity.sh`, because the batch had already launched) and **becomes the control**. Control
and treatment then share code, an artifact and a seed.

**What does not change.** No prediction or falsifier is edited. F4 stays exactly as locked and is now
*also* the cross-check on this amendment: if h=1 AP differs between `identity`-today and the preserved
cubes, that difference **is** the dropout change, measured.

**What the preserved cubes are still for.** (i) F1 — they proved `rescore.csv` reproduces bit-for-bit,
which stands. (ii) `identity`-today minus cubes-2026-08-12 now **measures what those three commits did
to the free-running path**, which is a byproduct worth having and was not otherwise on anyone's list.

**Scope note added.** Any share computed against the preserved cubes rather than against `identity`
would be contaminated; the verifier's control is the `identity` score CSV once it exists, and the
preserved-cube row is reported separately and labelled as 2026-08-12 code.
