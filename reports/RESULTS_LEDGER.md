# Results Ledger — HydraNet experiments (C-113 program & beyond)

**Purpose:** a durable, human-curated record of *what we ran, what we got, and what we now
know* — keyed to the **parameters/architecture** of each run, including things wandb does
**not** track (architecture variant, loss family, config knobs, qualitative read). This is our
sanity record; wandb holds the curves, this holds the *conclusions*.

**Living doc.** Append rows; never rewrite history. git-tracked via `git add -f` (reports/ is gitignored).

> ⚠️ **GAP: 2026-06-20 → 2026-08-17.** This ledger was not maintained for two months. Everything in
> that window — the ZINB epic (#167), the bloom epic (#193), the composition axis (#183), the v2
> scoreboard, the rollout-ruler artifact verdict (#263), the state-freeze probe (#277) and the
> feedback-realism probes (#278) — was recorded **only in per-dossier logs**. Each dossier reasons
> locally and correctly; nothing forced a claim to survive contact with the others, and that is
> exactly where narrative drift lives. The §Claims Ledger below is back-filled for the
> **rollout-collapse programme only**. The rest of the window is **NOT back-filled** — treat its
> conclusions as living in their dossiers until someone does the work, and do not read silence here
> as agreement.

---

## Claims Ledger — the rollout collapse (#258 / #262)

**Why this section exists.** The run ledger below is per-*run* and the narrative is chronological, and
chronological prose is what drifts: five mechanism stories were retracted on 2026-08-16/17 alone. This
table is per-*claim* and answers one question — **what do we believe right now, and how much weight can
it carry?**

**The distinction that does the work:**

- **MEASUREMENT** — a number a run produced. Overturned only by a bug in the harness or a re-run.
- **INFERENCE** — a story built on measurements. Overturned by thinking, and repeatedly has been.

Every claim below carries its scope. **The entire rollout-collapse programme to date is single-seed,
single-vehicle**, which is *below this project's own evidentiary bar* (≥3 seeds on validation; the v2
scoreboard ran 3 seeds × 300 lessons). Nothing here is a verdict. The dossiers say INDICATIVE and then
get reasoned from as though they did not — this table exists to make that impossible.

**Shared scope unless a row says otherwise:** seed 42 · 40 lessons · `truncated_smoke` ·
artifact `calibration_model_20260814_003058.pt` · 13 origins · 4 posterior samples · 35 steps ·
target `sb` where a single target is named · calibration partition.

### LIVE — measurements

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M1 | **On occurrence (AP), persistence beats every arm from h6 on.** h6 0.112 / h18 0.108 / h36 0.083 vs the best held arm 0.087 / 0.091 / 0.069 and free-running 0.028 / 0.007 / 0.008. At h1 the model wins (0.298 vs 0.146). **On `crps_all` the arms beat persistence at every horizon (CRPSS +0.20/+0.45/+0.41/+0.11) — and that win is an ARTIFACT at h6/18/36** by the audited rule; see §The persistence re-reference. | state-freeze EXP-03 | **High** — a baseline, not an inter-arm comparison. Metric-qualified: an unqualified version of this row was wrong. |
| M2 | **Gate AP collapses steeply then saturates:** 0.298 (h1) → 0.028 (h6) → 0.007 (h18), flat thereafter. ~5 steps hold most of the damage. | realism EXP-01 | **High** — large effect, reproduced across every arm's control. |
| M3 | **Scrambling only the LOCATIONS of a perfect field reproduces the collapse:** AP 0.3008 → 0.0097, against free-running 0.0070 — with active count and magnitudes held identical. | realism EXP-03 | **High** — 31× effect; direct manipulation. Confounded with geographic grounding (C-291). |
| M4 | **Sparsity alone is survivable.** At matched horizon, `thin:0.75` fires at a *similar* rate to the collapse and scores far better — h18: AP 0.2244 vs 0.0070 (**32×**) at `act_ratio` 0.332 vs 0.291; h36: AP 0.1898 vs 0.0083 (**23×**) at 0.317 vs 0.266. | realism EXP-03 | **High** — large, direct, holds at both horizons. ⚠️ EXP-03 quoted "0.33 vs 0.27", which pairs `thin` at **h18** with `identity` at **h36** — a cross-horizon conflation. The conclusion survives at matched horizons; the quoted pair did not exist in any single cell. |
| M5 | **Clustering spanning 100× moves AP not at all.** Fed clustering 0.011 → 1.064 (brackets the real 0.449); AP flat at ~0.007. | realism EXP-05 | **Medium-high** — a null over a wide dose range, but arms were **not byte-paired** (C-296), so read at one significant figure. |
| M6 | **The recurrence does not smear the gate.** Oracle holds Moran's I flat over 35 steps (sb 0.507 → 0.494 → 0.516); free-running over identical steps falls 0.409 → 0.192 → 0.178. | realism EXP-04 data, analysed 2026-08-17 | **High** — same architecture, same kernel, same step count; only the fed content differs. |
| M7 | **The gate's ranking stays structured while the draw does not.** At equal expected count, top-K clustering vs independent-draw clustering: 4.4× (step 1) → 15.5× (step 6) → 26.8× (step 12). | realism EXP-04 | **High** — same gate, same count, two draw rules. |
| M8 | **Freezing recurrent state partially recovers AP:** h18 0.0070 → 0.0912, h36 0.0083 → 0.0693. Ordering `all` ≥ `cell` > `hidden` > `none`. | state-freeze EXP-02 | **Medium** — real and pre-registered, but see I-D and C-292 for what it does *not* license. |
| M9 | **`crps_all` is blind to all of it.** Four arms score 0.1353 / 0.1352 / 0.1350 / 0.1346 at h18 while gate AP spans **13×**. | state-freeze EXP-02 | **High** — corroborates Epic #263 independently. |

### LIVE — inferences

| # | Claim | Rests on | Confidence |
|---|-------|----------|------------|
| I-A | **Occurrence carries ~89% of the damage, magnitude ~8%.** | M3, M4, E4 splice arms | **Low-medium.** The *ordering* (occurrence ≫ magnitude) is robust — the effects are 30×. The **89/8 split is a single-seed decomposition and should not be quoted as a number.** |
| I-B | **Clustering is a proxy for correct placement, not an independently sufficient property.** Right places + no clustering → collapse (M3); wrong places + right clustering → no recovery (M5). | M3 + M5 | **Medium-high** — the two arms bracket the claim from opposite ends. |
| I-C | **Coordinate channels never helped because they act on marginals while the failure is joint.** | M3, M5, C-152's 3-seed negative | **Medium** — a mechanism fitted to an already-established null. It explains, it does not predict. |
| I-D | **Some of the gap flows through the recurrent state (~23% of the oracle gap).** | M8 | **Low.** INDICATIVE. Recovers 23% *relative to a collapsed control* and still does not reach persistence (M1), so it is not a skill claim. **Which memory half is NOT established** (C-292). |
| I-E | **The independent Bernoulli draw discards usable ranking information.** | M7 | **Medium** — the information gap is measured (M7). That fixing it would help is **untested**, and M5 is a warning that a plausible fix can do nothing. |

### REPLICATED on a vehicle with skill — 2026-08-17

`reports/2026-08-17_vehicle_replication_dossier/`. Six arms on **`violet_visitor`** (160 lessons, `nb`,
REAL against climatology through h18 by the audited verdict), all falsifiers pass, GREEN. This directly
tests whether M3/M4/I-A above were artifacts of an undertrained vehicle. **They were not.**

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M10 | **The oracle does not degrade at all.** Fed the real field, gate AP holds 0.4745 → 0.4793 (h18) → 0.4577 (h36) over 36 steps. All free-running decay is attributable to fed-back content, not to the recurrence or the horizon. | replication EXP-02 | **High** — direct, large, and corroborates M6 on the headline metric. |
| M11 | **Occurrence is ~95% of the gap; magnitude ~0%.** E4a (real occurrence × the model's own **71%-inflated** magnitudes) recovers 95.3% at h18; E4b recovers 1.4% and is **negative at 4 of 6 horizons**. On smoke: 88.6 / 7.9. | replication EXP-02 | **Medium-high** — replicates across two vehicles, near-additive (86–99%). Still one seed each. |
| M12 | **Wrong placement is worse than the model's own errors.** `spatial_scramble` scores 0.0486 at h18 against a control of 0.2569 — 5× worse — while `thin:0.75` discards ¾ of true events and recovers 95.5%. | replication EXP-02 | **High** — 5× effect, direct manipulation, F6 confirms the transform bit. |
| M13 | **The Epic #263 board reproduces bit-for-bit** from preserved cubes: worst \|ΔAP\| = 0.00e+00 over 7 horizons, `N` identical. | replication EXP-00 | **High** — exact. |
| M14 | **The post-2026-08-12 inference commits are a no-op on this vehicle's free-running path** (incl. `a2eabeb` per-site LockedDropout): `identity` today vs cubes from 08-12, worst \|ΔAP\| = 0.00e+00. | replication EXP-02 | **High** — exact, on 7 horizons. |

**Two corrections this forces on earlier work:**

* **M3/I-A were measured under a floor effect.** On `truncated_smoke` the control was already at 0.0070,
  so `spatial_scramble`'s "+0.9% of the gap" was the distance between two numbers both pinned near zero —
  never a measurement of placement's importance. The smoke run **understated** it.
* **The share statistic `(arm − control)/(oracle − control)` does not apply to arms outside that
  interval.** `spatial_scramble` falls *below* the control on both vehicles, so its share is negative and
  meaningless as a fraction. Any decomposition must check the interval before quoting a share.

**Re-scoping, per the pre-committed decision rule:** I-A moves from "single-seed decomposition, do not
quote the number" to **replicated across two vehicles with the ordering robust and magnitude's share
falling to ~0**. It remains one seed per vehicle, so under the standing rule this is an **escalation
trigger** — second seed, third vehicle — not a conclusion.

---

### RETIRED — kept so they stay dead

| Claim | Retired | Why |
|-------|---------|-----|
| "It is the **cell state** — `cell` carries 89% of the freeze effect" | 2026-08-16 | Architecturally predetermined: `hs = o ⊙ tanh(hl)`, so hidden is a *readout* of cell and freezing cell constrains hidden by construction. No arm in the set separates them (C-292). |
| "The state result quantifies the mediator — **CONFIRMED AND QUANTIFIED**" | 2026-08-16 | Overclaimed against its own pre-registration, and no naive baseline existed. Persistence then beat every arm (M1). Downgraded to I-D. |
| "**The recurrence diffuses the gate**" (C-295's hypothesis) | 2026-08-17 | Falsified by M6 using data already on disk. Also backwards on its own terms: smoothing *raises* spatial autocorrelation — it is how `correlated_bernoulli` manufactures clustering from white noise — so diffusion predicts Moran's I going **up**, and it goes **down**. |
| "The target is **the spatial coherence** of the occurrence field" (EXP-03's framing) | 2026-08-16 | Too loose; corrected by M5. Superseded by I-B. |
| "Moran's I falls **0.50 → 0.16** by step 6" | 2026-08-17 | Not a trajectory. Paired the **oracle's** 0.507 with the **free-running** 0.16–0.19 as though one run produced both. Real free-running trajectory is 0.409 → 0.192 (M6). |
| "gated_NB beats climatology at h36" | 2026-08-15 | ARTIFACT 4/4 — zero-driven; 74.6% of the gap is true zeros, ΔAP −0.030, `size_ratio` 0.0 (Epic #263). |

### The persistence re-reference, 2026-08-17 — and a correction to this table

The first version of this section said, flatly, *"no arm anywhere beats persistence at any horizon ≥ 6."*
**That was unqualified and therefore wrong** — the sentence this table exists to prevent, written into the
table itself on day one. On `crps_all`, the **FAO-02 primary metric**, every arm beats persistence at every
horizon: CRPSS **+0.20 / +0.45 / +0.41 / +0.11** at h1/6/18/36.

So the honest question is not "does it beat persistence" but "**is the win real**" — and this repo already
has an audited, pre-registered rule for exactly that (`scripts/rollout_ruler_core.py`, Epic #263, 86 tests).
Applied to the best arm against persistence on `sb`:

| h | CRPSS | ΔAP | zero-share of the gap | verdict |
|--:|--:|--:|--:|---|
| 1 | +0.197 | **+0.1519** | 69.9% | **UNDECIDABLE** — no CI computed |
| 6 | +0.446 | −0.0257 | 79.9% | **ARTIFACT** |
| 18 | +0.409 | −0.0165 | 82.4% | **ARTIFACT** |
| 36 | +0.111 | −0.0141 | 73.4% | **ARTIFACT** |

**The model's CRPS win over persistence is an artifact at every horizon from h6** — the same verdict Epic
#263 reached for gated_NB against climatology, now reached against the naive baseline as well. The win is
bought by predicting near-zero on the 99% of cells that are zero, while occurrence gets *worse*.

**h1 is the only non-artifact signal in the entire programme.** It reads UNDECIDABLE solely because
`ci_excludes_zero=False` was passed — no bootstrap CI exists — while both point estimates say REAL
(CRPSS +0.197, ΔAP +0.152). It must not be read as a negative.

**Method caveat, stated rather than buried.** Persistence is scored as a **1-sample** forecast
(`_persistence_gathered` → `np.array([last])`) with `gate_source = frac(samples>0)`, i.e. a **binary**
ranking, while the arms use the continuous `gate-head`. That violates Epic #263's own method rule (*the
reference's S must equal the arms' cube width*). **Both biases run the same way as the conclusion:**

- On **AP**, persistence is *handicapped* — a binary ranking cannot order within its active set — and it
  still beats the model 15× at h18. Correcting would widen the gap.
- On **CRPS**, the arms' `2/(m*m)` estimator is biased *upward* at S>1, so the model's CRPS win is
  *understated*. Correcting would make the win bigger — and the ARTIFACT verdict does not depend on the
  win's size, only on ΔAP and zero-share.

The conclusion is therefore robust to the S mismatch. A matched-S re-score would sharpen the numbers, not
overturn them.

### What is NOT established

- **On occurrence (AP), no arm beats persistence at any horizon ≥ 6** — free-running reaches 7–25% of it, the best frozen arm 77–85%. Every "improvement" reported to date is measured against a collapsed control, not against this.
- **On `crps_all` the arms do beat persistence, and that win is an ARTIFACT** at h6/18/36 by the audited rule.
- **No result here is multi-seed or multi-vehicle.** Positive findings at n=1 have historically evaporated on proper runs; the same standard applies to everything above.
- **Nothing has been shown to fix the collapse.** Two inference-time interventions have been tried (copula M5, state freeze M8); neither reaches persistence.

### Verification pass, 2026-08-17

Every headline number above was recomputed from the committed CSVs rather than copied from the dossier
prose. Results:

| figure | verdict |
|---|---|
| persistence vs arms, all horizons (M1) | ✅ exact |
| freeze-arm AP ladder (M8) | ✅ exact |
| `crps_all` blindness, 4 arms at h18 (M9) | ✅ exact |
| oracle / scramble / free-running AP (M3) | ✅ exact |
| copula sweep AP flatness (M5) | ✅ exact |
| E4 decomposition **89% / 43% / 8%** | ✅ 88.6% / 42.6% / 7.9% |
| Moran's I "0.50 → 0.16" | ❌ **two arms quoted as one trajectory** — retired |
| `thin` activation "0.33 vs 0.27" | ❌ **two horizons quoted as one comparison** — corrected |
| "25× vs 2.6×" | ⚠️ true but the **peak**, not the range (4.4×→27×→14×) — restated as a range |

Three of nine headline figures were wrong or misleading, and **all three failed the same way**: values
pulled from different cells of the results grid and presented as a single comparison. None of the three
changed a conclusion — the effects are large enough to survive — but each was quotable, and two had already
propagated into the risk register. Registered as **C-298**.

One near-miss worth recording: "occurrence merely being **plausible** — 43%" looked wrong against
`spatial_scramble` (0.9%) until traced; "plausible" means `wrong_month:-60`, a *real* field from the wrong
month, and the figure is correct. **A verification pass produces false positives too** — the fix is to
trace the number to its arm, not to "correct" it on suspicion.

### The only work that is not measured against a collapsed control

Follows directly from the table above.

**1. Establish whether the h1 win is real — the single highest-value open question.** It is the only
non-artifact signal in the programme and it is currently UNDECIDABLE for a purely procedural reason: no
bootstrap CI was computed. Both point estimates say REAL. Needs a re-run (score-then-delete removed the
cubes) emitting **per-origin** scores so an origin-bootstrap CI can be formed, at matched S. Cheap,
decisive, and it either establishes the project's first genuine win over a naive baseline or removes the
last one.

**2. Stop treating the long-horizon rollout as having demonstrated value.** h6–h36 are ARTIFACT against
persistence. Any further rollout work should be justified as *research toward a future capability*, not as
improving something that currently works. This is a framing change, not an experiment.

**3. Make the collapsed control unusable as a reference.** Persistence and climatology should be emitted by
the scorer on every run, and the artifact verdict applied automatically, so no future dossier can report a
win against `none` without the naive comparison sitting beside it. Structural, cheap, and it prevents the
exact error this ledger was created to catch.

**Deferred until the above:** the top-K feedback arm, and every other inference-time intervention. They
compete to improve a rollout whose value over persistence is currently negative on occurrence and artifactual
on CRPS. Worth doing *after* there is a reference worth beating — not before.

### Standing rule adopted 2026-08-17

Cheap single-seed probes here have been **reliable at killing hypotheses and unreliable at establishing
them** — every retirement above has held, while the one positive claim (I-A's 89%) is the least
trustworthy line in the table. So:

1. Design cheap experiments to **eliminate**, and read a null as the informative outcome.
2. A **positive** result from a single-seed probe triggers **escalation** (≥3 seeds, second vehicle),
   never a conclusion.
3. A claim enters this table as MEASUREMENT or INFERENCE **before** it is argued from.

---

## Evaluation & selection criteria — adopted from FAO Pre-Release Note 05 (Topics C & D)

Source: `~/brain/2_projects/fao02/_dev_materials/prerelease_notes/fao_02_pre_release_note_05/`
(Topic C = model validity/selection; Topic D = ensemble construction). **We use these going forward.**

**Metrics** (cell–month level, temporal backtest protocol):
| Metric | Role | Better |
|--------|------|--------|
| **CRPS** | **primary ranking metric** | lower |
| **QS99** (99th-pct quantile score) | guardrail — tail sanity (catches *timid* models) | lower |
| **Brier** | guardrail — onset/hurdle probability calibration (`y>0`) | lower |
| **MCR** (mean pred ÷ mean obs) | guardrail — magnitude calibration | **closest to 1** |
| *Bounded?* (ours, C-113) | sanity gate — does the 36-step rollout stay in range or `expm1`-explode? | bounded |

**Eligibility (strict conjunction — Topic C.5):** a model is **Eligible** iff
1. **Superiority:** CRPS ≥ **5%** better than the baseline (C.2), *and*
2. **All guardrails non-inferior:** QS99 & Brier within **1%** of baseline; MCR calibration-distance `|MCR−1|` at least 1% closer-to-1 than baseline's (C.3/C.4).

Fail either → **Ineligible** (regardless of ranking). Ranking (CRPS order) ≠ admissibility (guardrails).

**Ensemble (Topic D):** eligible models only → **equal-weight mixture** of predictive samples →
**greedy forward selection** (seed = lowest-CRPS eligible; add the model giving the greatest
*strict* ensemble-CRPS reduction — no 5% margin intra-ensemble; stop when none strictly improves).

---

## Baseline reference

> The baseline that superiority/guardrails are measured against. (FAO uses an empirical heuristic
> baseline, Topic B; for the C-113 program our working reference is the SS-off clean run below until
> a formal baseline is set.) **TBD — set once the first clean eval lands.**

---

## Run ledger

Config fingerprint columns capture the *variation axes* (what we change between runs).
Metrics filled from `--evaluate`; *Bounded?* from the 36-step rollout / `diagnose_io_gain`.

| # | Date | Model · arch | Key params (variation axes) | wandb run | CRPS | QS99 | Brier | MCR | Bounded? | Eligibility | Notes (incl. non-wandb) |
|---|------|--------------|------------------------------|-----------|------|------|-------|-----|----------|-------------|--------------------------|
| R1 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | loss=tobit; **ss_epsilon_max=0.0 (SS OFF, pure TF)**; balancer=active(unfrozen); seeds 42/42; dropout 0.15; sigma{1.0,0.75,0.5}; onset_bias −7.0; log1p; total_lessons=80 | train `serene-plant-49` · eval `ro537u46` | os 0.04 ✓ / sb 0.7 ✓ / **ns 5.9e8 💥** | logged ✓ | logged ✓ | os 0.13 / sb 64⚠ / **ns 1.1e11 💥** | **NO — `lr_ns_best` explodes** | **Baseline (PATHOLOGICAL zero-point)** | C-113 reproduces on the clean SS-off baseline → explosion is NOT caused by scheduled sampling. **Head-specific**: `os` healthy, `sb` bounded but over-predicts ~64×, `ns` explodes (worst in posterior-sample-mean; robust CRPS for ns ~55–129 already ~1000× `os`). Active balancer + log1p. All 4 PRN-05 metrics emit (CRPS/QS99/Brier/MCR). This is the reference every fix is measured against. |
| R2 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | **sweep trial: dropout=0.10, lr=0.0005**; else = R1 baseline (tobit, SS off, sigma{1,.75,.5}, seeds 42, 80 lessons, active balancer, log1p) | sweep `ih3fc9u9` | os 0.04 ✓ / **sb 1.4e8 💥** / ns 50 ⚠ | logged ✓ | logged ✓ | os 2.0 / **sb 2.3e10 💥** / ns 3.0e4 ⚠ | **NO — sb explodes** | Diagnostic (sweep 1/9) | Explosion persists at lower dropout+lr. **Worst head SHIFTED to `lr_sb_best`** (R1 was `ns`) → runaway is not tied to one head; both sb & ns unstable, os consistently healthy. |
| R3 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | **sweep trial: dropout=0.10, lr=0.001**; else = R1 baseline | sweep `esa59shz` | os 0.04 ✓ / sb 0.16 ✓ / **ns 8.2e4 💥** | logged ✓ | logged ✓ | os 0.43 / sb 0.09 ✓ / **ns 3.8e7 💥** | **NO — ns explodes** | Diagnostic (sweep 2/9) | Explodes on `ns` (sb bounded & well-calibrated here). max\|metric\|≈1.8e9 — **less extreme than R1/R2 (~1e11)**. Pattern holding: *something* always explodes (sb or ns), magnitude varies, `os` always healthy. |
| R4 | 2026-06-20 | violet · HydraBNUNet06_LSTM4 | **post-#115 clean no-coords baseline** (refactored channel-role code); loss=**hurdle_nb** (bounded); n_posterior=8; static_channels=[] (no-coords floor); dropout 0.15; seeds 42; saved data (`-sa`), no `-re` | offline `20260620_162128-hhegi8bw` | sb 46.9 / ns 49.1 / os 49.7 (time-series CRPS) | sb 0.88 / ns 0.62 / os 0.63 (QS) | by_sb 0.69 / by_ns 0.71 / by_os 0.69 | sb 126 / ns 1552 / os 1077 | **YES — all finite, preds max ~120, NO explosion** | **Baseline (clean bounded reference for the coord experiment)** | First run on the post-#115 refactored code; all three census fixes hold in a real run (sidecar persists `static_channels: []` ✓). Hurdle-NB bounds the C-113 explosion (contrast R1–R3 ~1e11). Remaining pathology is **over-firing/miscalibration** (MCR ≫ 1 on ns/os: 1552/1077) — exactly the spatial signal CoordConv (#110) targets. This R4 is the no-coords anchor the coords smoke/comparison runs are measured against. |
| R5 | 2026-06-20 | violet · HydraBNUNet06_LSTM4 | **#110 CoordConv smoke** — R4 config + `static_channels=[row_coord,col_coord]`, `input_channels=5` (ADR-060/061 top-skip). One variable vs R4: coords on/off. Same total_lessons=40, n_posterior=8, dropout 0.15, seed 42, `-sa`, no `-re` | offline `20260620_164252-evzwoql4` | sb 49.5 / ns **79.4** / os 55.0 (CRPS — **all worse vs R4**) | sb 0.93 / ns 0.98 / os 0.71 (QS — worse) | by_sb 0.86 / by_ns 0.88 / by_os 0.84 (worse) | sb 134 / ns **2498** / os 1199 (worse) | YES — finite, bounded (ns pred mean 49→85) | **Diagnostic (first coord comparison — NOT a verdict; n=1/arm)** | **Engineering smoke PASS:** coords train+eval+reload with no crash → all three census fixes (C-157/158/159) hold with coords genuinely on. **Scientific result: coords HURT** — 12/12 metrics regressed vs R4, ns CRPS +62% (49→79), ns MCR 1552→2498, ns pred-mean 49→85 (MORE over-firing, opposite of hypothesis). Unanimous + large-magnitude, but **single seed/arm — no CI yet**; needs the #110 within-run-bootstrap decision rule (C-162) or multi-seed before a binding verdict. Prior now clearly negative for CoordConv-as-configured. |

**Status legend:** `Eligible` · `Ineligible` · `Baseline` (the reference) · `Diagnostic` (not a candidate, e.g. a probe) · `Failed` (run errored/exploded).

---

## What we know / wins (running narrative)

- **2026-06-07 — wandb training logging restored (C-132).** Single-run `-t` training now opens a
  `job_type="train"` run and logs per-lesson curves (was silently dropped by a stale phase-template
  override). Fix = delete the override; pinning test + fail-loud guard added. → all training results
  from here are observable. See `reports/2026-06-07_wandb_falsification/`.
- **2026-06-07 — clean baseline established (R1).** violet retrained with scheduled sampling OFF
  (`ss_epsilon_max=0.0`) = pure teacher forcing — the unfixed-model zero-point for C-113. Training
  healthy (loss↓).
- **2026-06-07 — C-113 REPRODUCED on the clean baseline (R1 eval).** Explosion is **head-specific**:
  `lr_ns_best` explodes (CRPS_mean ~6e8, MCR ~1e11), `lr_sb_best` bounded but over-predicts ~64×,
  `lr_os_best` healthy (CRPS ~0.04, MCR ~0.13). **Key inference:** the runaway is NOT caused by
  scheduled sampling (it was off) — it persists with pure teacher forcing + active balancer + log1p.
  The explosion concentrates in the **posterior-sample-mean** (a few runaway draws dominate), and is
  worst on the **non-state head**. Leads to chase: why `ns` specifically; the active-balancer (C-111)
  interaction; sample-mean vs robust-aggregate divergence. All 4 PRN-05 metrics confirmed emitted.

- **2026-06-08 ~00:40 — sweep progress + a data-availability flag.** 4/9 trials trained. R2 (0.1/0.0005, sb-explode) & R3 (0.1/0.001, ns-explode) confirmed via each run's *own* config — mappings correct. **BUT trials 3 (0.1/0.002) & 4 (0.15/0.0005) wrote train-only wandb summaries (13 keys, NO CRPS/MCR/eval)** — unlike trials 1–2 (88-key with eval). So their explosion status is **unreadable from wandb so far**. Open question for the morning: *does the sweep reliably evaluate every trial, or only some?* If sweep trials are train-only, the sweep tells us about training stability but NOT the rollout explosion (which is an eval-phase phenomenon) — meaning a sweep may be the wrong tool for the explosion question, and we'd need per-config `-e` runs. Will recheck at completion (eval may flush late) and do a careful full pass then. **Confirmed pattern from R1–R3 (the trials that DID eval): something always explodes (sb or ns), `os` always healthy.**

- **2026-06-08 ~01:09 — overnight sweep OOM-KILLED at trial ~5/9 (NOT restarted — would re-die).**
  Kernel log: `Out of memory: Killed process 2097902 (python) total-vm:33.7GB anon-rss:13.2GB global_oom`.
  **Mechanism — CORRECTED 2026-06-08:** my first claim ("RAM accumulates ~2.6 GB/trial across the
  sweep") is **RETRACTED — it's wrong.** Counter-evidence: `pink_pirate` (a *healthy* model) ran dozens
  of sweep trials over hours without OOM, so sweep trials DO free memory between them; a sweep does not
  use more RAM than the single run it's made of. The real distinguishing factor: **this model EXPLODES
  (C-113)**, and it died **mid-eval** (posterior sampling) where preds hit ~1e11 (`expm1`→inf). Leading
  (UNVERIFIED) hypothesis: the explosion balloons a single trial's eval memory past the limit — an OOM
  that is a **symptom of C-113**, not a sweep property. (R1, a single exploding run, survived its
  explosion, so it's not deterministic per trial — depends on blow-up severity.) NOT a CUDA wedge, NOT
  an in-process crash (external SIGKILL) — a true RAM OOM, cause not yet measured.
  **Salvaged (clean evals): R1, R2, R3 only** — trials 4–5 trained but their eval didn't complete/flush
  before the kill (they evaluate *after* training; sweep trials DO eval — earlier "train-only" was just
  mid-eval). **Sweep produced NO new eval rows beyond R1–R3.**
  **DECISION: not auto-restarted** — a fresh `-s` re-runs 1→9 and re-OOMs at ~trial 5 (infinite re-fail
  loop unattended). → registered as a risk (C-135). **Morning options:** (a) run trials individually
  (`-t -e` per config) so RAM frees between trials; (b) fix the cross-trial RAM accumulation (cf. ADR-047
  streaming `del`+`gc.collect`); (c) lower `n_posterior_samples`; (d) cap the sweep grid to ≤3–4 combos
  per process. The OOM is itself a finding: **the explosion isn't just a metric artifact — it inflates
  eval memory enough to kill the process.**

**Confirmed findings (3 clean evals, R1–R3): C-113 reproduces, SS-independent; the exploding head varies
(ns/sb), `os` always healthy; magnitude ~1e8–1e11; and the explosion is severe enough to OOM eval.**

- **2026-06-08 (pm) — collapse baseline + sweep-crash correction + magnitude-calibration dossier opened.**
  Today's **40-lesson** calibration run (`calibration_model_20260608_165326.pt`, eval `ffldgbxf`)
  **COLLAPSED**: MCR ≈ 0.002–0.03 (predicts ~0 everywhere); CRPS small (sb 0.13 / ns 0.05 / os 0.04) **only
  because ~95%+ cells are zero** — CRPS rewards the collapse. This is the **opposite** mode to R1–R3
  (80-lesson, explode) → framed as **two distinct failure modes**: collapse = zero-inflation reward;
  explosion = no rollout training. Verified (git): **no probability-coupled hurdle was ever implemented**
  (only C-45 ground-truth masking `aba45bc` + Tobit censoring `56194d2`; neither couples magnitude to
  predicted probability). New program → `reports/2026-06-08_magnitude_calibration_dossier/` (candidate #1 =
  minimal hurdle; judge on twCRPS+Coverage, MCR diagnostic only).
  **Sweep-crash correction (distinct from the ~01:09 RAM OOM above):** the 2026-06-08 *afternoon* sweep
  failures (`ikvegy30`/`3vidg2tr`/`4atlhfw0`/`x9zk3ujt`/`ksubmpw3`, ~14:3x) were a **CUDA `unspecified launch
  failure` — kernel `Xid 62` → `Xid 45`** at `model.to(device)` (`make()`), i.e. a **GPU-context fault**
  (one long-lived sweep process; trial 1 ran, then the context was poisoned — likely by a concurrent
  memory-intensive GPU job — so every later trial died identically). **Not** the RAM OOM logged at ~01:09;
  two different mechanisms. Fixed by `rmmod/modprobe nvidia_uvm`; a fresh single run trained + evaluated
  cleanly afterward. (This corrects/extends the ~01:09 entry: there were *two* distinct sweep-failure
  events on 2026-06-08.)

- **2026-06-09 — Arm-1 (hurdle, lognormal_nll, 40 lessons): magnitude UN-COLLAPSED one-step (directional); the rollout exploded.** *(Originally filed "FAIL → ZITD"; reframed 2026-06-09 — see the ⟳ note below.)*
  One variable vs the 40-lesson Tobit baseline (`loss_reg` tobit→lognormal_nll, sigma dict→0.9 scalar, +`hurdle_threshold=0`).
  Training completed HEALTHY; **eval FAILED — `views-evaluation` rejected the predictions ("Input contains infinity").**
  Verified from saved preds (`predictions_calibration_20260609_051916`, origin_0): **NOT a collapse — an autoregressive
  runaway.** Step-1 magnitudes are non-zero (sb 61 / ns 580 / os 91 — the hurdle *un-collapsed* the head vs baseline ~0.02),
  then the 36-step free-running rollout grows exponentially per step → expm1 → **INF by step ~13–15**. **Key finding:
  un-collapsing the magnitude head directly TRIGGERS the C-113 explosion — magnitude calibration and rollout stability are
  coupled; the head can't be fixed in isolation.** Per the binding stopping rule (dossier `2026-06-08_magnitude_calibration`)
  → **commit to ZITD/structural, no more loss tweaks.** ZITD is doubly motivated: its sub-exponential softplus link makes
  drift linear, dissolving exactly this expm1 explosion. Artifact `calibration_model_20260609_051916`.
  **⟳ Reframe 2026-06-09 (step-1 read; C-136 / M-R1 / M-R2):** the "FAIL → ZITD" verdict conflated two axes. A
  teacher-forced **step-1 `MCR_pos`** (rollout not engaged) shows the magnitude head moved **off ~0**: sb 0.11→**0.19**,
  ns 0.02→**1.29**, os 0.03→**0.73** vs Tobit. The explosion is purely the **untrained rollout** (Axis B). ⚠ **Direction,
  not values** — MCR is a diagnostic ratio (not a proper score); single-draw, 1 origin/seed, small n (131/59/50) → "un-collapsed"
  holds, "calibrated" does not. Proper-subset score + CI + 2nd seed = R4's readout (#93). **Revised next step:** hurdle +
  **rollout training** (cheap SS-middle probe → GTF), NOT a count-likelihood rebuild. Stopping rule intact (rollout ≠ a new loss).
  *(Numbers refined 2026-06-10 by the durable `scripts/mcr_readout.py` — see dossier `07`: the all-origins aggregate MCR is
  mean-dominated, the median positive cell still under-predicts, and sb barely moved. The step-1 figures above are the
  superseded origin-0 `/tmp` throwaway.)*

- **2026-06-10 — R4 (hurdle + scheduled sampling `ss_epsilon_max=0.5`, 40 lessons) → EXPLODED at eval (same as Arm-1).**
  The cheap rollout-training probe — SS-middle on the hurdle config, one variable vs Arm-1. Trained clean; eval failed
  "Input contains infinity". Durable readout (`scripts/mcr_readout.py`, `predictions_calibration_20260610_010843`):
  FULL-rollout MCR ≈ **3.4e33 / 3.1e33 / 7.5e33** (sb/ns/os) — indistinguishable from Arm-1; step-1 magnitude no better
  (sb 0.21→0.088). **Completes the scheduled-sampling bracket: 0.25→explode, 0.5→explode, 1.0→collapse — plain per-step
  SS is EXHAUSTED (proven, not assumed), even on the un-collapsed hurdle head.** → the real fix is **GTF / B1 rollout
  training (#78, cross-step gradients)** or the **count-likelihood head** (ZINB/hurdle-NB). Rollout dossier `07` EXP-02.

*(Append wins/lessons here as runs land — especially negatives and "looked-right-but-wasn't".)*
