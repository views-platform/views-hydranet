# Experiment log — vehicle replication

Append-only. Falsifier verdicts are recorded **before** any prediction is read.

---

## EXP-00 — F1, before any GPU time · 2026-08-17 02:00 · **PASS, EXACTLY**

The production cubes for `calibration_model_20260812_191742.pt` survived on disk. Quarantined to
`_quarantine_predictions_calibration_20260812_191742` (the `_quarantine_` prefix matters — a
`.REFERENCE` suffix would still match `glob("predictions_*")` and jam the driver's leftover guard
forever) and re-scored with the same scorer and the sha-pinned truth.

| h | 1 | 6 | 12 | 18 | 24 | 30 | 36 |
|---|--|--|--|--|--|--|--|
| worst \|ΔAP\| vs `rescore.csv` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Bit-for-bit on AP **and** `crps_all`, `N`=170430 throughout. **The shipped Epic #263 board is
reproducible from preserved artifacts** — worth recording independently of this experiment.

## EXP-01 — the smoke gate · 2026-08-17 02:40 · **PASS**

Two arms at 2 origins via `replication_smoke_entry.py` (the tested entry with `origins` truncated
through the `_EvaluationContext` NamedTuple `_replace` seam). Full chain proven in ~12 min against a
~2.6 h commitment: artifact loads → transform applies → cube written (exactly 2 origins) → scored →
deleted → directory clean. Fed-field rows exactly `2 × 4 × 35 × 3 = 840` for both arms.

The assertion that mattered: `active_fraction(thin:0.75) = 0.000875` against `0.25 × use_real =
0.000860`, **1.7% error** — the transform bit on the real vehicle, not merely on a fixture.

## EXP-02 — six arms · 2026-08-17 02:54–03:54 · **ALL FALSIFIERS PASS**

Ran in **59 minutes**, not the budgeted 2.6 h: `violet_visitor` is `nb`, and the ~26 min/arm figure
came from `truncated_smoke`'s `truncated_nb`, whose sampler is a known performance problem
(`c07a352`). Per-arm elapsed 362–620 s.

### Falsifiers, recorded first

| # | verdict | evidence |
|---|---|---|
| **F1** | PASS | worst \|ΔAP\| = 0.00e+00 across 7 horizons (EXP-00) |
| **F2** | PASS | `use_real` fed `active_fraction` 0.00357597, matching the real field; F6's relations are computed against it |
| **F3** | does not fire | oracle−control gap at h18 = **0.2224** ≫ 0.05 — there is a real gap to decompose |
| **F4** | PASS | h=1 AP = 0.474461375 for **all six arms**, worst \|ΔAP\| = 0.00e+00 |
| **F5** | PASS | `N` = 170430 in every scored row of every arm |
| **F6** | PASS | all five separation relations: af(scramble)≡af(use_real) 0.00e+00; clustering ratio 0.025; af(E4a)≡af(use_real) 0.00e+00; magnitudes differ 71.4%; af(thin)/af(use_real) within 1.0% of 0.25 |

### The control question, settled by measurement

AMENDMENT 1 replaced the preserved cubes with an `identity` arm run tonight, because three commits
touching the inference path landed after 2026-08-12 — notably `a2eabeb` (per-site LockedDropout,
independent MC-dropout masks) on a vehicle that evaluates with `dropout_rate: 0.15` and
`evaluation_mode: 'stochastic'`.

**`identity` (today) vs the preserved cubes (2026-08-12): worst \|ΔAP\| = 0.00e+00 across all seven
horizons.** Those commits are a **no-op on this vehicle's free-running path**. The amendment was still
correct — the equivalence could not be known without measuring it — and every share below is therefore
valid against either control.

### The result

Target `sb`, gate AP. Control = `identity` (≡ the preserved cubes).

| h | oracle (`use_real`) | control | gap | `thin:0.75` | E4a real-occ × model-mag | E4b model-occ × real-mag | `spatial_scramble` |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.4745 | 0.4745 | 0.000 | 0.4745 | 0.4745 | 0.4745 | 0.4745 |
| 6 | 0.4774 | 0.3924 | 0.085 | 0.4452 | 0.4652 | 0.3926 | 0.2110 |
| 12 | 0.4790 | 0.3226 | 0.156 | 0.4618 | 0.4674 | 0.3141 | 0.0945 |
| 18 | 0.4793 | 0.2569 | 0.222 | 0.4692 | 0.4689 | 0.2600 | 0.0486 |
| 24 | 0.4729 | 0.2060 | 0.267 | 0.4583 | 0.4704 | 0.2031 | 0.0300 |
| 30 | 0.4744 | 0.1699 | 0.304 | 0.4609 | 0.4630 | 0.1631 | 0.0230 |
| 36 | 0.4577 | 0.1370 | 0.321 | 0.4281 | 0.4626 | 0.1288 | 0.0188 |

**Share of the gap recovered:**

| h | occurrence (E4a) | magnitude (E4b) | `thin:0.75` | `spatial_scramble` |
|--:|--:|--:|--:|--:|
| 6 | 85.6% | 0.3% | 62.1% | **-213.2%** |
| 12 | 92.6% | -5.4% | 89.0% | **-145.8%** |
| 18 | 95.3% | 1.4% | 95.5% | **-93.7%** |
| 24 | 99.1% | -1.1% | 94.5% | **-66.0%** |
| 30 | 96.3% | -2.2% | 95.6% | **-48.3%** |
| 36 | 101.5% | -2.6% | 90.8% | **-36.9%** |

### Three findings

**1. The oracle does not degrade at all.** Fed the real field, gate AP is 0.4745 → 0.4793 → 0.4577
across 36 steps. Every point of decay in the free-running rollout is attributable to what is fed back,
not to the recurrence, the horizon or error accumulation in the architecture. This is the same
conclusion M6 reached on the gate's Moran's I, now on the headline metric and on a vehicle with skill.

**2. Occurrence is ~95% of the gap; magnitude is ~0%.** E4a hands the model the *correct occurrence*
and lets it keep its own magnitudes — which are **71% inflated** (26.5 vs the real 15.5) — and it
recovers 95.3% of the gap at h18. E4b is the mirror and recovers 1.4%, and is **negative at four of
six horizons**: giving the model true magnitudes while keeping its own occurrence makes it slightly
*worse*. On `truncated_smoke` the same split was 88.6 / 7.9. The decomposition is near-additive
(sum 86–99%).

**3. Wrong placement is worse than the model's own errors.** `spatial_scramble` — perfect marginals,
perfect magnitudes, permuted locations — scores **0.0486 at h18 against the control's 0.2569**. Not a
fraction of the gap but *below the control*, so its "share" is negative and the share statistic does
not apply to it. Meanwhile `thin:0.75` discards **three quarters of the true events** and still
recovers 95.5%. Throwing events away costs almost nothing; moving them costs everything.

### Predictions — read only after the above

| # | verdict | |
|---|---|---|
| **P1** | **CONFIRMED** | exact reproduction, 0.00e+00 |
| **P2** | **FALSIFIED** | the gap is *not* far smaller on a vehicle that has skill: 0.222 at h18 vs smoke's 0.294. The difference is relative, not absolute — smoke retains 2% of its oracle, violet retains 54% |
| **P3** | **FALSIFIED, and in the opposite direction** | `spatial_scramble` destroys *more*, not less. On smoke it sat at +0.9% of the gap; here it is −94%. The smoke measurement was **floor-limited** — its control was already at 0.0070, so scramble had no room to fall. The 0.9% was an artifact of the floor, not a measurement of placement's importance |
| **P4** | **FALSIFIED** | the occurrence/magnitude split differs by 6.7 points (95.3 vs 88.6), not the >10 predicted |

**Three of four predictions falsified, and the conclusion is the opposite of the hypothesis that
motivated the run.** The mechanism findings are **not** an artifact of an undertrained vehicle. They
replicate on the model with CI-backed skill and are *sharper* there: magnitude's contribution falls
from 7.9% to ~0%, and placement's importance rises from an apparent +0.9% to −94%.

### A methodological correction this forces

The share statistic `(arm − control) / (oracle − control)` assumes arms lie **between** control and
oracle. `spatial_scramble` does not, on either vehicle. Its smoke "share" of 0.9% was therefore never
a measure of placement's importance — it was the distance between two numbers both pinned near zero.
Any future decomposition must state whether an arm falls outside the interval before quoting a share.

### Scope

One seed (42), one vehicle for the treatment arms, one target (`sb`), 13 origins, S=16, calibration
partition. `truncated_smoke` differs from `violet_visitor` on **two** axes (40 vs 160 lessons *and*
`truncated_nb` vs `nb`) and this cannot separate them. Per the standing rule of 2026-08-17, a
replication is an **escalation trigger** — second seed, third vehicle — not a conclusion.
