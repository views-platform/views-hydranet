# 07 — Experiment log

Append-only. Every entry links to its pre-registration in `05_analysis_plan.md` and states the
verdict against the **pre-committed** falsifiers, including when none fired.

**Common method for M66–M70.** One origin (`origin_12` / dump `origin395`, forecast months
517–552), validation partition, target `sb`, all eight roster artifacts with the cell clamp on,
`S = 16` draws per member. Two instruments are used and they are **not interchangeable**:

| instrument | what it is | what it measures |
|---|---|---|
| **sampled** | mean/quantile of the composed draws in the cube | the shipped product, sampling noise included |
| **noiseless** | `gate × mu` (or `(gate ≥ τ) × mu`) from the body-mean dump | the model, with draw noise removed |

Placement is verified per target against the model's own gate (`corr = +1.0000`; wrong
orientation `+0.0186`), per C-322.

⚠️ **Single artifact per configuration — no seed replication.** C-119 puts training variance
around 20%, so differences between *adjacent* ranks below are inside the noise. The findings
called out as decisions (M68, M69, M70) are the ones whose effects are multiples of that.

---

## M66 — The pooled ensemble beats every member, and the reason is draws, not diversity

**Pre-registration:** `05_analysis_plan.md` Q3, falsifier **F2**.

Pooled 8 × 16 = 128 draws by equal-weight concat, matching `rusty_bucket`. AP on the composed
forecast, `sb`:

| | h1 | h18 | h36 |
|---|---|---|---|
| best single member | 0.3545 | 0.1804 | 0.1205 |
| **pooled (S=128)** | **0.4081** | **0.2203** | **0.1588** |

**F2 does NOT fire** — the pool beats its best member at h18 and at every other horizon tested.

**But the mechanism is sample count.** Subsampling the same 128-draw pool back to 16 draws
(20 random subsets) collapses it to member level, *below* the best member:

| h18 | AP |
|---|---|
| single member, mean of 8 (S=16) | 0.1293 |
| single member, best of 8 (S=16) | 0.1804 |
| **pool subsampled to S=16** | **0.1462** (sd 0.0175) |
| pool, all draws (S=128) | 0.2203 |

Monotone in S: k=2 → 0.1496, k=4 → 0.1860, k=8 → 0.2203.

**Reading.** The ensemble is currently buying variance reduction, not disagreement. The same
gain should be reachable by drawing more samples from one model — untested here, and the honest
test needs a re-emit at S=128 on a single artifact.

**Decision:** the ensemble stands; the *justification* for eight members does not rest on this
result. Feeds M67.

---

## M67 — The roster has almost no diversity

Gate-error correlation between members, h18, all 28 pairs: **0.96 – 0.99**.
Most redundant `blue_stranger`/`bright_starship` r=0.990; most complementary
`blazing_meteor`/`heavy_freighter` r=0.960 — still near-identical.

On the noiseless instrument with every model given a soft gate, AP@h18 spans
**0.2279 – 0.2856** — a 25% spread across eight architectures (nb and mixture_nb, three
compositions).

Leave-one-out on the pooled forecast (h18) moves AP by at most 0.0185, and `pink_pirate`'s
removal *improves* the pool by 0.0012.

**Reading.** The eight are close to interchangeable on occurrence. `heavy_freighter` is the one
genuine outlier in behaviour — 2–3× everyone's predicted mass (0.237 of truth at h18 vs ~0.08
for the rest; 0.403 at h36) and the lowest error correlation — so it earns its seat for being
different rather than for being good.

**Decision:** diversity is a real gap, not a solved property. The weakest and most redundant
members are the candidates to re-specify.

---

## M68 — `gate_threshold = 0.5` costs the three hard-gate models 2–4× their AP

Sweep over τ on each model's **own** gate and body, so nothing is retrained or re-emitted.
AP@h18, `sb`, noiseless:

| model | τ=0.0 | 0.05 | 0.10 | 0.20 | 0.50 | **soft** |
|---|---|---|---|---|---|---|
| `purple_alien` | 0.2720 | 0.2471 | 0.2244 | 0.1748 | 0.0993 | **0.2856** |
| `pink_pirate` | 0.2611 | 0.2299 | 0.2043 | 0.1635 | 0.0589 | **0.2845** |
| `bright_starship` | 0.2387 | 0.2145 | 0.1879 | 0.1519 | 0.0926 | **0.2752** |
| `blue_stranger` | 0.2497 | 0.2234 | 0.2012 | 0.1608 | 0.0954 | **0.2669** |
| `bold_comet` | 0.2375 | 0.2127 | 0.1808 | 0.1436 | 0.0675 | **0.2642** |
| `blazing_meteor` | 0.2264 | 0.1983 | 0.1802 | 0.1349 | 0.0969 | **0.2639** |
| `violet_visitor` | 0.2228 | 0.2035 | 0.1838 | 0.1509 | 0.0934 | **0.2588** |
| `heavy_freighter` | 0.1854 | 0.1767 | 0.1600 | 0.1318 | 0.0693 | **0.2279** |

**Monotone decline in τ for all eight, and the soft gate wins for all eight** — including
against τ=0, i.e. against no threshold at all. Multiplying by the gate carries ranking
information that *any* hard cut discards.

At τ=0.5, `bold_comet` emits on **23 of 13,110 cells** at h18.

What the threshold costs the three configured for it, at h18:
`bold_comet` 0.0675 → 0.2642 (**+291%**), `blazing_meteor` 0.0969 → 0.2639 (**+172%**),
`violet_visitor` 0.0934 → 0.2588 (**+177%**).

**Reading, and the limit of it.** The three hard-gate models rank last *as configured* and are
not weak models. But **what the threshold destroys is the ranking inside the FATALITY FIELD, not
the model's occurrence information** — and which of those matters depends on what is delivered.

This is the sharp edge, and M72 measures the other side of it: the scorer's AP is computed on the
**gate head** (`gate_source = 'gate-head'` on all 20 arms), and the gate is *identical* between a
soft and a threshold arm because the composition acts only on the emitted magnitude. On that
instrument soft and threshold differ by **≤0.009 AP at h18 and by exactly 0.0000 at h1**.

Both readings are correct about different products:

| if the delivered product is… | then the threshold… |
|---|---|
| the gate (`by_*`) **and** the fatality field, as two fields | costs no occurrence skill; the gate is untouched |
| a single fatality field | destroys its ranking — 99.8% of cells are exactly 0 and cannot be ordered |

**Decision:** τ=0.5 is not defensible **for a single-field delivery**, and is close to free if the
gate ships alongside. Since `rusty_bucket` pools the composed `lr_*` cubes, the pooled product is
the single-field case, so the roster should move off τ=0.5 — but the justification is delivery
shape, not model quality, and the earlier flat claim that "τ=0.5 is not defensible" was too broad.

This does *not* rescue the "threshold ≈ soft, statistically tied" reading `00_README` §2 flagged:
that reading was drawn from `crps_all`, and M72 shows the compositions differ by 1.3–4.3× on
`crps_none` and 1.3–3.4× on `act_ratio`. They are not tied; they are differently wrong.

---

## M69 — A multi-τ ensemble of one model cannot beat that model's soft gate

**Proposed by the chair:** seat the same model several times at different thresholds, for
diversity. Tested, and there is an analytic reason it cannot pay:
for τ ~ Uniform(0,1), **E_τ[(gate ≥ τ)] = gate exactly**, so averaging the hard composition over
evenly spaced τ *reconstructs the soft gate*.

Measured, h18, `sb`:

| model | 2 τ | 4 τ | 8 τ | 16 τ | 64 τ | its soft gate |
|---|---|---|---|---|---|---|
| `bold_comet` | 0.1293 | 0.1719 | 0.2272 | 0.2515 | 0.2602 | **0.2642** |
| `purple_alien` | 0.1608 | 0.2052 | 0.2417 | 0.2622 | 0.2809 | **0.2856** |

At 64 thresholds the mixed field is within **0.5–0.7%** of the soft field pointwise
(`max|mix − soft| / max(soft)`). It converges to the soft gate **from below and never crosses
it**.

**Reading.** A multi-τ roster is a lossy approximation of a composition already available as a
one-word config change. It adds seats without adding information.

**Decision:** do not spend roster slots on τ variation. Spend them on something the soft gate
does not already contain.

---

## M70 — For a point estimate, use the model's own expectation, not the average of its draws

**Context:** the next article wants point estimates. `output_type: 'point'` with
`aggregate_method` (ADR-021) offers exactly two collapses, `arithmetic_mean` and `median`.

AP@h18, `sb`, same cells and month for every column:

| collapse | what it is | five soft-gate models |
|---|---|---|
| `sample_median` | median of 16 draws | **0.038 – 0.065** |
| `sample_mean` | mean of 16 draws | 0.110 – 0.180 |
| **`analytic_mean`** | **`gate × mu`, exact** | **0.228 – 0.286** |

Gain from the analytic expectation over averaging draws: `pink_pirate` **+159%**,
`bright_starship` +87%, `blue_stranger` +85%, `heavy_freighter` +60%, `purple_alien` +58%.
Total predicted mass is unchanged (e.g. `purple_alien` 0.103 → 0.098 of truth), so this is
sharper ranking, not inflation.

**The median is a trap on this field.** It is non-zero on **0.05–0.14%** of 13,110 cells and
carries **0.3–1.5%** of true mass — an almost empty map. It is nonetheless the MAE-optimal point
estimate, which is the point: the optimal collapse is fixed by the loss, not chosen by taste.

⚠️ **Do not read the three hard-gate models in this comparison.** Their composition zeroes
99.8% of cells deterministically, so AP is dominated by tie structure rather than skill.

⚠️ `analytic_mean` is non-zero on **100%** of cells for soft-gate models (a dense body times a
non-zero gate). Better for ranking; a presentational decision for a paper.

**Decision:** if points are required, emit the expectation rather than collapsing samples. That
capability does not exist in `aggregate_method` today.

---

## M71 — The cell clamp transfers to roster models, and the effect is far larger than measured

**Pre-registration:** `05_analysis_plan.md` Q1, falsifier **F1**.
**Run:** 2026-09-07, before M66–M70. Two models emitted with the clamp on and off, both
compositions — the only four `_none` arms in the sweep.

AP, `sb`, clamp on minus clamp off:

| model | comp | h1 | h18 | h36 |
|---|---|---|---|---|
| `pink_pirate` | soft | **0.0000** | 0.2684 vs 0.0080 = **+0.2604** | 0.2017 vs 0.0093 = **+0.1924** |
| `pink_pirate` | threshold | **0.0000** | 0.2769 vs 0.0076 = **+0.2694** | 0.2033 vs 0.0093 = **+0.1940** |
| `violet_visitor` | soft | **0.0000** | 0.3512 vs 0.2569 = **+0.0943** | 0.2781 vs 0.1370 = **+0.1411** |
| `violet_visitor` | threshold | **0.0000** | 0.3488 vs 0.2571 = **+0.0917** | 0.2755 vs 0.1300 = **+0.1456** |

**F1 does NOT fire.** ΔAP@h36 is positive on both models and both compositions, so the clamp goes
on the roster and ADR-027 §2.1's production permission stands.

`crps_events` also improves at h18 and h36 on both models (e.g. `violet_visitor` soft 14.297 vs
14.823 at h18), so this is not AP bought at the cost of magnitude. The cost shows in `crps_none`,
which rises with the clamp (`pink_pirate` soft h18 0.0016 vs 0.0002) — the clamp keeps the model
firing, and some of that firing lands on quiet cells.

**The h1 delta is exactly 0.0000 on all four arms.** That is the pre-registered VOID condition
passing as a positive control: the clamp acts only from step 2, so h1 must be identical, and it is.

**Effect size vs prior.** ADR-027 §2.1 recorded **+0.0591 AP@h36** on four `fullzero_*` test
vehicles. On these production models it is **+0.14 to +0.19 at h36** — two to three times larger.
`pink_pirate` without the clamp is effectively dead from h18 (AP 0.008 against 0.268 clamped).

⚠️ **Two models, one origin, one seed each.** The direction is 4/4 across arms; the magnitude is
not established. The spread between the two models (+0.09 vs +0.26 at h18) is itself large and
unexplained.

---

## M72 — The composition changes how much is emitted and where it leaks, and nothing else measurable

**Pre-registration:** `05_analysis_plan.md` Q2.
**Run:** 2026-09-07. All 8 models × {soft_gate, threshold_gate} × clamp on = 16 arms.

`sb`, h18, clamp on, soft ÷ threshold:

| metric | soft / threshold | reading |
|---|---|---|
| `AP` | **1.00** (Δ ≤ 0.009; **exactly 0.0000 at h1**) | scored on the gate head — composition-blind |
| `crps_events` | **1.00** on all 8 | magnitude on event cells is unchanged |
| `n_false_pos` | **1.00** on all 8 | gate-driven, so identical |
| `act_ratio` | **1.28 – 3.42** | soft fires 1.3–3.4× more |
| `crps_none` | **1.12 – 4.30** | soft leaks 1.1–4.3× more |
| `size_ratio` | 0.0 vs 0.0 | saturated, reads nothing (known) |

**The composition is a firing-rate dial, not a skill lever.** It buys nothing on placement or on
event magnitude and costs leak in proportion to how much more it fires. Under M45 — *AP loss
scales with how much the model FIRES* — that is the shape of an intervention that should not help,
and on this instrument it does not.

`pink_pirate` is the outlier at 3.42× / 4.30×, against 1.3× for the rest. Unexplained.

**Q2 verdict:** the two compositions are **not tied and not close**, but they differ on
*emission*, not on skill. The decision between them is therefore a delivery-shape decision
(see M68), not a scoreboard decision — which is why the original `crps_all` reading found "no
difference" and why AP alone would have found the same.

---

## M73 — All eight pass the pre-deployment acceptance criteria; K3 is VOID

**Pre-registration:** `06_acceptance_criteria.md` (written 2026-09-07, **before** the runs).
**Run:** 2026-09-07, 300 lessons each, validation partition, cell clamp on, each model's own
configured composition.

⚠️ **Correction, 2026-09-08.** This entry first reported K1 using `act_ratio` computed on a
single origin. Neither was right: K1 is `recall_mass`, and the pre-registered instrument
`tools/k1_starvation.py` pools **all 13 origins**. The table below is that tool's output. The
verdict is unchanged; the metric and the scope were both wrong, and the single-origin numbers ran
about 2× high.

**K1 — starved.** `recall_mass(h36, sb) = Σ_{truth>0} pred_mean / Σ_{truth>0} truth`, over all 13
validation origins. Fires below **0.01**:

| model | h18 recall | h36 recall | waste% h36 | K1 fires? |
|---|---|---|---|---|
| `heavy_freighter` | 0.0584 | **0.0566** | 73.1 | no |
| `violet_visitor` | 0.0405 | **0.0314** | 64.3 | no |
| `bright_starship` | 0.0388 | **0.0306** | 69.7 | no |
| `purple_alien` | 0.0352 | **0.0281** | 67.7 | no |
| `pink_pirate` | 0.0304 | **0.0269** | 67.6 | no |
| `blue_stranger` | 0.0298 | **0.0245** | 68.3 | no |
| `blazing_meteor` | 0.0274 | **0.0170** | 65.3 | no |
| `bold_comet` | 0.0261 | **0.0162** | 68.9 | no |

**K1 PASS 8/8**, but read the margin: `bold_comet` clears the 1% floor by **1.6×**, not by an order
of magnitude. Every model places under 6% of true deaths on the right cells at h36.

`waste_frac` (share of predicted mass landing on zero-truth cells) is **64–73%** at h36, inside the
54–78% band `06` recorded and explicitly *not* a criterion — too high to be healthy, too uniform to
discriminate.

**Against the pre-registration's own numbers.** `06` recorded the calibration spread as
**0.008 → 0.067** and said a 1% threshold "fails the worst (`violet_visitor` at 0.008)". On
validation **with the clamp on**, `violet_visitor` reads **0.0314** — 4× better, and nothing fails.
Consistent with M71, where `violet_visitor` gains +0.14 AP@h36 from the clamp. The criterion was
calibrated on unclamped models and is now slack; not a reason to move it after the fact, but it
should be re-derived before reuse.

**K2 — bloom.** Fires if `crps_none(h36) > 10 × crps_none(h1)` **or** total predicted at h36 > 5× h1:

| model | `crps_none` h1 | h36 | ratio |
|---|---|---|---|
| `blazing_meteor` | 0.00497 | 0.00204 | 0.41 |
| `pink_pirate` | 0.00401 | 0.00195 | 0.48 |
| `blue_stranger` | 0.00486 | 0.00243 | 0.50 |
| `purple_alien` | 0.00449 | 0.00261 | 0.58 |
| `bold_comet` | 0.00424 | 0.00259 | 0.61 |
| `bright_starship` | 0.00517 | 0.00322 | 0.62 |
| `violet_visitor` | 0.00609 | 0.00378 | 0.62 |
| `heavy_freighter` | 0.00625 | 0.00533 | 0.85 |

**K2 PASS 8/8** — `crps_none` *falls* from h1 to h36 on every model (0.41–0.85), the opposite of a
bloom. ADR-070 sample-feedback plus the cell clamp is holding.

**K3 — worse than climatology: VOID.** `light_strider` produced no scores
(`results/validation/log_light_strider.txt`), traced to pipeline-core `507ae11` and filed as
**views-models#445** (29 models affected since 2026-08-02). Recorded **VOID, not PASS** — the
criterion was pre-registered and has not been tested, so **the roster has not been checked against a
baseline.** `06` also requires `light_strider` to be re-run on validation itself, which has not
happened either.

---

## M74 — Forecast maps for all three targets

27 PDFs in `results/maps/`, nine per target (eight models + the pooled ensemble), 4 × 5 panels:
gate / body mean / forecast p95 / observed truth × steps 1, 12, 18, 24, 36.

Built by re-emitting all eight with `--body-mean-dump`. **Verified non-destructive**: re-scoring
`purple_alien` after the re-emit reproduces the 2026-09-07 score CSV **byte-identically**, and the
check is not vacuous — the cube's `y_pred.npy` was rewritten at 15:48 on 2026-09-08 while the CSV
predates it.

Two method points fixed during the build, both of which had already produced a wrong reading once:

- **Row 3 shows the p95 of the draws, not the mean.** A first version plotted the mean and was read
  as "the forecast is an order of magnitude below reality in the peak cells". It is not: at the h1
  peak cell the draws were `[0, 144, 0, 0, 4, 0, 0, 0, 137, 0, ...]` against a truth of 159, whose
  mean is 10. A quantile rather than the max, because the pool has 8× the draws and a max is an
  order statistic that rises with S. Both the max and the mean are printed on every panel.
- **Scales are shared across models within a target, never across targets.** `sb`, `ns` and `os`
  differ by orders of magnitude; one scale would render two of the three as empty frames.

Placement verified per target against each model's own gate: `corr = +1.0000`, wrong orientation
`+0.0186` / `+0.0241` / `+0.1376` — reproducing C-322's own evidence.

---

## Open

- **K3** (worse than climatology) remains **VOID** pending views-models#445.
- `ns` and `os` have maps (27 PDFs in `results/maps/`) but no scored ranking; M66–M70 are `sb`.
- **views-datafactory#484** — two `lr_os_best` values in priogrid 149451 (Darfur, 2025-10 and
  2025-12) of 27,413 and 32,505, against a next-largest of 1,100 anywhere in the file, with a
  **2** in the month between them. Possible summary/aggregate assignment. `os` magnitude scores
  for month 552 should be treated as provisional until that is resolved.
