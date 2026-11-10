# Experiment log — epic #579, locating λ*

⛔ **Append-only. Negatives and VOIDs get the same treatment as wins.** ⚠️ E-1's log
(`reports/2026-11-04_delivered_gate_lookup/07_experiment_log.md`) still reads *"No entries yet."*
for a fully pre-registered epic with a measured baseline — a reader cannot tell that it did not run.
That is the failure this file exists not to repeat.

---

## E-0 · 2026-11-10 · reproduce the chain — ✅ V1 CLEARED

**Story** S2 #581. **Instruments** `2026-11-04_rank_coupled_body/tools/sim_rank_coupling.py` and
`2026-11-04_delivered_gate_lookup/tools/e1_baseline.py`, both **unmodified**.
**Artifacts** `results/s2_sim_repro.out`, `results/s2_e1_repro.out`.

`ap_ctl` reproduced **0.5667056384496834** vs committed, `|Δ| = 0.000e+00` at tol 1e-12. Both
committed `.out` files reproduced **bit-identically** apart from wall-clock. Truth maximum **2482**
confirms raw counts. ⇒ Nothing downstream is blocked on the artifact chain.

---

## E-1 · 2026-11-10 · the `ranks FIXED` ablation held nothing — ⛔ DEFECT, FIXED

**Story** S3 #582. **Instrument** `sim_rank_coupling.py:148`. **Guard**
`tests/test_rank_coupling_ablation.py` (3 tests, including the half that must fail against the
defect). **Artifacts** `results/s3_sim_fixed.out`.

`key = np.sort(key)` makes `argsort` the identity, so magnitudes were assigned in **cell-index
order** — a scramble, not a held rank map. Its column printed the largest spread in the whole output
(flat **3–6** across every decile) and was an artefact. Corrected to hold the cell-to-quantile map;
the column now rises monotonically **0.116 → 155.459**. ⚠️ Under the corrected hold the assigned
values are a resampling of the draw multiset **with repeats**, so that column's field total is not
comparable with the others' — printed as a caveat in the output. Diff confined to the ablation
column: REAL, the λ sweep, both controls and every gate unchanged.

---

## E-2 · 2026-11-10 · the gate convention, READ — ⚠️ transcription was right, and unverified

**Story** S4 #583. **Instrument** `tools/arm_config.py`. **Guard** `tests/test_arm_config.py`
(5 tests). **Artifact** `results/s4_gate_convention.out`.

All **11** delivered arms carry `gate_threshold = 0.20` and
`forecast_composition = 'threshold_gate'`, read by importing each arm's own config and asserted by
`G-T1`, which **raises** on disagreement rather than picking a winner. ⚠️ τ is an **unswept prior**,
set deliberately **below** this model's measured calibrated **0.26**. ⭐ Decision recorded: **B fires
on `gate >= τ`, deterministically** — A's own open set — so `PREREG.md` V2 becomes structurally
unsatisfiable. ⇒ 2026-11-04's Bernoulli-gate figures are not comparable with this epic's.

---

## E-3 · 2026-11-10 · seven folds — ⭐ THE 2026-11-04 NULL WAS AN INSTRUMENT ARTEFACT

**Story** S5 #584. **Instrument** `tools/rank_coupling_verdict.py`. **Artifact**
`results/s5_seven_folds.out`. 2,256 open cell-months, 45.83% events, R = 400, 35.7 s CPU.

| contrast | result |
|---|---|
| within-B, REAL vs the noise ordering | **−3.5763** fair-CRPS, CI **[−4.24, −3.06]**, **7/7** folds |
| ✅ **V3** no-op vs a permutation of itself | **+0.1257**, spans zero |
| ✅ **V4** oracle | **−13.27**, excludes zero |
| ⛔ B against A on CRPS | **+8.52**, 7/7 folds — B **loses** |
| ⛔ the oracle against A | **−1.16**, spans zero — a perfect separation only **ties** A |

⇒ Against the committed 2026-11-04 REAL gap of **−0.0137 [−0.1138, +0.0826]** (Bernoulli gate, one
month, spans zero): the same contrast under A's own gate over 7 folds is **−3.5763**, excluding
zero. **The null did not survive the instrument being fixed.**

⚠️ Two control structures were conflated and separated here: the no-op of the **ordering** (λ=0,
within B) is not the no-op of the **promotion** question (B = A). Both are now built.

---

## E-4 · 2026-11-10 · the promotion criterion — ⛔⛔ VOID

**Story** S6 #585. **Instrument** `tools/conditional_readout.py`. **Artifact**
`results/s6_conditional_readout.out`.

⛔ **`PREREG.md` §2's ratio criterion is undefined on its own headline bin.** Bin 0 is **exactly**
the 128 open cells where A issues a literal point mass at zero (`pred_A` max `== 0`, bit-exact), so
`mean_obs/mean_pred` divides by zero. ⇒ **Amendment A-1**, dated and appended: the criterion becomes
the within-bin **absolute conditional bias**, the ratio's own numerator.

On the amended criterion:

- ✅ **V3** the **exact** no-op (B rebuilt from A's own pool and index) reads **0.000e+00** ⇒ the
  binning, pairing and origin-blocked bootstrap are **sound**.
- ⛔ **V4 FIRES**: the oracle moves the **wrong way**, **−2.471 [−3.535, −1.108]**.

⇒ **VOID — not "B is worse".** V3 passing and V4 failing localise the fault to the **statistic**: a
bin-**mean** bias measures a bin's level, and the oracle's advantage is **placement within** the bin.
⚠️ The RP-1 blindness class, measured again (#477: 14 of 20 battery columns read the mean field only
— this is a 15th).

⛔ **No second amendment was made.** A further swap after seeing which arm wins is the post-hoc
selection `C-3531` forbids, and is the loop that produced four withdrawn definitions of done.

---

## E-5 · 2026-11-10 · σ̂, measured in-run — ✅ 9.04 σ̂

**Story** S7 #586. **Instrument** `tools/sigma_hat.py`. **Artifact** `results/s7_sigma_hat.out`.
6 control-form seeds × 7 folds, R = 200, 201 s CPU.

**σ̂_paired = 0.4441** fair-CRPS at **`ddof=1`, declared**; **margin = 0.8881**, both printed
**before** any comparison. Effect **4.0155 = 9.04 σ̂**; all **42** arm-fold signs negative; ordering
quality across seeds `ρ_event` **+0.3786 ± 0.0589**, range [+0.3127, +0.4424].
⚠️ `ddof=0` would have given 0.4054 — **1.0954×** smaller at n=6.

⛔ The borrowed σ̂ table was **not** used: wrong family, wrong dropout, wrong composition,
`S=256` vs 16, inference mode never stated.

---

## E-6 · 2026-11-10 · λ* — ⛔ V6 FIRES ⇒ ANSWER 3

**Story** S8 #587. **Instrument** `tools/locate_lambda_star.py`. **Artifact**
`results/s8_lambda_star.out`. 23 + 21 arms × 7 folds, R = 200, 86 s CPU. Margin **read** from E-5's
output, never transcribed.

⛔ **The λ dial cannot move the quantity λ\* was stated in.** Its oracle term is a **global** rank
over a 99.63%-zero field, so all events occupy the top **0.37%** of it (sd **0.0021** against noise
sd 1.0). `ρ_event` is flat at ≈ −0.019 from λ=0.00 to λ=0.95 and jumps to +0.996 at λ=1.00, while
the CRPS gap moves smoothly −0.14 → −4.26. ⇒ **λ is a separation dial.** ⚠️ 2026-11-04's
*"Spearman 0.68 at lam=0.75"* was therefore a separation figure. ⇒ **Amendment A-2**: an AUC column,
plus a **ν** dial that moves event ordering while holding AUC **exactly** fixed (`G-L3`,
**0.000e+00**).

⛔ **A second defect, caught by a gate rather than by inspection:** `G-L3` first failed at
**1.571e-03** on an invariant that cannot be violated. The dial was right; `argsort(argsort(x))` is
not a rank on tied data — it is a function of **cell order**, and A's prediction is quantised at
1/16. Replaced with midranks throughout; the oracle's Spearman went 0.9963 → 0.9981 and `G-L3` to
exactly zero. Guarded by `tests/test_rank_estimators.py` (5 tests, including the refusal half).

| | |
|---|---|
| ⭐⭐ **λ\*** | **ρ_event +0.4441**, bracketed **(+0.3965, +0.4441]** |
| our body (REAL) | **+0.3127 ± 0.0713**, AUC 0.6764 |
| the gate | **+0.4418 ± 0.0583**, AUC 0.7742 — ⭐ **at λ\*** |
| ⛔ **V6** | **FIRES** ⇒ `PREREG.md` **answer 3** |

⭐⭐ **The two dials say opposite things.** ν=1.00 (perfect event ordering, body's own separation) is
**−11.53** — still worse than A's **−12.11**. λ=1.00 (perfect separation) is **−13.27** — better than
A. ⇒ **Separation is the binding constraint, not magnitude ordering.**

---

## E-7 · 2026-11-10 · the verdict — ⚠️ ANSWER 3

**Story** S9 #588. **Artifact** `03_FINDING.md`. Predictions logged as **`W-134`, `W-135`, `W-136`**
in `reports/PREDICTION_LEDGER.md` — ⛔ **0 of 3 right.** The only thing that held was the structure
that made a null informative: a dial with a known answer at both ends, and a third outcome declared
in advance.

---

## E-8 · 2026-11-09 · τ swept, composition swept, CRPS attributed — ⭐⭐ THE HEADROOM IS 3.7–11.5%

**Story** S11. **Pre-registration** `11_PREREG_tau_and_decomposition.md`, committed **alone and
before the tool existed** (`9ab37ba`) — the device S1 could not honour. **Instrument**
`tools/tau_sweep_decomposition.py`. **Guard** `tests/test_tau_sweep.py` (7 tests, each with the
half that must fail against the defect). **Artifact** `results/s11_tau_sweep.out`. 6,689 cell-months
on one fixed population, 9 rows × 7 folds, R = 200 total B-cubes per row, **37.8 s** CPU.

⛔⛔ **This entry exists because Simon corrected E-3…E-7 on three counts and two of them changed the
result.** (a) The `+8.52` loss was attributed to **B** and belongs to **τ = 0.20**, an unswept prior
set deliberately below the measured calibrated 0.26. (b) *"B cannot say nothing here"* was false —
occurrence is the **gate's** job, and `ADR-069`'s composition axis is where it goes. (c) CRPS was
the wrong headline and `C-52` already said so.

### ⭐ First: the old number REPRODUCES, so it was the attribution that was wrong

`+2.814` on the fixed 6,689-cell population × `6689/2256` = **+8.34** plug-in, **+8.54** fair,
against E-3's committed **+8.52**. ⇒ The published figure was **correct as a number**; it was a mean
over the τ = 0.20 open set, and reporting it without the sweep made a property of τ look like a
property of B.

### The sweep, whole

| τ | open | event rate | `R*` | `CRPS_A` | `MCB_A` = ⭐ the prize | `MCB_B` | ΔCRPS [blocked CI] |
|---|---|---|---|---|---|---|---|
| 0.05 | 6,689 | 0.2081 | 0.1356 | 3.195 | **0.117** | 12.094 | **+11.977** [+11.53, +12.41] |
| 0.10 | 3,972 | 0.3134 | 0.1356 | 3.195 | 0.117 | 6.172 | +6.055 [+5.84, +6.29] |
| ⬅ **0.20 shipped** | 2,256 | 0.4583 | 0.1432 | 3.189 | 0.138 | 2.952 | **+2.814** [+2.71, +2.92] |
| ⭐ **0.26 calibrated** | 1,749 | 0.5369 | 0.1514 | 3.189 | 0.167 | 2.148 | **+1.981** [+1.89, +2.06] |
| 0.35 | 1,291 | 0.6166 | 0.1601 | 3.198 | 0.207 | 1.499 | +1.292 [+1.21, +1.38] |
| 0.50 | 877 | 0.7275 | 0.1793 | 3.220 | 0.297 | 1.014 | +0.716 [+0.64, +0.79] |
| 0.65 | 572 | 0.8182 | **0.1859** | 3.275 | 0.376 | 0.728 | +0.352 [+0.30, +0.40] |
| 0.80 | 303 | 0.8944 | 0.1462 | 3.428 | 0.387 | 0.544 | **+0.157** [+0.10, +0.21] |
| `soft_gate` | 6,689 | 0.2081 | 0.1790 | 3.269 | 0.346 | 3.817 | ⛔ **VOID** (V8) |

⇒ **B's penalty collapses by 76× across τ** (+11.977 → +0.157) **and never reaches zero.** Every CI
excludes zero. ⚠️ And tightening τ is not free: `CRPS_A` rises **3.189 → 3.428**, so the gate pays
for the events it closes.

### ⭐⭐ THE HEADROOM — what the epic's question actually wanted

`CRPS_ORD = UNC − DSC` is the best score **any** forecast isotone in the index can achieve, and a
rank-coupled emission **is** such a forecast by construction (a monotone map from the index to
magnitudes). ⇒ `MCB_A` is not an estimate of B's prize — **it is the prize, exactly.**

| | |
|---|---|
| ⭐⭐ the most a perfected B could **ever** take off A | **3.7% – 11.5%** of A's CRPS (0.117 – 0.387) |
| how far B is from that ceiling today | ⛔ **1.4× to 103× the whole prize** |
| ordering information available in the index, `R*` | **0.136 – 0.186** — ⇒ 81–86% of the question is not answerable from it |
| `MCB_A` at the shipped τ | **0.138** = **4.3%** of A's CRPS |

⭐⭐ **A is already within 4.3% of the ceiling its own ordering allows.** ⇒ The rank-coupled
emission is not competing against A's magnitudes; it is competing against **A's magnitudes already
being nearly an optimal use of A's ordering.** ⛔ And this is a statement about **the emission**:
`DSC` is held identical across the two arms by construction, so none of it is an ordering claim.

### Controls and gates

✅ **G-S1** the identity `ΔCRPS == MCB_B − MCB_A` reads **0.000e+00** on all 9 rows. ✅ **G-S4**
`UNC` spread across rows **0.000e+00** ⇒ the fixed population held. ✅ **G-S5** `n = 6,689` every
row, arms aligned. ✅ **G-S2/G-S3** `MCB ≥ 0`, `DSC ≥ 0`. ✅ **G-S6** S = 16 both arms. ✅ **G-S7**
the PAV refit is isotonic and weighted-mean preserving. ✅ **V8** `B/NOISE` worse than `B/REAL` on
all 8 `threshold_gate` rows.

⛔ **`soft_gate` is VOID by V8**: ordering by `gate × body_mean` scores **+2.687 worse than noise**.
⚠️ A hypothesis, not a claim, and deliberately not pursued here: the gate enters **twice** — once in
the index and once as the per-draw coin — which is the double-suppression
`views_hydranet/distributions/truncated_negative_binomial.py:6` already names. ⇒ **No statement is
made about `soft_gate`'s ΔCRPS**, and `P-6` is marked **UNEVALUABLE** rather than scored off a void
row. ⛔ That is the same discipline that forced `P-1` to be marked WRONG in E-7.

### ⛔ Two defects in my own instrument, both in one function

1. ⛔⛔ **`ecdf_cube` returned the survival function `1 − F`.** ⚠️ **It passed every structural gate
   in the run** — `G-S1`, `G-S4`, `G-S5` all at `0.000e+00` — because a wrong transform applied to
   **both arms** cancels in an identity. What caught it was reading the magnitudes: `CRPS_A` came
   back **673** against a climatological `UNC` of **3.561**, i.e. 190× worse than the pooled
   histogram. ⇒ **An identity gate cannot catch a shared transform.**
2. ⛔ **`searchsorted(..., side="right")` does not invert to `v <= yt[j]`** for a draw strictly
   between two grid points. Small here (most pool draws are observed counts and land on the grid)
   and wrong regardless. Both guarded by `tests/test_tau_sweep.py`.
3. ⚠️ **`couple`'s tie-break is spatial.** `np.argsort(..., kind="stable")` orders a tie block by
   **cell index**, and cell index on this grid is **geography** — so the bottom of the empirical
   pool was laid out along a spatial sweep. The tie block is not incidental: `x == 0` on **63.5%**
   of the firing set at τ = 0.05 and 5.7% at τ = 0.20, because A's body draws zero ~57% of the time.
   S11 breaks ties at random (`couple_random_ties`); ⛔ S5's committed instrument is **not** changed
   and its rows stand as published.

### Predictions — `W-137` … `W-140`

| # | verdict |
|---|---|
| **P-4** τ = 0.26 shrinks ΔCRPS > 20% | ✅ **RIGHT** — +2.814 → +1.981, **−29.6%** |
| **P-5** ΔCRPS stays positive at every τ | ✅ **RIGHT** — all 8 rows positive, all CIs exclude zero |
| **P-6** `soft_gate` beats τ = 0.20 | ⚠️ **UNEVALUABLE** — the row is VOID by V8 (measured +3.471 vs +2.814) |
| **P-7** `R*` falls as τ rises above 0.26 | ⛔ **WRONG** — it **rises** 0.1514 → 0.1859 to τ = 0.65, then falls to 0.1462 |

⭐ 2 right, 1 wrong, 1 unevaluable — the first entry in this epic with a right prediction in it.
