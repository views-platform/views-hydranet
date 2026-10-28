# RP-2 — the noise floor for B-1a and B-1b: one readout's noise EXCEEDS its signal

**Date** 2026-10-28 · **Cost** 27 min d4 + 10 min downstream, local CPU, $0 · **All 8 gates PASS**
**Scripts** `determinism_matrix.py` `mask_pairing_gate.py` `sigma_hat_aggregate.py` `run_downstream.sh`
— all committed **before** they ran (D-5) · **Output** `sigma_hat_aggregate.out`

**C-351 slots.** Population = 64,818 land cells, in-sample months 421–456, out-of-sample 457–469 ·
transform = as each instrument computes it (B-1a variogram ratio, B-1b pooled FSS over a
count-matched binarisation) · estimator = **pooled sample sd, ddof=1, n=3, 2 df** — locked in
advance at `2026-10-18_weight_decay_arms_prereg.md:1051`, not chosen after seeing the spread ·
CV = σ̂ / |mean| · ⛔ **not** normalised against any reference.

---

## 1 · What was measured, and why it is clean

Three true seed replicates — `silver_0_1_4` (42), `bronze` (43), `cobalt` (44) — identical configs,
identical data, identical git sha `2e73de3`, **differing only in the seed**. Scored through the
committed route (`d4_mu_on_the_grid.py` → `d12_variogram.py` → `body_scoreboard.py` /
`b1b_placement_baseline.py`), every call tagged.

⭐⭐ **There is no measurement-noise term to subtract, and that is measured rather than assumed.**
Training is bit-deterministic (M74: 190/190 weight tensors identical across two git shas and three
library versions) and **scoring is bit-deterministic** (G-9: `mu`/`theta`/`gate` reproduce at exactly
`0.000000e+00` across a seven-day gap). ⇒ **The spread below is the seed and nothing else.**

## 2 · ⛔⛔ THE FINDING: B-1b body-alone has σ̂ LARGER than its own mean

| window | n | mean | **σ̂** | **CV** | 4.97σ̂ bar |
|---|---|---|---|---|---|
| in | 1 | 0.1055 | **0.1796** | **1.703** | 0.8927 |
| in | 9 | 0.1969 | 0.3168 | **1.609** | 1.5743 |
| in | 65 | 0.4577 | 0.2858 | 0.624 | 1.4206 |
| out | 1 | 0.1019 | 0.1507 | **1.479** | 0.7489 |
| out | 33 | 0.2947 | **0.4288** | **1.455** | **2.1312** |

**CV > 1 at 11 of the 14 scales.** The across-seed standard deviation **exceeds the quantity being
measured**. At out/n=33 the honest bar for believing a single-arm difference is **2.1312 against a
mean of 0.2947** — ⛔ **the bar is 7.2× the value itself.**

⇒ **No single-run comparison on B-1b body-alone can ever clear its own noise floor.** It is not a
weak instrument; at n=1 it is **not an instrument**. Every number quoted from it from one run —
including the committed baselines — carries no information about the model.

## 3 · ⭐ One readout IS usable, and it is now identified rather than assumed

| window | n | mean | σ̂ | CV | bar |
|---|---|---|---|---|---|
| out | gated 1 | 0.6030 | 0.0327 | **0.054** | 0.1626 |
| out | gated 9 | 0.8509 | 0.0315 | 0.037 | 0.1564 |
| out | gated 65 | 0.9264 | 0.0227 | **0.024** | 0.1127 |

**B-1b gated, out-of-sample: CV 0.024–0.054.** ⇒ A difference of **≈ 0.16** in gated pooled FSS
clears the bar. **This is the placement instrument arms should be run against**, and that is now a
measured statement.

## 4 · ⚠️ B-1a is weakest exactly where it is quoted

The published B-1a headline is **d = 1**. Out of sample, E-W, d=1: mean **0.9523**, σ̂ **0.5073**,
bar **2.5214** — on a quantity whose whole observed range is ~0.4–1.5. ⛔ **Unusable at the headline
setting.** It steadies with scale (out/isotropic/d=32: σ̂ 0.1223, CV 0.156) but the bar there is
still **78% of the value**.

## 5 · ⭐⭐ σ̂ is NOT flat in the scale parameter — and it moves in OPPOSITE directions

| | σ̂ at the smallest scale | σ̂ at the largest | direction |
|---|---|---|---|
| B-1a, E-W, out | 0.5073 | 0.1560 | **falls** |
| B-1b gated, out | 0.0327 | 0.0227 | **falls** |
| B-1b body, out | 0.1507 | 0.4288 | ⛔ **rises** |

⇒ **A single scalar σ̂ would have been wrong in both directions at once.** Spans across the profile:
**7.9×** (B-1a), **18.9×** (B-1b). **D-4 was not ceremony here** — it is the difference between a
usable noise floor and a misleading one.

## 6 · P3 scored — one confirmed, one refuted, and the refutation is my fault not the data's

Declared in `FINDING_RP2_S0.md` §8 and committed **before** the aggregate ran.

**P3.1 ✅ CONFIRMED, 14 of 14.** CV(gated) < CV(body alone) at every `n`, both windows — in-sample
0.342…0.132 against 1.703…0.624; out-of-sample 0.054…0.024 against 1.479…1.144. ⇒ **The gate is the
seed-stable half of the model**, as predicted from the raw cube deltas (mean |Δ| 4.35 on `mu` against
0.0074 on `gate`).

**P3.2 ⛔ REFUTED.** I predicted CV(B-1a) > CV(B-1b gated) "at comparable scales". At the smallest
scale it is the reverse: **0.105 (B-1a in/E-W/d1) against 0.342 (B-1b in/gated/n1)**.

⛔ **And the real defect is the prediction, not the surprise.** `d` is a **variogram lag in grid
cells** and `n` is an **FSS window width**; they are not the same parameter and there is no mapping
between them. *"At comparable scales"* was therefore **not a testable statement** — I wrote a
prediction whose terms I could not evaluate, and only noticed when I had to score it. ⇒ **W-95**,
and the lesson is the one C-351 already states: a comparison needs its population **and its axis**
named, or it is not a comparison.

⚠️ P3.2's declared counter-case — *"both instruments discard magnitude, so a rank-preserving `mu`
perturbation would move neither"* — is **NOT** what happened. Both instruments moved plenty. The
counter-case is unused, not confirmed.

## 7 · What this hands the programme

1. ⛔ **B-1b's own closing line is now answerable.** `b1b_placement_baseline.out` ends *"M-3 (σ̂) is
   absent … these are BASELINES and B-1b carries NO decision rule."* σ̂ now exists — and it says the
   **gated out-of-sample** variant may carry a decision rule and the **body-alone** variant may not.
2. ⛔ **RP-5, RP-6 and RP-8 are unblocked but constrained**: any arm they run must be judged on the
   gated out-of-sample readout, against **4.97 σ̂ ≈ 0.16**, not on body-alone placement.
3. ⚠️ **The cost of three seeds is visible**: the bar falls to ≈3.0 σ̂ at n=5 and ≈2.5 σ̂ at n=10.
   Two more seeds would buy a ~40% tighter bar for ~17 min of CPU each.

## 8 · ⛔ What this does NOT establish

- **Not a σ̂ for `nb`.** All three replicates are **`truncated_nb` @ dropout 0.15**. The published
  B-1a 0.391 / B-1b 0.6267 are **`nb` @ 0.15 on a DELETED checkpoint**; the live `nb` control is
  @ **0.125**. B-1a on `truncated_nb` reads **1.462** against `nb`'s **0.391** — a **3.7× level
  shift from the family alone**, reproduced here from a live checkpoint. ⇒ σ̂ discharges the noise
  floor for **`truncated_nb` @ 0.15** and, for `nb`, only **provisionally as a transported CV**.
  ⛔ Amend the prereg's rule 7 to say which half is discharged.
- **Not the gate-slope σ̂** of the 2026-10-21 readout (M74, C-390). Different statistic, never
  quoted against this one (GLOSSARY §5e).
- **Not a verdict on any arm.** These are noise measurements against known-identical configs.
- **Not a tight σ̂.** n=3, 2 df, 95% CI **[0.52 σ̂, 6.28 σ̂ ]** — a **12.1-fold** span. Every bound
  is derived from the **upper** limit.
