# Experiment log — placement interventions

Append-only. Verdicts against the pre-registered falsifiers are recorded **before** predictions are read.
Negatives are recorded with the same prominence as wins.

---

### EXP-01 · copula re-test on a non-floor-limited vehicle · 2026-08-17 · **NULL — M5 CONFIRMED, with a mechanism**

- **Plan (pre-reg):** `05_analysis_plan.md` §4–§7, locked before execution.
- **Variable:** `feedback_length_scale` ∈ {1.0, 3.0, 8.0} — `correlated_bernoulli` replaces the
  independent Bernoulli on the **feedback path only**. Everything else identical to the control.
- **Driver / artifact / results:** `run_realism_arms.py --arms identity --length-scale <ℓ>` ·
  `calibration_model_20260812_191742.pt` (sha `909f44c0…`) · `results/score_violet_visitor_*_ls*.csv`
- **Control:** the already-scored `identity` arm from
  `reports/2026-08-17_vehicle_replication_dossier/` — not re-run.

#### Verdict vs falsifiers (plan §5) — recorded first

| # | verdict | evidence |
|---|---|---|
| **F1** silent no-op | **PASS** | clustering moved 0.0685 → 0.1061 at ℓ=1.0, a 1.55× rise (fires only below 1.2×) |
| **F2** h=1 identical | **PASS** | max \|ΔAP\| at h1 across all three arms = 0.00e+00 |
| **F3** support | **PASS** | `N` = 170430 in every scored row |
| **F4** firing-rate confound | **PASS, N/A** | `act_ratio` shift −1.7%; no arm gained, so nothing to confound |

#### Readout

**Dose — fed-field clustering (real field = 0.4473):**

| arm | clustering | % of real |
|---|--:|--:|
| control (independent draw) | 0.0685 | 15% |
| ℓ = 1.0 | 0.1061 | 24% |
| ℓ = 3.0 | 0.1144 | 26% |
| ℓ = 8.0 | 0.1056 | 24% |

**It saturates.** A 16× range of length scale plateaus at ~25% of the real field's clustering.

**Primary endpoint — gate AP, target `sb`:**

| h | control | ℓ=1.0 | ℓ=3.0 | ℓ=8.0 | best ΔAP |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.4745 | 0.4745 | 0.4745 | 0.4745 | +0.0000 |
| 6 | 0.3924 | 0.3876 | 0.3882 | 0.3886 | −0.0038 |
| 12 | 0.3226 | 0.3132 | 0.3069 | 0.3072 | −0.0094 |
| 18 | 0.2569 | 0.2538 | 0.2536 | 0.2523 | **−0.0031** |
| 24 | 0.2060 | 0.2032 | 0.2017 | 0.2027 | −0.0028 |
| 30 | 0.1699 | 0.1579 | 0.1630 | 0.1651 | −0.0048 |
| 36 | 0.1370 | 0.1303 | 0.1347 | 0.1292 | −0.0023 |

Every delta at every dose is **negative**. The best case anywhere is −0.0023.

#### Predictions

| # | verdict | |
|---|---|---|
| **P1** (primary) | **HOLDS — null** | \|ΔAP\| = 0.0031 at h18, inside the ±0.01 threshold |
| **P2** | **FALSIFIED** | ℓ=3.0 did **not** bracket the oracle's clustering; the sweep saturates at 26% |
| **P3** | **CONFIRMED** | every effect is negative or null, as skepticism §2 predicted |

#### Why — the mechanism, demonstrated not asserted

P2's failure is the interesting part. On `truncated_smoke`, ℓ=1.0 *overshot* the real clustering (0.494
vs 0.449). Here no dose gets past a quarter of it. **A marginal-preserving sampler cannot move a
confident gate.**

Established on a controlled synthetic sweep (`correlated_bernoulli` on hand-built gates with identical
expected active count, `scratchpad/skew.py`):

| gate | ℓ=1.0 | ℓ=3.0 | ℓ=8.0 |
|---|--:|--:|--:|
| uniform (unconfident) | 0.758 | 1.534 | 1.806 |
| skewed, 1000 cells @ p=0.40 | **0.097** | **0.113** | 0.425 |
| skewed, 600 cells @ p=0.70 | 0.086 | 0.144 | 0.214 |
| skewed, 500 cells @ p=0.90 | 0.033 | 0.051 | 0.025 |

The real run (0.106 → 0.114) lands on the `1000 @ 0.40` line. When probability concentrates on a
specific set of cells, `Φ(z) < p` is dominated by `p`; correlation can only reshuffle among cells of
comparable probability, and there are too few. **The marginals the copula must preserve have already
chosen the cells.**

⚠️ **A wrong explanation was published first and is corrected here.** The initial reading was that the
gate is too *diffuse* for the copula to clump. The synthetic sweep refutes it directly — a *uniform*
(maximally diffuse) gate reaches clustering 1.53 at ℓ=3.0, 13× what the real run achieved. The bound is
**skew**, not diffuseness: the gate is too **decided**, not too vague. Recorded because the first
version was stated with more confidence than a story deserves, twenty minutes before the test that
killed it.

#### Decision (plan §7)

Per the pre-committed rule for "P1 holds and F1–F4 clear": **M5 is confirmed on a vehicle that is not
floor-limited, and the inference-time sampling family is CLOSED.** It is closed more strongly than the
rule anticipated — not "clustering does not help" but "no marginal-preserving sampler can deliver the
clustering", which rules out the whole class rather than one instance.

**Consequence for the deferred top-K arm.** Top-K is *not* marginal-preserving, so this bound does not
apply to it. But the same measurement raises a new question: if the gate has already committed to its
cells, top-K selects nearly the ones the Bernoulli draw already fires, and gains little. M7's headroom
figure (top-K 4–27× more clustered than the draw) was measured on `truncated_smoke`, whose gate we now
know was diffuse and floor-limited. **Measure violet's gate structure before building anything** —
`--gate-probe`, ~10 min — rather than spending a build on headroom that may not exist.

#### Scope

One seed (42), one vehicle, one target (`sb`), 13 origins, S=16. **Not byte-paired** (C-296): the copula
consumes different RNG than the control, so read at one significant figure — which is why a −0.003 delta
is reported as a null, not as a small negative effect.
