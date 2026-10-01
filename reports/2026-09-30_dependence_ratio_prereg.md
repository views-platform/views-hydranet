# Pre-registration — what the architecture predicts for the pooled dependence ratio

**Written 2026-09-30, committed BEFORE the pooled 128-draw measurement lands.** The `models` session
asked what the emit path predicts for `Var_d(total) / Σ_cells Var_d(cell)` — variance across draws,
summed over 64,818 cells — before they measure it on the pooled cube. This is derived from the code, not
from their single-model readings (0.992 / 1.107 / 0.997 forecasting; ~0.95 validation).

Committing this before the result exists is deliberate: `2026-08-22_state_freeze_l300_dossier` records a
C-303 instance where falsifiers entered git **in the same commit as the score CSVs**, four hours after the
run. A prediction that arrives with the answer is not a prediction.

## 1. What the emit path can and cannot produce — MEASUREMENT (code reading)

| stage | stochasticity | cross-cell structure |
|---|---|---|
| **D** = `n_posterior_samples` = 4 | `LockedDropout.forward` draws `torch.rand_like(x)` — **elementwise, per position and per channel** (`locked_dropout.py:71`), cached per input shape and reused across the roll-forward | ⚠️ **The only possible source.** The mask itself is elementwise, but it acts on feature maps that subsequent **convolutions mix spatially**, so one resample shifts params over a receptive field — wide in the deep U-Net layers. Coupling is real but **local-to-regional, not global** |
| **K** = `n_head_samples` = 4 | `NegativeBinomial.sample` → `NBCore.sample(mu, theta, k, generator)`, drawn per cell given that cell's params | ⛔ **None.** Independent per cell by construction |
| **gate, `soft_gate`** | `compose_samples` → `torch.bernoulli(gate_k, generator)` on a `[..., k]` tensor | ⛔ **None.** Elementwise, independent per cell **and** per draw |
| **gate, `threshold_gate`** | `(gate_k >= τ)` — deterministic given the pass's gate | ⚠️ Within a pass, **all K draws share one zero pattern**; the pattern changes only across D |

⛔ **The decisive structural fact: the scored cube has no mechanism for spatial dependence at all beyond
the D-axis coupling.** `correlated_bernoulli` — the one function in the repo that manufactures spatial
correlation — is applied **only to the autoregressive feedback**, as a diagnostic, and
`hydranet_inference.py` states that the scored cube deliberately keeps independent sampling via
`to_cube_samples`. So a ratio near 1 is **architecturally expected**, not an empirical discovery about
conflict.

## 2. The sensitivity nobody has stated — and it changes how the number reads

`ratio = 1 + (Σ_{i≠j} Cov_ij) / (Σ_i Var_i)`. There are ~`N²` off-diagonal terms against `N` diagonal
ones, so for an average cross-cell correlation `ρ`, **`ratio ≈ 1 + ρ·N`**. At `N = 64,818`:

| ρ | predicted ratio |
|---|---|
| 1e-6 | 1.06 |
| 1e-5 | 1.65 |
| 1e-4 | 7.5 |
| 1e-3 | 65 |

⛔ **So "0.992–1.107" is not "essentially independent". It bounds the average cross-cell correlation at
roughly `ρ ≲ 1.6e-6` — independent to about six decimal places.** That is a far stronger statement than
it has been read as, and it is the single most useful thing in this note.

⚠️ **And why it is that small, given that D *should* couple cells:** the denominator carries the **full
per-cell NB sampling variance** in count space on a body with `ξ ≈ 0.8` and single cells reaching 1e4–1e5.
The between-pass shift in the field aggregate is swamped. **The ratio is a variance-*share* statistic, and
its denominator is dominated by head noise.**

## 3. Pre-registered predictions

**P1 — the pooled 128 ratio stays ≈1**, in `[0.9, 1.2]`, all three targets. *Agreeing with the `models`
session's number, and disagreeing with its reason:* their argument is that M67's 0.96–0.99 error
correlation means pooling adds no between-model factor. That is true but not load-bearing, because **the
single-model ratio was already ≈1** — pooling was never the question.

**P2 — ⛔ `threshold_gate` models show a systematically HIGHER ratio than `soft_gate` models.** Derived,
not guessed: a hard gate makes the cell's on/off state **identical across all K draws within a pass**, so
its variance is concentrated on the D axis where the only cross-cell coupling lives; a soft gate injects
independent per-cell Bernoulli noise into every draw, inflating the denominator. **This predicts the
`models` session's own falsifier fires** — they pre-registered *"a systematic difference between the
`soft_gate` and `threshold_gate` families"* as refuting their prior. Families: `soft_gate` =
blue_stranger, bright_starship, heavy_freighter, pink_pirate, purple_alien; `threshold_gate` =
blazing_meteor (τ=0.16), bold_comet (τ=0.14), violet_visitor (τ=0.20).

**P3 — if any target exceeds 1.3, the cause is the D axis and nothing else**, and the diagnostic is to
recompute the ratio **within a single pass** (K only, 4 draws). Within-pass it must be ≈1 by
construction; if it is not, the emit path is doing something undocumented.

## 4. Falsifiers

- **F1** — pooled ratio outside `[0.9, 1.2]` on any target ⇒ **P1 refuted**, and P3's within-pass
  decomposition is the required follow-up before any interpretation.
- **F2** — no systematic `soft_gate` vs `threshold_gate` separation ⇒ **P2 refuted.** That would mean the
  gate's contribution to draw variance is negligible against the body's, which is itself worth knowing and
  is the more likely way I am wrong.
- **F3** — within-pass (K-only) ratio materially above 1 ⇒ the code reading in §1 is wrong. This is the
  falsifier that would invalidate the whole note.
- ⚠️ **VOID, not refuted:** fewer than 3 models per composition family in the pooled cube; or a ratio
  computed across draws that mixes targets or months.

## 5. What the number licenses, and what it does not

⛔ **It does not license a claim about conflict.** A ratio of 1 measured on a cube whose sampler contains
no spatial-dependence mechanism is a statement about **our sampler**, not about whether conflict co-varies
across space. The two are routinely conflated and this is the place it would happen.

✅ **It does license the `models` session's operational conclusion, and by a stronger route than they
give:** no aggregate interval from this product — a country total, a regional total, anything summed —
carries usable width, because **nothing in the emit path could have given it any.** That holds under
*every* collapse, and it holds whichever way the measurement falls, which is why it outranks the collapse
comparison.

⚠️ **The gap worth watching.** If the architecture predicts a shared factor and the measurement shows
none, the factor exists in the model and **is not surviving into the emitted cube** — a plumbing question,
not a modelling one. The `models` session flagged `seed_t` deriving from `torch_seed` alone with no model
identity (`distributions/sampling.py:99`) as worth a look on that path. Agreed, and it is independent of
which way the ratio falls.

---

# OUTCOME — measured 2026-10-01, run `20261001_012905`, 8/8 models, pooled 128 draws

Scored against the predictions above, which were committed before the measurement existed.

| prediction | verdict |
|---|---|
| **P1** — pooled ratio in `[0.9, 1.2]`, all three targets | ⛔ **REFUTED. My own F1 fires on all three.** |
| **P2** — `threshold_gate` systematically higher than `soft_gate` | ✅ **SURVIVED**, direction only |
| **P3** — if anything exceeds 1.3, the cause is the D axis | ⛔ **WRONG PREMISE.** The cause is pooling, which P3 did not contemplate |
| **F3** — a within-pass ratio above 1 would invalidate §1's code reading | did not fire; §1's reading of the **per-model emit path** stands |

## P1: refuted, and the error is one this report has a name for

| target | pooled ratio (median) | range over 36 months | implied ρ |
|---|---|---|---|
| `lr_sb_best` | **9.8118** | 1.93 – 11.47 | ~1.4e-4 |
| `lr_ns_best` | **2.1638** | 1.42 – 2.77 | ~1.8e-5 |
| `lr_os_best` | **1.8823** | 1.01 – 3.02 | ~1.4e-5 |

⛔ **Pooling manufactures cross-cell dependence, and the single-model measurement did not predict it.**
Per-model ratios are 0.87–1.41; pooled `sb` is **9.81**. The dominant term is a **between-model level
shift** — model A runs hot across the whole field, B cold — which is a common factor over every cell.

⛔ **My error, named in this report's own vocabulary: §3.1 shape (c), measurement on the wrong object.** I
traced the per-model emit path, established correctly that it contains no spatial-dependence mechanism, and
then generalised that to the **pooled product**, which is one level up and outside the path I examined. The
code reading was right about the object I read. The object that mattered was not it.

✅ **What did survive is the sensitivity relation**, and it is what makes the result interpretable: `ratio
≈ 1 + ρ·N` at `N = 64,818` converts all three numbers into ρ directly, and the three implied ρ values
(1.4e-4, 1.8e-5, 1.4e-5) are what the ratios mean. **The framework held; the prediction made with it did
not.** Those are separable and only the second was wrong.

## P2: survived — derived from the architecture before the data existed

`lr_sb_best`, per model, 16 draws:

| `soft_gate` | | `threshold_gate` | |
|---|---|---|---|
| blue_stranger | 1.2251 | blazing_meteor | 1.4089 |
| bright_starship | 1.0726 | bold_comet | 1.2110 |
| heavy_freighter | 1.0472 | violet_visitor | 0.9967 |
| pink_pirate | 1.0502 | | |
| purple_alien | 0.8698 | | |
| **median-of-medians** | **1.0502** (n=5) | **median-of-medians** | **1.2110** (n=3) |

Ratio **1.153**, hard gate higher. **F2 did not fire.** The mechanism was derived, not guessed: a hard gate
makes a cell's on/off state identical across all K draws within a pass, concentrating variance onto the D
axis; a soft gate injects independent per-cell Bernoulli noise into every draw and inflates the denominator.

⚠️ **Scope, and it is the honest limit: n=3 against n=5, one target, and the ranges overlap** —
`violet_visitor` at 0.9967 sits below three of the five soft models. **Direction established, magnitude
not**, which is this repo's own phrase for exactly this situation. Nothing more should be claimed from
eight models.

## The operational conclusion was half wrong, and the half that saved it was the hedge

I wrote: *"no aggregate interval from this product carries usable width, because nothing in the emit path
could have given it any — and that holds whichever way the measurement falls."*

⛔ **The first clause is now false.** The pooled product carries width — ~**9.8× the independent-sampling
baseline** on `sb`. ✅ **The second clause is what kept the claim honest**, and it is the only reason this is
a correction rather than a retraction.

✅ **And the corrected version is better news than either session predicted:** country and regional totals
from the pooled product are **less falsely confident** than we both expected, and pooling specifically is
what buys that. ⚠️ It does nothing for the **location** half — MCR ~0.09, predicted totals ~10% of
observed — so the aggregate is honestly wide around a number that is far too low.

## A measured follow-up worth pre-registering for the next roster

The ratio tracks **signal density**: `sb` 1.99% of cells carrying signal → 9.81; `os` 0.74% → 1.88; `ns`
0.63% → 2.16. Consistent with between-model level differences having more cells to accumulate over.
**Testable on any future roster**, and it predicts that a denser target or a larger region raises the ratio.

## Cross-session corrections recorded

- **Mine:** P1 refuted; P3's premise wrong; the operational conclusion's first clause false.
- **The `models` session's**, volunteered: their asymmetric 16-draw caveat on `tower_point` is dead — more
  draws destroyed the estimator rather than rescuing it; and their M67 argument was wrong, because gate
  errors correlating 0.96–0.99 **does not preclude systematic level differences** between models. Highly
  correlated errors, differently-centred predictions.
