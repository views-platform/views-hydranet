# 03 — Harness and invariants

Every falsifier in `05_analysis_plan.md` is backed by a test **checked against a deliberate sabotage**, and
every arm additionally self-reports the field it actually fed. That second layer is what makes the
conclusions trustworthy: a fixture test proves a transform moves its axis on a hand-built field; only the
fed-field record proves it moved on the real one.

## Falsifier → guard map

| # | Falsifier | Guard | Sabotage it was checked against |
|---|---|---|---|
| **F1** | `use_real` ≠ `teacher_forced` | `test_F1_use_real_feeds_exactly_what_teacher_forced_feeds` (with and without statics) | `wrong_month:1` — a deliberate one-month error → red |
| **F2** | a transform touches the statics | `test_F2_no_transform_touches_the_static_channels` | a hand-built write into the static prefix → red |
| **F3** | the seam perturbs production | `test_F3_none_is_byte_identical_to_identity` | — |
| **F6** | an arm is a silent no-op | `fedfield_*.csv`, read per arm | **FIRED on `shuffle_months`** (see below) |

Both F1 and F3 were **re-checked on real data**, not only on fixtures: `use_real` reproduced the archived
`teacher_forced` row to 4 s.f. (0.2979 / 0.3008 / 0.2711) and `identity` reproduced the archived free-`sample`
row (0.2979 / 0.0070 / 0.0083). This repo has had fixture-level correctness flip in production; the archived
comparison is the check that catches it.

## F6 fired, and that is the harness's headline

`shuffle_months` was to destroy temporal persistence. It moved it **0.424 → 0.404 (5%)** — and scored
ΔAP −0.0415, a plausible middling effect that reads as "persistence matters somewhat". **Voided.** The score
alone was indistinguishable from a real result. See C-289.

## Transform orthogonality, confirmed outside the fixtures

| transform | must move | must NOT move | measured on real data |
|---|---|---|---|
| `spatial_scramble` | clustering | persistence, count, magnitude | clustering 0.447 → 0.009; persistence 0.424 (oracle 0.424); count and mean magnitude unchanged |
| `magnitude_perturb` | magnitude | clustering, count | clustering 0.447 unchanged; cost ~0 |
| `thin` / `inject` | count | — | **also perturb clustering** (→0.137 / 0.135); recorded as a confound, not claimed clean |

## Numeric-space invariant

The dynamic channels are `log1p(counts)` with **no standardisation** — the models configure
`transformations: {'log1p': [...]}` and `FeatureScaler` applies a stateless elementwise transform. All field
manipulation is in count space with an `expm1`/`log1p` round-trip, and `HydraNetInference.__init__` **raises**
if any dynamic feature is not in `transformations['log1p']`. An `asinh` feature would otherwise run every arm
on mis-scaled counts and emit plausible, wrong numbers.

## Sampler invariants (EXP-05)

| invariant | how it is held |
|---|---|
| marginals exactly preserved | Gaussian copula: Φ(z) is uniform for standard normal z, so `P(active_i) = p_i` whatever the correlation. Verified on the oracle at fixed gate: `n_active` 176–184 across all length scales |
| the smoothed field is standard normal | variance renormalised analytically; a test reproduces the un-renormalised case (variance < 0.1) to prove the guard is load-bearing |
| the **scored** cube is untouched | the correlated sampler is applied in `_sample_feedback` only; a test asserts it appears in neither `sampling.py` nor `composition.py` |
| the probe can return both answers | `topk` tie-breaking artifact (see C-291) — a test requires structured and smeared gates to separate before the probe is trusted |

## Operational invariants
Same controls as the state-freeze dossier: refuse-on-leftover-prediction-dir, `--artifact` required,
score-then-delete, disk preflight. `--targets=sb` must be passed in `=` form; the space form its own
docstring shows raises `IndexError`.
