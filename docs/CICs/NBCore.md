# Class Intent Contract: NBCore (`views_hydranet/distributions/nb_core.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-08-14
**Related ADRs:** ADR-067 (distribution-family subsystem — per-cell NB/ZINB), ADR-028 (Numerical
Stability Guards), ADR-008 (Error Propagation), ADR-009 (Boundary Contracts & Configuration
Validation), ADR-002 (Layering)

---

## 1. Purpose

> The single shared **Negative-Binomial count-math authority**: the closed-form NB log-pmf, its
> zero mass (`prob_zero` / `log_prob_zero`), and a generator-deterministic Gamma-Poisson `sample`,
> parameterised by per-cell mean `mu` and dispersion `theta`, plus the scalar link helpers and the
> weighted-mean reduction every family reuses.

`NBCore` is a stateless module of static/pure functions (`nb_core.py:109-110`) that all four
families — `nb`, `zinb`, `mixture_nb`, `truncated_nb` — and the two legacy losses **compose**
(`has-a`, not inherit): `NegativeBinomialFamily` (`negative_binomial.py:5`),
`ZINBFamily` (`zero_inflated_negative_binomial.py:4`, `NBCore` + a structural π spike),
`MixtureNBFamily` (`mixture_negative_binomial.py:4`, `NBCore` twice),
`TruncatedNBFamily` (`truncated_negative_binomial.py:15`), and the legacy
`TruncatedNBLoss` / `DenseNBLoss` (which reuse only the `inverse_softplus` link,
`truncated_nb_loss.py:28,51` / `dense_nb_loss.py:28,51`). One place owns the NB count math so a
stability fix (C-212) or a determinism fix (C-3) lands once, not per family.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** own any head, activation, or link application. It exposes the scalar inverse links
  (`inverse_softplus`, `logit`, `nb_core.py:27-46`) for informed init but never activates a head —
  a family's `activate` maps raw channels to `(mu, theta)` before calling in.
- Does **not** own the gate, composition, or `self_zeroed` policy. `NBCore` computes the raw NB
  quantities; whether a forecast is `gate × body` or structurally self-zeroed lives in the family
  (`self_zeroed` flag) and `ForecastComposer`, never here.
- Does **not** assemble the `D×K` cube or run the MC-dropout / rollout loop — `sample` returns the
  `k` per-cell aleatoric draws (`nb_core.py:153`); `to_cube_samples` (`sampling.py`) does the cube.
- Does **not** carry per-family policy: zero-inflation (`zinb`'s π mixture,
  `zero_inflated_negative_binomial.py:78-80`), zero truncation (`truncated_nb`'s hurdle body,
  `truncated_negative_binomial.py:92-105`), and the two-component mixture
  (`mixture_negative_binomial.py:88-93`) are assembled by the families *from* `NBCore` calls.
- Does **not** expose a `mean`: `E[Y] = mu` is trivially the `mu` parameter, so the family returns
  it directly (`negative_binomial.py:77-80`) rather than round-tripping through the core.
- Does **not** own the count↔log1p target transform — a family recovers raw counts via
  `to_raw_counts` before calling `NBCore.log_prob` (`negative_binomial.py:63-65`); `NBCore` works in
  raw count space throughout (`nb_core.py:114`).

---

## 3. Responsibilities and Guarantees

`NBCore` (`nb_core.py:109`) is stateless — all members are `@staticmethod`, no `__init__`, no fields.
Parameterisation matches `torch.distributions.NegativeBinomial(total_count=theta,
probs=mu/(mu+theta))` so `E[Y] = mu`, `Var[Y] = mu + mu**2/theta` (`nb_core.py:3-5,116-118`).

Static methods:
- `log_prob(mu, theta, y) -> Tensor` (`nb_core.py:112-118`) — log NB pmf `log P(Y=y | mu, theta)`,
  `y` in raw count space. Delegates to `torch.distributions.NegativeBinomial(..., validate_args=False)`.
- `prob_zero(mu, theta) -> Tensor` (`nb_core.py:120-124`) — closed form `(theta/(theta+mu))**theta`.
- `log_prob_zero(mu, theta) -> Tensor` (`nb_core.py:126-144`) — the C-212-stable
  `-theta * log1p(mu/theta)` form of `log P(Y=0)` (see §Invariants).
- `sample(mu, theta, k, generator=None) -> Tensor` (`nb_core.py:146-165`) — `k` count draws per cell
  → `[*mu.shape, k]`, **deterministic** under `generator` (C-3).

Module helpers (exported, reused by every family/loss):
- `inverse_softplus(y: float) -> float` (`nb_core.py:27-35`) — inverse-softplus link `x` s.t.
  `softplus(x) == y` for `y > 0`, via the overflow-safe `y + log1p(-exp(-y))` form; **raises** on
  `y <= 0`.
- `logit(p: float) -> float` (`nb_core.py:38-46`) — inverse-sigmoid link for `0 < p < 1`; **raises**
  outside the open interval.
- `check_param_target_shape(counts, mu) -> None` (`nb_core.py:49-59`) — the shared `nll` shape guard;
  **raises** `ValueError` when `counts.shape != mu.shape` (blocks a silent `[N,1]`-vs-`[N]` broadcast).
- `weighted_nll_mean(per_cell, weight, eps=1e-8) -> Tensor` (`nb_core.py:62-79`) — broadcast-normalized
  weighted mean of a per-cell loss (the shared family reduction). `weight=None` → plain mean; an
  all-zero weight yields a graph-connected `0` (zero gradient), not an error.
- `_clamp(mu, theta) -> (Tensor, Tensor)` (`nb_core.py:23-24`) and `_standard_gamma(concentration,
  generator) -> Tensor` (`nb_core.py:82-106`) are module-private internals.

Guarantees: pure, stateless computation; every public entry point clamps `mu, theta` away from `0`
via `_clamp` before use (`nb_core.py:115,123,143,154`), so the likelihood cannot blow up at the
boundary (C-199). `sample` returns non-negative integer counts (`torch.poisson`, `nb_core.py:165`)
and, given the same seed + params, byte-identical draws (C-3).

---

## 4. Key Invariants and Assumptions

- **Numerical stability — the C-212 `log_prob_zero` form (ADR-028).** `log_prob_zero` uses
  `-theta * log1p(mu/theta)` (`nb_core.py:144`), NOT the algebraically-equal
  `theta * log1p(-mu/(theta+mu))`. In float32, once `theta < ½·ulp(mu)` (e.g. `mu=1000`,
  `theta≈1e-5`) the sum `theta+mu` rounds to `mu`, so `mu/(theta+mu)` becomes **exactly** `1.0` and
  `log1p(-1) = -inf`; the forward value survived (ZINB's `logaddexp`/`where` masked it) but the
  **backward** hit `d/dz log1p(z)|₋₁ = 1/0 = inf` → `0·inf = NaN`, sprayed to every upstream param by
  the mean reduction — the ZINB lesson-18 gradient explosion (`nb_core.py:130-141`). **Who
  differentiates through it:** only `ZINBFamily.nll` puts `log_prob_zero` on the backward path (its
  zero mass mixes π with the NB zero, `zero_inflated_negative_binomial.py:78`). `nb`, `mixture_nb`,
  and `truncated_nb` call `log_prob_zero` too, but only inside the **no-backward** `prob_positive`
  scoring path (`-expm1(log_prob_zero)`, `negative_binomial.py:90`,
  `mixture_negative_binomial.py:126-127`, `truncated_negative_binomial.py:97,141`), so all-cell NB
  training was **unaffected** — which is why the bug hid until ZINB. The stable form also keeps
  `P(Y>0) = -expm1(log_prob_zero)` accurate in the `mu << theta` tail (avoids the
  `theta/(theta+mu) → 1` cancellation, C-201).
- **`_clamp` is a value-clamp, not a gradient-bound.** `_clamp` floors both `mu` and `theta` at
  `_EPS = 1e-6` via `clamp_min` (`nb_core.py:20,23-24`) — it bounds the *forward* value only and does
  NOT bound the `1/theta` (digamma) gradient as `theta → floor`. The feared exploding-θ gradient
  (C-205 / C-202) is **not owned here**: it is cancelled at the head channel by the softplus link
  (`dθ/d(raw) = sigmoid(raw) → 0` cancels the `1/θ` term), documented and tested at the family level
  (`test_theta_gradient_bound.py`), not by any floor in `NBCore`. Treat the `_EPS` floor as a
  boundary guard, not a gradient guard.
- **Determinism — generator-aware Gamma-Poisson (C-3 / S2 #121).** `torch.distributions` ignore a
  `torch.Generator`, so `sample` draws the NB as a Gamma-Poisson from generator-native primitives: a
  vectorised Marsaglia-Tsang Gamma over `torch.randn`/`torch.rand` (`_standard_gamma`,
  `nb_core.py:82-106`) then `torch.poisson`, all threading `generator` (`nb_core.py:96,98,105,165`).
  Same seed + same params ⇒ same draws; **a non-deterministic sampler is a regression, not an edge
  case.**
- **Independent-per-cell draws under a broadcast `theta`.** `sample` broadcasts `mu, theta` to one
  common per-cell shape *before* drawing (`nb_core.py:159`), so a per-target `theta` `[1,C,1,1]`
  against a per-cell `mu` `[B,C,H,W]` gets one **independent** Gamma per cell — not a per-channel draw
  tied across the grid (`nb_core.py:155-158`).
- **Assumptions:** `mu, theta` are per-cell tensors, broadcastable together; `y` / `counts` are in
  **raw count space** (the caller recovers counts from the log1p target before calling in). `k` is a
  positive int. `generator` is a seeded `torch.Generator` whenever determinism is required (it is, on
  the inference path). `inverse_softplus`/`logit` take Python floats.

---

## 5. Outputs and Side Effects

- Pure per-call computation: no I/O, no logging, no global state, no mutation of inputs.
- `log_prob` / `prob_zero` / `log_prob_zero` return per-cell tensors shaped like the broadcast params;
  `weighted_nll_mean` returns a scalar loss tensor; `inverse_softplus` / `logit` return Python floats;
  `check_param_target_shape` returns `None` (a guard).
- `sample` is **stochastic** but reproducible: it returns a `[*mu.shape, k]` tensor of non-negative
  integer counts, deterministic under the supplied `generator` (`nb_core.py:153,165`).

---

## 6. Failure Modes and Loudness

- `inverse_softplus(y)` **raises `ValueError`** for `y <= 0` — the softplus range is `(0, ∞)`, so a
  non-positive target is a caller bug, failed loud, never a silent `nan` (`nb_core.py:33-35`).
- `logit(p)` **raises `ValueError`** for `p` outside the open `(0, 1)` — fail-loud outside the sigmoid
  range (`nb_core.py:44-46`).
- `check_param_target_shape(counts, mu)` **raises `ValueError`** on a param/target shape mismatch,
  before any NB call, so a trailing-singleton target can never silently broadcast the log-pmf into a
  wrong-rank per-cell loss (`nb_core.py:55-59`).
- `weighted_nll_mean` fails loud on an **incompatible** weight shape (`torch.broadcast_to`,
  `nb_core.py:77`); an **all-zero** weight is intentionally NOT an error — it returns a graph-connected
  `0` (no supervised cells → zero gradient), guarded by `eps` on the divide (`nb_core.py:73,79`).
- Degenerate params (`mu` or `theta` at/below `0`) do NOT raise — `_clamp` floors them at `_EPS` so
  `log_prob`/`sample` stay finite (a boundary guard, `nb_core.py:23-24`; regression-tested at
  `test_nb_core.py:73-81`). This is the one deliberate non-loud path, and it is a *stability* guard,
  not a policy fallback.

Aligns with ADR-008 (errors propagate, no silent fallback) and ADR-028 (numerical guards are
explicit and documented, not incidental).

---

## 7. Boundaries and Interactions

- **Consumed by** the four families (`negative_binomial.py`, `zero_inflated_negative_binomial.py`,
  `mixture_negative_binomial.py`, `truncated_negative_binomial.py`), each of which **composes**
  `NBCore` (`has-a`) for its count math and adds its own zero policy on top. The legacy
  `TruncatedNBLoss` / `DenseNBLoss` reuse only `inverse_softplus`.
- **Depends only on** `torch` and the stdlib `math` (`nb_core.py:16-18`). It must not depend on the
  head, the training loop, models, inference, config, or framework/I-O layers (ADR-002). It is a leaf
  in the distributions subsystem.
- **Trusts** its callers to pass raw-count-space targets and activated `(mu, theta)`; it treats the
  torch NB reference as the authority for the positive-branch log-pmf (`nb_core.py:116-118`) and
  re-implements only the zero mass and the sampler (the two places torch's reference is unstable or
  generator-blind).

---

## 8. Examples of Correct Usage

```python
from views_hydranet.distributions.nb_core import NBCore, weighted_nll_mean, check_param_target_shape

# Inside a family's nll (mu, theta are the activated per-cell params; counts are raw-count space):
check_param_target_shape(counts, mu)
nll_per_cell = -NBCore.log_prob(mu, theta, counts)
loss = weighted_nll_mean(nll_per_cell, weight)          # scalar; weight=None -> plain mean

# Occurrence scoring (no-backward path) — stay accurate in the mu << theta tail:
p_pos = -torch.expm1(NBCore.log_prob_zero(mu, theta))   # P(Y>0), C-201

# Deterministic aleatoric draws for the sample cube:
draws = NBCore.sample(mu, theta, k=cfg.n_head_samples, generator=gen)  # [*mu.shape, k]
```

```python
from views_hydranet.distributions.nb_core import inverse_softplus
# Informed head init: seed the theta channel so softplus(raw) == the theta prior.
raw_theta = inverse_softplus(float(priors.get("theta", 1.0)))
```

---

## 9. Examples of Incorrect Usage

- Re-deriving `log P(Y=0)` inline as `theta * log(theta/(theta+mu))` or
  `theta * log1p(-mu/(theta+mu))` instead of calling `NBCore.log_prob_zero` — reintroduces the C-212
  float32 backward NaN that only ZINB differentiates through (`nb_core.py:130-141`).
- Calling `NBCore.sample` without a `generator` on the inference/rollout path — reintroduces the
  non-determinism the S2 #121 gate forbids (C-3).
- Passing a log1p-space target to `NBCore.log_prob` (it expects **raw counts**), or feeding a
  `[N, 1]` target against `[N]` params without `check_param_target_shape` — a silent wrong-rank
  broadcast.
- Relying on `_clamp`'s `_EPS` floor to bound the θ **gradient** — it bounds only the forward value;
  the gradient is bounded by the softplus link at the head, not here (C-205 / C-202).
- Adding a per-family behaviour (a π mixture, a truncation, a two-component blend) *into* `NBCore` —
  that policy belongs in the family; `NBCore` stays the single-NB count authority.

---

## 10. Test Alignment

- **Green — `tests/distributions/test_nb_core.py`:** `log_prob` vs the `torch.distributions.NegativeBinomial`
  reference (`:19-28`); `prob_zero` closed form (`:10-16`); sample shape / non-negative integers /
  mean recovery (`:31-40`); **determinism under a fixed generator seed** (`:43-50`, C-3);
  independent-per-cell draws under a broadcast `theta` (`:53-70`); boundary-clamp finiteness at
  degenerate params (`:73-81`); sample **variance** recovers `mu + mu²/theta` (`:123-137`, C-208) and
  empirical `P(Y=0)` matches the analytic NB(0) (`:158-170`); `inverse_softplus`/`logit` domain
  guards raise (`:140-155`).
- **C-212 regression guards (`test_nb_core.py`):** `log_prob_zero` forward AND backward finite at
  `probs → 1.0` saturation, with a self-validating premise that the regime is actually hit
  (`:84-103`); the positive `log_prob` branch stays finite in backward as a delegated-reference guard
  (`:106-120`).
- **C-205 / C-202 boundary (`tests/distributions/test_theta_gradient_bound.py`):** the near-`_EPS`
  θ head-channel gradient stays `O(1)` for `nb`/`zinb` across targets — confirming the `_clamp` floor
  is not the gradient bound (the softplus link is).
- **Composition coverage:** the shared-core reuse is exercised through the family suites
  (`test_negative_binomial.py`, `test_zero_inflated_negative_binomial.py`,
  `test_mixture_negative_binomial.py`, `test_truncated_negative_binomial.py`) and the D×K sampler /
  determinism (`test_sampler_dxk.py`).
- **Must never regress:** the C-212 stable `log_prob_zero` form, sampler determinism (C-3), the
  per-cell independence of broadcast-`theta` draws, and the fail-loud domain/shape guards.

---

## 11. Evolution Notes

- **Stable:** the `(mu, theta)` parameterisation matched to `torch`'s NB, the C-212 `log_prob_zero`
  form, generator-deterministic sampling, and the exported link/reduction helper set.
- **Expected to change:** a heavier-tailed count core (e.g. bulk + GPD tail, PIG) would be added as a
  *sibling* core alongside `NBCore`, composed by a new family — not by subclassing `NBCore` or bolting
  a tail onto it (mirrors the `DistributionFamily` evolution note).
- **Would require revisiting this contract:** any reparametrization of `log_prob` / `log_prob_zero`
  (re-audit the float32 backward at `probs → 1.0`), any change to the sampler primitives (re-verify
  determinism and the NB variance recovery), or turning the `_EPS` value-clamp into a
  gradient-bounding device (currently deliberately not that — C-205).

---

## End of Contract

This document defines the **intended meaning** of `NBCore` and its module helpers.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
