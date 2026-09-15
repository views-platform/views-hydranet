# 02 — Design: occurrence dropout

**Derived from S1 (#313), not chosen.** Every parameter below cites the measurement in
`07_experiment_log.md` / `results/s1/error_profile_sb.json`. Where a choice is *not* derived from
data, it says so — that distinction is the point of writing this down.

## 1. The family — derived

S1's pre-registered rule selected **`occurrence_dropout`**: at h18 the model silences **99.6%** of
true events while firing on **0.003%** of true zeros — FN/FP = **36,870×**, with the dominant rate's
CV across 13 origins at **0.002**. The model does not jitter; it goes silent. So the training
corruption **removes events from the input**.

**Two hazards dissolve as a consequence:**

* It **cannot manufacture occurrence.** Dropout only removes. Skepticism-ledger item 1 (M45: AP loss
  scales with how much the model fires) has no path to fire through this design.
* It **cannot produce negative log-counts.** Dropout writes `0`, and `log1p(0) = 0` is on-manifold.
  The off-manifold hazard that made the naive Gaussian design dangerous simply does not exist here.

## 2. The rate — derived

**Not** the raw FN rate. At h1 the model already silences 81% of events with barely any recursion —
that is its own timid gate, not rollout damage. The rollout-induced part is the growth on top, and it
fits a **constant per-step dropout**:

| h | survival of h1-kept events | implied p |
|---|---|---|
| 6 | 0.30201 | 0.2129 |
| 12 | 0.08134 | 0.2040 |
| **18** | **0.02169** | **0.2018** |
| 24 | 0.00456 | 0.2090 |
| 36 | 0.00055 | 0.1932 |

**p = 0.204**, spread across horizons 0.020. Fitting `S(h) = (1−p)^(h−1)` with that single constant
reproduces the observed survival with relative residuals of **0.3% (h12), 5.0% (h18), 5.7% (h6)**,
degrading to 14.9% at h24 and 38.1% at h36.

⚠️ **The h36 residual is not evidence against the fit.** Survival there is 0.055%, so the relative
residual is a ratio of two near-zero numbers — the same instability that makes S1's h36 *magnitude*
column (n=3 cells) unusable. At **h18, the horizon the decision rule keys on, the fit is within 5%.**

## 3. The accumulation — ⚠️ A DESIGN CHOICE, NOT A MEASUREMENT

`SanchezGonzalez2020` found random-walk (accumulating) noise beat i.i.d., because rollout error
accumulates — and S1's monotone FN curve says ours accumulates too. So a dropped cell **stays
dropped**.

But accumulation cannot run unbounded here, and the reason is structural:

> **A training window is 348 steps. Deployment rolls 36.** Accumulating at p=0.204 over 348 steps
> leaves `0.796^347 ≈ 10⁻³⁵` of the input — the model would train on an empty field.

So dropout **accumulates within a segment and resets between segments**, with the segment length
taken from the config's `time_steps` (**36** — the deployment horizon), read explicitly at the call
site rather than hardcoded. Each segment then mimics one deployment rollout.

**This is the one parameter S1 does not determine.** The measurement gives the *rate*; the
*reset cadence* is forced by the window/horizon mismatch and chosen to match the deployment horizon.
Flagged here for S3's audit rather than buried.

## 4. What is not touched

- **Statics.** Geometry is *"always the true values, never sampled"*. ⚠️ No arm in this fleet has any
  (`static_channels` is empty — measured at S0), so this guard **cannot be exercised by any real
  config** and must be tested against a **synthetic** one. C-309.
- **The Stage-5 diagnostic biopsy** (`training_engine.py:920-926`) — a clean-performance probe.
- **The pushforward self-fed forward** (`training_engine.py:695`) — **the deliberate fork, decided:
  NOT noised.** Its input is already the model's own output, which is *already* silenced by exactly
  the process being modelled. Dropping again would double-count the error and make the pushforward
  arm a different experiment from the one #289 pre-registered.

## 5. Configuration

One field, `input_noise_dropout: float | None`, default `None` = off = byte-identical.
`gt=0.0` rather than `ge=0.0`: a dropout of 0.0 is a no-op indistinguishable from off, which is
precisely the **C-324** inert-knob signature that cost 276 minutes of GPU on #308. The same reasoning
`ss_feedback_grad_clip` records.

The scale is a **config field from day one** — the hardcoded `0.5` flip probability in `random_flips`
is open debt **C-85** and a second instance is not acceptable.
