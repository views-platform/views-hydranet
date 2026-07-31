# 02 — Design: isolate + calibrate the bulk body

## The structural frame (why this decomposition)
The outcome is a composite of (at least) three latent processes with different predictability/uncertainty:
- **Occurrence** (conflict vs none) — the *gate*. Level fully predictable; a knob exists (`pos_weight`).
  **FROZEN** here (dense-mse+wBCE `pos_weight=2` already ties white_ranger on Brier).
- **Bulk magnitude** — the *body* on the bottom ~97–99% of positives. A slowly-varying latent level (why
  persistence is strong); its expected value IS predictable once the tail is removed. **THIS program.**
- **Extreme tail** — top ~1–3%. Value irreducible ([[project_amount_ceiling_wall]]), only *risk* predictable
  ([[project_volatility_ceiling_predictable]]); must be carried as mass/width, not a point. **PARKED.**

Independence: gate ⊥ (bulk, tail); bulk~tail correlated (shared drivers) but we park the tail now.
Conflict history only (no new features yet).

## The problem, precisely
dense-mse fits the log-**median** → `expm1` → underestimates the tail-dominated mean → **timid**
(`ratio_med` 0.05–0.11). Every prior body loss went **timid** (mae/huber/shrinkage/pareto/NB) or **exploded**
(count_mean OOS collapse; quantile fan). The survey's decisive point: the body mean is un-calibratable
*only because* it's the mean of an **infinite-variance** tail (ξ≈0.8) — **remove the tail and the bulk mean
becomes finite, stable, learnable.**

## The MVP (two moves — the second needs the first)
1. **Stabilizer — outlier-robust per-cell winsorize/cap of the TARGET.** Cap `y_{t+1}` at `k ×` a robust
   running statistic of the cell's own recent positives (rolling median/MAD). Removes the infinite-variance
   tail from the training signal. **Necessary, not sufficient** — plain MSE on a capped target still gives
   the (capped) median, still biased low. It *permits* a lifter to run without exploding.
2. **Lifter — the magnitude dial.** A moderate-τ **log-space pinball** on the (capped) body: τ=0.5 = median
   (today), τ≈0.65–0.75 lifts toward the mean. **τ is the knob** (the body analog of `pos_weight`). Bounded
   (log space) ⇒ no count-space explosion; the cap stops it chasing extremes (why the lab's τ=0.99
   tail-pinball over-fired and wrecked Brier). Alternative lifter = `count_mean` on the capped target
   (proven to lift the mean; the cap is exactly what makes its exp-gradient stable). Decision: start with
   the **τ-pinball** (explicit dial); `count_mean`-on-capped as the fallback.

## Why this is new (not a re-run of a killed arm)
Winsorized-TARGET body loss has **never been tried** (zero "winsor" hits in either repo). Every prior tail
treatment pushed the tail *up* (pinball@0.99, count_mean) or *dampened* it (Basu, Pareto) — nobody excluded
the top 1–3% of *targets* so the bulk could calibrate. The lab's `count_mean` lifted (0.10→0.62) then
collapsed OOS — the winsorize is the specific new variable that should prevent that regime-overfit.

## Scope guards
Gate frozen · tail parked · conflict-history features only · one variable · measured strictly per `03`.
Success is bulk `ratio_med`→[0.7,1.3] at T=0 with Brier/CRPS/QS99 held; graduate on validation.
