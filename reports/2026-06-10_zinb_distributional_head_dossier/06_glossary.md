# 06 — Glossary

**Date:** 2026-06-10 · Append-only.

- **ZINB** — Zero-Inflated Negative Binomial: a count distribution mixing a point mass at 0 (probability π) with a negative-binomial body. Here: one likelihood over onset + magnitude.
- **π (gate)** — zero-inflation probability; here `1 − sigmoid(cls)`, reusing the classification head.
- **μ** — count-space NB mean; here `softplus` on the regression heads (sub-exponential by design).
- **θ** — NB dispersion (over-dispersion) parameter; MVP = one learnable scalar per target.
- **Hurdle** — the factorization `E[y] = P(y>0) · E[y | y>0]`; the onset gate × the positive body. Here it is the *structure* of the ZINB, not a separate config.
- **Count-target bridge** — recovering raw counts via `expm1(log1p y_true)` (exact, bounded target) to feed the count loss; never applied to predictions.
- **Explosion-check gate** — read-only 36-step `diagnose_io_gain` run on `E[y]`; a go/no-go, not a clamp.
- **MCR** — `mean(ŷ)/mean(y)`; magnitude-calibration **diagnostic** (→1), never the optimization target.
- **CRPS / QS99 / Brier** — FAO PRN-05 metrics: primary score / tail-sanity / onset-calibration.
- **Parity** — flag-off ⇒ byte-identical to the current head.
- **The two exits** — ship the ZINB head, or revert to `e029e63`.
- **CIRCLE** — the chair's brake word: stop, drop the proposal, return to the next checklist box.
