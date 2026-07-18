# 06 — Glossary

**Bulk** — the bottom ~97–99% of NON-ZERO observations; the "normal range" of conflict magnitude. Where we
want the expected-value (body) calibrated. Defined for the metric by `truth ≤ cut` (98th-pct of positive
training truths; also 97/99).

**Tail (extreme)** — top ~1–3% of non-zero (`truth > cut`). Value irreducible from covariates (amount-ceiling
WALL); only its *risk* is predictable. PARKED — carried later as mass/width, not chased as a point.

**Gate** — the occurrence head P(y>0). FROZEN (dense-mse+wBCE `pos_weight=2`). Its Brier must not regress.

**Body / bulk magnitude** — the predicted expected value E[y] on positive cells. Currently TIMID.

**Timid prophet** — under-fires positives (predicts ~5–11% of truth; `ratio_med` 0.05–0.11) → wins the
zero-dominated pooled CRPS by going quiet on positives. The failure this program targets.

**Winsorize / cap (stabilizer)** — replace the target's outliers with a robust per-cell cap so the
infinite-variance tail can't drag/explode the fit. Necessary, not sufficient (doesn't itself lift the bias).

**Magnitude dial (lifter)** — the tunable that raises predicted body magnitude. MVP: moderate-τ log-space
pinball, **τ the knob** (0.5=median/timid, ↑ lifts toward mean). Fallback: `count_mean` on the capped target.

**`ratio_med`** — median over bulk-positive cells of E[y]/truth. THE headline (per-cell → un-cancellable).
Target [0.7, 1.3]. NOT MCR (pooled mean — banned, it hid the failure ×3).

**`within2x` / `within2x_rescaled`** — % of bulk-positive cells with E[y]/truth in [0.5, 2]; rescaled divides
by the median ratio to isolate scatter from bias (genuine sharpness vs a global rescale).

**Guardrails** — Brier (gate), CRPS, QS99 at T=0; must not regress vs baseline (else it's over-firing).

**Same-seed A/B** — baseline dense-mse pw2 vs +winsorize+dial, identical seed/config; the delta is the effect.

**T=0** — the first forecast month per origin. The rollout (steps 2–36) is NEVER pooled into a T=0 read
(the trap that faked the quantile "degeneration").
