"""AMENDMENT 1 to `../2026-10-26_instrument_battery_prereg.md`: F-4 fired, and its bar was
unsatisfiable.

F-4 asked for |sampled - analytic| / |analytic| <= 1% at K = 4096. But the sampled estimator is a
mean of K squared differences, and its OWN relative standard error at K = 4096 is ~2% -- larger
than the bar it had to clear. ⛔ No correct closed form could have passed. Same defect class as the
structure-penalty prereg's P-1, whose +-8.6% tolerance was finer than its grid's 22.7% resolution.
Fourth instance. Logged as W-69.

⛔ The test is REPLACED, not weakened, and the replacement is recorded rather than applied
silently. A mis-specified threshold cannot tell "noisy estimator" from "wrong formula".
CONVERGENCE can: under a correct closed form the relative error must fall like 1/sqrt(K). If it
plateaus at any level, the formula is wrong and every VS-p number is VOID as §6 says.

Deliberately self-contained -- it re-derives the three pieces it needs rather than importing from
`instrument_battery.py`, which does its work at module level. Importing it would run the whole
battery as a side effect of checking one identity.
"""

from pathlib import Path

import numpy as np

CTL = Path(
    "/home/simon/views-diagnostics/hf_control_20261025/diagnostics/251026_1043/"
    "02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
)
OFFSETS = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0 / np.sqrt(2))]   # the three shortest lags


def control_field() -> tuple[np.ndarray, np.ndarray]:
    """-> (composed forecast, truth) in COUNTS, [steps*targets, 32, 32]."""
    z = np.load(CTL)
    d = {k: z[k].astype(np.float64) for k in ("y_reg", "y_hat_reg", "y_hat_cls")}
    gate, body, truth = (np.moveaxis(d[k], -1, 1).reshape(-1, 32, 32)
                         for k in ("y_hat_cls", "y_hat_reg", "y_reg"))
    return gate * np.expm1(body), np.expm1(truth)


def main() -> None:
    cp, ct = control_field()
    v = 0.25 * ct.var()
    sd = np.sqrt(v)
    rng = np.random.default_rng(7)

    print("AMENDMENT 1 — does the sampled estimate CONVERGE to the closed form like 1/sqrt(K)?")
    print("  identity under test:  E|X_i - X_j|^2 = (m_i - m_j)^2 + v_i + v_j")
    print(f"  {'K':>8s} {'rel err':>10s}   {'ratio to previous':>20s}")
    prev = None
    for k_draws in (256, 1024, 4096, 16384, 65536):
        num = den = 0.0
        for dr, dc, w in OFFSETS:
            xa = cp[:, dr:, dc:]
            xb = cp[:, : cp.shape[1] - dr, : cp.shape[2] - dc]
            run = np.zeros_like(xa)
            for k in range(k_draws):
                diff = ((xa + rng.normal(0, sd, xa.shape))
                        - (xb + rng.normal(0, sd, xb.shape)))
                run += (diff * diff - run) / (k + 1)
            ana = (xa - xb) ** 2 + 2.0 * v
            num += w * float(np.abs(run - ana).mean())
            den += w * float(np.abs(ana).mean())
        rel = num / den
        ratio = "" if prev is None else f"{prev / rel:.2f}x  (2.00 expected)"
        print(f"  {k_draws:8d} {rel:10.4%}   {ratio:>20s}")
        prev = rel
    print("\n  A ratio of 2.00 per 4x in K is 1/sqrt(K) exactly. A PLATEAU would mean the closed")
    print("  form is wrong and every VS-p number in the battery is VOID.")


if __name__ == "__main__":
    main()
