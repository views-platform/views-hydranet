"""Q0.1 and C.1-C.3 — does the model express escalation, and does freezing flatten it?

M55 showed the cell clamp is a PLACEMENT fix: it moves AP and Brier and leaves `size_ratio` at 0.
This asks the next question, which is about *dynamics* rather than level: can the model say which
places are getting WORSE, and does pinning the state destroy that ability?

Three measures, and the order matters:

* **Q0.1 direction skill (TRUTH-REFERENCED)** -- rank correlation between the predicted change
  ``mu(h) - mu(1)`` and the true change, on a FIXED COHORT of cells truly active at h1. It is
  computed first and **gates the other two**: if the model has no escalation skill to begin with,
  "freezing destroys it" is unanswerable, not a null. Reporting a null there would be the C-318
  mistake in a new costume -- a number produced where there is nothing to measure.
* **C.1 dispersion (INTERNAL)** -- spread across cells of the predicted log-ratio. Near zero means
  the whole field moves together and the model expresses no PER-CELL dynamics at all.
* **C.2/C.3** -- both of the above, per arm.

The cohort is fixed at h1 by TRUTH, not by the model's own firing, so no arm can change its own
denominator and nothing here is subject to the survivorship that C-319 records.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave1_data import (  # noqa: E402
    RAW,
    RESULTS,
    arm_fields,
    build_unit_grid,
    load_origins,
    load_truth,
    truth_vec,
)

HS = (6, 12, 18, 24, 36)
EPS = 1e-3  # small against mu on truly-active cells, which is O(0.1-10)


def direction_skill(mu_1, mu_h, truth_1, truth_h):
    """Spearman rho between predicted and true change on the cells active at h1.

    Returns ``(rho, n)``. ``rho`` is NaN when the cohort is too small or either series is constant
    -- a correlation is undefined there, and returning 0.0 would read as "no skill" when the truth
    is "no measurement".
    """
    from scipy.stats import spearmanr

    cohort = truth_1 > 0
    n = int(cohort.sum())
    if n < 10:
        return float("nan"), n
    dp = mu_h[cohort] - mu_1[cohort]
    dt = truth_h[cohort] - truth_1[cohort]
    if np.ptp(dp) == 0 or np.ptp(dt) == 0:
        return float("nan"), n
    return float(spearmanr(dp, dt).statistic), n


def dispersion(mu_1, mu_h, truth_1):
    """Spread of the predicted per-cell log-ratio on the h1-active cohort.

    Zero spread = every cell scaled by the same factor, i.e. no per-cell dynamics whatever the
    field-level trend does.

    ``EPS`` puts a small FLOOR on this: for cells whose ``mu`` is itself of order ``EPS`` the ratio
    is pulled toward 1, so a perfectly uniform rescaling reads ~0.002 rather than exactly 0. That
    is two orders of magnitude below the signal a real per-cell spread produces, but it means the
    measure should be read as "materially above the floor", never as "non-zero".
    """
    cohort = truth_1 > 0
    if cohort.sum() < 10:
        return float("nan")
    r = np.log((mu_h[cohort] + EPS) / (mu_1[cohort] + EPS))
    return float(np.std(r))


def arm_rows(arm_dir, origins, umap, tm, horizons=HS):
    """Pool the cohort across origins, then measure once per horizon."""
    acc = {h: {"mu1": [], "muh": [], "t1": [], "th": []} for h in horizons}
    for m0, units, _gate, mu in arm_fields(arm_dir, origins, umap):
        t1 = truth_vec(tm, m0, units, 0)  # truth at h=1
        for h in horizons:
            acc[h]["mu1"].append(mu[0])
            acc[h]["muh"].append(mu[h - 1])
            acc[h]["t1"].append(t1)
            acc[h]["th"].append(truth_vec(tm, m0, units, h - 1))
    out = {}
    for h in horizons:
        d = {k: np.concatenate(v) for k, v in acc[h].items()}
        rho, n = direction_skill(d["mu1"], d["muh"], d["t1"], d["th"])
        out[h] = {"rho": rho, "n_cohort": n, "dispersion": dispersion(d["mu1"], d["muh"], d["t1"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fullzero_fortytwo")
    ap.add_argument(
        "--arms", default="identity,identity_freezehidden,identity_freezecell,identity_freezeall"
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    origins = load_origins()
    umap = build_unit_grid(str(RAW))
    tm = load_truth(origins, (1,) + HS)
    recs = []
    for lb in [x.strip() for x in a.arms.split(",") if x.strip()]:
        d = RESULTS / f"bodymean_{a.model}_{lb}"
        if not d.is_dir():
            print(f"skip {lb}: no dump")
            continue
        try:
            rows = arm_rows(d, origins, umap, tm)
        except ValueError as exc:
            print(f"skip {lb}: {exc}")
            continue
        print(f"\n=== {a.model} / {lb} ===")
        print(f"{'h':>3} {'rho (Q0.1)':>12} {'dispersion':>12} {'cohort n':>10}")
        for h, r in rows.items():
            print(f"{h:>3} {r['rho']:>12.4f} {r['dispersion']:>12.4f} {r['n_cohort']:>10}")
            recs.append({"model": a.model, "arm": lb, "h": h, **r})
    if a.out and recs:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
