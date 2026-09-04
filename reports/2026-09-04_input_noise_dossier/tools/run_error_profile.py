#!/usr/bin/env python3
"""run_error_profile.py — S1 (#313): the model's free-running error, per horizon and origin.

Feeds the pre-registered S1 -> S2 selection rule in ``scripts/input_noise_gate.py``. Reuses the
frozen rollout-ruler machinery (``gather_all_horizons``, ``_support_keys``, ``_truth_map``) rather
than re-deriving support or truth, so this sits on the same support the screens score on.

Usage:
    run_error_profile.py --pred-dir <predictions_*> [--target sb] [--horizons 1,6,12,18,24,36]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_HN / "scripts"))
sys.path.insert(0, str(_HN / "reports" / "2026-07-25_t0_rollout_skill_dossier" / "tools"))
sys.path.insert(0, str(_HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools"))

import error_profile as ep  # noqa: E402
from input_noise_gate import MAX_CV, cv, rule_md5, select_design  # noqa: E402
from rollout_skill_score import _support_keys, _truth_map, gather_all_horizons  # noqa: E402

_TRUTH = (
    _HN / "reports/2026-07-28_datafactory_migration_dossier/tools/v2_truth"
    "/calibration_datafactory_df.parquet"
)
#: h18 is the primary horizon the decision rule keys on (05_analysis_plan.md §4).
PRIMARY_H = 18


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--target", default="sb")
    ap.add_argument("--horizons", default="1,6,12,18,24,36")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    horizons = tuple(int(h) for h in a.horizons.split(","))
    if PRIMARY_H not in horizons:
        raise SystemExit(f"h{PRIMARY_H} is the primary horizon and must be measured")
    lr, by = f"lr_{a.target}_best", f"by_{a.target}_best"

    support = _support_keys(a.pred_dir, lr, horizons=horizons)
    if not support:
        raise SystemExit("empty support — no cell is present at every requested horizon")
    g = gather_all_horizons(a.pred_dir, lr, by)
    months = {m0 + h - 1 for (m0, _) in support for h in horizons}
    tmap = _truth_map(str(_TRUTH), lr, months)

    per_h = {}
    for h in horizons:
        rates = ep.origin_rates(ep.per_cell(g, support, tmap, h), h)
        fn = [r.fn_rate for r in rates]
        fp = [r.fp_rate for r in rates]
        per_h[h] = {
            "n_origins": len(rates),
            "n_cells": rates[0].n_cells if rates else 0,
            "act_true": _mean([r.act_true for r in rates]),
            "fn_rate": _mean(fn),
            "fp_rate": _mean(fp),
            "fn_rate_hard": _mean([r.fn_rate_hard for r in rates]),
            "mag_err_median": _mean([r.mag_err_median for r in rates]),
            "cv_fn": cv(fn),
            "cv_fp": cv(fp),
            "per_origin": [r.__dict__ for r in rates],
        }

    p = per_h[PRIMARY_H]
    dominant_cv = p["cv_fn"] if p["fn_rate"] >= p["fp_rate"] else p["cv_fp"]
    verdict = select_design(p["fn_rate"], p["fp_rate"], dominant_cv)

    out = {
        "pred_dir": a.pred_dir,
        "target": a.target,
        "horizons": list(horizons),
        "primary_h": PRIMARY_H,
        "rule_md5": rule_md5(),
        "max_cv": MAX_CV,
        "dominant_cv": dominant_cv,
        "verdict": verdict,
        "per_horizon": per_h,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    print(
        f"{'h':>4}{'act_true':>11}{'FN rate':>10}{'FP rate':>10}{'FN hard':>10}"
        f"{'mag err':>10}{'CV(FN)':>9}{'CV(FP)':>9}"
    )
    for h in horizons:
        r = per_h[h]
        print(
            f"{h:>4}{r['act_true']:>11.6f}{r['fn_rate']:>10.4f}{r['fp_rate']:>10.6f}"
            f"{r['fn_rate_hard']:>10.4f}{r['mag_err_median']:>10.4f}"
            f"{r['cv_fn']:>9.3f}{r['cv_fp']:>9.3f}"
        )
    print(
        f"\nPRIMARY h{PRIMARY_H}: FN={p['fn_rate']:.4f} FP={p['fp_rate']:.6f} "
        f"dominant CV={dominant_cv:.3f} (gate at {MAX_CV})"
    )
    print(f"DESIGN: {verdict['design']}   STOP={verdict['stop']}")
    print(f"  {verdict['reason']}")
    return 3 if verdict["stop"] else 0


def _mean(xs):
    import math

    v = [x for x in xs if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    sys.exit(main())
