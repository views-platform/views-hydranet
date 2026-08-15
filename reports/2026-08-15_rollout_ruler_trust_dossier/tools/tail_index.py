#!/usr/bin/env python3
"""tail_index.py — the 9 C-224 numbers: T_u for one arm vs the FAO-02 climatology.

Epic #263 / S5 (#269). Emits ``diag_Tu`` for target `sb`, h in {1, 18, 36}, q in
{0.99, 0.995, 0.999} — the exact grid pre-registered in ``05_analysis_plan.md``. **No
optimisation over q**, no other arms, no bootstrap: the stopping condition is nine numbers in
one table (`SCOPE.md`).

Every column is ``diag_``-prefixed and the output carries ``role=DIAGNOSTIC``. It is reported
and never selected on; `verdict_token` reads no ``diag_*`` key and a test asserts that.

Usage:
    python tools/tail_index.py <arm>=<pred_dir> [--target sb]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
_LODE = _HN / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"
for _p in (str(_LODE), str(_HN / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from lodestar_score import crps_ensemble  # noqa: E402
from rollout_ruler_core import (  # noqa: E402
    climatology_resample,
    reference_sample_width,
    taillardat_index,
)

_spec = importlib.util.spec_from_file_location("mde", _HERE / "mde.py")
_mde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mde)

HORIZONS = (1, 18, 36)
QUANTILES = (0.99, 0.995, 0.999)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", help="arm=pred_dir")
    ap.add_argument("--target", default="sb")
    ap.add_argument("--out", default=str(_HERE.parent / "results" / "tail_index.md"))
    args = ap.parse_args()
    label, _, pred = args.arm.partition("=")
    pred = Path(pred)

    roll = _mde._load("rollout_skill_score", _mde._ROLL)
    v2 = _mde._load(
        "v2_ruler",
        _HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_ruler.py",
    )

    rows = []
    for h in HORIZONS:
        months = set()
        for odir in sorted(pred.glob("origin_*")):
            t = np.load(odir / f"lr_{args.target}_best" / "identifiers.npz")["time"]
            months.add(int(t.min()) + h - 1)
        tmap = roll._truth_map(str(v2.V2_TRUTH), f"lr_{args.target}_best", months)
        a = _mde.per_origin_crps(pred, args.target, h, tmap)

        support = [(m0, int(u)) for m0, (_c, _t, uu) in a.items() for u in uu]
        # CANONICAL, matching rescore_v2 and mde. This call previously omitted window_anchor
        # entirely and hardcoded n_samples=16 — so it selected the SLIDING pool and published
        # nine diag_Tu numbers against a different climatology than the headline, while the
        # experiment log recorded the divergence as fixed in "all three" tools. It was fixed in
        # two. climatology_resample now requires all three keywords for that reason.
        anchor_pool = set(range(_mde.CLIM_ANCHOR - 36 + 1, _mde.CLIM_ANCHOR + 1))
        hist_months = anchor_pool | {m for (m0, _u) in support for m in range(m0 - 36, m0)}
        hist = roll._truth_map(str(v2.V2_TRUTH), f"lr_{args.target}_best", hist_months)
        gathered = climatology_resample(
            hist,
            support,
            (h,),
            window=36,
            n_samples=reference_sample_width(
                (c.shape[1] for _m0, (c, _t, _u) in a.items()), where=label
            ),
            seed=_mde.CLIM_SEED,
            window_anchor=_mde.CLIM_ANCHOR,
        )

        cf, cg = [], []
        for m0, (cube, tt, uu) in a.items():
            clim = np.stack([gathered[(m0, h, int(u))][0] for u in uu])
            cf.append(crps_ensemble(tt, cube))
            cg.append(crps_ensemble(tt, clim))
        cf, cg = np.concatenate(cf), np.concatenate(cg)

        for q in QUANTILES:
            out = taillardat_index(cf, cg, q=q)
            out.update(h=h, target=args.target, model=label)
            rows.append(out)

    lines = [
        f"# C-224 tail diagnostic — `{label}` vs the FAO-02 climatology",
        "",
        f"Epic #263 / S5 (#269). Target `{args.target}`. Taillardat2023 §3.3: CRPS treated as a "
        "random variable, its **distribution** compared rather than its expectation.",
        "",
        "> **`diag_Tu` is not used in any decision rule in this dossier.** It is a DIAGNOSTIC. "
        "`verdict_token` reads no `diag_*` key, and "
        "`test_no_diag_column_reaches_the_decision_rule` asserts that by inspecting its source.",
        "",
        "| h | q | u | m (model) | m (ref) | gamma model | gamma ref | **diag_Tu** |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        tu = "n/a" if not np.isfinite(r["diag_Tu"]) else f"**{r['diag_Tu']:+.4f}**"
        cap = " †" if r.get("diag_gamma_at_pwm_ceiling") else ""
        lines.append(
            f"| {r['h']} | {r['diag_q']} | {r['diag_u']:.4g} | {r['diag_m_model']} | "
            f"{r['diag_m_ref']} | {r['diag_gamma_model']:.3f}{cap} | "
            f"{r['diag_gamma_ref']:.3f}{cap} | {tu} |"
        )
    lines += [
        "",
        "`T_u > 0` means the model's CRPS tail is *further* from a fitted GPD than the "
        "reference's. **It does NOT mean better.** Taillardat §3.3 is explicit that an "
        "inflated, mis-calibrated 'extremist' forecaster scores HIGH — pinned by the green "
        "test `test_extremist_forecast_gets_a_HIGH_index`, whose passing condition is that "
        "this metric is gameable.",
        "",
        "`n/a` means the index is **undefined**, not bad: the pooled threshold left one arm "
        "with fewer than 50 exceedances, so no GPD could be fitted to it.",
        "",
        "**† marks a row where the fitted `gamma` is at the PWM saturation ceiling.** "
        "`gpd_pwm_fit` is a probability-weighted-moment estimator: it is consistent only for "
        "`gamma < 0.5`, and its `a1` moment ceases to exist at `gamma >= 1`. Measured on exact "
        "GPD quantiles it saturates — a true shape of 0.9/1.0/1.1/1.3 fits to "
        "0.86/0.92/0.96/0.99. A marked `gamma` is therefore a **lower bound**, not an "
        "estimate, and cannot distinguish a heavy tail from an infinite-mean one (`gamma >= 1`) "
        "— which is the regime this diagnostic exists to probe. A scipy MLE fit valid across "
        "this range already exists at "
        "`reports/2026-07-15_volatility_ceiling_dossier/tools/s5_tail.py::gpd_xi`; nothing "
        "cross-checks the two (registered).",
        "",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {args.out}  ({len(rows)} rows)")
    for r in rows:
        print(
            f"  h={r['h']:>2} q={r['diag_q']}  m={r['diag_m_model']:>5}/{r['diag_m_ref']:<5} "
            f"Tu={r['diag_Tu']:+.4f}"
            if np.isfinite(r["diag_Tu"])
            else f"  h={r['h']:>2} q={r['diag_q']}  UNDEFINED — {r['reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
