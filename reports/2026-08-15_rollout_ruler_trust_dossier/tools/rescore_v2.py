#!/usr/bin/env python3
"""rescore_v2.py — re-score the surviving cubes on the trustworthy ruler, and rule.

Epic #263 / S6 (#270). Mirrors ``score_v2_horizons.score_horizons_v2``'s loop exactly — same
cross-arm support intersection (G4), same ``_metric_row``, same per-arm ``del g`` OOM guard
(C-252) — with the **FAO-02 climatology** injected where ``add_persistence`` would go. It then
joins the three things the ruler previously lacked:

* ``crpss_vs_clim`` — a skill score, because ``crps_all`` finally has a denominator (S3);
* ``zero_share_of_gap`` — how much of the gap is confident zeros (S1);
* ``delta_AP`` — whether the occurrence ranking moved the same way.

Every emitted headline row passes ``require_headline_columns``, so a bare ``crps_all`` cannot
be reported (C-219). The verdict token comes from the **pre-registered** rule in
``05_analysis_plan.md`` via ``verdict_token``; it is applied, not chosen.

Usage:
    python tools/rescore_v2.py <arm>=<pred_dir> ... [--targets sb,ns,os]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parents[3]
_V2T = _HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "tools"
_ROLLT = _HN / "reports" / "2026-07-25_t0_rollout_skill_dossier" / "tools"
_LODE = _HN / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"

for _p in (str(_LODE), str(_ROLLT), str(_V2T), str(_HN / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rollout_ruler_core import (  # noqa: E402
    climatology_resample,
    crps_gap_decomposition,
    crps_skill_score,
    require_headline_columns,
    verdict_token,
)

HORIZONS = (1, 6, 12, 18, 24, 30, 36)  # pre-registered in 05_analysis_plan.md
CLIM = "climatology"
CI_HORIZONS = (1, 18, 36)  # where the origin-block CI is computed (D5 needs it at h=36)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rescore(registry, targets, horizons=HORIZONS, n_samples=64, seed=42, window_anchor=456):
    """One row per (arm, target, horizon), including the climatology arm."""
    v2 = _load("score_v2_horizons", _V2T / "score_v2_horizons.py")
    roll = _load("rollout_skill_score", _ROLLT / "rollout_skill_score.py")
    ruler = _load(
        "v2_ruler",
        _HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_ruler.py",
    )
    truth_parquet = str(ruler.V2_TRUTH)

    rows = []
    for tgt in targets:
        lr_full = f"lr_{tgt}_best"
        # G4: identical (origin, cell) support across ALL arms at EVERY horizon, from
        # identifiers only (no cube load).
        support = sorted(
            set.intersection(
                *[roll._support_keys(e[1], e[2].format(t=tgt), horizons) for e in registry]
            )
        )
        if not support:
            raise ValueError(f"[{tgt}] EMPTY cross-arm support — FAIL")
        n_origins = len({m0 for (m0, _u) in support})

        months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
        tmap = roll._truth_map(truth_parquet, lr_full, months)
        # the climatology needs PRE-origin truth too
        anchor_ends = {456 if window_anchor is None else int(window_anchor)}
        anchor_ends |= {m0 - 1 for (m0, _u) in support}
        hist_months = {m for e in anchor_ends for m in range(e - 36 + 1, e + 1)}
        hist = roll._truth_map(truth_parquet, lr_full, hist_months)

        per_arm = {}
        for entry in list(registry):
            label, pdir, lr_tmpl, by_tmpl = entry[0], entry[1], entry[2], entry[3]
            g = roll.gather_all_horizons(
                pdir, lr_tmpl.format(t=tgt), by_tmpl.format(t=tgt) if by_tmpl else None
            )
            thr = v2._resolve_thresh(entry[4] if len(entry) > 4 else None, tgt)
            for h in horizons:
                r = v2._metric_row(label, g, support, tmap, tgt, h, thr)
                r["n_origins"] = n_origins
                per_arm.setdefault(label, {})[h] = r
            del g  # C-252: one arm's cubes in RAM at a time

        # CANONICAL convention by default: pool fixed at train_end (456 for pgm calibration),
        # matching views-baseline's ConflictologyModel. window_anchor=None gives the sliding
        # alternative — the open question raised as views-baseline #82.
        gc_ = climatology_resample(
            hist, support, horizons, n_samples=n_samples, seed=seed, window_anchor=window_anchor
        )
        for h in horizons:
            r = v2._metric_row(CLIM, gc_, support, tmap, tgt, h, None)
            r["n_origins"] = n_origins
            per_arm.setdefault(CLIM, {})[h] = r
        del gc_

        for label, by_h in per_arm.items():
            for h, r in by_h.items():
                ref = per_arm[CLIM][h]
                if label == CLIM:
                    # The reference against itself: zero gap by construction, and its CI
                    # trivially contains zero. These are measurements, not defaults.
                    r.update(
                        crpss_vs_clim=0.0,
                        zero_share_of_gap=0.0,
                        delta_AP=0.0,
                        ci_lo=0.0,
                        ci_hi=0.0,
                        ci_excludes_zero=False,
                    )
                else:
                    d = crps_gap_decomposition(r, ref)
                    r["crpss_vs_clim"] = crps_skill_score(
                        r["crps_all"], ref["crps_all"], ref_n_samples=n_samples
                    )
                    r["zero_share_of_gap"] = d["zero_share"]
                    r["delta_AP"] = r["AP"] - ref["AP"]
                    r["zero_part"] = d["zero_part"]
                    r["event_part"] = d["event_part"]
                require_headline_columns(r, where=f"{label} {tgt} h={h}")
                rows.append(r)

        # --- origin-block CI on the CRPS differential vs climatology (C-221/C-254) ---------
        # EVERY horizon gets one. A rule that cannot receive an input is a rule biased by
        # omission, and restricting this to three horizons is what left 48 of 84 rows unable
        # to reach REAL. Hoisting the climatology (below) makes all seven cheaper than three
        # were, so there is no cost argument for partial coverage.
        _add_origin_block_ci(rows, registry, tgt, tmap, hist, support, horizons, n_samples, seed)
    return rows


def _add_origin_block_ci(rows, registry, tgt, tmap, hist, support, horizons, n_samples, seed):
    """Fill ci_lo / ci_hi / ci_excludes_zero for EVERY (arm, horizon) row of this target.

    Two things this deliberately does not do, both of which it used to:

    * **It does not build the climatology per arm.** Under a fixed ``window_anchor`` the draws
      depend only on the cell, so they are identical across arms *and* horizons
      (`climatology_resample` stores the same array object for every h). Rebuilding it inside
      the arm loop was ~83% of this function's cost for no difference in result.
    * **It does not derive the cell universe from one arm's coverage.** That is C-277: the CI
      would then describe a different population than the point estimate it annotates, which
      is computed on the cross-arm intersection (G4). The caller's `support` is passed in.
    """
    mde = _load("mde", Path(__file__).resolve().parent / "mde.py")
    gw = _load("gw_stratified", _V2T / "gw_stratified.py")
    from lodestar_score import crps_ensemble

    # One climatology for this target, on the INTERSECTED support, for all horizons at once.
    gclim = climatology_resample(
        hist, support, horizons, n_samples=n_samples, seed=seed, window_anchor=456
    )
    by_origin = {}
    for m0, u in support:
        by_origin.setdefault(m0, []).append(u)

    for h in horizons:
        for label, pdir, *_ in registry:
            a = mde.per_origin_crps(Path(pdir), tgt, h, tmap)
            d, gid = [], []
            for m0, units in by_origin.items():
                if m0 not in a:
                    continue
                cube, tt, uu = a[m0]
                pos = {int(x): i for i, x in enumerate(uu)}
                idx = [pos[u] for u in units if u in pos]
                if not idx:
                    continue
                keep = [u for u in units if u in pos]
                clim = np.stack([gclim[(m0, h, int(u))][0] for u in keep])
                dd = crps_ensemble(tt[idx], cube[idx]) - crps_ensemble(tt[idx], clim)
                d.append(dd)
                gid.append(np.full(dd.size, m0))
            if not d:
                raise ValueError(f"[{tgt}] {label} h={h}: no cells on the intersected support")
            d, gid = np.concatenate(d), np.concatenate(gid)
            lo, hi, _ = gw._bootstrap_mean_ci(d, gid, 2000, seed, 0.90, "origin")
            for r in rows:
                if r["model"] == label and r["target"] == tgt and r["h"] == h:
                    r["ci_lo"], r["ci_hi"] = lo, hi
                    r["ci_excludes_zero"] = bool(lo > 0 or hi < 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="arm=pred_dir")
    ap.add_argument("--targets", default="sb,ns,os")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    registry = []
    for spec in args.arms:
        lab, _, pth = spec.partition("=")
        registry.append((lab, pth, "lr_{t}_best", "by_{t}_best"))

    rows = rescore(registry, tuple(args.targets.split(",")))

    # Persist the verdict. It is the epic's deliverable (SCOPE.md: "a number and a verdict
    # token"), and until now it existed only on stdout — so the published tables were hand
    # transcriptions with nothing to diff or re-derive against. Computing it here also runs
    # the decision rule over EVERY row, so a missing input raises at write time instead of
    # silently defaulting inside some downstream consumer.
    for r in rows:
        r["verdict"] = verdict_token(
            zero_share=float(r["zero_share_of_gap"]),
            delta_ap=float(r["delta_AP"]),
            crpss=float(r["crpss_vs_clim"]),
            ci_excludes_zero=bool(r["ci_excludes_zero"]),
        )

    cols = sorted({k for r in rows for k in r})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rescore.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if not np.isfinite(r["crps_all"])]
    if bad:
        raise SystemExit(f"D1 FAIL: {len(bad)} rows have a non-finite crps_all")

    print(f"wrote {out / 'rescore.csv'}  ({len(rows)} rows)")
    print(f"\nheadline — target sb, vs the FAO-02 climatology ({rows[0]['n_origins']} origins):")
    print(
        f"  {'arm':<18} {'h':>3} {'crps_all':>9} {'CRPSS':>8} {'zero_sh':>8} "
        f"{'dAP':>7} {'verdict':>12}"
    )
    for r in sorted(
        (x for x in rows if x["target"] == "sb" and x["model"] != CLIM),
        key=lambda x: (x["model"], x["h"]),
    ):
        if r["h"] not in (1, 18, 36):
            continue
        print(
            f"  {r['model']:<18} {r['h']:>3} {r['crps_all']:>9.4f} "
            f"{r['crpss_vs_clim']:>8.4f} {r['zero_share_of_gap']:>8.3f} "
            f"{r['delta_AP']:>7.3f} {r['verdict']:>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
