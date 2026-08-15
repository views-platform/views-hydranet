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
    reference_sample_width,
    require_headline_columns,
    verdict_token,
)

HORIZONS = (1, 6, 12, 18, 24, 30, 36)  # pre-registered in 05_analysis_plan.md
CLIM = "climatology"
CLIM_SEED = 0  # pre-registered in 05_analysis_plan.md; NOT a free parameter
CLIM_ANCHOR = 456  # pgm calibration train_end — the canonical ConflictologyModel pool


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _arm_sample_width(registry) -> int:
    """The reference's S, read off the arms' cubes. Header-only — no cube is loaded."""
    audit = _load("partition_audit", Path(__file__).resolve().parent / "partition_audit.py")
    return reference_sample_width(
        (shp[1] for e in registry for shp in audit.cube_shapes(Path(e[1])).values()),
        where="the scored arms",
    )


def rescore(registry, targets, horizons=HORIZONS, seed=CLIM_SEED, window_anchor=CLIM_ANCHOR):
    """One row per (arm, target, horizon), including the climatology arm.

    There is deliberately **no ``n_samples`` parameter.** The reference's draw count is derived
    from the arms' cube width (``reference_sample_width``), because the two must agree:
    ``crps_ensemble``'s ``2/(m*m)`` normalisation leaves an ``O(1/S)`` bias that cancels
    between equal-width ensembles and does not cancel between unequal ones, and the reference's
    AP is quantised to ``S + 1`` rank levels. The shipped 2026-08-15 board drew the reference
    at 64 against arms at 16; both effects biased it toward ARTIFACT, the direction of its own
    conclusion. Making it a parameter is what let them diverge, so it is not one.
    """
    v2 = _load("score_v2_horizons", _V2T / "score_v2_horizons.py")
    roll = _load("rollout_skill_score", _ROLLT / "rollout_skill_score.py")
    ruler = _load(
        "v2_ruler",
        _HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_ruler.py",
    )
    truth_parquet = str(ruler.V2_TRUTH)
    n_samples = _arm_sample_width(registry)

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
        anchor_ends = {CLIM_ANCHOR if window_anchor is None else int(window_anchor)}
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
        _add_origin_block_ci(
            rows,
            registry,
            tgt,
            tmap,
            hist,
            support,
            horizons,
            n_samples,
            seed,
            window_anchor,
            v2,
        )
    return rows


def _add_origin_block_ci(
    rows, registry, tgt, tmap, hist, support, horizons, n_samples, seed, window_anchor, v2
):
    """Fill ci_lo / ci_hi / ci_excludes_zero for EVERY (arm, horizon) row of this target.

    Two things this deliberately does not do, both of which it used to:

    * **It does not build the climatology per arm.** Under a fixed ``window_anchor`` the draws
      depend only on the cell, so they are identical across arms *and* horizons
      (`climatology_resample` stores the same array object for every h). Rebuilding it inside
      the arm loop was ~83% of this function's cost for no difference in result.
    * **It does not derive the cell universe from one arm's coverage.** That is C-277: the CI
      would then describe a different population than the point estimate it annotates, which
      is computed on the cross-arm intersection (G4). The caller's `support` is passed in.
    * **It does not choose its own climatology convention or its own threshold gate.** Both
      used to be re-decided here — `window_anchor` was hardcoded to 456 while the point
      estimate honoured the caller's, and the registry was unpacked as `label, pdir, *_`,
      discarding the threshold spec that `rescore` applies at emit time. Either one produces a
      CI describing a strictly different forecast than the number it annotates, which is the
      same class as C-277 one layer down. Every parameter now arrives from the caller.
    """
    mde = _load("mde", Path(__file__).resolve().parent / "mde.py")
    gw = _load("gw_stratified", _V2T / "gw_stratified.py")
    from lodestar_score import crps_ensemble

    # One climatology for this target, on the INTERSECTED support, for all horizons at once.
    gclim = climatology_resample(
        hist, support, horizons, n_samples=n_samples, seed=seed, window_anchor=window_anchor
    )
    by_origin = {}
    for m0, u in support:
        by_origin.setdefault(m0, []).append(u)

    for h in horizons:
        for entry in registry:
            label, pdir = entry[0], entry[1]
            if v2._resolve_thresh(entry[4] if len(entry) > 4 else None, tgt) is not None:
                # `rescore` applies a threshold gate at emit time; `per_origin_crps` reads raw
                # y_pred.npy and never loads the gate channel, so it cannot. Scoring a th_gated
                # arm here would annotate a gated point estimate with an ungated CI — two
                # different forecasts in one row, the C-277 class. Latent (main() builds only
                # 4-tuples), so it fails loud rather than growing gating support speculatively.
                raise NotImplementedError(
                    f"{label}: the origin-block CI cannot honour a threshold gate — "
                    "per_origin_crps scores the ungated body. Teach it the gate channel "
                    "before scoring a th_gated arm, or the CI will describe a different "
                    "forecast than the estimate it annotates."
                )
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
            if not (np.isfinite(lo) and np.isfinite(hi)):
                # bool(nan > 0 or nan < 0) is False, which would record "could not measure" as
                # "measured, and it straddles zero" — indistinguishable in the CSV, and exactly
                # the not-measured-is-not-no defect verdict_token's signature exists to delete.
                raise ValueError(
                    f"[{tgt}] {label} h={h}: the origin-block CI is non-finite "
                    f"([{lo}, {hi}]), so ci_excludes_zero is not measurable. A NaN reached the "
                    "differential — check the cubes and the truth lookup."
                )
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
    #
    # `prereg` marks the ONE cell the rule was pre-registered for. 05_analysis_plan.md:95 reads
    # "Decision rule — pre-committed. Evaluated at **sb, h36**". Every other cell is the same
    # rule applied to data it was not registered against, with no multiplicity control across
    # 84 of them — informative, but not a pre-registered result, and the shipped 2026-08-15 log
    # read the full grid as one ("a monotone decay of genuine skill") without saying so. The
    # column exists so a reader cannot fail to see which row carries the pre-registration.
    for r in rows:
        if r["model"] == CLIM:
            # The reference does not rule on itself. Its crpss/ΔAP/CI are the self-comparison
            # constants set above, so a token here would be arithmetic about a definition —
            # and 21 of the shipped board's 25 UNDECIDABLE rows were exactly that.
            r["verdict"] = "reference"
            r["prereg"] = False
            continue
        r["verdict"] = verdict_token(
            zero_share=float(r["zero_share_of_gap"]),
            delta_ap=float(r["delta_AP"]),
            crpss=float(r["crpss_vs_clim"]),
            ci_excludes_zero=bool(r["ci_excludes_zero"]),
        )
        r["prereg"] = bool(r["target"] == "sb" and int(r["h"]) == 36)

    # D1 BEFORE the write, so a failing falsifier cannot leave a published artifact behind
    # (csv_decompose.py:129 already gets this ordering right; this one did not).
    bad = [r for r in rows if not np.isfinite(r["crps_all"])]
    if bad:
        raise SystemExit(f"D1 FAIL: {len(bad)} rows have a non-finite crps_all")

    cols = sorted({k for r in rows for k in r})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rescore.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

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
