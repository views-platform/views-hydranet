#!/usr/bin/env python3
"""mde.py — the minimum detectable effect of the origin-block bootstrap at P = 13.

Epic #263 / S4 (#268). **C-254's real content.** Giacomini & White's asymptotics are in the
number of out-of-sample forecast *periods* ``P``, not the cell cross-section — so despite
~170k cell-months per horizon, the effective inferential sample is **13 origins**. Worse, at a
fixed ``h`` those 13 origins contribute 13 *adjacent* months, so the origin-block bootstrap
fixes within-origin dependence but not between-origin serial dependence.

The honest response is to **report the MDE, not to fix it** (`SCOPE.md` #12: at P=13 an HAC
regression would be *less* honest, not more). This tool computes it from the real per-origin
CRPS differentials of a headline arm pair, so a null result can be read as "no difference"
or "no power" — the distinction C-2/C-254 exist to preserve.

Method: per-cell CRPS for each arm at one horizon, differenced; the differential is averaged
within each origin; then a block bootstrap over the 13 origin means gives the null CI. The MDE
is the half-width of that CI — the smallest true effect that would clear zero at the stated
level.

Memory: cubes are memmapped and only the rows for the requested horizon are read, so this
touches ~13k x 16 floats per origin, not the 471,960-row cube.

Usage:
    python tools/mde.py <armA>=<pred_dir> <armB>=<pred_dir> [--target sb] [--h 36]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parents[3]
_LODE = _HN / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"
_GW = _HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "tools" / "gw_stratified.py"
_ROLL = (
    _HN / "reports" / "2026-07-25_t0_rollout_skill_dossier" / "tools" / "rollout_skill_score.py"
)

if str(_LODE) not in sys.path:
    sys.path.insert(0, str(_LODE))
from lodestar_score import crps_ensemble  # noqa: E402

# The climatology's convention, identical to rescore_v2's. Kept as named constants in both
# rather than passed between them: these two tools are independent entry points, and the
# invariant that matters is that they name the same values, which a test asserts.
CLIM_SEED = 0  # pre-registered in 05_analysis_plan.md
CLIM_ANCHOR = 456  # pgm calibration train_end


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_origin_crps(pred_dir: Path, target: str, h: int, truth_map: dict) -> dict:
    """{m0: per-cell CRPS array} for one arm at one horizon. Reads only the horizon's rows."""
    out = {}
    for odir in sorted(glob.glob(str(pred_dir / "origin_*"))):
        tdir = Path(odir) / f"lr_{target}_best"
        ids = np.load(tdir / "identifiers.npz")
        time, unit = ids["time"], ids["unit"]
        m0 = int(time.min())
        sel = np.where(time == m0 + h - 1)[0]
        if sel.size == 0:
            continue
        cube = np.load(tdir / "y_pred.npy", mmap_mode="r")[sel]
        truth = np.array([truth_map.get((int(m0 + h - 1), int(u)), 0.0) for u in unit[sel]])
        out[m0] = (np.asarray(cube, dtype=float), truth, unit[sel])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs=2, help="two arm=pred_dir specs (candidate, reference)")
    ap.add_argument("--target", default="sb")
    ap.add_argument("--h", type=int, default=36)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    gw = _load("gw_stratified", _GW)
    roll = _load("rollout_skill_score", _ROLL)
    v2 = _load(
        "v2_ruler",
        _HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_ruler.py",
    )

    labels, dirs = [], []
    for spec in args.arms:
        lab, _, pth = spec.partition("=")
        labels.append(lab)
        dirs.append(Path(pth))

    # truth for the scored month of every origin at this horizon
    months = set()
    for odir in sorted(glob.glob(str(dirs[0] / "origin_*"))):
        t = np.load(Path(odir) / f"lr_{args.target}_best" / "identifiers.npz")["time"]
        months.add(int(t.min()) + args.h - 1)
    tmap = roll._truth_map(str(v2.V2_TRUTH), f"lr_{args.target}_best", months)

    a = per_origin_crps(dirs[0], args.target, args.h, tmap)
    if labels[1] == "climatology":
        # Build the FAO-02 reference (S3) on exactly the arm's support, so the pair is
        # comparable cell-for-cell. Needs pre-origin truth, hence the wider month set.
        if str(_HN / "scripts") not in sys.path:
            sys.path.insert(0, str(_HN / "scripts"))
        from rollout_ruler_core import climatology_resample

        from rollout_ruler_core import reference_sample_width

        support = [(m0, int(u)) for m0, (_c, _t, uu) in a.items() for u in uu]
        # The pool actually requested is [ANCHOR-35, ANCHOR], so the anchor's months must be
        # fetched explicitly. Deriving hist_months from the sliding window alone happened to
        # cover it only because the earliest origin is 457; point this at origins starting
        # later and truth_map.get(..., 0.0) would silently pad the reference with zeros and
        # make it too timid, with no exception. rescore_v2 unions the anchor in; this did not.
        anchor_pool = set(range(CLIM_ANCHOR - 36 + 1, CLIM_ANCHOR + 1))
        hist_months = anchor_pool | {m for (m0, _u) in support for m in range(m0 - 36, m0)}
        hist = roll._truth_map(str(v2.V2_TRUTH), f"lr_{args.target}_best", hist_months)
        missing = anchor_pool - set(m for (m, _u) in hist)
        if missing == anchor_pool:
            raise SystemExit(
                f"the climatology pool {min(anchor_pool)}-{max(anchor_pool)} has no truth at "
                "all; the reference would be resampled entirely from zero-padding."
            )
        # CANONICAL params, stated explicitly — climatology_resample requires all three, so a
        # convention can no longer be selected by omitting a keyword (which is how this file's
        # sibling tail_index.py published nine numbers against a different reference).
        gathered = climatology_resample(
            hist,
            support,
            (args.h,),
            window=36,
            n_samples=reference_sample_width(
                (c.shape[1] for _m0, (c, _t, _u) in a.items()), where=labels[0]
            ),
            seed=CLIM_SEED,
            window_anchor=CLIM_ANCHOR,
        )
        b = {}
        for m0, (_c, tt, uu) in a.items():
            cube = np.stack([gathered[(m0, args.h, int(u))][0] for u in uu])
            b[m0] = (cube, tt, uu)
    else:
        b = per_origin_crps(dirs[1], args.target, args.h, tmap)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no shared origins between the two arms")

    diffs, group_ids, per_origin_mean = [], [], {}
    for m0 in shared:
        ca, ta, ua = a[m0]
        cb, tb, ub = b[m0]
        # Align by cell id, NOT by position. Boolean masking preserves each source array's own
        # order; it does not reindex to `keep`. With ua=[3,1,2] and ub=[1,2,3] both masks are
        # all-True, no shape error fires, and the subtraction silently pairs cell 3 against
        # cell 1. The origin means survive that (each origin's sum is unchanged) but the
        # per-cell vector does not — and it is the per-cell vector the iid bootstrap and the
        # heavy-tail reading below are computed from. rescore_v2 uses an explicit unit->index
        # map; this did not.
        pa = {int(x): i for i, x in enumerate(ua)}
        pb = {int(x): i for i, x in enumerate(ub)}
        if len(pa) != len(ua) or len(pb) != len(ub):
            raise SystemExit(f"origin {m0}: duplicate unit ids — the pairing is ambiguous")
        keep = sorted(set(pa) & set(pb))
        ia = np.array([pa[u] for u in keep], dtype=int)
        ib = np.array([pb[u] for u in keep], dtype=int)
        d = crps_ensemble(ta[ia], ca[ia]) - crps_ensemble(tb[ib], cb[ib])
        diffs.append(d)
        group_ids.append(np.full(d.size, m0))
        per_origin_mean[m0] = float(d.mean())

    d_all = np.concatenate(diffs)
    g_all = np.concatenate(group_ids)

    lo_o, hi_o, _ = gw._bootstrap_mean_ci(d_all, g_all, args.n_boot, args.seed, args.ci, "origin")
    lo_i, hi_i, _ = gw._bootstrap_mean_ci(d_all, g_all, args.n_boot, args.seed, args.ci, "obs")

    observed = float(d_all.mean())
    mde = (hi_o - lo_o) / 2.0
    mde_iid = (hi_i - lo_i) / 2.0
    if mde <= 0 or mde_iid <= 0:
        raise SystemExit(
            f"degenerate CI (origin half-width {mde:.6g}, iid {mde_iid:.6g}). Both are zero "
            "when the two arms are identical, which makes the width ratio undefined rather "
            "than infinite — scoring an arm against itself is not a measurable comparison."
        )

    lines = [
        "# Minimum detectable effect at P = 13",
        "",
        f"Epic #263 / S4 (#268). Pair: `{labels[0]}` − `{labels[1]}`, target `{args.target}`, "
        f"h = {args.h}. {len(shared)} origins, N = {d_all.size:,} cell-months. "
        f"{args.n_boot} bootstrap replicates, {int(args.ci * 100)}% CI, seed {args.seed}.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| observed mean CRPS differential | {observed:+.6g} |",
        f"| origin-block CI | [{lo_o:+.6g}, {hi_o:+.6g}] |",
        f"| **MDE (origin-block, half-width)** | **{mde:.6g}** |",
        f"| iid-over-cells CI (for contrast only) | [{lo_i:+.6g}, {hi_i:+.6g}] |",
        f"| iid half-width | {mde_iid:.6g} |",
        f"| **origin-block ÷ iid width** | **{mde / mde_iid:.2f}×** |",
        f"| separable from 0 at {int(args.ci * 100)}%? | "
        f"{'YES' if (lo_o > 0 or hi_o < 0) else 'NO'} |",
        "",
        "## Per-origin mean differential",
        "",
        "| origin (m0) | mean Δcrps |",
        "|---:|---:|",
    ]
    for m0 in shared:
        lines.append(f"| {m0} | {per_origin_mean[m0]:+.6g} |")

    lines += [
        "",
        "## Reading this",
        "",
        f"An effect smaller than **{mde:.6g}** cannot be distinguished from zero at "
        f"{int(args.ci * 100)}% with {len(shared)} origins, however many cells there are. "
        "A null on this pair therefore means *either* no difference *or* no power — the "
        "distinction C-254 exists to preserve.",
        "",
        (
            f"The iid-over-cells bootstrap gives a CI **{mde_iid / mde:.2f}× WIDER** here, not "
            "narrower. That is the opposite of the direction C-221/C-253 anticipate, and it is "
            "worth stating plainly rather than asserting the expected sign: the per-cell CRPS "
            "differential on this pair is dominated by a handful of heavy-tail conflict cells, "
            "so resampling 170k individual cells shakes those extremes in and out and inflates "
            "the spread, while the 13 per-origin means are comparatively stable. Which bootstrap "
            "is wider depends on the balance between within-origin correlation (favours "
            "origin-block being wider, the C-253 synthetic case) and tail heaviness (favours iid "
            "being wider, this case)."
            if mde_iid > mde
            else f"The iid-over-cells bootstrap gives a CI **{mde / mde_iid:.2f}× narrower**. "
            "That is the overconfidence C-221/C-253 warn about: it treats ~13k spatially "
            "co-active cells per origin as independent observations."
        ),
        "",
        "**Neither reading licenses using the iid bootstrap.** It is reported as a contrast "
        "only. The origin block is the correct unit because the 36-month futures of adjacent "
        "origins overlap; that is a structural fact about the design, not an empirical result "
        "about which interval happens to be wider on one pair.",
        "",
        "## Why this is not called a Giacomini–White test",
        "",
        "The models are trained **once** and scored at 13 rolling origins, so the forecasting "
        'scheme is **fixed**, satisfying Giacomini & White 2006 §3.2 Comment 3 (*"Expanding '
        'window forecasting schemes are ruled out by assumption"*). That holds **by '
        "construction**: one prediction directory is the output of one training run, so its 13 "
        "origins necessarily share an artifact — nothing checks them pairwise, because there is "
        "nothing that could differ. What `partition_audit.py` adds is that the artifact's "
        "sha256 is resolved and recorded, and now fails loud when it cannot be, so the claim is "
        "auditable after the fact rather than merely asserted.",
        "",
        "But `gw_stratified` is a **bootstrap on the mean loss differential**, not a GW "
        "conditional-predictive-ability regression: there is no analytic variance estimator and "
        'no conditioning test function. Calling it "the GW test" would be overclaiming. At '
        "P = 13 an HAC/Newey–West regression would be *less* honest, not more, which is why "
        "upgrading it is out of scope (`SCOPE.md` #12).",
        "",
        "Residual, stated plainly: at a fixed horizon the 13 origins are **adjacent months**, so "
        "the origin blocks are not independent either. The block bootstrap fixes within-origin "
        "dependence, not between-origin serial dependence. **Report the MDE; do not try to fix "
        "it.**",
        "",
    ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MDE.md").write_text("\n".join(lines))
    print(f"wrote {out / 'MDE.md'}")
    print(f"  observed  {observed:+.6g}")
    print(f"  MDE       {mde:.6g}   (origin-block, {int(args.ci * 100)}%)")
    rel = f"{mde_iid / mde:.2f}x WIDER" if mde_iid > mde else f"{mde / mde_iid:.2f}x narrower"
    print(f"  iid       {mde_iid:.6g}   ({rel} — contrast only, never for a claim)")
    print(f"  separable from 0: {'YES' if (lo_o > 0 or hi_o < 0) else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
