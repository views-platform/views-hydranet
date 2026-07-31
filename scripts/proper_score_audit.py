#!/usr/bin/env python
"""proper_score_audit.py — judge saved predictions with PROPER scores, not the eyeball/FSS metric.

WHY THIS EXISTS (2026-06-21, expert-method-review gate). We have been ranking heads by a custom
FSS / area-ratio "sharpness" scorecard and by eye. An 8-persona methodology panel warned, near
unanimously, that this is an IMPROPER diagnostic: on a 99.7%-zero map a model that predicts
near-zero everywhere looks "sharp" and "wins" even if badly calibrated (Gneiting2007 linear-score
trap; Lerch2017 forecaster's dilemma; Taillardat2023 CRPS tail-blindness). So the headline finding
we were about to act on — "the negative-binomial loss is ~10x blurrier than shrinkage" — might be a
measurement artifact. This tool re-judges the SAME saved runs with honest scores so a near-zero
cheat cannot win, scoring ALL cells (NOT mcr_readout's positive-only subset — that is the dilemma).

Read-only. No GPU, no model, no training. Reuses the C-136 fail-loud join guards from mcr_readout
and the grid reconstruction from sharpness_scorecard.

Per target x {STEP-1 (out-of-sample first step, no rollout), FULL (all steps; rollout parked, shown
only as a labelled aside)}:
  - CRPS            proper; already in views_evaluation; all cells.
  - SCRPS           Bolin&Wallin 2023; scale-invariant so the 99.7% zeros don't drown the signal.
  - twCRPS@[0,1]    threshold-weighted, proper; weights the positive region without conditioning.
  - randomized PIT  calibration, count-aware. Reported on ALL cells AND positive-truth cells (the
                    positive PIT is a CONDITIONAL diagnostic, not for selection). NOTE: a
                    continuous hurdle mean piles PIT at 0 on zero-truth cells (Czado2009 discrete
                    caveat) — read the positive-cell PIT for magnitude calibration.
  - MCR (all & pos), mean pred / mean truth — diagnostics.
Plus a SHARPNESS CROSS-CHECK reusing the old scorecard (area-ratio, FSS@1) computed on the 8-sample
MEAN field AND on a SINGLE draw — to test whether any "blur" is the MC-average, not the model.

FAO-02 COMPLIANCE (2026-07-27, C-167 close-out): this tool is a DIAGNOSTIC. **`twCRPS` and
`randomized PIT` are FAO-02-REJECTED and must NEVER enter a selection/decision rule** (see
`reference_fao02_locked_eval_framework.md`: CRPS primary + QS99/Brier/MCR guardrails only).
Selection uses CRPS (proper, primary) with `size_ratio` (magnitude) and FSS/conc1% as a
*directional* anti-smearing guard — twCRPS/PIT are read here only to corroborate, never to decide.

Usage:
    python scripts/proper_score_audit.py <pred_dir> [--baseline <dir>] [--raw <parquet>] \
        [--targets ...] [--out <md>]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os

import numpy as np

# Reuse trusted helpers from the sibling scripts (scripts/ is on sys.path[0] when run directly).
from mcr_readout import aligned_truth, load_truth_index  # C-136 fail-loud join guards
from sharpness_scorecard import _score_field, build_unit_grid, to_grid
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_crps_native,
    calculate_twcrps_native,
)

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
_EPS = 1e-12


# ---- pure scoring functions (unit-tested) ----
def crps_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean CRPS over cells (proper; lower better). Thin wrapper on the pipeline implementation."""
    return calculate_crps_native(y_true, y_pred)


def scrps_ensemble(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-cell SCRPS (Bolin&Wallin 2023), REWARD form (higher better), scale-invariant.

    SCRPS = -E|X-y| / E|X-X'| - 0.5*log E|X-X'|, with ensemble estimators
    E|X-y| = mean_i|x_i-y| and E|X-X'| = (2/S^2) * sum_j (2j-S+1) x_(j)  (sorted identity).
    """
    sp = np.sort(y_pred, axis=1)
    s = sp.shape[1]
    mae = np.mean(np.abs(sp - y_true[:, None]), axis=1)
    w = (2 * np.arange(s) - s + 1).astype(float)
    beta = 2.0 * np.sum(w[None, :] * sp, axis=1) / (s * s)  # E|X-X'| >= 0
    beta = np.maximum(beta, _EPS)
    return -(mae / beta) - 0.5 * np.log(beta)


def randomized_pit(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Count-aware randomized PIT: P(X<y) + U*P(X=y), U~Uniform. Uniform under calibration."""
    below = np.mean(y_pred < y_true[:, None], axis=1)
    eq = np.mean(y_pred == y_true[:, None], axis=1)
    return below + rng.uniform(size=len(y_true)) * eq


def pit_noncalibration(pits: np.ndarray, nbins: int = 10) -> tuple[np.ndarray, float]:
    """Histogram freq + flatness score (mean |freq - 1/nbins|; 0 = perfectly uniform)."""
    hist, _ = np.histogram(pits, bins=nbins, range=(0.0, 1.0))
    total = hist.sum()
    freq = hist / total if total else np.full(nbins, np.nan)
    return freq, float(np.mean(np.abs(freq - 1.0 / nbins)))


def _mcr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mt = y_true.mean()
    return float(y_pred.mean() / mt) if mt > 0 else float("nan")


# ---- collection (ALL cells — NOT positive-only) ----
def _collect_all(pred_dir: str, target: str, truth_idx):
    """Pool (y_true, y_pred over ALL finite cells) across origins → {STEP-1, FULL}."""
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    if not origins:
        raise FileNotFoundError(f"no origin_* dirs under {pred_dir}")
    acc = {"STEP-1": ([], []), "FULL": ([], [])}
    for origin in origins:
        d = os.path.join(origin, target)
        yp_p, ip = os.path.join(d, "y_pred.npy"), os.path.join(d, "identifiers.npz")
        if not (os.path.exists(yp_p) and os.path.exists(ip)):
            continue
        yp = np.load(yp_p).astype(np.float64)  # (N, S)
        ids = np.load(ip)
        tm, un = ids["time"], ids["unit"]
        truth = aligned_truth(truth_idx, tm, un, target)
        finite = np.isfinite(yp).all(axis=1) & np.isfinite(truth)
        seed = int(tm.min())
        for tag, sel in (("STEP-1", tm == seed), ("FULL", np.ones_like(tm, bool))):
            m = sel & finite
            acc[tag][0].append(truth[m])
            acc[tag][1].append(yp[m])
    out = {}
    for tag, (ts, ps) in acc.items():
        out[tag] = (
            (np.concatenate(ts), np.concatenate(ps)) if ts else (np.array([]), np.zeros((0, 1)))
        )
    return out


def _score_subset(y_true: np.ndarray, y_pred: np.ndarray, salt: int) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    rng = np.random.default_rng(2024 + salt)
    _, nc_all = pit_noncalibration(randomized_pit(y_true, y_pred, rng))
    posm = y_true > 0
    if posm.any():
        _, nc_pos = pit_noncalibration(randomized_pit(y_true[posm], y_pred[posm], rng))
    else:
        nc_pos = float("nan")
    return {
        "n": int(len(y_true)),
        "n_pos": int(posm.sum()),
        "crps": crps_mean(y_true, y_pred),
        "scrps": float(np.mean(scrps_ensemble(y_true, y_pred))),
        "twcrps0": calculate_twcrps_native(y_true, y_pred, threshold=0.0),
        "twcrps1": calculate_twcrps_native(y_true, y_pred, threshold=1.0),
        "pit_noncal_all": nc_all,
        "pit_noncal_pos": nc_pos,
        "mcr_all": _mcr(y_true, y_pred),
        "mcr_pos": _mcr(y_true[posm], y_pred[posm]) if posm.any() else float("nan"),
        "mean_pred": float(y_pred.mean()),
        "mean_true": float(y_true.mean()),
    }


# ---- sharpness cross-check: mean field vs single draw (read-only) ----
def _sharpness(pred_dir, target, truth_idx, umap, readout, scales=(1, 3, 5, 11)):
    """area_ratio + FSS@1 averaged over STEP-1 timesteps, using mean field or a single draw."""
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    rows = []
    for origin in origins:
        d = os.path.join(origin, target)
        yp_p, ip = os.path.join(d, "y_pred.npy"), os.path.join(d, "identifiers.npz")
        if not (os.path.exists(yp_p) and os.path.exists(ip)):
            continue
        yp = np.load(yp_p).astype(np.float64)
        vals = yp.mean(axis=1) if readout == "mean" else yp[:, 0]
        ids = np.load(ip)
        tm, un = ids["time"], ids["unit"]
        truth = aligned_truth(truth_idx, tm, un, target)
        seed = int(tm.min())
        for t in np.unique(tm[tm == seed]):
            m = tm == t
            pg = to_grid(vals[m], un[m], umap)
            og = to_grid(truth[m], un[m], umap)
            f, area, _ = _score_field(pg, og, list(scales))
            rows.append((f[1], area))
    if not rows:
        return float("nan"), float("nan")
    return float(np.nanmean([r[0] for r in rows])), float(np.nanmean([r[1] for r in rows]))


def _config_tag(pred_dir):
    """Auto-read loss_reg/output_distribution from the sidecar so the run is never mis-stated."""
    spec = importlib.util.spec_from_file_location(
        "sharpness_scorecard",
        os.path.join(os.path.dirname(__file__), "sharpness_scorecard.py"),
    )
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    cfg = sc.read_config_block(pred_dir, None)
    keys = ["loss_reg", "loss_class", "output_distribution", "reg_activation"]
    return " · ".join(f"{k}={cfg.get(k)}" for k in keys if k in cfg) or "(sidecar not found)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pred_dir")
    ap.add_argument("--baseline", default=None, help="second predictions dir to contrast")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = args.raw
    if raw is None:
        hits = sorted(glob.glob("data/raw/*calibration*.parquet")) or sorted(
            glob.glob(
                os.path.join(
                    os.path.dirname(args.pred_dir.rstrip("/")),
                    "..",
                    "raw",
                    "*calibration*.parquet",
                )
            )
        )
        if not hits:
            ap.error("could not auto-find a calibration raw parquet; pass --raw")
        raw = hits[0]
    truth = {t: load_truth_index(raw, t) for t in args.targets}
    umap = build_unit_grid(raw)

    runs = [("RUN", args.pred_dir)]
    if args.baseline:
        runs.insert(0, ("BASELINE", args.baseline))

    lines = [
        "# Proper-score audit (honest yardstick vs the FSS scorecard)",
        "",
        f"raw actuals: `{raw}`",
        "",
    ]
    for label, d in runs:
        lines += [f"## {label}: `{d}`", f"**CONFIG (auto):** {_config_tag(d)}", ""]
        lines += [
            "| target | subset | CRPS↓ | SCRPS↑ | twC@0↓ | twC@1↓ | "
            "PITnc(all/pos) | MCR(all/pos) | mean p/t |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for t in args.targets:
            coll = _collect_all(d, t, truth[t])
            for tag in ("STEP-1", "FULL"):
                yt, yp = coll[tag]
                m = _score_subset(yt, yp, salt=hash(t + tag) % 1000)
                if m["n"] == 0:
                    lines.append(f"| {t} | {tag} | n=0 | | | | | | |")
                    continue
                # SCRPS is undefined for a near-degenerate (overconfident, ~0-spread) ensemble:
                # E|X-X'|->0 blows up the -mae/beta term. Flag rather than print a
                # meaningless number.
                sc = f"{m['scrps']:.2f}" if abs(m["scrps"]) < 1e5 else "degen*"
                lines.append(
                    f"| {t} | {tag} | {m['crps']:.4f} | {sc} | "
                    f"{m['twcrps0']:.4f} | {m['twcrps1']:.4f} | "
                    f"{m['pit_noncal_all']:.3f}/{m['pit_noncal_pos']:.3f} | "
                    f"{m['mcr_all']:.3f}/{m['mcr_pos']:.3f} | "
                    f"{m['mean_pred']:.3f}/{m['mean_true']:.3f} |"
                )
        lines += [
            "",
            "**Sharpness cross-check (STEP-1) — FSS@1 / area-ratio, mean field vs single draw:**",
            "| target | FSS@1 mean | area mean | FSS@1 single | area single |",
            "|---|---|---|---|---|",
        ]
        for t in args.targets:
            fm, am = _sharpness(d, t, truth[t], umap, "mean")
            fs, as_ = _sharpness(d, t, truth[t], umap, "single")
            lines.append(f"| {t} | {fm:.2f} | {am:.2f}× | {fs:.2f} | {as_:.2f}× |")
        lines += [
            "",
            "> `degen*` SCRPS = ensemble spread ~0 (overconfident under-firing) → SCRPS "
            "undefined; use CRPS/twCRPS. PITnc(all) is dominated by the 99.7% zero cells "
            "(Czado2009 discrete caveat): a model with a clean point-mass at 0 scores ~uniform "
            "there; one that spreads mass onto zeros scores high. PITnc(pos) is the magnitude "
            "calibration on positive cells.",
            "",
        ]

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
