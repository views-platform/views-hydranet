#!/usr/bin/env python
"""gate_reliability.py — the pre-registered C-147 onset-gate calibration readout.

Is the hurdle-NB onset gate ``π = sigmoid(cls)`` (saved as the ``by_*`` predictions) calibrated
against binary onset truth ``1[lr_* > 0]``? A class-weighted Bernoulli NLL can minimise while π is
miscalibrated, biasing every ``E[y] = P(y>0)·μ/(1−NB0)`` (C-147, pre-registered in
reports/2026-06-10_zinb_distributional_head_dossier/05_analysis_plan.md but never run).

Read-only on saved artifacts: no GPU/model/training. The (month_id, priogrid_gid) join guards
(unique index + every-cell-matched, C-136) are copied **verbatim** from ``scripts/mcr_readout.py``
so pairing is identical; Brier / reliability / ECE are self-contained.

Axes (matching mcr_readout): ``y_pred.npy`` is ``(N_cell_months, n_samples)`` — the second
axis is **samples**, not forecast steps. The rollout axis is **STEP-1** (each origin's seed month,
teacher-forced) vs **FULL** (all months, rollout engaged): comparing them shows whether any
miscalibration is present from the start (gate/loss) or grows with the rollout (feedback).

Usage:
    python scripts/gate_reliability.py <predictions_dir> [--baseline <dir>] [--raw <parquet>] \
        [--pairs by_sb_best:lr_sb_best by_ns_best:lr_ns_best by_os_best:lr_os_best] \
        [--n-boot 1000] [--bins 10] [--out <report.md>]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

DEFAULT_PAIRS = ["by_sb_best:lr_sb_best", "by_ns_best:lr_ns_best", "by_os_best:lr_os_best"]


# --- C-136 join guards: copied verbatim from scripts/mcr_readout.py (identical pairing) ---
def load_all_truth(raw_parquet: str, targets: list[str]) -> dict[str, pd.Series]:
    """Read the raw parquet ONCE; return {target: (month_id, priogrid_gid)->target Series}.

    Reads once (the 5M-row parquet ×N reads OOMs an 8GB box). Same C-136 uniqueness guard as
    mcr_readout.load_truth_index, applied once on the shared index.
    """
    raw = pd.read_parquet(raw_parquet).reset_index()
    for col in ("month_id", "priogrid_gid", *targets):
        if col not in raw.columns:
            raise ValueError(f"raw parquet missing {col!r}; columns: {list(raw.columns)[:12]}")
    raw = raw[["month_id", "priogrid_gid", *targets]].set_index(["month_id", "priogrid_gid"])
    if not raw.index.is_unique:
        n_dup = int(raw.index.duplicated().sum())
        raise ValueError(
            f"raw (month_id, priogrid_gid) index is NOT unique ({n_dup} dups) — "
            f"the join would be ambiguous. Refusing (guard the join, C-136)."
        )
    return {t: raw[t] for t in targets}


def aligned_truth(
    truth_idx: pd.Series, months: np.ndarray, units: np.ndarray, target: str
) -> np.ndarray:
    """Align actuals to (month, unit) prediction keys; FAIL LOUD on any unmatched cell."""
    keys = list(zip(months.tolist(), units.tolist()))
    truth = truth_idx.reindex(keys).to_numpy(dtype=np.float64)
    n_missing = int(np.isnan(truth).sum())
    if n_missing:
        raise ValueError(
            f"{target}: {n_missing}/{len(truth)} prediction cells had NO matching actual — "
            f"refusing to silently drop (guard the join, C-136)."
        )
    return truth


# --- calibration metrics (self-contained) ---
def brier(pi: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pi - y) ** 2))


def brier_baserate(y: np.ndarray) -> float:
    """Brier of the constant predictor that always emits the prevalence = ρ(1−ρ)."""
    rho = float(y.mean())
    return rho * (1.0 - rho)


def reliability_table(pi: np.ndarray, y: np.ndarray, n_bins: int):
    """Equal-width bins over [0,1]; per-bin (mean_pi, emp_freq, count). Returns (rows, ece)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, n = [], 0.0, len(y)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (pi >= lo) & (pi < hi if b < n_bins - 1 else pi <= hi)
        c = int(m.sum())
        if c == 0:
            rows.append((lo, hi, c, float("nan"), float("nan")))
            continue
        mp, ef = float(pi[m].mean()), float(y[m].mean())
        rows.append((lo, hi, c, mp, ef))
        ece += (c / n) * abs(mp - ef)
    return rows, float(ece)


def _bootstrap_ci(pi: np.ndarray, y: np.ndarray, n_boot: int, salt: int):
    """95% bootstrap CI on (Brier, mean_pi), resampling cells with replacement."""
    n = len(y)
    if n == 0 or n_boot <= 0:
        return {"brier": (float("nan"),) * 2, "mean_pi": (float("nan"),) * 2}
    rng = np.random.default_rng(1234 + salt)
    bs, mp = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bs[b] = np.mean((pi[idx] - y[idx]) ** 2)
        mp[b] = pi[idx].mean()
    return {
        "brier": tuple(np.percentile(bs, [2.5, 97.5])),
        "mean_pi": tuple(np.percentile(mp, [2.5, 97.5])),
    }


def _collect_gate(pred_dir: str, by_target: str, truth_idx: pd.Series, lr_target: str):
    """Pool (π_cell, binary_truth) across origin_* for STEP-1 (seed month) and FULL (all months).

    π_cell = mean over posterior samples of the saved by_* prediction. truth = 1[lr_* > 0].
    """
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    if not origins:
        raise FileNotFoundError(f"no origin_* dirs under {pred_dir}")
    s_pi, s_y, f_pi, f_y = [], [], [], []
    n_skipped = 0
    rng_lo, rng_hi = np.inf, -np.inf
    for origin in origins:
        d = os.path.join(origin, by_target)
        ypath, ipath = os.path.join(d, "y_pred.npy"), os.path.join(d, "identifiers.npz")
        if not (os.path.exists(ypath) and os.path.exists(ipath)):
            n_skipped += 1
            continue
        yp = np.load(ypath).astype(np.float64)  # (N, S=samples)
        ids = np.load(ipath)
        tm, un = ids["time"], ids["unit"]
        rng_lo, rng_hi = min(rng_lo, float(yp.min())), max(rng_hi, float(yp.max()))
        pi_cell = yp.mean(axis=1)  # E[π] per cell over posterior samples
        truth = aligned_truth(truth_idx, tm, un, lr_target)
        ybin = (truth > 0).astype(np.float64)
        finite = np.isfinite(pi_cell)
        seed = int(tm.min())
        for sel, PI, Y in ((tm == seed, s_pi, s_y), (np.ones_like(tm, bool), f_pi, f_y)):
            m = sel & finite
            PI.append(pi_cell[m])
            Y.append(ybin[m])
    if n_skipped:
        print(f"  [note] {by_target}: {n_skipped} origin(s) missing — skipped")
    # F-alignment-artifact guard: π must be a probability in [0,1].
    if not (rng_lo >= -1e-6 and rng_hi <= 1.0 + 1e-6):
        raise ValueError(
            f"{by_target}: saved predictions range [{rng_lo:.4g}, {rng_hi:.4g}] is NOT in [0,1] — "
            f"these are not gate probabilities sigmoid(cls). Refusing (F-alignment-artifact)."
        )

    def cat(ps, ys):
        return (np.concatenate(ps), np.concatenate(ys)) if ps else (np.array([]), np.array([]))

    return cat(s_pi, s_y), cat(f_pi, f_y), (rng_lo, rng_hi)


def _readout(pi: np.ndarray, y: np.ndarray, n_boot: int, bins: int, salt: int) -> dict:
    if len(y) == 0:
        return {"n": 0}
    b, b0 = brier(pi, y), brier_baserate(y)
    rows, ece = reliability_table(pi, y, bins)
    ci = _bootstrap_ci(pi, y, n_boot, salt)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_pi": float(pi.mean()),
        "brier": b,
        "brier_baserate": b0,
        "skill": float(1.0 - b / b0) if b0 > 0 else float("nan"),
        "ece": ece,
        "ci": ci,
        "rows": rows,
    }


def _fmt_block(tag: str, m: dict) -> list[str]:
    if m.get("n", 0) == 0:
        return [f"**{tag}** — n=0", ""]
    ci = m["ci"]
    out = [
        f"**{tag}** (n={m['n']:,})",
        "",
        f"- prevalence (onset base rate): **{m['prevalence']:.4%}**",
        f"- mean π: **{m['mean_pi']:.4f}** "
        f"[95% CI {ci['mean_pi'][0]:.3f}, {ci['mean_pi'][1]:.3f}]",
        f"- Brier: **{m['brier']:.4f}** [95% CI {ci['brier'][0]:.4f}, {ci['brier'][1]:.4f}]  "
        f"(base-rate predictor Brier = {m['brier_baserate']:.4f})",
        f"- **Brier skill score vs base-rate: {m['skill']:+.3f}**  (<0 ⇒ worse than prevalence)",
        f"- ECE: **{m['ece']:.4f}**",
        "",
        "| π bin | n | mean π | empirical onset freq |",
        "|------|---|--------|----------------------|",
    ]
    for lo, hi, c, mp, ef in m["rows"]:
        if c == 0:
            continue
        out.append(f"| [{lo:.1f},{hi:.1f}) | {c:,} | {mp:.3f} | {ef:.4f} |")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pred_dir", help="predictions_<run> dir (origin_*/{by_target}/y_pred.npy)")
    ap.add_argument("--baseline", default=None, help="optional second predictions dir to contrast")
    ap.add_argument("--raw", default=None, help="raw calibration parquet (default: auto-glob)")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="by_target:lr_target pairs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default=None, help="optional markdown file to write the readout to")
    args = ap.parse_args()

    raw = args.raw
    if raw is None:
        base = os.path.dirname(args.pred_dir.rstrip("/"))
        hits = sorted(glob.glob("data/raw/*calibration*.parquet")) or sorted(
            glob.glob(os.path.join(base, "..", "raw", "*calibration*.parquet"))
        )
        if not hits:
            ap.error("could not auto-find a calibration raw parquet; pass --raw")
        raw = hits[0]

    pairs = [tuple(p.split(":")) for p in args.pairs]
    lr_targets = sorted({lr for _, lr in pairs})
    truth = load_all_truth(raw, lr_targets)

    lines = ["# C-147 gate-reliability readout", "", f"raw actuals: `{raw}`", ""]
    runs = [("RUN", args.pred_dir)]
    if args.baseline:
        runs.insert(0, ("BASELINE", args.baseline))

    for label, d in runs:
        lines += [f"## {label}: `{d}`", ""]
        for pi_idx, (by_t, lr_t) in enumerate(pairs):
            (s_pi, s_y), (f_pi, f_y), rng = _collect_gate(d, by_t, truth[lr_t], lr_t)
            salt = pi_idx * 2  # stable per-target seed → reproducible bootstrap CIs (not hash())
            s = _readout(s_pi, s_y, args.n_boot, args.bins, salt)
            f = _readout(f_pi, f_y, args.n_boot, args.bins, salt + 1)
            lines += [f"### {by_t}  (π range on disk: [{rng[0]:.4g}, {rng[1]:.4g}])", ""]
            lines += _fmt_block("STEP-1 (teacher-forced seed month)", s)
            lines += _fmt_block("FULL (all months, rollout engaged)", f)

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
