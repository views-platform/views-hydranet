"""Assemble the v2 scoreboard from the per-arm/-seed score CSVs in results/.

Reads results/score_<arm>_<seed>.csv (arm ∈ {gated_NB, th_gated_NB, ZINB}, seed ∈ {42,43,44}) plus
results/score_light_strider.csv (baseline). Aggregates each metric across the AVAILABLE seeds
(mean ± sample-std; n_seeds reported so partial runs are honest), and renders:
  - a headline crps_all table (arm × target × horizon),
  - a full per-(target,horizon) metric table (arm × metric, mean±std),
  - a per-seed appendix,
to results/SCOREBOARD.md. Runnable incrementally — aggregates whatever seeds are present.

Usage: python assemble_scoreboard.py            # writes results/SCOREBOARD.md
       python assemble_scoreboard.py --print     # also echo the headline to stdout
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ARMS = ["gated_NB", "th_gated_NB", "ZINB"]
BASE = "light_strider"
TARGETS = ["sb", "ns", "os"]
HORIZONS = [1, 18, 36]
# ⛔ #537: crps_concentration and crps_effective_n must appear BESIDE crps_all here, not only in
# the score CSVs. This file renders what its own docstring calls a "headline crps_all table", reads
# exactly the CSVs score_v2_horizons.py writes, and never calls require_headline_columns — so
# without these two entries the human-facing SCOREBOARD.md stays a bare pooled crps_all table,
# which is the thing the guard exists to prevent. ⚠️ Score CSVs written before 2026-11-04 lack the
# columns and will render as the missing-value marker; that is correct and visible, not a bug.
METRICS = ["crps_all", "crps_concentration", "crps_effective_n",
           "crps_none", "crps_events", "AP", "Brier",
           "size_ratio", "mcr_all", "pos_mcr", "mcr_none"]
METRIC_LABEL = {"crps_all": "CRPS_all",
                "crps_concentration": "CRPS_conc(top band)",
                "crps_effective_n": "CRPS_eff_n",
                "crps_none": "CRPS_none", "crps_events": "CRPS_events",
                "AP": "AP", "Brier": "Brier", "size_ratio": "size_ratio",
                "mcr_all": "MCR_all", "pos_mcr": "MCR_events", "mcr_none": "MCR_none(leak)"}


def _load():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "score_*.csv"))):
        base = os.path.basename(f)[len("score_"):-len(".csv")]
        m = re.match(r"^(.*)_(\d+)$", base)
        if m and m.group(1) in ARMS:
            arm, seed = m.group(1), int(m.group(2))
        elif base == BASE:
            arm, seed = BASE, None
        else:
            continue
        d = pd.read_csv(f)
        d["arm"], d["seed"] = arm, seed
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fmt(mean, std, n):
    if np.isnan(mean):
        return "—"
    if n is None or n <= 1:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _agg(df, arm, tgt, h, metric):
    sub = df[(df.arm == arm) & (df.target == tgt) & (df.h == h)][metric].to_numpy(dtype=float)
    sub = sub[~np.isnan(sub)]
    if sub.size == 0:
        return np.nan, np.nan, 0
    return float(sub.mean()), (float(sub.std(ddof=1)) if sub.size > 1 else 0.0), sub.size


def main():
    df = _load()
    out = ["# v2 (datafactory) scoreboard — 3 arms × 3 seeds × 300 lessons, horizon-resolved", ""]
    if df.empty:
        out.append("_No score CSVs yet._")
        open(os.path.join(RES, "SCOREBOARD.md"), "w").write("\n".join(out))
        print("no data yet")
        return
    present = {a: sorted(df[df.arm == a].seed.dropna().unique().astype(int)) for a in ARMS}
    out.append("Arms/seeds present: " + "; ".join(f"{a}={present[a]}" for a in ARMS) +
               f"; baseline={'yes' if (df.arm == BASE).any() else 'no'}")
    out.append("Values = mean±std across available seeds. Truth = frozen v2 datafactory "
               "(months 121–504 × 13,110 cells). h=1/18/36 = first/middle/last forecast month.")
    out.append("")

    # headline: crps_all
    out.append("## Headline — CRPS_all (lower is better)")
    for tgt in TARGETS:
        out.append(f"\n### target = {tgt}")
        hdr = "| arm | " + " | ".join(f"h={h}" for h in HORIZONS) + " |"
        out.append(hdr)
        out.append("|" + "---|" * (len(HORIZONS) + 1))
        for arm in ARMS + [BASE]:
            cells = []
            for h in HORIZONS:
                mean, std, n = _agg(df, arm, tgt, h, "crps_all")
                cells.append(_fmt(mean, std, n))
            out.append(f"| {arm} | " + " | ".join(cells) + " |")

    # full metric tables per (target, horizon)
    out.append("\n## Full metric tables (arm × metric)")
    for tgt in TARGETS:
        for h in HORIZONS:
            out.append(f"\n### target = {tgt}, h = {h}")
            out.append("| arm | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " |")
            out.append("|" + "---|" * (len(METRICS) + 1))
            for arm in ARMS + [BASE]:
                cells = [_fmt(*_agg(df, arm, tgt, h, m)) for m in METRICS]
                out.append(f"| {arm} | " + " | ".join(cells) + " |")

    # per-seed appendix (crps_all only, to keep it compact)
    out.append("\n## Per-seed appendix — CRPS_all")
    for tgt in TARGETS:
        out.append(f"\n### target = {tgt}")
        out.append("| arm | seed | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
        out.append("|" + "---|" * (len(HORIZONS) + 2))
        for arm in ARMS:
            for seed in present[arm]:
                cells = []
                for h in HORIZONS:
                    v = df[(df.arm == arm) & (df.seed == seed) & (df.target == tgt) & (df.h == h)]
                    cells.append(f"{v['crps_all'].iloc[0]:.4f}" if len(v) else "—")
                out.append(f"| {arm} | {seed} | " + " | ".join(cells) + " |")

    open(os.path.join(RES, "SCOREBOARD.md"), "w").write("\n".join(out))
    print(f"wrote results/SCOREBOARD.md ({len(df)} score rows; "
          f"arms present: {[a for a in ARMS if present[a]]})")
    if "--print" in sys.argv:
        # echo headline crps_all sb
        print("\nHeadline CRPS_all (sb):")
        for arm in ARMS + [BASE]:
            print(f"  {arm:14s} " + "  ".join(
                f"h{h}={_fmt(*_agg(df, arm, 'sb', h, 'crps_all'))}" for h in HORIZONS))


if __name__ == "__main__":
    main()
