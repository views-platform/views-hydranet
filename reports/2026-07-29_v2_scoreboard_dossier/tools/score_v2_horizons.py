"""V2 HORIZON SCOREBOARD RULER — the frozen rollout ruler, on datafactory v2 truth, + MCR columns.

Dossier 2026-07-29_v2_scoreboard. This is a **versioned** scorer: it does NOT mutate the frozen
lodestar T=0 tool (2026-07-17) nor the closed rollout ruler (2026-07-25). It REUSES both verbatim:
the frozen metric primitives (crps_ensemble, average_precision, brier, apply_threshold_gate,
_resolve_thresh) and the rollout ruler gather/support/truth machinery (gather_all_horizons,
`_support_keys`, `_truth_map`, `_persistence_gathered`). The ONLY thing re-implemented here is the
per-(model,target,h) scoring loop, so that two Magnitude-Calibration-Ratio columns can be emitted
beside the existing ones:

  * ``pos_mcr``  — mean(E[y]/truth) over EVENT cells  (= mcr_events; already in the frozen tools)
  * ``mcr_all``  — mean(pred)/mean(true) over ALL cells   (the repo/glossary canonical MCR)
  * ``mcr_none`` — mean(pred) over TRUE-ZERO cells   (leaked mass, count space; the bloom signal;
        the ratio is division-by-zero on zero cells, so we report the absolute mass)

Default truth = the frozen v2 datafactory truth (`v2_ruler.V2_TRUTH`); default horizons = h=1/18/36
(first / middle / last forecast month). h=1 byte-reproduces the frozen lodestar T=0 (the anchor,
tested in `tests/test_score_v2_horizons.py`).

CLI:
  score_v2_horizons.py "<label>|<pred_dir>|lr_{t}_best|by_{t}_best[|thresh]" ... \
      [--targets sb,ns,os] [--horizons 1,18,36] [--persistence] [--truth <parquet>] [--out csv]
  (omit <truth> to use the frozen v2 datafactory truth)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HN = "/home/simon/Documents/scripts/views_platform/views-hydranet"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # own dir (activation_metrics)
sys.path.insert(0, f"{_HN}/reports/2026-07-17_lodestar_eval_dossier/tools")
sys.path.insert(0, f"{_HN}/reports/2026-07-25_t0_rollout_skill_dossier/tools")
sys.path.insert(0, f"{_HN}/reports/2026-07-28_datafactory_migration_dossier/tools")

from activation_metrics import activation_frequency, topk_occurrence  # noqa: E402
from lodestar_score import (  # noqa: E402
    _resolve_thresh,
    apply_threshold_gate,
    average_precision,
    brier,
    crps_ensemble,
)
from rollout_skill_score import (  # noqa: E402
    _persistence_gathered,
    _support_keys,
    _truth_map,
    gather_all_horizons,
)

try:
    from v2_ruler import V2_TRUTH  # noqa: E402  frozen datafactory truth path
except Exception:  # pragma: no cover - v2_ruler import is a convenience default only
    V2_TRUTH = Path(
        f"{_HN}/reports/2026-07-28_datafactory_migration_dossier/tools/v2_truth"
        "/calibration_datafactory_df.parquet"
    )

DEFAULT_HORIZONS = (1, 18, 36)  # first / middle / last forecast month


def _metric_row(label, g, support, tmap, tgt, h, thr):
    """Score one (model, target, horizon). Mirrors the frozen rollout ruler metric block,
    then adds mcr_all + mcr_none. g = gathered dict (origin_m0, h, unit) -> (cs, gate)."""
    cs = np.stack([g[(m0, h, u)][0] for (m0, u) in support])  # (N, S)
    has_gate = g[(support[0][0], h, support[0][1])][1] is not None
    p = (
        np.array([g[(m0, h, u)][1] for (m0, u) in support])
        if has_gate
        else (cs > 0).mean(1)
    )
    if thr is not None:
        if not has_gate:
            raise ValueError(f"[{tgt}] {label}: th_gated needs a gate — FAIL")
        cs = apply_threshold_gate(cs, p, thr)  # body-only; p (AP/Brier) unchanged
    truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in support], dtype=float)
    ev = truth > 0
    ze = ~ev
    c = crps_ensemble(truth, cs)
    ey = cs.mean(1)
    ratio = ey[ev] / truth[ev] if ev.sum() else np.array([])
    tmean = float(truth.mean())
    # #258 activation-aware metrics (crps_all/MCR are blind to the collapse): occurrence frequency,
    # gate precision@k, and the named risk — composed magnitude on the gate's false positives.
    af = activation_frequency(cs, truth)
    tk = topk_occurrence(p, ey, truth)  # rank by the gate p; magnitude = composed mean ey
    return dict(
        target=tgt,
        model=label,
        h=h,
        N=len(support),
        n_event=int(ev.sum()),
        AP=average_precision((truth > 0).astype(float), p),
        Brier=brier((truth > 0).astype(float), p),
        crps_all=float(c.mean()),
        crps_events=float(c[ev].mean()) if ev.sum() else float("nan"),
        crps_none=float(c[ze].mean()) if ze.sum() else float("nan"),
        size_ratio=float(np.median(ratio)) if ratio.size else float("nan"),
        pos_mcr=float(np.mean(ratio)) if ratio.size else float("nan"),  # = mcr_events
        mcr_all=float(ey.mean() / tmean) if tmean > 0 else float("nan"),
        mcr_none=float(ey[ze].mean()) if ze.sum() else float("nan"),  # leaked mass (count space)
        act_pred=af["act_pred"],  # P(pred>0): the composed activation frequency
        act_true=af["act_true"],  # P(truth>0): the base rate
        act_ratio=af["act_ratio"],  # <<1 = #258 collapse, >>1 = bloom, ~1 = calibrated occurrence
        precision_at_k=tk["precision_at_k"],  # gate skill at "fire k = #true positives"
        mag_on_false_pos=tk["mag_on_false_pos"],  # THE #258 risk: magnitude on gate false-pos
        mag_on_true_pos=tk["mag_on_true_pos"],  # magnitude on correctly-fired cells (contrast)
        n_false_pos=tk["n_false_pos"],
        gate_source=("gate-head" if has_gate else "frac(samples>0)"),
    )


def score_horizons_v2(
    registry,
    targets=("sb", "ns", "os"),
    horizons=DEFAULT_HORIZONS,
    truth_parquet=str(V2_TRUTH),
    add_persistence=False,
):
    """registry: list of (label, pred_dir, lr_tmpl, by_tmpl_or_None[, thresh_spec]). One row per
    model×target×horizon, with the frozen metric set + mcr_all + mcr_none. Identical (origin,cell)
    support intersected across all arms at every scored horizon (G4)."""
    horizons = tuple(horizons)
    thresh_by_label = {e[0]: (e[4] if len(e) > 4 else None) for e in registry}
    rows = []
    for tgt in targets:
        lr_full = "lr_" + tgt + "_best"
        support = sorted(
            set.intersection(
                *[_support_keys(e[1], e[2].format(t=tgt), horizons) for e in registry]
            )
        )
        if not support:
            raise ValueError(f"[{tgt}] EMPTY fixed support across models/horizons — FAIL")
        months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
        tmap = _truth_map(truth_parquet, lr_full, months)
        for entry in list(registry):
            label, pdir, lr_tmpl, by_tmpl = entry[0], entry[1], entry[2], entry[3]
            g = gather_all_horizons(
                pdir, lr_tmpl.format(t=tgt), by_tmpl.format(t=tgt) if by_tmpl else None
            )
            thr = _resolve_thresh(thresh_by_label.get(label), tgt)
            for h in horizons:
                rows.append(_metric_row(label, g, support, tmap, tgt, h, thr))
            del g  # free this arm's cubes before the next (OOM guard)
        if add_persistence:
            gp = _persistence_gathered(tmap, support, horizons)
            for h in horizons:
                rows.append(_metric_row("persistence", gp, support, tmap, tgt, h, None))
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = ["sb", "ns", "os"]
    horizons = DEFAULT_HORIZONS
    truth = str(V2_TRUTH)
    add_persistence = False
    out_csv = "v2_scoreboard.csv"
    entries = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--targets"):
            targets = a.split("=", 1)[1].split(",")
        elif a.startswith("--horizons"):
            horizons = tuple(int(x) for x in a.split("=", 1)[1].split(","))
        elif a == "--persistence":
            add_persistence = True
        elif a.startswith("--truth"):
            truth = a.split("=", 1)[1]
        elif a.startswith("--out"):
            out_csv = a.split("=", 1)[1]
        else:
            parts = a.split("|")
            entries.append(
                (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3] if len(parts) > 3 and parts[3] else None,
                    parts[4] if len(parts) > 4 and parts[4] else None,
                )
            )
        i += 1
    rows = score_horizons_v2(
        entries, targets, horizons=horizons, truth_parquet=truth, add_persistence=add_persistence
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows: {df.model.nunique()} models × "
          f"{len(targets)} targets × {len(horizons)} h)")
    for tgt in targets:
        sub = df[df.target == tgt]
        print(f"\n=== {tgt}: crps_all / mcr_none by horizon ===")
        print(sub.pivot_table(index="model", columns="h", values="crps_all")
              .to_string(float_format=lambda x: f"{x:.4f}"))
