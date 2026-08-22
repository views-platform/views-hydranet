"""ROLLOUT SKILL RULER (T>0) — the frozen lodestar T=0 scorer, indexed by horizon.

EXP-1 (dossier 2026-07-25_t0_rollout_skill). Reuses the FROZEN lodestar primitives VERBATIM
(imported, not reimplemented): `crps_ensemble`, `average_precision`, `brier`, `apply_threshold_gate`,
`_resolve_thresh`, `BASE_RATE`, `_grid_col`, `MONTH_COL`.

`gather_all_horizons` generalizes the lodestar's `gather_t0`: instead of `sel = t == m0` (T=0 only),
keep EVERY month per origin and index by **horizon h = month − origin_m0 + 1**, so **h=1 == the lodestar
T=0** (the faithfulness anchor, F1). Scoring at horizon h uses IDENTICAL (origin, cell) support across all
horizons and all models (the G4 discipline on the horizon axis).

Metrics per (model, target, h) = the frozen lodestar set: crps_all / crps_events / crps_none + AP + Brier
+ size_ratio + pos_mcr. NO twCRPS / PIT / QS99 (not in the frozen tool; adding them = a new metric, out of
scope — chair ruling §6b). CIs (optional) = block bootstrap over ORIGINS (never iid over cells; C-221).

CLI:
  rollout_skill_score.py --selftest
  rollout_skill_score.py <truth_parquet> "<label>|<pred_dir>|lr_{t}_best|by_{t}_best[|thresh]" ... \
      [--targets sb,ns,os] [--persistence] [--boot 1000] [--out rollout_skill.csv]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

# --- reuse the FROZEN lodestar primitives verbatim (import, never reimplement) ---
sys.path.insert(
    0, "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-07-17_lodestar_eval_dossier/tools"
)
from lodestar_score import (  # noqa: E402
    MONTH_COL,
    _grid_col,
    _resolve_thresh,
    apply_threshold_gate,
    average_precision,
    brier,
    crps_ensemble,
)

HORIZONS = tuple(range(1, 37))  # h=1..36


def gather_all_horizons(pred_dir: str, lr_name: str, by_name: "str | None"):
    """Return dict ``(origin_m0, h, unit) -> (count_samples[S], switch_prob or None)``.

    ``origin_m0`` = the earliest month in an origin dir (= the lodestar T=0 month for that origin).
    ``h = month − origin_m0 + 1`` so the ``h=1`` slice reproduces ``gather_t0`` byte-for-byte.
    """
    out = {}
    origins = sorted(glob.glob(pred_dir + "/origin_*"), key=lambda p: int(p.split("_")[-1]))
    if not origins:
        raise FileNotFoundError(f"no origins in {pred_dir}")
    for od in origins:
        idf = np.load(f"{od}/{lr_name}/identifiers.npz", allow_pickle=True)
        t, u = idf["time"], idf["unit"]
        m0 = int(t.min())
        cs_all = np.load(f"{od}/{lr_name}/y_pred.npy")  # (n_total, S)
        sw_all = np.load(f"{od}/{by_name}/y_pred.npy") if by_name else None
        for month in sorted({int(x) for x in t}):
            h = month - m0 + 1
            sel = t == month
            cs = cs_all[sel]
            sw = sw_all[sel].mean(1) if sw_all is not None else None
            for i, unit in enumerate(u[sel]):
                out[(m0, h, int(unit))] = (cs[i], None if sw is None else float(sw[i]))
    return out


def _fixed_support(gathered_by_label: dict, horizons=HORIZONS):
    """(origin_m0, unit) keys present at EVERY horizon for EVERY model — the identical support (G4)."""
    per_label_full = []
    for g in gathered_by_label.values():
        seen = {}
        for (m0, h, u) in g:
            seen.setdefault((m0, u), set()).add(h)
        per_label_full.append({k for k, hs in seen.items() if set(horizons).issubset(hs)})
    return sorted(set.intersection(*per_label_full)) if per_label_full else []


def _support_keys(pred_dir: str, lr_name: str, horizons=HORIZONS):
    """(origin_m0, unit) present at EVERY horizon — from identifiers ONLY (no y_pred load; cheap)."""
    seen = {}
    for od in sorted(glob.glob(pred_dir + "/origin_*"), key=lambda p: int(p.split("_")[-1])):
        idf = np.load(f"{od}/{lr_name}/identifiers.npz", allow_pickle=True)
        t, u = idf["time"], idf["unit"]
        m0 = int(t.min())
        for tt, uu in zip(t, u):
            seen.setdefault((m0, int(uu)), set()).add(int(tt) - m0 + 1)
    return {k for k, hs in seen.items() if set(horizons).issubset(hs)}


def _truth_map(truth_parquet: str, lr_full: str, months: set):
    tdf = pd.read_parquet(truth_parquet).reset_index()
    grid = _grid_col(tdf)
    tm = {}
    for m in months:
        sub = tdf[tdf[MONTH_COL] == m].set_index(grid)[lr_full]
        d = sub.to_dict()
        for u, v in d.items():
            tm[(m, int(u))] = float(v)
    return tm


def _persistence_gathered(truth_map, support, horizons=HORIZONS, months_loaded=None):
    """Persistence 'model': feed truth[m0-1] as a 1-sample forecast at every h.

    **GH #282.** `truth_map` is built from the SCORED months `{m0+h-1}`, which do not include the
    history month `m0-1`. It is present only *by accident* — origins are usually consecutive, so
    origin k's history is origin k-1's h=1 forecast month — and the FIRST origin has no such
    neighbour. Its history therefore fell through `.get(..., 0.0)` and its entire persistence
    forecast became zeros, silently understating the baseline by 4-10% at 13 origins and by
    everything at one.

    A missing **cell** inside a loaded month is a legitimate 0.0 (`_truth_map` builds a dense
    per-month dict). A missing **month** is not a measurement at all. Pass `months_loaded` to keep
    them distinguishable. Left optional rather than required so that archived analyses which
    reconstruct this call still run — but they run WRONG, so **both production callers pass it**
    (`score_horizons` here and `score_v2_horizons.score_horizons_v2`). An earlier version of this
    docstring claimed every live caller passed it while `score_horizons` did not — the guard was
    wired into one scorer and asserted for both.
    """
    if months_loaded is not None:
        needed = {m0 - 1 for (m0, _u) in support}
        missing = sorted(needed - set(months_loaded))
        if missing:
            raise ValueError(
                f"persistence needs history months {missing}, which were never loaded — "
                f"refusing to score them as zeros (GH #282)"
            )
        # `months_loaded` is what was REQUESTED. `_truth_map` yields no keys for a month absent
        # from the parquet, so an origin whose history predates the truth data would pass the check
        # above and be scored as zeros anyway — the exact #282 failure, one level down. Verify the
        # month actually MATERIALISED.
        absent = sorted(m for m in needed if not any((m, u) in truth_map for (_m0, u) in support))
        if absent:
            raise ValueError(
                f"persistence history months {absent} were requested but the truth frame yielded "
                f"no cells for them — refusing to score them as zeros (GH #282)"
            )
    out = {}
    for (m0, u) in support:
        last = truth_map.get((m0 - 1, u), 0.0)  # last observed month before T=0
        for h in horizons:
            out[(m0, h, u)] = (np.array([last], dtype=float), None)
    return out


def score_horizons(truth_parquet, registry, targets, horizons=HORIZONS, add_persistence=False):
    """registry: list of (label, pred_dir, lr_tmpl, by_tmpl_or_None[, thresh_spec]). Returns row dicts
    (one per model×target×horizon)."""
    thresh_by_label = {e[0]: (e[4] if len(e) > 4 else None) for e in registry}
    rows = []
    for tgt in targets:
        lr_full = "lr_" + tgt + "_best"
        # (1) fixed support from IDENTIFIERS only (cheap) — intersect across arms (C-217/G4)
        support = sorted(
            set.intersection(
                *[_support_keys(e[1], e[2].format(t=tgt), horizons) for e in registry]
            )
        )
        if not support:
            raise ValueError(f"[{tgt}] EMPTY fixed support across models/horizons — FAIL")
        # history months included for the persistence baseline — see GH #282
        months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
        months |= {m0 - 1 for (m0, _u) in support}
        tmap = _truth_map(truth_parquet, lr_full, months)
        # (2) stream arm-by-arm: load one arm's cubes, score all h, FREE before next (OOM guard)
        arms = list(registry)
        for entry in arms:
            label, pdir, lr_tmpl, by_tmpl = entry[0], entry[1], entry[2], entry[3]
            g = gather_all_horizons(
                pdir, lr_tmpl.format(t=tgt), by_tmpl.format(t=tgt) if by_tmpl else None
            )
            thr = _resolve_thresh(thresh_by_label.get(label), tgt)
            for h in horizons:
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
                    cs = apply_threshold_gate(cs, p, thr)
                truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in support], dtype=float)
                ev = truth > 0
                c = crps_ensemble(truth, cs)
                ey = cs.mean(1)
                ratio = ey[ev] / truth[ev] if ev.sum() else np.array([])
                rows.append(
                    dict(
                        target=tgt,
                        model=label,
                        h=h,
                        N=len(support),
                        n_event=int(ev.sum()),
                        AP=average_precision((truth > 0).astype(float), p),
                        Brier=brier((truth > 0).astype(float), p),
                        crps_all=float(c.mean()),
                        crps_events=float(c[ev].mean()) if ev.sum() else float("nan"),
                        crps_none=float(c[~ev].mean()) if (~ev).sum() else float("nan"),
                        size_ratio=float(np.median(ratio)) if ratio.size else float("nan"),
                        pos_mcr=float(np.mean(ratio)) if ratio.size else float("nan"),
                        gate_source=("gate-head" if has_gate else "frac(samples>0)"),
                    )
                )
            del g  # free this arm's cubes before loading the next (OOM guard)
        # persistence: last observed truth[m0-1] held for all h (cheap; streamed last)
        if add_persistence:
            gp = _persistence_gathered(tmap, support, horizons, months_loaded=months)
            for h in horizons:
                cs = np.stack([gp[(m0, h, u)][0] for (m0, u) in support])
                p = (cs > 0).mean(1)
                truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in support], dtype=float)
                ev = truth > 0
                c = crps_ensemble(truth, cs)
                ey = cs.mean(1)
                ratio = ey[ev] / truth[ev] if ev.sum() else np.array([])
                rows.append(
                    dict(
                        target=tgt, model="persistence", h=h, N=len(support),
                        n_event=int(ev.sum()),
                        AP=average_precision((truth > 0).astype(float), p),
                        Brier=brier((truth > 0).astype(float), p),
                        crps_all=float(c.mean()),
                        crps_events=float(c[ev].mean()) if ev.sum() else float("nan"),
                        crps_none=float(c[~ev].mean()) if (~ev).sum() else float("nan"),
                        size_ratio=float(np.median(ratio)) if ratio.size else float("nan"),
                        pos_mcr=float(np.mean(ratio)) if ratio.size else float("nan"),
                        gate_source="frac(samples>0)",
                    )
                )
    return rows


def block_bootstrap_crps(truth_parquet, registry, tgt, model_label, horizons=HORIZONS, B=1000, seed=0):
    """Block bootstrap over ORIGINS (C-221): resample the origin set with replacement, recompute
    crps_all(h). Returns {h: (lo, hi)} 90% CIs. Deterministic given seed."""
    label, pdir, lr_tmpl, by_tmpl = (
        model_label,
        dict((e[0], e[1]) for e in registry)[model_label],
        dict((e[0], e[2]) for e in registry)[model_label],
        dict((e[0], e[3]) for e in registry)[model_label],
    )
    g = gather_all_horizons(pdir, lr_tmpl.format(t=tgt), by_tmpl.format(t=tgt) if by_tmpl else None)
    support = _fixed_support({label: g}, horizons)
    origins = sorted({m0 for (m0, _u) in support})
    by_origin = {o: [(m0, u) for (m0, u) in support if m0 == o] for o in origins}
    months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
    tmap = _truth_map(truth_parquet, "lr_" + tgt + "_best", months)
    rng = np.random.default_rng(seed)
    out = {}
    for h in horizons:
        vals = []
        for _b in range(B):
            pick = rng.choice(origins, size=len(origins), replace=True)
            cells = [c for o in pick for c in by_origin[o]]
            cs = np.stack([g[(m0, h, u)][0] for (m0, u) in cells])
            truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in cells], dtype=float)
            vals.append(float(crps_ensemble(truth, cs).mean()))
        out[h] = (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))
    return out


def _selftest():
    # h indexing: month m0 -> h=1 ; m0+35 -> h=36
    assert (457 - 457 + 1) == 1 and (492 - 457 + 1) == 36
    # persistence constructor: last observed held for all h
    tmap = {(456, 7): 5.0}
    pg = _persistence_gathered(tmap, [(457, 7)], horizons=(1, 2, 3))
    assert pg[(457, 1, 7)][0][0] == 5.0 and pg[(457, 3, 7)][0][0] == 5.0
    # fixed support: only cells with ALL horizons, intersected across models
    gA = {(457, 1, 7): (np.zeros(2), None), (457, 2, 7): (np.zeros(2), None),
          (457, 1, 8): (np.zeros(2), None)}  # cell 8 missing h=2
    gB = {(457, 1, 7): (np.zeros(2), None), (457, 2, 7): (np.zeros(2), None)}
    assert _fixed_support({"a": gA, "b": gB}, horizons=(1, 2)) == [(457, 7)]
    # crps reused from the frozen tool (point mass at truth => 0)
    assert abs(crps_ensemble(np.array([3.0]), np.array([[3.0, 3.0]]))[0]) < 1e-9
    print("SELFTEST PASS — horizon indexing + persistence + fixed-support + frozen-crps reuse verified.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        _selftest()
        sys.exit(0)
    truth_parquet = args[0]
    targets = ["sb", "ns", "os"]
    add_persistence = False
    boot = 0
    out_csv = "rollout_skill.csv"
    entries = []
    for a in args[1:]:
        if a.startswith("--targets"):
            targets = a.split("=", 1)[1].split(",")
        elif a == "--persistence":
            add_persistence = True
        elif a.startswith("--boot"):
            boot = int(a.split("=", 1)[1])
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
    rows = score_horizons(truth_parquet, entries, targets, add_persistence=add_persistence)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows: {df.model.nunique()} models × {len(targets)} targets × 36 h)")
    # compact crossover print: crps_all at h in {1,6,12,24,36}
    for tgt in targets:
        print(f"\n=== {tgt}: crps_all by horizon ===")
        sub = df[df.target == tgt].pivot_table(index="model", columns="h", values="crps_all")
        cols = [c for c in (1, 3, 6, 12, 18, 24, 36) if c in sub.columns]
        print(sub[cols].to_string(float_format=lambda x: f"{x:.4f}"))
