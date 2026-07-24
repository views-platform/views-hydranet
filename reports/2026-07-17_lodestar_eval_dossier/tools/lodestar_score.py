"""LODESTAR RULER — the one frozen T=0 scorer (dossier 2026-07-17, spec 03 §C).

Scores model families on IDENTICAL support: same months, same cells (intersection), same truth, same
metrics. Switch (where is conflict): AP + Brier. Size-guesser (how many): all/event/zero-CRPS + size-ratio.
Deterministic. Fail-loud. FROZEN once the self-test passes — a change is a new version and re-scores all.

CLI:
  lodestar_score.py <truth_parquet> "<label>|<pred_dir>|lr_{t}_best|by_{t}_best" "<label2>|...|" [--targets sb,ns,os]
  (empty by-template => count-only model; P(conflict) is then the fraction of count samples > 0)
  Optional 5th field = th_gated composition (ADR-068): a-priori τ as a float ('0.5') or
  'baserate' (fixed per-target BASE_RATE) — zeros the body where gate < τ. Needs a by-template.
  Absent => classic soft/self path, byte-identical:  "...|lr_{t}_best|by_{t}_best|0.5"
  lodestar_score.py --selftest
"""

from __future__ import annotations
import glob
import sys
import numpy as np
import pandas as pd

MONTH_COL = "month_id"


# ---- metric primitives (hand-verifiable; covered by --selftest) ----
def crps_ensemble(truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Per-cell CRPS, energy form on sorted samples. truth (N,), samples (N,S) -> (N,)."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    a = np.abs(s - truth[:, None]).mean(1)
    w = 2 * np.arange(m) - m + 1
    b = (2.0 / (m * m)) * (w[None, :] * s).sum(1)
    return a - 0.5 * b


def average_precision(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (sklearn convention). Rank-based."""
    from sklearn.metrics import average_precision_score

    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(average_precision_score(y_bin, score))


def brier(y_bin: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_bin) ** 2))


# ---- th_gated composition (opt-in body transform; ADR-068 th_gated_<body>) ----
#: A-PRIORI per-target base rates (global conflict prevalence, fixed BEFORE any scoring — the
#: "retain above chance" recall-favoring threshold). NEVER computed from the scored months
#: (Goodhart; cf. the different-months scar). Pre-registered in the dossier 05_analysis_plan.
BASE_RATE = {"sb": 0.0077, "ns": 0.0034, "os": 0.0041}


def apply_threshold_gate(cs: np.ndarray, p: np.ndarray, thresh: float | None) -> np.ndarray:
    """Hard threshold-gate the body: keep a cell's FULL sample vector where ``gate >= thresh``,
    zero it entirely where ``gate < thresh`` (th_gated_<body>, ADR-068). ``cs`` (N,S) count
    samples, ``p`` (N,) gate prob. ``thresh=None`` => identity (soft/self arms byte-identical).

    Boundary is inclusive (``>=`` keeps). Note this is a BODY transform — the gate ``p`` itself is
    unchanged, so AP/Brier (which score ``p``) are identical to the soft-gated arm; only the count
    metrics (CRPS/E[y]/size-ratio) move. That is the whole point: same gate, decisive composition.
    """
    if thresh is None:
        return cs
    keep = (p >= thresh).astype(cs.dtype)  # (N,)
    return cs * keep[:, None]


# ---- gather ----
def _grid_col(df: pd.DataFrame) -> str:
    for c in ("priogrid_id", "priogrid_gid"):
        if c in df.columns:
            return c
    raise KeyError(f"no grid id column in {list(df.columns)}")


def gather_t0(pred_dir: str, lr_name: str, by_name: str | None):
    """Return dict (month,unit) -> (count_samples[S], switch_prob or None). T=0 = first month per origin."""
    out = {}
    origins = sorted(glob.glob(pred_dir + "/origin_*"), key=lambda p: int(p.split("_")[-1]))
    if not origins:
        raise FileNotFoundError(f"no origins in {pred_dir}")
    for od in origins:
        idf = np.load(f"{od}/{lr_name}/identifiers.npz", allow_pickle=True)
        t, u = idf["time"], idf["unit"]
        m0 = int(t.min())
        sel = t == m0
        cs = np.load(f"{od}/{lr_name}/y_pred.npy")[sel]  # (n, S)
        sw = None
        if by_name:
            sw = np.load(f"{od}/{by_name}/y_pred.npy")[sel].mean(1)  # (n,) switch prob
        for i, unit in enumerate(u[sel]):
            out[(m0, int(unit))] = (cs[i], None if sw is None else float(sw[i]))
    return out


def _resolve_thresh(spec: "str | None", tgt: str) -> "float | None":
    """A-priori τ for a th_gated arm: None (no threshold), 'baserate' -> fixed per-target
    BASE_RATE, or a literal float string. Never derived from the scored truth."""
    if spec is None:
        return None
    if spec == "baserate":
        return BASE_RATE[tgt]
    return float(spec)


def score_models(truth_parquet: str, registry: list, targets: list, coverage_tol=0.20):
    """registry: list of (label, pred_dir, lr_tmpl, by_tmpl_or_None[, thresh_spec_or_None]).

    ``thresh_spec`` (optional 5th field) opts a model into the th_gated composition: a-priori τ
    as a float string or 'baserate'. Absent/None => classic soft/self path, byte-identical.
    Returns list of row dicts.
    """
    tdf = pd.read_parquet(truth_parquet).reset_index()
    grid = _grid_col(tdf)
    thresh_by_label = {e[0]: (e[4] if len(e) > 4 else None) for e in registry}
    rows = []
    for tgt in targets:
        lr_full = "lr_" + tgt + "_best"
        gathered = {}
        for entry in registry:
            label, pred_dir, lr_tmpl, by_tmpl = entry[0], entry[1], entry[2], entry[3]
            lr = lr_tmpl.format(t=tgt)
            by = by_tmpl.format(t=tgt) if by_tmpl else None
            gathered[label] = gather_t0(pred_dir, lr, by)
        # common (month,unit) support across ALL models
        keysets = [set(g.keys()) for g in gathered.values()]
        common = sorted(set.intersection(*keysets))
        if len(common) == 0:
            raise ValueError(f"[{tgt}] EMPTY common support across models — FAIL (F1)")
        # coverage check
        for label, g in gathered.items():
            miss = 1 - len(common) / max(len(g), 1)
            if miss > coverage_tol:
                print(
                    f"  ⚠️  [{tgt}] {label} covers only {len(common)}/{len(g)} of its cells "
                    f"(missing {miss:.0%}) — row provisional (F5)"
                )
        # truth on common support
        months = sorted({m for m, _ in common})
        trmap = {}
        for m in months:
            sub = tdf[tdf[MONTH_COL] == m].set_index(grid)[lr_full]
            for mm, uu in common:
                if mm == m:
                    trmap[(mm, uu)] = float(sub.get(uu, 0.0))
        truth = np.array([trmap[k] for k in common], dtype=float)
        ybin = (truth > 0).astype(float)
        if not np.isfinite(truth).all():
            raise ValueError(f"[{tgt}] non-finite truth — FAIL (F1)")
        ev = truth > 0
        ze = ~ev
        for label, g in gathered.items():
            cs = np.stack([g[k][0] for k in common])  # (N, S)
            # switch prob: gate if present else fraction of count samples > 0
            has_gate = g[common[0]][1] is not None
            p = np.array([g[k][1] for k in common]) if has_gate else (cs > 0).mean(1)
            # th_gated (opt-in): hard-threshold the BODY at a-priori τ. Needs a real gate.
            thresh = _resolve_thresh(thresh_by_label.get(label), tgt)
            if thresh is not None:
                if not has_gate:
                    raise ValueError(
                        f"[{tgt}] {label}: th_gated needs a gate (by-template), none given — FAIL"
                    )
                cs = apply_threshold_gate(cs, p, thresh)  # body-only; p (AP/Brier) unchanged
            c = crps_ensemble(truth, cs)
            ey = cs.mean(1)
            ratio = ey[ev] / truth[ev] if ev.sum() else np.array([])
            sr = float(np.median(ratio)) if ratio.size else float("nan")  # size-ratio (MEDIAN)
            pm = float(np.mean(ratio)) if ratio.size else float("nan")  # pos-mcr (MEAN)
            if not np.isfinite(c).all():
                raise ValueError(f"[{tgt}] {label} non-finite CRPS — FAIL (F1)")
            rows.append(
                dict(
                    target=tgt,
                    model=label,
                    N=len(common),
                    n_event=int(ev.sum()),
                    n_zero=int(ze.sum()),
                    AP=average_precision(ybin, p),
                    Brier=brier(ybin, p),
                    crps_all=float(c.mean()),
                    crps_events=float(c[ev].mean()),
                    crps_none=float(c[ze].mean()),
                    size_ratio=sr,
                    pos_mcr=pm,
                    gate_source=("gate-head" if has_gate else "frac(samples>0)"),
                    thresh=(float(thresh) if thresh is not None else float("nan")),
                )
            )
    return rows


# ---- self-test (gate to freeze) ----
def _selftest():
    # CRPS: sample point-mass at truth => 0
    assert abs(crps_ensemble(np.array([3.0]), np.array([[3.0, 3.0, 3.0]]))[0]) < 1e-9
    # CRPS: samples all 0, truth 1 => a=1, b=0 => 1.0
    assert abs(crps_ensemble(np.array([1.0]), np.array([[0.0, 0.0]]))[0] - 1.0) < 1e-9
    # CRPS energy form vs brute force on a small case
    tr = np.array([2.0])
    sm = np.array([[0.0, 1.0, 4.0]])
    a = np.abs(sm[0] - 2).mean()
    b = np.mean([abs(x - y) for x in sm[0] for y in sm[0]])
    assert abs(crps_ensemble(tr, sm)[0] - (a - 0.5 * b)) < 1e-9
    # event/zero split
    truth = np.array([0.0, 0.0, 5.0, 3.0])
    ev = truth > 0
    c = np.array([0.1, 0.2, 9.0, 7.0])
    assert abs(c[ev].mean() - 8.0) < 1e-9 and abs(c[~ev].mean() - 0.15) < 1e-9
    # Brier: perfect prob => 0 ; worst => 1
    assert abs(brier(np.array([1.0, 0.0]), np.array([1.0, 0.0]))) < 1e-9
    assert abs(brier(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9
    # AP: perfect ranking => 1.0
    assert (
        abs(average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9
    )
    # size ratio: E[y]=truth => 1.0
    # --- th_gated composition (opt-in; body-only transform, additive) ---
    cs = np.array([[2.0, 4.0], [3.0, 5.0], [1.0, 1.0]])  # (3 cells, 2 samples)
    p = np.array([0.9, 0.2, 0.6])  # per-cell gate prob
    # thresh None => identity (existing arms byte-identical)
    assert np.array_equal(apply_threshold_gate(cs, p, None), cs)
    # thresh 0.5 => zero the entire sample vector of cells with gate < 0.5 (cell 1 only)
    g = apply_threshold_gate(cs, p, 0.5)
    assert np.array_equal(g[0], cs[0]) and np.array_equal(g[2], cs[2])  # kept (0.9, 0.6 >= 0.5)
    assert np.array_equal(g[1], np.zeros(2))  # zeroed (0.2 < 0.5)
    # boundary: gate == thresh is KEPT (>=)
    assert np.array_equal(apply_threshold_gate(cs, np.array([0.5, 0.5, 0.5]), 0.5), cs)
    # a-priori base rates are fixed constants (never fit on scored truth)
    assert set(BASE_RATE) == {"sb", "ns", "os"} and all(0 < v < 0.05 for v in BASE_RATE.values())
    print("SELFTEST PASS — ruler metric primitives + th_gated composition verified.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
        sys.exit(0)
    truth_parquet = sys.argv[1]
    targets = ["sb", "ns", "os"]
    entries = []
    for a in sys.argv[2:]:
        if a.startswith("--targets"):
            targets = a.split("=", 1)[1].split(",") if "=" in a else targets
            continue
        parts = a.split("|")
        label, pdir, lr_tmpl = parts[0], parts[1], parts[2]
        by_tmpl = parts[3] if len(parts) > 3 and parts[3] else None
        thresh_spec = parts[4] if len(parts) > 4 and parts[4] else None  # th_gated τ or 'baserate'
        entries.append((label, pdir, lr_tmpl, by_tmpl, thresh_spec))
    rows = score_models(truth_parquet, entries, targets)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for tgt in targets:
        print(
            f"\n=== lr_{tgt}_best (N={df[df.target == tgt]['N'].iloc[0]}, "
            f"events={df[df.target == tgt]['n_event'].iloc[0]}) ==="
        )
        sub = df[df.target == tgt][
            [
                "model",
                "AP",
                "Brier",
                "crps_all",
                "crps_events",
                "crps_none",
                "size_ratio",
                "pos_mcr",
                "thresh",
            ]
        ]
        print(sub.to_string(index=False))
    out = sys.argv[1].rsplit("/", 1)[0] if False else "lodestar.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
