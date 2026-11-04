"""EXP-03, the BODY: does the upsample arm forecast HIGHER or LOWER on cells that fire?

⭐ WHY. Every EXP-02/03 readout so far is the GATE (AP). AP is rank-based: it is exactly invariant
to monotone gate miscalibration and it says nothing whatever about magnitude. This asks Simon's
question instead -- among cells that fire, does the forecast go up or down, and how is the
distribution shaped.

**C-351 slots.** Population = 13 origins x 64,818 global land cells = 842,634 cell-months PER
HORIZON per arm. Transform = NONE; the draws are already count space. Estimator = the arm's 16
posterior draws; `E[Y]` is their mean; CRPS from the FROZEN ruler, imported, plug-in on 16 members.
Truth = the arms' shared global pull, `lr_sb_best`. ⛔ NOT `V2_TRUTH`.

⛔ The ratio is SWEPT across quantiles, never reported as `size_ratio`. `size_ratio` is a median
and was measured to be EXACTLY invariant to a factor-1000 error on up to 49% of event cells
(M3 Finding C), which is precisely the quantity under discussion here.

⚠️ Equal S matters and is asserted: all arms are S=16, so the plug-in CRPS bias is common and
differences are interpretable while levels are not comparable to anything scored at another S.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
sys.path.insert(0, str(REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
from lodestar_score import crps_ensemble  # noqa: E402

#: ⛔ FAO-02 is LOCKED: CRPS primary, with QS99/Brier/MCR as guardrails. QS99 is therefore a
#: LICENSED tail-sensitive score and is used as such here -- no new score is invented (twCRPS,
#: LogScore and PIT are all REJECTED by that framework and are not used).
#: ⚠️ M5/#482 records that QS99's definition as `t0_score.py` computes it has NOT been reconciled
#: with FAO-02's guardrail wording. Treated here as a within-experiment comparison only.
#: ⚠️ And with S=16 draws the 0.99 quantile IS essentially the top member, so QS99 on this cube is
#: a statement about the ensemble MAXIMUM, not about a well-estimated 99th percentile.
TAU = 0.99

#: Truth bands for the stratified CRPS. ⭐ The point: `crps_events` pools cells whose truth spans
#: 1 to 2,482, and CRPS is in count units, so a handful of enormous cells can dominate the mean and
#: hide everything in the bulk. Bands are fixed BEFORE looking at any arm.
BANDS = ((1, 2), (3, 5), (6, 20), (21, 100), (101, 10**9))

DELIVERY = "data/generated/calibration_delivery_20261103"
SEEDS = ["fortytwo", "fortythree", "fortyfour", "fortyfive", "fortyseven"]
HORIZONS = (1, 36)
QS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
print(__doc__, flush=True)

tb = pq.read_table(MODELS / "ctlfullzero_fortytwo" / "data/raw/calibration_datafactory_df.parquet",
                   columns=["month_id", "priogrid_id", "lr_sb_best"])
_mo, _un = tb["month_id"].to_numpy(), tb["priogrid_id"].to_numpy()
CELLS, M0 = np.unique(_un), int(_mo.min())
GRID = np.zeros((int(_mo.max()) - M0 + 1, CELLS.size), np.float64)
GRID[_mo - M0, np.searchsorted(CELLS, _un)] = tb["lr_sb_best"].to_numpy()
del tb, _mo, _un
print(f"truth grid {GRID.shape}\n", flush=True)


def _qs(truth, draws, tau):
    """Pinball loss at `tau` against the ensemble's tau-quantile. Lower is better."""
    q = np.quantile(draws, tau, axis=1)
    d = truth - q
    return float(np.maximum(tau * d, (tau - 1.0) * d).mean())


def arm_stats(arm: str) -> dict:
    """Per horizon: pooled CRPS split + magnitude distributions on event cells."""
    acc = {h: dict(draws=[], truth=[]) for h in HORIZONS}
    with tempfile.TemporaryDirectory() as td:
        #: ⚠️ the tarball carries each ARM's own run timestamp, so it must be globbed, never
        #: spelled -- hardcoding one arm's name made this script die on the second arm.
        tars = sorted((MODELS / arm / DELIVERY / "draws").glob("*_lr.tar.zst"))
        assert len(tars) == 1, f"{arm}: {len(tars)} lr draw archives, want exactly 1"
        subprocess.run(["tar", "-I", "zstd", "-xf", str(tars[0]), "-C", td,
                        "--wildcards", "*/lr_sb_best/*"], check=True)
        root = next(Path(td).glob("predictions_*"))
        for p in sorted(root.glob("origin_*"), key=lambda q: int(q.name.split("_")[1])):
            d = p / "lr_sb_best"
            y = np.load(d / "y_pred.npy")
            z = np.load(d / "identifiers.npz")
            tt, uu = z["time"].astype(np.int64), z["unit"].astype(np.int64)
            m0 = int(tt.min())
            assert y.shape[1] == 16, f"{arm}: S={y.shape[1]}, not 16"
            for h in HORIZONS:
                month = m0 + h - 1
                sel = tt == month
                assert sel.sum() == CELLS.size, f"{arm} h{h}: {sel.sum()} rows, want {CELLS.size}"
                acc[h]["draws"].append(y[sel].astype(np.float64))
                acc[h]["truth"].append(GRID[tt[sel] - M0, np.searchsorted(CELLS, uu[sel])])
            del y
    out = {}
    for h in HORIZONS:
        cs = np.concatenate(acc[h]["draws"]); tr = np.concatenate(acc[h]["truth"])
        c = crps_ensemble(tr, cs); ey = cs.mean(1); ev = tr > 0
        ratio = ey[ev] / tr[ev]
        fires = ey > 0
        out[h] = dict(
            crps_all=float(c.mean()), crps_events=float(c[ev].mean()),
            crps_none=float(c[~ev].mean()),
            n=int(tr.size), n_event=int(ev.sum()),
            ey_event_q={str(q): float(np.quantile(ey[ev], q)) for q in QS},
            ratio_q={str(q): float(np.quantile(ratio, q)) for q in QS},
            silent_on_event=float((cs[ev] == 0).all(1).mean()),
            qs99=float(_qs(tr, cs, TAU)), qs99_events=float(_qs(tr[ev], cs[ev], TAU)),
            crps_band={f"{lo}-{hi if hi < 10**9 else 'inf'}": (
                float(c[(tr >= lo) & (tr <= hi)].mean()) if ((tr >= lo) & (tr <= hi)).any()
                else float("nan")) for lo, hi in BANDS},
            n_band={f"{lo}-{hi if hi < 10**9 else 'inf'}": int(((tr >= lo) & (tr <= hi)).sum())
                    for lo, hi in BANDS},
            crps_share_band={f"{lo}-{hi if hi < 10**9 else 'inf'}": float(
                c[(tr >= lo) & (tr <= hi)].sum() / c.sum()) for lo, hi in BANDS},
            fires_share=float(fires.mean()),
            ey_when_fires_q={str(q): float(np.quantile(ey[fires], q)) for q in QS},
            truth_event_q={str(q): float(np.quantile(tr[ev], q)) for q in QS},
        )
    return out


res = {}
for s in SEEDS:
    for a in (f"ctlfullzero_{s}", f"upctwofullzero_{s}"):
        res[a] = arm_stats(a)
    print(f"  seed {s} done", flush=True)

def show(h, key, label, fmt="{:+.4f}", sub=None):
    print(f"\n  {label}   (h{h})")
    print(f"  {'seed':<12}{'control':>12}{'upctwo':>12}{'Δ':>12}   direction")
    print("  " + "-" * 60)
    deltas = []
    for s in SEEDS:
        c = res[f"ctlfullzero_{s}"][h]; t = res[f"upctwofullzero_{s}"][h]
        cv, tv = (c[key][sub], t[key][sub]) if sub else (c[key], t[key])
        d = tv - cv; deltas.append(d)
        print(f"  {s:<12}{cv:>12.4f}{tv:>12.4f}{d:>+12.4f}   {'HIGHER' if d>0 else 'lower'}")
    a = np.array(deltas)
    print(f"  {'':<12}{'':>12}{'':>12}{'':>12}   {int((a>0).sum())}/5 higher, "
          f"mean Δ {a.mean():+.4f}, sd {a.std(ddof=1):.4f}, unanimous: {bool(np.all(a>0) or np.all(a<0))}")

for h in HORIZONS:
    print("\n" + "=" * 92)
    print(f"h{h}  —  n = {res['ctlfullzero_fortytwo'][h]['n']:,} cell-months, "
          f"{res['ctlfullzero_fortytwo'][h]['n_event']:,} TRUE event cells")
    print("=" * 92)
    show(h, "crps_all", "crps_all   (FAO-02 PRIMARY, lower is better)")
    show(h, "crps_events", "crps_events   (on true event cells, lower is better)")
    show(h, "crps_none", "crps_none   (on true zero cells, lower is better)")
    show(h, "silent_on_event", "share of TRUE EVENT cells where ALL 16 draws are ZERO")
    print(f"\n  E[Y] on TRUE EVENT cells — the distribution, swept   (h{h})")
    print(f"  {'quantile':<10}" + "".join(f"{s[:9]:>11}" for s in SEEDS))
    for q in QS:
        line = f"  q{q:<9.2f}"
        for s in SEEDS:
            c = res[f"ctlfullzero_{s}"][h]["ey_event_q"][str(q)]
            t = res[f"upctwofullzero_{s}"][h]["ey_event_q"][str(q)]
            line += f"{t-c:>+11.4f}"
        print(line + "   <- Δ (upctwo − control)")
    print(f"  {'(control)':<10}" + "".join(
        f"{res[f'ctlfullzero_{s}'][h]['ey_event_q'][str(0.50)]:>11.4f}" for s in SEEDS)
        + "   <- control median, for scale")
    print(f"  {'(truth)':<10}" + "".join(
        f"{res[f'ctlfullzero_{s}'][h]['truth_event_q'][str(0.50)]:>11.4f}" for s in SEEDS)
        + "   <- TRUTH median on the same cells")

json.dump(res, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print(f"\nwrote {Path(__file__).with_suffix('.json')}")
