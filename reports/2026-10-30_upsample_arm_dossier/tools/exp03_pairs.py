"""EXP-03 readout: ΔAP at h1 and h36 across the COMPLETE seed pairs, taken off the GATE.

⭐ WHY OFF THE GATE. `P-23` reads AP from the gate. The delivered `gate_endpoints.tar.zst` carries,
per origin, `by_sb_best/y_pred_h{1,36}.npy` [cells, 16] with `identifiers_h{1,36}.npz` holding
`time`/`unit` IN BAND. ⇒ no (H,W)->priogrid arithmetic and none of the lr/gate row-alignment the
plan's Step 4 warns about, because AP needs the gate alone.

**C-351 slots.** Population = 13 origins x 64,818 global land cells = 842,634 cell-months PER
HORIZON, per arm. Transform = none (the gate is already a probability). Estimator = posterior-mean
gate (mean over the arm's 16 draws), pooled over origins; AP from the FROZEN ruler, imported.
Truth = the arms' shared global pull, `lr_sb_best > 0`. ⛔ NOT `V2_TRUTH` (retired Africa crop).

**G-0 gates everything.** Seeds 42/43 are published at h1 as +0.0049 / +0.0047. This must
reproduce both within 0.0005 or it REFUSES to print a single figure.

Paired origin-block bootstrap, n_boot=400 (the committed default in
`scripts/ap_block_bootstrap.py`): one origin draw per replicate scores BOTH arms of a pair on the
same resampled origin set, so the correlation between arms is carried rather than assumed away.
"""
import json, subprocess, sys, tempfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

#: ⛔ views-models renamed the twelve arms (#702/#704), so a directory may NOT be spelled
#: from a pre-registered name. `exp03_arms` derives the mapping from each delivery's own
#: manifests and refuses an ambiguous one. Pre-registered names stay the identifiers in
#: every readout below, because that is what PREREG.md and the ledgers use.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp03_arms import DELIVERY, REPO, arm_dir, arm_directories  # noqa: E402

_ARMS = arm_directories()
sys.path.insert(0, str(REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
from lodestar_score import average_precision  # noqa: E402

SEEDS = ["fortytwo", "fortythree", "fortyfour", "fortyfive", "fortysix", "fortyseven"]
HORIZONS = (1, 36)
N_BOOT, BOOT_SEED, CI = 400, 0, 0.90
PUBLISHED_H1 = {"fortytwo": 0.0049, "fortythree": 0.0047}
G0_TOL = 0.0005

print(__doc__, flush=True)

tb = pq.read_table(arm_dir("ctlfullzero_fortytwo", arms=_ARMS) / "data/raw/calibration_datafactory_df.parquet",
                   columns=["month_id", "priogrid_id", "lr_sb_best"])
_mo, _un = tb["month_id"].to_numpy(), tb["priogrid_id"].to_numpy()
CELLS, M0 = np.unique(_un), int(_mo.min())
GRID = np.zeros((int(_mo.max()) - M0 + 1, CELLS.size), np.float64)
GRID[_mo - M0, np.searchsorted(CELLS, _un)] = tb["lr_sb_best"].to_numpy()
del tb, _mo, _un
print(f"truth grid {GRID.shape} from the shared pull; {int((GRID > 0).sum()):,} positive "
      f"cell-months over all 384 months\n", flush=True)


def load_arm(arm):
    """-> {h: {origin: (posterior-mean gate, binary truth)}}. Extracts sb only, then discards."""
    out = {h: {} for h in HORIZONS}
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["tar", "-I", "zstd", "-xf", str(arm_dir(arm, arms=_ARMS) / DELIVERY /
                        "gate_endpoints.tar.zst"), "-C", td,
                        "--wildcards", "*/by_sb_best/*"], check=True)
        root = Path(td) / "gate_endpoints"
        for h in HORIZONS:
            for p in sorted(root.glob("origin_*"), key=lambda q: int(q.name.split("_")[1])):
                o = int(p.name.split("_")[1])
                g = np.load(p / "by_sb_best" / f"y_pred_h{h}.npy").astype(np.float64)
                z = np.load(p / "by_sb_best" / f"identifiers_h{h}.npz")
                tt, uu = z["time"].astype(np.int64), z["unit"].astype(np.int64)
                assert g.shape[0] == tt.size == uu.size, f"{arm} o{o} h{h} shape"
                assert np.unique(tt).size == 1, f"{arm} o{o} h{h}: >1 month in one slice"
                out[h][o] = (g.mean(1), (GRID[tt - M0, np.searchsorted(CELLS, uu)] > 0
                                         ).astype(np.float64))
    return out


def ap_of(per_origin, origins):
    return float(average_precision(
        np.concatenate([per_origin[o][1] for o in origins]),
        np.concatenate([per_origin[o][0] for o in origins])))


def one_cell(task, cache):
    s, h = task
    c, t = cache[f"ctlfullzero_{s}"][h], cache[f"upctwofullzero_{s}"][h]
    origins = sorted(c)
    assert sorted(t) == origins, f"seed {s} h{h}: origin sets differ between arms"
    ap_c, ap_t = ap_of(c, origins), ap_of(t, origins)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(origins, size=len(origins), replace=True)
        boots[b] = ap_of(t, pick) - ap_of(c, pick)
    lo, hi = np.quantile(boots, [(1 - CI) / 2, 1 - (1 - CI) / 2])
    return dict(seed=s, h=h, ap_ctl=ap_c, ap_trt=ap_t, diff=ap_t - ap_c, lo=float(lo),
                hi=float(hi), mde=float(hi - lo) / 2, n_org=len(origins),
                n=int(sum(c[o][0].size for o in origins)))


PAIRS = [s for s in SEEDS if f"upctwofullzero_{s}" in _ARMS]
for s in SEEDS:
    if s not in PAIRS:
        print(f"⛔ seed {s}: INCOMPLETE — upctwofullzero_{s} absent, the pair cannot be read",
              flush=True)
CACHE = {}
for s in PAIRS:
    for a in (f"ctlfullzero_{s}", f"upctwofullzero_{s}"):
        CACHE[a] = load_arm(a)
    print(f"  loaded seed {s}", flush=True)

with Pool(min(10, len(PAIRS) * 2)) as pool:
    rows = pool.map(partial(one_cell, cache=CACHE), [(s, h) for s in PAIRS for h in HORIZONS])

print("\n" + "=" * 100)
print("G-0  does this reproduce the two PUBLISHED h1 values?")
print("=" * 100)
ok = True
for s, want in PUBLISHED_H1.items():
    got = next((r["diff"] for r in rows if r["seed"] == s and r["h"] == 1), None)
    d = abs(got - want) if got is not None else float("inf")
    print(f"  seed {s:<12} published {want:+.4f}   computed "
          f"{'n/a' if got is None else f'{got:+.4f}'}   |Δ| {d:.5f}   "
          f"{'PASS' if d <= G0_TOL else 'FAIL'}")
    ok &= d <= G0_TOL
if not ok:
    raise SystemExit("\nVOID: G-0 FAILED. This pipeline does not reproduce the two published h1 "
                     "values, so nothing below is a measurement of what it claims. No figure from "
                     "this run may be quoted.")

print("\n" + "=" * 100)
print("ΔAP = upctwo − its OWN-SEED control · gate · sb · paired origin-block bootstrap · 90% CI")
print("=" * 100)
for h in HORIZONS:
    hr = sorted((r for r in rows if r["h"] == h), key=lambda r: SEEDS.index(r["seed"]))
    print(f"\n  h{h}     (n = {hr[0]['n']:,} cell-months over {hr[0]['n_org']} origins, per seed)")
    print(f"  {'seed':<12}{'AP ctl':>9}{'AP upctwo':>11}{'ΔAP':>10}{'90% CI':>23}{'MDE':>9}   resolved?")
    print("  " + "-" * 88)
    for r in hr:
        print(f"  {r['seed']:<12}{r['ap_ctl']:>9.4f}{r['ap_trt']:>11.4f}{r['diff']:>+10.4f}"
              f"   [{r['lo']:+.4f}, {r['hi']:+.4f}]{r['mde']:>9.4f}   "
              f"{'YES' if (r['lo'] > 0 or r['hi'] < 0) else 'no (straddles 0)'}")
    d = np.array([r["diff"] for r in hr])
    print(f"  ⇒ {len(d)} pairs: mean {d.mean():+.5f}  between-seed sd {d.std(ddof=1):.5f}  "
          f"min {d.min():+.4f}  max {d.max():+.4f}  signs {''.join('+' if x > 0 else '-' for x in d)}")
    print(f"    unanimous sign: {bool(np.all(d > 0) or np.all(d < 0))}   "
          f"median within-seed MDE: {np.median([r['mde'] for r in hr]):.5f}")

json.dump(rows, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print(f"\nwrote {Path(__file__).with_suffix('.json')}")
