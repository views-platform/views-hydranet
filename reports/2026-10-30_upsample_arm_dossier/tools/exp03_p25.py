"""P-25 — is the body untouched? Δ `crps_all` at h18, from data already on disk.

**The prediction** (`PREREG.md` §5b): *"The body is untouched"* — threshold
**|Δ `crps_all`| @h18 ≤ 0.006**, where Δ is each arm minus its **own-seed** control, so the
symmetric difference between the two configs is exactly ``{"model"}``.

⛔ **POPULATION: seeds 42 and 43 ONLY.** `PREREG.md` §3's population slot reads *"the bake-off's
own: `fullzero_*` controls, seeds **42** and **43**, L=300, paired"*. The four later seeds arrived
with `PREREG_EXP03_SEEDS.md` and are **not** part of this prediction. They are reported **beside**
the verdict as context, explicitly labelled as outside the registered population — reporting them
inside it would be moving the goalposts after the numbers exist.

⭐ **Why h18 is permitted here**, when §6 device 2 forbids an h18 verdict: §3's estimator slot
provisions scoring *"at **h1/6/12/18/24/30/36**, plus the oracle ceiling"*, so device 2 scopes to
§6's own verdict branch for `P-23`. Full reading in `FINDING_EXP03.md` §12 (#541). ⛔ No frozen
instrument's `HORIZONS` constant is changed; h18 is computed here.

**C-351 slots.** Population = 13 origins × 64,818 global land cells = **842,634 cell-months** at
h18, per arm. Transform = none (count space). Estimator = the arm's own **S = 16** posterior draws;
CRPS via the **frozen** plug-in `crps_ensemble`, imported. Truth = the arms' shared global pull,
`lr_sb_best`. ⛔ NOT `V2_TRUTH`, the retired 13,110-cell Africa crop.

⭐ **It also reports the concentration figures (#536/#537) beside every CRPS**, which is the new
default — and this is the **first measurement of `crps_effective_n` on real per-cell CRPS from a
delivered arm**. Everything quoted before this was a proxy.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp03_arms import DELIVERY, REPO, arm_dir, arm_directories  # noqa: E402

sys.path.insert(0, str(REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
sys.path.insert(0, str(REPO / "scripts"))
from lodestar_score import crps_ensemble  # noqa: E402
from rollout_ruler_core import crps_concentration  # noqa: E402

#: ⛔ §3's population. The later seeds are context, never part of the verdict.
PREREG_SEEDS = ("fortytwo", "fortythree")
CONTEXT_SEEDS = ("fortyfour", "fortyfive", "fortyseven")
H = 18
THRESHOLD = 0.006

print(__doc__, flush=True)
_ARMS = arm_directories()

tb = pq.read_table(
    arm_dir("ctlfullzero_fortytwo", arms=_ARMS) / "data/raw/calibration_datafactory_df.parquet",
    columns=["month_id", "priogrid_id", "lr_sb_best"],
)
mo, un = tb["month_id"].to_numpy(), tb["priogrid_id"].to_numpy()
CELLS, M0 = np.unique(un), int(mo.min())
GRID = np.zeros((int(mo.max()) - M0 + 1, CELLS.size), np.float64)
GRID[mo - M0, np.searchsorted(CELLS, un)] = tb["lr_sb_best"].to_numpy()
del tb, mo, un
print(f"truth grid {GRID.shape} from the arms' shared pull\n", flush=True)


def score(arm: str) -> dict:
    """-> crps_all at h18 plus its concentration, from the arm's own delivered draws."""
    per_cell, truth = [], []
    with tempfile.TemporaryDirectory() as td:
        tars = sorted((arm_dir(arm, arms=_ARMS) / DELIVERY / "draws").glob("*_lr.tar.zst"))
        assert len(tars) == 1, f"{arm}: {len(tars)} lr draw archives, want exactly 1"
        subprocess.run(
            ["tar", "-I", "zstd", "-xf", str(tars[0]), "-C", td, "--wildcards", "*/lr_sb_best/*"],
            check=True,
        )
        root = next(Path(td).glob("predictions_*"))
        for p in sorted(root.glob("origin_*"), key=lambda q: int(q.name.split("_")[1])):
            y = np.load(p / "lr_sb_best" / "y_pred.npy")
            z = np.load(p / "lr_sb_best" / "identifiers.npz")
            tt, uu = z["time"].astype(np.int64), z["unit"].astype(np.int64)
            sel = tt == int(tt.min()) + H - 1
            assert sel.sum() == CELLS.size, f"{arm}: h{H} has {sel.sum()} rows, want {CELLS.size}"
            tr = GRID[tt[sel] - M0, np.searchsorted(CELLS, uu[sel])]
            per_cell.append(crps_ensemble(tr, y[sel].astype(np.float64)))
            truth.append(tr)
            del y
    c, t = np.concatenate(per_cell), np.concatenate(truth)
    conc = crps_concentration(c, t, where=f"{arm} h{H}")
    return {"crps_all": float(c.mean()), "n": int(c.size), **conc}


res = {}
for s in PREREG_SEEDS + CONTEXT_SEEDS:
    for a in (f"ctlfullzero_{s}", f"upctwofullzero_{s}"):
        if a in _ARMS:
            res[a] = score(a)
    print(f"  scored seed {s}", flush=True)


def block(label: str, seeds, *, verdict: bool) -> list[float]:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    print(f"  {'seed':<12}{'crps_all ctl':>14}{'crps_all arm':>14}{'Δ':>11}"
          f"{'|Δ| ≤ 0.006?':>14}{'eff n ctl':>12}{'top band':>10}")
    print("  " + "-" * 92)
    ds = []
    for s in seeds:
        c, t = res.get(f"ctlfullzero_{s}"), res.get(f"upctwofullzero_{s}")
        if c is None or t is None:
            print(f"  {s:<12}  INCOMPLETE — the pair cannot be read")
            continue
        d = t["crps_all"] - c["crps_all"]
        ds.append(d)
        print(f"  {s:<12}{c['crps_all']:>14.6f}{t['crps_all']:>14.6f}{d:>+11.6f}"
              f"{('YES' if abs(d) <= THRESHOLD else 'NO'):>14}"
              f"{c['crps_effective_n']:>12.2f}{c['crps_concentration']:>10.4f}")
    if ds and verdict:
        a = np.array(ds)
        ok = bool(np.all(np.abs(a) <= THRESHOLD))
        print(f"\n  ⇒ max |Δ| = {np.abs(a).max():.6f} against the pre-registered {THRESHOLD}")
        print(f"  ⇒ P-25 VERDICT: {'CONFIRMED' if ok else 'FALSIFIED'} "
              f"({'every' if ok else 'not every'} pre-registered seed is within threshold)")
    return ds


block(f"P-25 — THE VERDICT. h{H}, seeds 42 and 43, the PRE-REGISTERED population (§3)",
      PREREG_SEEDS, verdict=True)
block("CONTEXT ONLY — ⛔ outside P-25's registered population, NOT part of the verdict",
      CONTEXT_SEEDS, verdict=False)

print("\n" + "=" * 100)
print(f"⭐ crps_effective_n on REAL per-cell CRPS at h{H} — first measurement, not a proxy")
print("=" * 100)
for a in sorted(res):
    r = res[a]
    print(f"  {a:<28} eff n {r['crps_effective_n']:>10.2f} of {r['n']:,}   "
          f"top band {r['crps_concentration']:.4f}   cells {r['crps_band_n']['101+']}")
