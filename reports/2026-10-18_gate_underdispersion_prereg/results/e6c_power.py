"""E6c - the realised window union and the window-level bootstrap sd(b).

PREREG Amendment 5: sufficiency is a MEASURED property of the realised draw, never a nominal
product. sd(b) is bootstrapped over WINDOWS (the sampling unit), not cells. No window count is
sufficient a priori. Power gate: sd(b) < 0.027.

Needs no forward pass: the crop population is read off the global-pass gate already in the npz.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
DIAG = HN / "reports/2026-10-12_underfire_diagnosis"
EV = HN / "reports/2026-10-16_pos_weight_arm_outcome/evidence"
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(HN))
from vehicle import GRID_COLS, GRID_ROWS, TARGETS, grid_rowcol, pull  # noqa: E402

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.utils.curriculum import CurriculumLearner  # noqa: E402
from views_hydranet.utils.feature_scaler import FeatureScaler  # noqa: E402
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402
from views_hydranet.utils.volume_sampler import VolumeSampler  # noqa: E402

DIM, MARGIN = 32, 13
N_WIN = [30, 60, 120, 300]
EPS = 1e-9


def lg(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def ml_slope(x, yy, iters=80):
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        W = p * (1 - p)
        r = yy - p
        H = np.array([[W.sum(), (W * x).sum()], [(W * x).sum(), (W * x * x).sum()]])
        try:
            d = np.linalg.solve(H + 1e-9 * np.eye(2), np.array([r.sum(), (r * x).sum()]))
        except np.linalg.LinAlgError:
            return np.nan
        a += d[0]
        b += d[1]
        if np.abs(d).max() < 1e-11:
            break
    return b


ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
spec = importlib.util.spec_from_file_location(
    "hp", MODELS / "amber_kestrel/configs/config_hyperparameters.py")
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)
cfg = hp.get_hp_config()

df = pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
tr = df[(df.month_id >= 121) & (df.month_id <= 456)].copy()
r, c = grid_rowcol(tr.priogrid_id.to_numpy())
tr["row"], tr["col"] = r, c
tr["c_id"] = 0
tr = tr.rename(columns=dict(zip(TARGETS, cfg["regression_targets"])))
handler = VolumeHandler.from_df(FeatureScaler(cfg).fit_transform(tr), cfg)
sampler = VolumeSampler(handler, cfg)
planner = CurriculumLearner(cfg, handler)
tvh = sampler.get_train_volume()
p_row, p_col = tvh.spatial_offset
h_max = tvh.data.shape[1]
print(f"train volume {tvh.data.shape}  spatial_offset=({p_row}, {p_col})")

# cache activity per subject (no rng consumed), so _generate_window is called with identical
# rng draw order to get_batch
act = {}
for name in planner.subjects:
    idx = tvh.channel_map.index(name)
    act[name] = np.count_nonzero(tvh.data[..., idx], axis=0)


def coords(vh):
    """Recover (r0, c0) from the returned handler's spatial_offset."""
    nr, nc = vh.spatial_offset
    return int(p_row + (h_max - DIM - nr)), int(nc - p_col)


# validate the formula against the content-proven window at lesson 0
tgt0, thr0 = planner.get_lesson(0)
vh0 = sampler._generate_window(tvh, act[tgt0], thr0)
got = coords(vh0)
ok = np.array_equal(
    tvh.data[:vh0.data.shape[0], got[0]:got[0] + DIM, got[1]:got[1] + DIM, :],
    np.asarray(vh0.data))
print(f"coord formula on lesson 0 -> {got}   content-verified: {ok}")
if not ok:
    raise SystemExit("⛔ coord formula does not reproduce the window content - stop")

# draw windows under the locked rule: get_lesson(w mod 30) for w in range(N)
rows = []
for w in range(max(N_WIN)):
    tgt, thr = planner.get_lesson(w % 30)
    vh = sampler._generate_window(tvh, act[tgt], thr)
    rows.append(coords(vh))

# land-cell lookup for the npz population
cells = np.sort(df.priogrid_id.unique())
rr, cc = grid_rowcol(cells)
land_idx = -np.ones((GRID_ROWS, GRID_COLS), dtype=np.int64)
land_idx[rr - 1, cc - 1] = np.arange(len(cells))

z = np.load(EV / "p2_result_amber.npz")
gate = z["gate"].mean(0)                 # (13, n_land)  global-pass posterior mean
truth = z["truth"]                       # (13, n_land)
b_grid = ml_slope(lg(gate.ravel().astype(np.float64)), truth.ravel().astype(np.float64))
print(f"\nb_grid (bin-free, full population) = {b_grid:.4f}\n")

def win_cells(r0, c0):
    sl = land_idx[r0 + MARGIN:r0 + DIM - MARGIN, c0 + MARGIN:c0 + DIM - MARGIN].ravel()
    return sl[sl >= 0]

print(f"{'N win':>6} {'nominal':>10} {'UNION':>9} {'dup':>6} {'cell-mo':>9} "
      f"{'pos':>6} {'b_crop':>8} {'sd(b) win-boot':>15} {'gate':>7}")
rng = np.random.default_rng(42)
for N in N_WIN:
    per = [win_cells(*rows[i]) for i in range(N)]
    nominal = sum(len(p) for p in per)
    union = np.unique(np.concatenate(per)) if nominal else np.array([], dtype=np.int64)
    if union.size == 0:
        print(f"{N:>6} {nominal:>10} {0:>9}  no land in interiors")
        continue
    g = gate[:, union].ravel().astype(np.float64)
    y = truth[:, union].ravel().astype(np.float64)
    bc = ml_slope(lg(g), y)
    bs = []
    for _ in range(200):
        pick = rng.integers(0, N, N)
        u = np.unique(np.concatenate([per[i] for i in pick]))
        if u.size < 20:
            continue
        gb = gate[:, u].ravel().astype(np.float64)
        yb = truth[:, u].ravel().astype(np.float64)
        if not 0 < yb.mean() < 1:
            continue
        v = ml_slope(lg(gb), yb)
        if np.isfinite(v):
            bs.append(v)
    sd = np.std(bs, ddof=1) if len(bs) > 1 else np.nan
    print(f"{N:>6} {nominal:>10} {union.size:>9} {nominal/union.size:>6.1f}x "
          f"{union.size*13:>9} {int(y.sum()):>6} {bc:>8.4f} {sd:>15.4f} "
          f"{'PASS' if sd < 0.027 else 'FAIL':>7}")
u300 = np.unique(np.concatenate([win_cells(*rows[i]) for i in range(300)]))
g300 = gate[:, u300].ravel().astype(np.float64)
y300 = truth[:, u300].ravel().astype(np.float64)
b_crop300 = ml_slope(lg(g300), y300)
print(f"\nF5' selection confound  b_grid - b_crop(300) : {b_grid - b_crop300:+.4f}")
