"""E6d PRECONDITION - prove window->grid alignment by CONTENT, not by offset arithmetic.

A misaligned crop would manufacture exactly the crop-vs-window difference E6d measures, so this
must pass before any forward pass runs. Coordinates are recovered by locating the sampler's window
data inside the global grid
alignment is then true by construction rather than by assumption.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
DIAG = HN / "reports/2026-10-12_underfire_diagnosis"
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(HN))
from vehicle import TARGETS, grid_rowcol, pull  # noqa: E402

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.utils.feature_scaler import FeatureScaler  # noqa: E402
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402

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
print(f"handler data shape {handler.data.shape}  axes {handler.axes}")

from views_hydranet.utils.curriculum import CurriculumLearner  # noqa: E402
from views_hydranet.utils.volume_sampler import VolumeSampler  # noqa: E402

sampler = VolumeSampler(handler, cfg)
planner = CurriculumLearner(cfg, handler)
target, threshold = planner.get_lesson(0)
batch, qualified = sampler.get_batch(target, threshold, batch_size=1)
w = batch[0]
print(f"lesson 0: target={target} threshold={threshold} qualified_count={qualified}")
print(f"window data shape {w.data.shape}")

# locate the window inside the handler's grid by FULL-BLOCK equality (all T, all C).
# Heuristic single-slice matching is ambiguous: channels 3/4 are constant-valued and match
# anywhere flat. Full-block equality is decisive and assumes nothing about the geometry.
H = handler.data
needle = np.asarray(w.data)
dim = cfg["window_dim"]
Tw = needle.shape[0]
# cheap prefilter on one slice, then full-block confirm
t_i, c_i = 299, 4
pat = needle[t_i, :, :, c_i]
cands = [(r0, c0)
         for r0 in range(H.shape[1] - dim + 1)
         for c0 in range(H.shape[2] - dim + 1)
         if np.array_equal(H[t_i, r0:r0 + dim, c0:c0 + dim, c_i], pat)]
print(f"\nprefilter candidates: {len(cands)}")
hits = [(r0, c0) for r0, c0 in cands
        if np.array_equal(H[:Tw, r0:r0 + dim, c0:c0 + dim, :], needle)]
print(f"full-block (all {Tw} T x all {H.shape[3]} C) matches: {len(hits)} -> {hits[:4]}")
if len(hits) != 1:
    print("⛔ NOT UNIQUELY LOCATABLE - alignment unproven, E6d must not run")
    raise SystemExit(1)
r0, c0 = hits[0]
print(f"✅ ALIGNMENT PROVEN: window sits at (r0, c0) = ({r0}, {c0})")
print(f"   handler grid {H.shape[1]}x{H.shape[2]} vs vehicle volume 360x720 -> "
      f"{'SAME INDEX SPACE' if (H.shape[1], H.shape[2]) == (360, 720) else '⛔ DIFFERENT'}")

# ---- eligible-anchor-set size at every hot-end lesson index: E6c's first input ----
print("\n=== qualified_count per lesson index 0-29 (the recal selection rule) ===")
qs = []
for wi in range(30):
    tgt, thr = planner.get_lesson(wi)
    _, q = sampler.get_batch(tgt, thr, batch_size=1)
    qs.append((wi, tgt, int(thr), int(q)))
for wi, tgt, thr, q in qs:
    print(f"  lesson {wi:>2}  target={tgt:<12} threshold={thr:>4}  qualified={q:>6}")
qq = np.array([q for *_, q in qs])
print(f"\n  qualified_count over the 30 recal indices: min {qq.min()}  "
      f"median {np.median(qq):.0f}  max {qq.max()}")
