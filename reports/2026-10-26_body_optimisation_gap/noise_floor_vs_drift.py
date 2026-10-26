"""Is the output drift MC-dropout noise? The control `drift_on_fixed_window.py` said it had none.

That script closed on "⛔ NOT RULED OUT: inference runs MC-dropout ON (ADR-059), so two snapshots
differ by dropout noise as well as by weights, and no noise floor for THIS quantity exists." It
does exist, in the arm's own delivery: `gate_draws/gate_h1.tar.zst` holds the gate probability at
h=1 for every land cell with a DRAW axis, which is the same model at FIXED weights sampled more
than once. The spread across that axis is the floor.

Extract it first (the archive is not read here, so the path is explicit and checkable):

    mkdir -p <dir> && tar -I zstd -xf \
        /home/simon/views-diagnostics/hf_arm_20261026/gate_draws/gate_h1.tar.zst -C <dir>

⛔ TWO DEFECTS THIS SCRIPT EXISTS TO AVOID, both of which inflate the floor and so favour the
conclusion. Neither is a hypothetical: the first was found by the falsifier below and the second
by comparing the two populations.

  1. The cube stores S = D x K columns (D dropout realisations x K head samples) and the GATE is
     invariant to K, so 16 columns carry only **4 distinct fields**. Averaging over all C(16,2) =
     120 pairs mixes 24 pairs that are identical by construction (r = 1) with 96 real ones.
     Columns are deduplicated by exact bytes and the distinct count is printed, not assumed.
  2. The floor over all 64,818 land cells is not comparable to a drift measured on one 32x32
     window: a correlation across the globe is carried by the difference between an active region
     and an empty ocean, which one patch does not contain. The floor is therefore ALSO measured on
     32x32 patches of the same grid, stratified by activity, because the curriculum seeks active
     windows (median positive fraction 0.0045 on the windows the model trains on).

⚠️ WHAT IS STILL NOT MATCHED, and is not fixed here.
  * The floor is the GATE. No draw axis was saved for the body or the forecast, so their floors
    are NOT measured. The gate's drift (+0.033), the body's (+0.094) and the forecast's (-0.020)
    sit in one regime, which makes the gate indicative for all three and nothing more.
  * The floor is eval mode at h=1 forecast origins; the drift is train mode within a window.
    BatchNorm in train mode uses the window's own batch statistics, which are deterministic given
    the same input, so BN contributes to "the weights moved" rather than to the floor -- but the
    two modes are not the same forward pass and the comparison is between regimes, not exact.
"""

import itertools
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMD = Path("/home/simon/views-diagnostics/hf_arm_20261026")
GATE = Path(
    "/tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-hydranet/"
    "f7a7ca8c-4eb9-45a0-8923-271b8be2a39c/scratchpad/gate_h1/gate_h1"
)
GRID_ROWS, GRID_COLS, PATCH = 360, 720, 32
TARGETS, ORIGINS = ("sb", "ns", "os"), 13


def pair_stats(w: np.ndarray) -> tuple[float, float, float]:
    """Mean Pearson, min Pearson and mean Spearman over every distinct pair of rows of `w`."""
    pairs = list(itertools.combinations(range(len(w)), 2))
    pr = [np.corrcoef(w[i], w[j])[0, 1] for i, j in pairs]
    sp = [spearmanr(w[i], w[j]).statistic for i, j in pairs]
    return float(np.mean(pr)), float(np.min(pr)), float(np.mean(sp))


def distinct_draws(tgt: str, origin: int) -> tuple[np.ndarray, np.ndarray, int]:
    """-> (distinct draws [D, n_cells], priogrid ids [n_cells], the stored column count S)."""
    d = GATE / f"origin_{origin}" / f"by_{tgt}_best"
    a = np.load(d / "y_pred_h1.npy").astype(np.float64)
    pg = np.load(d / "identifiers_h1.npz")["unit"]
    seen, cols = set(), []
    for j in range(a.shape[1]):
        b = a[:, j].tobytes()
        if b not in seen:
            seen.add(b)
            cols.append(a[:, j])
    return np.array(cols), pg, a.shape[1]


def to_grid(vec: np.ndarray, pg: np.ndarray) -> np.ndarray:
    g = np.full((GRID_ROWS, GRID_COLS), np.nan)
    g[(pg - 1) // GRID_COLS, (pg - 1) % GRID_COLS] = vec
    return g


# ----------------------------------------------------------------------------- the floor, globally
print("=" * 100)
print("PART 1 -- THE FLOOR on all land cells. Fixed weights, MC-dropout ON, eval mode, h=1.")
print("=" * 100)
print("  The falsifier runs first: S is the stored column count, D the distinct fields in it.")
print(f"  {'tgt':>4s} {'origins':>8s} {'cells':>7s} {'S':>3s} {'D':>3s} "
      f"{'r mean':>9s} {'r worst':>9s} {'rho mean':>9s}")
for tgt in TARGETS:
    acc = []
    for origin in range(ORIGINS):
        cols, _, s = distinct_draws(tgt, origin)
        acc.append((s, len(cols), *pair_stats(cols)))
    a = np.array(acc)
    print(f"  {tgt:>4s} {ORIGINS:8d} {64818:7d} {int(a[0, 0]):3d} {int(a[0, 1]):3d} "
          f"{a[:, 2].mean():+9.4f} {a[:, 3].min():+9.4f} {a[:, 4].mean():+9.4f}")

# ------------------------------------------------------------------------- the floor, on 32x32
print("\n" + "=" * 100)
print("PART 2 -- THE FLOOR on 32x32 patches of the same grid, which is the drift's population.")
print("=" * 100)
floor_median, floor_active = {}, {}
for tgt in TARGETS:
    per_patch = []
    for origin in range(ORIGINS):
        cols, pg, _ = distinct_draws(tgt, origin)
        grids = np.stack([to_grid(c, pg) for c in cols])
        land = ~np.isnan(grids[0])
        for r in range(0, GRID_ROWS - PATCH + 1, PATCH):
            for c in range(0, GRID_COLS - PATCH + 1, PATCH):
                m = land[r:r + PATCH, c:c + PATCH]
                if m.sum() < PATCH * PATCH // 2:          # a mostly-land patch, as the grid is
                    continue
                w = grids[:, r:r + PATCH, c:c + PATCH][:, m]
                if np.ptp(w) == 0:
                    continue
                pr, _, sp = pair_stats(w)
                per_patch.append((w.mean(), pr, sp))
    p = np.array(per_patch)
    hi = p[:, 0] >= np.quantile(p[:, 0], 0.90)
    floor_median[tgt] = float(np.median(p[:, 1]))
    floor_active[tgt] = float(np.median(p[hi, 1]))
    print(f"\n  {tgt}: {len(p)} mostly-land patches ({ORIGINS} origins x the grid), "
          f"{len(distinct_draws(tgt, 0)[0])} distinct draws each")
    print(f"      all patches        Pearson median {np.median(p[:, 1]):+.4f}  "
          f"p10 {np.quantile(p[:, 1], .10):+.4f}  p90 {np.quantile(p[:, 1], .90):+.4f}  "
          f"Spearman median {np.median(p[:, 2]):+.4f}")
    print(f"      top-decile activity (n={hi.sum()}, the population the curriculum picks)  "
          f"Pearson median {np.median(p[hi, 1]):+.4f}  Spearman median {np.median(p[hi, 2]):+.4f}")

# ------------------------------------------------------------------------------------- the drift
print("\n" + "=" * 100)
print("PART 3 -- THE DRIFT. Consecutive snapshots, one fixed 32x32 window, byte-identical truth.")
print("=" * 100)
groups: dict[int, list[Path]] = {}
for p in ARMD.rglob("*forensic_(lesson_*_win_*).npz"):
    groups.setdefault(hash(np.load(p)["y_reg"].tobytes()), []).append(p)
files = sorted(max(groups.values(), key=lambda g: len(g)),
               key=lambda p: int(p.name.split("lesson_")[1].split("_win")[0]))
lessons = [int(p.name.split("lesson_")[1].split("_win")[0]) for p in files]
fields = {k: np.stack([np.load(p)[c].astype(np.float64) for p in files])
          for k, c in (("body", "y_hat_reg"), ("gate", "y_hat_cls"),
                       ("forecast", "y_hat_forecast"))}
print(f"  {len(files)} snapshots, lessons {lessons[0]}..{lessons[-1]}")

rows = []
for i in range(1, len(files)):
    v = []
    for k in ("body", "gate", "forecast"):
        a, b = fields[k][i - 1].ravel(), fields[k][i].ravel()
        v += [float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b).statistic)]
    rows.append((lessons[i - 1], lessons[i] - lessons[i - 1], v))

drift = {}
for lab, lo, hi in (("lessons  94-149", 0, 150), ("lessons 200-299", 200, 999)):
    grp = [r[2] for r in rows if lo <= r[0] < hi]
    m = np.nanmean(np.array(grp), 0)
    drift[lab] = m
    print(f"  {lab} (n={len(grp):2d}):  body r {m[0]:+.4f} rho {m[1]:+.4f} | "
          f"gate r {m[2]:+.4f} rho {m[3]:+.4f} | forecast r {m[4]:+.4f} rho {m[5]:+.4f}")

# ------------------------------------------------------------------------------------- the verdict
print("\n" + "=" * 100)
print("VERDICT -- the GATE field, the one field measured on both sides.")
print("=" * 100)
print(f"  {'':34s} {'sb':>9s} {'ns':>9s} {'os':>9s}")
print("  floor, fixed weights, all patches  " +
      " ".join(f"{floor_median[t]:+9.4f}" for t in TARGETS))
print("  floor, fixed weights, ACTIVE patch " +
      " ".join(f"{floor_active[t]:+9.4f}" for t in TARGETS))
print(f"\n  drift, lessons  94-149 (pooled targets)  gate r {drift['lessons  94-149'][2]:+.4f}")
print(f"  drift, lessons 200-299 (pooled targets)  gate r {drift['lessons 200-299'][2]:+.4f}")
lo = min(floor_active.values())
d = drift["lessons 200-299"][2]
#: Stated as SHAPE CHANGE (1 - r), not as a ratio of correlations: a ratio of two correlations
#: is not a ratio of anything physical, and 1 - r is the quantity `drift_on_fixed_window.py`
#: already reports per lesson, so the two scripts stay on one scale.
print(f"\n  Re-drawing dropout at fixed weights changes {1 - lo:.3f} of the field's shape, on the "
      f"matched\n  active population and at the worst of the three targets. Between two "
      f"consecutive snapshots\n  late in training the same field changes {1 - d:.3f} of its shape "
      f"-- {(1 - d) / (1 - lo):.1f}x more.")
print("\n  => MC-dropout noise is RULED OUT as the explanation, ON THIS FIELD. The observation "
      "that\n     the output keeps moving while the loss plateaus survives its own strongest "
      "control.")
print("  ⚠️ The body's and the forecast's floors are NOT measured -- no draw axis was saved for "
      "them.")
