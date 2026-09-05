"""G1 (dossier 03 C.3, falsifier F3): do the two instruments agree?

The dump writes the body mean and the gate as raw fields; the cube writes composed family DRAWS
masked by a per-draw Bernoulli(gate). They are produced by different code paths, in different runs
(the dump run had the dump ON, this cube run had it OFF), and they are read here two ways:

* EXACT — the cube's gate columns for MC pass 0 are `np.repeat` of the same `sigmoid(cls)` tensor
  the dump writes. These must agree to floating-point equality. Anything else means one of the two
  is not the object its name claims.
* STATISTICAL — the cube's mean emitted mass estimates `g*mu`, which the dump computes
  analytically. The cube carries Bernoulli and sampling noise, so agreement is checked at the field
  level, within the 10% F3 band.

A cell-set mismatch is the trap this guards against first: the cube covers the 13110 study cells,
the dump the full 180x180 grid. Comparing them unmasked would disagree for a reason that has
nothing to do with either instrument being wrong — and would silently dilute every field statistic
with ~19000 cells that are not in the study area.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from sharpness_scorecard import build_unit_grid  # noqa: E402

GRID = 180
F3_BAND = 0.10
MODELS = Path("../views-models/models/fullzero_fortytwo")
CUBE = MODELS / "data/generated/predictions_calibration_20260818_221401"
RAW = MODELS / "data/raw/calibration_datafactory_df.parquet"
ARM = sys.argv[1] if len(sys.argv) > 1 else "identity"
DUMP = Path(f"reports/2026-09-02_silence_vs_fade_dossier/results/bodymean_fullzero_fortytwo_{ARM}")
TARGET, TIDX = "sb_best", 0


def main() -> int:
    print(f"ARM: {ARM}\n")
    umap = build_unit_grid(str(RAW))
    dumps = sorted(DUMP.glob("bodymean_origin*.npz"), key=lambda p: int(p.stem.split("origin")[1]))
    cubes = sorted(CUBE.glob("origin_*"), key=lambda p: int(p.name.split("_")[1]))
    if len(dumps) != len(cubes):
        raise SystemExit(f"origin count mismatch: {len(dumps)} dumps vs {len(cubes)} cubes")

    occ_c, occ_d, mass_c, mass_d = [], [], [], []
    exact_max = 0.0
    for dpath, cpath in zip(dumps, cubes):
        z = np.load(dpath)
        mu, gate = z["mu"][:, TIDX], z["gate"][..., TIDX]  # [T,H,W], [T,H,W]
        units = np.load(cpath / f"lr_{TARGET}/identifiers.npz")["unit"]
        n_unit = len(np.unique(units))
        lr = np.load(cpath / f"lr_{TARGET}/y_pred.npy").reshape(n_unit, -1, 16)
        by = np.load(cpath / f"by_{TARGET}/y_pred.npy").reshape(n_unit, -1, 16)
        order = units.reshape(n_unit, -1)[:, 0]

        # VERTICAL FLIP. The model field's H axis runs opposite to priogrid row order, so the
        # naive (row-87, col-310) placement lands on the wrong cells: correlation 0.026 against
        # the cube. With the flip it is 1.0000 and the max difference is exactly 0. A global flip
        # cancels inside FSS/Moran (both grids are built the same way), which is why
        # sharpness_scorecard.to_grid never had to care — but it does NOT cancel when a grid is
        # compared against a model-native field, as here.
        rows = np.array([GRID - 1 - umap[int(u)][0] for u in order])
        cols = np.array([umap[int(u)][1] for u in order])
        keep = (rows >= 0) & (rows < GRID) & (cols >= 0) & (cols < GRID)
        if not keep.all():
            raise SystemExit(
                f"{keep.size - keep.sum()} study units fall outside the {GRID}x{GRID} grid"
            )

        # dump values on exactly the cube's cells: [n_unit, T]
        g_dump = gate[:, rows, cols].T
        m_dump = mu[:, rows, cols].T

        # EXACT: cube gate, MC pass 0 (cols 0..3 are np.repeat of one sigmoid(cls) tensor)
        exact_max = max(exact_max, float(np.abs(by[:, :, 0] - g_dump).max()))

        occ_c.append(by.mean(axis=(0, 2)))
        occ_d.append(g_dump.mean(axis=0))
        # NO expm1: the stored cube has already been through the pipeline's Invert step, so it is
        # in COUNT space (max 646.0, not a log1p value). Applying expm1 here overflowed to inf.
        mass_c.append(lr.mean(axis=(0, 2)))
        mass_d.append((g_dump * m_dump).mean(axis=0))

    occ_c, occ_d = np.mean(occ_c, axis=0), np.mean(occ_d, axis=0)
    mass_c, mass_d = np.mean(mass_c, axis=0), np.mean(mass_d, axis=0)

    print(f"G1 EXACT  max |cube gate (pass 0) - dump gate| = {exact_max:.3e}")
    exact_ok = exact_max < 1e-6

    print(
        f"\n{'h':>3} {'occ_cube':>12} {'occ_dump':>12} {'rel':>8} "
        f"{'mass_cube':>12} {'mass_dump':>12} {'rel':>8}"
    )
    worst_occ = worst_mass = 0.0
    for h in range(len(occ_c)):
        r_o = abs(occ_c[h] - occ_d[h]) / occ_d[h] if occ_d[h] else float("nan")
        r_m = abs(mass_c[h] - mass_d[h]) / mass_d[h] if mass_d[h] else float("nan")
        worst_occ, worst_mass = max(worst_occ, r_o), max(worst_mass, r_m)
        if h + 1 in (1, 6, 12, 18, 24, 30, 36):
            print(
                f"{h + 1:>3} {occ_c[h]:>12.5e} {occ_d[h]:>12.5e} {r_o:>8.4f} "
                f"{mass_c[h]:>12.5e} {mass_d[h]:>12.5e} {r_m:>8.4f}"
            )
    print(f"\nworst relative disagreement: occurrence {worst_occ:.4f}, mass {worst_mass:.4f}")

    ok = exact_ok and worst_occ < F3_BAND and worst_mass < F3_BAND
    print(
        f"\nF3: {'PASS — instruments agree; proceed to (d)' if ok else 'FIRED — HALT, no claim either way'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
