"""E-1 loader — the delivered gate at h1, the truth, and the emission's own draws, ALIGNED.

Epic #559, story #563. Pre-registration: ``PREREG.md``, committed 2026-11-04 before this file
existed (``D-5``).

⭐⭐ WHY THIS FILE IS THE LOAD-BEARING ONE, AND IT IS NOT THE SCORER. Every scorer this epic
needs already exists, frozen and self-tested (``03_harness_and_invariants.md`` §B). The
marginal risk is a **one-step misalignment** between the gate, the truth and the draws, which
would produce a plausible WRONG number that survives every statistical check downstream. The
gates below exist for that and nothing else.

**C-351 slots, every one.**
  population   64,818 global ``land`` cells x 13 delivered origins; at h1 the origins are
               months 457..469, one month per origin, so a fold's score window is exactly
               64,818 cells.
  truth        ``lr_sb_best`` from the arms' shared global pull. ⚠️ **RAW COUNTS, no
               transform**, despite the ``lr_`` prefix — verified by reading the column
               maximum (113,395) and the single cell behind it (``priogrid 148759, month
               504``).
  index        the per-cell **mean of the gate's 16 MC-dropout draws**. ⭐ The construction
               the published ``exp03_pairs`` readout used, and the reason is at
               ``hydranet_inference.py:929-931``: the scorer ranks on the gate averaged over
               all posterior columns, and a single pass reproduces its AP only to ~11%.
  emission     the delivered composed draws, ``[2333448, 16]`` per origin = 36 horizons x
               64,818 cells x 16 draws, **unit-major with stride 36**, so h1 is ``[0::36]``.
               ⛔ Already composed (post-``threshold_gate``, tau=0.20) and in **count space**
               — h1 max 728.0001 against a truth max of 552 at month 457, so there is no
               ``expm1`` anywhere.
  inference    ⭐ **DELIVERED AUTOREGRESSIVE, h = 1.** The slot whose absence is ``C-399``.
  arm          ``ctlfullzero_fortytwo`` (control, seed 42). ⛔ One arm; a test of a method.

⛔ THE GATES, declared before the run:

  G-0  ⭐⭐ COPY GATE, exact, and it is load-bearing. The per-cell gate mean, pooled over all
       13 origins (842,634 cell-months), through the FROZEN ``average_precision``
       (``lodestar_score.py:36``), must reproduce ``ap_ctl = 0.5667056384496834`` from
       ``reports/2026-10-30_upsample_arm_dossier/results/exp03_pairs.json``, record
       ``{"seed": "fortytwo", "h": 1}``. Tolerance 1e-12 — the construction is reproduced step
       for step from ``exp03_pairs.py:58-84``, so near-exactness is the right demand
       (``W-96``).
       ⛔ On failure this module raises ``SystemExit`` and NOTHING else is reported. It is
       ``PREREG.md`` §6 device 4's single void condition.
       ⭐ Why this is a legitimate copy gate under ``W-98`` (*"a copy gate's anchor is the
       committed implementation, never a ledger row"*): the anchor is a committed artifact
       produced by a committed script through the frozen ruler, and the figure is this arm's
       OWN absolute AP on exactly the population loaded here. Wrong arm, wrong origin set,
       wrong target or a different draw average and the number moves.

  G-4  ⭐ LEAKAGE, by COUNTING and not by argument. For each of the 7 folds the intersection
       of the fit months with the score month must be empty, asserted on the month arrays
       themselves. Reports the intersection size per fold; a non-zero anywhere is fatal.

  G-5  ALIGNMENT. Per origin, ``np.array_equal`` on three things: the gate's ``unit`` array,
       the draws' h1 ``unit`` array, and the truth grid's cell order. ⚠️ This is the gate that
       catches the failure every other gate would survive.

  G-8  STRUCTURAL, recomputed and not quoted. Over h1 months 457..469 the truth must give
       842,634 cell-months, **2,905** events (0.3448%) and a maximum of **2,482**.

⚠️ C-347. ``python <path>`` puts this file's own directory on ``sys.path[0]``, not the repo root,
so ``views_hydranet`` would resolve to the installed wheel rather than the working tree. Nothing
here imports ``views_hydranet``, but the repo root is inserted explicitly anyway so that a later
import cannot acquire the hazard silently.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

#: ⭐ Reuse the committed resolver rather than spelling delivery paths. It derives each arm's
#: directory from the two manifests and refuses an ambiguous one, which matters because
#: views-models RENAMED all twelve arms (#702/#704) and no directory matches the old spellings.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "2026-10-30_upsample_arm_dossier/tools")
)
from exp03_arms import DELIVERY, REPO, arm_dir, arm_directories  # noqa: E402

sys.path.insert(0, str(REPO))  # C-347, see the module docstring
sys.path.insert(0, str(REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
from lodestar_score import average_precision  # noqa: E402

ARM = "ctlfullzero_fortytwo"
TARGET = "sb"
N_ORIGINS = 13
N_CELLS = 64_818
N_DRAWS = 16
N_HORIZONS = 36

#: The pre-registered rolling design: fixed six-month fit, one score month, 7 folds.
#: ⭐ FIXED length and not expanding, so fit-window length never confounds which month is scored.
FIT_LEN = 6
H1_MONTHS = tuple(range(457, 470))  # origin_0..origin_12 at h1
FOLDS = tuple(
    (tuple(range(m - FIT_LEN, m)), m) for m in range(H1_MONTHS[0] + FIT_LEN, H1_MONTHS[-1] + 1)
)

#: G-0's anchor, read from the committed artifact at import time rather than typed in.
_ANCHOR_JSON = REPO / "reports/2026-10-30_upsample_arm_dossier/results/exp03_pairs.json"
G0_TOL = 1e-12

#: G-8's structural expectations over h1 months 457..469.
G8_CELL_MONTHS = 842_634
G8_EVENTS = 2_905
G8_MAX = 2482.0


def anchor_ap() -> float:
    """-> the committed ``ap_ctl`` for this arm at h1. Raises if the record is absent."""
    rows = json.loads(_ANCHOR_JSON.read_text())
    hit = [r for r in rows if r.get("seed") == "fortytwo" and r.get("h") == 1]
    if len(hit) != 1:
        raise SystemExit(
            f"e1_data: expected exactly one {{seed: fortytwo, h: 1}} record in {_ANCHOR_JSON}, "
            f"found {len(hit)}. G-0 has no anchor and nothing may be reported."
        )
    row = hit[0]
    #: ⛔ The anchor's own population must match ours, or the gate compares two different things.
    if int(row["n"]) != G8_CELL_MONTHS or int(row["n_org"]) != N_ORIGINS:
        raise SystemExit(
            f"e1_data: the anchor was computed on n={row['n']} over {row['n_org']} origins, but "
            f"this loader reads n={G8_CELL_MONTHS} over {N_ORIGINS}. Not the same population; G-0 "
            f"would be a consistency check and no copy gate may be claimed (W-120)."
        )
    return float(row["ap_ctl"])


def truth_grid() -> tuple[np.ndarray, np.ndarray, int]:
    """-> ``(GRID[month, cell], CELLS, M0)`` in RAW COUNTS.

    Built exactly as ``exp03_pairs.py:47-52`` builds it, from the arms' shared global pull, so that
    G-0 compares like with like.
    """
    tb = pq.read_table(
        arm_dir(ARM, arms=arm_directories()) / "data/raw/calibration_datafactory_df.parquet",
        columns=["month_id", "priogrid_id", "lr_sb_best"],
    )
    mo, un = tb["month_id"].to_numpy(), tb["priogrid_id"].to_numpy()
    cells, m0 = np.unique(un), int(mo.min())
    grid = np.zeros((int(mo.max()) - m0 + 1, cells.size), np.float64)
    grid[mo - m0, np.searchsorted(cells, un)] = tb["lr_sb_best"].to_numpy()
    return grid, cells, m0


def load_gate_h1() -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    """-> ``{origin: (gate_mean[N_CELLS], units, month)}`` at h1.

    Extraction is scoped to ``*/by_sb_best/*`` so the h36 arrays and the other two targets
    are never
    written to disk.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    src = arm_dir(ARM, arms=arm_directories()) / DELIVERY / "gate_endpoints.tar.zst"
    with tempfile.TemporaryDirectory(prefix="e1_gate_") as td:
        subprocess.run(
            ["tar", "-I", "zstd", "-xf", str(src), "-C", td, "--wildcards", "*/by_sb_best/*"],
            check=True,
        )
        root = Path(td) / "gate_endpoints"
        for p in sorted(root.glob("origin_*"), key=lambda q: int(q.name.split("_")[1])):
            o = int(p.name.split("_")[1])
            g = np.load(p / "by_sb_best" / "y_pred_h1.npy").astype(np.float64)
            z = np.load(p / "by_sb_best" / "identifiers_h1.npz")
            tt, uu = z["time"].astype(np.int64), z["unit"].astype(np.int64)
            if not (g.shape == (N_CELLS, N_DRAWS) and tt.size == uu.size == N_CELLS):
                raise SystemExit(
                    f"e1_data: origin {o} gate has shape {g.shape} with {tt.size} identifiers; "
                    f"expected {(N_CELLS, N_DRAWS)} and {N_CELLS}."
                )
            if np.unique(tt).size != 1:
                raise SystemExit(
                    f"e1_data: origin {o}'s h1 slice spans {np.unique(tt).size} months, not 1. "
                    f"A multi-month slice would pool two populations under one label."
                )
            out[o] = (g.mean(1), uu, int(tt[0]))
    if len(out) != N_ORIGINS:
        raise SystemExit(
            f"e1_data: found {len(out)} origins in the gate archive, expected {N_ORIGINS}. "
            f"A partially-rsynced delivery looks exactly like a complete one to a name lookup."
        )
    return out


def load_draws_h1() -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    """-> ``{origin: (draws[N_CELLS, N_DRAWS], units, month)}`` for the DELIVERED emission at h1.

    ⛔ Streams ONE origin at a time and deletes each extracted array before the next, because every
    ``y_pred.npy`` is 149 MB uncompressed and the disk sits at 96% (38 GB free, #448).
    """
    out: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    delivery = arm_dir(ARM, arms=arm_directories()) / DELIVERY
    archives = sorted((delivery / "draws").glob("*_lr.tar.zst"))
    if len(archives) != 1:
        raise SystemExit(
            f"e1_data: expected exactly one *_lr.tar.zst under {delivery / 'draws'}, "
            f"found {len(archives)}: {[a.name for a in archives]}."
        )
    src = archives[0]
    for o in range(N_ORIGINS):
        with tempfile.TemporaryDirectory(prefix=f"e1_draws_o{o}_") as td:
            subprocess.run(
                [
                    "tar",
                    "-I",
                    "zstd",
                    "-xf",
                    str(src),
                    "-C",
                    td,
                    "--wildcards",
                    f"*/origin_{o}/lr_{TARGET}_best/*",
                ],
                check=True,
            )
            found = sorted(Path(td).glob(f"*/origin_{o}/lr_{TARGET}_best/y_pred.npy"))
            if len(found) != 1:
                raise SystemExit(
                    f"e1_data: origin {o}: expected one y_pred.npy, found {len(found)}."
                )
            ids = np.load(found[0].parent / "identifiers.npz")
            tt, uu = ids["time"].astype(np.int64), ids["unit"].astype(np.int64)
            month = int(tt.min())
            sel = tt == month
            if int(sel.sum()) != N_CELLS:
                raise SystemExit(
                    f"e1_data: origin {o}: {int(sel.sum())} rows at h1, expected {N_CELLS}."
                )
            #: ⛔ Select by MONTH and then assert the stride, rather than trusting [0::36] alone.
            stride_rows = np.arange(0, N_CELLS * N_HORIZONS, N_HORIZONS)
            if not np.array_equal(np.flatnonzero(sel), stride_rows):
                raise SystemExit(
                    f"e1_data: origin {o}: the h1 rows are not the unit-major stride-{N_HORIZONS} "
                    f"rows. The archive layout has changed and positional reads are unsafe."
                )
            y = np.load(found[0], mmap_mode="r")
            if y.shape != (N_CELLS * N_HORIZONS, N_DRAWS):
                raise SystemExit(
                    f"e1_data: origin {o}: draws have shape {y.shape}, expected "
                    f"{(N_CELLS * N_HORIZONS, N_DRAWS)}."
                )
            out[o] = (np.asarray(y[sel], dtype=np.float64), uu[sel], month)
            del y
    return out


# ── the gates ────────────────────────────────────────────────────────────────────────────────


def gate_0_ap_copy(
    gate: dict[int, tuple[np.ndarray, np.ndarray, int]],
    grid: np.ndarray,
    cells: np.ndarray,
    m0: int,
) -> tuple[bool, float, float]:
    """-> (passed, computed AP, |delta|). Reproduces ``exp03_pairs.ap_of`` step for step."""
    origins = sorted(gate)
    score = np.concatenate([gate[o][0] for o in origins])
    y_bin = np.concatenate(
        [
            (grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] > 0).astype(np.float64)
            for o in origins
        ]
    )
    ap = float(average_precision(y_bin, score))
    delta = abs(ap - anchor_ap())
    return delta <= G0_TOL, ap, delta


def gate_4_leakage() -> tuple[bool, list[int]]:
    """-> (passed, per-fold intersection sizes). Counted, never argued."""
    sizes = [len(set(fit) & {sco}) for fit, sco in FOLDS]
    return all(s == 0 for s in sizes), sizes


def gate_5_alignment(
    gate: dict[int, tuple[np.ndarray, np.ndarray, int]],
    draws: dict[int, tuple[np.ndarray, np.ndarray, int]],
    cells: np.ndarray,
) -> tuple[bool, list[int]]:
    """-> (passed, the origins that failed). Three-way ``np.array_equal`` per origin."""
    bad = [
        o
        for o in sorted(gate)
        if not (
            np.array_equal(gate[o][1], draws[o][1])
            and np.array_equal(gate[o][1], cells)
            and gate[o][2] == draws[o][2]
        )
    ]
    return not bad, bad


def gate_8_structural(grid: np.ndarray, m0: int) -> tuple[bool, int, int, float]:
    """-> (passed, cell-months, events, max). Recomputed from the pull, not quoted."""
    rows = grid[np.array(H1_MONTHS) - m0, :]
    n, ev, mx = int(rows.size), int((rows > 0).sum()), float(rows.max())
    return (n == G8_CELL_MONTHS and ev == G8_EVENTS and mx == G8_MAX), n, ev, mx


def run_gates() -> dict[str, object]:
    """Load everything, run every gate, print the block, and refuse on G-0."""
    t0 = time.time()
    grid, cells, m0 = truth_grid()
    gate = load_gate_h1()
    draws = load_draws_h1()

    ok0, ap, d0 = gate_0_ap_copy(gate, grid, cells, m0)
    ok4, sizes = gate_4_leakage()
    ok5, bad5 = gate_5_alignment(gate, draws, cells)
    ok8, n8, ev8, mx8 = gate_8_structural(grid, m0)

    print(
        f"  {'PASS' if ok0 else '⛔ FAIL'}  G-0  COPY: pooled h1 gate AP {ap:.16f} "
        f"vs committed {anchor_ap():.16f}  |Δ| {d0:.3e}  (tol {G0_TOL:.0e})"
    )
    print(
        f"  {'PASS' if ok4 else '⛔ FAIL'}  G-4  LEAKAGE: per-fold fit∩score sizes {sizes} "
        f"over {len(FOLDS)} folds"
    )
    print(
        f"  {'PASS' if ok5 else '⛔ FAIL'}  G-5  ALIGNMENT: gate units == draws units == truth "
        f"cell order on {len(gate) - len(bad5)}/{len(gate)} origins"
        + (f"; failed {bad5}" if bad5 else "")
    )
    print(
        f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8  STRUCTURAL: {n8:,} cell-months, {ev8:,} events "
        f"({ev8 / n8:.4%}), max {mx8:.0f}  (expected {G8_CELL_MONTHS:,} / {G8_EVENTS:,} / "
        f"{G8_MAX:.0f})"
    )

    if not ok0:
        raise SystemExit(
            f"\nVOID: G-0 FAILED. Computed {ap!r} against the committed "
            f"{anchor_ap()!r}. The loader "
            f"is not reading the population the published numbers came from, and PREREG.md §6 "
            f"device 4 makes this the single void condition. Nothing else is reported."
        )
    if not (ok4 and ok5 and ok8):
        raise SystemExit("\nVOID: a structural gate failed. See the block above.")

    print(
        f"\n  loaded {len(gate)} origins, {len(FOLDS)} folds, "
        f"{len(FOLDS) * N_CELLS:,} scored cell-months, in {time.time() - t0:.1f} s"
    )
    return {"grid": grid, "cells": cells, "m0": m0, "gate": gate, "draws": draws}


if __name__ == "__main__":
    print(__doc__, flush=True)
    print("=" * 100)
    print("E-1 LOADER — the delivered gate at h1, the truth, and the emission's draws")
    print("=" * 100)
    run_gates()
