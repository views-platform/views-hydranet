"""E-1 baseline — the delivered emission's OWN CRPS at h1, per fold, split by truth.

Epic #559, story #564. Pre-registration: ``PREREG.md``, committed before any instrument existed
(``D-5``). Loader: ``e1_data.py`` (#563), whose four gates passed with ``G-0`` bit-exact.

⭐ **This is the comparator the whole experiment rests on.** The lookup in #565 is measured against
the number this file produces and against the ``B = 1`` floor, on the same cell-months.

⛔⛔ **IT IS NOT `M89`'s NUMBER.** `M89` reports `AR-h1` at 0.052109 plug-in / 0.050680 fair with a
floor of 0.060323, and those are **a different population** (all 13 months, bins borrowed from the
teacher-forced fit window) on **a different checkpoint** (the 2026-10-12 ``violet_visitor``
vehicle,
not ``ctlfullzero_fortytwo``). They are an order-of-magnitude sanity anchor and nothing else.
Quoting
them as this epic's comparator would be a `W-51` two-variable comparison.

**C-351 slots.** Population: 64,818 global ``land`` cells x 7 score months (463..469) = 453,726
cell-months, and each fold's floor is fitted on its own 6 months (388,908). Truth: ``lr_sb_best``,
raw counts, no transform. Estimator: **BOTH** — ``plug-in-crps`` ``e1 - A/(2m^2)`` and
``fair-crps``
``e1 - A/(2m(m-1))``, printed side by side with the estimator named on every line (``GLOSSARY.md``
§5e). Inference mode: ⭐ **DELIVERED AUTOREGRESSIVE, h = 1**. Arm: ``ctlfullzero_fortytwo``.
⛔ No h36 number is produced, in either direction.

⚠️ **WHAT TO EXPECT, pre-registered in ``PREREG.md`` §5b as P-30 rather than discovered here.** On
**99.4569%** of h1 cells the emission issues a literal point mass at zero, so its CRPS there is
exactly the truth value: *perfect* on a true zero and the full count on an event. 99.66% of cells
are
true zeros, so the emission wins the background outright and the whole comparison turns on events,
which carry ~96% of the pooled score. ⇒ The background and event halves are reported
**separately**,
with ``n`` beside every figure, because a pooled mean hides the only thing that moves.

⛔ THE GATES, declared before the run:

  G-2   ⭐ On every cell where all 16 draws are 0, ``plug-in == fair == |y|`` **exactly**, with no
        tolerance. The pairwise term ``A`` is 0 for a degenerate ensemble, so both estimators
        reduce
        to the same expression — and this is what confines the estimator asymmetry against #565's
        **exactly** scored lookup to the minority of cells that have spread. Both counts printed.

  G-P1  PROVENANCE, by AST. The ``crps_ensemble`` lifted into this file must be **structurally
        identical** to ``reports/2026-10-29_delivered_forecast/ar_score.py``'s. ⚠️ Compared as a
        parsed AST rather than as bytes, following ``l3c_three_target_scores.py``'s ``G-S1``,
        because a formatting difference is not a behavioural one and a byte test would fail on it.
        ⭐ An ADDITION beyond ``PREREG.md`` §7's gate list. It strengthens and relaxes nothing,
        so it is not a change to the verdict rule under §6 device 6.

  G-3   ⭐ ``PREREG.md`` §7's gate, in the form available here. At ``B = 1`` there are no interior
        edges, so every scored cell gets the SAME distribution and the floor **cannot depend on the
        conditioning index**. Four destructions of the index — shuffled, zeroed, inverted, and the
        fit index shuffled — must each leave it **bit-unchanged**. This is `M89`'s own ``G-4``
        reused, which measured spread ``0.0e+00`` across its eight variants, and it catches a
        ``B = 1`` path that accidentally bins.
        ⚠️ Scoped honestly: #565 runs the full assign x tie ladder; assign and tie belong to the
        lookup, not to the floor.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import e1_data  # noqa: E402

sys.path.insert(0, str(e1_data.REPO / "reports/2026-10-28_rp3_l3"))
from binned_cdf import binned_cdf_crps  # noqa: E402
from binned_cdf import self_test as binned_cdf_self_test  # noqa: E402

_AR_SCORE = e1_data.REPO / "reports/2026-10-29_delivered_forecast/ar_score.py"


# ── the lifted scorer. ⛔ LIFTED, NOT EDITED — see G-P1. ──────────────────────────────────────
def crps_ensemble(Y: np.ndarray, z: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Per-row CRPS of an m-member ensemble ``Y`` [N, m] against truth ``z`` [N].

    Returns (plug-in `1/m^2`, fair `1/(m(m-1))`). Same pairwise identity, row-wise.
    """
    Ys = np.sort(Y, axis=1)
    m = Ys.shape[1]
    e1 = np.abs(Ys - z[:, None]).mean(1)
    w = 2.0 * np.arange(m) - (m - 1)
    A = 2.0 * (Ys * w).sum(1)
    return e1 - A / (2.0 * m * m), e1 - A / (2.0 * m * (m - 1))


def _fn_ast(path: Path, name: str) -> str:
    """-> the dumped AST of one top-level function, with its docstring stripped."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise SystemExit(f"e1_baseline: no top-level function {name!r} in {path}")


def gate_p1_provenance() -> tuple[bool, str]:
    """-> (passed, a short note). The lifted scorer must match ar_score.py's structurally."""
    mine = _fn_ast(Path(__file__), "crps_ensemble")
    theirs = _fn_ast(_AR_SCORE, "crps_ensemble")
    return mine == theirs, f"{len(mine)} vs {len(theirs)} AST chars"


def gate_2_degenerate_estimators(
    draws: np.ndarray, truth: np.ndarray
) -> tuple[bool, int, int, float]:
    """-> (passed, n degenerate cells, n cells with spread, worst |deviation|).

    On an all-zero ensemble the pairwise term is 0, so both estimators must equal ``|y|`` exactly.
    """
    plug, fair = crps_ensemble(draws, truth)
    deg = draws.max(axis=1) == 0
    if not deg.any():
        return False, 0, int((~deg).sum()), float("nan")
    worst = max(
        float(np.abs(plug[deg] - np.abs(truth[deg])).max()),
        float(np.abs(fair[deg] - np.abs(truth[deg])).max()),
    )
    return worst == 0.0, int(deg.sum()), int((~deg).sum()), worst


def _split(vals: np.ndarray, truth: np.ndarray) -> dict[str, tuple[float, int]]:
    """-> {'all'|'background'|'event': (mean, n)}. ⛔ n travels with every mean (C-351)."""
    ev = truth > 0
    return {
        "all": (float(vals.mean()), int(vals.size)),
        "background": (float(vals[~ev].mean()), int((~ev).sum())),
        "event": (float(vals[ev].mean()), int(ev.sum())),
    }


def gate_3_floor_is_index_independent(gate: dict, grid: np.ndarray, m0: int) -> tuple[bool, float]:
    """-> (passed, worst |deviation|). ``PREREG.md`` §7's **G-3**, in the form available here.

    At ``B = 1`` there are no interior edges, so every scored cell receives the **same**
    distribution — the fit window's unconditional empirical CDF — and the floor therefore **cannot
    depend on the conditioning index**. Destroying the index must leave it bit-unchanged.

    ⭐ This is `M89`'s own ``G-4`` reused, which measured spread ``0.0e+00`` across all eight
    of its index / rule / tie variants. It catches the real failure: a ``B = 1`` path that
    accidentally bins, which would make the floor a weak conditional model instead of
    "no model at all".

    ⚠️ Scoped honestly: #565 runs the full assign x tie version across the whole ladder. Here only
    the index is varied, because assign and tie belong to the lookup and not to the floor.

    ⛔ An earlier draft of this gate permuted the SCORED TRUTH instead, which is **vacuous** — a
    mean
    over the same multiset against one shared distribution is invariant by construction. That would
    have been a second tautological gate immediately after the #563 review flagged the first.
    """
    fit_months, sco = e1_data.FOLDS[0]
    y_fit = np.concatenate([grid[m - m0] for m in fit_months])
    x_fit = np.concatenate([gate_of(gate, m) for m in fit_months])
    y_sco, x_sco = grid[sco - m0], gate_of(gate, sco)
    rng = np.random.default_rng(0)
    base = binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, 1)[1]
    variants = [
        binned_cdf_crps(x_fit, y_fit, rng.permutation(x_sco), y_sco, 1)[1],  # index shuffled
        binned_cdf_crps(x_fit, y_fit, np.zeros_like(x_sco), y_sco, 1)[1],  # index destroyed
        binned_cdf_crps(x_fit, y_fit, -x_sco, y_sco, 1)[1],  # index inverted
        binned_cdf_crps(rng.permutation(x_fit), y_fit, x_sco, y_sco, 1)[1],  # fit index shuffled
    ]
    worst = max(abs(base - v) for v in variants)
    return worst == 0.0, worst


def gate_of(gate: dict, month: int) -> np.ndarray:
    """-> the per-cell gate mean for one h1 month. Raises if that month was not delivered."""
    hit = [o for o in gate if gate[o][2] == month]
    if len(hit) != 1:
        raise SystemExit(
            f"e1_baseline: month {month} is carried by {len(hit)} origins, expected exactly 1."
        )
    return gate[hit[0]][0]


def run() -> None:
    t0 = time.time()
    print(__doc__, flush=True)
    print("=" * 100)
    print("E-1 BASELINE — the delivered emission's own CRPS at h1, per fold")
    print("=" * 100)

    ok_p1, note_p1 = gate_p1_provenance()
    print(
        f"  {'PASS' if ok_p1 else '⛔ FAIL'}  G-P1 PROVENANCE: lifted crps_ensemble matches "
        f"ar_score.py structurally  ({note_p1})"
    )
    if not ok_p1:
        raise SystemExit(
            "\nVOID: G-P1 failed. The lifted scorer is not the committed one, so every number "
            "below would be from a different implementation than the record's."
        )
    binned_cdf_self_test()
    print("  PASS  G-1a binned_cdf.self_test() — the frozen floor scorer's own identity check")

    loaded = e1_data.run_gates()
    grid, m0, gate, draws = (loaded["grid"], loaded["m0"], loaded["gate"], loaded["draws"])

    print()
    print("-" * 100)
    print(
        "1 · THE DELIVERED EMISSION, SCORED  ⭐ fair-crps is the headline; plug-in-crps beside it"
    )
    print("-" * 100)
    print(
        f"  {'fold':<6}{'month':<7}{'estimator':<12}{'all':>13}{'background':>15}{'event':>13}"
        f"{'n_event':>9}"
    )

    per_fold = {}
    for i, (_fit, sco) in enumerate(e1_data.FOLDS, start=1):
        o = [k for k in gate if gate[k][2] == sco][0]
        y = grid[sco - m0]
        plug, fair = crps_ensemble(draws[o][0], y)
        sp, sf = _split(plug, y), _split(fair, y)
        per_fold[sco] = {"plug": sp, "fair": sf, "plug_cell": plug, "fair_cell": fair, "truth": y}
        for label, s in (("plug-in-crps", sp), ("fair-crps", sf)):
            print(
                f"  {i:<6}{sco:<7}{label:<12}{s['all'][0]:>13.6f}{s['background'][0]:>15.6f}"
                f"{s['event'][0]:>13.4f}{s['event'][1]:>9,}"
            )

    print()
    print("  pooled over the 7 score months, 453,726 cell-months:")
    for label, key in (("plug-in-crps", "plug_cell"), ("fair-crps", "fair_cell")):
        allv = np.concatenate([per_fold[m][key] for m in per_fold])
        allt = np.concatenate([per_fold[m]["truth"] for m in per_fold])
        s = _split(allv, allt)
        print(
            f"    {label:<13} all {s['all'][0]:.6f} (n {s['all'][1]:,})   "
            f"background {s['background'][0]:.6f} (n {s['background'][1]:,})   "
            f"event {s['event'][0]:.4f} (n {s['event'][1]:,})"
        )

    print()
    print("-" * 100)
    print("2 · THE B=1 FLOOR — the unconditional empirical CDF of each fold's OWN fit window")
    print("-" * 100)
    print(
        f"  {'fold':<6}{'fit window':<18}{'score':<8}{'plug-in':>12}{'fair':>12}"
        f"{'fit members':>14}"
    )
    for i, (fit_months, sco) in enumerate(e1_data.FOLDS, start=1):
        y_fit = np.concatenate([grid[m - m0] for m in fit_months])
        x_fit = np.concatenate([gate_of(gate, m) for m in fit_months])
        p, f, members, empty = binned_cdf_crps(x_fit, y_fit, gate_of(gate, sco), grid[sco - m0], 1)
        print(
            f"  {i:<6}{f'{fit_months[0]}-{fit_months[-1]}':<18}{sco:<8}{p:>12.6f}{f:>12.6f}"
            f"{members:>14,.0f}"
        )
        assert empty == 0.0, f"fold {i}: {empty} of the score cells fell in an empty bin at B=1"

    print()
    print("-" * 100)
    print("3 · THE GATES")
    print("-" * 100)
    o0 = [k for k in gate if gate[k][2] == e1_data.FOLDS[0][1]][0]
    ok2, n_deg, n_spread, worst = gate_2_degenerate_estimators(
        draws[o0][0], grid[e1_data.FOLDS[0][1] - m0]
    )
    print(
        f"  {'PASS' if ok2 else '⛔ FAIL'}  G-2   on {n_deg:,} all-zero-draw cells "
        f"({n_deg / (n_deg + n_spread):.4%}) plug-in == fair == |y|, worst |dev| {worst:.1e}; "
        f"{n_spread:,} cells carry spread"
    )
    ok3, dev3 = gate_3_floor_is_index_independent(gate, grid, m0)
    print(
        f"  {'PASS' if ok3 else '⛔ FAIL'}  G-3   the B=1 floor is unchanged under 4 destructions "
        f"of the conditioning index, worst |dev| {dev3:.1e}"
    )

    if not (ok2 and ok3):
        raise SystemExit("\nVOID: a gate failed. See the block above.")

    print()
    print("-" * 100)
    print("4 · ⚠️ LIMITS")
    print("-" * 100)
    print("  · ⛔ NOT M89's number. M89's AR-h1 0.052109 / 0.050680 and floor 0.060323 are a")
    print("    DIFFERENT population (13 months, borrowed teacher-forced bins) on a DIFFERENT")
    print("    checkpoint (2026-10-12 violet_visitor). Sanity anchor only.")
    print("  · ⛔ h1 only. No h36 number, in either direction (PREREG §6 device 2).")
    print("  · ⚠️ n=1: one checkpoint, one seed, one arm, one target. No sigma-hat exists.")
    print("  · ⚠️ The emission is scored from 16 SAMPLED draws; #565's lookup is scored EXACTLY.")
    print("    G-2 confines that asymmetry to the cells with spread, it does not remove it.")
    print(f"\n  {time.time() - t0:.1f} s, CPU, zero GPU, nothing retrained.")


if __name__ == "__main__":
    run()
