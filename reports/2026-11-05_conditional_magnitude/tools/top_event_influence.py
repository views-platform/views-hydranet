"""How much of the conditional-magnitude profile is driven by the largest few observations?

Committed BEFORE it runs (``D-5``). Answers a direct question from Simon about the Tigray
aggregate, and then the general version of it.

⛔⛔ THIS SCRIPT IS DELIBERATELY OUTCOME-CONDITIONED, AND THAT IS THE POINT

It removes cells selected by the SIZE OF THEIR OBSERVED OUTCOME. That is exactly what
``conditional_magnitude.py``'s docstring forbids -- for a **calibration** statistic. Here it is
an **influence** measure: the question *"how much of this number comes from the largest k
observations?"* cannot be asked without conditioning on the outcome, and the answer is a
statement about the estimator's robustness, not about the forecast's calibration.

⇒ ⛔ No row produced here may be quoted as a calibration or bias figure. Every number is an
influence figure. ⚠️ And removal here is a SENSITIVITY probe, never a remedy: the `models` session's
reading of Vesco et al. and Weidmann 2015 is that under-reporting is non-linear in event size while
counts are more accurate than locations, so thresholding on magnitude discards the reliable half.
⛔ Nothing in this file licenses winsorising or capping the data.

PART 1 · THE SPECIFIC CELL (verified before this script existed, re-asserted here as a gate)

The Tigray aggregate is priogrid **148759, month 504**, 113,395 fatalities -- UCDP's total
for a year of war across three Ethiopian regions landing in one cell-month. The h1 window is
months **457..469**. ⇒ Month 504 is reached only at h36. ``G-T1`` asserts that cell contributes
exactly **0** inside the window, so its influence on every number in
``results/conditional_magnitude.out`` is **zero by construction, not by luck**.

PART 2 · THE GENERAL VERSION

The window's own largest observations could carry the profile even without a known aggregate. So:
report where the top-K observations landed in the prediction bins, and recompute the whole profile
with the top-k removed for k in {0, 1, 5, 12, 25}.

⭐ The in-window top 12 do NOT carry the summary-event signature. The known artefact is priogrid
150919 -- 48,183 fatalities in **6 nonzero months of 336**, ringed by zeros. Every one of the
top 12
sits in a cell with **49 to 137 nonzero months of 384**, and ranks 1, 2, 5 and 7 are the SAME cell
(178273) in four consecutive months -- a sustained episode, not an aggregate. ``G-T2`` recomputes
that nonzero-month floor rather than quoting it.

SLOTS (``C-351``): population, transform and estimator are ``conditional_magnitude.py``'s,
unchanged. The only new free parameter is k, and all five values are reported (``D-4``).

⛔ THE GATES

    G-0   ⭐⭐ COPY, exact: ap_ctl = 0.5667056384496834. VOID -- nothing reported on failure
    G-5   per-origin alignment
    G-8   structural: 842,634 cell-months / 2,905 events / max 2,482
    G-T1  ⭐ priogrid 148759 contributes exactly 0.0 inside months 457..469, and 113,395 at 504
    G-T2  ⭐ every top-12 cell has >= 20 nonzero months of 384, so none carries the 6-month
          summary-event signature. ⛔ Refuses if any does -- in which case the window is NOT clean
          and the profile must be re-read
    G-T3  non-vacuity: each k must remove exactly k observations and leave >= 100 events, or that
          row is refused rather than reported thin

PRE-REGISTERED PREDICTIONS -- next free is P-35

    P-35  ⭐⭐ The SHAPE survives. With the top 25 observations removed, the ratio in the lowest
          prediction bin still exceeds the ratio in the highest by more than 2.0x, at B=8 and
          B=16. ~75%
    P-36  The lowest bin's ratio moves by less than 25% relative when the top 25 are removed,
          because the largest observations cannot be in the lowest prediction bin: bin 0 at B=8
          holds 530 cells with mean_obs 2.56, total ~1,357, which is less than the single largest
          observation of 2,482. ⭐ Arithmetic, not a guess. ~80%
    P-37  The TOP bin's ratio falls when the top observations are removed, because that is where
          they land. ~70%

FALSIFIERS

    F5  the lowest-vs-highest ratio gap drops below 2.0x at any k ⇒ ⛔ the profile IS carried by a
        handful of observations and the finding is withdrawn to "not demonstrated"
    F6  a gate fails ⇒ no number reported
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conditional_magnitude as cm  # noqa: E402
import e1_data  # noqa: E402

TIGRAY_CELL, TIGRAY_MONTH, TIGRAY_VALUE = 148_759, 504, 113_395.0
K_SWEEP = (0, 1, 5, 12, 25)
TOP_N_INSPECT = 12
G_T2_MIN_NONZERO_MONTHS = 20
G_T3_MIN_EVENTS = 100
P35_MIN_GAP = 2.0


def gate_t1_tigray_is_outside(
    grid: np.ndarray, cells: np.ndarray, m0: int
) -> tuple[bool, float, float]:
    """-> (passed, in-window total for that cell, its month-504 value). Recomputed, not quoted."""
    ci = int(np.searchsorted(cells, TIGRAY_CELL))
    if cells[ci] != TIGRAY_CELL:
        raise SystemExit(f"top_event_influence: priogrid {TIGRAY_CELL} is not in the pull at all.")
    inside = float(grid[np.array(e1_data.H1_MONTHS) - m0, ci].sum())
    at504 = float(grid[TIGRAY_MONTH - m0, ci])
    return (inside == 0.0 and at504 == TIGRAY_VALUE), inside, at504


def gate_t2_no_summary_signature(
    grid: np.ndarray, top: list[tuple[float, int, int]]
) -> tuple[bool, int]:
    """-> (passed, the smallest nonzero-month count among the top cells).

    ⛔ The known artefact (priogrid 150919) has SIX nonzero months of 336 and is ringed by
    zeros. A top observation in a cell nonzero only a handful of times is an aggregate, not a
    battle.
    """
    counts = [int((grid[:, ci] > 0).sum()) for _v, ci, _mi in top]
    return (min(counts) >= G_T2_MIN_NONZERO_MONTHS), min(counts)


def main() -> None:
    print("=" * 110)
    print("TOP-EVENT INFLUENCE ON THE CONDITIONAL-MAGNITUDE PROFILE")
    print("=" * 110)
    print("  ⛔ EVERY NUMBER HERE IS OUTCOME-CONDITIONED BY DESIGN and is an INFLUENCE figure.")
    print("     None may be quoted as a calibration or bias figure. Removal is a sensitivity")
    print("     probe, NEVER a remedy -- see the docstring on Vesco et al. and Weidmann 2015.")

    grid, cells, m0 = e1_data.truth_grid()
    gate = e1_data.load_gate_h1()
    draws = e1_data.load_draws_h1()

    ok0, ap, d0 = e1_data.gate_0_ap_copy(gate, grid, cells, m0)
    print(f"\n  {'PASS' if ok0 else '⛔ FAIL'}  G-0  ap {ap:.16f}  |delta| {d0:.3e}")
    if not ok0:
        raise SystemExit("VOID: G-0 did not reproduce the anchor. Nothing is reported.")
    ok5, bad5 = e1_data.gate_5_alignment(gate, draws, cells)
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok5 else '⛔ FAIL'}  G-5  misaligned {bad5}")
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8  {n8:,} cell-months  {ev8:,} events  max {mx8}")

    okt1, inside, at504 = gate_t1_tigray_is_outside(grid, cells, m0)
    print(
        f"  {'PASS' if okt1 else '⛔ FAIL'}  G-T1  priogrid {TIGRAY_CELL}: in-window total "
        f"{inside:,.0f}, month {TIGRAY_MONTH} value {at504:,.0f}"
    )
    print(
        f"        ⇒ its influence on every conditional-magnitude number is EXACTLY {inside:,.0f}"
    )

    W = np.array(e1_data.H1_MONTHS)
    rows = grid[W - m0, :]
    flat = rows.ravel()
    order = np.argsort(flat)[::-1][: max(K_SWEEP + (TOP_N_INSPECT,))]
    top = [(float(flat[k]), int(k) % rows.shape[1], int(k) // rows.shape[1]) for k in order]

    okt2, min_nz = gate_t2_no_summary_signature(grid, top[:TOP_N_INSPECT])
    print(
        f"  {'PASS' if okt2 else '⛔ FAIL'}  G-T2  smallest nonzero-month count among the top "
        f"{TOP_N_INSPECT}: {min_nz} of 384 (floor {G_T2_MIN_NONZERO_MONTHS}; the known artefact "
        f"priogrid 150919 has 6)"
    )
    if not (ok5 and ok8 and okt1 and okt2):
        raise SystemExit("⛔ F6: a gate failed. No number is reported.")

    origins = sorted(gate)
    pred = np.concatenate([cm.predicted(draws[o][0]) for o in origins])
    obs = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    open_m = np.concatenate([(draws[o][0] > 0).any(axis=1) for o in origins])
    p_all, o_all = pred[open_m], obs[open_m]

    print(f"\n  OPEN set {p_all.size:,} cells, {(o_all > 0).sum():,} events")
    print(f"\n  WHERE THE TOP {TOP_N_INSPECT} IN-WINDOW OBSERVATIONS LANDED (B=8 prediction bins)")
    edges8 = cm.equal_count_edges(p_all, 8)
    print(
        f"    {'rank':>4} {'value':>8} {'priogrid':>9} {'month':>6} {'open?':>6} "
        f"{'predicted':>10} {'B=8 bin':>8}"
    )
    for i, (v, ci, mi) in enumerate(top[:TOP_N_INSPECT], 1):
        oi = [o for o in origins if gate[o][2] == int(W[mi])][0]
        flatpos = origins.index(oi) * e1_data.N_CELLS + int(np.searchsorted(cells, cells[ci]))
        is_open = bool(open_m[flatpos])
        pv = float(pred[flatpos])
        b = int(np.clip(np.searchsorted(edges8[1:-1], pv, side="right"), 0, 7)) if is_open else -1
        print(
            f"    {i:>4} {v:>8,.0f} {cells[ci]:>9} {W[mi]:>6} {str(is_open):>6} "
            f"{pv:>10.4f} {(b if is_open else 'closed'):>8}"
        )

    print("\n  THE PROFILE WITH THE TOP-k OBSERVATIONS REMOVED (open set, B=8 and B=16)")
    for b in (8, 16):
        print(f"\n   B={b}")
        print(
            f"    {'k':>4} {'n':>7} {'events':>7} {'ratio bin0':>11} {'ratio top':>10} "
            f"{'gap':>7} {'pooled':>8}"
        )
        for k in K_SWEEP:
            keep = np.ones(o_all.size, bool)
            if k:
                thr_idx = np.argsort(o_all)[::-1][:k]
                keep[thr_idx] = False
            removed = int((~keep).sum())
            ev = int((o_all[keep] > 0).sum())
            if removed != k or ev < G_T3_MIN_EVENTS:
                print(
                    f"    {k:>4}  ⛔ G-T3 refused: removed {removed} (wanted {k}), "
                    f"{ev} events left (floor {G_T3_MIN_EVENTS})"
                )
                continue
            t = cm.bin_table(p_all[keep], o_all[keep], b)
            okm2, _nb, nbp = cm.gate_m2_non_vacuous(t)
            okm3, _ = cm.gate_m3_bins_are_ordered(t)
            if not (okm2 and okm3):
                print(f"    {k:>4}  ⛔ G-M2/G-M3 refused ({nbp} bins with an event)")
                continue
            r0, rt = t[0]["ratio"], t[-1]["ratio"]
            pooled = float(o_all[keep].mean() / p_all[keep].mean())
            print(
                f"    {k:>4} {p_all[keep].size:>7,} {ev:>7,} {r0:>11.3f} {rt:>10.3f} "
                f"{r0 / rt:>7.2f} {pooled:>8.4f}"
            )

    print(f"\n  ⛔ F5 fires if the gap falls below {P35_MIN_GAP} at any k. P-35..P-37 are scored")
    print("     in the dossier against the thresholds in this docstring.")


if __name__ == "__main__":
    main()
