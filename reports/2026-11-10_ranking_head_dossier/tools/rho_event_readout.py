"""rho_event_readout.py — the ρ_event instrument for the ranking body, with its control pinned.

⭐⭐ **Built to the one recipe in this repo that produced an instrument whose headline did NOT have
to be withdrawn.** `reports/2026-09-30_evaluation_learning_report.md` §9c records that
`ordering_by_index.py` was the fifth instrument of its dossier and the first to survive, and names
the three things that were different. All three are implemented here:

1. ⭐⭐ **The estimator is IMPORTED, never copied.** `spearman` and `midrank` come from
   `../../2026-11-10_rank_coupling_verdict/tools/rank_coupling_verdict.py` — the module that
   published `+0.3127`. `tests/test_rho_event_readout.py` asserts the **same function objects**,
   so a near-copy cannot drift in. Every earlier comparison in that dossier was against a
   near-copy, and both of its HOLDs were about a baseline that moved.
2. ⭐⭐ **`G-1` pins the control by arithmetic.** This tool must reproduce the delivered body's
   `ρ_event = +0.3127` AND the gate's `+0.4418` from the committed artifacts, to `_TOL`, or it
   ⛔ **refuses to print any new number.** The comparison is legitimate because a gate says so, not
   because the author believed it was.
3. ⭐ **The swept axis is the thing in question.** The question is *which index orders event cells
   best*, so the index is what varies: one row per score field, no single number to move.

And one gate from a different lesson — §0 item 2, which is the most expensive thing in that report:
a delivery served **0.0 for all 2,330,712 cells** on global land, in all three targets, and **every
structural check passed** (valid manifest, right coverage, file size 98.1% of a real run's). ⛔ So
`G-2` counts **distinct values**, not files. The comparator is printed beside it: the gate takes
**64,732 distinct values over 64,818 cells**, which is §9c `N9`'s explanation for why it wins.

**Population, per `C-351`:** 7 folds (fixed six-month fit, one score month, months 463–469), `sb`,
h1, truth `lr_sb_best` in **raw counts, no transform**. `ρ_event` is Spearman on **midranks**
between the index and the truth over the **event cells** of the firing set —
`open_m & (truth > 0)` at `τ` read from the arm config — measured on the **global** 360×720 grid,
never within a patch. Reported as mean ± SD over the 7 folds at **`ddof=1`, declared**.

⛔ **What this tool does NOT do.** It does not score a verdict. `00_README` §6's kill rule and
`04_roadmap` phase 3's pre-commitment both apply to every number it prints: a `ρ_event` from an
unconverged arm is an **instrument reading**, not evidence for or against the ranking body.

Usage::

    python rho_event_readout.py                      # the control table alone; G-1 and G-2 only
    python rho_event_readout.py --score FIELD.npz     # add a row for a score field on disk

A `--score` file is an `.npz` with `score` of shape `[n_months, 64818]` and `months` of shape
`[n_months]`, the month indices matching this design's score months. Exit 0 = every gate passed;
exit 1 = a gate failed and nothing was reported.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE.parents[1]
# ⛔ Relative to this file, never an absolute /home path: §0 of the evaluation report counts 21
# tracked tool files carrying hardcoded user paths, which is why none of them runs anywhere else.
for _p in (
    _REPORTS / "2026-11-10_rank_coupling_verdict/tools",
    _REPORTS / "2026-11-04_delivered_gate_lookup/tools",
    _REPORTS / "2026-11-04_rank_coupled_body/tools",
    _REPORTS / "2026-10-30_upsample_arm_dossier/tools",
):
    sys.path.insert(0, str(_p))

import arm_config  # noqa: E402
import e1_data  # noqa: E402
from rank_coupling_verdict import spearman  # noqa: E402

#: ⭐⭐ `G-1`'s anchors — the published figures this tool must reproduce before it may print
#: anything else. `03_FINDING.md` of `2026-11-10_rank_coupling_verdict`, and the finish line in
#: `00_README` §⭐⭐. ⛔ Editing these is editing the control; do not, without re-measuring.
_ANCHOR_BODY = 0.3127
_ANCHOR_GATE = 0.4418
_ANCHOR_BODY_SD = 0.0713
_ANCHOR_GATE_SD = 0.0583
#: Four decimals is what the published figures carry, so the gate is set at the last printed digit.
_TOL = 5e-5

#: `G-2`'s floor on distinct values, as a share of the FIRING cells scored here — ⚠️ a different
#: population from §9c `N9`'s 64,732-of-64,818, which is the whole global grid. Measured on this
#: population 2026-11-10: GATE **1.0000**, BODY (the draws' mean) **0.7796**. ⛔ The floor is set
#: well below both, because its job is to catch a DEGENERATE field — the §0.2 all-zero delivery —
#: not to rank resolution. A score at or below it is tied enough that `ρ_event` would be mostly
#: reporting its own ties, and it is refused rather than printed.
_DISTINCT_FLOOR = 0.10

DDOF = 1


def rho_event(index: np.ndarray, truth: np.ndarray, event_m: np.ndarray) -> float:
    """-> ρ_event for one fold: Spearman on midranks, over the event cells of the firing set."""
    return spearman(index[event_m], truth[event_m])


def distinct_share(x: np.ndarray) -> float:
    """-> the share of distinct values in ``x``. §9c `N9`: resolution is why an index wins."""
    return float(np.unique(x).size) / float(x.size) if x.size else 0.0


def _fold_inputs():
    """-> per fold: (score_month, truth, open_m, event_m, gate, draws). One pass, cached by e1."""
    grid, _cells, m0 = e1_data.truth_grid()
    gates = {mo: g for (g, _u, mo) in e1_data.load_gate_h1().values()}
    draws = {mo: d for (d, _u, mo) in e1_data.load_draws_h1().values()}
    tau = arm_config.resolve_tau()
    out = []
    for _fit, score_month in e1_data.FOLDS:
        g = gates[score_month]
        truth = grid[score_month - m0].astype(np.float64)
        open_m = arm_config.open_set(g, tau)
        out.append(
            {
                "month": score_month,
                "truth": truth,
                "open_m": open_m,
                "event_m": open_m & (truth > 0),
                "gate": g,
                "draws": draws[score_month].astype(np.float64),
            }
        )
    return out, tau


def _summarise(per_fold: list[float]) -> tuple[float, float]:
    a = np.asarray(per_fold, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=DDOF))


def _load_score(path: Path, months: list[int]) -> dict[int, np.ndarray]:
    """-> {month: score field}, refusing anything that does not cover this design's months."""
    with np.load(path) as z:
        if "score" not in z or "months" not in z:
            raise SystemExit(
                f"G-0 REFUSED: {path} must carry 'score' [n_months, n_cells] and 'months' "
                f"[n_months]; it carries {sorted(z.keys())}."
            )
        score, got = np.asarray(z["score"], dtype=np.float64), [int(m) for m in z["months"]]
    if score.ndim != 2 or score.shape[0] != len(got):
        raise SystemExit(
            f"G-0 REFUSED: score has shape {score.shape} against {len(got)} months — the two axes "
            "do not agree, so no row may be scored."
        )
    missing = [m for m in months if m not in got]
    if missing:
        raise SystemExit(
            f"G-0 REFUSED: the score file is missing score months {missing}. Partial coverage "
            "would score a different population from the control (C-351)."
        )
    return {m: score[got.index(m)] for m in months}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", type=Path, default=None, help="an .npz score field to add as a row")
    ap.add_argument("--label", default="SCORE", help="the row label for --score")
    args = ap.parse_args()

    t0 = time.time()
    print(__doc__.split("Usage::")[0])
    print("=" * 100)
    print("  ρ_event READOUT — the ranking body's instrument, control pinned")
    print("=" * 100)

    folds, tau = _fold_inputs()
    months = [f["month"] for f in folds]
    n_events = int(sum(int(f["event_m"].sum()) for f in folds))
    print(
        f"  τ = {tau} (read from the arm config) · {len(folds)} folds · score months "
        f"{months[0]}..{months[-1]} · {n_events} event cell-months · ddof={DDOF}"
    )

    rows: list[tuple[str, list[float], list[float]]] = []
    rows.append(("BODY (draws' mean)", [], []))
    rows.append(("GATE", [], []))
    for f in folds:
        body = f["draws"].mean(axis=1)
        rows[0][1].append(rho_event(body, f["truth"], f["event_m"]))
        rows[0][2].append(distinct_share(body[f["open_m"]]))
        rows[1][1].append(rho_event(f["gate"], f["truth"], f["event_m"]))
        rows[1][2].append(distinct_share(f["gate"][f["open_m"]]))

    if args.score is not None:
        by_month = _load_score(args.score, months)
        rows.append((args.label, [], []))
        for f in folds:
            s = np.asarray(by_month[f["month"]], dtype=np.float64)
            if s.shape != f["truth"].shape:
                raise SystemExit(
                    f"G-0 REFUSED: the score field for month {f['month']} has shape {s.shape} "
                    f"against the truth grid's {f['truth'].shape}."
                )
            rows[2][1].append(rho_event(s, f["truth"], f["event_m"]))
            rows[2][2].append(distinct_share(s[f["open_m"]]))

    # ── G-1 · the control, pinned by arithmetic before anything else is printed ────────────────
    body_mean, body_sd = _summarise(rows[0][1])
    gate_mean, gate_sd = _summarise(rows[1][1])
    print("\n  ⭐⭐ G-1 · THE CONTROL PIN — the published figures, recomputed here")
    ok = True
    for name, got, want, got_sd, want_sd in (
        ("BODY", body_mean, _ANCHOR_BODY, body_sd, _ANCHOR_BODY_SD),
        ("GATE", gate_mean, _ANCHOR_GATE, gate_sd, _ANCHOR_GATE_SD),
    ):
        hit = abs(got - want) <= _TOL and abs(got_sd - want_sd) <= _TOL
        ok &= hit
        print(
            f"    {name:<5} ρ_event {got:+.4f} ± {got_sd:.4f}   published {want:+.4f} ± "
            f"{want_sd:.4f}   {'PASS' if hit else '⛔ FAIL'}"
        )
    if not ok:
        print(
            f"\n  ⛔⛔ G-1 FAILED at tol {_TOL:g}. NOTHING is reported. Either the committed "
            "artifacts changed, or the estimator did, or this tool's population does not match "
            "the one that published those figures. Find out which before reading any row."
        )
        return 1
    print(f"    ⇒ the estimator and the population reproduce both published figures to {_TOL:g}.")

    # ── G-2 · resolution, because a file that exists is not a readout ──────────────────────────
    print("\n  ⛔ G-2 · DISTINCT VALUES — §0.2: a delivery served 0.0 for 2,330,712 cells")
    print("     and every structural check passed. This counts values, not files.")
    # ⚠️ The gate FAILS only on a --score row. The two control rows are known to be coarse —
    # that coarseness is §9c `N9`'s finding, not a defect to refuse — so they are printed and not
    # gated. Gating them would make this tool refuse to run on the very data it is pinned to.
    gated_label = rows[2][0] if len(rows) > 2 else None
    g2 = True
    for label, _rho, dist in rows:
        share = float(np.mean(dist))
        hit = share >= _DISTINCT_FLOOR
        if label == gated_label:
            g2 = hit
        print(
            f"    {label:<22} distinct {share:.4f} of firing cells   "
            f"{'ok' if hit else '⚠️ quantised'}"
        )
    if len(rows) > 2 and not g2:
        print(
            f"\n  ⛔⛔ G-2 FAILED: '{rows[2][0]}' is below the {_DISTINCT_FLOOR:.2f} floor, so "
            "its ρ_event is mostly reporting its own ties. NOT reported."
        )
        return 1

    # ── the table, which is the output ────────────────────────────────────────────────────────
    print("\n  THE READOUT · ρ_event on event cells, global grid, mean ± SD over folds")
    print(f"    {'index':<22}{'ρ_event':>10}{'±sd':>9}{'vs BODY':>10}{'vs GATE':>10}")
    for label, rho, _dist in rows:
        m, sd = _summarise(rho)
        print(
            f"    {label:<22}{m:>+10.4f}{sd:>9.4f}{m - body_mean:>+10.4f}{m - gate_mean:>+10.4f}"
        )

    print(
        "\n  ⚠️ Every number above is an INSTRUMENT READING. `04_roadmap` phase 3's "
        "pre-commitment\n     and `00_README` §6's kill rule both apply: no verdict about the "
        "ranking body may cite\n     a ρ_event taken outside `05_analysis_plan`'s criterion, "
        "which does not yet exist."
    )
    print(f"\n  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
