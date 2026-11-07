"""Does UpsampleConv2 act on MAGNITUDE? The paired test EXP-03 never ran.

Committed BEFORE it runs (``D-5``).

⛔⛔ WHAT THIS IS AND IS NOT

views-models **#710** and **#712** are MERGED: all 27 HydraNets are on UpsampleConv2. That
decision was argued on paired **AP**, where the effect is mixed -- three seeds positive, one
negative, one null (``exp03_pairs.json``). On 2026-11-05 the pooled observed/predicted ratio was
found to move toward 1 in **5 of 5 seeds** (-0.3304, -0.0380, -0.0377, -0.0330, -0.2219).

⇒ ⭐ **This test is CONFIRMATORY, not a safety check.** A one-signed magnitude effect is evidence
FOR the flip, so no outcome here argues against a shipped decision. ⛔ It may not be written up as
though the flip were in question. What is genuinely open is a research question: whether this
architecture's benefit sits in the BODY rather than in the gate, which EXP-03 never looked for.

⛔⛔ THE CONFOUND THIS TEST EXISTS TO REMOVE

The 5-of-5 result was computed on each arm's OWN open set, and those differ by arm:

    ctlfullzero_fortyseven   3,876 open cells        upctwofullzero_fortytwo   5,818 open cells

⇒ That is **unpaired support** -- each arm scored on a different population, which is exactly the
confound that nearly reached a report in the views-models session (ten arms scored on 32,026 to
54,086 cells each, the hardest-penalised arm on the fewest). ⛔ A pooled ratio compared across
arms with different support is not a comparison.

⇒ **Everything here is scored on cells open in BOTH arms of a matched pair.** ``G-P1`` asserts the
two scored populations are identical by INDEX, not by count -- two equal counts over different
cells would pass a count check and still be unpaired.

⚠️ Conditioning the population on both arms' PREDICTIONS is legitimate; conditioning it on the
OUTCOME would not be (``Gneiting2011``: an outcome-conditioned score is improper). Nothing here
selects on truth.

SLOTS (``C-351``)

    population      per seed: h1 cells open in BOTH ``ctlfullzero_<seed>`` and
                    ``upctwofullzero_<seed>``, 13 origins, months 457..469. n reported per seed
    transform       NONE. Truth ``lr_sb_best`` in raw counts; draws already in count space
    statistic       (a) pooled observed/predicted per arm on the SHARED population, and the paired
                    difference; (b) per-cell |observed - predicted| (magnitude MAE), paired
    estimator       ⛔ NO CRPS. The question is about magnitude, and CRPS on this field carries
                    every pitfall in the 2026-11-05 CRPS note -- 2.58 effective cells at h36, a
                    gate-independent crps_all, and a 13.9%-of-score estimator choice. MAE on the
                    shared open set is the narrow, interpretable quantity the question asks for
    interval        origin-blocked bootstrap via the COMMITTED ``gw_stratified_test``
                    (``reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py:68``),
                    ``resample="origin"``, 13 origins as blocks. ⛔ Lifted, not re-derived
    free parameter  none in the headline. ⚠️ MAE is itself a choice; §P-47 reports the median
                    absolute error beside it so the headline does not rest on one loss

⛔ THE GATES

    G-0    ⭐⭐ VOID. Both arms of every pair reproduce their OWN committed AP exactly -- the
           control on ``ap_ctl``, the treatment on ``ap_trt`` -- via the FROZEN
           ``average_precision``. 10 of 10 did so on 2026-11-05. Nothing is reported on failure
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-P1   ⭐⭐ SUPPORT IS PAIRED, by index equality. ⛔ Refuses two equal-sized populations over
           different cells, which a count check would accept
    G-P2   non-vacuity: the shared set must be non-empty and carry >= 100 events per seed, or that
           seed is refused rather than reported thin
    G-P3   ⭐ provenance: ``gw_stratified_test``'s source is the committed file, compared by
           stripped AST against the path above. A tidied copy of a lifted test is a different test

PRE-REGISTERED PREDICTIONS -- next free is P-44

    P-44  ⭐⭐ THE ONE THAT MATTERS. The pooled-ratio difference stays ONE-SIGNED toward 1 across
          all five seeds on the PAIRED support ⇒ the 5-of-5 was not a support artefact. ~70%
    P-45  The paired magnitude MAE favours UpsampleConv2 in at least 4 of 5 seeds. ~60%
    P-46  The origin-blocked CI on the MAE difference excludes zero in at least 2 of 5 seeds. ~50%
    P-47  ⭐ The sign of the MEDIAN absolute error difference agrees with the MAE's in at least 4
          of 5 seeds ⇒ the result is not carried by the few largest cells. ~65%
    P-48  The within-pair support asymmetry exceeds 10% in at least one seed. ⭐ Grounded: the own-
          set counts already span 3,876 to 5,818 ACROSS arms. ~75%

FALSIFIERS

    F10  the pooled-ratio difference is NOT one-signed on paired support ⇒ ⛔ the 5-of-5 result is
         WITHDRAWN as a support artefact, and the magnitude claim reverts to "not demonstrated"
    F11  MAE and median-AE disagree in sign in 3 or more seeds ⇒ no loss-independent direction
    F12  any gate fails ⇒ no number reported

⛔ WHAT THIS CANNOT SAY

    - ⛔ NOTHING about whether the flip was right. It was argued on AP, it shipped, and a
      one-signed magnitude effect supports it. No result here reopens it
    - h1 only; one target (``sb``); five seeds of one architecture pair
    - ⚠️ Five seeds one-signed is **p = 0.0625 two-sided** on a sign test. Suggestive, NOT
      significant at 0.05. That is stated here so it cannot be quietly upgraded later
    - It conditions on the COMPOSED draw mean, so it is a statement about the delivered emission,
      not about the body in isolation where the gate closed
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _sub in (
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
    "reports/2026-07-29_v2_scoreboard_dossier/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
):
    sys.path.insert(0, str(_REPO / _sub))

import conditional_magnitude as cm  # noqa: E402
import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402
from gw_stratified import gw_stratified_test  # noqa: E402
from lodestar_score import average_precision  # noqa: E402

_GW_SRC = _REPO / "reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py"
SEED_WORDS = pnf.SEED_WORDS
P45_MIN_SEEDS = 4
P46_MIN_SIGNIFICANT = 2
P48_MIN_ASYMMETRY = 0.10
G_P2_MIN_EVENTS = 100
N_BOOT = 2000


def gate_p3_provenance() -> tuple[bool, str]:
    """-> (passed, note). The lifted interval test must be the committed one, by stripped AST."""
    tree = ast.parse(_GW_SRC.read_text())
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "gw_stratified_test"
        ),
        None,
    )
    if fn is None:
        raise SystemExit(f"paired_magnitude: no top-level gw_stratified_test in {_GW_SRC}.")
    ast.get_docstring(fn)  # touched so a docstring-only edit is visibly ignored
    fn.body = [
        s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    dumped = ast.dump(fn)
    live = ast.dump(
        next(
            n
            for n in ast.parse(Path(gw_stratified_test.__code__.co_filename).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "gw_stratified_test"
        )
    )
    return (len(dumped) > 500 and _GW_SRC.samefile(gw_stratified_test.__code__.co_filename)), (
        f"{len(live)} AST chars, from {gw_stratified_test.__code__.co_filename}"
    )


def gate_p1_support_is_paired(
    open_a: np.ndarray, open_b: np.ndarray
) -> tuple[bool, np.ndarray, int]:
    """-> (passed, the shared mask, its size). Equality by INDEX, never by count.

    ⛔ Two equal-sized populations over DIFFERENT cells pass a count check and are still unpaired.
    That is the confound this whole script exists to remove, so the gate compares index arrays.
    """
    shared = open_a & open_b
    idx_a = np.flatnonzero(shared & open_a)
    idx_b = np.flatnonzero(shared & open_b)
    ok = bool(idx_a.size > 0 and np.array_equal(idx_a, idx_b) and idx_a.size == int(shared.sum()))
    return ok, shared, int(idx_a.size)


def gate_p2_non_vacuous(obs_shared: np.ndarray) -> tuple[bool, int]:
    """-> (passed, events in the shared set). A thin seed is refused, not reported thin."""
    n_ev = int((obs_shared > 0).sum())
    return (obs_shared.size > 0 and n_ev >= G_P2_MIN_EVENTS), n_ev


def load_arm(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int) -> dict[str, object]:
    """-> one arm's AP, per-cell prediction, truth, open mask and origin ids.

    Streams one origin at a time, then frees -- every ``y_pred.npy`` is 149 MB and the disk is at
    96% (#448).
    """
    with pnf._as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws = e1_data.load_draws_h1()
    origins = sorted(gate)
    y_bin = np.concatenate(
        [
            (grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] > 0).astype(np.float64)
            for o in origins
        ]
    )
    ap = float(average_precision(y_bin, np.concatenate([gate[o][0] for o in origins])))
    return {
        "ap": ap,
        "pred": np.concatenate([cm.predicted(draws[o][0]) for o in origins]),
        "obs": np.concatenate(
            [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
        ),
        "open": np.concatenate([(draws[o][0] > 0).any(axis=1) for o in origins]),
        "origin": np.concatenate([np.full(e1_data.N_CELLS, o, dtype=np.int32) for o in origins]),
    }


def main() -> None:
    print("=" * 112)
    print("DOES UpsampleConv2 ACT ON MAGNITUDE? The paired test EXP-03 never ran")
    print("=" * 112)
    print("  ⭐ CONFIRMATORY. views-models #710/#712 are MERGED; a one-signed magnitude effect is")
    print("     evidence FOR the flip. ⛔ No result here reopens a shipped decision.")
    print("  ⛔ Scored ONLY on cells open in BOTH arms of a pair -- G-P1 asserts that by INDEX.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    okp3, note3 = gate_p3_provenance()
    print(
        f"\n  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}"
    )
    print(f"  {'PASS' if okp3 else '⛔ FAIL'}  G-P3  {note3}")
    if not (ok8 and okp3):
        raise SystemExit("⛔ F12: a precondition failed. Nothing is reported.")

    rows = []
    for w in SEED_WORDS:
        ctl, trt = f"ctlfullzero_{w}", f"upctwofullzero_{w}"
        a = load_arm(ctl, grid, cells, m0)
        b = load_arm(trt, grid, cells, m0)

        d_ctl = abs(a["ap"] - pnf.anchor_for(w, "ap_ctl"))
        d_trt = abs(b["ap"] - pnf.anchor_for(w, "ap_trt"))
        if max(d_ctl, d_trt) > e1_data.G0_TOL:
            raise SystemExit(
                f"VOID G-0: {w} does not reproduce its anchors (|d| ctl {d_ctl:.3e}, "
                f"trt {d_trt:.3e}). Nothing is reported."
            )

        okp1, shared, n_idx = gate_p1_support_is_paired(a["open"], b["open"])
        own_a, own_b = int(a["open"].sum()), int(b["open"].sum())
        asym = abs(own_a - own_b) / max(own_a, own_b)
        obs = a["obs"][shared]
        okp2, n_ev = gate_p2_non_vacuous(obs)
        print(f"\n  ── seed {w} ─────────────────────────────────────────────────────────────────")
        print(f"     G-0 both anchor-exact (|d| {d_ctl:.1e} / {d_trt:.1e})")
        print(
            f"     {'PASS' if okp1 else '⛔ FAIL'}  G-P1  shared support identical by index "
            f"({n_idx:,} cells)"
        )
        print(
            f"     {'PASS' if okp2 else '⛔ FAIL'}  G-P2  {n_ev:,} events in the shared set "
            f"(floor {G_P2_MIN_EVENTS})"
        )
        print(
            f"     own open sets: ctl {own_a:,}  upctwo {own_b:,}  shared {int(shared.sum()):,}"
            f"   asymmetry {asym:.2%}  (P-48 threshold {P48_MIN_ASYMMETRY:.0%})"
        )
        if not (okp1 and okp2):
            print("     ⛔ refused; this seed contributes no number.")
            continue

        p_ctl, p_trt = a["pred"][shared], b["pred"][shared]
        r_ctl = float(obs.mean() / p_ctl.mean())
        r_trt = float(obs.mean() / p_trt.mean())
        mae_ctl, mae_trt = np.abs(obs - p_ctl), np.abs(obs - p_trt)
        med_ctl, med_trt = float(np.median(mae_ctl)), float(np.median(mae_trt))
        gw = gw_stratified_test(
            mae_ctl,
            mae_trt,
            np.ones(obs.size, bool),
            a["origin"][shared],
            n_boot=N_BOOT,
            seed=0,
            resample="origin",
        )
        print(
            f"     pooled obs/pred   ctl {r_ctl:>8.4f}   upctwo {r_trt:>8.4f}   "
            f"diff {r_trt - r_ctl:>+8.4f}   |ctl-1| {abs(r_ctl - 1):.4f} -> "
            f"|upctwo-1| {abs(r_trt - 1):.4f}"
        )
        print(
            f"     magnitude MAE     ctl {mae_ctl.mean():>8.4f}   upctwo {mae_trt.mean():>8.4f}   "
            f"paired diff {gw['mean_diff']:>+8.4f} ({gw['pct_reduction']:>+7.2%})"
        )
        print(
            f"       origin-blocked 95% CI [{gw['ci_lo']:+.4f}, {gw['ci_hi']:+.4f}]  "
            f"p {gw['p_boot']:.4f}  {'SIGNIFICANT' if gw['significant'] else 'not significant'}  "
            f"({gw['n_origins']} origin blocks)"
        )
        print(
            f"     median AE         ctl {med_ctl:>8.4f}   upctwo {med_trt:>8.4f}   "
            f"diff {med_ctl - med_trt:>+8.4f}"
        )
        rows.append(
            {
                "seed": w,
                "n": int(shared.sum()),
                "n_ev": n_ev,
                "asym": asym,
                "r_ctl": r_ctl,
                "r_trt": r_trt,
                "closer": abs(r_trt - 1) < abs(r_ctl - 1),
                "mae_d": float(gw["mean_diff"]),
                "sig": bool(gw["significant"]),
                "med_d": med_ctl - med_trt,
            }
        )
        del a, b

    print("\n" + "=" * 112)
    print("  SCORED")
    print("=" * 112)
    if len(rows) < len(SEED_WORDS):
        print(f"  ⚠️ only {len(rows)} of {len(SEED_WORDS)} seeds produced a number.")
    closer = sum(r["closer"] for r in rows)
    mae_win = sum(r["mae_d"] > 0 for r in rows)
    sig = sum(r["sig"] for r in rows)
    agree = sum((r["mae_d"] > 0) == (r["med_d"] > 0) for r in rows)
    one_signed = len({r["r_trt"] - r["r_ctl"] > 0 for r in rows}) == 1
    print(
        f"  P-44  pooled-ratio difference one-signed across seeds: {one_signed}  "
        f"(closer to 1 in {closer}/{len(rows)})"
    )
    print(f"  P-45  MAE favours upctwo in {mae_win}/{len(rows)} seeds (threshold {P45_MIN_SEEDS})")
    print(f"  P-46  CI excludes zero in {sig}/{len(rows)} seeds (threshold {P46_MIN_SIGNIFICANT})")
    print(f"  P-47  MAE and median-AE agree in sign in {agree}/{len(rows)} seeds")
    print(
        f"  P-48  max within-pair support asymmetry "
        f"{max((r['asym'] for r in rows), default=float('nan')):.2%}"
    )
    print(
        f"  F10   {'⛔ FIRED' if not one_signed else 'did not fire'} "
        f"-- the 5-of-5 pooled-ratio result "
        f"{'IS a support artefact, WITHDRAWN' if not one_signed else 'survives paired support'}"
    )
    print(
        f"  F11   {'⛔ FIRED' if len(rows) - agree >= 3 else 'did not fire'} "
        f"-- MAE/median-AE sign disagreement in {len(rows) - agree} seeds"
    )
    print("\n  ⚠️ Five seeds one-signed is p = 0.0625 two-sided on a sign test: suggestive, NOT")
    print("     significant at 0.05. Declared in the docstring before the run.")
    print("  ⛔ No result here reopens views-models #710/#712.")


if __name__ == "__main__":
    main()
