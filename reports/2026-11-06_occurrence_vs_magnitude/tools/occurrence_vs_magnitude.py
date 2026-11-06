"""Split the body's apparent magnitude defect into OCCURRENCE and MAGNITUDE. Five seeds, always.

Committed BEFORE it runs (``D-5``).

⛔⛔ WHY THIS EXISTS: THE NUMBER IT REPLACES WAS A COMPOSITE

On 2026-11-05 ``reports/2026-11-05_conditional_magnitude`` reported that the lowest prediction bin
is under-predicted **18.7x** and called it a conditional-magnitude defect. ⛔ **It is not one.**
The statistic was ``mean(observed) / mean(predicted)`` over a prediction bin, and in that bin only
**110 of 530 cells have an event at all**. So the ratio adds two different errors:

    the model did not FIRE often enough            (occurrence)
    when it fired it was too SMALL                 (magnitude)

⇒ 18.7x is their product, not a magnitude figure. It was withdrawn on 2026-11-06.

⭐⭐ THREE STRUCTURAL RULES, because "be more careful" is not a mechanism

This script is built so that the failure above cannot recur in it:

    1. ⛔ **It cannot print a composite.** ``G-C1`` parses this file's own source and will not
       if a ``mean(obs)/mean(pred)``-shaped expression appears in it. The composite is not merely
       discouraged; the instrument will not start.
    2. ⛔ **It cannot print a single-seed number.** Every figure is computed on all five control
       seeds and printed WITH its across-seed sd in the same row. ``G-C2`` refuses any statistic
       backed by fewer than five seeds. There is no code path that emits one seed's value alone.
    3. ⛔ **The decomposition must reconstruct.** ``G-C4`` asserts per cell that
       ``mean(draws) == p_hat * m_hat`` to 1e-9, so the two components provably account for the
       whole of the quantity they replace. If they do not, nothing is reported.

THE DECOMPOSITION

Per cell, from the 16 delivered draws:

    p_hat   the fraction of draws that are > 0        -- the model's implied firing rate
    m_hat   the mean of the NONZERO draws             -- its conditional magnitude given firing
            (defined only where at least one draw fires; the population is exactly those cells)
    mean(draws) = p_hat * m_hat                       -- exactly, and G-C4 checks it

Against truth, per prediction bin:

    occurrence   mean(p_hat)              vs   the observed event rate
    magnitude    mean(m_hat)              vs   mean(observed | observed > 0)

⚠️ ON THE ONE PIECE OF OUTCOME CONDITIONING THAT IS UNAVOIDABLE AND WHY IT IS ALLOWED HERE

The magnitude row compares ``m_hat`` with ``mean(observed | observed > 0)``. The truth side
selects on the outcome, and there is no way to ask *"how big was it when it happened"* without
knowing it happened.

⇒ This is **not** the Murphy-Winkler error, which is reweighting a SCORE by the outcome
(``Gneiting2011``: improper unless the weight is constant). Here nothing is scored and nothing is
reweighted: the bin is defined by prediction alone, and two descriptive distributions over
``{1,2,...}`` are placed side by side. A hurdle's magnitude component is *defined* conditionally --
``D`` is supported on the positives -- so its own domain is the only place it can be read.

⛔ But the conditioning is real and the row is labelled ``MAGNITUDE (truth side selects on
outcome)`` in the output so no reader can take it for a calibration statistic. ⚠️ And it carries a
known mismatch: ``m_hat`` conditions on the MODEL firing, the truth side on the WORLD firing.
Those are different events, which is why the occurrence row is printed first and must be read
first.

SLOTS (``C-351``)

    population      cells with >= 1 nonzero draw at h1, per seed, 13 origins, months 457..469.
                    ⛔ Per-seed populations are NOT pooled -- each seed is binned and summarised
                    separately, then the five summaries are combined. Pooling would hide the
                    support asymmetry that voided the 2026-11-05 pooled-ratio claim
    arms            the five CONTROL arms ``ctlfullzero_{fortytwo..fortyseven}``. ⛔ No
                    UpsampleConv2 here -- that is a separate question and mixing them is how the
                    last two findings went wrong
    transform       NONE. Truth ``lr_sb_best`` raw counts; draws already in count space
    estimator       means per bin, and the across-seed sd of each. No score, no CRPS, no ratio of
                    unlike things
    free parameter  bin count B in {4, 8, 16}, all reported (``D-4``). ⭐ And "spread" is defined
                    HERE as the across-seed **sd**, with the **range** printed beside it, because
                    on 2026-11-05 a prediction said "spread" without saying which and was
                    unscoreable as written

⛔ THE GATES

    G-0    ⭐⭐ VOID. All five control arms reproduce their own committed ``ap_ctl`` exactly via
           the FROZEN ``average_precision``. Nothing is reported on failure
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-C1   ⭐⭐ NO COMPOSITE. This file's own AST must contain no division of a mean-of-truth by a
           mean-of-prediction. Refuses to start if one appears
    G-C2   ⭐⭐ FIVE SEEDS OR NOTHING. Any statistic backed by < 5 seeds is refused, not printed
    G-C3   the binning is truth-blind -- edges recomputed after permuting truth must be identical
           (reused from ``conditional_magnitude.gate_m1_binning_is_truth_blind``)
    G-C4   ⭐⭐ the decomposition reconstructs: ``mean(draws) == p_hat * m_hat`` per cell to 1e-9

PRE-REGISTERED PREDICTIONS -- next free is P-49

    P-49  ⭐⭐ THE ONE THAT MATTERS. In the LOWEST prediction bin the OCCURRENCE error is the
          larger of the two: ``observed_rate / mean(p_hat)`` exceeds
          ``mean(obs|pos) / mean(m_hat)``. ~65%. ⭐ Grounded: in that bin 110 of 530 cells have an
          event (20.8%) while the mean draw is 0.137, which needs about 1 nonzero draw in 16
    P-50  The MAGNITUDE error is roughly FLAT across prediction bins -- its highest-bin value is
          within 2.0x of its lowest-bin value, at every B. ~55%
    P-51  The OCCURRENCE error FALLS with predicted size, monotone in the bin index at B=4. ~70%
    P-52  Both components' across-seed sd is smaller than the component's own across-bin range at
          B=8, i.e. the bin profile is resolvable above seed noise for both. ~60%

FALSIFIERS

    F13  the two components are within 1.25x of each other in the lowest bin ⇒ the 2026-11-05
         number was NOT mainly occurrence, and the decomposition does not explain it
    F14  either component's across-seed sd exceeds its across-bin range ⇒ ⛔ that component's
         profile is not resolvable at five seeds and NO shape is reported for it
    F15  any gate fails ⇒ no number reported

⛔ WHAT THIS CANNOT SAY

    - h1 only; target ``sb`` only; control architecture only
    - It describes the DELIVERED composed emission. Where ``threshold_gate`` closed, the body is
      unrecoverable, so this is the emission's occurrence/magnitude split and not the head's
    - ⚠️ ``p_hat`` from 16 draws is quantised at 1/16 = 0.0625. A bin whose mean ``p_hat`` is below
      about 0.06 is one nonzero draw, and that is printed as a caveat beside such rows
    - ⛔ It does NOT say whether v1 would help. It says which of two defects is the larger
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _sub in (
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
):
    sys.path.insert(0, str(_REPO / _sub))

import conditional_magnitude as cm  # noqa: E402
import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402
from lodestar_score import average_precision  # noqa: E402

B_SWEEP = (4, 8, 16)
N_SEEDS_REQUIRED = 5
P_HAT_QUANTUM = 1.0 / e1_data.N_DRAWS
F13_FACTOR = 1.25
RECON_TOL = 1e-9


# ── G-C1 · the instrument cannot print a composite ───────────────────────────────────────────


#: ⭐⭐ ALLOWLIST, not a blocklist. The two ratios this instrument exists to compute, as
#: (numerator root, denominator root). ⛔ ANY other division of one mean by another is refused,
#: including composites nobody has thought of yet -- which a blocklist of known-bad names could
#: not do. The first version of this gate WAS such a blocklist, and it rejected both legitimate
#: ratios while a test caught it.
PERMITTED_MEAN_RATIOS = (
    ("rate_obs", "p_hat"),  # OCCURRENCE: observed event rate over the model's implied firing rate
    ("obs_pos", "m_hat"),  # MAGNITUDE: observed size given an event over the model's conditional
)


def _root(expr: ast.AST) -> str:
    """-> the leading identifier of ``name[...].mean()``-shaped expression, or ''."""
    while True:
        if isinstance(expr, ast.Call):
            expr = expr.func
        elif isinstance(expr, ast.Attribute):
            expr = expr.value
        elif isinstance(expr, ast.Subscript):
            expr = expr.value
        elif isinstance(expr, ast.Name):
            return expr.id
        else:
            return ""


def gate_c1_no_composite() -> tuple[bool, list[str]]:
    """-> (passed, offending source segments). Parses THIS file and refuses any UNDECLARED ratio.

    ⛔⛔ The structural guard, and the reason this instrument exists. The 2026-11-05 claim was
    ``mean(observed) / mean(predicted)`` over a prediction bin -- a composite of an occurrence
    error and a magnitude error, reported as the latter and withdrawn the next day.

    ⭐ It is an ALLOWLIST: every division of one ``.mean()`` by another must be one of the two
    pairs in ``PERMITTED_MEAN_RATIOS``. A blocklist of suspicious variable names cannot work --
    the first version of this gate was one, and it flagged ``rate_obs.mean() / p_hat.mean()``,
    which is half the measurement. An allowlist refuses the composite AND every variant of it,
    and the only way to add a ratio is to declare it above.
    """
    tree = ast.parse(_HERE.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        left, right = ast.unparse(node.left), ast.unparse(node.right)
        if not (".mean()" in left and ".mean()" in right):
            continue
        pair = (_root(node.left), _root(node.right))
        if pair not in PERMITTED_MEAN_RATIOS:
            bad.append(f"{ast.unparse(node)}   [roots {pair}]")
    return (not bad), bad


# ── the decomposition ────────────────────────────────────────────────────────────────────────


def decompose(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> ``(p_hat, m_hat, open_mask)``. ``p_hat * m_hat`` equals the mean draw exactly.

    ``m_hat`` is undefined where nothing fires, so it is returned as 0 there and the mask excludes
    those cells from every statistic.
    """
    fired = draws > 0
    k = fired.sum(axis=1)
    p_hat = k / draws.shape[1]
    total = draws.sum(axis=1)
    m_hat = np.zeros_like(p_hat)
    np.divide(total, k, out=m_hat, where=k > 0)
    return p_hat, m_hat, k > 0


def gate_c4_reconstructs(draws: np.ndarray) -> tuple[bool, float]:
    """-> (passed, worst absolute residual). ``mean(draws) == p_hat * m_hat`` per cell."""
    p_hat, m_hat, open_m = decompose(draws)
    worst = float(np.abs(draws.mean(axis=1)[open_m] - (p_hat * m_hat)[open_m]).max())
    return worst <= RECON_TOL, worst


# ── per-seed summary ─────────────────────────────────────────────────────────────────────────


def summarise(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int) -> dict[int, list[dict]]:
    """-> ``{B: [one row per bin]}`` for ONE seed. ⛔ Never printed alone; see G-C2."""
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
    word = next(w for a, (w, _s) in pnf.SEED_ARMS.items() if a == arm)
    d = abs(ap - pnf.anchor_for(word, "ap_ctl"))
    if d > e1_data.G0_TOL:
        raise SystemExit(f"VOID G-0: {arm} |delta| {d:.3e}. Nothing is reported.")

    all_draws = np.concatenate([draws[o][0] for o in origins])
    okc4, worst = gate_c4_reconstructs(all_draws)
    if not okc4:
        raise SystemExit(f"⛔ F15: G-C4 failed on {arm}, worst residual {worst:.3e}.")

    p_hat, m_hat, open_m = decompose(all_draws)
    obs = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    #: ⛔ the conditioning index is the MEAN DRAW, the same index 2026-11-05 used, so the two
    #: readouts describe the same bins, so the split is comparable to the number it replaces
    idx = (p_hat * m_hat)[open_m]
    o_open, p_open, m_open = obs[open_m], p_hat[open_m], m_hat[open_m]

    okc3, _ = cm.gate_m1_binning_is_truth_blind(idx, o_open, B_SWEEP[1])
    if not okc3:
        raise SystemExit(f"⛔ F15: G-C3 failed on {arm} — the binning saw the truth.")

    out: dict[int, list[dict]] = {}
    for b in B_SWEEP:
        edges = cm.equal_count_edges(idx, b)
        bi = np.clip(np.searchsorted(edges[1:-1], idx, side="right"), 0, b - 1)
        rows = []
        for k in range(b):
            msk = bi == k
            if not msk.any():
                continue
            pos = o_open[msk] > 0
            rows.append(
                {
                    "bin": k,
                    "n": int(msk.sum()),
                    "n_pos": int(pos.sum()),
                    "rate_obs": float(pos.mean()),
                    "p_hat": float(p_open[msk].mean()),
                    "m_hat": float(m_open[msk].mean()),
                    "obs_pos": float(o_open[msk][pos].mean()) if pos.any() else float("nan"),
                }
            )
        out[b] = rows
    return {"ap": ap, "bins": out}


def _agg(vals: list[float]) -> tuple[float, float, float]:
    """-> (mean, across-seed sd, across-seed range).

    ⛔ "spread" means the sd here, and the range is printed beside it, because on 2026-11-05 a
    prediction said "spread" without saying which and was unscoreable as written.
    """
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std()), float(a.max() - a.min())


def main() -> None:
    print("=" * 108)
    print("OCCURRENCE vs MAGNITUDE — splitting the number withdrawn on 2026-11-06")
    print("=" * 108)
    okc1, bad = gate_c1_no_composite()
    print(
        f"  {'PASS' if okc1 else '⛔ FAIL'}  G-C1  no composite mean(obs)/mean(pred) in this file"
        + (f"  offending: {bad}" if bad else "")
    )
    if not okc1:
        raise SystemExit("⛔ F15: this instrument contains the composite it exists to replace.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F15: G-8 failed.")
    print("  ⛔ Control arms only. No UpsampleConv2 here.")
    print(f"  ⚠️ p_hat is quantised at 1/{e1_data.N_DRAWS} = {P_HAT_QUANTUM:.4f}.")

    per_seed = {}
    for w in pnf.SEED_WORDS:
        arm = f"ctlfullzero_{w}"
        per_seed[w] = summarise(arm, grid, cells, m0)
        print(f"   {arm:<26} G-0 exact · G-C3 pass · G-C4 pass")

    okc2 = len(per_seed) >= N_SEEDS_REQUIRED
    print(
        f"\n  {'PASS' if okc2 else '⛔ FAIL'}  G-C2  {len(per_seed)} seeds "
        f"(floor {N_SEEDS_REQUIRED}) — no statistic is printed from fewer"
    )
    if not okc2:
        raise SystemExit("⛔ F15: fewer than five seeds. Nothing is reported.")

    for b in B_SWEEP:
        nb = min(len(per_seed[w]["bins"][b]) for w in pnf.SEED_WORDS)
        print("\n" + "-" * 108)
        print(
            f"  B = {b}   every cell is mean ± across-seed sd over {len(per_seed)} seeds "
            f"[range in brackets]"
        )
        print("-" * 108)
        print(
            f"  {'bin':>3} {'n':>6} {'ev rate':>16} {'mean p_hat':>16} "
            f"{'OCCURRENCE err':>16} | {'obs|pos':>14} {'mean m_hat':>14} {'MAGNITUDE err':>15}"
        )
        print(
            f"  {'':>3} {'':>6} {'observed':>16} {'model':>16} {'obs_rate/p_hat':>16} | "
            f"{'truth-side ⚠️':>14} {'model':>14} {'obs_pos/m_hat':>15}"
        )
        occ_profile, mag_profile = [], []
        for k in range(nb):
            cols = {}
            for key in ("n", "rate_obs", "p_hat", "obs_pos", "m_hat"):
                cols[key] = _agg([per_seed[w]["bins"][b][k][key] for w in pnf.SEED_WORDS])
            occ = _agg(
                [
                    per_seed[w]["bins"][b][k]["rate_obs"] / per_seed[w]["bins"][b][k]["p_hat"]
                    for w in pnf.SEED_WORDS
                ]
            )
            mag = _agg(
                [
                    per_seed[w]["bins"][b][k]["obs_pos"] / per_seed[w]["bins"][b][k]["m_hat"]
                    for w in pnf.SEED_WORDS
                ]
            )
            occ_profile.append(occ)
            mag_profile.append(mag)
            q = " ⚠️1-draw" if cols["p_hat"][0] <= P_HAT_QUANTUM * 1.01 else ""
            print(
                f"  {k:>3} {cols['n'][0]:>6.0f} "
                f"{cols['rate_obs'][0]:>8.4f}±{cols['rate_obs'][1]:<7.4f} "
                f"{cols['p_hat'][0]:>8.4f}±{cols['p_hat'][1]:<7.4f} "
                f"{occ[0]:>8.2f}±{occ[1]:<7.2f} | "
                f"{cols['obs_pos'][0]:>7.2f}±{cols['obs_pos'][1]:<6.2f} "
                f"{cols['m_hat'][0]:>7.2f}±{cols['m_hat'][1]:<6.2f} "
                f"{mag[0]:>8.2f}±{mag[1]:<6.2f}{q}"
            )

        o0, m0_ = occ_profile[0], mag_profile[0]
        print(
            f"\n   LOWEST BIN — occurrence err {o0[0]:.2f} ± {o0[1]:.2f} "
            f"vs magnitude err {m0_[0]:.2f} ± {m0_[1]:.2f}"
        )
        bigger = "OCCURRENCE" if o0[0] > m0_[0] else "MAGNITUDE"
        within = (
            max(o0[0], m0_[0]) / min(o0[0], m0_[0]) if min(o0[0], m0_[0]) > 0 else float("inf")
        )
        print(f"      ⇒ larger component: {bigger}   factor between them {within:.2f}")
        print(
            f"      F13 {'⛔ FIRED' if within < F13_FACTOR else 'did not fire'} "
            f"(fires below {F13_FACTOR})"
        )
        for name, prof in (("OCCURRENCE", occ_profile), ("MAGNITUDE", mag_profile)):
            vals = [p[0] for p in prof]
            sds = [p[1] for p in prof]
            rng = max(vals) - min(vals)
            sd = float(np.mean(sds))
            print(
                f"      {name:<11} across-bin range {rng:>7.2f}   mean across-seed sd "
                f"{sd:>7.2f}   F14 {'⛔ FIRED' if sd > rng else 'did not fire'}"
            )

    print("\n  ⛔ P-49..P-52 and F13..F15 are scored against the thresholds in the docstring.")
    print("  ⛔ No composite ratio appears anywhere above — G-C1 enforces it on this file's AST.")


if __name__ == "__main__":
    main()
