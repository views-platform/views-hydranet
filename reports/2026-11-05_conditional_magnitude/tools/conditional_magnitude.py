"""Is the delivered emission's magnitude proportional to what it predicted? Binned by PREDICTION.

Committed BEFORE it runs (``D-5``). The gates and the pre-registered predictions below are the
contract; the run may not change them.

⛔⛔ WHY THIS, AND WHY BINNED BY PREDICTION

``reports/2026-11-04_rank_coupled_body_programme.md`` §1.1 and §1.2 assert that the body's
conditional magnitude is defective, and build a whole programme on it. The evidence cited is
``M90`` -- a dispersion measurement on a **teacher-forced** cube -- and ``ADR-073`` §0, an analytic
argument about the likelihood. ⇒ **Nobody has measured the delivered autoregressive forecast's
conditional magnitude.** This does that, once, on one arm.

⭐ Binned by the model's own predicted magnitude, never by the outcome. Conditioning a magnitude
statistic on the observation is the Murphy-Winkler calibration-vs-refinement error, and separately
``Gneiting2011`` proves a score conditioned on the outcome is **improper** ("the composite score is
improper, unless the weight" is constant). Our own `S4` event/background split has that defect and
is flagged for revision. ``G-M1`` enforces the BINNING mechanically rather than by discipline: the
bin edges are recomputed after the truth is permuted and must come back byte-identical.

⛔⛔ ONE COLUMN BREAKS THAT RULE, AND IS NAMED SO IT CANNOT BE USED BY ACCIDENT

``ratio_obs_conditioned`` divides ``mean(observed | observed > 0)`` by ``mean(predicted | observed
> 0)``. The denominator **selects cells by their outcome**, which is exactly the defect the
paragraph above says this instrument avoids. It was written into the first committed version of
this file by the same author who wrote that paragraph, found on reading the first run's output, and
is kept rather than deleted because the quantity is descriptively useful: *among the cells that did
have an event, how large was it against what was predicted?*

⇒ ⛔ **It may not be used for any calibration or bias inference.** The clean column is ``ratio``,
whose numerator and denominator are both taken over the whole prediction bin. ``mean_obs_pos`` is a
plain description of the outcome distribution inside a prediction bin and is not a ratio, so it
does not inherit the problem. ``test_the_obs_conditioned_column_is_named_as_such`` pins the name so
it cannot be quietly shortened back to something that reads innocent.

SLOTS (``C-351`` -- population, transform, estimator, normalisation, named with every number)

    population      arm ``ctlfullzero_fortytwo``, target ``sb``, DELIVERED AUTOREGRESSIVE h1,
                    13 origins, months 457..469, 64,818 global ``land`` cells = 842,634 cell-months
    transform       NONE. Truth is ``lr_sb_best`` in RAW COUNTS; the draws are already in count
                    space (max 728.0001 at h1 against a truth max of 2,482) -- no ``expm1``
    conditioning    per-cell MEAN of the 16 composed draws. ⛔ The COMPOSED draws: the emission
                    ships ``forecast_composition='threshold_gate'`` at tau=0.20, so this is a
                    statement about the delivered forecast, NOT about the body in isolation
    estimator       ratio of means per bin, and the conditional pair restricted to observed > 0.
                    ⛔ Both printed, both named. No CRPS is computed here and none is implied
    free parameter  bin count B in {4, 8, 16, 32, 64}, equal-count. ⛔ ALL swept and ALL reported
                    (``D-4``) -- three claims in this programme have been withdrawn because one
                    number crossed the value it was testing as its bin count moved

TWO POPULATIONS, REPORTED SEPARATELY AND NEVER POOLED

    (A) OPEN cells -- any of the 16 draws nonzero. ⭐ The cells on which the model actually makes a
        magnitude statement. ``threshold_gate`` passes body draws through UNCHANGED above tau, so
        on this set the composed draw IS the body draw
    (B) ALL 842,634 cell-months, where the prediction is exactly 0 on ~99.46%. Reported for
        completeness; ⛔ a ratio over (B) is dominated by cells carrying no prediction at all and
        is not the quantity in the question

⛔ THE GATES, declared before the run. A gate that cannot fire is not a gate; ``test_conditional_
magnitude.py`` shows each of G-M1..G-M3 refusing a deliberately broken input.

    G-0    ⭐⭐ COPY, exact, anchored on a COMMITTED IMPLEMENTATION (``W-98``): the gate mean
           reproduces ap_ctl = 0.5667056384496834 over all 13 origins via the FROZEN
           ``average_precision``. ⛔ VOID condition -- if this fails NOTHING is reported
    G-4    leakage, by counting: a construction gate reused from ``e1_data``
    G-5    per-origin alignment of the gate's units, the draws' units and the truth cell order
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed not quoted
    G-M1   ⭐ THE BINNING CANNOT SEE THE TRUTH. Edges recomputed after permuting truth must be
           byte-identical. ⛔ Refuses a binner that peeks
    G-M2   ⭐ NON-VACUITY. The open set must be non-empty and at least 2 bins must contain a
           positive outcome, or the ratio is undefined in every bin and no number is reported
    G-M3   ⭐ The bins must be ORDERED: mean predicted strictly increasing across bins. A
           non-monotone sequence means the binning is broken, not that the model is interesting

PRE-REGISTERED PREDICTIONS -- global numbering, next free is P-31 (P-27..P-30 are the E-1 set)

    P-31  ⭐⭐ THE ONE THAT MATTERS. observed/predicted is NOT FLAT across prediction bins.
          Threshold: max-over-bins / min-over-bins of the ratio exceeds 2.0, at EVERY B. ~70%
    P-32  The ratio EXCEEDS 1 in every bin -- the body is timid everywhere, not only on average.
          ⭐ Grounded, not guessed: the foundation measured size-ratio ~0.02 pooled. ~65%
    P-33  The direction is MONOTONE in predicted size. ⚠️ ~50% -- genuinely uncertain, and said so
          before the run rather than after
    P-34  The open set is 0.4473% of cell-months = 3,760..3,780 of 842,634. ⭐ Grounded:
          ``draw_sweep.out``'s G-3 measured threshold_gate leaving 0.004473 open at EVERY K

FALSIFIERS -- pre-committed

    F1  ratio flat within 2.0x at every B ⇒ ⛔ no conditional-magnitude defect is detectable at h1
        on this arm, and §1.1's premise is "NOT DEMONSTRATED on the delivered forecast". ⛔ That is
        a result about the premise, not licence to re-argue the test
    F2  fewer than 100 open cells carry a positive outcome ⇒ one event, not a measurement
    F3  the DIRECTION flips as B moves ⇒ no direction quotable (``D-4``)
    F4  any gate fails ⇒ no number is reported at all

⛔ WHAT THIS CANNOT SAY, committed before the run

    - Nothing about h36, in either direction. ``crps_effective_n`` is 2.58 there against ~490 here
    - n=1: one arm, one seed, one checkpoint, one target
    - It conditions on the COMPOSED draw mean. Where the gate closed the body is unrecoverable, so
      this measures the DELIVERED EMISSION's conditional magnitude, not the body's
    - ⚠️ A flat ratio would NOT prove the body is well calibrated -- only that this instrument, on
      the open set, at h1, cannot see a defect
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))

import e1_data  # noqa: E402

B_SWEEP = (4, 8, 16, 32, 64)
#: ``draw_sweep.out`` G-3: threshold_gate leaves this share of cell-month-passes open at every K.
OPEN_SHARE = 0.004473
P34_LO, P34_HI = 3_760, 3_780
P31_SPREAD = 2.0
F2_MIN_POSITIVE = 100


def predicted(draws: np.ndarray) -> np.ndarray:
    """-> per-cell predicted magnitude: the mean of the 16 composed draws.

    ⛔ Deliberately the mean and not the median: on the open set most cells still have a majority
    of zero draws, so a median collapses to 0 and the conditioning index would stop sorting --
    which is exactly the degeneracy that closed ``M89`` (1,321 distinct values, 99.66% at 0.0).
    """
    return draws.mean(axis=1)


def equal_count_edges(x: np.ndarray, b: int) -> np.ndarray:
    """-> ``b + 1`` equal-count edges of ``x``.

    ⛔ Reads ``x`` ONLY -- the truth is not a parameter of this function, which is what makes
    ``G-M1`` an identity rather than a tolerance.
    """
    return np.quantile(x, np.linspace(0.0, 1.0, b + 1))


def bin_table(pred: np.ndarray, obs: np.ndarray, b: int) -> list[dict[str, float]]:
    """-> one row per bin: n, n_pos, mean predicted, mean observed, and two ratios.

    ⭐ ``ratio`` is the clean one -- both means over the whole bin, nothing selected by outcome.
    ⛔ ``ratio_obs_conditioned`` restricts both means to cells with a positive outcome and is
    therefore outcome-conditioned; see the module docstring. Reported, never inferred from.

    ⚠️ ``np.searchsorted`` on interior edges only, so the top bin is closed -- otherwise the single
    largest prediction lands in a bin of its own and its ratio is one cell's.
    """
    edges = equal_count_edges(pred, b)
    idx = np.clip(np.searchsorted(edges[1:-1], pred, side="right"), 0, b - 1)
    rows = []
    for k in range(b):
        m = idx == k
        if not m.any():
            continue
        p, o = pred[m], obs[m]
        pos = o > 0
        rows.append(
            {
                "bin": k,
                "n": int(m.sum()),
                "n_pos": int(pos.sum()),
                "mean_pred": float(p.mean()),
                "mean_obs": float(o.mean()),
                "ratio": float(o.mean() / p.mean()) if p.mean() > 0 else float("nan"),
                "mean_obs_pos": float(o[pos].mean()) if pos.any() else float("nan"),
                "ratio_obs_conditioned": (
                    float(o[pos].mean() / p[pos].mean())
                    if pos.any() and p[pos].mean() > 0
                    else float("nan")
                ),
            }
        )
    return rows


# ── the three new gates ──────────────────────────────────────────────────────────────────────


def gate_m1_binning_is_truth_blind(
    pred: np.ndarray, obs: np.ndarray, b: int, binner=equal_count_edges
) -> tuple[bool, float]:
    """-> (passed, max |edge delta|). Permute the truth; the edges must not move at all.

    ⛔ The gate calls ``binner`` with the SAME signature the measurement uses, so substituting a
    truth-peeking binner is what breaks it -- which is the only way to break a gate on an identity.
    """
    rng = np.random.default_rng(0)
    a = np.asarray(binner(pred, b), dtype=np.float64)
    shuffled = obs.copy()
    rng.shuffle(shuffled)
    try:
        c = np.asarray(binner(pred, b, shuffled), dtype=np.float64)  # type: ignore[call-arg]
    except TypeError:
        #: the honest binner does not accept truth at all ⇒ it cannot depend on it
        return True, 0.0
    if a.shape != c.shape:
        return False, float("inf")
    worst = float(np.abs(a - c).max())
    return worst == 0.0, worst


def gate_m2_non_vacuous(rows: list[dict[str, float]]) -> tuple[bool, int, int]:
    """-> (passed, n bins, n bins carrying a positive outcome). Two bins minimum."""
    with_pos = sum(1 for r in rows if r["n_pos"] > 0)
    return (len(rows) >= 2 and with_pos >= 2), len(rows), with_pos


def gate_m3_bins_are_ordered(rows: list[dict[str, float]]) -> tuple[bool, int]:
    """-> (passed, the first offending bin index, or -1). Mean predicted strictly increasing."""
    mp = [r["mean_pred"] for r in rows]
    for k in range(1, len(mp)):
        if not mp[k] > mp[k - 1]:
            return False, k
    return True, -1


# ── the run ──────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()
    print("=" * 110)
    print("CONDITIONAL MAGNITUDE OF THE DELIVERED EMISSION, BINNED BY PREDICTION")
    print("=" * 110)
    print(
        f"  arm {e1_data.ARM}  target {e1_data.TARGET}  h1  months "
        f"{e1_data.H1_MONTHS[0]}..{e1_data.H1_MONTHS[-1]}"
    )
    print("  ⛔ DELIVERED AUTOREGRESSIVE. Truth lr_sb_best, RAW COUNTS, no transform.")
    print("  ⛔ Binned by PREDICTION ONLY. G-M1 enforces it.")
    print("  ⛔ The RATIO-OBS-COND column IS outcome-conditioned and may NOT be used for")
    print("     calibration inference. The clean column is `ratio`.")

    grid, cells, m0 = e1_data.truth_grid()
    gate = e1_data.load_gate_h1()
    draws = e1_data.load_draws_h1()

    ok0, ap, d0 = e1_data.gate_0_ap_copy(gate, grid, cells, m0)
    print(
        f"\n  {'PASS' if ok0 else '⛔ FAIL'}  G-0  ap {ap:.16f}  anchor "
        f"{e1_data.anchor_ap():.16f}  |delta| {d0:.3e}"
    )
    if not ok0:
        raise SystemExit("VOID: G-0 did not reproduce the anchor. Nothing is reported.")
    ok4, inter = e1_data.gate_4_leakage()
    print(f"  {'PASS' if ok4 else '⛔ FAIL'}  G-4  per-fold overlap {inter}")
    ok5, bad5 = e1_data.gate_5_alignment(gate, draws, cells)
    print(f"  {'PASS' if ok5 else '⛔ FAIL'}  G-5  misaligned origins {bad5}")
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8  {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not (ok4 and ok5 and ok8):
        raise SystemExit("⛔ F4: a gate failed. No number is reported.")

    origins = sorted(gate)
    pred = np.concatenate([predicted(draws[o][0]) for o in origins])
    obs = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    open_m = np.concatenate([(draws[o][0] > 0).any(axis=1) for o in origins])

    share = float(open_m.mean())
    n_open, n_open_pos = int(open_m.sum()), int((obs[open_m] > 0).sum())
    print(
        f"\n  OPEN set: {n_open:,} of {obs.size:,} cell-months = {share:.6%}  "
        f"(P-34 expects {P34_LO:,}..{P34_HI:,}; G-3's share {OPEN_SHARE:.6f})"
    )
    print(
        f"  of those, {n_open_pos:,} carry a positive outcome  (F2 fires below {F2_MIN_POSITIVE})"
    )
    if n_open_pos < F2_MIN_POSITIVE:
        raise SystemExit(f"⛔ F2 FIRED: only {n_open_pos} open cells carry an event.")

    for label, mask in (("(A) OPEN cells", open_m), ("(B) ALL cell-months", np.ones_like(open_m))):
        p, o = pred[mask], obs[mask]
        print("\n" + "-" * 110)
        print(
            f"  {label}   n {p.size:,}   mean_pred {p.mean():.6f}   mean_obs {o.mean():.6f}   "
            f"pooled ratio {o.mean() / p.mean() if p.mean() > 0 else float('nan'):.4f}"
        )
        print("-" * 110)
        for b in B_SWEEP:
            rows = bin_table(p, o, b)
            okm1, w1 = gate_m1_binning_is_truth_blind(p, o, b)
            okm2, nb, nbp = gate_m2_non_vacuous(rows)
            okm3, bad3 = gate_m3_bins_are_ordered(rows)
            flags = (
                f"G-M1 {'PASS' if okm1 else '⛔FAIL'} (|d| {w1:g}) · "
                f"G-M2 {'PASS' if okm2 else '⛔FAIL'} ({nbp}/{nb} bins w/ event) · "
                f"G-M3 {'PASS' if okm3 else f'⛔FAIL at bin {bad3}'}"
            )
            print(f"\n   B={b:<3}  {flags}")
            if not (okm1 and okm2 and okm3):
                print("      ⛔ F4: gate failed at this B; no row reported.")
                continue
            print(
                f"      {'bin':>4} {'n':>8} {'n_pos':>7} {'mean_pred':>11} {'mean_obs':>11} "
                f"{'ratio':>9} {'mean_obs|pos':>13} {'RATIO-OBS-COND':>15}"
            )
            for r in rows:
                print(
                    f"      {r['bin']:>4} {r['n']:>8,} {r['n_pos']:>7,} {r['mean_pred']:>11.5f} "
                    f"{r['mean_obs']:>11.5f} {r['ratio']:>9.3f} {r['mean_obs_pos']:>13.3f} "
                    f"{r['ratio_obs_conditioned']:>15.3f}"
                )
            fin = [r["ratio"] for r in rows if np.isfinite(r["ratio"]) and r["ratio"] > 0]
            if len(fin) >= 2:
                spread = max(fin) / min(fin)
                mono_up = all(fin[k] >= fin[k - 1] for k in range(1, len(fin)))
                mono_dn = all(fin[k] <= fin[k - 1] for k in range(1, len(fin)))
                print(
                    f"      ⇒ ratio spread max/min {spread:.3f}  "
                    f"(P-31 threshold {P31_SPREAD})  monotone "
                    f"{'UP' if mono_up else 'DOWN' if mono_dn else 'NEITHER'}  "
                    f"all>1 {all(r > 1.0 for r in fin)}"
                )

    print(f"\n  done in {time.time() - t0:.1f} s")
    print("  ⛔ P-31..P-34 and F1..F4 are scored in the dossier, against the thresholds in this")
    print("     docstring. Changing a threshold is a change to the pre-registration.")


if __name__ == "__main__":
    main()
