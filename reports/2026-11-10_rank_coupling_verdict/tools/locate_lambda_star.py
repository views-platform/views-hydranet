"""S8 — ⭐ THE HEADLINE: locate lambda*, the ordering quality at which the mechanism becomes visible.

Epic #579, story S8 #587. Pre-registration: ``../PREREG.md``, committed before any tool existed.

⭐⭐ THE QUESTION, and it is the whole epic in one sentence: **how good does the body's ranking have
to be before the rank-coupled empirical draw is measurable at all?** Call that quality lambda*.
Compare it to what our body actually achieves. If lambda* is above us, the question moves from the
draw to the ranking, with a target. If it is below us, the mechanism is already measurable and the
2026-11-04 null was an instrument artefact.

⛔ WHY THE DIAL IS TRUSTWORTHY. It has a known right answer at BOTH ends: at ``lam = 0`` the
ordering is noise, so the test MUST read zero; at ``lam = 1`` it is truth-derived, so the test MUST
see it. S5 #584 §2a measured both and both behaved. That is what licenses reading the middle.

⛔⛔ THE ORACLE IS A CALIBRATION AXIS, NOT AN ARM (``PREREG.md`` §8 device 1). No number built on
truth is a statement about the model. The dial exists to calibrate the instrument's sensitivity;
only the REAL and GATE rankers are statements about the model.

⛔ THE CRITERION IS THE ONE WITH DEMONSTRATED POWER. The within-B gap in fair CRPS, each ordering
against the NOISE ordering, holding the gate, the open set, the pool and ``S`` fixed. ⚠️ Not S6
#585's bin-mean conditional bias, which is **VOID**: its potency control failed because a bin-mean
statistic cannot see placement within a bin.

⛔ THE MARGIN IS READ, NOT TRANSCRIBED. ``2 * sigma-hat_paired`` is parsed out of
``../results/s7_sigma_hat.out``, which S7 #586 measured in-run at a declared ``ddof=1`` across the
6 control-form seeds. A transcribed margin is the defect this epic already registered once.

⚠️ THE SCALE AXIS IS VACUOUS, BY CONSTRUCTION, AND IS ASSERTED RATHER THAN PADDED.
``PREREG.md`` §1 lists ``scale in {raw, log1p}`` among the swept axes. ⛔ A rank coupling depends on
the ordering ONLY, and ``log1p`` is strictly monotone on non-negative counts, so it cannot change a
single rank. ⇒ The two scales are the SAME arm. This file asserts that identity once (``G-L2``) and
then reports one column, because printing an identical column twice would inflate a grid without
adding a measurement.

**C-351 slots.** Population: the gate-open set (``gate >= tau``, tau read from the arm config) over
score months 463-469, 7 folds, cell counts printed by S5. Truth: ``lr_sb_best``, raw counts.
Ordering quality: Spearman vs truth on **EVENT cells** -- ⛔ the population matters, see
``rho_open`` in S5's output, where tie-breaking on the true zeros understates every arm. Estimator:
fair CRPS, mean over the open set; CIs origin-blocked over the 7 folds; sd at ``ddof=1``, declared.
Inference mode: ⭐ DELIVERED AUTOREGRESSIVE, h = 1. Arm: ``ctlfullzero_fortytwo``.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_rank_coupled_body/tools"))

import arm_config  # noqa: E402
import e1_data  # noqa: E402

from sim_rank_coupling import crps_ensemble  # noqa: E402
from rank_coupling_verdict import (  # noqa: E402
    SEED,
    _fold_pool,
    auc,
    couple,
    lambda_key,
    spearman,
)

R = 200
N_BOOT = 10_000
DDOF = 1
#: ⭐ DENSIFIED. ``sim_rank_coupling.py`` swept 6 points and left the whole interval above 0.75
#: unmeasured, which is precisely where the answer turned out NOT to be.
LAMS = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))
#: ⭐⭐ ``PREREG.md`` A-2's event-ordering dial. THIS is the axis lambda* is located on, because the
#: lambda dial above was MEASURED not to move ordering among events at all.
NUS = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))
_MARGIN_RE = re.compile(r"MARGIN = [0-9.]+ \* sigma-hat\s+([0-9.]+)")
_SIGMA_RE = re.compile(r"sigma-hat_paired \(ddof=(\d+)\)\s+([0-9.]+)")


def read_margin() -> tuple[float, float, int]:
    """-> (margin, sigma-hat, ddof) parsed from S7's committed output. ⛔ Never transcribed."""
    out = _HERE.parents[0] / "results/s7_sigma_hat.out"
    if not out.exists():
        raise SystemExit(
            f"locate_lambda_star: {out} is absent. ⛔ S7 #586 must run first: the margin is a "
            "MEASURED input, and a transcribed one is the defect this epic registered."
        )
    text = out.read_text()
    m, s = _MARGIN_RE.search(text), _SIGMA_RE.search(text)
    if not m or not s:
        raise SystemExit(f"locate_lambda_star: cannot parse sigma-hat / margin out of {out}")
    return float(m.group(1)), float(s.group(2)), int(s.group(1))


def nu_key(real_key: np.ndarray, truth: np.ndarray, event_m: np.ndarray, nu: float,
           rng: np.random.Generator) -> np.ndarray:
    """-> the REAL key with its EVENT-cell ordering dialled from REAL (nu=0) to perfect (nu=1).

    ⭐⭐ ``PREREG.md`` A-2's dial, and the one lambda* is stated on. The measured defect it replaces:
    the lambda dial's oracle term is a GLOBAL rank over a 99.63%-zero field, so every event sits in
    the top 0.37% of it and the noise term dominates among events at every lambda < 1. ⇒ lambda
    moves separation, never event ordering.

    ⛔ THE INVARIANT THAT MAKES THIS A ONE-VARIABLE DIAL: the key's VALUE MULTISET on the event
    cells is reassigned, never rewritten. ⇒ AUC, the open set, and every separation-driven effect
    are held EXACTLY fixed, and the only thing that moves is the order of the events among
    themselves.

    ⭐ ``nu = 0`` reproduces ``real_key`` exactly, which makes it an exact no-op control.
    """
    out = real_key.astype(np.float64).copy()
    idx = np.flatnonzero(event_m)
    if idx.size < 3:
        return out
    v = np.sort(out[idx])                      # the multiset, preserved
    t, k = truth[idx], out[idx]
    zt = np.argsort(np.argsort(t + rng.random(t.size) * 1e-9)) / max(t.size - 1, 1)
    zk = np.argsort(np.argsort(k)) / max(k.size - 1, 1)
    s_blend = nu * zt + (1.0 - nu) * zk
    out[idx[np.argsort(s_blend, kind="stable")]] = v
    return out


def main() -> None:
    t0 = time.time()
    print(__doc__)
    print("=" * 100)
    print("S8 · LOCATING LAMBDA*")
    print("=" * 100)

    margin, sigma, ddof = read_margin()
    tau = arm_config.resolve_tau()
    print(f"  margin  {margin:.4f} fair-crps  = 2 * sigma-hat {sigma:.4f} (ddof={ddof}), "
          f"READ from results/s7_sigma_hat.out")
    print(f"  tau = {tau}, read from the arm config; R = {R}; lambda grid = {len(LAMS)} points")

    grid, _cells, m0 = e1_data.truth_grid()
    gates = {mo: g for (g, _u, mo) in e1_data.load_gate_h1().values()}
    draws = {mo: d for (d, _u, mo) in e1_data.load_draws_h1().values()}

    # ── build every arm's per-fold score in one pass over the folds
    arm_names = ([f"lam={lam:.2f}" for lam in LAMS]
                 + [f"nu={nu:.2f}" for nu in NUS] + ["REAL", "GATE"])
    per_fold: dict[str, list[float]] = {a: [] for a in arm_names}
    per_fold_rho: dict[str, list[float]] = {a: [] for a in arm_names}
    per_fold_auc: dict[str, list[float]] = {a: [] for a in arm_names}
    g_l2, g_l3_auc = [], []

    for i, (fit_months, score_month) in enumerate(e1_data.FOLDS, start=1):
        g = gates[score_month]
        a_draws = draws[score_month].astype(np.float64)
        truth = grid[score_month - m0].astype(np.float64)
        pool = _fold_pool(grid, m0, fit_months)
        open_m = arm_config.open_set(g, tau)
        open_idx = np.flatnonzero(open_m)
        event_m = open_m & (truth > 0)

        keys = {f"lam={lam:.2f}": lambda_key(truth, lam, np.random.default_rng(SEED + i))
                for lam in LAMS}
        keys["REAL"] = a_draws.mean(axis=1)
        keys["GATE"] = g.astype(np.float64)
        for nu in NUS:
            keys[f"nu={nu:.2f}"] = nu_key(
                keys["REAL"], truth, event_m, nu, np.random.default_rng(777 + i)
            )

        # ── G-L2: log1p cannot change a rank, so the scale axis is one arm, not two
        raw = keys["REAL"][open_m]
        g_l2.append(np.array_equal(np.argsort(np.argsort(raw)),
                                   np.argsort(np.argsort(np.log1p(raw)))))

        rng = np.random.default_rng(SEED + i)
        for name, key in keys.items():
            acc = np.zeros(open_idx.size)
            for _ in range(R):
                yb = couple(pool, open_idx, key, g.size, e1_data.N_DRAWS, rng)
                _p, f = crps_ensemble(yb[open_m], truth[open_m])
                acc += f
            per_fold[name].append(float((acc / R).mean()))
            per_fold_rho[name].append(spearman(key[event_m], truth[event_m]))
            per_fold_auc[name].append(auc(key[open_m], event_m[open_m]))
        # ⭐ G-L3's invariant: the nu dial must hold SEPARATION exactly fixed while it moves
        # ordering. The value multiset on event cells is reassigned, so every nu arm must share
        # the REAL arm's AUC to machine precision.
        g_l3_auc.append(max(
            abs(per_fold_auc[f"nu={nu:.2f}"][-1] - per_fold_auc["REAL"][-1]) for nu in NUS
        ))
        print(f"  fold {i} · month {score_month} · {len(keys)} arms scored", flush=True)

    print(f"\n  {'PASS' if all(g_l2) else '⛔ FAIL'}  G-L2  log1p leaves every rank unchanged on "
          f"{len(g_l2)}/{len(g_l2)} folds ⇒ the raw/log1p scale axis is ONE arm, not two")
    worst = max(g_l3_auc)
    print(f"  {'PASS' if worst < 1e-12 else '⛔ FAIL'}  G-L3  the nu dial holds SEPARATION exactly "
          f"fixed: worst |AUC(nu) - AUC(REAL)| = {worst:.3e} over {len(g_l3_auc)} folds")

    # ── the sweep, reported WHOLE
    boot = np.random.default_rng(SEED)
    idx = boot.integers(0, len(e1_data.FOLDS), (N_BOOT, len(e1_data.FOLDS)))
    noise = np.array(per_fold["lam=0.00"])

    def row(name: str) -> dict:
        d = np.array(per_fold[name]) - noise
        bs = d[idx].mean(axis=1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return dict(
            name=name,
            rho=float(np.mean(per_fold_rho[name])),
            rho_sd=float(np.std(per_fold_rho[name], ddof=DDOF)),
            auc=float(np.mean(per_fold_auc[name])),
            gap=float(d.mean()),
            lo=float(lo),
            hi=float(hi),
            excludes=bool(lo > 0 or hi < 0),
            signs="".join("-" if x < 0 else "+" for x in d),
            clears=bool(abs(d.mean()) > margin and (lo > 0 or hi < 0)),
        )

    print("\n" + "-" * 100)
    print("1 · ⭐ THE SWEEP, REPORTED WHOLE  (PREREG.md §8 device 3: quoting the best cell of a")
    print("    grid without the grid beside it is not a result)")
    print("-" * 100)
    print("    gap = fair-crps(this ordering) - fair-crps(NOISE ordering). NEGATIVE = the ordering")
    print(f"    HELPED. 'clears' = |gap| > margin {margin:.4f} AND the CI excludes zero.")
    print()
    def show(rs):
        for r in rs:
            mark = "✅" if r["clears"] else "  "
            print(f"  {r['name']:<12}{r['auc']:>8.4f}{r['rho']:>+11.4f}{r['rho_sd']:>8.4f}"
                  f"{r['gap']:>+10.4f}   [{r['lo']:>+9.4f}, {r['hi']:>+9.4f}]{mark:>9}  "
                  f"{r['signs']}")

    hdr = (f"  {'arm':<12}{'AUC':>8}{'rho_event':>11}{'±sd':>8}{'gap':>10}"
           f"{'95% CI of the gap':>26}{'clears':>9}  signs")
    print("  ⚠️ THE LAMBDA GRID. ⛔ Read it in the AUC column, NOT in rho_event: PREREG.md A-2")
    print("     measured that this dial moves event-vs-zero SEPARATION and leaves ordering among")
    print("     events untouched (rho_event flat until lambda=1.00). It is a gate-like axis.")
    print(hdr)
    rows = [row(f"lam={lam:.2f}") for lam in LAMS]
    show(rows)

    print("\n  ⭐⭐ THE NU GRID — PREREG.md A-2's dial, and the axis LAMBDA* IS LOCATED ON.")
    print("     It moves ordering AMONG EVENT CELLS while holding separation (AUC) exactly fixed")
    print("     at the body's own. nu=0.00 reproduces REAL exactly ⇒ a second exact no-op.")
    print("     ⛔ gaps here are still measured against the NOISE ordering, so they are comparable")
    print("        with every other row in this file.")
    print(hdr)
    nu_rows = [row(f"nu={nu:.2f}") for nu in NUS]
    show(nu_rows)

    print("\n  ⭐ THE REAL ORDERINGS — the only arms that are statements about the model:")
    real_rows = [row("REAL"), row("GATE")]
    print(hdr)
    show(real_rows)

    # ── lambda*
    print("\n" + "-" * 100)
    print("2 · ⭐⭐ LAMBDA*")
    print("-" * 100)
    print("  ⭐ lambda* = the EVENT-ORDERING quality at which the improvement over the body's")
    print("     CURRENT ordering (nu=0.00 == REAL) first exceeds the margin. ⛔ Measured on nu,")
    print("     because PREREG.md A-2 showed lambda cannot move this quantity.")
    base = nu_rows[0]
    print(f"\n  nu=0.00 (== REAL): gap {base['gap']:+.4f}, rho_event {base['rho']:+.4f}, "
          f"AUC {base['auc']:.4f}")
    print(f"  ⇒ the incremental margin to beat: |gap(nu) - gap(nu=0)| > {margin:.4f}")

    lam_star = None
    print(f"\n  {'nu':<10}{'rho_event':>11}{'gap':>10}{'increment vs nu=0':>20}{'clears':>9}")
    for r in nu_rows:
        inc = r["gap"] - base["gap"]
        ok = abs(inc) > margin and inc < 0
        print(f"  {r['name']:<10}{r['rho']:>+11.4f}{r['gap']:>+10.4f}{inc:>+20.4f}"
              f"{'✅' if ok else '  ':>9}")
        if ok and lam_star is None:
            lam_star = r
    if lam_star is None:
        print("\n  ⛔⛔ THE SWEEP COULD NOT BRACKET LAMBDA* ON THIS AXIS. Not even a PERFECT event")
        print("      ordering (nu=1.00), at the body's own separation, improves on the body's")
        print("      CURRENT ordering by the margin. ⇒ Within B, event-ordering quality is NOT the")
        print("      binding constraint — separation is. That is a result, and it is stated as one.")
    else:
        prev_i = nu_rows.index(lam_star) - 1
        prev = nu_rows[prev_i] if prev_i >= 0 else None
        print(f"\n  ⇒ ⭐⭐ LAMBDA* = {lam_star['name']}, at rho_event {lam_star['rho']:+.4f}")
        if prev is not None:
            print(f"     bracketed: rho_event in ({prev['rho']:+.4f}, {lam_star['rho']:+.4f}]")

    print("\n" + "-" * 100)
    print("3 · ⛔ V6 — IS LAMBDA* ABOVE OUR ORDERING?  (PREREG.md §5, declared in advance)")
    print("-" * 100)
    real = real_rows[0]
    gate = real_rows[1]
    print(f"  our body's ordering (REAL), rho_event   {real['rho']:+.4f} ± {real['rho_sd']:.4f}"
          f"   AUC {real['auc']:.4f}")
    print(f"  the gate's ordering,        rho_event   {gate['rho']:+.4f} ± {gate['rho_sd']:.4f}"
          f"   AUC {gate['auc']:.4f}")
    if lam_star is None:
        print("\n  ⛔⛔ V6 IS UNEVALUABLE AS POSED, and that is itself the epic's finding.")
        print("      lambda* does not exist on the event-ordering axis: a PERFECT event ordering,")
        print("      at the body's own separation, does not improve on the body's current ordering")
        print("      by 2*sigma-hat. ⇒ The epic's question — 'how good must the body's RANKING be'")
        print("      — has the answer 'the ranking among events is not what the gain rides on'.")
        print("      What the gain rides on is SEPARATION, which is the GATE's job, and the gate")
        print(f"      already out-ranks the body here ({gate['rho']:+.4f} vs {real['rho']:+.4f}).")
    elif lam_star["rho"] > real["rho"]:
        print(f"  lambda* at rho_event                    {lam_star['rho']:+.4f}")
        print("\n  ⛔⛔ V6 FIRES — lambda* is ABOVE our ordering. This is PREREG.md's ANSWER 3,")
        print("      declared in advance as a RESULT and not a failure: the question moves from")
        print("      the empirical draw to the body's RANKING, with a numerical target.")
        print(f"      the gap to close: {lam_star['rho'] - real['rho']:+.4f} in rho_event.")
    else:
        print(f"  lambda* at rho_event                    {lam_star['rho']:+.4f}")
        print("\n  ✅ V6 DOES NOT FIRE — lambda* is AT OR BELOW our ordering. ⇒ The mechanism is")
        print("     measurable at the ordering quality the body already has, and the 2026-11-04")
        print("     null was an instrument artefact, not a property of the idea.")
        print(f"     headroom: our ordering exceeds lambda* by "
              f"{real['rho'] - lam_star['rho']:+.4f} in rho_event.")
        print(f"     REAL clears the margin: {'YES' if real['clears'] else 'NO'} "
              f"(|gap| {abs(real['gap']):.4f} vs margin {margin:.4f}, "
              f"{abs(real['gap']) / sigma:.2f} sigma-hat)")

    print("\n  ⚠️ WHAT THIS DOES NOT SAY. That the ordering carries measurable information is NOT")
    print("     that B should replace A. On CRPS against A, B LOSES by +8.52 (S5 #584 §2b) and")
    print("     even the oracle only ties A. The promotion criterion is S6 #585's, and it is VOID.")
    print("     ⇒ The verdict is S9 #588's to write, from all three readouts, not from this one.")

    print(f"\n  {time.time() - t0:.1f} s, CPU, R={R} couplings per arm per fold, zero GPU.")


if __name__ == "__main__":
    main()
