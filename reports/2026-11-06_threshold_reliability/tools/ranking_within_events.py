"""⭐⭐ AMONG PLACES WHERE CONFLICT HAPPENED, CAN WE RANK THE BIG FROM THE SMALL? And how far up?

Committed BEFORE it runs (``D-5``). Five seeds, always.

⛔⛔ THE QUESTION, IN SIMON'S WORDS, AND WHY THE EXISTING TABLE CANNOT ANSWER IT

    *"It's perfectly possible that we can rank the bulk of the body but not the tail. But it's also
    perfectly possible that it is only the lower end of the bulk we can rank. Does the instrument
    differentiate?"*

``threshold_reliability.py`` measured discrimination at ten thresholds and found a smooth decay,
``DSC/UNC`` running 33.78% at ``t = 0`` down to 0.13% at ``t = 250``. ⛔ **But every one of those
rows is scored against all 842,634 cell-months, and 99.66% of them are zero.** So "above 1" is
overwhelmingly still the conflict/no-conflict question — which is why it reads 33.27%, almost
identical to "above 0"'s 33.78%. ⇒ **The low rows are inflated by the one thing the model is
already good at**, and that table cannot say how well we rank *among places where something
happened*.

This instrument fixes exactly that one thing: **the population is the event cells.**

⭐ THE POPULATION IS THE SAME FOR EVERY SEED, WHICH MAKES THE SPREAD MEAN SOMETHING

The event cells are a property of the TRUTH, and the truth does not vary across seeds. ⇒ all five
seeds are scored on the identical 2,905 cell-months, so the across-seed spread is **pure model
variation** with no population variation mixed in. ``G-E2`` pins the count.

⛔⛔ THE ONE THING THIS INSTRUMENT MAY NOT DO, MADE STRUCTURAL

Selecting cells on ``obs > 0`` is **outcome conditioning**. For a *calibration* claim that is the
Murphy-Winkler error and the score becomes improper (``Gneiting2011``: *"the composite score is
improper, unless the weight"* is constant). ⇒ ``MCB`` IS NOT COMPUTED, NOT RETURNED AND NOT
PRINTED, and ``G-E1`` parses this file's own AST to keep it that way.

⭐ Discrimination is a different matter. *"Among this fixed set of cells, does our ranking separate
the ones above ``t`` from the ones below?"* is a well-posed question about a ranking on a stated
population, and it is the question Simon asked. ⛔ But it is **not** a calibration statement and no
row here may ever be quoted as one.

THE THREE RANKING VARIABLES — the first moment, the second, and a free cross-check

    mean_draw   the mean of the 16 delivered draws. ⭐⭐ THE FIRST MOMENT. "Can the predicted
                magnitude rank the actual magnitude?" is Simon's question verbatim
    sd_draw     the sd of the 16 delivered draws. ⭐ The SECOND moment. ⚠️ This does NOT test
                "can we rank the variance" — per-cell variance is unobservable from one
                observation. It tests whether the predicted SPREAD carries magnitude information,
                which is the answerable half of that question and is stated as such
    frac_above  the fraction of draws exceeding ``t``. The quantity the previous table used, kept
                only so the two instruments can be compared

⚠️ Ranking is invariant to any monotone transform, so **none of these readouts depends on count vs
log1p space.** Simon said so and he is right; it is recorded here because it makes the question
well-posed independently of that argument.

⭐⭐ DSC NEEDS A NULL FLOOR TOO, FOR THE SAME REASON MCB DID

An isotonic refit on 2,905 points extracts some apparent signal from pure noise, so ``DSC > 0`` is
**not** evidence of ranking ability. Each row therefore carries

    DSC_0   the DSC an UNINFORMATIVE ranking would show on this population, from 400 random
            permutations of the ranking variable. Mean and 99th percentile

⛔ A permutation null, not a parametric one: it destroys the association while keeping the
population, the outcome and the tie structure of the ranking variable exactly as they are. ⚠️ This
is the correction the first threshold-reliability run shipped without and had to be amended for;
it is built in here from the start.

SLOTS (``C-351``)

    population      ⭐ **event cells only**: the 2,905 cell-months of 842,634 with
                    ``lr_sb_best > 0``, h1, months 457..469. ⛔ Outcome-selected, deliberately,
                    and therefore discrimination-only
    arms            the five CONTROL arms ``ctlfullzero_{fortytwo..fortyseven}``
    transform       NONE, and it would not matter: every readout is rank-based
    estimator       Brier score, plug-in, on the selected population; ``UNC - S_rc`` via PAV.
                    ⛔ NO MCB. ⛔ No CRPS
    normalisation   ``DSC/UNC`` — the share of the question the ranking answers — plus its own
                    permutation floor on the same scale
    free parameter  NONE. Thresholds are fixed below and reported in full
    thresholds      1, 2, 5, 10, 25, 50, 100, 250. ⛔ ``t = 0`` is omitted because every cell in
                    this population is an event, so the row would be empty by construction

⛔ THE GATES

    G-0    ⭐⭐ VOID. All five control arms reproduce their own committed ``ap_ctl`` exactly
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-E1   ⭐⭐ NO CALIBRATION STATISTIC. This file's AST must contain no assignment or print of a
           miscalibration term. Refuses to start if one appears
    G-E2   ⭐ the population is exactly the event cells, the SAME ones for every seed, and its
           size equals ``G-8``'s event count
    G-E3   ⭐⭐ no DSC is reported without its permutation floor
    G-E4   ⭐⭐ the ranking variables are truth-blind BY SIGNATURE: ``rank_values(draws, t)``
           cannot receive the outcome, so it cannot read it. ⚠️ The first version checked this at
           runtime by comparing two identical expressions and could not fire
    G-C2   ⭐⭐ FIVE SEEDS OR NOTHING

PRE-REGISTERED PREDICTIONS -- next free is P-57

    P-57  ⭐⭐ THE ONE THAT MATTERS, and it is Simon's hypothesis stated as a threshold. Ranking
          ability among events DECAYS with magnitude: ``mean_draw``'s ``DSC/UNC`` at ``t = 1`` is
          at least 3x its value at ``t = 100``. ~70%. ⭐ Grounded: the full-population table
          already decays 26.13% -> 1.99% over that span, though confounded by occurrence
    P-58  ⛔ The decay starts INSIDE THE BULK, not at the tail: ``mean_draw`` is above its own
          floor at ``t = 10`` and at or below it by ``t = 100``. ~55%. ⭐ Grounded: the
          full-population knee sat between t=10 and t=50, and this population removes the
          occurrence signal that was propping up the low rows
    P-59  ``mean_draw`` beats ``sd_draw`` at every threshold where either clears its floor. ~60%.
          ⚠️ Deliberately the boring direction: the two are strongly coupled in a negative
          binomial, since the variance is ``mu + mu^2/theta`` and therefore monotone in the mean,
          so they may be near-indistinguishable rather than ordered
    P-60  At least one threshold at or below 250 is NOT RESOLVABLE on all three variables. ~65%.
          ⭐ Grounded: only 10 of the 2,905 event cells exceed 250

FALSIFIERS

    F20  ⛔ ``mean_draw`` clears its floor at EVERY threshold ⇒ there is no measured loss of
         ranking ability within events, Simon's hypothesis is not supported on this population,
         and the research question must be re-posed rather than answered
    F21  ⛔ ``mean_draw`` is below its floor at EVERY threshold including ``t = 1`` ⇒ the model
         cannot rank magnitude among events AT ALL, and the decay question is moot because there
         is nothing to decay. ⭐ This is the branch that would kill the rank-coupled body outright
    F22  across-seed sd of ``DSC/UNC`` exceeds the figure itself at ``t = 1`` ⇒ ⛔ the headline
         row is seed noise and no decay profile is reported
    F23  any gate fails ⇒ ⛔ no number reported

⛔ WHAT THIS CANNOT SAY

    - h1 only; target ``sb`` only; control architecture only; one checkpoint
    - ⛔⛔ **NOTHING about calibration.** The population is outcome-selected; see ``G-E1``
    - ⛔ It does NOT answer "can we rank the variance". Per-cell variance is unobservable from one
      realisation. It answers only whether predicted spread carries magnitude information
    - ⚠️ 2,905 cells, thinning to 10 above 250. The floor column is what makes the thin rows
      readable; without it they would all look like findings
    - It ranks by the DELIVERED composed emission's draws. ⛔ So a null here still cannot
      separate *the ranking carries no information* from *the composition destroyed it* — that
      needs the gate as the index, and it is the NEXT run, not this one
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _sub in (
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
):
    sys.path.insert(0, str(_REPO / _sub))

import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402
from lodestar_score import average_precision  # noqa: E402

THRESHOLDS = (1, 2, 5, 10, 25, 50, 100, 250)
RANKERS = ("mean_draw", "sd_draw", "frac_above")
N_SEEDS_REQUIRED = 5
N_PERMUTATIONS = 400
FLOOR_Q = 0.99
NULL_SEED = 20261106

#: ⛔⛔ Names that would denote a calibration term. ``G-E1`` refuses this file if any is bound or
#: printed, because the population is outcome-selected and a calibration figure read off it would
#: be improper (``Gneiting2011``). ⭐ A deny-list is the right shape HERE, unlike ``G-C1``'s
#: allow-list, because the forbidden thing is a known, named quantity rather than an open set of
#: ratios somebody might invent.
FORBIDDEN_TERMS = ("mcb", "miscalibration", "mcb_null", "s_bar", "calibration_error")


def gate_e1_no_calibration_statistic(source: str | None = None) -> tuple[bool, list[str]]:
    """-> (passed, offending names). ⛔⛔ No miscalibration term may be computed in this file.

    Parses the file's own AST and refuses any assignment, keyword or dict key naming a
    calibration quantity. ⭐ The population here is selected on the outcome, which is exactly the
    condition under which a calibration score stops being proper — so the instrument is built
    unable to produce one rather than merely discouraged from printing it.
    """
    tree = ast.parse(_HERE.read_text() if source is None else source)

    #: ⛔ Skip the deny-list's own definition. Its elements ARE the forbidden strings, so a naive
    #: walk reports this gate as violating itself — which it did on the first run, and is the
    #: cheapest possible illustration of why a gate must be tested rather than reasoned about.
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(tgt, ast.Name) and tgt.id == "FORBIDDEN_TERMS" for tgt in node.targets
        ):
            skip.update(id(d) for d in ast.walk(node))

    bad: list[str] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id.lower() in FORBIDDEN_TERMS:
                bad.append(f"assignment to {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.lower() in FORBIDDEN_TERMS:
                bad.append(f"string key {node.value!r}")
        elif isinstance(node, ast.keyword) and (node.arg or "").lower() in FORBIDDEN_TERMS:
            bad.append(f"keyword {node.arg}")
    return (not bad), bad


# ── the discrimination statistic ─────────────────────────────────────────────────────────────


def pav(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """-> the weighted isotonic (non-decreasing) least-squares fit, standard stack form.

    ⚠️ Duplicated from ``threshold_reliability.py`` rather than imported: that module is the
    committed instrument behind ``M95`` and importing from it here would couple a published
    readout to this file's edits. WET, deliberately, and the two copies are pinned identical by
    a test.
    """
    means: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for xi, wi in zip(x.astype(np.float64), w.astype(np.float64)):
        means.append(float(xi))
        weights.append(float(wi))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            weights[-2] = total
            counts[-2] += counts[-1]
            means.pop()
            weights.pop()
            counts.pop()
    return np.repeat(np.asarray(means), np.asarray(counts))


def discrimination(rank_values: np.ndarray, exceeded: np.ndarray) -> tuple[float, float]:
    """-> ``(DSC, UNC)`` for one ranking variable against one threshold's 0/1 outcome.

    ⭐ Ties in the ranking variable are POOLED before the isotonic fit. Without that, cells the
    model cannot distinguish would be ordered arbitrarily and the fit would read that arbitrary
    order as signal — which matters here because the draws are quantised and ties are common.

    ``UNC`` is the outcome's own variance on this population; ``DSC = UNC - S_rc`` is the part of
    it the ranking accounts for. ⛔ No calibration quantity is formed or returned.
    """
    y = exceeded.astype(np.float64)
    levels, inverse = np.unique(rank_values, return_inverse=True)
    n_k = np.bincount(inverse, minlength=levels.size).astype(np.float64)
    s_k = np.bincount(inverse, weights=y, minlength=levels.size)
    fit = pav(s_k / n_k, n_k)
    total = float(n_k.sum())
    s_rc = float((s_k * (fit - 1.0) ** 2 + (n_k - s_k) * fit**2).sum() / total)
    ybar = float(s_k.sum() / total)
    unc = ybar * (1.0 - ybar)
    return unc - s_rc, unc


def discrimination_floor(
    rank_values: np.ndarray, exceeded: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """-> ``(mean DSC_0, FLOOR_Q-quantile DSC_0)`` under an UNINFORMATIVE ranking.

    ⭐⭐ A PERMUTATION null. Shuffling the ranking variable destroys its association with the
    outcome while leaving the population, the outcome's base rate and the ranking variable's tie
    structure exactly as they are. ⇒ whatever DSC comes back is the DSC that means NOTHING IS
    THERE, at this sample size and this tie structure.

    ⛔ Built in from the start, because shipping ``threshold_reliability.py``'s first run without
    the equivalent floor on MCB produced a readout that had to be amended the same day.
    """
    vals = np.array(
        [discrimination(rng.permutation(rank_values), exceeded)[0] for _ in range(N_PERMUTATIONS)]
    )
    return float(vals.mean()), float(np.quantile(vals, FLOOR_Q))


def rank_values(draws: np.ndarray, t: float) -> dict[str, np.ndarray]:
    """-> ``{ranker name: per-cell value}``. ⛔ The truth is NOT a parameter of this function.

    ⭐⭐ That is the whole of ``G-E4``, and it is why this is a function rather than a dict literal
    inside the loop. A ranker that cannot RECEIVE the outcome cannot read it, so truth-blindness
    is a property of the signature rather than something a runtime comparison has to establish.

    ⚠️ The first version of this file tried to check it at runtime by recomputing each ranker
    after permuting the truth and comparing — but both sides were the same expression over the
    same ``draws``, so the comparison was trivially true and the gate could not fire. The fix is
    structural: remove the truth from the function's reach.
    """
    return {
        "mean_draw": draws.mean(axis=1),
        "sd_draw": draws.std(axis=1),
        "frac_above": (draws > t).mean(axis=1),
    }


def gate_e4_rankers_cannot_see_the_truth() -> tuple[bool, str]:
    """-> (passed, detail). ``rank_values`` must take the draws and a threshold, and nothing else.

    ⛔ Checked by SIGNATURE, so adding an outcome parameter — the change that would make outcome
    conditioning possible — fails the gate before any number is produced. A test adds such a
    parameter and asserts the refusal.
    """
    params = list(inspect.signature(rank_values).parameters)
    if params != ["draws", "t"]:
        return False, f"rank_values takes {params}; only ('draws', 't') is permitted"
    source = inspect.getsource(rank_values)
    body = source.split('"""', 2)[-1]
    for forbidden in ("obs", "truth", "exceeded", "y_bin", "grid"):
        if forbidden in body:
            return False, f"rank_values' body mentions {forbidden!r}"
    return True, "rank_values takes (draws, t) only and its body names no outcome"


def verdict(dsc_unc: float, sd: float, floor: float) -> str:
    """-> one of exactly three tokens, assigned by arithmetic. ⛔ The only way to state a row.

    ⭐ The factor uses the seed-conservative edge ``value - sd``, the amendment adopted for
    ``threshold_reliability.py`` on 2026-11-06, so a stated factor can only rise with more seeds.
    """
    if sd > abs(dsc_unc):
        return "NOT RESOLVABLE"
    if floor <= 0.0:
        return "NOT RESOLVABLE"
    edge = dsc_unc - sd
    if edge <= floor:
        return "NO RANKING ABILITY"
    return f"RANKS {edge / floor:.1f}x floor"


# ── per-seed pass ────────────────────────────────────────────────────────────────────────────


def profile_seed(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int) -> dict:
    """-> ``{ranker: {threshold: row}}`` for ONE seed. ⛔ Never printed alone; see ``G-C2``."""
    with pnf._as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws_by_origin = e1_data.load_draws_h1()
    origins = sorted(gate)

    obs_all = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    y_bin = (obs_all > 0).astype(np.float64)
    ap = float(average_precision(y_bin, np.concatenate([gate[o][0] for o in origins])))
    word = next(w for a, (w, _s) in pnf.SEED_ARMS.items() if a == arm)
    if abs(ap - pnf.anchor_for(word, "ap_ctl")) > e1_data.G0_TOL:
        raise SystemExit(f"VOID G-0: {arm} differs from its committed ap_ctl. Nothing reported.")

    draws_all = np.concatenate([draws_by_origin[o][0] for o in origins])

    #: ⭐ THE POPULATION. Selected on the truth, which does not vary across seeds, so every seed
    #: is scored on the identical cells and the across-seed spread is pure model variation.
    event = obs_all > 0
    obs, draws = obs_all[event], draws_all[event]

    rng = np.random.default_rng(NULL_SEED + pnf.SEED_ARMS[arm][1])
    out: dict[str, dict[int, dict]] = {r: {} for r in RANKERS}
    for t in THRESHOLDS:
        exceeded = obs > t
        values = rank_values(draws, t)
        for name in RANKERS:
            dsc, unc = discrimination(values[name], exceeded)
            f_mean, f_q = discrimination_floor(values[name], exceeded, rng)
            if not (np.isfinite(f_mean) and np.isfinite(f_q)):
                raise SystemExit(f"⛔ F23: G-E3 — no floor for {name} at t={t}")
            out[name][t] = {
                "dsc_unc": dsc / unc if unc > 0 else float("nan"),
                "floor_unc": f_q / unc if unc > 0 else float("nan"),
                "n_above": int(exceeded.sum()),
                "n_levels": int(np.unique(values[name]).size),
            }
    return {"ap": ap, "n_pop": int(event.sum()), "rankers": out}


def _agg(vals: list[float]) -> tuple[float, float]:
    """-> (mean, across-seed sd). ⛔ "spread" means the sd here, stated so it cannot drift."""
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def main() -> None:
    print("=" * 112)
    print(
        "RANKING WITHIN EVENTS — among places where conflict happened, can we rank big vs small?"
    )
    print("=" * 112)
    ok_e1, bad = gate_e1_no_calibration_statistic()
    print(
        f"  {'PASS' if ok_e1 else '⛔ FAIL'}  G-E1  no calibration statistic in this file"
        + (f"  offending: {bad}" if bad else "")
    )
    if not ok_e1:
        raise SystemExit("⛔ F23: the population is outcome-selected; a calibration term is void.")
    ok_e4, detail_e4 = gate_e4_rankers_cannot_see_the_truth()
    print(f"  {'PASS' if ok_e4 else '⛔ FAIL'}  G-E4  {detail_e4}")
    if not ok_e4:
        raise SystemExit("⛔ F23: a ranking variable can reach the outcome.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F23: G-8 failed.")
    print("  ⛔ Control arms only. ⛔ DISCRIMINATION ONLY — no calibration claim is sayable here.")
    print("  ⚠️ Ranking is invariant to monotone transforms, so count vs log1p does not apply.")

    per_seed = {}
    for w in pnf.SEED_WORDS:
        arm = f"ctlfullzero_{w}"
        per_seed[w] = profile_seed(arm, grid, cells, m0)
        print(f"   {arm:<26} G-0 exact · G-E3 pass")

    pops = {per_seed[w]["n_pop"] for w in pnf.SEED_WORDS}
    ok_e2 = len(pops) == 1 and pops.pop() == ev8
    print(
        f"  {'PASS' if ok_e2 else '⛔ FAIL'}  G-E2  population is the {ev8:,} event cells, "
        f"identical for every seed"
    )
    if not ok_e2:
        raise SystemExit("⛔ F23: G-E2 failed — the population is not the event cells.")

    ok_c2 = len(per_seed) >= N_SEEDS_REQUIRED
    print(f"  {'PASS' if ok_c2 else '⛔ FAIL'}  G-C2  {len(per_seed)} seeds (floor 5)")
    if not ok_c2:
        raise SystemExit("⛔ F23: fewer than five seeds.")

    rows: dict[str, list[dict]] = {}
    for name in RANKERS:
        print("\n" + "=" * 112)
        label = {
            "mean_draw": "⭐⭐ mean of the 16 draws — THE FIRST MOMENT, Simon's question verbatim",
            "sd_draw": "⭐ sd of the 16 draws — the SECOND moment (NOT 'can we rank variance')",
            "frac_above": "fraction of draws above t — the previous table's index, for comparison",
        }[name]
        print(f"  RANKED BY: {label}")
        print("=" * 112)
        print(f"  {'t':>5} {'n>t':>6} {'lv':>5} {'DSC/UNC':>18} {'floor (q99)':>12}   VERDICT")
        print(f"  {'':>5} {'of 2905':>6} {'ties':>5} {'share ranked ± sd':>18} {'noise here':>12}")
        rows[name] = []
        for t in THRESHOLDS:
            vals = _agg([per_seed[w]["rankers"][name][t]["dsc_unc"] for w in pnf.SEED_WORDS])
            floor = _agg([per_seed[w]["rankers"][name][t]["floor_unc"] for w in pnf.SEED_WORDS])
            n_above = per_seed[pnf.SEED_WORDS[0]]["rankers"][name][t]["n_above"]
            lv = _agg([float(per_seed[w]["rankers"][name][t]["n_levels"]) for w in pnf.SEED_WORDS])
            v = verdict(vals[0], vals[1], floor[0])
            rows[name].append({"t": t, "val": vals, "floor": floor, "verdict": v})
            print(
                f"  {t:>5} {n_above:>6} {lv[0]:>5.0f} "
                f"{vals[0]:>9.2%} ± {vals[1]:<6.2%} {floor[0]:>11.2%}   {v}"
            )

    print("\n" + "=" * 112)
    print("  THE READOUT. ⛔ These sentences are the result.")
    print("=" * 112)
    for name in RANKERS:
        print(f"\n  ranked by {name}:")
        for r in rows[name]:
            print(f"    among events, above {r['t']:<4} : {r['verdict']}")

    print("\n  FALSIFIERS")
    m = rows["mean_draw"]
    f20 = all("RANKS" in r["verdict"] for r in m)
    f21 = all("RANKS" not in r["verdict"] for r in m)
    print(
        f"    F20 {'⛔ FIRED' if f20 else 'did not fire'} — mean_draw clears its floor EVERYWHERE "
        "⇒ no measured loss of ranking ability within events"
    )
    print(
        f"    F21 {'⛔ FIRED' if f21 else 'did not fire'} — mean_draw below its floor EVERYWHERE "
        "⇒ no magnitude ranking at all; this branch kills the rank-coupled body"
    )
    f22 = m[0]["val"][1] > abs(m[0]["val"][0])
    print(
        f"    F22 {'⛔ FIRED' if f22 else 'did not fire'} — t=1 sd {m[0]['val'][1]:.2%} "
        f"vs value {m[0]['val'][0]:.2%}"
    )

    ranked = [r for r in m if "RANKS" in r["verdict"]]
    if ranked:
        top = max(r["t"] for r in ranked)
        print(f"\n  ⭐⭐ mean_draw ranks up to and including {top} fatalities, and not above it.")
    else:
        print("\n  ⛔⛔ mean_draw does not clear its floor at ANY threshold within events.")


if __name__ == "__main__":
    main()
