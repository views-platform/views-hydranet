"""What is the conditional-magnitude profile's NOISE FLOOR? Two floors, both measured.

Committed BEFORE it runs (``D-5``).

⛔⛔ WHY THIS EXISTS

``results/conditional_magnitude.out`` reports a ratio profile running ~18.7 in the lowest
prediction bin to ~1.0 in the highest, a gap of 17.9x at B=8. ⛔ **That is a signal with no
denominator.** The `models` session's closing point, arrived at after seven retractions of its own,
is the general form: *"is this metric discriminating?" has no answer independent of the model
family -- only "discriminating against THIS family's noise"*. A spread is not a finding until it is
measured against the spread the same instrument produces when the answer should not change.

⭐ TWO FLOORS, AND THE SECOND IS TIGHTER THAN THE FIRST

    SEED floor      five arms, seeds 42/43/44/45/47, one per seed. How much the profile moves
                    when the seed changes and nothing else does
    RE-RUN floor    ⭐⭐ five ``lander``/``orbiter`` PAIRS. Each pair is byte-identical in every
                    config file except its own name -- same ``torch_seed``, same ``np_seed``, same
                    git sha, same queryset, same partitions. ⇒ Their difference is pure run-to-run
                    nondeterminism: how much the profile moves when NOTHING changes

⇒ A signal must clear the seed floor to mean anything about seeds. If it does not clear the RE-RUN
floor, the instrument is simply unstable and no seed reading is possible at all.

⚠️ Seed **46** (``quiet_orbiter``) is EXCLUDED. ``exp03_pairs.json`` carries committed ``ap_ctl``
anchors at h1 for 42, 43, 44, 45 and 47 but not 46, so ``G-N0`` -- the void condition -- cannot be
anchored for it. ⛔ An unanchored arm is not cheaper than no arm; it is a population we cannot
prove we loaded correctly.

⭐ A BUILT-IN TEST NOBODY DESIGNED

``exp03_pairs.json``'s records are keyed by SEED, not by arm directory, and the two members of a
pair share a seed. ⇒ If ``G-0`` reproduces the same committed ``ap_ctl`` **exactly** for both
members, the two runs are bit-identical and the re-run floor is exactly zero. If only one matches,
the pair genuinely differs and the floor is real. Either answer is informative, so ``G-N4``
REPORTS it rather than refusing on it.

SLOTS (``C-351``): population, transform and estimator are ``conditional_magnitude.py``'s,
unchanged, re-run per arm. The only new free parameter is the bin count, swept as before.

⛔ WHY ``e1_data.ARM`` IS MUTATED, AND HOW THAT IS BOUNDED

``e1_data`` hardcodes one arm, and its ``anchor_ap`` hardcodes the literal ``"fortytwo"`` and is
``lru_cache``d -- so calling it for another arm would silently return seed 42's anchor. ⇒ This
script does NOT reuse ``anchor_ap``. It carries its own seed-keyed lookup and its own G-0, and it
sets ``e1_data.ARM`` only inside ``_as_arm``, which restores it on every path including an
exception. ``G-N1`` asserts the restore. ⛔ ``e1_data.py`` itself is not edited: three committed
scripts import it and E-1's result is already published against it.

⛔ THE GATES

    G-N0  ⭐⭐ VOID. Each of the five seed-floor arms reproduces ITS OWN committed ap_ctl to
          within G0_TOL via the FROZEN ``average_precision``. Fewer than 4 passing ⇒ F9
    G-N1  ``e1_data.ARM`` is restored after every arm. Asserted, not assumed
    G-N2  non-vacuity: the five committed anchors must be DISTINCT. If they were equal, G-N0
          would be satisfied by loading the same arm five times
    G-N4  ⭐ measurement, not refusal: per pair, the AP delta between members, and whether both
          reproduce the shared anchor
    G-M2/G-M3  reused from ``conditional_magnitude`` per arm per B

PRE-REGISTERED PREDICTIONS -- next free is P-38

    P-38  ⭐⭐ THE ONE THAT MATTERS. The profile's SHAPE clears the seed floor: the across-seed
          spread of the bin0/top gap is less than HALF the gap's own mean. ~70%
    P-39  The RE-RUN floor is smaller than the SEED floor on the same statistic. ~75%
    P-40  The pooled ratio's across-seed sd is below 0.15 on a level of ~1.68. ~60%
    P-41  ⭐ The bin0 ratio is the NOISIEST of the three statistics in relative terms, because it
          is an extreme bin of a skewed distribution. ~80%

FALSIFIERS

    F7  the gap's across-seed spread exceeds the gap's own mean ⇒ ⛔ the conditional-magnitude
        SHAPE is not established, and the finding is withdrawn to "not demonstrated on one arm"
    F8  the re-run floor EXCEEDS the seed floor ⇒ the instrument is unstable under nothing
        changing and no seed interpretation is possible. ⛔ Reported as an instrument failure,
        not as a result about seeds
    F9  fewer than 4 arms pass G-N0 ⇒ insufficient population for any floor

⛔ WHAT THIS CANNOT SAY

    - It is still h1, still one target, still one architecture. Five seeds of one model is a
      floor for THAT model, not for HydraNet in general
    - ⚠️ It measures the floor of the INSTRUMENT on the delivered cubes. It says nothing about
      refit variance beyond these five runs, and nothing about a retrained architecture
    - A gap that clears both floors is not evidence the programme's remedy works. It is only
      evidence the profile is real
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conditional_magnitude as cm  # noqa: E402
import e1_data  # noqa: E402
from lodestar_score import average_precision  # noqa: E402

#: arm directory -> (seed word used by exp03_pairs.json, seed int). ⛔ 46 is absent on purpose.
SEED_ARMS = {
    "calm_orbiter": ("fortytwo", 42),
    "keen_orbiter": ("fortythree", 43),
    "mellow_orbiter": ("fortyfour", 44),
    "nimble_orbiter": ("fortyfive", 45),
    "rapid_orbiter": ("fortyseven", 47),
}
#: byte-identical config pairs: same seed, same sha, differing only in their own name.
RERUN_PAIRS = (
    ("calm_lander", "calm_orbiter", "fortytwo"),
    ("keen_lander", "keen_orbiter", "fortythree"),
    ("mellow_lander", "mellow_orbiter", "fortyfour"),
    ("nimble_lander", "nimble_orbiter", "fortyfive"),
    ("rapid_lander", "rapid_orbiter", "fortyseven"),
)
B_REPORT = (8, 16)
P38_MAX_SPREAD_FRACTION = 0.5
P40_MAX_POOLED_SD = 0.15
F9_MIN_ARMS = 4
_ANCHORS = _REPO / "reports/2026-10-30_upsample_arm_dossier/results/exp03_pairs.json"


def anchor_for(seed_word: str) -> float:
    """-> the committed ``ap_ctl`` at h1 for ONE seed.

    ⛔ Deliberately not ``e1_data.anchor_ap``, which hardcodes ``"fortytwo"`` and is lru_cached:
    calling it per arm would return seed 42's anchor five times and G-N0 would pass vacuously.
    """
    rows = [
        r
        for r in json.loads(_ANCHORS.read_text())
        if r.get("seed") == seed_word and r.get("h") == 1
    ]
    if len(rows) != 1:
        raise SystemExit(
            f"profile_noise_floor: expected exactly one {{seed: {seed_word}, h: 1}} record in "
            f"{_ANCHORS}, found {len(rows)}. G-N0 has no anchor for this arm."
        )
    r = rows[0]
    if int(r["n"]) != e1_data.G8_CELL_MONTHS or int(r["n_org"]) != e1_data.N_ORIGINS:
        raise SystemExit(
            f"profile_noise_floor: the {seed_word} anchor was computed on n={r['n']} over "
            f"{r['n_org']} origins, not {e1_data.G8_CELL_MONTHS} over {e1_data.N_ORIGINS}. "
            "Comparing it to ours would compare two different populations."
        )
    return float(r["ap_ctl"])


@contextlib.contextmanager
def _as_arm(arm: str):
    """Point ``e1_data``'s loaders at ``arm``, and restore on every path including an exception."""
    was = e1_data.ARM
    e1_data.ARM = arm
    try:
        yield
    finally:
        e1_data.ARM = was


def profile(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int) -> dict[str, object]:
    """-> one arm's AP and its ratio profile at each reported B. Loads, measures, discards."""
    with _as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws = e1_data.load_draws_h1()
    if e1_data.ARM != "ctlfullzero_fortytwo":
        raise SystemExit(f"⛔ G-N1: e1_data.ARM was left as {e1_data.ARM!r} after loading {arm}.")

    origins = sorted(gate)
    score = np.concatenate([gate[o][0] for o in origins])
    y_bin = np.concatenate(
        [
            (grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] > 0).astype(np.float64)
            for o in origins
        ]
    )
    ap = float(average_precision(y_bin, score))

    pred = np.concatenate([cm.predicted(draws[o][0]) for o in origins])
    obs = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    open_m = np.concatenate([(draws[o][0] > 0).any(axis=1) for o in origins])
    p, o = pred[open_m], obs[open_m]

    out: dict[str, object] = {
        "arm": arm,
        "ap": ap,
        "n_open": int(open_m.sum()),
        "n_open_pos": int((o > 0).sum()),
        "pooled": float(o.mean() / p.mean()),
    }
    for b in B_REPORT:
        t = cm.bin_table(p, o, b)
        okm2, _nb, nbp = cm.gate_m2_non_vacuous(t)
        okm3, _ = cm.gate_m3_bins_are_ordered(t)
        out[f"B{b}"] = (
            None
            if not (okm2 and okm3)
            else {
                "bin0": t[0]["ratio"],
                "top": t[-1]["ratio"],
                "gap": t[0]["ratio"] / t[-1]["ratio"],
                "bins_with_event": nbp,
            }
        )
    return out


def _spread(xs: list[float]) -> tuple[float, float, float]:
    """-> (mean, sd, max-min). Population sd, five points; reported alongside the range."""
    a = np.asarray(xs, float)
    return float(a.mean()), float(a.std()), float(a.max() - a.min())


def main() -> None:
    print("=" * 110)
    print("THE CONDITIONAL-MAGNITUDE PROFILE'S NOISE FLOOR -- seed, and pure re-run")
    print("=" * 110)
    print("  ⛔ A spread is not a finding until it is measured against the spread the same")
    print("     instrument produces when the answer should not change.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(
        f"\n  {'PASS' if ok8 else '⛔ FAIL'}  G-8  {n8:,} cell-months  {ev8:,} events  max {mx8}"
    )

    anchors = {w: anchor_for(w) for _a, (w, _s) in SEED_ARMS.items()}
    distinct = len(set(anchors.values())) == len(anchors)
    print(f"  {'PASS' if distinct else '⛔ FAIL'}  G-N2  the five committed anchors are distinct")
    for w, v in anchors.items():
        print(f"        {w:>12}  ap_ctl {v:.16f}")
    if not (ok8 and distinct):
        raise SystemExit("⛔ a precondition failed. Nothing is reported.")

    results: dict[str, dict[str, object]] = {}
    arms = sorted({a for a in SEED_ARMS} | {a for p in RERUN_PAIRS for a in p[:2]})
    print(f"\n  loading {len(arms)} arms, one at a time\n")
    n_anchored = 0
    for arm in arms:
        r = profile(arm, grid, cells, m0)
        results[arm] = r
        word = SEED_ARMS.get(arm, (None,))[0] or next(
            (w for la, oa, w in RERUN_PAIRS if arm in (la, oa)), None
        )
        anc = anchor_for(word) if word else float("nan")
        d = abs(r["ap"] - anc)
        hit = d <= e1_data.G0_TOL
        n_anchored += int(hit and arm in SEED_ARMS)
        print(
            f"   {arm:<16} ap {r['ap']:.16f}  anchor({word}) |delta| {d:.3e}  "
            f"{'ANCHOR-EXACT' if hit else 'differs'}  open {r['n_open']:,}  "
            f"pooled {r['pooled']:.4f}"
        )

    print(
        f"\n  {'PASS' if n_anchored >= F9_MIN_ARMS else '⛔ FAIL'}  G-N0  "
        f"{n_anchored} of {len(SEED_ARMS)} seed-floor arms reproduce their own anchor exactly "
        f"(F9 fires below {F9_MIN_ARMS})"
    )
    if n_anchored < F9_MIN_ARMS:
        raise SystemExit("⛔ F9 FIRED: insufficient anchored arms for any floor.")

    print("\n" + "-" * 110)
    print("  ⭐ G-N4 · THE RE-RUN PAIRS: identical config, identical seed, different run")
    print("-" * 110)
    print(
        f"    {'pair':<10} {'AP lander':>18} {'AP orbiter':>18} {'|AP delta|':>11} "
        f"{'both anchor-exact?':>19}"
    )
    for la, oa, w in RERUN_PAIRS:
        a1, a2 = results[la]["ap"], results[oa]["ap"]
        anc = anchor_for(w)
        both = abs(a1 - anc) <= e1_data.G0_TOL and abs(a2 - anc) <= e1_data.G0_TOL
        print(f"    {w:<10} {a1:>18.12f} {a2:>18.12f} {abs(a1 - a2):>11.3e} {str(both):>19}")

    print("\n" + "-" * 110)
    print("  THE FLOORS")
    print("-" * 110)
    for stat_name, getter in (
        ("pooled ratio", lambda r: r["pooled"]),
        *[
            (f"B={b} bin0 ratio", (lambda b: lambda r: (r[f"B{b}"] or {}).get("bin0"))(b))
            for b in B_REPORT
        ],
        *[
            (f"B={b} top ratio", (lambda b: lambda r: (r[f"B{b}"] or {}).get("top"))(b))
            for b in B_REPORT
        ],
        *[
            (f"B={b} GAP bin0/top", (lambda b: lambda r: (r[f"B{b}"] or {}).get("gap"))(b))
            for b in B_REPORT
        ],
    ):
        seed_vals = [getter(results[a]) for a in SEED_ARMS]
        rerun_vals = [
            abs(getter(results[la]) - getter(results[oa]))
            for la, oa, _w in RERUN_PAIRS
            if getter(results[la]) is not None and getter(results[oa]) is not None
        ]
        if any(v is None for v in seed_vals):
            print(f"\n   {stat_name:<20} ⛔ refused: a per-arm profile was non-vacuous-FAILED")
            continue
        mean, sd, rng = _spread([float(v) for v in seed_vals])
        rr_mean = float(np.mean(rerun_vals)) if rerun_vals else float("nan")
        print(
            f"\n   {stat_name:<20} mean {mean:>9.4f}   SEED sd {sd:>8.4f} "
            f"({sd / mean if mean else float('nan'):>7.2%})   SEED range {rng:>8.4f}   "
            f"RE-RUN mean|diff| {rr_mean:>8.4f}"
        )
        print("      per-seed: " + "  ".join(f"{float(v):.4f}" for v in seed_vals))

    print("\n  ⛔ F7/F8/F9 and P-38..P-41 are scored in the dossier against the thresholds in")
    print("     this docstring. Changing a threshold is a change to the pre-registration.")


if __name__ == "__main__":
    main()
