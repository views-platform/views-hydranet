"""⭐⭐ Where does CRPS actually LIVE on this population? — the session's headline instrument finding.

Reads the committed `results/exp03_magnitude.json` (produced by `exp03_magnitude.py`) and reports,
per truth-magnitude band: the cell count, that band's CRPS per cell, and ⭐ **that band's share of
the TOTAL CRPS**. Then the arm's per-band effect, and QS99.

**Why it exists.** Simon's objection, in his words: *"manipulated nature of the data makes enormous
CRPS very hard to use."* That is a claim about the instrument, and it is measurable rather than
arguable. FAO-02 locks CRPS as the primary score; this does not propose a replacement, it
DECOMPOSES the locked score by truth magnitude — which is permitted where a new score is not.

⚠️ Bands were fixed BEFORE any arm was compared (they are a property of the truth distribution, not
tuned to an effect). ⛔ Nothing here is an interval: the per-band arm effects are sign counts across
five seeds, and the between-seed sd is printed beside every one of them precisely so they cannot be
read as resolved effects.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = ["fortytwo", "fortythree", "fortyfour", "fortyfive", "fortyseven"]
d = json.load(open(HERE.parent / "results" / "exp03_magnitude.json"))
print(__doc__)

for h in ("1", "36"):
    a = d["ctlfullzero_fortytwo"][h]
    print("=" * 98)
    print(f"h{h}   WHERE DOES CRPS LIVE?   (control, seed 42 — a property of the DATA, not the arm)")
    print("=" * 98)
    print(f"  {'truth band':<12}{'n cells':>10}{'% of cells':>12}{'crps/cell':>13}{'% of TOTAL crps':>18}")
    for k in a["n_band"]:
        n = a["n_band"][k]
        print(f"  {k:<12}{n:>10,}{100 * n / a['n']:>11.3f}%{a['crps_band'][k]:>13.3f}"
              f"{100 * a['crps_share_band'][k]:>17.2f}%")
    top = list(a["n_band"])[-1]
    print(f"  ⇒ the {a['n_band'][top]:,} cells of band {top} are {100*a['n_band'][top]/a['n']:.3f}% "
          f"of the population and carry {100*a['crps_share_band'][top]:.1f}% of the score.")

    print(f"\n  --- the ARM, per band: Δ crps/cell (lower = better), 5 seeds ---")
    print(f"  {'band':<12}{'ctl':>11}{'upctwo':>11}{'mean Δ':>11}{'sd':>10}  signs       unanimous")
    for k in a["n_band"]:
        c = np.array([d[f"ctlfullzero_{s}"][h]["crps_band"][k] for s in SEEDS])
        t = np.array([d[f"upctwofullzero_{s}"][h]["crps_band"][k] for s in SEEDS])
        dd = t - c
        u = bool(np.all(dd > 0) or np.all(dd < 0))
        print(f"  {k:<12}{c.mean():>11.3f}{t.mean():>11.3f}{dd.mean():>+11.3f}{dd.std(ddof=1):>10.3f}  "
              f"{''.join('+' if x > 0 else '-' for x in dd):<11} "
              f"{('YES ' + ('WORSE' if dd.mean() > 0 else 'BETTER')) if u else 'no'}")

    print()
    for key, lab in (("qs99", "QS99 all cells"), ("qs99_events", "QS99 event cells")):
        c = np.array([d[f"ctlfullzero_{s}"][h][key] for s in SEEDS])
        t = np.array([d[f"upctwofullzero_{s}"][h][key] for s in SEEDS])
        dd = t - c
        u = bool(np.all(dd > 0) or np.all(dd < 0))
        print(f"  {lab:<20} ctl {c.mean():>10.4f}  upctwo {t.mean():>10.4f}  Δ {dd.mean():>+9.4f}  "
              f"signs {''.join('+' if x > 0 else '-' for x in dd)}  "
              f"{'UNANIMOUS' if u else 'not unanimous'}")
    print()
