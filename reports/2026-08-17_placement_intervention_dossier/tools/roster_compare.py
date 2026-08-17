#!/usr/bin/env python3
"""roster_compare.py - is the quiet-gate diagnosis a property of one model or of the family?

EXP-02 found, on violet_visitor, that the gate KEEPS ITS SHAPE (Moran's I dips then recovers) and
LOSES ITS NERVE (gate_mean down 12x, committed cells 92 -> 9 against reality's ~116). Everything
expensive we might build next rests on that being general rather than one vehicle's property.

Prints three things per model, which are the three ways the diagnosis could fail to generalise:
  * gate_mean decay      - does confidence collapse at all?
  * Moran's I trajectory - does the SHAPE hold (violet) or collapse (truncated_smoke)?
  * committed cells      - the absolute count, against reality's ~116. A ratio is not enough; that
                           lesson cost a nearly-recommended top-K build this morning.

Reads whatever gate CSVs exist, so it is safe to run mid-batch.
"""

from __future__ import annotations

import collections
import csv
import glob
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
STEPS = (1, 6, 12, 18, 24, 35)
REAL_CELLS = 115.9  # mean active cells per origin-step in the real field (use_real, target sb)


def load(path: Path) -> dict[int, dict[str, float]]:
    rows = [r for r in csv.DictReader(open(path)) if int(r["target_idx"]) == 0]
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["step"])].append(r)
    out = {}
    for st, g in by.items():
        m = lambda k: statistics.mean(float(r[k]) for r in g)  # noqa: E731
        out[st] = {
            "gate_mean": m("gate_mean"),
            "moran": m("gate_moran_i"),
            "n": m("indep_n_active"),
        }
    return out


def main() -> None:
    files = sorted(glob.glob(str(RES / "gate_*_identity.csv")))
    if not files:
        print("no gate CSVs yet")
        return
    models = {
        Path(f).stem.replace("gate_", "").replace("_identity", ""): load(Path(f)) for f in files
    }

    for title, key, fmt in [
        ("CONFIDENCE - gate_mean (does it collapse?)", "gate_mean", "%9.5f"),
        ("SHAPE - Moran's I (holds = violet, collapses = smoke)", "moran", "%9.3f"),
        ("COMMITMENT - cells fired, against reality's %.0f" % REAL_CELLS, "n", "%9.1f"),
    ]:
        print(f"\n### {title}")
        print("%-17s" % "model" + "".join("%9s" % f"h{s}" for s in STEPS) + "   ratio")
        for name, d in sorted(models.items()):
            vals = [d[s][key] if s in d else None for s in STEPS]
            cells = "".join((fmt % v) if v is not None else "%9s" % "-" for v in vals)
            first, last = vals[0], vals[-1]
            ratio = f"{first / last:6.1f}x" if (first and last) else "     -"
            print("%-17s%s   %s" % (name, cells, ratio))

    print("\n### VERDICT per model")
    print("%-17s %-22s %-22s %s" % ("model", "confidence", "shape", "commitment @h18"))
    for name, d in sorted(models.items()):
        if 1 not in d or 18 not in d:
            continue
        conf = d[1]["gate_mean"] / d[18]["gate_mean"] if d[18]["gate_mean"] else float("inf")
        mor = d[18]["moran"] / d[1]["moran"] if d[1]["moran"] else float("nan")
        n18 = d[18]["n"]
        print(
            "%-17s %-22s %-22s %.1f cells (%.0fx short)"
            % (
                name,
                f"falls {conf:.0f}x" if conf > 2 else f"stable ({conf:.1f}x)",
                f"holds ({mor:.0%} of h1)" if mor > 0.6 else f"COLLAPSES ({mor:.0%})",
                n18,
                REAL_CELLS / n18 if n18 else float("inf"),
            )
        )


if __name__ == "__main__":
    main()
