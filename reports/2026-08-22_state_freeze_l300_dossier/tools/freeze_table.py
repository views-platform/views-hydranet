#!/usr/bin/env python3
"""freeze_table.py — assemble the state-freeze table, keyed by SEED as well as arm.

**The bug this replaces.** The first finisher built its table from the score CSV's `model` column,
which for these arms is just ``none``/``hidden``/``cell``/``all`` — it does **not identify the
seed**. Both seeds therefore collapsed onto four rows, last-write-wins, and the baseline lookup
silently matched nothing so the comparison section rendered empty. The seed lives in the FILENAME.

That is the same defect class as the `aggregate_seeds` label collision fixed the day before
(C-303/C-304 territory): *keying on a label that does not identify the run when the filename does*.
Hence this is a tested module rather than a heredoc — a results tool with no test is how the first
one shipped wrong.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys

ARMS = ("none", "hidden", "cell", "all")


def _order(arms):
    """Bare modes in canonical order, then weighted arms by ascending weight."""
    bare = [a for a in ARMS if a in arms]
    weighted = sorted((a for a in arms if "@" in a), key=lambda a: float(a.split("@")[1]))
    return bare + weighted


HS = (1, 6, 18, 36)
#: published free-running AP@h18 (ledger M34) — the `none` arm must reproduce these
BASELINE_H18 = {"fullzero_fortythree": 0.3318, "fullzero_fortytwo": 0.3298}
# `cell_0.5` is the weighted-anchor arm (`cell@0.5`, `@`->`_` for the filename). Without the
# optional suffix these were silently skipped — correct per "ignore, do not guess", but the
# summary then omitted four measured arms without saying so.
_FNAME = re.compile(
    r"^score_(?P<seed>fullzero_[a-z]+)_(?P<arm>none|hidden|cell|all)(?P<w>_[0-9.]+)?\.csv$"
)


def read_results(results_dir: str, target: str = "sb") -> dict[str, dict[str, dict[int, float]]]:
    """{seed: {arm: {h: AP}}}, keyed from the FILENAME, not the CSV's model column."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "score_*.csv"))):
        m = _FNAME.match(os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            rows = {
                int(r["h"]): float(r["AP"]) for r in csv.DictReader(fh) if r["target"] == target
            }
        if rows:
            arm = m["arm"] + (m["w"].replace("_", "@") if m["w"] else "")
            out.setdefault(m["seed"], {})[arm] = rows
    return out


def check_h1_invariant(data) -> list[str]:
    """h1 must be identical across arms: there is no feedback at step 1, so nothing to freeze."""
    problems = []
    for seed, arms in data.items():
        vals = {a: r.get(1) for a, r in arms.items() if r.get(1) is not None}
        if len(set(round(v, 6) for v in vals.values())) > 1:
            problems.append(f"{seed}: h1 differs across arms {vals} — freezing cannot affect h1")
    return problems


def check_control_reproduces(data, tol: float = 5e-4) -> list[str]:
    """The `none` arm must reproduce the published free-running number for that seed (M34)."""
    problems = []
    for seed, arms in data.items():
        exp = BASELINE_H18.get(seed)
        got = arms.get("none", {}).get(18)
        if exp is None:
            # A seed with no published baseline cannot be checked — say so, because `render` prints
            # "every `none` arm reproduces its published value" and would otherwise assert a
            # falsifier that never ran for this seed.
            problems.append(
                f"{seed}: no published baseline in BASELINE_H18 — the control check did NOT run"
            )
            continue
        if got is None:
            problems.append(f"{seed}: no `none` arm at h18 — the control check could not run")
            continue
        if abs(got - exp) > tol:
            problems.append(
                f"{seed}: control h18 {got:.4f} != published {exp:.4f} — vehicle mismatch"
            )
    return problems


def render(data) -> str:
    L = ["# State-freeze at L=300 — auto-assembled", ""]
    problems = check_h1_invariant(data) + check_control_reproduces(data)
    L += (
        ["## ⚠️ FALSIFIER FAILED — do not read the table", ""] + [f"- {p}" for p in problems] + [""]
        if problems
        else [
            "**Falsifiers pass:** h1 identical across arms (no feedback at step 1), and every `none` "
            "arm reproduces its published free-running value (M34).",
            "",
        ]
    )
    for seed in sorted(data):
        L += [
            f"## `{seed}`",
            "",
            "| arm | " + " | ".join(f"h{h}" for h in HS) + " | h18 vs none |",
            "|---|" + "--:|" * (len(HS) + 1),
        ]
        none18 = data[seed].get("none", {}).get(18)
        for arm in _order(data[seed]):
            r = data[seed].get(arm)
            if not r:
                continue
            d = "—" if arm == "none" or none18 is None else f"{r[18] - none18:+.4f}"
            L.append(
                f"| `{arm}` | "
                + " | ".join(f"{r.get(h, float('nan')):.4f}" for h in HS)
                + f" | {d} |"
            )
        L.append("")
    L += ["## Mean over seeds (h18)", "", "| arm | mean | seeds |", "|---|--:|---|"]
    all_arms = _order({a for s in data for a in data[s]})
    for arm in all_arms:
        v = [data[s][arm][18] for s in sorted(data) if arm in data[s] and 18 in data[s][arm]]
        if v:
            L.append(
                f"| `{arm}` | {statistics.mean(v):.4f} | {', '.join(f'{x:.4f}' for x in v)} |"
            )
    L += [
        "",
        "*Auto-generated. Falsifiers only — no paired CI, no verdict. See `07_experiment_log.md`.*",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="sb")
    a = ap.parse_args()
    data = read_results(a.results, a.target)
    if not data:
        raise SystemExit(f"no score_<seed>_<arm>.csv files in {a.results}")
    with open(a.out, "w") as fh:
        fh.write(render(data))
    print(f"wrote {a.out} ({len(data)} seed(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
