#!/usr/bin/env python3
"""coupling_check.py — Check B (#291): are one-step skill and rollout robustness coupled?

Criterion is in `05_analysis_plan.md`, committed BEFORE this ran (`6ec3c3c`, 22:14:04):

    decoupled (CLOSE #291) iff there is a lesson step with
        dT0 > 2 * sigma_T0   AND   dRetention < 1 * sigma_retention

Zhuang et al. 2025 state the Horizon Forcing premise with its own scope condition — *"in chaotic
systems controlling long-term error necessitates controlling short-term error"*. Conflict counts
are not chaotic in the Lyapunov sense, and M27 is the registered falsifier.

**The robustness requirement from 05 §Check B is mandatory here**: M26/M27 quote L=300 from ONE
seed, but four eps=0 seeds now exist. The check runs on both the single-seed series (what the
ledger claims) and the multi-seed L=300 mean. If they disagree on the verdict, neither closes.

Sigmas are the L=160 seed spreads (n=6), the only lesson count with enough arms to measure noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_HN = Path(__file__).resolve().parents[3]
_SS = _HN / "reports" / "2026-08-17_ss_retention_dossier" / "results"
_LC = _HN / "reports" / "2026-08-18_lesson_curve_dossier" / "results"

#: the six L=160 eps=0 arms — the only lesson count with enough seeds to measure seed noise
SIGMA_ARMS = [
    _SS / "score_longzero_fortytwo.csv",
    _SS / "score_longzero_fortythree.csv",
    _SS / "score_longzero_fortyfour.csv",
    _SS / "score_longzero_fortyfive.csv",
    _SS / "score_longzero_fortysix.csv",
    _LC / "score_longzero_fortyseven.csv",
]
#: single-seed series as the ledger's M26/M27 present it (seed 42 throughout above 160)
SINGLE = {
    160: _SS / "score_longzero_fortytwo.csv",
    300: _LC / "score_fullzero_fortytwo.csv",
    600: _LC / "score_sixhundredzero_fortytwo.csv",
}
#: every L=300 eps=0 arm, for the multi-seed mean the criterion requires
L300_ALL = [
    _LC / "score_fullzero_fortytwo.csv",
    _SS / "score_fullzero_fortythree.csv",
    _SS / "score_fullzero_fortyfour.csv",
    _SS / "score_fullzero_fortyfive.csv",
]


def h1_h18(path: Path) -> tuple[float, float]:
    with open(path) as fh:
        rows = {int(r["h"]): float(r["AP"]) for r in csv.DictReader(fh) if r["target"] == "sb"}
    return rows[1], rows[18]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0s, rets = [], []
    for p in SIGMA_ARMS:
        c, f = h1_h18(p)
        t0s.append(c)
        rets.append(f / c)
    sigma_t0 = statistics.stdev(t0s)
    sigma_ret = statistics.stdev(rets)

    single = {}
    for lessons, p in SINGLE.items():
        c, f = h1_h18(p)
        single[lessons] = {"t0": c, "retention": f / c}

    multi = []
    for p in L300_ALL:
        c, f = h1_h18(p)
        multi.append({"t0": c, "retention": f / c})
    l300_multi = {
        "t0": statistics.mean(x["t0"] for x in multi),
        "retention": statistics.mean(x["retention"] for x in multi),
        "retention_sd": statistics.stdev(x["retention"] for x in multi),
        "n": len(multi),
    }

    print(f"sigma_T0 (L=160, n={len(t0s)})        = {sigma_t0:.4f}")
    print(f"sigma_retention (L=160, n={len(rets)}) = {sigma_ret:.4f}")
    print()
    print("--- M27 reproduction from the CSVs (05 falsifier 3) ---")
    d_t0 = single[600]["t0"] - single[300]["t0"]
    d_ret = single[600]["retention"] - single[300]["retention"]
    print(f"  T=0 300->600      = {d_t0:+.5f}   (ledger says +0.0213)")
    print(f"  sigma_T0          = {sigma_t0:.4f}    (ledger says 0.0077)")
    print(f"  retention 300->600= {d_ret:+.5f}   (ledger says +0.0014)")
    repro = (
        abs(d_t0 - 0.0213) < 5e-4 and abs(sigma_t0 - 0.0077) < 5e-4 and abs(d_ret - 0.0014) < 5e-4
    )
    print(f"  M27 REPRODUCES: {'YES' if repro else 'NO — do not read further'}")
    print()

    rows = []
    for series, l300 in (
        ("single-seed (ledger)", single[300]),
        (f"multi-seed n={l300_multi['n']}", l300_multi),
    ):
        dt = single[600]["t0"] - l300["t0"]
        dr = single[600]["retention"] - l300["retention"]
        dec = (dt > 2 * sigma_t0) and (dr < 1 * sigma_ret)
        rows.append(
            {
                "series": series,
                "d_t0": dt,
                "d_ret": dr,
                "d_t0_sigmas": dt / sigma_t0,
                "d_ret_sigmas": dr / sigma_ret,
                "decoupled": dec,
            }
        )
        print(
            f"{series:<22} 300->600:  dT0 = {dt:+.5f} ({dt / sigma_t0:.2f}sigma)   "
            f"dRet = {dr:+.5f} ({dr / sigma_ret:.2f}sigma)   -> {'DECOUPLED' if dec else 'coupled'}"
        )

    agree = len({r["decoupled"] for r in rows}) == 1
    verdict = (
        "CLOSE #291 (decoupled)"
        if agree and rows[0]["decoupled"]
        else "KEEP #291 OPEN"
        if agree
        else "DISAGREE — neither closes"
    )
    print()
    print("criterion: dT0 > 2*sigma AND dRet < 1*sigma  =>  decoupled => CLOSE")
    print(f"VERDICT: {verdict}")

    Path(a.out).write_text(
        json.dumps(
            {
                "sigma_t0": sigma_t0,
                "sigma_retention": sigma_ret,
                "m27_reproduces": repro,
                "single": single,
                "l300_multi": l300_multi,
                "steps": rows,
                "series_agree": agree,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
