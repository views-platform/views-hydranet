#!/usr/bin/env python3
"""verify_sweep.py — check the harness invariants, then emit the four-state verdict.

Order matters: **invariants first, verdict second.** If two arms scored the same cube, or the arms
sit on different supports, the numbers are not a result and no test on them means anything. Take 1
(2026-08-14) read four numbers off a grid that could not discriminate; the point of this file is that
such a reading is not reachable without first passing checks that would have failed there.

Never raises — a crash here would replace a partial result with none. Anything unexpected becomes a
VOID with the reason attached.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

D = Path(__file__).resolve().parents[1]
RES = D / "results"
H_STAR = 18
THETA = 0.30
REF_N = 170430


def _arms() -> dict[str, dict]:
    out = {}
    for p in sorted(RES.glob("arm_*.json")):
        try:
            meta = json.loads(p.read_text())
        except Exception:
            continue
        label = meta.get("label")
        sc = RES / f"score_{label}.csv"
        if not sc.exists():
            continue
        rows = {int(r["h"]): r for r in csv.DictReader(open(sc)) if r["target"] == "sb"}
        if H_STAR in rows and 1 in rows:
            meta["rows"] = rows
            out[label] = meta
    return out


def _perm_p(treated: list[float], control: list[float]) -> float:
    """Exact one-sided permutation p for 'treated is lower', on seed-level values."""
    obs = sum(treated) / len(treated) - sum(control) / len(control)
    pool = treated + control
    k, n = len(treated), len(pool)
    hits = tot = 0
    for idx in combinations(range(n), k):
        t = [pool[i] for i in idx]
        c = [pool[i] for i in range(n) if i not in idx]
        if sum(t) / k - sum(c) / (n - k) <= obs:
            hits += 1
        tot += 1
    return hits / tot


def main() -> int:
    arms = _arms()
    problems: list[str] = []
    lines: list[str] = []

    lines.append(f"Arms with a score CSV: {len(arms)}")
    for lab, m in sorted(arms.items()):
        lines.append(
            f"  {lab:26s} eps={m['ss_epsilon_max']} seed={m['torch_seed']} "
            f"AP@h{H_STAR}={float(m['rows'][H_STAR]['AP']):.4f}"
        )

    # ---- invariants, before any verdict -----------------------------------------------------
    for lab, m in arms.items():
        for h, r in m["rows"].items():
            if int(r["N"]) != REF_N:
                problems.append(f"F3: {lab} h{h} N={r['N']} != {REF_N}")
    heads = {m.get("head") for m in arms.values()}
    if len(heads) > 1:
        problems.append(f"F6: arms span {len(heads)} repo HEADs — not comparable")
    hashes = [(lab, m.get("weight_sha256")) for lab, m in arms.items()]
    for (a, ha), (b, hb) in combinations(hashes, 2):
        if ha and hb and ha == hb:
            problems.append(f"F5: {a} and {b} share a weight hash — the same model twice")
    for (a, ma), (b, mb) in combinations(sorted(arms.items()), 2):
        for h in sorted(set(ma["rows"]) & set(mb["rows"])):
            if abs(float(ma["rows"][h]["AP"]) - float(mb["rows"][h]["AP"])) < 1e-12:
                problems.append(f"F4: {a} and {b} share an identical AP at h{h} — one cube scored twice")

    ctrl = {k: v for k, v in arms.items() if float(v["ss_epsilon_max"]) == 0.0}
    trt = {k: v for k, v in arms.items() if float(v["ss_epsilon_max"]) > 0.0}

    # ---- post-hoc floor gate on this sweep's OWN controls ------------------------------------
    sys.path.insert(0, str(D.parents[1]))
    try:
        from scripts.floor_gate import floor_gate

        for lab, m in ctrl.items():
            r = m["rows"][H_STAR]
            g = floor_gate(
                ap_control=float(r["AP"]),
                n_cells=int(r["N"]),
                n_event=int(r["n_event"]),
                horizon=H_STAR,
                target="sb",
            )
            if g["clauses"]["FG-A"]["verdict"] != "PASS":
                problems.append(f"post-hoc floor gate: control {lab} fails FG-A")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"post-hoc floor gate could not run: {exc!r}")

    # ---- the verdict -------------------------------------------------------------------------
    verdict, detail = "VOID", ""
    if problems:
        detail = "harness invariants failed — the numbers are not a result"
    elif len(ctrl) < 3 or len(trt) < 3:
        verdict, detail = (
            "UNDERPOWERED",
            f"{len(ctrl)} control and {len(trt)} treated arms; the exact test cannot reach p<=0.05 "
            "with fewer than 3 per side",
        )
    else:
        c = [float(m["rows"][H_STAR]["AP"]) for m in ctrl.values()]
        t = [float(m["rows"][H_STAR]["AP"]) for m in trt.values()]
        cr = [float(m["rows"][H_STAR]["AP"]) / float(m["rows"][1]["AP"]) for m in ctrl.values()]
        tr = [float(m["rows"][H_STAR]["AP"]) / float(m["rows"][1]["AP"]) for m in trt.values()]
        p = _perm_p(t, c)
        diff = sum(t) / len(t) - sum(c) / len(c)
        rdiff = sum(tr) / len(tr) - sum(cr) / len(cr)
        agree = (diff < 0) == (rdiff < 0)
        mde = None
        cif = RES / f"ap_ci_{next(iter(ctrl))}.json"
        if cif.exists():
            mde = json.loads(cif.read_text())[str(H_STAR)]["mde"]
        lines += [
            "",
            f"control AP@h{H_STAR}: {[round(x, 4) for x in c]}",
            f"treated AP@h{H_STAR}: {[round(x, 4) for x in t]}",
            f"mean difference: {diff:+.4f}   retention difference: {rdiff:+.4f}   "
            f"endpoints agree: {agree}",
            f"exact one-sided permutation p = {p:.4f}",
            f"MDE_AP(h{H_STAR}) = {mde:.4f}" if mde else "MDE unavailable",
        ]
        theta_abs = THETA * (sum(c) / len(c))
        if p <= 0.05 and mde and abs(diff) >= 3 * mde and agree:
            verdict, detail = "EFFECT", f"p={p:.4f}, |diff|={abs(diff):.4f} >= 3*MDE={3 * mde:.4f}"
        elif p > 0.05 and mde and abs(diff) + 3 * mde < theta_abs:
            verdict, detail = (
                "NULL",
                f"p={p:.4f} and the interval excludes a {THETA:.0%} effect ({theta_abs:.4f})",
            )
        else:
            verdict, detail = (
                "UNDERPOWERED",
                f"p={p:.4f} but the interval does not exclude a {THETA:.0%} effect "
                f"({theta_abs:.4f}) — cannot distinguish 'no effect' from 'could not tell'",
            )

    head = [f"# {verdict}", "", detail, ""]
    if problems:
        head += ["## Blocking", ""] + [f"- {x}" for x in problems] + [""]
    head += [
        "Pre-registration: `05_analysis_plan.md` (LOCKED). Invariants are checked before the verdict; "
        "a null is declared only when the interval excludes the pre-registered effect.",
        "",
        "⚠️ Per §3.1 this cannot settle the roster observation — those models trained with "
        "`ss_feedback='mean'`, which C-259 forbids.",
        "",
    ]
    (RES / "VERDICT.md").write_text("\n".join(head + lines) + "\n")
    print(f"verify_sweep: {verdict} ({len(arms)} arms, {len(problems)} problem(s))")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        try:
            (RES / "VERDICT.md").write_text(f"# VOID\n\nverify_sweep itself crashed: {exc!r}\n")
        except Exception:
            pass
        print(f"verify_sweep crashed: {exc!r}")
        sys.exit(0)
