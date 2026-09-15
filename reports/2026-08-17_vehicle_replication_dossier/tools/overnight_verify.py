#!/usr/bin/env python3
"""overnight_verify.py — assemble MORNING_REPORT.md and check the pre-registered falsifiers.

Runs after every arm and from the driver's EXIT trap, so the report always exists and always
describes a real state. It must therefore **never raise**: a crash here would replace a partial
result with no result at all. Every check is wrapped, and anything unexpected becomes an AMBER
line rather than a traceback.

Checks F4/F5/F6 from `05_analysis_plan.md` (LOCKED). It does NOT read P1-P4 — predictions are for
a human to read after the falsifiers have been recorded.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

DOS = Path(__file__).resolve().parents[1]
RES = DOS / "results"
OVN = RES / "overnight"
MODEL = "violet_visitor"
ARMS = [
    "use_real",
    "spatial_scramble",
    "occurrence_real_magnitude_model",
    "occurrence_model_magnitude_real",
    "thin_0.75",
]
REF_N = 170430  # 13110 cells x 13 origins


def _rows(path: Path, target: str = "sb") -> dict[int, dict]:
    try:
        return {int(r["h"]): r for r in csv.DictReader(open(path)) if r["target"] == target}
    except Exception:
        return {}


def _fed(label: str) -> dict[str, float] | None:
    """Mean fed-field statistics for target 0, or None when the arm has no record."""
    p = RES / f"fedfield_{MODEL}_{label}.csv"
    if not p.exists():
        return None
    try:
        rows = [r for r in csv.DictReader(open(p)) if r["target_idx"] == "0"]
        if not rows:
            return None
        out = {"n_rows": float(len(rows))}
        for k in ("active_fraction", "neighbour_pairs_per_active", "mean_magnitude_on_active"):
            vals = [float(r[k]) for r in rows if float(r.get("n_active", 1)) > 0]
            out[k] = statistics.mean(vals) if vals else float("nan")
        return out
    except Exception:
        return None


def main() -> int:
    lines: list[str] = []
    problems: list[str] = []
    ambers: list[str] = []

    # AMENDMENT 1 (2026-08-17 02:58): the control is `identity` run TONIGHT, not the preserved
    # 2026-08-12 cubes. Three commits touching the inference path landed after those cubes were
    # written — notably a2eabeb, per-site LockedDropout, which changes MC-dropout masks, and this
    # vehicle evaluates with dropout live (`evaluation_mode: stochastic`, `dropout_rate: 0.15`).
    # Scoring treatment arms against a pre-change control would confound each transform's effect
    # with a dropout change. The preserved cubes are still reported, labelled, as the 2026-08-12
    # code path — the difference between them and `identity` measures what those commits did.
    preserved = _rows(RES / f"score_{MODEL}_production_reference.csv")
    identity = _rows(RES / f"score_{MODEL}_identity.csv")
    ref = identity or preserved
    control_is = (
        "identity (tonight's code)" if identity else "PRESERVED 2026-08-12 cubes — PROVISIONAL"
    )
    scores = {a: _rows(RES / f"score_{MODEL}_{a}.csv") for a in ARMS}
    done = {a: (RES / f"{MODEL}_{a}_DONE").exists() for a in ARMS}
    complete = [a for a in ARMS if done[a] and scores[a]]
    if not identity:
        ambers.append(
            "control is still the PRESERVED 2026-08-12 cubes; the `identity` arm (chained after "
            "the batch) had not finished. Shares below are provisional — see AMENDMENT 1."
        )

    # ---------------- state ----------------
    lines.append("## Run state\n")
    lines.append(f"- **control in use: {control_is}**")
    lines.append(
        f"- preserved 2026-08-12 cubes: {'scored' if preserved else '**MISSING**'} "
        "(F1 reference; reproduced rescore.csv bit-for-bit)"
    )
    lines.append(f"- `identity` (tonight's code): {'scored' if identity else 'pending'}")
    for a in ARMS:
        mark = (
            "done"
            if done[a]
            else ("**FAILED**" if (OVN / f"FAIL_{a}.txt").exists() else "pending")
        )
        lines.append(f"- `{a}`: {mark}")
    if not ref:
        problems.append("the production reference score is missing — nothing is comparable")

    # ---------------- F5: identical support ----------------
    lines.append("\n## F5 — support (`N`) identical across arms\n")
    bad_n = []
    for a in complete:
        for h, r in scores[a].items():
            if int(r["N"]) != REF_N:
                bad_n.append(f"{a} h{h} N={r['N']}")
    if bad_n:
        problems.append("F5 FIRED — arms scored on different supports: " + "; ".join(bad_n[:6]))
        lines.append(f"**FIRED** — {len(bad_n)} row(s) off {REF_N}: {bad_n[:6]}")
    else:
        lines.append(
            f"pass — every scored row has N={REF_N}" if complete else "no arms scored yet"
        )

    # ---------------- F4: h=1 identical across arms ----------------
    lines.append("\n## F4 — h=1 identical across arms (step 1 has no feedback)\n")
    if ref and complete:
        base = {k: float(ref[1][k]) for k in ("AP", "Brier", "crps_all")}
        lines.append("| arm | AP@h1 | dAP vs control |")
        lines.append("|---|--:|--:|")
        lines.append(f"| control | {base['AP']:.9f} | — |")
        worst = 0.0
        for a in complete:
            if 1 not in scores[a]:
                continue
            v = float(scores[a][1]["AP"])
            d = abs(v - base["AP"])
            worst = max(worst, d)
            lines.append(f"| `{a}` | {v:.9f} | {d:.2e} |")
        if worst > 1e-6:
            problems.append(
                f"F4 FIRED — h=1 AP differs across arms by {worst:.2e} (> 1e-6). Step 1 has no "
                "feedback, so something other than the feedback path moved."
            )
            lines.append(f"\n**FIRED** — worst |dAP@h1| = {worst:.2e}")
        else:
            lines.append(f"\npass — worst |dAP@h1| = {worst:.2e}")
    else:
        lines.append("not evaluable yet")

    # ---------------- F6: the transforms bit on THIS vehicle ----------------
    lines.append("\n## F6 — arm separation on the real field\n")
    fr = _fed("use_real")
    if fr is None:
        lines.append("`use_real` fed-field record absent — cannot check separation")
    else:
        lines.append("| relation | expected | observed | |")
        lines.append("|---|---|---|---|")

        def rel(name, got, exp, ok):
            lines.append(f"| {name} | {exp} | {got} | {'ok' if ok else '**FAIL**'} |")
            if not ok:
                problems.append(f"F6 FIRED — {name}: {got} vs expected {exp}")

        f_ss = _fed("spatial_scramble")
        if f_ss:
            d = abs(f_ss["active_fraction"] - fr["active_fraction"])
            rel("af(spatial_scramble) == af(use_real)", f"{d:.2e}", "< 1e-6", d < 1e-6)
            r = f_ss["neighbour_pairs_per_active"] / max(fr["neighbour_pairs_per_active"], 1e-12)
            rel("clustering destroyed", f"{r:.3f}", "< 0.5", r < 0.5)
        f_e4 = _fed("occurrence_real_magnitude_model")
        if f_e4:
            d = abs(f_e4["active_fraction"] - fr["active_fraction"])
            rel("af(occ_real_mag_model) == af(use_real)", f"{d:.2e}", "< 1e-6", d < 1e-6)
            m = abs(f_e4["mean_magnitude_on_active"] - fr["mean_magnitude_on_active"]) / max(
                fr["mean_magnitude_on_active"], 1e-12
            )
            rel("magnitudes swapped", f"{m:.1%}", "> 5%", m > 0.05)
        f_th = _fed("thin_0.75")
        if f_th:
            exp = 0.25 * fr["active_fraction"]
            r = abs(f_th["active_fraction"] - exp) / max(exp, 1e-12)
            rel("af(thin:0.75) == 0.25 x af(use_real)", f"{r:.1%}", "within 5%", r < 0.05)

    # ---------------- the comparison ----------------
    lines.append("\n## The comparison — gate AP, target sb\n")
    if ref:
        hs = sorted(ref)
        lines.append("| h | control | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
        lines.append("|--:|" + "--:|" * (len(ARMS) + 1))
        for h in hs:
            cells = [f"{float(ref[h]['AP']):.4f}"]
            for a in ARMS:
                got = scores.get(a, {})
                cells.append(f"{float(got[h]['AP']):.4f}" if h in got else "-")
            lines.append(f"| {h} | " + " | ".join(cells) + " |")

        # F3 + the decomposition, only once the oracle and control both exist
        if "use_real" in complete and 18 in scores["use_real"]:
            o = float(scores["use_real"][18]["AP"])
            c = float(ref[18]["AP"])
            gap = o - c
            lines.append(
                f"\n**Oracle-control gap at h18 = {gap:.4f}** "
                f"(oracle {o:.4f}, control {c:.4f}); on `truncated_smoke` it was 0.2938."
            )
            if gap < 0.05:
                problems.append(
                    f"F3 FIRED — oracle-control gap at h18 is {gap:.4f} < 0.05 AP. Nothing to "
                    "decompose on this vehicle; the E4 shares are noise and must NOT be quoted."
                )
                lines.append("\n**F3 FIRED — do not quote shares.**")
            else:
                lines.append("\n| component | share of the gap | truncated_smoke |")
                lines.append("|---|--:|--:|")
                smoke = {
                    "occurrence_real_magnitude_model": "88.6%",
                    "occurrence_model_magnitude_real": "7.9%",
                    "spatial_scramble": "0.9%",
                }
                for a, s in smoke.items():
                    if a in complete and 18 in scores[a]:
                        sh = (float(scores[a][18]["AP"]) - c) / gap
                        lines.append(f"| `{a}` | {sh:6.1%} | {s} |")

    # ---------------- byproduct: what the post-2026-08-12 commits did ----------------
    if identity and preserved:
        lines.append("\n## Byproduct — `identity` (tonight) vs preserved 2026-08-12 cubes\n")
        lines.append("Same artifact, same seed, same origins; different code. The gap below IS the")
        lines.append("effect of d3a2626 / c07a352 / a2eabeb (per-site LockedDropout) on the")
        lines.append("free-running path.\n")
        lines.append("| h | identity (today) | preserved (08-12) | diff |")
        lines.append("|--:|--:|--:|--:|")
        for h in sorted(identity):
            if h in preserved:
                a, b = float(identity[h]["AP"]), float(preserved[h]["AP"])
                lines.append(f"| {h} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")

    # ---------------- verdict ----------------
    if (OVN / "ANOMALIES.txt").exists():
        ambers.append("ANOMALIES.txt exists — read it")
    if not (OVN / "RUN_COMPLETE").exists():
        ambers.append("RUN_COMPLETE absent — the run did not finish cleanly")
    if len(complete) < len(ARMS):
        ambers.append(f"{len(ARMS) - len(complete)} arm(s) did not complete")

    verdict = "RED" if problems else ("AMBER" if ambers else "GREEN")
    head = [
        f"# Morning report — vehicle replication on `{MODEL}`",
        "",
        f"**VERDICT: {verdict}**",
        "",
    ]
    if problems:
        head += (
            ["## Falsifiers fired / blocking problems", ""] + [f"- {p}" for p in problems] + [""]
        )
    if ambers:
        head += ["## Warnings", ""] + [f"- {a}" for a in ambers] + [""]
    head += [
        "Pre-registration: `05_analysis_plan.md` (LOCKED before the run). Falsifiers above "
        "are recorded before predictions P1-P4 are read.",
        "",
    ]

    (OVN / "MORNING_REPORT.md").write_text("\n".join(head + lines) + "\n")
    print(f"verify: {verdict} ({len(complete)}/{len(ARMS)} arms, {len(problems)} problem(s))")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # never let the reporter be the reason there is no report
        try:
            (OVN / "MORNING_REPORT.md").write_text(
                f"# Morning report\n\n**VERDICT: RED**\n\nthe verifier itself crashed: {exc!r}\n"
            )
        except Exception:
            pass
        print(f"verify crashed: {exc!r}")
        sys.exit(0)
