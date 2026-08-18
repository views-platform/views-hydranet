#!/usr/bin/env python3
"""verify_curve.py — read the arms, apply the locked rule, render the verdict.

This file is deliberately thin. The decision rule lives in `scripts/lesson_curve_gate.py`,
tracked and unit-tested, because a tracked test may not runtime-load the gitignored `reports/`
tree — a rule that lived only here would be a rule with no test in CI. What remains is reading
and rendering, and `curve_verdict` runs its falsifiers before its rule, so an invariant failure
can never be read past.

Never raises. A crash would replace a partial result with none, so anything unexpected becomes a
VOID with the reason attached.

Where the data lives, and why it is split:
  * L=160 (seeds 42-45) — `2026-08-17_ss_retention_dossier/results`, written by that dossier's
    unmodified `run_arm.sh`. Those three seed arms are arms 1-3 of its parked sweep and are shared,
    not duplicated (SCOPE.md).
  * L>=300 controls and every oracle — this dossier's `results`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
RES = D / "results"
SS_RES = D.parents[0] / "2026-08-17_ss_retention_dossier" / "results"

sys.path.insert(0, str(D.parents[1]))
from scripts.lesson_curve_gate import (  # noqa: E402
    ANCHOR_L,
    H_BASE,
    H_STAR,
    K_PRED,
    THETA,
    curve_verdict,
)


def _sb_rows(path: Path) -> dict[int, dict]:
    return {int(r["h"]): r for r in csv.DictReader(open(path)) if r["target"] == "sb"}


def _read_arms() -> tuple[list[dict], list[str]]:
    """One record per arm that has both a config record and a score CSV."""
    notes: list[str] = []
    arms: list[dict] = []
    for res_dir in (SS_RES, RES):
        if not res_dir.is_dir():
            continue
        for meta_path in sorted(res_dir.glob("arm_*.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:  # noqa: BLE001
                notes.append(f"unreadable {meta_path.name}: {exc!r}")
                continue
            label = meta.get("label")
            score = res_dir / f"score_{label}.csv"
            if not label or not score.exists():
                continue
            rows = _sb_rows(score)
            if H_STAR not in rows or H_BASE not in rows:
                notes.append(f"{label}: score CSV lacks h{H_BASE} or h{H_STAR}")
                continue

            ci = None
            for d in (SS_RES, RES):
                p = d / f"ap_ci_{label}.json"
                if p.exists():
                    ci = json.loads(p.read_text())
                    break

            # the oracle is always this dossier's product, whichever dossier holds the control
            oracle_path = RES / f"score_{label}_use_real.csv"
            oracle = _sb_rows(oracle_path) if oracle_path.exists() else None

            arms.append(
                {
                    "label": label,
                    "total_lessons": int(meta["total_lessons"]),
                    "torch_seed": int(meta["torch_seed"]),
                    "ss_epsilon_max": float(meta.get("ss_epsilon_max", 0.0)),
                    "ap_h1": float(rows[H_BASE]["AP"]),
                    "ap_h18": float(rows[H_STAR]["AP"]),
                    "oracle_h1": float(oracle[H_BASE]["AP"]) if oracle else None,
                    "oracle_h18": float(oracle[H_STAR]["AP"]) if oracle else None,
                    "n_cells": int(rows[H_STAR]["N"]),
                    "n_event": int(rows[H_STAR]["n_event"]),
                    "mde_f": float(ci[str(H_STAR)]["mde"]) if ci else None,
                    "head": meta.get("head"),
                    "weight_sha256": meta.get("weight_sha256"),
                    "source": res_dir.parent.name.replace("2026-08-", "").replace("_dossier", ""),
                    "retention_ci": _retention_ci(label),
                }
            )
    return arms, notes


def _retention_ci(label: str) -> dict | None:
    for d in (SS_RES, RES):
        p = d / f"ret_ci_{label}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                return None
    return None


def _render(arms: list[dict], v: dict, notes: list[str]) -> str:
    lines = [f"# {v['state']}", "", v["detail"], ""]
    if v["problems"] or notes:
        lines += ["## Blocking", ""] + [f"- {x}" for x in v["problems"] + notes] + [""]
    lines += [
        f"Pre-registration: `05_analysis_plan.md` (LOCKED), rule md5 `{v['rule_md5']}`. "
        "Invariants "
        "run before the verdict; a null is declared only when the prediction bound excludes the "
        f"pre-registered effect theta = {THETA}.",
        "",
        "⚠️ One seed per lesson point above 160. Per the standing rule a positive here is an "
        "escalation trigger, not a conclusion.",
        "",
        "| arm | L | seed | source | C=AP h1 | F=AP h18 | O=oracle h18 | R=F/C | ±MDE(R) | FG-A |",
        "|---|--:|--:|---|--:|--:|--:|--:|--:|---|",
    ]
    for a in sorted(arms, key=lambda x: (x["total_lessons"], x["torch_seed"])):
        o = f"{a['O']:.4f}" if a.get("O") is not None else "—"
        rci = a.get("retention_ci")
        rm = f"{rci['mde']:.4f}" if rci and "mde" in rci else "—"
        lines.append(
            f"| `{a['label']}` | {a['total_lessons']} | {a['torch_seed']} | {a['source']} | "
            f"{a['C']:.4f} | {a['F']:.4f} | {o} | {a['R']:.4f} | {rm} | {a.get('fg_a', '?')} |"
        )
    lines.append("")

    if v["decomposition"]:
        lines += [
            f"**Decomposition** `log F = log C + log R`, against L={ANCHOR_L} seed "
            f"{sorted(a['torch_seed'] for a in arms if a['total_lessons'] == ANCHOR_L)[0]}:",
            "",
            "| L | dlog F | from the ceiling | from retention | oracle O(L) |",
            "|--:|--:|--:|--:|--:|",
        ]
        for d in v["decomposition"]:
            o = f"{d['oracle_h18']:.4f}" if d["oracle_h18"] is not None else "not measured"
            lines.append(
                f"| {d['total_lessons']} | {d['dlog_F']:+.4f} | {d['dlog_C']:+.4f} | "
                f"{d['dlog_R']:+.4f} | {o} |"
            )
        lines.append("")

    if v["sigma_seed_r"] is not None:
        lines += [
            f"**sigma_seed at L={ANCHOR_L}** (n={len(v['anchor'])}): retention "
            f"{v['sigma_seed_r']:.4f}, AP@h{H_STAR} {v['sigma_seed_f']:.4f}",
            "",
            f"- mean R {v['mean_r']:.4f} → prediction bound **{v['bound_r']:.4f}** "
            f"(mean + {K_PRED} x sigma)",
            f"- mean F {v['mean_f']:.4f} → prediction bound **{v['bound_f']:.4f}**",
            f"- k x sigma_seed(R) = **{K_PRED * v['sigma_seed_r']:.4f}** against theta = {THETA}"
            + (
                "  ← a null is declarable"
                if K_PRED * v["sigma_seed_r"] < THETA
                else "  ← too wide for a null; only a large rise is detectable"
            ),
            "- mean MDE_F over the anchor: "
            + (f"{v['mde_f']:.4f}" if v["mde_f"] else "unavailable"),
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    arms, notes = _read_arms()
    v = curve_verdict(arms)
    (RES / "VERDICT.md").write_text(_render(arms, v, notes))
    (RES / "curve_state.json").write_text(
        json.dumps(
            {
                "state": v["state"],
                "rule_md5": v["rule_md5"],
                "n_arms": len(arms),
                "n_anchor": len(v["anchor"]),
                "sigma_seed_r": v["sigma_seed_r"],
                "bound_r": v["bound_r"],
                "longest": v["longest"],
                "problems": v["problems"] + notes,
            },
            indent=2,
        )
        + "\n"
    )
    n_bad = len(v["problems"] + notes)
    print(f"verify_curve: {v['state']} ({len(arms)} arms, {n_bad} problem(s))")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        try:
            (RES / "VERDICT.md").write_text(f"# VOID\n\nverify_curve itself crashed: {exc!r}\n")
        except Exception:
            pass
        print(f"verify_curve crashed: {exc!r}")
        sys.exit(0)
