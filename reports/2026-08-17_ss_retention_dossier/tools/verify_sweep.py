#!/usr/bin/env python3
"""verify_sweep.py — read the sweep's arms, apply the locked rule, render the verdict.

Thin by design. The decision rule lives in `scripts/ss_sweep_gate.py`, tracked and unit-tested,
because a tracked test may not runtime-load the gitignored `reports/` tree — a rule that lived only
here would be a rule with no test in CI. What remains is reading and rendering, and `sweep_verdict`
runs its falsifiers before its rule, so an invariant failure can never be read past.

Rewritten 2026-08-21. The previous version was missing the pre-registration's §4 **guard**
(`|ΔAP(h1)| ≤ 3 × MDE_AP(h1)`), so a run in which scheduled sampling wrecked one-step skill would
have been reported as a retention effect. It also took the MDE from one arbitrary control, and
exited 0 on a crash.

Where the arms live, and why it is split:
  * this dossier's `results/` — every arm the sweep runs.
  * `2026-08-18_lesson_curve_dossier/results/` — `fullzero_fortytwo` (L=300, ε=0, seed 42) was
    produced by the lesson curve, is identical in construction to an ε=0 sweep arm, and carries the
    floor-gate licence. It counts as the seed-42 control rather than being retrained.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
RES = D / "results"
CURVE_RES = D.parents[0] / "2026-08-18_lesson_curve_dossier" / "results"
#: the one arm the lesson curve already produced that is a valid ε=0 control here
BORROWED = {"fullzero_fortytwo"}
LESSONS = 300

sys.path.insert(0, str(D.parents[1]))
from scripts.ss_sweep_gate import (  # noqa: E402
    ALPHA,
    GUARD_K,
    H_BASE,
    H_STAR,
    THETA,
    sweep_verdict,
)


def _sb_rows(path: Path) -> dict[int, dict]:
    return {int(r["h"]): r for r in csv.DictReader(open(path)) if r["target"] == "sb"}


def _f(v, nd: int) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _guard_word(v: dict) -> str:
    if v["guard_ok"] is None:
        return "unevaluable"
    return "OK" if v["guard_ok"] else "VIOLATED"


_FP_CACHE: dict[str, str | None] = {}
_CODE_PATHS = (
    "views_hydranet",
    "reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py",
)


def _code_fingerprint(head: str | None) -> str | None:
    """Git tree/blob hashes of the result-producing code — identical iff the code is.

    None, never a sentinel, when a SHA cannot be resolved: a sentinel would be identical for every
    unresolvable arm and would make F6 assert a comparability it never checked.
    """
    if not head:
        return None
    if head not in _FP_CACHE:
        parts = []
        for path in _CODE_PATHS:
            r = subprocess.run(
                ["git", "-C", str(D.parents[1]), "rev-parse", f"{head}:{path}"],
                capture_output=True,
                text=True,
            )
            parts.append(r.stdout.strip() if r.returncode == 0 else None)
        _FP_CACHE[head] = "+".join(parts) if all(parts) else None
    return _FP_CACHE[head]


def _read_arms() -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    other: list[str] = []
    arms: list[dict] = []
    for res_dir in (RES, CURVE_RES):
        if not res_dir.is_dir():
            continue
        for meta_path in sorted(res_dir.glob("arm_*.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:  # noqa: BLE001
                notes.append(f"unreadable {meta_path.name}: {exc!r}")
                continue
            label = meta.get("label")
            if not label:
                notes.append(f"{meta_path.name}: no label field")
                continue
            # the curve dossier holds many arms; take only the one that is a valid control here
            if res_dir is CURVE_RES and label not in BORROWED:
                continue
            if any(a["label"] == label for a in arms):
                continue
            # This sweep is defined at ONE training length. The SS dossier's results also hold the
            # lesson curve's L=160 stage-1 arms; pooling them would confound SS with length. Scope
            # here and NOTE it — the mixed-length falsifier stays as a backstop for anything that
            # slips through by another route.
            if int(meta.get("total_lessons", -1)) != LESSONS:
                other.append(label)
                continue
            score = res_dir / f"score_{label}.csv"
            if not score.exists():
                notes.append(f"{label}: has a config record but no score CSV — arm did not finish")
                continue
            rows = _sb_rows(score)
            if H_STAR not in rows or H_BASE not in rows:
                notes.append(f"{label}: score CSV lacks h{H_BASE} or h{H_STAR}")
                continue

            ci = None
            for d in (res_dir, RES, CURVE_RES):
                p = d / f"ap_ci_{label}.json"
                if p.exists():
                    ci = json.loads(p.read_text())
                    break

            arms.append(
                {
                    "label": label,
                    "total_lessons": int(meta["total_lessons"]),
                    "torch_seed": int(meta["torch_seed"]),
                    "ss_epsilon_max": float(meta.get("ss_epsilon_max", 0.0)),
                    "ap_h1": float(rows[H_BASE]["AP"]),
                    "ap_h18": float(rows[H_STAR]["AP"]),
                    "n_cells": int(rows[H_STAR]["N"]),
                    "n_event": int(rows[H_STAR]["n_event"]),
                    "n_origins": int(float(ci[str(H_STAR)]["n_origins"])) if ci else None,
                    "mde_h1": float(ci[str(H_BASE)]["mde"]) if ci else None,
                    "mde_h18": float(ci[str(H_STAR)]["mde"]) if ci else None,
                    "head": meta.get("head"),
                    "code_fingerprint": _code_fingerprint(meta.get("head")),
                    "weight_sha256": meta.get("weight_sha256"),
                    "borrowed": res_dir is CURVE_RES,
                    "size_ratio": float(rows[H_STAR].get("size_ratio", "nan") or "nan"),
                }
            )
    if other:
        notes.append(
            f"{len(other)} arm(s) at another lesson count ignored — this sweep is "
            f"L={LESSONS}: {', '.join(sorted(other))}"
        )
    return arms, notes


def _render(arms: list[dict], v: dict, notes: list[str]) -> str:
    lines = [f"# {v['state']}", "", v["detail"], ""]
    if v["problems"]:
        lines += ["## Blocking", ""] + [f"- {x}" for x in v["problems"]] + [""]
    if notes:
        lines += ["## Notes (not blocking)", ""] + [f"- {x}" for x in notes] + [""]
    lines += [
        f"Pre-registration: `05_analysis_plan.md` (LOCKED, AMENDMENT 1 → L={LESSONS}), rule md5 "
        f"`{v['rule_md5']}`. Falsifiers run before the verdict. Direction is pre-registered: "
        f"**SS lowers AP@h{H_STAR}**, one-sided, alpha={ALPHA}.",
        "",
        "| arm | eps | seed | L | AP h1 | AP h18 | retention | size_ratio | src |",
        "|---|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for a in sorted(arms, key=lambda x: (x["ss_epsilon_max"], x["torch_seed"])):
        lines.append(
            f"| `{a['label']}` | {a['ss_epsilon_max']} | {a['torch_seed']} | "
            f"{a['total_lessons']} | {a['ap_h1']:.4f} | {a['ap_h18']:.4f} | "
            f"{a['ap_h18'] / a['ap_h1']:.4f} | {_f(a.get('size_ratio'), 4)} | "
            f"{'curve' if a['borrowed'] else 'sweep'} |"
        )
    lines.append("")

    if v["n_control"] and v["n_treated"]:
        lines += [
            f"**{v['n_control']} control vs {v['n_treated']} treated.**",
            "",
            f"- mean AP@h{H_STAR}: control {_f(v['mean_control_h18'], 4)} → treated "
            f"{_f(v['mean_treated_h18'], 4)}  (**{_f(v['diff_h18'], 4)}**)",
            f"- mean retention difference: {_f(v['diff_retention'], 4)}  "
            f"(endpoints agree: {v['endpoints_agree']})",
            f"- exact one-sided permutation **p = {_f(v['p_value'], 4)}**",
            f"- mean MDE_AP(h{H_STAR}) = {_f(v['mde_h18'], 4)}",
            f"- **anchor guard**: mean dAP(h1) = {_f(v['guard_delta_h1'], 4)} against "
            f"{GUARD_K} x MDE_AP(h1) = "
            f"{_f(None if v['mde_h1'] is None else GUARD_K * v['mde_h1'], 4)} → {_guard_word(v)}",
            "",
        ]
        if v["censored"]:
            lines += [
                f"⚠️ **Magnitude CENSORED** for {', '.join(v['censored'])} — below 2 x prevalence, "
                "so the effect size is a lower bound, never a point estimate.",
                "",
            ]
    lines += [
        f"⚠️ Per §3.1 this cannot settle what the roster showed — those models trained with "
        f"`ss_feedback='mean'`, which C-259 forbids. A null here answers the forward-looking "
        f"question only. A NULL requires the interval to exclude theta = {THETA:.0%} of the "
        f"control mean.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    arms, notes = _read_arms()
    v = sweep_verdict(arms)
    (RES / "VERDICT.md").write_text(_render(arms, v, notes))
    (RES / "sweep_state.json").write_text(
        json.dumps(
            {
                "state": v["state"],
                "rule_md5": v["rule_md5"],
                "n_control": v["n_control"],
                "n_treated": v["n_treated"],
                "p_value": v["p_value"],
                "diff_h18": v["diff_h18"],
                "guard_ok": v["guard_ok"],
                "censored": v["censored"],
                "problems": v["problems"] + notes,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"verify_sweep: {v['state']} ({v['n_control']}c/{v['n_treated']}t, "
        f"{len(v['problems'] + notes)} problem(s))"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        # Both files together, and a NON-ZERO exit: a caller must be able to tell a crash from a
        # verdict, and a stale state file is worse than none (lesson-curve audit, finding 3).
        for path, body in (
            (RES / "VERDICT.md", f"# VOID\n\nverify_sweep itself crashed: {exc!r}\n"),
            (
                RES / "sweep_state.json",
                json.dumps({"state": "VOID", "problems": [f"crashed: {exc!r}"]}, indent=2) + "\n",
            ),
        ):
            try:
                path.write_text(body)
            except Exception:  # noqa: BLE001, S110
                pass
        print(f"verify_sweep crashed: {exc!r}")
        sys.exit(1)
