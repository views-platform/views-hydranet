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
import subprocess
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
    THETA,
    curve_verdict,
)


def _sb_rows(path: Path) -> dict[int, dict]:
    return {int(r["h"]): r for r in csv.DictReader(open(path)) if r["target"] == "sb"}


def _f(v, nd: int) -> str:
    """Format a possibly-absent number for a markdown cell."""
    return "—" if v is None else f"{v:.{nd}f}"


_SHIP_COLS = ("crps_all", "crps_events", "size_ratio", "mcr_all", "Brier")


def _ship_cols(row: dict) -> dict:
    """The FAO-02 battery columns, absent ones reported as None rather than raised."""
    out = {}
    for c in _SHIP_COLS:
        try:
            out[c.lower()] = float(row[c])
        except (KeyError, TypeError, ValueError):
            out[c.lower()] = None
    return out


def _read_arms() -> tuple[list[dict], list[str]]:
    """One record per arm that has both a config record and a score CSV."""
    notes: list[str] = []
    treated: list[str] = []
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
            if not label:
                notes.append(f"{meta_path.name}: no label field")
                continue
            # The lesson curve is the eps=0 axis. The SS sweep writes its arms into the SS
            # dossier's results, which this reader also scans, so eps>0 arms appear here. They are
            # a different intervention: SCOPED OUT with a note, never silently dropped and never
            # pooled. The gate's own eps check stays as a backstop for anything that reaches it by
            # another route.
            if float(meta.get("ss_epsilon_max", 0.0)) != 0.0:
                treated.append(label)
                continue
            score = res_dir / f"score_{label}.csv"
            if not score.exists():
                # An arm with a config record but no score is a HALF-COMPLETED arm, not an absent
                # one. Silently dropping it is how a partial run looks like a complete one.
                notes.append(f"{label}: has a config record but no score CSV — arm did not finish")
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
                    # The FAO-02 ship battery is already in every score CSV, carried through so the
                    # curve can say whether training moves MAGNITUDE and not just occurrence. Read
                    # defensively: it is a bonus readout, not a gate, and one missing column must
                    # not turn a valid verdict into "verify_curve itself crashed".
                    "ship": {h: _ship_cols(rows[h]) for h in sorted(rows)},
                    "n_cells": int(rows[H_STAR]["N"]),
                    "n_event": int(rows[H_STAR]["n_event"]),
                    "mde_f": float(ci[str(H_STAR)]["mde"]) if ci else None,
                    "head": meta.get("head"),
                    "code_fingerprint": _code_fingerprint(meta.get("head")),
                    "weight_sha256": meta.get("weight_sha256"),
                    "source": res_dir.parent.name.replace("2026-08-", "").replace("_dossier", ""),
                    "retention_ci": _retention_ci(label),
                }
            )
    if treated:
        notes.append(
            f"{len(treated)} scheduled-sampling arm(s) (eps>0) scoped out — the lesson "
            f"curve is the eps=0 axis: {', '.join(sorted(treated))}"
        )
    return arms, notes


_FP_CACHE: dict[str, str | None] = {}
#: Paths whose content can change a result: the training/inference package, and the scorer.
_CODE_PATHS = (
    "views_hydranet",
    "reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py",
)


def _code_fingerprint(head: str | None) -> str | None:
    """Git tree/blob hashes of the result-producing code at `head` — identical iff the code is.

    F6's question is "were these arms produced by the same code", and a commit id answers a
    different one: four arms built across docs-only commits span two HEADs and one codebase.
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
            # None, never a sentinel string. A sentinel would be IDENTICAL for every arm whose SHA
            # is unreachable (routine after a rebase/squash/gc on this branch), collapsing them all
            # to one apparent "code version" and making F6 assert a comparability it never checked.
            parts.append(r.stdout.strip() if r.returncode == 0 else None)
        _FP_CACHE[head] = "+".join(parts) if all(parts) else None
    return _FP_CACHE[head]


def _retention_ci(label: str) -> dict | None:
    for d in (SS_RES, RES):
        p = d / f"ret_ci_{label}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                return None
    return None


def by_len_seed_ship(arms: list[dict]) -> list[dict]:
    return sorted(arms, key=lambda a: (a["total_lessons"], a["torch_seed"]))


def _render(arms: list[dict], v: dict, notes: list[str]) -> str:
    lines = [f"# {v['state']}", "", v["detail"], ""]
    # Blocking and Notes are different things and must not be merged: a problem forces VOID, a note
    # records something skipped. Printing a note under "Blocking" would misreport a healthy run.
    if v["problems"]:
        lines += ["## Blocking", ""] + [f"- {x}" for x in v["problems"]] + [""]
    if notes:
        lines += ["## Notes (not blocking)", ""] + [f"- {x}" for x in notes] + [""]
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
            "| L | dlog F | from T=0 skill | from retention | the ceiling O(L) |",
            "|--:|--:|--:|--:|--:|",
        ]
        for d in v["decomposition"]:
            o = f"{d['oracle_h18']:.4f}" if d["oracle_h18"] is not None else "not measured"
            lines.append(
                f"| {d['total_lessons']} | {d['dlog_F']:+.4f} | {d['dlog_C']:+.4f} | "
                f"{d['dlog_R']:+.4f} | {o} |"
            )
        lines.append("")

    # ---- the ship panel: FAO-02's primary is CRPS and its guardrails are magnitude ----------
    if any(a.get("ship") for a in arms):
        lines += [
            "### Ship battery (FAO-02) — does training move MAGNITUDE, or only occurrence?",
            "",
            "FAO-02 selects on **CRPS** with a **strict conjunction** of QS99 / Brier / **MCR**"
            " guardrails, on the **validation** partition. These arms are scored on"
            " **calibration**, so nothing here is a ship decision — but `size_ratio` and `MCR`"
            " are the blocker, and they are free to read off arms already produced.",
            "",
            "| arm | L | crps_all h18 | crps_events h18 | size_ratio h18 | MCR h18 | |MCR-1| |",
            "|---|--:|--:|--:|--:|--:|--:|",
        ]
        for a in by_len_seed_ship(arms):
            s = a["ship"].get(H_STAR)
            if not s:
                continue
            lines.append(
                f"| `{a['label']}` | {a['total_lessons']} | {_f(s['crps_all'], 5)} | "
                f"{_f(s['crps_events'], 2)} | {_f(s['size_ratio'], 4)} | {_f(s['mcr_all'], 5)} | "
                f"{_f(None if s['mcr_all'] is None else abs(s['mcr_all'] - 1), 4)} |"
            )
        lines += [
            "",
            "**MCR = mean(prediction) / mean(truth); 1.0 is right-sized.** `size_ratio` is the"
            " median guess on cells that truly had conflict. A model can win `crps_all` on a"
            " 99%-zero field by emitting almost nothing — which is why FAO-02 makes the magnitude"
            " guardrail a hard conjunct and why Epic #263 called that pattern an ARTIFACT.",
            "",
        ]

    if v["sigma_seed_r"] is not None:
        lines += [
            f"**sigma_seed at L={ANCHOR_L}** (n={len(v['anchor'])}): retention "
            f"{v['sigma_seed_r']:.4f}, AP@h{H_STAR} {v['sigma_seed_f']:.4f}",
            "",
            f"- mean R {v['mean_r']:.4f} → prediction bound **{v['bound_r']:.4f}** "
            f"(mean + {v['k_used']:.3f} x sigma, n={len(v['anchor'])})",
            f"- mean F {v['mean_f']:.4f} → prediction bound **{v['bound_f']:.4f}**",
            # k_used, NOT K_PRED. K_PRED is the n=4 reference; the verdict applies _k_for_n(n).
            # Reading the annotation from one and the verdict from the other let VERDICT.md print
            # "too wide for a null" directly beneath a PLATEAU heading.
            f"- k x sigma(R) = **{v['k_used'] * v['sigma_seed_r']:.4f}** vs theta = {THETA}"
            + (
                "  ← a null is declarable"
                if v["k_used"] * v["sigma_seed_r"] < THETA
                else "  ← too wide for a null; only a large rise is detectable"
            ),
            "- mean MDE_F over the anchor: "
            + (f"{v['mde_f']:.4f}" if v["mde_f"] is not None else "unavailable"),
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
    # blocking and non-blocking counted separately — see the note in verify_sweep.py
    print(
        f"verify_curve: {v['state']} ({len(arms)} arms, {len(v['problems'])} blocking, "
        f"{len(notes)} note(s))"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        # Both files must be updated together. Writing only VERDICT.md left curve_state.json
        # holding the PREVIOUS stage's verdict, so a driver reading it would act on a state that
        # no longer exists — e.g. a stale RISING sending it into a 900-lesson arm. The exit code
        # must also be non-zero, or a caller cannot tell a crash from a verdict.
        for path, body in (
            (RES / "VERDICT.md", f"# VOID\n\nverify_curve itself crashed: {exc!r}\n"),
            (
                RES / "curve_state.json",
                json.dumps(
                    {"state": "VOID", "problems": [f"verify_curve crashed: {exc!r}"]}, indent=2
                )
                + "\n",
            ),
        ):
            try:
                path.write_text(body)
            except Exception:  # noqa: BLE001, S110
                pass
        print(f"verify_curve crashed: {exc!r}")
        sys.exit(1)
