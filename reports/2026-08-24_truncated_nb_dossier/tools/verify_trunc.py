#!/usr/bin/env python3
"""verify_trunc.py — the §5 decision rule, re-run by the queue after every arm.

Pre-registration: `../05_analysis_plan.md`, LOCKED before this file existed. This implements §5,
§6 and nothing else.

Unlike the ITF pilot's 2v2 screen, this is a **4v4**: the exact one-sided permutation floor is
`1/C(8,4) = 0.0143`, so a significance statement is available and §5's four-state rule applies
(EFFECT / NULL / UNDERPOWERED / VOID) rather than a direction screen.

**The guard the shared scheduler cannot give us.** `run_queue.sh:ensure_arm` proves a reused arm
matches on `total_lessons:torch_seed:ss_epsilon_max:ss_reverse` — `output_distribution` is NOT in
that tuple. On the resume path an `nb` arm sitting at a `trunc*` name would be silently reused and
scored as the treatment. So every arm's config is read and asserted here, before any result is.
Kept local rather than edited into the shared scheduler, which four other dossiers depend on.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

_D = Path(__file__).resolve().parents[1]
_HN = _D.parents[1]
RES = _D / "results"
_MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")

H_STAR = 18
#: §5 magnitude guardrails — reported at every horizon, never summarised away.
GUARD_COLS = ("crps_all", "size_ratio", "mag_on_false_pos", "n_false_pos")
HORIZONS = (1, 6, 12, 18, 24, 30, 36)

#: seed -> (arm label, control label, published control AP@h18, control CSV relative to reports/)
PAIRS = {
    42: ("truncfullzero_fortytwo", "fullzero_fortytwo", 0.3298,
         "2026-08-18_lesson_curve_dossier/results/score_fullzero_fortytwo.csv"),
    43: ("truncfullzero_fortythree", "fullzero_fortythree", 0.3318,
         "2026-08-17_ss_retention_dossier/results/score_fullzero_fortythree.csv"),
    44: ("truncfullzero_fortyfour", "fullzero_fortyfour", 0.3058,
         "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfour.csv"),
    45: ("truncfullzero_fortyfive", "fullzero_fortyfive", 0.3352,
         "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfive.csv"),
}


def _row(path: Path, h: int, target: str = "sb") -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["target"] == target and int(r["h"]) == h:
                return r
    return None


def _hp(model: str) -> dict | None:
    """Resolve an arm's config by executing it — the same readback the builder uses."""
    p = _MODELS / model / "configs" / "config_hyperparameters.py"
    if not p.is_file():
        return None
    text = p.read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(p), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns["get_hp_config"]()


def _permutation_p(treat: list[float], ctrl: list[float]) -> float:
    """Exact one-sided permutation p for 'treatment > control' over the pooled 8 values.

    Paired by seed in the data, but the test is over labellings: with n=4 per side there are
    C(8,4)=70 splits, so the floor is 1/70 = 0.0143. Enumerated, not sampled — at this size a
    Monte-Carlo p would add noise to a quantity that has an exact value.
    """
    pooled = treat + ctrl
    obs = sum(treat) / len(treat) - sum(ctrl) / len(ctrl)
    idx = range(len(pooled))
    n_ge = 0
    total = 0
    for pick in combinations(idx, len(treat)):
        a = [pooled[i] for i in pick]
        b = [pooled[i] for i in idx if i not in pick]
        total += 1
        if sum(a) / len(a) - sum(b) / len(b) >= obs - 1e-12:
            n_ge += 1
    return n_ge / total


def main() -> int:
    rows: list[dict] = []
    problems: list[str] = []

    for seed, (arm, ctrl_name, published, ctrl_rel) in PAIRS.items():
        # ── F1/identity: prove the arm is the treatment BEFORE reading any of its numbers ──
        hp = _hp(arm)
        if hp is None:
            problems.append(f"seed {seed}: arm {arm} has no config — cannot verify identity")
            continue
        if hp.get("output_distribution") != "truncated_nb":
            problems.append(
                f"seed {seed}: {arm} has output_distribution="
                f"{hp.get('output_distribution')!r}, not 'truncated_nb'. The queue's identity "
                "tuple does not cover this key, so a control could be scored as the treatment."
            )
            continue
        ctrl_hp = _hp(ctrl_name)
        if ctrl_hp is not None:
            diff = {k for k in set(hp) | set(ctrl_hp) if hp.get(k) != ctrl_hp.get(k)}
            if diff != {"output_distribution"}:
                problems.append(
                    f"seed {seed}: {arm} differs from {ctrl_name} in {sorted(diff)}, expected "
                    "exactly ['output_distribution'] — not single-variable (F1)."
                )
                continue

        c_row = _row(_HN / "reports" / ctrl_rel, H_STAR)
        if c_row is None:
            problems.append(f"seed {seed}: control CSV missing at {ctrl_rel}")
            continue
        ctrl_ap = float(c_row["AP"])
        if abs(ctrl_ap - published) > 5e-4:
            problems.append(
                f"seed {seed}: control AP@h18 {ctrl_ap:.4f} != published {published:.4f} — the "
                "comparison is not seed-matched"
            )

        a_row = _row(RES / f"score_{arm}.csv", H_STAR)
        a1 = _row(RES / f"score_{arm}.csv", 1)
        c1 = _row(_HN / "reports" / ctrl_rel, 1)
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "control": ctrl_ap,
                "trunc": None if a_row is None else float(a_row["AP"]),
                "delta": None if a_row is None else float(a_row["AP"]) - ctrl_ap,
                "h1_delta": (
                    None if (a1 is None or c1 is None) else float(a1["AP"]) - float(c1["AP"])
                ),
                "floor": (
                    "PASS" if (RES / f"FLOORGATE_{arm}_PASS").exists()
                    else "FAIL" if (RES / f"FLOORGATE_{arm}_FAIL").exists()
                    else "unknown"
                ),
            }
        )

    done = [r for r in rows if r["trunc"] is not None]
    did_not_train = [r for r in done if r["floor"] == "FAIL"]

    if problems:
        state, detail = "VOID", "; ".join(problems)
    elif did_not_train:
        state = "DID NOT TRAIN"
        detail = (
            f"{len(did_not_train)} arm(s) failed the floor gate: "
            f"{', '.join(r['arm'] for r in did_not_train)}. F3: an arm that did not train is "
            "reported as such, NEVER as 'truncated_nb is worse' — that is exactly how "
            "truncated_smoke wasted three days (C-299)."
        )
    elif len(done) < len(PAIRS):
        state = f"INCOMPLETE ({len(done)}/{len(PAIRS)})"
        detail = "no verdict until all four arms are scored."
    else:
        treat = [r["trunc"] for r in done]
        ctrl = [r["control"] for r in done]
        p = _permutation_p(treat, ctrl)
        deltas = [r["delta"] for r in done]
        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        mean_d = sum(deltas) / len(deltas)
        # MDE from the per-seed paired bootstrap when present; else unavailable, and §5 says an
        # unevaluable binding clause is never a silent pass.
        mdes = []
        for r in done:
            f = RES / f"ap_ci_{r['arm']}.json"
            if f.exists():
                try:
                    mdes.append(float(json.loads(f.read_text())[str(H_STAR)]["mde"]))
                except Exception:  # noqa: BLE001 - a malformed CI must not crash the verifier
                    pass
        mde = max(mdes) if mdes else None
        if mde is None:
            state, detail = "PROVISIONAL", (
                f"p={p:.4f}, mean ΔAP={mean_d:+.4f}, but no ap_ci_*.json was readable, so the "
                "3×MDE clause of §5 could not be evaluated. Never a silent pass."
            )
        elif p <= 0.05 and abs(mean_d) >= 3 * mde and same_sign:
            state = "EFFECT" if mean_d > 0 else "EFFECT (NEGATIVE)"
            detail = (
                f"p={p:.4f} (floor 0.0143), mean ΔAP={mean_d:+.4f} ≥ 3×MDE={3 * mde:.4f}, all "
                "four seeds agree in sign."
            )
        else:
            state = "NULL / UNDERPOWERED"
            detail = (
                f"p={p:.4f}, mean ΔAP={mean_d:+.4f}, 3×MDE={3 * mde:.4f}, signs agree="
                f"{same_sign}. §5 distinguishes NULL from UNDERPOWERED on whether the CI on the "
                "mean difference excludes a 30% relative effect — read the CI before calling it."
            )

    out = [f"# {state}", "", detail, ""]
    out += [
        "**4v4 — exact one-sided permutation floor `1/C(8,4)` = 0.0143.**",
        "",
        "| seed | arm | control | truncated_nb | Δ AP@h18 | Δ AP@h1 | floor |",
        "|--:|---|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        def f(v, n=4):
            return "—" if v is None else f"{v:.{n}f}"
        out.append(
            f"| {r['seed']} | `{r['arm']}` | {f(r['control'])} | {f(r['trunc'])} | "
            f"{f(r['delta'])} | {f(r['h1_delta'])} | {r['floor']} |"
        )

    # ── §5 magnitude guardrails: an AP win with a crps_all regression is a TRADE, not a win ──
    out += ["", "## Magnitude guardrails (§5) — reported, never summarised away", ""]
    out += ["| seed | h | " + " | ".join(f"Δ {c}" for c in GUARD_COLS) + " |",
            "|--:|--:|" + "--:|" * len(GUARD_COLS)]
    for seed, (arm, _cn, _pub, ctrl_rel) in PAIRS.items():
        for h in HORIZONS:
            a = _row(RES / f"score_{arm}.csv", h)
            c = _row(_HN / "reports" / ctrl_rel, h)
            if a is None or c is None:
                continue
            cells = []
            for col in GUARD_COLS:
                try:
                    cells.append(f"{float(a[col]) - float(c[col]):+.4f}")
                except (KeyError, ValueError):
                    cells.append("—")
            out.append(f"| {seed} | {h} | " + " | ".join(cells) + " |")
    out += [
        "",
        "⚠️ **A gain in AP accompanied by a regression in `crps_all` is a TRADE, not a win** (§5). "
        "The family's author named this risk himself: a truncated body gives the gate's false "
        "positives full magnitude, and `crps_all` is blind to it.",
        "",
        "⚠️ **Registered false-negative mode (§7):** the gate is retrained alongside the "
        "body, so a NULL closes *'swapping the body fixes rollout skill'*, NOT *'the double-zero diagnosis "
        "was wrong'* — M44's decomposition stands on its own measurement either way.",
    ]

    (RES / "VERDICT.md").write_text("\n".join(out) + "\n")
    (RES / "trunc_state.json").write_text(
        json.dumps({"state": state, "detail": detail, "rows": rows}, indent=2)
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - a crashed verifier must stop the queue, not pass it
        print(f"verify_trunc: CRASHED -> VOID: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
