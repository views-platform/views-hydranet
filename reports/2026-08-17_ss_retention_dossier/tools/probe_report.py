#!/usr/bin/env python3
"""probe_report.py — render the placement probe, against its pre-registered predictions.

The probe asks ONE question: when scheduled sampling made the rollout worse, did it damage the
model's ability to PLACE events, or its ability to USE a field once placed?

`occurrence_real_magnitude_model` is the discriminator. It hands the model a perfectly-placed
occurrence field and keeps the model's own magnitudes. If the two models converge under that arm,
the damage is in the occurrence field SS taught the model to emit. If the SS model stays behind
even with perfect placement handed to it, SS damaged something else — its use of its input.

Reports the pre-registered predictions verbatim and marks each HOLDS / FAILS, so the reading cannot
drift from what was committed to before the arms ran.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
RES = D / "results"
CURVE_RES = D.parents[0] / "2026-08-18_lesson_curve_dossier" / "results"
H = 18
PAIR = {"fullzero_fortytwo": "eps=0.0", "fullhalf_fortytwo": "eps=0.5"}
ARMS = ["occurrence_real_magnitude_model", "spatial_scramble", "thin_0.75"]


def ap(label: str, suffix: str = "") -> float | None:
    for d in (RES, CURVE_RES):
        p = d / f"score_{label}{suffix}.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if r["target"] == "sb" and int(r["h"]) == H:
                    return float(r["AP"])
    return None


def main() -> int:
    lines = ["# Placement probe — where did scheduled sampling do its damage?", ""]
    lines += [
        "Inference-only, on two frozen artifacts differing in exactly one config key "
        "(`ss_epsilon_max`). Control and oracle already existed and were not re-run.",
        "",
        f"| model | control AP h{H} | oracle (ceiling) | "
        + " | ".join(f"`{a}`" for a in ARMS)
        + " |",
        "|---|--:|--:|" + "--:|" * len(ARMS),
    ]
    data: dict[str, dict] = {}
    for m, tag in PAIR.items():
        ctrl, orac = ap(m), ap(m, "_use_real")
        row = {"ctrl": ctrl, "oracle": orac}
        cells = []
        for a in ARMS:
            v = ap(m, f"_{a}")
            row[a] = v
            cells.append("—" if v is None else f"{v:.4f}")
        data[m] = row
        lines.append(
            f"| `{m}` ({tag}) | {ctrl:.4f} | {orac:.4f} | " + " | ".join(cells) + " |"
        )
    lines.append("")

    z, h = data["fullzero_fortytwo"], data["fullhalf_fortytwo"]
    e4a = "occurrence_real_magnitude_model"

    lines += ["## Pre-registered predictions", ""]
    preds = []
    if z.get(e4a) is not None and h.get(e4a) is not None:
        gap_ctrl = z["ctrl"] - h["ctrl"]
        gap_e4a = z[e4a] - h[e4a]
        closed = 1 - (gap_e4a / gap_ctrl) if gap_ctrl else float("nan")
        preds.append(
            (
                "**P1** Handing both models perfect occurrence closes MOST of the gap between "
                "them (>60%) ⇒ SS's damage is in the occurrence field it emits",
                f"gap {gap_ctrl:+.4f} → {gap_e4a:+.4f}, **{closed:.0%} closed**",
                closed > 0.60,
            )
        )
        preds.append(
            (
                "**P2** If instead the gap SURVIVES perfect occurrence (<30% closed) ⇒ SS damaged "
                "the model's use of its input, not its placement",
                f"{closed:.0%} closed",
                closed < 0.30,
            )
        )
    if z.get("spatial_scramble") is not None and h.get("spatial_scramble") is not None:
        dz = z["spatial_scramble"] - z["ctrl"]
        dh = h["spatial_scramble"] - h["ctrl"]
        preds.append(
            (
                "**P3** `spatial_scramble` falls below BOTH controls — destroying placement is "
                "worse than either model's own output",
                f"eps=0 {dz:+.4f}, eps=0.5 {dh:+.4f}",
                dz < 0 and dh < 0,
            )
        )
    if z.get("thin_0.75") is not None and h.get("thin_0.75") is not None:
        rz = (z["thin_0.75"] - z["ctrl"]) / (z["oracle"] - z["ctrl"])
        rh = (h["thin_0.75"] - h["ctrl"]) / (h["oracle"] - h["ctrl"])
        preds.append(
            (
                "**P4** `thin:0.75` recovers ≥60% of each model's own gap — a quarter of the true "
                "events, correctly placed, is still enough (M4/M15)",
                f"eps=0 {rz:.0%}, eps=0.5 {rh:.0%}",
                rz >= 0.60 and rh >= 0.60,
            )
        )
    if not preds:
        lines.append("_No arm has completed yet._")
    for text, obs, ok in preds:
        lines += [f"- {text}", f"  - observed: {obs} → **{'HOLDS' if ok else 'FAILS'}**"]
    lines += [
        "",
        "⚠️ **One seed, one vehicle, one dose, one target (`sb`), h\\*=18.** `spatial_scramble` "
        "carries C-291's confound: destroying clustering also breaks alignment with the statics. "
        "The share statistic `(arm − control)/(oracle − control)` is meaningless for an arm that "
        "falls OUTSIDE that interval, which `spatial_scramble` does — quote its sign, never its "
        "share.",
        "",
    ]
    (RES / "PROBE.md").write_text("\n".join(lines) + "\n")
    print(f"probe_report: {sum(1 for *_, ok in preds if ok)}/{len(preds)} predictions hold")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        try:
            (RES / "PROBE.md").write_text(f"# VOID\n\nprobe_report crashed: {exc!r}\n")
        except Exception:  # noqa: BLE001, S110
            pass
        print(f"probe_report crashed: {exc!r}")
        sys.exit(1)
