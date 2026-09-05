"""Assemble whatever Wave 1 has produced so far. MUST NEVER RAISE.

Modelled on 2026-08-17_vehicle_replication_dossier/tools/overnight_verify.py: it runs after every
arm and from the launcher's EXIT trap, so it sees partial states by design. Anything unexpected
becomes an AMBER line in the report rather than a traceback that costs the morning its summary.
"""

from __future__ import annotations

import csv
import glob
import os
import traceback
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "MORNING_REPORT.md"
N_ORIGINS = 13
SEEDS = ["fullzero_fortytwo", "fullzero_fortythree", "fullzero_fortyfour", "fullzero_fortyfive"]
ARMS = [
    ("none", "identity"),
    ("hidden", "identity_freezehidden"),
    ("cell", "identity_freezecell"),
    ("all", "identity_freezeall"),
]
# Archived on the pass-0 instrument but from the UNCHANGED cube path, so the re-run must reproduce
# them exactly. If it does not, something other than the dump changed and the wave is suspect.
ANCHOR = {
    ("fullzero_fortytwo", "identity"): 0.3298395823400329,
    ("fullzero_fortytwo", "identity_freezecell"): 0.3621885544392029,
}
HS = [1, 6, 12, 18, 24, 30, 36]


def _ap(path, h, target="sb"):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("target") == target and row.get("h") == str(h):
                return float(row["AP"])
    return None


def _row(path, h, target="sb"):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("target") == target and row.get("h") == str(h):
                return row
    return {}


def main() -> int:
    lines, amber = ["# Wave 1 — morning report", ""], []
    try:
        done = RES / "overnight" / "RUN_COMPLETE"
        lines.append(f"- run state: **{'COMPLETE' if done.exists() else 'IN PROGRESS'}**")
        if done.exists():
            lines.append(f"  - {done.read_text().strip()}")
        hb = RES / "overnight" / "HEARTBEAT"
        if hb.exists():
            lines.append(f"- last heartbeat: {hb.read_text().strip()}")
        ph = RES / "overnight" / "PHASE"
        if ph.exists():
            lines.append(f"- phase: {ph.read_text().strip()}")
        lines.append("")

        lines += [
            "## Arms",
            "",
            "| seed | arm | scored | origins | n_passes | AP@h18 |",
            "|---|---|---|---|---|---|",
        ]
        for m in SEEDS:
            for arm, label in ARMS:
                sc = RES / f"score_{m}_{label}.csv"
                dump = RES / f"bodymean_{m}_{label}"
                npz = sorted(glob.glob(str(dump / "*.npz")))
                passes = "-"
                if npz:
                    try:
                        import numpy as np

                        vals = {int(np.load(f)["n_passes"]) for f in (npz[0], npz[-1])}
                        passes = ",".join(str(v) for v in sorted(vals))
                        if vals != {4}:
                            amber.append(
                                f"{m}/{label}: n_passes={passes}, expected 4 "
                                "(pass-0 instrument mixed in?)"
                            )
                    except Exception as exc:  # noqa: BLE001
                        passes = f"?({type(exc).__name__})"
                ap = _ap(sc, 18) if sc.exists() else None
                mark = "yes" if sc.exists() else "—"
                cnt = f"{len(npz)}/{N_ORIGINS}"
                if npz and len(npz) != N_ORIGINS:
                    amber.append(f"{m}/{label}: {cnt} origins — INCOMPLETE dump")
                lines.append(
                    f"| {m} | {arm} | {mark} | {cnt} | {passes} | "
                    f"{'%.6f' % ap if ap is not None else '—'} |"
                )
        lines.append("")

        lines += ["## Reproduction falsifier (the cube path was NOT changed)", ""]
        for (m, label), want in ANCHOR.items():
            sc = RES / f"score_{m}_{label}.csv"
            if not sc.exists():
                lines.append(f"- {m}/{label}: not yet run")
                continue
            got = _ap(sc, 18)
            ok = got is not None and abs(got - want) < 1e-12
            lines.append(
                f"- {m}/{label} AP@h18 = {got} vs archived {want} — "
                f"**{'EXACT' if ok else 'MISMATCH'}**"
            )
            if not ok:
                amber.append(f"{m}/{label}: AP@h18 does not reproduce the archived value")
        lines.append("")

        lines += ["## Gate and body, by horizon (seed-42 first)", ""]
        for m in SEEDS:
            have = [(a, lb) for a, lb in ARMS if (RES / f"score_{m}_{lb}.csv").exists()]
            if not have:
                continue
            lines += [
                f"### {m}",
                "",
                "| h | "
                + " | ".join(f"AP {a}" for a, _ in have)
                + " | "
                + " | ".join(f"sizeR {a}" for a, _ in have)
                + " |",
                "|---" * (1 + 2 * len(have)) + "|",
            ]
            for h in HS:
                aps, srs = [], []
                for _a, lb in have:
                    r = _row(RES / f"score_{m}_{lb}.csv", h)
                    aps.append(f"{float(r['AP']):.4f}" if r.get("AP") else "—")
                    srs.append(f"{float(r['size_ratio']):.4f}" if r.get("size_ratio") else "—")
                lines.append(f"| {h} | " + " | ".join(aps) + " | " + " | ".join(srs) + " |")
            lines.append("")

        anom = RES / "overnight" / "ANOMALIES.txt"
        if anom.exists() and anom.read_text().strip():
            amber.append("ANOMALIES.txt is non-empty:")
            amber += ["  " + ln for ln in anom.read_text().strip().split("\n")]
        rl = RES / "overnight" / "run.log"
        if rl.exists():
            fails = [
                ln for ln in rl.read_text().split("\n") if "FAILED" in ln or "GUARD FAIL" in ln
            ]
            if fails:
                amber.append("run.log failures:")
                amber += ["  " + f for f in fails[-12:]]
    except Exception:  # noqa: BLE001
        amber.append("verifier hit an unexpected error (report is partial):")
        amber += ["  " + ln for ln in traceback.format_exc().strip().split("\n")[-6:]]

    lines += ["## AMBER", ""] + ([f"- {a}" for a in amber] if amber else ["- none"])
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n")
        print(f"[verify] wrote {OUT} ({len(amber)} amber)")
    except Exception:  # noqa: BLE001
        print("[verify] COULD NOT WRITE REPORT:\n" + "\n".join(lines[-20:]))
    return 0


if __name__ == "__main__":
    os._exit(main())
