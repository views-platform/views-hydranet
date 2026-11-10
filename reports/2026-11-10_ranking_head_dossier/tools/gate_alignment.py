"""gate_alignment.py — ⛔ the gate that must pass before any dumped field may be read.

A field from `score_on_the_grid.py` is indexed by **fed** month. Whether feeding month `t` yields
the h1 forecast for `t` or for `t+1` is a **convention**, and getting it wrong by one is
**silent**: it shifts every number by a month and still returns a plausible answer. ⚠️ The nearest
scar is `C-399` (Tier 1) — a reconstruction that looked fine and was feeding the model its own
answers.

⭐ So the offset is fixed **BY MEASUREMENT, not by reading the code.** `score_on_the_grid.py`
collects the **gate** on every run whatever `--channel` asked for, and the **delivered** gate at h1
is on disk for all 13 origins (`e1_data.load_gate_h1`). This sweeps the candidate offsets and
requires **exactly one** to resolve. ⛔ If none does, or more than one does, the dumped field is
not trustworthy and nothing may be read off it.

⚠️ **The LEVEL correlation cannot resolve a month** and is kept only as a reported diagnostic: the
gate field is too temporally smooth, so every offset reads ≈0.93–0.98. The resolving statistic is
the **ANOMALY** correlation — each field minus its across-month mean — which is what
`2026-11-08_ordering_discrimination/tools/mu_ordering.py` established and what its thresholds were
set on. This tool reuses those thresholds unchanged so the two cannot drift apart.

⛔⛔ **What a PASS does NOT establish.** The anomaly `r` at the winning offset is **not 1.0**, and
two explanations are live and are **not separated by anything on disk**: (a) the dumping checkpoint
may not be the delivered weights (`C-317` — `4acbf4d3…` is the retrained control), and (b) the dump
is **teacher-forced** while the delivery is autoregressive (`C-399`). ⇒ A PASS fixes the **month
offset and the cell order**. It does **not** license comparing a dumped field with a delivered
number; `rho_event_readout.py` refuses that separately, on the stored `input_mode`.

Usage::

    python gate_alignment.py ../results/score_grid_<tag>.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE.parents[1]
for _p in (
    _REPORTS / "2026-11-04_delivered_gate_lookup/tools",
    _REPORTS / "2026-10-30_upsample_arm_dossier/tools",
):
    sys.path.insert(0, str(_p))

import e1_data  # noqa: E402

#: ⭐ Unchanged from `mu_ordering.py`'s pre-registration, so the two instruments cannot drift.
OFFSETS = (-2, -1, 0, 1, 2)
ANOM_R = 0.30
ANOM_MARGIN = 0.10


def anomaly_r(a: np.ndarray, b: np.ndarray) -> float:
    """-> the correlation of the two fields' across-month ANOMALIES, pooled over months/cells."""
    aa, ba = a - a.mean(0), b - b.mean(0)
    return float(np.corrcoef(aa.ravel(), ba.ravel())[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("field", type=Path, help="an .npz written by score_on_the_grid.py")
    args = ap.parse_args()

    print(__doc__.split("Usage::")[0])
    print("=" * 100)
    with np.load(args.field) as z:
        gate = np.asarray(z["gate"], dtype=np.float64)
        months = [int(m) for m in z["months"]]
        cells = np.asarray(z["land_cells"])
        mode = str(z["input_mode"]) if "input_mode" in z else "UNDECLARED"
        channel = str(z["channel"]) if "channel" in z else "UNDECLARED"
    print(
        f"  {args.field.name} · channel={channel} · input_mode={mode} · "
        f"{len(months)} fed months {months[0]}..{months[-1]}"
    )

    _grid, e1_cells, _m0 = e1_data.truth_grid()
    same_cells = np.array_equal(cells, np.asarray(e1_cells))
    print(f"\n  ⛔ G-CELLS · cell order identical to e1_data's: {same_cells}")
    if not same_cells:
        print(
            "     REFUSED: cell index is geography here (C-289/C-419), so a permuted field "
            "would score a spatial sweep against the truth and return a plausible number."
        )
        return 1

    delivered = {mo: g for (g, _u, mo) in e1_data.load_gate_h1().values()}
    dmonths = sorted(delivered)
    print(f"  delivered gate on disk: {len(dmonths)} origins, months {dmonths[0]}..{dmonths[-1]}")

    print(
        f"\n  {'offset':>7}{'LEVEL r':>12}{'ANOMALY r':>12}{'months':>9}   (level is a "
        "DIAGNOSTIC — it cannot resolve a month)"
    )
    rows = []
    for off in OFFSETS:
        pairs = [(months.index(dm + off), dm) for dm in dmonths if dm + off in months]
        if not pairs:
            print(f"  {off:>+7}{'—':>12}{'—':>12}{0:>9}   (outside the dumped range)")
            continue
        A = np.array([gate[i] for i, _ in pairs])
        B = np.array([delivered[dm] for _, dm in pairs])
        lv = float(np.mean([np.corrcoef(A[k], B[k])[0, 1] for k in range(len(pairs))]))
        an = anomaly_r(A, B)
        rows.append((off, lv, an, len(pairs)))
        print(f"  {off:>+7}{lv:>+12.6f}{an:>+12.6f}{len(pairs):>9}")

    if not rows:
        print("\n  ⛔ REFUSED: no candidate offset overlaps the delivered months at all.")
        return 1
    rows.sort(key=lambda r: -r[2])
    best, runner = rows[0], rows[1]
    margin = best[2] - runner[2]
    print(
        f"\n  ⇒ best offset {best[0]:+d} at anomaly r {best[2]:+.6f}; runner-up "
        f"{runner[0]:+d} at {runner[2]:+.6f}; margin {margin:+.6f}"
    )
    ok = best[2] >= ANOM_R and margin >= ANOM_MARGIN
    print(
        f"  {'⭐ PASS' if ok else '⛔ FAIL'} G-ALIGN (anomaly r >= {ANOM_R}, "
        f"margin >= {ANOM_MARGIN})"
    )
    if not ok:
        print(
            "     REFUSED: the offset is not resolved, so the field's month indexing is "
            "unestablished and nothing may be read off it."
        )
        return 1
    print(
        f"\n  ⇒ feeding month t yields the h1 forecast for month t{best[0]:+d} — established by "
        "measurement, not by reading the code."
    )
    print(
        "\n  ⛔⛔ And what this does NOT establish: the anomaly r is not 1.0, and the dumping "
        "checkpoint's\n     identity (C-317) and the teacher-forced input mode (C-399) are both "
        "live explanations\n     for the gap. They are NOT separated by anything on disk. A PASS "
        "fixes the offset and the\n     cell order; it does not license comparing this field to "
        "a delivered number."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
