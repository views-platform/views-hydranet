"""rho_event_teacher_forced.py — ρ_event for TEACHER-FORCED fields, in a table of their own.

**Serves:** *"if we changed the 'magnitude head' to a 'ranking head' do we think we could rank
better?"* — this is the instrument that reads the answer, in the one regime where a patch-trained
ranking head and the body can be compared with **one variable changed**.

⛔⛔ **WHY THIS IS A SECOND FILE AND NOT A FLAG ON `rho_event_readout.py`.** That tool's control
rows are the **delivered autoregressive** emission, and its `G-TF` refuses any field whose stored
`input_mode` is not `"delivered"`. The refusal is correct and stays: comparing a teacher-forced
field with the delivered `+0.3127` is the `C-399` carry — teacher-forced AUC **0.99777** against
the delivered **0.72340**, with no decay in horizon. ⇒ A teacher-forced score must be read against
a **teacher-forced control from the same dumper**, in its own table. This is that table, and its
refusal is the **mirror**: it rejects a `"delivered"` field just as firmly.

⭐ **What makes the comparison legitimate is that only the loss differs.** Both rows come from
`score_on_the_grid.py`, same forward loop, same volume, same teacher-forced inputs, same cells,
same months, same firing set. The control is the body's `mu` field (`--channel mu`); the arm is a
ranking head's score (`--channel latent`). The patch→global ordering axis is what moves.

⭐ **The month offset is re-resolved from each field's own gate, never assumed.** A dumped field is
indexed by **fed** month; the h1 forecast it carries is for a *different* month, and being wrong by
one is silent. `gate_alignment.py` fixes it by measurement, and this tool calls the same sweep with
the same thresholds on every field it loads, so a field whose offset does not resolve is refused
rather than read.

⚠️ **The firing set is the DELIVERED gate's**, at `τ` from the arm config — identical for control
and arm, and identical to the delivered table's population definition. The population is held
fixed on purpose; only the ranking index varies (`C-351`).

⛔ **What this tool does NOT do.** It does not score a verdict. `00_README` §6's kill rule and
`04_roadmap` phase 3's pre-commitment apply to every number it prints: a `ρ_event` from an
unconverged arm is an **instrument reading**, not evidence for or against the ranking body.

Usage::

    python rho_event_teacher_forced.py --control ../results/score_grid_CONTROL_MU_<tag>.npz
    python rho_event_teacher_forced.py --control <mu>.npz --arm <score>.npz --label RANK

Exit 0 = every gate passed; exit 1 = a gate failed and nothing was reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# ⭐⭐ IMPORTED, never copied — §9c's first rule, and `tests/test_rho_event_teacher_forced.py`
# asserts the SAME FUNCTION OBJECTS as the delivered readout and the alignment gate, so a
# near-copy of either cannot drift in. Every earlier comparison in the §9c dossier was against a
# near-copy, and both of its HOLDs were about a baseline that moved.
from gate_alignment import ANOM_MARGIN, ANOM_R, OFFSETS, anomaly_r  # noqa: E402
from rho_event_readout import (  # noqa: E402
    _DISTINCT_FLOOR,
    DDOF,
    _fold_inputs,
    _summarise,
    distinct_share,
    e1_data,
    rho_event,
)

#: ⛔ The mode this table accepts. The mirror of `rho_event_readout`'s `G-TF`.
_TF_MODE = "teacher_forced"

#: ⭐⭐ `G-1`'s pin — the teacher-forced control's own published figures. ⛔ UNSET until the
#: control has been measured once and the number written in here. While it is `None` this
#: tool prints the control and **refuses to score an arm**, because an arm row read against
#: an unverified
#: control is exactly the shape §9c's withdrawn headlines had.
#: ⭐ MEASURED 2026-11-10 on `score_grid_CONTROL_MU_v20261012.npz` (sha `4acbf4d3…`, 7 folds,
#: months 463-469, ddof=1): **+0.5189 ± 0.0473**, distinct share **1.000000**.
#: ⚠️⚠️ THIS IS NOT THE DELIVERED `+0.3127` AND MAY NOT BE SET BESIDE IT. The same body's
#: delivered autoregressive `ρ_event` is `+0.3127 ± 0.0713`; the +0.2062 difference is the
#: `C-399` teacher-forcing carry and the `C-317` checkpoint identity together, and nothing on
#: disk separates them. ⇒ A ranking arm's bar in THIS table is **+0.5189**, not `+0.3127`.
_ANCHOR_TF_MU: float | None = 0.5189
_ANCHOR_TF_MU_SD: float | None = 0.0473
_TOL = 5e-5


def resolve_offset(gate: np.ndarray, months: list[int]) -> int:
    """-> the fed→forecast month offset, by measurement against the delivered gate.

    ⛔ Raises `SystemExit` unless exactly one candidate resolves at `gate_alignment`'s own
    thresholds. Those thresholds are imported, not restated, so the two instruments cannot drift.
    """
    delivered = {mo: g for (g, _u, mo) in e1_data.load_gate_h1().values()}
    rows = []
    for off in OFFSETS:
        pairs = [(months.index(dm + off), dm) for dm in sorted(delivered) if dm + off in months]
        if not pairs:
            continue
        a = np.array([gate[i] for i, _ in pairs])
        b = np.array([delivered[dm] for _, dm in pairs])
        rows.append((off, anomaly_r(a, b), len(pairs)))
    if len(rows) < 2:
        raise SystemExit(
            "G-ALIGN REFUSED: fewer than two candidate offsets overlap the delivered months, so "
            "no margin can be computed and the month indexing is unestablished."
        )
    rows.sort(key=lambda r: -r[1])
    best, runner = rows[0], rows[1]
    margin = best[1] - runner[1]
    if best[1] < ANOM_R or margin < ANOM_MARGIN:
        raise SystemExit(
            f"G-ALIGN REFUSED: best offset {best[0]:+d} at anomaly r {best[1]:+.6f}, margin "
            f"{margin:+.6f} — below the thresholds (r >= {ANOM_R}, margin >= {ANOM_MARGIN}). The "
            "field's month indexing is unestablished, so nothing may be read off it."
        )
    print(
        f"  ⭐ G-ALIGN offset {best[0]:+d} · anomaly r {best[1]:+.6f} · margin {margin:+.6f} "
        f"⇒ the forecast for month m is read at fed index m{best[0]:+d}"
    )
    return int(best[0])


def forecast_index(fed: list[int], off: int) -> dict[int, int]:
    """-> {forecast month: index into the fed axis}, for an offset resolved by `resolve_offset`.

    ⛔⛔ **THE DIRECTION OF THIS MAP IS THE WHOLE POINT OF `G-ALIGN`, AND IT IS EASY TO INVERT.**
    The sweep pairs the dumped field at fed month `dm + off` with the DELIVERED field for month
    `dm`. ⇒ the forecast for month `m` is read at fed index `m + off`, so the field at fed month
    `t` carries the forecast for `t - off`. ⭐ `mu_ordering.py` does exactly this in code
    (`src = fed_idx[month + best]`); this is the same map, not a re-derivation.

    ⚠️ Both tools' PROSE said the inverse ("feeding month t yields the forecast for t+off"), which
    is wrong in sign. mu_ordering's code was right and only its sentence was not. An inverted map
    here was caught on 2026-11-10 by the month-coverage refusal in `load_tf_field`, **not** by
    anyone reading either sentence — which is why this map is a named function with its own test.
    """
    return {m - off: i for i, m in enumerate(fed)}


def load_tf_field(path: Path, months: list[int], cells: np.ndarray) -> dict[int, np.ndarray]:
    """-> {forecast month: field}, refusing anything not teacher-forced, not on these cells.

    ⛔⛔ Three refusals, each one a defect this repository has already paid for:

    * **`input_mode`** — must be ``teacher_forced``. A ``delivered`` field belongs in
      `rho_event_readout.py`'s table, and mixing the two in either direction is `C-399`.
    * **`land_cells`** — element for element, not merely in length. Cell index IS geography
      here (`C-289`, `C-419`), so a permuted field scores a spatial sweep and returns a
      plausible number.
    * **the offset** — resolved by measurement from the field's own gate, above.
    """
    with np.load(path) as z:
        keys = sorted(z.keys())
        for need in ("score", "months", "gate", "land_cells"):
            if need not in keys:
                raise SystemExit(
                    f"G-0 REFUSED: {path.name} must carry {need!r}; it carries {keys}. This table "
                    "re-resolves the month offset from the field's own gate, so the gate is "
                    "required, not optional."
                )
        mode = str(z["input_mode"]) if "input_mode" in z else "UNDECLARED"
        if mode != _TF_MODE:
            raise SystemExit(
                f"G-TF REFUSED: this field was dumped with input_mode={mode!r}, and every row of "
                f"this table is {_TF_MODE!r}. A delivered field belongs in rho_event_readout.py; "
                "putting the two in one table is the C-399 carry in the other direction."
            )
        got_cells = np.asarray(z["land_cells"])
        if got_cells.shape != cells.shape or not np.array_equal(got_cells, cells):
            raise SystemExit(
                f"G-CELLS REFUSED: the field carries {got_cells.size} cells against this design's "
                f"{cells.size}, or in a different ORDER. Cell index is geography here "
                "(C-289/C-419), so a permuted field scores a spatial sweep."
            )
        channel = str(z["channel"]) if "channel" in z else "UNDECLARED"
        fed = [int(m) for m in z["months"]]
        gate = np.asarray(z["gate"], dtype=np.float64)
        score = np.asarray(z["score"], dtype=np.float64)
    if score.ndim != 2 or score.shape[0] != len(fed):
        raise SystemExit(
            f"G-0 REFUSED: score has shape {score.shape} against {len(fed)} fed months — the two "
            "axes do not agree, so no row may be scored."
        )
    print(f"  {path.name} · channel={channel} · input_mode={mode} · {len(fed)} fed months")
    off = resolve_offset(gate, fed)
    forecast = {m: score[i] for m, i in forecast_index(fed, off).items()}
    missing = [m for m in months if m not in forecast]
    if missing:
        raise SystemExit(
            f"G-0 REFUSED: after the {off:+d} offset the field covers forecast months "
            f"{min(forecast)}..{max(forecast)} and is missing {missing}. Partial coverage would "
            "score a different population (C-351)."
        )
    return {m: forecast[m] for m in months}


def _score_rows(field: dict[int, np.ndarray], folds: list[dict]) -> tuple[list[float], float]:
    """-> (per-fold ρ_event, the minimum distinct share over the folds' firing cells)."""
    rhos, shares = [], []
    for f in folds:
        x = field[f["month"]]
        rhos.append(rho_event(x, f["truth"], f["event_m"]))
        shares.append(distinct_share(x[f["open_m"]]))
    return rhos, min(shares)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--control",
        type=Path,
        required=True,
        help="the body's teacher-forced mu field (score_on_the_grid.py --channel mu)",
    )
    ap.add_argument("--arm", type=Path, default=None, help="a ranking head's score field")
    ap.add_argument("--label", default="RANK", help="the row label for --arm")
    args = ap.parse_args()

    print(__doc__.split("Usage::")[0])
    print("=" * 100)
    folds, tau, cells = _fold_inputs()
    months = [f["month"] for f in folds]
    print(
        f"  population: {len(folds)} folds, score months {months[0]}..{months[-1]}, sb h1, "
        f"firing set = DELIVERED gate >= tau {tau} · truth in raw counts · ddof={DDOF}"
    )

    print("\n  CONTROL")
    ctl = load_tf_field(args.control, months, cells)
    ctl_rhos, ctl_share = _score_rows(ctl, folds)
    ctl_mean, ctl_sd = _summarise(ctl_rhos)
    if ctl_share < _DISTINCT_FLOOR:
        print(
            f"\n  ⛔ FAIL G-2: the control takes only {ctl_share:.6f} distinct values over the "
            f"firing cells, below the {_DISTINCT_FLOOR} floor — it would be reporting its own "
            "ties. (§0.2: a delivery served 0.0 for all 2,330,712 cells and every structural "
            "check passed.)"
        )
        return 1

    rows = [("BODY mu (TF)", ctl_mean, ctl_sd, ctl_share)]

    if _ANCHOR_TF_MU is None:
        print(
            f"\n  ⚠️⚠️ G-1 PIN UNSET. The teacher-forced control measures "
            f"{ctl_mean:+.4f} ± {ctl_sd:.4f} over {len(folds)} folds, distinct share "
            f"{ctl_share:.6f}.\n     ⇒ Write these into `_ANCHOR_TF_MU` / `_ANCHOR_TF_MU_SD` and "
            "re-run, so that every later\n     reading must reproduce the control by arithmetic "
            "before it may print an arm row."
        )
        if args.arm is not None:
            print(
                "\n  ⛔ REFUSED to score --arm: an arm row read against an UNVERIFIED control is "
                "the exact shape\n     of the headlines §9c had to withdraw. Pin the control "
                "first."
            )
            return 1
    else:
        d_mean = abs(ctl_mean - _ANCHOR_TF_MU)
        d_sd = abs(ctl_sd - (_ANCHOR_TF_MU_SD or 0.0))
        ok = d_mean <= _TOL and d_sd <= _TOL
        print(
            f"\n  {'⭐ PASS' if ok else '⛔ FAIL'} G-1 · control {ctl_mean:+.4f} ± {ctl_sd:.4f} "
            f"against the pin {_ANCHOR_TF_MU:+.4f} ± {_ANCHOR_TF_MU_SD:.4f} "
            f"(|Δ| {d_mean:.2e} / {d_sd:.2e}, tol {_TOL:.0e})"
        )
        if not ok:
            print(
                "     REFUSED: the control did not reproduce, so the comparison is not the one "
                "that was pinned and no arm row may be printed."
            )
            return 1
        if args.arm is not None:
            print(f"\n  ARM · {args.label}")
            arm = load_tf_field(args.arm, months, cells)
            arm_rhos, arm_share = _score_rows(arm, folds)
            arm_mean, arm_sd = _summarise(arm_rhos)
            rows.append((args.label, arm_mean, arm_sd, arm_share))
            if arm_share < _DISTINCT_FLOOR:
                print(
                    f"\n  ⛔ FAIL G-2: the arm takes only {arm_share:.6f} distinct values over "
                    f"the firing cells, below the {_DISTINCT_FLOOR} floor."
                )
                return 1

    print(f"\n  {'index':<16}{'rho_event':>12}{'SD':>10}{'distinct':>12}")
    for label, mean, sd, share in rows:
        print(f"  {label:<16}{mean:>+12.4f}{sd:>10.4f}{share:>12.6f}")
    if len(rows) == 2:
        delta = rows[1][1] - rows[0][1]
        print(
            f"\n  ⇒ {rows[1][0]} − BODY mu (TF) = {delta:+.4f} rho_event, both teacher-forced, "
            "one variable changed: the loss."
        )
        print(
            "\n  ⛔ This is an INSTRUMENT READING, not a verdict. 04_roadmap phase 3's "
            "pre-commitment applies:\n     it says nothing for or against the ranking body until "
            "the arm is converged and its dispersion known."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
