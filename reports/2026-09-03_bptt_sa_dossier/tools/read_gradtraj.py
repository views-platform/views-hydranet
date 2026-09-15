#!/usr/bin/env python3
"""read_gradtraj.py — CREEP or JUMP? The pre-committed readout for the GRAD-TRAJ probe.

The rule below is written BEFORE the run and is committed with the launcher. That ordering is the
whole point: SCREEN-2's rule enumerated three numeric outcomes and had no branch for the run that
actually happened (C-320, fourth instance), and the fix is to fix the branches first — including
the one that says the data does not decide.

WHAT IS MEASURED
    The engine's opt-in per-lesson CSV: lesson, raw_grad_norm (PRE-clip), loss_reg, loss_cls,
    gate_logit_mean. `raw_grad_norm` is the L2 norm over all model parameters, computed before
    `clip_grad_norm_`, so it is the quantity clipping would hide.

WHY A CONTROL
    Gradient norms move a lot early in training. Without `trajdetached` over the SAME lessons a
    rise in `trajattached` cannot be called abnormal. Both arms are compared on the identical
    lesson index range.

THE RULE (pre-committed)
    Let W_late  = the last 10 logged lessons of the attached arm.
    Let W_early = lessons 15..25 — post-warmup (ss_warmup_lessons=15), so epsilon is already
                  pinned at its max of 0.5 and is no longer changing. Any later movement is
                  therefore NOT a dose effect.

    CREEP  requires BOTH  (a) attached median over W_late >= 3x its own median over W_early, and
                          (b) attached median over W_late >= 3x the CONTROL's median over W_late.
           Read: the gradient grew, and grew in a way the control does not.

    JUMP   requires BOTH  (a) attached median over W_late <  1.5x its own median over W_early, and
                          (b) attached median over W_late <  1.5x the control's median over W_late.
           Read: the gradient was ordinary right up to the lesson it became NaN.

    Anything else is AMBIGUOUS and is reported as AMBIGUOUS. A partial rise does not get rounded
    to whichever story is more convenient.

    Spearman rho of raw_grad_norm against lesson over the post-warmup range is reported for both
    arms as SUPPORTING evidence. It does not enter the rule -- a monotone rise from 1.0 to 1.4 is
    rho=1.0 and means nothing here.

WHAT THIS PROBE CANNOT DO
    n=1 seed, one configuration. It separates two mechanism families; it does not identify the
    mechanism, and CREEP would not by itself prove the Jacobian-product story (sigma_max=7.76,
    issue #294) over any other route to a growing gradient.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "gradtraj"
ARMS = {"attached": RES / "traj_attached.csv", "detached": RES / "traj_detached.csv"}

WARMUP = 15  # config ss_warmup_lessons: epsilon is pinned at 0.5 from here on
EARLY = (15, 25)
LATE_N = 10
CREEP_FACTOR = 3.0
JUMP_FACTOR = 1.5


def _read(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open() as fh:
        rows = []
        for r in csv.DictReader(fh):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                # A row written mid-crash can be short or carry 'nan'. float('nan') parses, so
                # this only drops genuinely malformed rows -- and says so rather than skipping.
                print(f"  ! unparseable row in {path.name}: {r}")
        return rows


def _median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median of an empty window — the probe has no data to read")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def _window(rows: list[dict[str, float]], lo: int, hi: int) -> list[float]:
    return [r["raw_grad_norm"] for r in rows if lo <= r["lesson"] <= hi]


def main() -> int:
    data = {k: _read(p) for k, p in ARMS.items()}
    print("=" * 78)
    print("GRAD-TRAJ readout — CREEP or JUMP?")
    print("=" * 78)

    for name, rows in data.items():
        n = len(rows)
        last = rows[-1]["lesson"] if rows else None
        print(f"{name:9s}: {n:3d} lessons logged, last lesson index {last}")
    if not data["attached"]:
        print("\nATTACHED ARM PRODUCED NO TRAJECTORY. Nothing to read; the probe did not run.")
        return 2

    att, det = data["attached"], data["detached"]
    late_lo = att[-1]["lesson"] - LATE_N + 1
    a_late = _window(att, late_lo, att[-1]["lesson"])
    a_early = _window(att, *EARLY)
    if not a_early or not a_late:
        print(
            f"\nATTACHED died before lesson {EARLY[1]} — no post-warmup window. Cannot apply the"
        )
        print("rule; report the raw trajectory instead.")
        for r in att:
            print(f"  lesson {int(r['lesson']):3d}  grad {r['raw_grad_norm']:12.4f}")
        return 2

    print(
        f"\nwindows: early = lessons {EARLY[0]}-{EARLY[1]}, late = lessons "
        f"{int(late_lo)}-{int(att[-1]['lesson'])}"
    )
    ma_e, ma_l = _median(a_early), _median(a_late)
    self_ratio = ma_l / ma_e if ma_e else float("inf")
    print(
        f"\nattached  median grad early {ma_e:12.4f}   late {ma_l:12.4f}   "
        f"late/early = {self_ratio:.2f}x"
    )

    ctrl_ratio = None
    if det:
        d_late = _window(det, late_lo, att[-1]["lesson"])
        d_early = _window(det, *EARLY)
        if d_late:
            md_l = _median(d_late)
            ctrl_ratio = ma_l / md_l if md_l else float("inf")
            md_e = _median(d_early) if d_early else float("nan")
            print(
                f"control   median grad early {md_e:12.4f}"
                f"   late {md_l:12.4f}   attached/control late = {ctrl_ratio:.2f}x"
            )
        else:
            print(
                f"control   has NO rows in lessons {int(late_lo)}-{int(att[-1]['lesson'])} — "
                "the control was capped too early to compare."
            )
    else:
        print("control   MISSING — the comparison the rule requires cannot be made.")

    for name, rows in (("attached", att), ("detached", det)):
        post = [r for r in rows if r["lesson"] >= WARMUP]
        if len(post) >= 3:
            rho = _spearman([r["lesson"] for r in post], [r["raw_grad_norm"] for r in post])
            print(
                f"  (supporting) {name:9s} spearman rho(grad, lesson) post-warmup = {rho:+.3f} "
                f"over {len(post)} lessons"
            )

    print("\nlast 12 lessons of the attached arm:")
    print(
        f"  {'lesson':>7} {'grad_norm':>13} {'loss_reg':>11} {'loss_cls':>11} {'gate_logit':>11}"
    )
    for r in att[-12:]:
        print(
            f"  {int(r['lesson']):7d} {r['raw_grad_norm']:13.4f} {r['loss_reg']:11.4f} "
            f"{r['loss_cls']:11.4f} {r['gate_logit_mean']:11.4f}"
        )

    print("\n" + "-" * 78)
    if ctrl_ratio is None:
        print("VERDICT: INCOMPLETE — no control window. The rule needs both comparisons.")
        return 3
    if self_ratio >= CREEP_FACTOR and ctrl_ratio >= CREEP_FACTOR:
        print(f"VERDICT: CREEP — the gradient grew {self_ratio:.1f}x over its own post-warmup")
        print(f"         baseline and ran {ctrl_ratio:.1f}x the control. A real explosion that")
        print(
            "         develops during training. A stabiliser (feedback-path clipping, GTF alpha)"
        )
        print("         is now worth GPU time; it was not before this line.")
    elif self_ratio < JUMP_FACTOR and ctrl_ratio < JUMP_FACTOR:
        print(f"VERDICT: JUMP — the gradient tracked the control ({ctrl_ratio:.2f}x) and its own")
        print(f"         baseline ({self_ratio:.2f}x) right up to the lesson it went non-finite.")
        print("         That is a numerical DEFECT in the straight-through path, not an unstable")
        print("         objective. Do NOT buy a stabiliser. Find the defect.")
    else:
        print(f"VERDICT: AMBIGUOUS — self {self_ratio:.2f}x, control {ctrl_ratio:.2f}x. Neither")
        print(
            f"         the CREEP bar ({CREEP_FACTOR}x on both) nor the JUMP bar "
            f"({JUMP_FACTOR}x on both) is met."
        )
        print("         Reported as ambiguous by the pre-committed rule. It is not rounded.")
    print("-" * 78)
    print("SCOPE: n=1 seed, one configuration. This separates two mechanism FAMILIES. It does not")
    print("       identify the mechanism, and CREEP would not on its own single out the Jacobian-")
    print(
        "       product story (sigma_max=7.76, #294) over any other route to a growing gradient."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
