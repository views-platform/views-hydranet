#!/usr/bin/env python3
"""state_range.py — compute §4's `f` and render the verdict. Analysis only; captures nothing.

`f` = the fraction of R2 (deployment) per-cell values falling OUTSIDE their channel's [1%, 99%]
interval built from the pooled R1 (training-like) distribution.

By construction a 98% interval puts **f = 0.02** there by chance, which is what §4's thresholds are
multiples of (>=0.20 is 10x chance, <=0.05 is 2.5x). Nothing here is tuned to the observed values.

Registered choices this file implements, all fixed BEFORE it first ran:
  * AMENDMENT 5(a) the interval pools all three curriculum ratios; per-ratio `f` is secondary.
  * AMENDMENT 5(b) the verdict renders on the CELL half (M39); `hidden` is always reported.
  * §4                a seed split is INCONCLUSIVE, never resolved by picking a seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

RATIOS = (0.665, 0.35, 0.05)


def _interval(r1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel [1%, 99%] bounds of the pooled training-like distribution. `r1` is [C, N]."""
    return torch.quantile(r1, 0.01, dim=1), torch.quantile(r1, 0.99, dim=1)


def _f(r2: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> float:
    """Fraction of R2 cells outside their own channel's interval. `r2` is [C, N]."""
    outside = (r2 < lo[:, None]) | (r2 > hi[:, None])
    return float(outside.float().mean())


def _verdict(f: float) -> str:
    if f >= 0.20:
        return "OUT-OF-RANGE"
    if f <= 0.05:
        return "IN-RANGE"
    return "INCONCLUSIVE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--seeds", nargs="+", default=["fortytwo", "fortythree"])
    args = ap.parse_args()
    res = Path(args.results)

    report: dict = {"per_seed": {}}
    for seed in args.seeds:
        meta = json.loads((res / f"regimes_{seed}.json").read_text())
        if meta["F3"]["verdict"] != "PASS":
            raise SystemExit(f"{seed}: F3 did not pass — §4 must not be consulted (HARD STOP)")

        r2 = torch.load(res / f"r2_state_{seed}.pt")
        per_ratio_raw = {
            r: torch.load(res / f"r1_state_{seed}_ratio{r}.pt") for r in RATIOS
        }
        entry: dict = {"f3_worst_interior": meta["F3"]["worst_interior_rel_abs_diff"]}

        for half in ("cell", "hidden"):
            r2h = r2[half][0].reshape(r2[half].shape[1], -1)  # [C, N]
            pooled = torch.cat([per_ratio_raw[r][half] for r in RATIOS], dim=1)
            lo, hi = _interval(pooled)
            f_primary = _f(r2h, lo, hi)
            entry[half] = {
                "f": f_primary,
                "verdict": _verdict(f_primary),
                "n_channels": int(r2h.shape[0]),
                "n_r1_cells_per_channel": int(pooled.shape[1]),
                "r1_interval_width_mean": float((hi - lo).mean()),
                "r2_abs_max": float(r2h.abs().max()),
                "r1_abs_max": float(pooled.abs().max()),
                "per_ratio_f_SECONDARY": {
                    str(r): _f(r2h, *_interval(per_ratio_raw[r][half])) for r in RATIOS
                },
            }
        # AMENDMENT 5(b): the halves must be compared, not silently reconciled.
        entry["half_split"] = entry["cell"]["verdict"] != entry["hidden"]["verdict"]
        report["per_seed"][seed] = entry

    cell_verdicts = {s: report["per_seed"][s]["cell"]["verdict"] for s in args.seeds}
    if len(set(cell_verdicts.values())) > 1:
        report["verdict"] = "INCONCLUSIVE — SEED-SPLIT"
        report["note"] = (
            f"seeds disagree on the cell half: {cell_verdicts}; §4 forbids picking one"
        )
    else:
        report["verdict"] = next(iter(cell_verdicts.values()))

    (res / "STATE_RANGE.json").write_text(json.dumps(report, indent=1))

    print(f"\n=== §4 VERDICT (cell half): {report['verdict']} ===")
    print("    chance rate for a 98% interval = 0.02;  OUT >= 0.20,  IN <= 0.05\n")
    for seed, e in report["per_seed"].items():
        print(f"  seed {seed}  (F3 interior {e['f3_worst_interior']:.4f} PASS)")
        for half in ("cell", "hidden"):
            h = e[half]
            sec = "  ".join(f"{k}:{v:.3f}" for k, v in h["per_ratio_f_SECONDARY"].items())
            print(
                f"    {half:6} f={h['f']:.4f} -> {h['verdict']:12} "
                f"|R2|max={h['r2_abs_max']:7.3f}  |R1|max={h['r1_abs_max']:7.3f}"
            )
            print(f"           per-ratio f (SECONDARY, not the verdict): {sec}")
        if e["half_split"]:
            print("    ** HALF-SPLIT: cell and hidden land in different branches **")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
