#!/usr/bin/env python3
"""paired_ci.py — paired origin-block CI on AP(cell) - AP(none) for ONE seed.

Both cubes must already be on disk under `data/generated/paired_{arm}` (see
`reemit_for_paired_ci.sh`). One origin draw per replicate scores BOTH arms on the same resampled
cell set, so the correlation between two arms of the same model is carried rather than assumed
away — which is the whole reason M40 measured a paired MDE of 0.0086 against an unpaired 0.0541.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.ap_block_bootstrap import ap_diff_origin_block_ci  # noqa: E402

MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--h", type=int, default=18)
    p.add_argument("--target", default="sb")
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gen = MODELS / a.model / "data" / "generated"
    cell, none = gen / "paired_cell", gen / "paired_none"
    for d in (cell, none):
        if not d.is_dir():
            raise SystemExit(f"paired_ci: missing cube {d} — run reemit_for_paired_ci.sh first")
    truth = MODELS / a.model / "data" / "raw" / "calibration_datafactory_df.parquet"
    if not truth.is_file():
        raise SystemExit(f"paired_ci: missing truth parquet {truth}")

    res = ap_diff_origin_block_ci(
        pred_dir_a=str(cell), pred_dir_b=str(none), truth_parquet=str(truth),
        target=a.target, h=a.h, n_boot=a.n_boot, seed=0, ci=0.90,
    )
    res["model"], res["h"], res["target"] = a.model, a.h, a.target
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"  {a.model} h{a.h}: " + "  ".join(f"{k}={v}" for k, v in res.items() if k != "model"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
