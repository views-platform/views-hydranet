#!/usr/bin/env python
"""compare_run_determinism.py — exact determinism check for two runs of the SAME model.

Two runs with the same seed + same data must be BIT-IDENTICAL. This compares them by the
**reliable** signals only: the weight-TENSOR hash (NOT the .pt file sha — that embeds
non-deterministic zip mtimes; see reports/reproducibility_runbook.md §1) and `np.array_equal`
on every saved `y_pred.npy`.

Usage:
    python scripts/compare_run_determinism.py <art1.pt> <art2.pt> [<preds_dir1> <preds_dir2>]

Exit code 0 = identical (deterministic), 1 = differs, 2 = bad args.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch


def state_dict_hash(path: str) -> str:
    """Hash model weights by tensor bytes (sorted keys). Reliable, unlike the .pt file sha."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def compare_predictions(dir1: str, dir2: str):
    """Exact-compare every origin_*/{target}/y_pred.npy. Returns (n_cmp, n_mismatch, n_missing)."""
    d1, d2 = Path(dir1), Path(dir2)
    rels = sorted(p.relative_to(d1) for p in d1.glob("origin_*/*/y_pred.npy"))
    n_cmp = n_mism = n_missing = 0
    for rel in rels:
        f2 = d2 / rel
        if not f2.exists():
            n_missing += 1
            print(f"  MISSING in B: {rel}")
            continue
        a, b = np.load(d1 / rel), np.load(f2)
        n_cmp += 1
        if a.shape != b.shape or not np.array_equal(a, b):
            n_mism += 1
            print(f"  MISMATCH: {rel}  (shapes {a.shape} vs {b.shape})")
    return n_cmp, n_mism, n_missing


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(2)

    h1, h2 = state_dict_hash(args[0]), state_dict_hash(args[1])
    weights_ok = h1 == h2
    print(f"weight-tensor hash A: {h1[:16]}…")
    print(f"weight-tensor hash B: {h2[:16]}…")
    print(f"WEIGHTS IDENTICAL: {weights_ok}")

    preds_ok = True
    if len(args) >= 4:
        n_cmp, n_mism, n_missing = compare_predictions(args[2], args[3])
        preds_ok = n_mism == 0 and n_missing == 0
        print(f"y_pred files compared: {n_cmp} | mismatches: {n_mism} | missing in B: {n_missing}")
        print(f"PREDICTIONS IDENTICAL: {preds_ok}")

    verdict = weights_ok and preds_ok
    print("\n" + ("✅ DETERMINISTIC — fixed-fixed" if verdict else "❌ NON-DETERMINISTIC — STOP"))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
