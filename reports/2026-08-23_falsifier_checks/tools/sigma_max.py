#!/usr/bin/env python3
"""sigma_max.py — Check D (#294): is the ConvLSTM recurrent map expansive enough for GTF?

Threshold registered in GitHub issue #294 (2026-08-22); estimator registered in
`05_analysis_plan.md` AMENDMENT 1 (`e631f74`, 23:08:39) — both BEFORE this ran.

Hess et al. 2023 derive ``alpha = 1 - 1/sigma_max`` and state it **requires sigma_max >= 1**.
M41 measured our empirical optimum at w ~ 0.1, which if ``w == alpha`` implies sigma_max ~ 1.11.

**A conv's operator norm is not its kernel's matrix norm.** Power iteration is therefore run on
the convolution as an operator at the real spatial size, via ``conv2d`` / ``conv_transpose2d``.
Data-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CELLS = (1, 2, 3, 4)
GATES = ("Whi", "Whf", "Whc", "Who")


def conv_operator_norm(w: torch.Tensor, hw: tuple[int, int], iters: int, seed: int = 0) -> float:
    """Largest singular value of the conv operator ``v -> conv2d(v, w, padding=same)``.

    Power iteration on ``A^T A`` with matrix-free products: ``conv2d`` is A, ``conv_transpose2d``
    is A^T for the same weight and padding. Operates at the REAL field size because a
    convolution's spectrum depends on it.
    """
    cin = w.shape[1]
    pad = w.shape[-1] // 2
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, cin, *hw, generator=g)
    v /= v.norm()
    sigma = 0.0
    for _ in range(iters):
        u = F.conv2d(v, w, padding=pad)
        v_new = F.conv_transpose2d(u, w, padding=pad)
        n = v_new.norm()
        if n == 0:
            return 0.0
        v = v_new / n
        sigma = float(n.sqrt())
    return sigma


def jacobian_bound(norms: dict[str, float], m_cell: float) -> dict[str, float]:
    """Block bounds for one LSTM cell at ``M = max|hl|``. See AMENDMENT 1 for the derivation.

    |sigma'| <= 1/4, |tanh'| <= 1, gates and tanh outputs bounded by 1.
    """
    dhl_dhs = 0.25 * m_cell * norms["Whf"] + 0.25 * norms["Whi"] + norms["Whc"]
    dhl_dhl = 1.0  # max f < 1, bounded by 1
    dhs_dhl = 1.0  # max(o * tanh'(hl') * f) <= 1
    dhs_dhs = 0.25 * norms["Who"] + dhl_dhs
    # 2x2 block matrix: bound the operator norm by the Frobenius norm of the block-norm matrix
    frob = (dhl_dhl**2 + dhl_dhs**2 + dhs_dhl**2 + dhs_dhs**2) ** 0.5
    return {
        "dhl_dhl": dhl_dhl,
        "dhl_dhs": dhl_dhs,
        "dhs_dhl": dhs_dhl,
        "dhs_dhs": dhs_dhs,
        "bound": frob,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(a.artifact, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    sd = sd.get("model_state_dict", sd)

    hw = (a.height, a.width)
    norms, drift = {}, {}
    for k in CELLS:
        for gate in GATES:
            key = f"{gate}_{k}.weight"
            if key not in sd:
                raise SystemExit(f"missing {key} in the artifact")
            w = sd[key].float()
            s50 = conv_operator_norm(w, hw, 50)
            s200 = conv_operator_norm(w, hw, 200)
            norms[f"{gate}_{k}"] = s200
            drift[f"{gate}_{k}"] = abs(s200 - s50) / s200 if s200 else 0.0

    worst_drift = max(drift.values())
    converged = worst_drift <= 0.01

    print(f"field {hw[0]}x{hw[1]}   power iteration 200 steps\n")
    print("recurrent conv OPERATOR norms (data-free):")
    print(f"  {'cell':<6} " + " ".join(f"{g:>8}" for g in GATES))
    for k in CELLS:
        print(f"  {k:<6} " + " ".join(f"{norms[f'{g}_{k}']:>8.4f}" for g in GATES))
    print(
        f"\nfalsifier — max drift 50 vs 200 iters: {worst_drift:.2%} "
        f"({'CONVERGED' if converged else 'NOT CONVERGED — bound unreadable'})"
    )

    print("\nJacobian bound per cell, as a function of M = max|hl|:")
    print(f"  {'M':>6} " + " ".join(f"{'cell ' + str(k):>10}" for k in CELLS) + f" {'max':>10}")
    rows = {}
    for m in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        b = [jacobian_bound({g: norms[f"{g}_{k}"] for g in GATES}, m)["bound"] for k in CELLS]
        rows[m] = b
        print(f"  {m:>6.0f} " + " ".join(f"{x:>10.4f}" for x in b) + f" {max(b):>10.4f}")

    # where does the bound cross 1?
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        bm = max(
            jacobian_bound({g: norms[f"{g}_{k}"] for g in GATES}, mid)["bound"] for k in CELLS
        )
        if bm < 1.0:
            lo = mid
        else:
            hi = mid
    crossing = hi

    b_at_1 = max(rows[1.0])
    verdict = (
        "CLOSE #294 (bound < 1 => sigma_max < 1 definitively)"
        if b_at_1 < 1.0
        else "INCONCLUSIVE — an upper bound above 1 licenses nothing (AMENDMENT 1)"
    )
    print(f"\nbound crosses 1.0 at M = {crossing:.4f}")
    print(f"bound at M=1: {b_at_1:.4f}")
    print(f"\nVERDICT: {verdict}")
    if not converged:
        print("...but the power iteration did not converge, so this is VOID.")

    Path(a.out).write_text(
        json.dumps(
            {
                "artifact": a.artifact,
                "field": list(hw),
                "conv_operator_norms": norms,
                "drift_50_vs_200": drift,
                "converged": converged,
                "bound_by_M": {str(k): v for k, v in rows.items()},
                "bound_at_M1": b_at_1,
                "crosses_one_at_M": crossing,
                "verdict": "VOID" if not converged else verdict,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
