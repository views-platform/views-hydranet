#!/usr/bin/env python
"""hurdle_decompose.py — split hurdle E[y] = pi*body into GATE vs BODY contributions.

WHY (2026-06-21 batch, diagnostics #1/#2/#18). The proper-score gate showed the count/hurdle
model leaks predicted mass onto truly-zero cells. This attributes that leak (and the ns/os
positive under-firing) to the GATE (pi = P(y>0), the `by_*` head) vs the conditional BODY
(mu/(1-NB0), recovered as E[y]/pi). Read-only on saved predictions.

DATA NOTE (verified in-tool, not assumed): the saved `lr_*` y_pred is E[y] in COUNT space (the
output path inverse-transforms the log1p compose before writing) and `by_*` is the gate prob pi
in [0,1]. The tool asserts pi in [0,1] and reports the E[y] range so a reader can confirm
count-space, bounded values (not log1p, not exploded).

Attribution (per target x {STEP-1, FULL}):
  - zero-cell LEAK (#1/#2): on truth==0 cells, leak = mean(pi*body). Counterfactual reductions:
    set pi->prevalence rho (calibrated gate) vs set body->m_pos (mean positive magnitude); the
    larger reduction names the dominant lever.
  - positive UNDER-FIRING (#18): on truth>0 cells, recover by firing the gate (pi->1) vs fixing
    the body (body->truth); the larger recovery names the dominant lever.

This tool emits ONLY the DATA block (numbers). Interpretation goes in the dossier readout
(the DATA-vs-READING discipline).

Usage:
    python scripts/hurdle_decompose.py <pred_dir> [--raw <parquet>] [--targets ...] [--out <md>]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from mcr_readout import aligned_truth, load_truth_index  # C-136 fail-loud join guards

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
_PI_FLOOR = 1e-4  # by_* min observed ~1.7e-4; floor the division, never silently drop


# ---- pure functions (unit-tested) ----
def conditional_body(ey: np.ndarray, pi: np.ndarray, floor: float = _PI_FLOOR) -> np.ndarray:
    """Recover the conditional positive mean body = E[y]/pi (pi floored to avoid blow-up)."""
    return ey / np.maximum(pi, floor)


def attribute_zero_leak(pi: np.ndarray, body: np.ndarray, rho: float, m_pos: float) -> dict:
    """Zero cells: is the leak pi*body gate-driven (pi>>rho) or body-driven (body>>m_pos)?"""
    total = float(np.mean(pi * body))
    gate_corrected = float(np.mean(rho * body))  # set pi -> prevalence
    body_corrected = float(np.mean(pi * m_pos))  # set body -> reference magnitude
    red_gate = max(0.0, total - gate_corrected)
    red_body = max(0.0, total - body_corrected)
    denom = red_gate + red_body
    return {
        "total_leak": total,
        "pi_mean": float(np.mean(pi)),
        "body_mean": float(np.mean(body)),
        "rho": rho,
        "m_pos": m_pos,
        "red_gate": red_gate,
        "red_body": red_body,
        "frac_gate": red_gate / denom if denom > 0 else float("nan"),
        "frac_body": red_body / denom if denom > 0 else float("nan"),
        "dominant": "gate" if red_gate >= red_body else "body",
    }


def attribute_pos_underfire(pi: np.ndarray, body: np.ndarray, truth: np.ndarray) -> dict:
    """Positive-truth cells: is under-firing the gate (pi low) or the body (body<truth)?"""
    pred = float(np.mean(pi * body))
    recover_gate = max(0.0, float(np.mean(1.0 * body)) - pred)  # fire the gate fully
    recover_body = max(0.0, float(np.mean(pi * truth)) - pred)  # body == truth
    denom = recover_gate + recover_body
    return {
        "pi_mean": float(np.mean(pi)),
        "body_mean": float(np.mean(body)),
        "truth_mean": float(np.mean(truth)),
        "pred_mean": pred,
        "recover_gate": recover_gate,
        "recover_body": recover_body,
        "frac_gate": recover_gate / denom if denom > 0 else float("nan"),
        "frac_body": recover_body / denom if denom > 0 else float("nan"),
        "dominant": "gate" if recover_gate >= recover_body else "body",
    }


# ---- IO / collection ----
def _gate_target(lr_target: str) -> str:
    return lr_target.replace("lr_", "by_")


def _collect(pred_dir, lr_target, truth_idx):
    """Pool (truth, pi, E[y]) over all origins for STEP-1 (seed month) and FULL. Count space."""
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    by_target = _gate_target(lr_target)
    acc = {"STEP-1": [[], [], []], "FULL": [[], [], []]}  # truth, pi, ey
    ey_max = 0.0
    pi_lo, pi_hi = 1.0, 0.0
    for origin in origins:
        lrd = os.path.join(origin, lr_target)
        byd = os.path.join(origin, by_target)
        lp, ip = os.path.join(lrd, "y_pred.npy"), os.path.join(lrd, "identifiers.npz")
        bp = os.path.join(byd, "y_pred.npy")
        if not all(os.path.exists(p) for p in (lp, ip, bp)):
            continue
        ey = np.load(lp).astype(np.float64).mean(axis=1)  # E[y], count space
        pi = np.load(bp).astype(np.float64).mean(axis=1)  # gate pi
        ids = np.load(ip)
        tm, un = ids["time"], ids["unit"]
        truth = aligned_truth(truth_idx, tm, un, lr_target)
        finite = np.isfinite(ey) & np.isfinite(pi) & np.isfinite(truth)
        if finite.any():
            ey_max = max(ey_max, float(np.nanmax(ey[finite])))
            pi_lo = min(pi_lo, float(np.nanmin(pi[finite])))
            pi_hi = max(pi_hi, float(np.nanmax(pi[finite])))
        seed = int(tm.min())
        for tag, sel in (("STEP-1", tm == seed), ("FULL", np.ones_like(tm, bool))):
            m = sel & finite
            acc[tag][0].append(truth[m])
            acc[tag][1].append(pi[m])
            acc[tag][2].append(ey[m])
    out = {}
    for tag, (t, p, e) in acc.items():
        out[tag] = (
            (np.concatenate(t), np.concatenate(p), np.concatenate(e))
            if t
            else (np.array([]), np.array([]), np.array([]))
        )
    return out, ey_max, (pi_lo, pi_hi)


def _fmt_zero(tag, zl, rho, m_pos):
    return (
        f"    - ZERO-cell leak: mean E[y]={zl['total_leak']:.4f} | "
        f"pi_mean={zl['pi_mean']:.3f} (rho={rho:.4f}) "
        f"body_mean={zl['body_mean']:.2f} (m_pos={m_pos:.2f}) -> "
        f"**{zl['dominant']}**-driven (gate {zl['frac_gate']:.0%} / body {zl['frac_body']:.0%})"
    )


def _fmt_pos(pu):
    return (
        f"    - POSITIVE under-firing: pred={pu['pred_mean']:.2f} vs "
        f"truth={pu['truth_mean']:.2f} | pi_mean={pu['pi_mean']:.3f} "
        f"body_mean={pu['body_mean']:.2f} -> **{pu['dominant']}**-driven "
        f"(gate {pu['frac_gate']:.0%} / body {pu['frac_body']:.0%})"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = args.raw
    if raw is None:
        base = os.path.dirname(args.pred_dir.rstrip("/"))
        hits = sorted(glob.glob("data/raw/*calibration*.parquet")) or sorted(
            glob.glob(os.path.join(base, "..", "raw", "*calibration*.parquet"))
        )
        if not hits:
            ap.error("could not auto-find a calibration raw parquet; pass --raw")
        raw = hits[0]
    truth = {t: load_truth_index(raw, t) for t in args.targets}

    lines = [
        "# Hurdle decomposition — GATE (pi) vs BODY (mu) attribution",
        "",
        "## WHAT THE DATA SHOWS (numbers only; interpretation lives in the dossier readout)",
        f"pred_dir: `{args.pred_dir}`",
        "",
    ]
    for t in args.targets:
        coll, ey_max, (pi_lo, pi_hi) = _collect(args.pred_dir, t, truth[t])
        # in-tool VERIFY: pi is a probability, E[y] is bounded count-space (not log1p)
        assert -1e-9 <= pi_lo and pi_hi <= 1.0 + 1e-6, f"{t}: gate not in [0,1]"
        lines += [
            f"### {t} (pi in [{pi_lo:.4f},{pi_hi:.3f}]; E[y] max={ey_max:.1f}, count-space)",
            "",
        ]
        for tag in ("STEP-1", "FULL"):
            tr, pi, ey = coll[tag]
            if len(tr) == 0:
                lines.append(f"- {tag}: n=0")
                continue
            rho = float(np.mean(tr > 0))
            posm = tr > 0
            m_pos = float(np.mean(tr[posm])) if posm.any() else float("nan")
            body = conditional_body(ey, pi)
            zm = ~posm
            lines.append(
                f"- **{tag}** (n={len(tr)}, rho={rho:.4f}, mean_pos_truth={m_pos:.2f})"
            )
            if zm.any():
                zl = attribute_zero_leak(pi[zm], body[zm], rho, m_pos)
                lines.append(_fmt_zero(tag, zl, rho, m_pos))
            if posm.any():
                lines.append(_fmt_pos(attribute_pos_underfire(pi[posm], body[posm], tr[posm])))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
