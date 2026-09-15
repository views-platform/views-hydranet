"""Split the free-running forecast into OCCURRENCE and MAGNITUDE, without conditioning.

The question — does the model make FEWER forecasts or SMALLER ones — turns on one identity:

    mean_cells(g*mu)  =  mean_cells(g)  x  [ sum(g*mu) / sum(g) ]
    ----------------     --------------     -------------------
      EMITTED MASS         OCCURRENCE            MAGNITUDE

It is exact (it is the definition of a weighted mean), and every cell contributes at every
horizon. Nothing is thresholded and nothing is conditioned, so the survivorship rival — a downward
shift in the whole distribution leaving only upper-tail cells above a firing bar, propping up a
conditional mean — has nothing to select on.

Two magnitudes are reported, not one. The identity's term is **gate-weighted**, so a gate that
collapsed *non-uniformly* (staying high only where mu is large) could hold it flat while the
typical cell shrank. `mag_unweighted` is the unweighted mean of mu: co-primary, not a footnote.

`mag_tau_*` is the deliberately conditioned statistic — the `threshold_gate` composition, which is
what the suspect claim was originally read off. Sweeping tau varies the strength of the selection,
turning the rival from a hypothesis defended against by construction into a dose-response
measurement (dossier 05, amendment A1). It measures whether the STATISTIC is a selection
artifact; it does NOT model a run fed back under `threshold_gate`, which changes the trajectory.

Inputs are the raw fields written by `HydraNetInference._dump_body_mean`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

#: Selection strengths for the survivorship dose-response (dossier 05 A1.4).
DEFAULT_TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)

#: Written where a conditioned statistic had NO contributing cells. `float('nan')`, never 0.0 and
#: never a negative in-band marker: C-318 was an in-band -1.0 UNDEFINED sentinel averaged as if it
#: were a magnitude, which published "18.4 -> -0.8". NaN cannot be averaged into a plausible number
#: by accident, and `n_above` carries the count so a reader can see the support.
UNDEFINED = float("nan")


def decompose(mu: np.ndarray, gate: np.ndarray, taus=DEFAULT_TAUS) -> list[dict]:
    """Decompose one arm's fields into per-(target, horizon) occurrence and magnitude records.

    Args:
        mu: ``[T, n_reg, H, W]`` count-space body mean ``E[Y|body]``, un-composed.
        gate: ``[T, H, W, n_cls]`` per-cell ``P(y>0)``.
        taus: selection strengths for the conditioned dose-response.

    Returns:
        One record per (target, horizon). ``horizon`` is 1-based.
    """
    if mu.ndim != 4 or gate.ndim != 4:
        raise ValueError(
            f"expected 4-D mu [T,n_reg,H,W] and gate [T,H,W,n_cls]; got {mu.shape}, {gate.shape}"
        )
    t, n_reg, h, w = mu.shape
    if gate.shape[:3] != (t, h, w):
        raise ValueError(f"mu {mu.shape} and gate {gate.shape} disagree on T/H/W")
    if gate.shape[3] < n_reg:
        raise ValueError(f"gate has {gate.shape[3]} channels, need at least n_reg={n_reg}")
    if not np.isfinite(mu).all() or not np.isfinite(gate).all():
        raise ValueError("non-finite value in the dumped fields; refusing to summarise (ADR-003)")
    if (gate < 0).any() or (gate > 1).any():
        raise ValueError(
            "gate outside [0,1]; it is not a probability, so the identity does not hold"
        )

    records = []
    for j in range(n_reg):
        for step in range(t):
            g = gate[step, :, :, j].ravel()
            m = mu[step, j].ravel()
            sum_g = float(g.sum())
            rec = {
                "target_index": j,
                "horizon": step + 1,
                "n_cells": g.size,
                # OCCURRENCE — read straight off the gate field. Exact, not estimated.
                "occurrence": float(g.mean()),
                # EMITTED MASS — the composed forecast, mean over all cells.
                "emitted_mass": float((g * m).mean()),
                # MAGNITUDE, gate-weighted: the identity's second factor.
                "mag_gate_weighted": float((g * m).sum() / sum_g) if sum_g > 0 else UNDEFINED,
                # MAGNITUDE, unweighted: co-primary, sees a non-uniform gate collapse.
                "mag_unweighted": float(m.mean()),
            }
            for tau in taus:
                above = g >= tau
                n_above = int(above.sum())
                key = f"{tau:g}".replace(".", "p")
                rec[f"n_above_{key}"] = n_above
                # The conditioned statistic. NaN when unsupported — see UNDEFINED.
                rec[f"mag_tau_{key}"] = float(m[above].mean()) if n_above else UNDEFINED
            records.append(rec)
    return records


def assert_no_sentinel_survived(records: list[dict]) -> None:
    """Fail loud if an unsupported statistic could be mistaken for a measurement (dossier 03 C.4).

    A conditioned magnitude with zero contributing cells must be NaN, and a NaN must never coexist
    with a non-zero support count. This is the mechanical descendant of C-318.
    """
    for rec in records:
        for key, value in rec.items():
            if not key.startswith("mag_tau_"):
                continue
            n = rec[f"n_above_{key[len('mag_tau_') :]}"]
            if n == 0 and not math.isnan(value):
                raise ValueError(f"{key} has no support but is not NaN: {value!r} (C-318 class)")
            if n > 0 and math.isnan(value):
                raise ValueError(f"{key} has support n={n} but is NaN — the field is inconsistent")


def load_arm(dump_dir: Path, taus=DEFAULT_TAUS) -> list[dict]:
    """Decompose every origin dumped for one arm, tagging each record with its origin."""
    files = sorted(dump_dir.glob("bodymean_origin*.npz"))
    if not files:
        raise SystemExit(f"no bodymean_origin*.npz under {dump_dir}")
    out = []
    for f in files:
        z = np.load(f)
        for rec in decompose(z["mu"], z["gate"], taus):
            rec["origin"] = int(z["origin"])
            out.append(rec)
    assert_no_sentinel_survived(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    records = load_arm(args.dump_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
