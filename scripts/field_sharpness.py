#!/usr/bin/env python3
"""field_sharpness.py — spatial-structure statistics for one emitted field vs its truth.

Composes three EXISTING, tested statistics rather than adding a fourth. Nothing here implements a
new metric; the only new thing is the **ratio**, and the reason it exists is below.

* ``fss`` / ``matched_pred_thresh`` / ``_score_field`` — `scripts/sharpness_scorecard.py`
* ``moran_i`` — `views_hydranet/utils/gate_field_structure.py`

**Why FSS and not a Fourier high-frequency fraction.** `matched_pred_thresh` picks the threshold at
which the model fires in *exactly as many cells as truth has events*. FSS then asks: given the
same event count, are they in the right places? The activation rate — which moves by more than
9x across the horizon in some arms — cannot drive the answer. A spectral metric would have to
correct for it after the fact, and there is no FFT anywhere in this repo to reuse.

**The FSS ratio is NOT the blur detector — established by measurement, not assumed.**
The first design made ``fss@1 / fss@11`` primary, reasoning that blur costs the fine scale
disproportionately. It does — but a *displaced* forecast, of identical sharpness in the wrong
place, costs it far MORE: on synthetic fields, blur at sigma=4 gives ratio **0.752** while a
3-cell displacement gives **0.267**. As a blur detector it would have reported displacement as
blur, with confidence.

**The blur detectors are the INTRINSIC statistics** — computed on the prediction alone, with no
reference to truth, so a wrong-place forecast cannot move them:

===============  ==========  ==============  ==============  ==============
statistic        perfect     blurred (s=2)   displaced (3)   thinned 50%
===============  ==========  ==============  ==============  ==============
``moran_i``      0.636       **0.968**       0.636           0.292
``conc1pct``     0.0277      **0.0967**      0.0277          0.0274
``fss_ratio``    1.000       0.897           **0.267**       0.876
===============  ==========  ==============  ==============  ==============

``moran_i`` is primary: it rises monotonically with blur, is unmoved by displacement, and falls
under thinning — it separates all three failure modes. ``conc1pct`` is the independent second
(note it rises under blur; the opposite direction was predicted before measuring, and was wrong).
``fss_ratio`` is reported as *agreement* context and must not be read as sharpness.

`tests/test_field_sharpness.py` pins every one of these behaviours on synthetic fields with known
answers, so the instrument is demonstrated fit for purpose before it is pointed at real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HN))
sys.path.insert(0, str(_HN / "scripts"))

from sharpness_scorecard import _score_field  # noqa: E402

from views_hydranet.utils.gate_field_structure import moran_i  # noqa: E402

SCALES = (1, 3, 5, 11)


def field_sharpness(pred_grid: np.ndarray, truth_grid: np.ndarray) -> dict[str, float]:
    """Spatial-structure summary of ``pred_grid`` against ``truth_grid``.

    Returns ``fss_1``, ``fss_11``, ``fss_ratio`` (the primary readout), ``conc1pct`` (share of
    predicted mass in the top 1% of active cells) and ``moran_i`` (rook-adjacency autocorrelation
    of the prediction, threshold-free and computed without reference to truth).

    Blur predicts: ``fss_ratio`` DOWN, ``conc1pct`` DOWN, ``moran_i`` UP. The three are reported
    together because agreement is what makes the reading robust and disagreement is itself the
    finding — see the dossier's S3 falsifier.
    """
    if pred_grid.shape != truth_grid.shape:
        raise ValueError(
            f"field_sharpness: shape mismatch, pred={pred_grid.shape} truth={truth_grid.shape}. "
            "Both must be the same [H, W] grid or the comparison is meaningless."
        )
    fss_scores, _area_ratio, conc = _score_field(pred_grid, truth_grid, list(SCALES))
    f1, f11 = fss_scores[1], fss_scores[11]
    ratio = float(f1 / f11) if f11 not in (0.0, None) and np.isfinite(f11) else float("nan")
    return {
        "fss_1": float(f1),
        "fss_11": float(f11),
        "fss_ratio": ratio,
        "conc1pct": float(conc),
        "moran_i": float(moran_i(torch.from_numpy(np.ascontiguousarray(pred_grid)))),
    }
