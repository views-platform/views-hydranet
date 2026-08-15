"""Activation-aware metrics for the occurrence/magnitude split (#258 — the rollout collapse).

``crps_all`` / MCR go BLIND when the field is ~99% zero: a forecast that almost never fires still
scores well because the zeros dominate the mean. That blindness is what let the collapse hide
(views-hydranet#258 — the models under-activate 4–115× yet MCR≈1). These metrics look at the thing
that actually broke: WHERE and HOW OFTEN the composed forecast says a conflict occurs, and how much
magnitude it dumps on cells it wrongly fires.

Pure numpy, no torch, no I/O — so the degenerate-forecast red-team can unit-test that they SEE the
defects (collapse, bloom, imprecise-gate-with-full-magnitude) before they are trusted on cubes.

  * :func:`activation_frequency` — ``P(pred>0)`` vs ``P(truth>0)``: ``act_ratio << 1`` = collapse,
    ``>> 1`` = bloom. The single number the collapse investigation needed and MCR could not give.
  * :func:`topk_occurrence` — at the operating point "fire exactly as many cells as truly fire"
    (top-k by the occurrence score, ``k = #true positives`` — threshold-free):
      - ``precision_at_k``   : fraction of fired cells that are truly positive (the gate's skill).
      - ``mag_on_false_pos`` : mean composed magnitude on the FIRED-BUT-ZERO cells — the named #258
                               risk (truncating the body gives the gate's false positives FULL
                               magnitude). This is the falsifier ``crps_all`` cannot see.
      - ``mag_on_true_pos``  : mean composed magnitude on the fired-and-correct cells (contrast).
"""

from __future__ import annotations

import numpy as np


def activation_frequency(cs: "np.ndarray", truth: "np.ndarray") -> dict:
    """``P(pred>0)`` (over all cell×sample draws of the composed cube ``cs``) vs ``P(truth>0)``.

    ``act_ratio = act_pred / act_true`` is the collapse/bloom dial: ~1 is calibrated occurrence,
    ``<< 1`` is the #258 collapse, ``>> 1`` is the self-zeroed/ZINB bloom. ``act_true`` is the base
    rate; ``nan`` ratio if the truth window has no positives (undefined, not zero)."""
    cs = np.asarray(cs, dtype=float)
    truth = np.asarray(truth, dtype=float)
    act_pred = float((cs > 0).mean()) if cs.size else float("nan")
    act_true = float((truth > 0).mean()) if truth.size else float("nan")
    ratio = act_pred / act_true if act_true and act_true > 0 else float("nan")
    return {"act_pred": act_pred, "act_true": act_true, "act_ratio": ratio}


def topk_occurrence(score: "np.ndarray", magnitude: "np.ndarray", truth: "np.ndarray") -> dict:
    """Fire the top-k cells by ``score`` (``k = #true positives``) and measure precision + the
    magnitude that lands on the false positives.

    ``score`` — per-cell occurrence signal (the gate probability). ``magnitude`` — per-cell
    composed mean forecast (``E[y]``). ``truth`` — per-cell truth. Threshold-free: firing exactly
    ``k`` cells puts precision and false-positive magnitude on the SAME operating point. Ties in
    ``score`` break deterministically (stable sort) — a constant score therefore scores near the
    base rate, correctly reporting "no discrimination". ``k = 0`` -> all-``nan``."""
    score = np.asarray(score, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    truth = np.asarray(truth, dtype=float)
    is_pos = truth > 0
    k = int(is_pos.sum())
    if k == 0 or score.size == 0:
        return {
            "precision_at_k": float("nan"),
            "mag_on_false_pos": float("nan"),
            "mag_on_true_pos": float("nan"),
            "k": k,
            "n_false_pos": 0,
        }
    fired = np.argsort(score, kind="stable")[::-1][:k]  # the k highest-scoring cells
    fired_pos = is_pos[fired]
    fp = fired[~fired_pos]
    tp = fired[fired_pos]
    return {
        "precision_at_k": float(fired_pos.mean()),
        "mag_on_false_pos": float(magnitude[fp].mean()) if fp.size else 0.0,
        "mag_on_true_pos": float(magnitude[tp].mean()) if tp.size else float("nan"),
        "k": k,
        "n_false_pos": int(fp.size),
    }
