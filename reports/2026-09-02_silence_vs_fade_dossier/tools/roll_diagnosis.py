"""Is the rolled arm's forecast the clamp's forecast MOVED, or is it rubbish?

EXP-3 left this open, and the obvious test — roll the truth back and re-score — is not valid here:
`torch.roll` wraps on a torus while only 13,110 of the 32,400 grid cells are study cells, so
rolling truth wraps land onto ocean; the model's INPUT was never rolled, only its memory, so a
clean displacement was never guaranteed; and AP scores study cells only, so a displaced forecast
could score zero merely by landing outside the mask.

So compare the FIELDS instead. `torch.roll` is circular, so circular cross-correlation is the exact
matched operation — no windowing, no edge effects, no mask dependence, no truth involved:

* peak at offset (90, 90), high correlation  -> the forecast is INTACT and DISPLACED
* peak at offset (0, 0), high correlation    -> the rolled memory was IGNORED; output follows input
* no sharp peak / low correlation anywhere   -> the forecast is BROKEN, not moved

The three outcomes are distinguishable, which is what the AP-based test could not manage.
"""

from __future__ import annotations

import numpy as np


def circular_xcorr_peak(a: np.ndarray, b: np.ndarray) -> tuple[tuple[int, int], float, float]:
    """Peak of the circular cross-correlation of two 2-D fields.

    Returns ``((dy, dx), peak_r, r_at_zero)`` where ``(dy, dx)`` is the shift that best maps ``a``
    onto ``b`` (i.e. ``np.roll(a, (dy, dx), (0, 1)) ~ b``), and the two values are Pearson
    correlations. Both fields are mean-centred and norm-scaled, so the values are comparable
    across horizons where the field's overall level changes by orders of magnitude.
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"need two 2-D fields of equal shape; got {a.shape} and {b.shape}")
    x = a.astype(np.float64) - a.mean()
    y = b.astype(np.float64) - b.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        # A constant field has no spatial structure to match, so "where does it best align" is not
        # a question with an answer. Refuse rather than return an arbitrary argmax of noise.
        raise ValueError("a field is constant; circular cross-correlation is undefined for it")
    corr = np.fft.irfft2(np.fft.rfft2(y) * np.conj(np.fft.rfft2(x)), s=x.shape) / (nx * ny)
    idx = int(np.argmax(corr))
    dy, dx = divmod(idx, corr.shape[1])
    return (int(dy), int(dx)), float(corr.max()), float(corr[0, 0])


def field(npz_path, *, horizon: int, target: int = 0, which: str = "gate") -> np.ndarray:
    """One 2-D field from a body-mean dump. ``horizon`` is 1-based."""
    z = np.load(npz_path)
    if which == "gate":
        return z["gate"][horizon - 1, :, :, target]
    if which == "mu":
        return z["mu"][horizon - 1, target]
    if which == "emitted":  # gate x mu, the composed point forecast
        return z["gate"][horizon - 1, :, :, target] * z["mu"][horizon - 1, target]
    raise ValueError(f"which must be gate|mu|emitted, got {which!r}")
