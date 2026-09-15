"""The sharpness instrument must be demonstrated fit for purpose BEFORE it is pointed at real data.

The diagnostic these support asks whether the model's emitted field becomes spatially blurrier
across a 36-step rollout. A metric that cannot separate a deliberately blurred field from its own
source would answer that question confidently and wrongly, which is worse than not asking it.

**These tests already changed the design.** The dossier's first plan made ``fss_ratio``
(``fss@1 / fss@11``) the primary readout, reasoning that blur costs the fine scale
disproportionately. It does — but displacement costs it far more, so the ratio detects *wrong
place*, not *blurry*. That was found here, on synthetic fields, before any GPU time was spent.

Three failure modes are simulated, and the instrument must tell them apart:

* **blur** — the field's own structure is smoothed
* **displacement** — identical structure, wrong location
* **thinning** — fewer events, same structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from field_sharpness import field_sharpness  # noqa: E402


def _truth(seed: int = 0, n_clusters: int = 20, size: int = 90) -> np.ndarray:
    """A sparse clustered field, shaped like the real grid: mostly zero, a few 5x5 blobs."""
    rng = np.random.default_rng(seed)
    t = np.zeros((size, size))
    for _ in range(n_clusters):
        r, c = rng.integers(6, size - 6, 2)
        t[r - 2 : r + 3, c - 2 : c + 3] += rng.random((5, 5)) * 3
    return t


def _displaced(t: np.ndarray, shift: int) -> np.ndarray:
    """Same field, wrong place — sharpness identical by construction."""
    return np.roll(np.roll(t, shift, axis=0), shift, axis=1)


def test_a_field_against_itself_is_perfect():
    """The floor case. If this is not 1.0 the instrument is broken before anything else matters."""
    t = _truth()
    s = field_sharpness(t.copy(), t)
    assert s["fss_1"] == pytest.approx(1.0)
    assert s["fss_11"] == pytest.approx(1.0)
    assert s["fss_ratio"] == pytest.approx(1.0)


def test_morans_i_rises_monotonically_with_blur():
    """PRIMARY detector. Blur raises spatial autocorrelation; more blur raises it more."""
    t = _truth()
    vals = [field_sharpness(t.copy(), t)["moran_i"]]
    for sigma in (1.0, 2.0, 4.0):
        vals.append(field_sharpness(gaussian_filter(t, sigma), t)["moran_i"])
    assert all(b > a for a, b in zip(vals, vals[1:])), (
        f"Moran's I is not monotone in blur: {vals}. The primary detector does not detect."
    )
    assert vals[-1] > 0.95, f"heavy blur should drive Moran's I near 1, got {vals[-1]:.4f}"


def test_morans_i_is_blind_to_displacement():
    """The property that makes it a BLUR detector rather than a wrongness detector.

    Displacement leaves the field's own structure untouched, so an intrinsic statistic must not
    move. If this fails, Moran's I is measuring agreement with truth and cannot separate the two.
    """
    t = _truth()
    base = field_sharpness(t.copy(), t)["moran_i"]
    for shift in (3, 8, 20):
        moved = field_sharpness(_displaced(t, shift), t)["moran_i"]
        assert moved == pytest.approx(base, abs=0.05), (
            f"shift={shift} moved Moran's I {base:.4f} -> {moved:.4f}; an intrinsic statistic "
            "must be unmoved by relocating an unchanged field."
        )


def test_concentration_moves_under_blur_and_not_under_displacement():
    """Independent second detector, on the direction MEASURED rather than the one predicted.

    The dossier's first draft predicted conc1pct would FALL under blur. It rises: blur spreads mass
    into many near-zero cells, and the top 1% of that larger active set holds a larger share. The
    direction is pinned here so the write-up cannot quietly adopt whichever sign fits.
    """
    t = _truth()
    base = field_sharpness(t.copy(), t)["conc1pct"]
    blurred = field_sharpness(gaussian_filter(t, 2.0), t)["conc1pct"]
    assert blurred > base * 1.5, (
        f"blur should raise conc1pct well above {base:.4f}, got {blurred:.4f}"
    )
    for shift in (3, 8, 20):
        moved = field_sharpness(_displaced(t, shift), t)["conc1pct"]
        assert moved == pytest.approx(base, abs=1e-6), (
            f"shift={shift}: conc1pct moved {base:.4f} -> {moved:.4f}; it must be intrinsic."
        )


def test_the_fss_ratio_is_NOT_a_blur_detector():
    """CHARACTERISATION of the finding that changed the design.

    ``fss_ratio`` falls under blur, so it tempts as a sharpness readout. It falls FURTHER under
    displacement — a field of identical sharpness in the wrong place. Anything reading a declining
    ratio as evidence of blur would be reporting displacement.

    This test exists so that reading cannot be reintroduced without deliberately breaking it.
    """
    t = _truth()
    blurred = field_sharpness(gaussian_filter(t, 4.0), t)["fss_ratio"]
    displaced = field_sharpness(_displaced(t, 3), t)["fss_ratio"]
    assert displaced < blurred, (
        f"displacement ratio {displaced:.3f} is not below heavy-blur {blurred:.3f} — the "
        "confound this test documents has changed, so the primary readout should be revisited."
    )


def test_thinning_is_separable_from_blur():
    """The third failure mode: fewer events, same structure. Must not masquerade as blur.

    Thinning LOWERS Moran's I while blur raises it, so the two are separable by sign — which is
    what lets a real result distinguish 'the model went quiet' from 'the model went blurry'.
    """
    t = _truth()
    rng = np.random.default_rng(7)
    thin = t * (rng.random(t.shape) < 0.5)
    base = field_sharpness(t.copy(), t)["moran_i"]
    thinned = field_sharpness(thin, t)["moran_i"]
    blurred = field_sharpness(gaussian_filter(t, 2.0), t)["moran_i"]
    assert thinned < base < blurred, (
        f"thinning {thinned:.4f} / base {base:.4f} / blur {blurred:.4f} — thinning and blur must "
        "move Moran's I in OPPOSITE directions or the diagnostic cannot tell them apart."
    )


def test_shape_mismatch_fails_loud():
    """A silent broadcast between differently-shaped grids would compare unrelated cells."""
    with pytest.raises(ValueError, match="shape mismatch"):
        field_sharpness(np.zeros((10, 10)), np.zeros((10, 12)))
