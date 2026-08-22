"""`ap_diff_origin_block_ci` must resample origins ONCE per replicate and feed BOTH ARMS.

The companion to `ap_ratio_origin_block_ci`, pairing across **arms** instead of **horizons**. Two
arms of the same model on the same origins move together; an unpaired interval built from two
independent bootstraps discards that correlation and reports a width dominated by variance the
comparison never faces. That is the #281 construction, and the first test pins it.

Driven through a synthetic support so they always run — the real cubes are 2.5 GB each and are
deleted after scoring by design.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.ap_block_bootstrap as abb

_ORIGINS = (101, 102, 103, 104, 105)
_UNITS = tuple(range(24))
_H = 18


def _arm(strength: float, n_origins: int = 5, seed: int = 7, units=_UNITS, origins=None):
    """A support whose gate tracks truth with the given `strength`. Truth is seed-locked so two
    arms built at different strengths still share labels — which is what makes them pairable."""
    origins = list(origins if origins is not None else _ORIGINS[:n_origins])
    # truth is seed-locked and covers every origin either arm might use, so two arms built at
    # different strengths still share labels — which is what makes them pairable at all.
    truth_rng = np.random.default_rng(1234)
    truth = {
        (o, u): float(truth_rng.random() < 0.25)
        for o in sorted(set(_ORIGINS) | set(origins))
        for u in _UNITS
    }
    rng = np.random.default_rng(seed)
    support = [(o, u) for o in origins for u in units]
    g, tmap = {}, {}
    for o in origins:
        for u in units:
            y = truth[(o, u)]
            tmap[(o + _H - 1, u)] = y
            p = y * strength + rng.random() * (1 - strength)
            g[(o, _H, u)] = (np.full(4, y), float(p))
    by_origin = {o: [(o, u) for u in units] for o in origins}
    return g, support, origins, by_origin, tmap, True


def _patch_pair(monkeypatch, bundle_a, bundle_b):
    calls = {"n": 0}

    def fake(**kw):
        calls["n"] += 1
        return bundle_a if kw["pred_dir"] == "A" else bundle_b

    monkeypatch.setattr(abb, "_load_indexed", fake)
    return calls


def _args(**over):
    base = dict(
        pred_dir_a="A",
        pred_dir_b="B",
        truth_parquet="unused",
        target="sb",
        h=_H,
        n_boot=40,
        seed=0,
    )
    base.update(over)
    return base


# ------------------------------------------------------------------ the property that matters


def test_both_arms_see_the_SAME_resampled_cells_every_replicate(monkeypatch):
    """THE reason this function exists. If each arm drew its own origins, the difference would
    carry variance the paired comparison never faces and the interval would be too wide."""
    seen: list[list] = []
    real_ap_fn = abb._ap_fn

    def spy(g, tmap, has_gate, h):
        f = real_ap_fn(g, tmap, has_gate, h)

        def wrapped(cells):
            seen.append(list(cells))
            return f(cells)

        return wrapped

    monkeypatch.setattr(abb, "_ap_fn", spy)
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3))
    abb.ap_diff_origin_block_ci(**_args(n_boot=5))
    # 2 point estimates + 5 replicates x 2 arms
    boot = seen[2:]
    assert len(boot) == 10
    for i in range(0, 10, 2):
        assert boot[i] == boot[i + 1], "the two arms were scored on DIFFERENT resampled cells"


def test_point_estimate_is_exactly_the_difference_of_the_two_point_aps(monkeypatch):
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3))
    r = abb.ap_diff_origin_block_ci(**_args())
    assert r["diff"] == pytest.approx(r["ap_a"] - r["ap_b"])


def test_interval_brackets_the_point_and_mde_is_the_half_width(monkeypatch):
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3))
    r = abb.ap_diff_origin_block_ci(**_args())
    assert r["lo"] <= r["diff"] <= r["hi"]
    assert r["mde"] == pytest.approx((r["hi"] - r["lo"]) / 2)


def test_pairing_is_tighter_than_scoring_the_arms_independently(monkeypatch):
    """The #281 claim, measured. Correlated arms paired must give a narrower interval than the
    same arms bootstrapped independently and differenced."""
    a, b = _arm(0.9), _arm(0.85)  # highly correlated: same truth, similar strength
    _patch_pair(monkeypatch, a, b)
    paired = abb.ap_diff_origin_block_ci(**_args(n_boot=200))

    rng = np.random.default_rng(0)
    ap_a = abb._ap_fn(a[0], a[4], True, _H)
    ap_b = abb._ap_fn(b[0], b[4], True, _H)
    unpaired = np.empty(200)
    for i in range(200):
        pa = rng.choice(a[2], size=len(a[2]), replace=True)
        pb = rng.choice(
            b[2], size=len(b[2]), replace=True
        )  # SEPARATE draw = the wrong construction
        unpaired[i] = ap_a([c for o in pa for c in a[3][o]]) - ap_b(
            [c for o in pb for c in b[3][o]]
        )
    unpaired_mde = (np.percentile(unpaired, 95) - np.percentile(unpaired, 5)) / 2
    assert paired["mde"] < unpaired_mde, (
        f"paired mde {paired['mde']:.4f} not tighter than unpaired {unpaired_mde:.4f}"
    )


def test_deterministic_at_a_fixed_seed(monkeypatch):
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3))
    a = abb.ap_diff_origin_block_ci(**_args())
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3))
    b = abb.ap_diff_origin_block_ci(**_args())
    assert a == b


def test_mismatched_support_raises_rather_than_scoring_different_cells(monkeypatch):
    """Two arms on different cell sets are not pairable. Refuse — do not silently intersect."""
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3, units=tuple(range(20))))
    with pytest.raises(ValueError, match="do not share a support"):
        abb.ap_diff_origin_block_ci(**_args())


def test_mismatched_origins_raises(monkeypatch):
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3, origins=(101, 102, 103, 104, 199)))
    with pytest.raises(ValueError, match="support|origin set"):
        abb.ap_diff_origin_block_ci(**_args())


def test_fewer_than_three_origins_raises(monkeypatch):
    """Inherited from _load_indexed: a block bootstrap over 2 units is meaningless."""

    def fake(**_kw):
        raise ValueError("ap_diff_origin_block_ci: fewer than 3 origins")

    monkeypatch.setattr(abb, "_load_indexed", fake)
    with pytest.raises(ValueError, match="3 origins"):
        abb.ap_diff_origin_block_ci(**_args())


def test_reports_h_and_counts_so_a_result_cannot_be_misread(monkeypatch):
    _patch_pair(monkeypatch, _arm(0.9), _arm(0.3))
    r = abb.ap_diff_origin_block_ci(**_args())
    assert r["h"] == float(_H)
    assert r["n_origins"] == 5.0
    assert r["n_support"] == 5.0 * len(_UNITS)
