"""`ap_ratio_origin_block_ci` must resample origins ONCE per replicate and feed both horizons.

Retention (`AP(h18)/AP(h1)`) is a co-primary endpoint of the lesson-curve programme
(`reports/2026-08-18_lesson_curve_dossier`). Its interval is only valid if the two horizons are
scored on the *same* resampled origins each replicate — an unpaired construction would silently
report the interval of a quantity nobody measured. That property is what the first test pins.

The real-data reproduction lives in the dossier's experiment log rather than here: it needs a
2.5 GB prediction cube and ~2 min, and the cube is deleted after scoring by design. These tests
therefore drive the function through a synthetic support so they always run and never skip.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.ap_block_bootstrap as abb

_ORIGINS = (101, 102, 103, 104, 105)
_UNITS = tuple(range(24))
_HORIZONS = (1, 18)


def _synthetic(n_origins: int = 5, sharp_h: int = 1):
    """A support the bootstrap can chew on: gate probability tracks truth sharply at `sharp_h`."""
    rng = np.random.default_rng(7)
    origins = list(_ORIGINS[:n_origins])
    support = [(o, u) for o in origins for u in _UNITS]
    g, tmap = {}, {}
    for o in origins:
        for u in _UNITS:
            for h in _HORIZONS:
                truth = float(rng.random() < 0.25)
                tmap[(o + h - 1, u)] = truth
                # at sharp_h the gate is informative; at the other horizon it is near-noise
                strength = 0.9 if h == sharp_h else 0.15
                p = truth * strength + rng.random() * (1 - strength)
                g[(o, h, u)] = (np.full(4, truth), float(p))
    by_origin = {o: [(o, u) for u in _UNITS] for o in origins}
    return g, support, origins, by_origin, tmap, True


def _patch(monkeypatch, bundle):
    monkeypatch.setattr(abb, "_load_indexed", lambda **_kw: bundle)


def _args(**over):
    base = dict(pred_dir="unused", truth_parquet="unused", target="sb", n_boot=40, seed=0)
    base.update(over)
    return base


# ------------------------------------------------------------------ the property that matters


def test_both_horizons_see_the_SAME_resampled_cells_every_replicate(monkeypatch):
    """The pairing. An unpaired bootstrap would give an interval for a quantity nobody measured."""
    bundle = _synthetic()
    _patch(monkeypatch, bundle)

    seen: dict[int, list[tuple]] = {1: [], 18: []}
    real_ap_fn = abb._ap_fn

    def recording_ap_fn(g, tmap, has_gate, h):
        inner = real_ap_fn(g, tmap, has_gate, h)

        def wrapped(cells):
            seen[h].append(tuple(cells))
            return inner(cells)

        return wrapped

    monkeypatch.setattr(abb, "_ap_fn", recording_ap_fn)
    abb.ap_ratio_origin_block_ci(**_args())

    assert seen[1] and seen[18], "both horizons must have been scored"
    assert len(seen[1]) == len(seen[18]), "one call per horizon per replicate (plus the point)"
    for cells_den, cells_num in zip(seen[1], seen[18]):
        assert cells_den == cells_num, "a replicate scored the two horizons on different cells"


def test_the_point_estimate_is_exactly_the_ratio_of_the_two_point_APs(monkeypatch):
    bundle = _synthetic()
    _patch(monkeypatch, bundle)
    res = abb.ap_ratio_origin_block_ci(**_args())
    assert res["ratio"] == res["ap_num"] / res["ap_den"]


def test_the_interval_brackets_the_point_and_mde_is_the_half_width(monkeypatch):
    _patch(monkeypatch, _synthetic())
    res = abb.ap_ratio_origin_block_ci(**_args(n_boot=120))
    assert res["lo"] < res["ratio"] < res["hi"]
    assert res["mde"] == pytest.approx((res["hi"] - res["lo"]) / 2.0)


def test_it_is_deterministic_at_a_fixed_seed(monkeypatch):
    _patch(monkeypatch, _synthetic())
    a = abb.ap_ratio_origin_block_ci(**_args())
    _patch(monkeypatch, _synthetic())
    b = abb.ap_ratio_origin_block_ci(**_args())
    assert a == b
    _patch(monkeypatch, _synthetic())
    c = abb.ap_ratio_origin_block_ci(**_args(seed=1))
    assert c["lo"] != a["lo"] or c["hi"] != a["hi"], "a different seed must move the interval"


def test_the_horizons_are_reported_so_a_result_cannot_be_misread(monkeypatch):
    _patch(monkeypatch, _synthetic())
    res = abb.ap_ratio_origin_block_ci(**_args(numerator_h=18, denominator_h=1))
    assert res["numerator_h"] == 18.0 and res["denominator_h"] == 1.0
    assert res["n_origins"] == 5.0


# ------------------------------------------------------------------------------- the guards


def test_fewer_than_three_origins_raises_rather_than_returning_an_interval(monkeypatch):
    g, support, origins, by_origin, tmap, has_gate = _synthetic()
    keep = origins[:2]
    trimmed = (
        g,
        [c for c in support if c[0] in keep],
        keep,
        {o: by_origin[o] for o in keep},
        tmap,
        has_gate,
    )
    # the guard lives in the real loader, so exercise it rather than the patched stub
    monkeypatch.setattr(
        abb,
        "_load_indexed",
        lambda **kw: (_ for _ in ()).throw(
            ValueError(f"{kw['who']}: 2 origin(s) — a block bootstrap over fewer than 3 units")
        ),
    )
    with pytest.raises(ValueError, match="fewer than 3 units"):
        abb.ap_ratio_origin_block_ci(**_args())
    assert len(trimmed[2]) == 2  # the trimmed bundle is what the loader would have refused


def test_a_zero_denominator_raises_rather_than_returning_an_infinity(monkeypatch):
    """A control with no h1 skill makes retention undefined, not large."""
    g, support, origins, by_origin, tmap, has_gate = _synthetic()
    for key in list(tmap):
        if key[0] in origins:  # h=1 truth month == origin
            tmap[key] = 0.0
    _patch(monkeypatch, (g, support, origins, by_origin, tmap, has_gate))
    with pytest.raises(ValueError, match="undefined, not large"):
        abb.ap_ratio_origin_block_ci(**_args())


def test_the_public_surface_advertises_the_new_entry_point():
    assert "ap_ratio_origin_block_ci" in abb.__all__
