"""Silence-vs-fade: the occurrence/magnitude decomposition and its ability to FALSIFY.

The claim under test is that during free-running the model loses cells, not size. The decomposition
rests on an exact identity, so the first thing to assert is that the identity actually holds in the
code. The second — and the one that matters more — is that the instrument can return the answer
that KILLS the claim. A statistic that can only produce the confirming reading is worthless, and
the reason this dossier exists is that the previous reading came from a statistic nobody had
checked in that direction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-09-02_silence_vs_fade_dossier"
sys.path.insert(0, str(_D / "tools"))

dc = pytest.importorskip("decompose")


def _fields(mu_t0, mu_t1, gate_t0, gate_t1):
    """Two-step [T,n_reg,H,W] / [T,H,W,n_cls] fields from flat per-cell vectors."""
    n = len(mu_t0)
    mu = np.stack([np.asarray(mu_t0), np.asarray(mu_t1)]).reshape(2, 1, 1, n).astype(np.float32)
    gate = (
        np.stack([np.asarray(gate_t0), np.asarray(gate_t1)]).reshape(2, 1, n, 1).astype(np.float32)
    )
    return mu, gate


def _by_horizon(records):
    return {r["horizon"]: r for r in records}


# ── the identity ─────────────────────────────────────────────────────────


def test_the_identity_holds_exactly():
    """`emitted_mass == occurrence * mag_gate_weighted` — the spine of the whole design."""
    rng = np.random.default_rng(0)
    mu = rng.gamma(2.0, 5.0, size=(3, 2, 6, 6)).astype(np.float32)
    gate = rng.uniform(0, 1, size=(3, 6, 6, 2)).astype(np.float32)

    for rec in dc.decompose(mu, gate):
        assert rec["emitted_mass"] == pytest.approx(
            rec["occurrence"] * rec["mag_gate_weighted"], rel=1e-5
        )


def test_occurrence_is_the_gate_field_verbatim():
    """Occurrence is an observation, not an estimate: it must be the gate's own mean."""
    mu = np.ones((1, 1, 4, 4), dtype=np.float32)
    gate = np.full((1, 4, 4, 1), 0.25, dtype=np.float32)
    assert dc.decompose(mu, gate)[0]["occurrence"] == pytest.approx(0.25)


# ── can it falsify? ──────────────────────────────────────────────────────


def test_a_genuine_fade_is_detected():
    """If the body really shrinks, `mag_unweighted` must fall. The claim must be killable."""
    base = np.linspace(1.0, 100.0, 50)
    mu, gate = _fields(base, base * 0.3, np.full(50, 0.5), np.full(50, 0.5))

    h = _by_horizon(dc.decompose(mu, gate))
    ratio = h[2]["mag_unweighted"] / h[1]["mag_unweighted"]
    assert ratio == pytest.approx(0.3, rel=1e-4), "a 70% shrink in the body went unseen"


def test_survivorship_is_detected_as_a_rise_in_tau():
    """The rival's signature: mu is UNCHANGED, but the gate collapses onto the big cells.

    A conditioned statistic is propped up by exactly this, and the dose-response is what exposes it
    — harsher selection, higher apparent magnitude. Meanwhile the unweighted mean, which conditions
    on nothing, correctly reports that nothing shrank. If this test ever fails, the tau sweep
    cannot see the mechanism it was added (05 amendment A1) to measure.
    """
    n = 100
    base = np.linspace(1.0, 100.0, n)
    collapsed = np.where(base >= base[-10], 0.9, 0.001)  # gate survives only on the top 10 cells
    mu, gate = _fields(base, base, np.full(n, 0.9), collapsed)

    h = _by_horizon(dc.decompose(mu, gate))

    assert h[2]["mag_unweighted"] == pytest.approx(h[1]["mag_unweighted"], rel=1e-6)
    # 50.5 -> 95.5 on this fixture: the conditioned statistic nearly doubles while nothing shrank.
    assert h[2]["mag_tau_0p5"] > 1.5 * h[1]["mag_tau_0p5"], "selection did not show up in tau"
    assert h[2]["occurrence"] < 0.2 * h[1]["occurrence"]


def test_a_uniform_gate_collapse_leaves_magnitude_flat():
    """The claim's own signature: cells are lost, size is not. Both magnitudes must stay flat."""
    n = 100
    base = np.linspace(1.0, 100.0, n)
    mu, gate = _fields(base, base, np.full(n, 0.8), np.full(n, 0.004))

    h = _by_horizon(dc.decompose(mu, gate))
    assert h[2]["occurrence"] == pytest.approx(0.005 * h[1]["occurrence"], rel=1e-3)
    assert h[2]["mag_unweighted"] == pytest.approx(h[1]["mag_unweighted"], rel=1e-6)
    assert h[2]["mag_gate_weighted"] == pytest.approx(h[1]["mag_gate_weighted"], rel=1e-6)


# ── the C-318 guard ──────────────────────────────────────────────────────


def test_unsupported_conditioned_statistic_is_nan_not_a_number():
    """C-318: an in-band -1.0 UNDEFINED was averaged as a magnitude, publishing `18.4 -> -0.8`.

    NaN is chosen precisely because it cannot be averaged into a plausible number by accident.
    """
    mu, gate = _fields([5.0] * 10, [5.0] * 10, np.full(10, 0.9), np.full(10, 0.01))
    h = _by_horizon(dc.decompose(mu, gate))

    assert h[2]["n_above_0p5"] == 0
    assert math.isnan(h[2]["mag_tau_0p5"])
    assert not math.isnan(h[1]["mag_tau_0p5"])


def test_sentinel_guard_rejects_an_unsupported_value_that_is_not_nan():
    """The guard must actually fire — a guard that cannot fail is decoration."""
    bad = [{"n_above_0p5": 0, "mag_tau_0p5": -1.0}]
    with pytest.raises(ValueError, match="C-318 class"):
        dc.assert_no_sentinel_survived(bad)


def test_sentinel_guard_rejects_nan_with_support():
    bad = [{"n_above_0p5": 7, "mag_tau_0p5": float("nan")}]
    with pytest.raises(ValueError, match="inconsistent"):
        dc.assert_no_sentinel_survived(bad)


# ── fail loud on bad input ───────────────────────────────────────────────


def test_non_finite_input_is_refused():
    mu = np.ones((1, 1, 2, 2), dtype=np.float32)
    mu[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dc.decompose(mu, np.full((1, 2, 2, 1), 0.5, dtype=np.float32))


def test_gate_outside_unit_interval_is_refused():
    """If the gate is not a probability, the identity fails and the readout is meaningless."""
    with pytest.raises(ValueError, match="not a probability"):
        dc.decompose(
            np.ones((1, 1, 2, 2), dtype=np.float32),
            np.full((1, 2, 2, 1), 1.5, dtype=np.float32),
        )


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="disagree on T/H/W"):
        dc.decompose(
            np.ones((2, 1, 3, 3), dtype=np.float32),
            np.full((2, 4, 4, 1), 0.5, dtype=np.float32),
        )


def test_tau_selection_matches_the_production_threshold_gate():
    """`compose_samples` keeps a cell where `gate >= tau`. The sweep must use the same boundary.

    Found by mutation: swapping `>=` for `>` survived every other test, because a gate landing
    exactly on tau is measure-zero in random fixtures. It is not measure-zero in intent — the sweep
    exists to model the production `threshold_gate` composition, and a sweep using a different
    comparison is modelling a composition the repo does not implement.
    """
    n = 4
    mu = np.array([10.0, 20.0, 30.0, 40.0])
    gate = np.array([0.5, 0.5, 0.4, 0.4])  # two cells exactly ON tau=0.5
    mu_f, gate_f = _fields(mu, mu, gate, gate)

    rec = _by_horizon(dc.decompose(mu_f, gate_f))[1]
    assert rec["n_above_0p5"] == 2, (
        "cells exactly on tau must be KEPT, as compose_samples keeps them"
    )
    assert rec["mag_tau_0p5"] == pytest.approx(15.0)
