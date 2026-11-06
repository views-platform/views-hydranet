"""The decomposition's structural gates must FIRE, not merely pass.

⭐⭐ ``G-C1`` is the one that matters. It exists because on 2026-11-05 a composite
``mean(obs)/mean(pred)`` was reported as a magnitude defect and withdrawn the next day. The gate
makes that expression unwritable in its own file — so the test that matters is the one proving it
REFUSES a file that contains it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-06_occurrence_vs_magnitude/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "occurrence_vs_magnitude.py").exists(),
    reason="the occurrence-vs-magnitude dossier tools are not present in this tree",
)

if (_TOOLS / "occurrence_vs_magnitude.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-05_conditional_magnitude/tools"))
    import occurrence_vs_magnitude as dc  # noqa: E402


# ── G-C1 · the instrument cannot contain the composite it replaces ───────────────────────────


def test_gate_c1_passes_on_the_real_instrument():
    ok, bad = dc.gate_c1_no_composite()
    assert ok, f"the shipped instrument contains a composite: {bad}"


def test_gate_c1_FIRES_on_a_file_containing_the_WITHDRAWN_composite(tmp_path, monkeypatch):
    """⛔⛔ The gate's entire purpose. This is the exact expression that was withdrawn."""
    fake = tmp_path / "occurrence_vs_magnitude.py"
    fake.write_text(
        "import numpy as np\n\n\ndef headline(obs, pred):\n    return obs.mean() / pred.mean()\n"
    )
    monkeypatch.setattr(dc, "_HERE", fake)
    ok, bad = dc.gate_c1_no_composite()
    assert not ok, "G-C1 accepted a file computing mean(obs)/mean(pred)"
    assert any("obs.mean() / pred.mean()" in b for b in bad), bad


def test_gate_c1_FIRES_on_the_composite_under_DIFFERENT_variable_names(tmp_path, monkeypatch):
    """⛔ Renaming the variables must not get the composite past the gate."""
    fake = tmp_path / "occurrence_vs_magnitude.py"
    fake.write_text("def h(observed, predicted):\n    return observed.mean() / predicted.mean()\n")
    monkeypatch.setattr(dc, "_HERE", fake)
    assert dc.gate_c1_no_composite()[0] is False


def test_gate_c1_ALLOWS_the_two_legitimate_component_ratios(tmp_path, monkeypatch):
    """⭐ Non-vacuity, and it is the sharp edge of the gate.

    ``rate_obs / p_hat`` and ``obs_pos / m_hat`` are the two component errors this instrument is
    FOR. If the gate rejected them it would reject the whole measurement, and a gate that rejects
    everything is as useless as one that rejects nothing.
    """
    fake = tmp_path / "occurrence_vs_magnitude.py"
    fake.write_text(
        "def occ(rate_obs, p_hat):\n    return rate_obs.mean() / p_hat.mean()\n\n\n"
        "def mag(obs_pos, m_hat):\n    return obs_pos.mean() / m_hat.mean()\n"
    )
    monkeypatch.setattr(dc, "_HERE", fake)
    ok, bad = dc.gate_c1_no_composite()
    assert ok, f"G-C1 rejected the component ratios it exists to permit: {bad}"


def test_gate_c1_ignores_a_division_that_is_not_of_two_means(tmp_path, monkeypatch):
    fake = tmp_path / "occurrence_vs_magnitude.py"
    fake.write_text("def f(obs, pred):\n    return obs.sum() / pred.sum()\n")
    monkeypatch.setattr(dc, "_HERE", fake)
    assert dc.gate_c1_no_composite()[0] is True


def test_gate_c1_FIRES_on_an_UNDECLARED_ratio_nobody_anticipated(tmp_path, monkeypatch):
    """⭐⭐ Why it is an allowlist. A composite under names no blocklist would have held.

    ``severity`` over ``intensity`` means nothing to a list of suspicious words, and that is
    exactly the kind of statistic that produced the withdrawn number under a different name.
    """
    fake = tmp_path / "occurrence_vs_magnitude.py"
    fake.write_text("def f(severity, intensity):\n    return severity.mean() / intensity.mean()\n")
    monkeypatch.setattr(dc, "_HERE", fake)
    ok, bad = dc.gate_c1_no_composite()
    assert not ok, "an allowlist must refuse a ratio it has never heard of"
    assert "severity" in bad[0]


def test_the_allowlist_is_exactly_the_two_declared_component_ratios():
    """⛔ Adding a third pair here is a change to the pre-registration, so pin the two."""
    assert dc.PERMITTED_MEAN_RATIOS == (("rate_obs", "p_hat"), ("obs_pos", "m_hat"))


def test_root_resolves_the_shapes_the_gate_actually_meets():
    import ast as _ast

    for src, want in (
        ("obs.mean()", "obs"),
        ("o_open[msk].mean()", "o_open"),
        ("o_open[msk][pos].mean()", "o_open"),
        ("np.asarray(x).mean()", "np"),
        ("(a + b).mean()", ""),
    ):
        assert dc._root(_ast.parse(src, mode="eval").body) == want, src


# ── the decomposition itself, and G-C4 ───────────────────────────────────────────────────────


def test_decompose_splits_the_mean_draw_exactly():
    draws = np.array([[0.0, 0.0, 4.0, 8.0], [1.0, 1.0, 1.0, 1.0]])
    p_hat, m_hat, open_m = dc.decompose(draws)
    assert np.allclose(p_hat, [0.5, 1.0])
    assert np.allclose(m_hat, [6.0, 1.0])
    assert open_m.all()
    assert np.allclose(p_hat * m_hat, draws.mean(axis=1)), "the identity G-C4 checks must hold"


def test_decompose_marks_a_fully_degenerate_cell_CLOSED_rather_than_dividing_by_zero():
    draws = np.zeros((2, 16))
    draws[0, 3] = 5.0
    p_hat, m_hat, open_m = dc.decompose(draws)
    assert open_m.tolist() == [True, False]
    assert m_hat[1] == 0.0, "a closed cell must not produce a nan or an inf"
    assert np.isfinite(m_hat).all()


def test_gate_c4_passes_on_real_shaped_draws():
    rng = np.random.default_rng(0)
    draws = np.where(rng.random((500, 16)) < 0.3, rng.lognormal(1.0, 1.2, (500, 16)), 0.0)
    ok, worst = dc.gate_c4_reconstructs(draws)
    assert ok and worst <= dc.RECON_TOL


def test_gate_c4_FIRES_when_the_decomposition_IS_WRONG(monkeypatch):
    """⛔ Substitute a p_hat that is off by a draw; the identity must break and be caught."""
    real = dc.decompose

    def broken(draws):
        p, m, o = real(draws)
        return p + 1.0 / draws.shape[1], m, o

    monkeypatch.setattr(dc, "decompose", broken)
    draws = np.zeros((4, 16))
    draws[:, :4] = 3.0
    ok, worst = dc.gate_c4_reconstructs(draws)
    assert not ok and worst > dc.RECON_TOL


def test_p_hat_is_quantised_at_one_over_the_draw_count():
    """⚠️ The caveat the output prints: with 16 draws, p_hat can only take 17 values."""
    rng = np.random.default_rng(1)
    draws = np.where(rng.random((200, 16)) < 0.5, 1.0, 0.0)
    p_hat, _m, _o = dc.decompose(draws)
    assert set(np.round(p_hat * 16).astype(int)).issubset(set(range(17)))
    assert dc.P_HAT_QUANTUM == 1 / 16


# ── the aggregator never emits a single-seed number ──────────────────────────────────────────


def test_agg_returns_mean_sd_and_range_and_the_sd_is_the_population_one():
    mean, sd, rng = dc._agg([1.0, 2.0, 3.0])
    assert mean == 2.0 and rng == 2.0
    assert abs(sd - (2 / 3) ** 0.5) < 1e-12, "a sample sd gives 1.0; the docstring says population"


def test_the_prereg_constants_are_the_ones_the_docstring_declares():
    assert dc.B_SWEEP == (4, 8, 16)
    assert dc.N_SEEDS_REQUIRED == 5
    assert dc.F13_FACTOR == 1.25
    assert dc.RECON_TOL == 1e-9
    assert dc.P_HAT_QUANTUM == 1 / 16


def test_this_module_name_is_not_one_that_already_exists_elsewhere():
    """⛔⛔ The collision this file caused on 2026-11-06, pinned so it cannot come back.

    Two dossiers already shipped a module called ``decompose.py``. Naming a third one that broke
    12 pre-existing tests in ``test_silence_vs_fade_decompose.py``, because every dossier test
    inserts its own ``tools`` dir on ``sys.path`` and pytest shares ``sys.modules`` for the whole
    session -- so whichever test imported first won the name.
    """
    stem = "occurrence_vs_magnitude"
    assert (_TOOLS / f"{stem}.py").exists()
    clashes = [
        p
        for p in (_REPO / "reports").glob(f"*/tools/{stem}.py")
        if p.resolve() != (_TOOLS / f"{stem}.py").resolve()
    ]
    assert not clashes, f"the module name is no longer unique across dossiers: {clashes}"
    assert not (_TOOLS / "decompose.py").exists(), "the colliding name is back"
