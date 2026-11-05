"""The noise-floor probe's preconditions must FIRE, not merely pass.

The load-bearing ones are ``anchor_for`` (which exists only because ``e1_data.anchor_ap``
hardcodes one seed and is lru_cached, so reusing it would pass G-N0 five times on seed 42's
anchor) and ``_as_arm`` (which mutates a module global and must restore it on every path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-05_conditional_magnitude/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "profile_noise_floor.py").exists(),
    reason="the conditional-magnitude dossier tools are not present in this tree",
)

if (_TOOLS / "profile_noise_floor.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
    import e1_data  # noqa: E402
    import profile_noise_floor as pnf  # noqa: E402


# ── anchor_for · the reason this script does not reuse e1_data.anchor_ap ──────────────────────


def test_anchor_for_returns_a_DIFFERENT_value_per_seed():
    """⭐⭐ G-N2's substance. If these coincided, G-N0 would pass by loading one arm five times."""
    vals = {w: pnf.anchor_for(w) for _a, (w, _s) in pnf.SEED_ARMS.items()}
    assert len(set(vals.values())) == len(vals), f"anchors are not distinct: {vals}"
    assert vals["fortytwo"] == 0.5667056384496834


def test_anchor_for_does_NOT_silently_return_seed_42s_anchor():
    """⛔ The exact trap: e1_data.anchor_ap hardcodes 'fortytwo' AND is lru_cached."""
    assert pnf.anchor_for("fortythree") != e1_data.anchor_ap()
    assert pnf.anchor_for("fortytwo") == e1_data.anchor_ap()


def test_anchor_for_REFUSES_a_seed_with_no_committed_anchor():
    """⛔ Seed 46 is excluded for this reason; the refusal must be reportable.

    A bare KeyError is not a reportable refusal, which is the whole gate contract.
    """
    with pytest.raises(SystemExit, match="has no anchor"):
        pnf.anchor_for("fortysix")


def test_anchor_for_REFUSES_an_anchor_computed_on_a_DIFFERENT_POPULATION(tmp_path, monkeypatch):
    """⛔ An anchor from another n or another origin count compares two different things."""
    rows = json.loads(pnf._ANCHORS.read_text())
    for r in rows:
        if r.get("seed") == "fortytwo" and r.get("h") == 1:
            r["n"] = 13_110 * 384  # the retired Africa+ME extent
    fake = tmp_path / "exp03_pairs.json"
    fake.write_text(json.dumps(rows))
    monkeypatch.setattr(pnf, "_ANCHORS", fake)
    with pytest.raises(SystemExit, match="two different populations"):
        pnf.anchor_for("fortytwo")


# ── _as_arm · a bounded mutation of a module global ──────────────────────────────────────────


def test_as_arm_points_the_loader_at_the_requested_arm_and_restores_it():
    was = e1_data.ARM
    with pnf._as_arm("ctlfullzero_fortythree"):
        assert e1_data.ARM == "ctlfullzero_fortythree"
    assert e1_data.ARM == was


def test_as_arm_restores_the_global_EVEN_WHEN_THE_BODY_RAISES():
    """⛔⛔ The load-bearing case. Without the finally, one failed load leaves every later arm
    reading the wrong directory and G-N1 would be the only thing between that and a silent
    five-arm measurement of one arm.
    """
    was = e1_data.ARM
    with pytest.raises(RuntimeError, match="deliberate"):
        with pnf._as_arm("ctlfullzero_fortyfour"):
            assert e1_data.ARM == "ctlfullzero_fortyfour"
            raise RuntimeError("deliberate")
    assert e1_data.ARM == was, "the module global was left pointing at the wrong arm"


def test_as_arm_nests_without_losing_the_outermost_value():
    was = e1_data.ARM
    with pnf._as_arm("upctwofullzero_fortytwo"):
        with pnf._as_arm("ctlfullzero_fortyseven"):
            assert e1_data.ARM == "ctlfullzero_fortyseven"
        assert e1_data.ARM == "upctwofullzero_fortytwo"
    assert e1_data.ARM == was


# ── the declared population ──────────────────────────────────────────────────────────────────


def test_seed_46_is_excluded_and_the_reason_still_holds():
    """⚠️ If an anchor for 46 is ever committed, this test should be revisited, not deleted."""
    assert 46 not in {s for _w, s in pnf.SEED_ARMS.values()}
    assert "ctlfullzero_fortysix" not in pnf.SEED_ARMS
    with pytest.raises(SystemExit):
        pnf.anchor_for("fortysix")


def test_every_form_pair_shares_one_seed_and_differs_only_in_ARCHITECTURE():
    """⛔⛔ The gate on the error that voided half this pre-registration.

    The arms are named by EXP02_MANIFEST's ``arm:`` line, not by directory. ``ctlfullzero_*`` is
    the control and ``upctwofullzero_*`` is UpsampleConv2. I previously read the directory names
    (``calm_lander`` / ``calm_orbiter``), found their configs identical, and reported "seed only".
    """
    for ctl, trt, word in pnf.FORM_PAIRS:
        assert ctl == f"ctlfullzero_{word}" and trt == f"upctwofullzero_{word}"
        assert pnf.anchor_for(word, "ap_ctl") > 0.0
        assert pnf.anchor_for(word, "ap_trt") > 0.0
        assert pnf.anchor_for(word, "ap_ctl") != pnf.anchor_for(word, "ap_trt")


def test_anchor_for_REFUSES_a_column_the_record_does_not_carry():
    with pytest.raises(SystemExit, match="has no 'ap_nonsense' column"):
        pnf.anchor_for("fortytwo", "ap_nonsense")


def test_seed_arms_are_keyed_by_DELIVERED_ARM_NAME_not_directory():
    """⛔ A directory name here is the bug that voided P-39. Pin it."""
    assert set(pnf.SEED_ARMS) == {f"ctlfullzero_{w}" for w in pnf.SEED_WORDS}
    assert not any("lander" in a or "orbiter" in a for a in pnf.SEED_ARMS)


def test_the_prereg_constants_are_the_ones_the_docstring_declares():
    assert pnf.B_REPORT == (8, 16)
    assert pnf.P38_MAX_SPREAD_FRACTION == 0.5
    assert pnf.P40_MAX_POOLED_SD == 0.15
    assert pnf.F9_MIN_ARMS == 4
    assert len(pnf.SEED_ARMS) == 5 and len(pnf.FORM_PAIRS) == 5
    assert pnf.SEED_WORDS == ("fortytwo", "fortythree", "fortyfour", "fortyfive", "fortyseven")


def test_spread_is_a_population_sd_and_a_range():
    mean, sd, rng = pnf._spread([1.0, 2.0, 3.0])
    assert mean == 2.0 and rng == 2.0
    assert abs(sd - (2 / 3) ** 0.5) < 1e-12, (
        "a sample sd would give 1.0; the docstring says population"
    )
