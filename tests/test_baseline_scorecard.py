"""Gates of ``reports/2026-11-07_baseline_scorecard/tools/baseline_scorecard.py``.

⛔ House rule: every gate the instrument declares is shown HERE to FIRE on an input it is supposed
to reject. A gate that only ever passes is decorative, and three gates in this dossier's siblings
were found this week to be exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-07_baseline_scorecard/tools"

#: ⛔ ``reports/`` is gitignored-but-tracked, so these tools CAN be absent from a tree that was
#: built another way. Skip rather than error — the same pattern as the sibling dossier tests, and
#: the property ``test_falsify_zero_surprises.py::test_P5b`` requires of every tracked test.
pytestmark = pytest.mark.skipif(
    not (_TOOLS / "baseline_scorecard.py").exists(),
    reason="the baseline-scorecard dossier tools are not present in this tree",
)

if (_TOOLS / "baseline_scorecard.py").exists():
    for _sub in (
        "reports/2026-11-04_delivered_gate_lookup/tools",
        "reports/2026-07-17_lodestar_eval_dossier/tools",
        "reports/2026-11-05_conditional_magnitude/tools",
        "reports/2026-11-06_threshold_reliability/tools",
        "reports/2026-11-07_baseline_scorecard/tools",
    ):
        if str(_REPO / _sub) not in sys.path:
            sys.path.insert(0, str(_REPO / _sub))

    import baseline_scorecard as bs  # noqa: E402
    import e1_data  # noqa: E402
    import ordering_by_index as obi  # noqa: E402
    import ordering_by_separation as obs_mod  # noqa: E402
    import profile_noise_floor as pnf  # noqa: E402

# ── the roster ───────────────────────────────────────────────────────────────────────────────


def test_there_are_exactly_ten_arms_five_of_each_form() -> None:
    assert len(bs.ARMS) == bs.N_ARMS_REQUIRED == 10
    names = [a for a, _w, _c in bs.ARMS]
    assert len(set(names)) == 10
    assert sum(n.startswith("ctlfullzero_") for n in names) == 5
    assert sum(n.startswith("upctwofullzero_") for n in names) == 5


def test_every_control_arm_reads_ap_ctl_and_every_upctwo_arm_reads_ap_trt() -> None:
    for arm, _word, col in bs.ARMS:
        assert col == ("ap_ctl" if arm.startswith("ctlfullzero_") else "ap_trt")


def test_the_roster_is_the_one_profile_noise_floor_already_scores() -> None:
    """⛔ If these drift apart, ``pnf.anchor_for`` has no record for an arm and G-0 cannot run."""
    expected = {f"ctlfullzero_{w}" for w in pnf.SEED_WORDS} | {
        f"upctwofullzero_{w}" for w in pnf.SEED_WORDS
    }
    assert {a for a, _w, _c in bs.ARMS} == expected


def test_the_two_anchor_columns_carry_DIFFERENT_numbers_for_every_seed() -> None:
    """⭐⭐ THE NON-VACUITY CHECK ON G-0's COLUMN CHOICE.

    Scoring an UpsampleConv2 arm against ``ap_ctl`` is a one-token mistake. It is only CAUGHT if
    the two columns actually differ — otherwise G-0 would pass on the wrong anchor and the five
    new arms would be silently unanchored. This proves they differ, per seed.
    """
    for word in pnf.SEED_WORDS:
        ctl = pnf.anchor_for(word, "ap_ctl")
        trt = pnf.anchor_for(word, "ap_trt")
        assert ctl != trt, word
        assert abs(ctl - trt) > e1_data.G0_TOL, word


# ── G-B3: the calibration population may not be outcome-selected ─────────────────────────────


def test_gate_b3_FIRES_when_every_truth_in_the_calibration_set_is_POSITIVE() -> None:
    ok, detail = bs.gate_b3_calibration_population_is_not_outcome_selected(
        np.array([1.0, 7.0, 3.0, 250.0])
    )
    assert not ok
    assert "outcome-selected" in detail


def test_gate_b3_passes_on_a_prediction_selected_set_that_contains_zeros() -> None:
    ok, detail = bs.gate_b3_calibration_population_is_not_outcome_selected(
        np.array([0.0, 0.0, 0.0, 4.0])
    )
    assert ok
    assert "3 of 4" in detail


def test_gate_b3_FIRES_on_an_empty_set_rather_than_dividing_by_nothing() -> None:
    ok, _ = bs.gate_b3_calibration_population_is_not_outcome_selected(np.array([]))
    assert not ok


# ── G-B5: the two populations must not be the same set ───────────────────────────────────────


def test_gate_b5_FIRES_when_the_open_set_and_the_event_set_are_the_same_size() -> None:
    ok, _ = bs.gate_b5_the_two_populations_are_distinct(e1_data.G8_EVENTS, e1_data.G8_EVENTS)
    assert not ok


def test_gate_b5_FIRES_when_the_event_count_is_not_the_committed_one() -> None:
    ok, detail = bs.gate_b5_the_two_populations_are_distinct(5_000, e1_data.G8_EVENTS - 1)
    assert not ok
    assert f"{e1_data.G8_EVENTS:,}" in detail


def test_gate_b5_passes_on_the_real_shape() -> None:
    ok, _ = bs.gate_b5_the_two_populations_are_distinct(4_481, e1_data.G8_EVENTS)
    assert ok


# ── G-B1 / G-B2: the control columns are pinned to the published figures ─────────────────────


def test_gate_b1_b2_passes_exactly_at_the_published_figures() -> None:
    ok, _ = bs.gate_b1_b2_controls_reproduce_the_published_figures(
        bs.M98_MEAN_DRAW_070, bs.M99_GATE_070
    )
    assert ok


@pytest.mark.parametrize("which", ["mean_draw", "gate"])
def test_gate_b1_b2_FIRES_when_EITHER_control_column_drifts(which: str) -> None:
    """⛔ Both columns must bind. A gate that only watches one leaves the other free to move."""
    md = bs.M98_MEAN_DRAW_070 + (bs.PIN_TOL * 2 if which == "mean_draw" else 0.0)
    g = bs.M99_GATE_070 + (bs.PIN_TOL * 2 if which == "gate" else 0.0)
    ok, detail = bs.gate_b1_b2_controls_reproduce_the_published_figures(md, g)
    assert not ok
    assert f"tol {bs.PIN_TOL}" in detail


def test_the_pin_tolerance_actually_binds_rather_than_admitting_anything() -> None:
    """⭐ A tolerance is only a gate if some value fails it. 0.04 in, 0.06 out."""
    inside, _ = bs.gate_b1_b2_controls_reproduce_the_published_figures(
        bs.M98_MEAN_DRAW_070 + 0.04, bs.M99_GATE_070
    )
    outside, _ = bs.gate_b1_b2_controls_reproduce_the_published_figures(
        bs.M98_MEAN_DRAW_070 + 0.06, bs.M99_GATE_070
    )
    assert inside and not outside


def test_the_pinned_constants_are_the_published_ones() -> None:
    """⚠️ Transcribed from ``RESULTS_LEDGER.md``; if someone edits them the pin means nothing."""
    assert bs.M98_MEAN_DRAW_070 == 8.61
    assert bs.M99_GATE_070 == 6.32


def test_gate_b1_b2_FIRES_on_a_non_finite_control_column() -> None:
    """An arm that never crosses 0.70 propagates ``inf``; the pin must refuse, not compare."""
    ok, _ = bs.gate_b1_b2_controls_reproduce_the_published_figures(float("inf"), bs.M99_GATE_070)
    assert not ok


# ── G-C3: ten arms or nothing, and _agg must not hide a short column ─────────────────────────


def test_agg_does_NOT_drop_non_finite_values() -> None:
    """⛔ The point of G-C3: a filtered mean prints a nine-arm figure under a ten-arm head."""
    m, s = bs._agg([1.0, 2.0, float("inf")])
    assert not np.isfinite(m)
    assert not np.isfinite(s)


def test_agg_matches_a_plain_mean_when_everything_is_finite() -> None:
    m, s = bs._agg([2.0, 4.0])
    assert m == pytest.approx(3.0)
    assert s == pytest.approx(1.0)


# ── the estimator is IMPORTED, not copied ────────────────────────────────────────────────────


def test_the_estimator_is_the_SAME_function_object_as_the_published_instruments() -> None:
    """⭐⭐ Why this file cannot drift from ``M98``/``M99``: there is only one of each function."""
    assert bs.resolution_curve is obs_mod.resolution_curve
    assert bs.crossing is obs_mod.crossing
    assert bs.crossing_support is obs_mod.crossing_support
    assert bs.PairSet is obs_mod.PairSet
    assert bs.PairView is obs_mod.PairView
    assert bs.RESOLUTION_TARGETS is obs_mod.RESOLUTION_TARGETS
    assert bs.mean_draw is obi.mean_draw
    assert bs.gate_only is obi.gate_only


def test_the_two_indices_are_the_ones_G_I1_already_certifies_truth_blind() -> None:
    """⭐ Importing them means ``ordering_by_index``'s signature gate covers this file too."""
    ok, detail = obi.gate_i1_indices_cannot_see_the_truth()
    assert ok, detail
    names = {n for n, _f in obi.INDEXES}
    assert {"mean_draw", "gate"} <= names


def test_the_headline_target_is_one_of_the_reported_targets() -> None:
    assert bs.HEADLINE_TARGET in bs.RESOLUTION_TARGETS


# ── the formatter must not print a thin or absent crossing as a number ───────────────────────


def test_sep_prints_never_for_a_crossing_that_does_not_exist() -> None:
    assert bs._sep(float("inf"), 0) == "never"


def test_sep_marks_a_crossing_whose_support_is_below_the_floor() -> None:
    assert "⛔thin" in bs._sep(8.61, bs.MIN_PAIRS - 1)


def test_sep_leaves_a_well_supported_crossing_clean() -> None:
    assert bs._sep(8.61, bs.MIN_PAIRS) == "8.61x"


# ── the calibration axis is swept, not chosen ────────────────────────────────────────────────


def test_the_calibration_bin_count_is_swept_and_the_headline_is_inside_the_sweep() -> None:
    """⛔ An unswept bin count is the defect that cost this dossier two days (``D-4``)."""
    assert len(bs.CALIB_SWEEP) >= 3
    assert bs.CALIB_B in bs.CALIB_SWEEP


def test_the_arm_restore_default_was_captured_and_is_a_real_arm() -> None:
    assert bs._E1_DEFAULT_ARM
    assert bs._E1_DEFAULT_ARM == e1_data.ARM
