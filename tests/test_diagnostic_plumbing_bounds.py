"""S8/#361: the diagnostic plumbing must not kill, mislabel, or silently neuter the run.

Three defects, none of which touches a production forecast, all of which decide whether a
diagnostic run finishes and whether its output can be read afterwards:

* **(a)** the two stats buffers grew without bound on an engine held across every origin. The
  first gate-probe run was SIGKILLed at `rc=137` — the OOM killer — and the file recorded the
  symptom without bounding the cause.
* **(b)** the body-mean dump stored `mu` target-second and `gate` target-last, wrote one `n_reg`
  taken from `mu`'s axis, and documented neither layout.
* **(c)** the `shuffle_months` derangement was built for **every** arm, so a two-step rollout
  raised `RuntimeError` for `identity`, `thin` and the rest — and a one-step rollout made
  `shuffle_months` itself run as the control.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from views_hydranet.utils.hydranet_inference import (  # noqa: E402
    DIAGNOSTIC_STATS_MAX_RECORDS,
    HydraNetInference,
)

from .test_feedback_transform_seam import _cfg, _RecordingModel, _tensor  # noqa: E402


def _inference(*, feedback_transform=None, time_steps=5):
    cfg = _cfg()
    cfg["time_steps"] = time_steps
    cfg["steps"] = list(range(1, time_steps + 1))
    return HydraNetInference(
        _RecordingModel(), cfg, device="cpu", feedback_transform=feedback_transform
    )


# ─────────────────────────────────────────────────── (a) the buffers are bounded


class TestADiagnosticCannotOOMTheRunItDiagnoses:
    def test_the_buffer_stops_growing_at_the_ceiling(self):
        inf = _inference()
        overshoot = 500
        for i in range(DIAGNOSTIC_STATS_MAX_RECORDS + overshoot):
            inf._append_diagnostic_stat(
                inf.feedback_field_stats, {"i": i}, label="feedback_field_stats"
            )
        assert len(inf.feedback_field_stats) == DIAGNOSTIC_STATS_MAX_RECORDS, (
            "the stats buffer grew past its ceiling — at D=256 over 13 origins this is hundreds "
            "of MB held for the whole run, on top of the DxK cube (S8/#361)"
        )

    def test_the_records_it_refused_are_counted_not_silently_dropped(self):
        """A short buffer read as if complete is a prefix of the run — early origins only — and
        averaging a column over it is a biased readout that looks exactly like an unbiased one."""
        inf = _inference()
        overshoot = 37
        for i in range(DIAGNOSTIC_STATS_MAX_RECORDS + overshoot):
            inf._append_diagnostic_stat(
                inf.gate_structure_stats, {"i": i}, label="gate_structure_stats"
            )
        assert inf.diagnostic_stats_dropped == {"gate_structure_stats": overshoot}

    def test_a_run_that_fits_records_everything_and_reports_no_drops(self):
        """Guards the cap against being so tight that it truncates ordinary runs."""
        inf = _inference()
        for i in range(1000):
            inf._append_diagnostic_stat(
                inf.feedback_field_stats, {"i": i}, label="feedback_field_stats"
            )
        assert len(inf.feedback_field_stats) == 1000
        assert inf.diagnostic_stats_dropped == {}

    def test_the_first_refusal_warns_once_and_says_what_to_do(self, caplog):
        inf = _inference()
        with caplog.at_level("WARNING"):
            for i in range(DIAGNOSTIC_STATS_MAX_RECORDS + 5):
                inf._append_diagnostic_stat(
                    inf.feedback_field_stats, {"i": i}, label="feedback_field_stats"
                )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, "the ceiling warning must fire once, not once per record"
        assert "posterior_D" in warnings[0].getMessage()


# ─────────────────────────────────────────────── (b) the dump describes itself


class TestTheDumpIsSelfDescribing:
    """`mu` is `[T, n_reg, H, W]` and `gate` is `[T, H, W, n_cls]`. A consumer writing
    `mu[:, j] * gate[:, j]` broadcasts an `(H, W)` field against a `(W, n_cls)` slice — an error on
    most grids, and a silently WRONG per-target field wherever `W == n_cls`."""

    @staticmethod
    def _dump(tmp_path, *, n_reg=3, n_cls=3, t=2, h=4, w=5):
        inf = _inference()
        inf.body_mean_dump_dir = str(tmp_path)
        mu = np.zeros((t, n_reg, h, w), dtype=np.float32)
        gate = np.zeros((t, h, w, n_cls), dtype=np.float32)
        inf._dump_body_mean(mu, gate, origin=9, n_passes=4)
        return np.load(tmp_path / "bodymean_origin9.npz")

    def test_each_array_carries_its_own_layout(self, tmp_path):
        z = self._dump(tmp_path)
        assert str(z["mu_layout"]) == "T,n_reg,H,W"
        assert str(z["gate_layout"]) == "T,H,W,n_cls"

    def test_the_layouts_let_a_consumer_find_the_channel_axis_without_reading_the_source(
        self, tmp_path
    ):
        z = self._dump(tmp_path, n_reg=3, n_cls=3)
        pairs = (("mu", "mu_layout", "n_reg"), ("gate", "gate_layout", "n_cls"))
        for array, layout, name in pairs:
            axis = str(z[layout]).split(",").index(name)
            assert z[array].shape[axis] == int(z[name]), (
                f"{array}'s declared layout {str(z[layout])!r} does not match its shape "
                f"{z[array].shape} at the {name} axis"
            )

    def test_n_cls_is_read_from_the_gate_not_assumed_equal_to_n_reg(self, tmp_path):
        """The gate head is not guaranteed to carry exactly `n_reg` channels, and the dump does not
        slice it — so the file has to say how many it has."""
        z = self._dump(tmp_path, n_reg=3, n_cls=5)
        assert int(z["n_reg"]) == 3
        assert int(z["n_cls"]) == 5


# ──────────────────────────────────────── (c) the derangement belongs to one arm


class TestTheDerangementIsBuiltOnlyForTheArmThatUsesIt:
    @pytest.mark.parametrize("arm", ["identity", "thin:0.25", "use_real", "hold_last_real"])
    def test_a_short_rollout_does_not_raise_for_an_arm_that_never_shuffles(self, arm):
        """`time_steps == 2` leaves one step to permute. No derangement of one element exists, so
        the loop exhausted and raised — for arms that never read `_month_shuffle` (S8/#361)."""
        inf = _inference(feedback_transform=arm, time_steps=2)
        inf.predict(_tensor(), 3, 0, ["feat"])

    def test_shuffle_months_refuses_a_rollout_too_short_to_shuffle(self):
        inf = _inference(feedback_transform="shuffle_months", time_steps=2)
        with pytest.raises(ValueError, match="time_steps must be >= 3"):
            inf.predict(_tensor(), 3, 0, ["feat"])

    def test_shuffle_months_refuses_a_one_step_rollout_rather_than_becoming_the_control(self):
        """At `time_steps == 1` there is nothing to permute, `_month_shuffle` is empty, and
        `.get(step, step)` feeds the TRUE month at every step — the control wearing the
        treatment's name (C-331). It used to do exactly that, silently."""
        inf = _inference(feedback_transform="shuffle_months", time_steps=1)
        with pytest.raises(ValueError, match="the control, wearing the treatment's name"):
            inf.predict(_tensor(), 3, 0, ["feat"])

    def test_a_long_enough_shuffle_still_deranges_every_step(self):
        inf = _inference(feedback_transform="shuffle_months", time_steps=5)
        inf.predict(_tensor(), 3, 0, ["feat"])
        assert inf._month_shuffle, "shuffle_months built no permutation"
        assert all(src != dst for src, dst in inf._month_shuffle.items()), (
            "a fixed point survived: that step feeds the TRUE month while being scored as "
            "'persistence destroyed'"
        )
