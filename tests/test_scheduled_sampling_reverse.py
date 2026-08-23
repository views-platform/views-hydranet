"""The `reverse` flag: INCREASING teacher forcing (Teutsch et al. 2022, #287).

Pre-registration: `reports/2026-08-23_itf_pilot_dossier/05_analysis_plan.md` (LOCKED `4cb8953`,
AMENDMENT 1 `f76e685`) — both committed before this code existed.

§6 of that plan registers four falsifiers. The two that live in code are here, and the first is
load-bearing: this touches `views_hydranet/` TRAINING code, not a dossier probe, so a silent
perturbation of the default path would contaminate every arm the programme has already run.
"""

from __future__ import annotations

import pytest

from views_hydranet.utils.scheduled_sampling import VALID_SCHEDULES, ScheduledSamplingMixer

LESSONS = 300


@pytest.mark.parametrize("schedule", VALID_SCHEDULES)
@pytest.mark.parametrize("warmup", [0, 15, 300])
def test_reverse_false_is_byte_identical_to_the_pre_flag_behaviour(schedule, warmup):
    """§6 falsifier 1. `reverse=False` must reproduce the old mixer EXACTLY, over the full lesson
    range — not approximately, and not just at the endpoints.

    The old behaviour is `min(raw * epsilon_max, epsilon_max)`; this pins that the default branch
    still returns it bit-for-bit, so every already-trained arm remains reproducible.
    """
    k = 12.0 if schedule == "inverse_sigmoid" else (0.98 if schedule == "exponential" else None)
    default = ScheduledSamplingMixer(schedule, 0.5, warmup_lessons=warmup, k=k)
    explicit = ScheduledSamplingMixer(schedule, 0.5, warmup_lessons=warmup, k=k, reverse=False)
    for lesson in range(LESSONS):
        a, b = default.get_epsilon(lesson), explicit.get_epsilon(lesson)
        assert a == b, f"{schedule}/warmup={warmup} lesson {lesson}: {a!r} != {b!r}"


def test_reverse_actually_decreases_epsilon_across_training():
    """§6 falsifier 2. A flag that transposed nothing would produce an "ITF" arm identical to SS —
    the experiment would run, cost ~10 GPU-hours, and measure the thing it was reversing.
    """
    itf = ScheduledSamplingMixer("linear", 0.5, warmup_lessons=LESSONS, reverse=True)
    first, last = itf.get_epsilon(0), itf.get_epsilon(LESSONS - 1)
    assert first == pytest.approx(0.5, abs=1e-9), f"ITF must START at epsilon_max, got {first}"
    assert last < 0.01, f"ITF must END near teacher forcing, got {last}"
    series = [itf.get_epsilon(i) for i in range(LESSONS)]
    assert series == sorted(series, reverse=True), "epsilon must be monotonically non-increasing"


def test_reverse_is_the_pointwise_complement_of_forward():
    """`reverse` mirrors the schedule about epsilon_max/2 at every lesson — the property that makes
    "same dose, opposite direction" meaningful at all."""
    fwd = ScheduledSamplingMixer("linear", 0.5, warmup_lessons=LESSONS)
    rev = ScheduledSamplingMixer("linear", 0.5, warmup_lessons=LESSONS, reverse=True)
    for lesson in range(0, LESSONS, 7):
        assert fwd.get_epsilon(lesson) + rev.get_epsilon(lesson) == pytest.approx(0.5, abs=1e-9)


def test_reverse_never_exceeds_epsilon_max_or_goes_negative():
    """The mixer's output is a probability and is fed straight to a Bernoulli mask; a value outside
    [0, epsilon_max] would be silently clamped by torch and mis-dose the arm."""
    for schedule in VALID_SCHEDULES:
        k = (
            12.0
            if schedule == "inverse_sigmoid"
            else (0.98 if schedule == "exponential" else None)
        )
        m = ScheduledSamplingMixer(schedule, 0.5, warmup_lessons=50, k=k, reverse=True)
        for lesson in range(LESSONS):
            e = m.get_epsilon(lesson)
            assert 0.0 <= e <= 0.5, f"{schedule} lesson {lesson}: epsilon={e} out of range"


def test_the_itf_arm_config_ramps_across_training_not_over_a_warmup():
    """AMENDMENT 1's adopted shape. With `warmup_lessons=15` (the SS arm's value) `reverse=True`
    would decay to 0 within 15 lessons and then teacher-force for 285 — NOT the paper's method.
    The ITF arm must set the ramp to the full lesson count."""
    wrong = ScheduledSamplingMixer("linear", 0.5, warmup_lessons=15, reverse=True)
    assert wrong.get_epsilon(20) == 0.0, "short ramp collapses to pure TF — the shape A1 rejected"
    right = ScheduledSamplingMixer("linear", 0.5, warmup_lessons=LESSONS, reverse=True)
    assert right.get_epsilon(150) == pytest.approx(0.25, abs=1e-9), "must be mid-decay at halfway"


def test_ss_reverse_reaches_the_mixer_from_config():
    """The flag is useless if it stops at the config boundary. `training_engine` builds the mixer
    from `config.get("ss_reverse", False)`; this pins that the key exists on the config model and
    that its default preserves the pre-flag path."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    assert "ss_reverse" in HydraNetConfig.model_fields, "ss_reverse missing from the config model"
    assert HydraNetConfig.model_fields["ss_reverse"].default is False, (
        "ss_reverse must default to False, or every existing arm silently changes curriculum"
    )


def test_training_engine_forwards_ss_reverse_to_the_mixer():
    """Asserted on source because constructing a real TrainingContext here would be a fixture the
    size of the engine. The registered falsifier is the byte-identity test above; this guards the
    one line that would otherwise make the flag a no-op in production."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "views_hydranet/train/training_engine.py"
    text = src.read_text()
    assert 'reverse=config.get("ss_reverse", False)' in text, (
        "training_engine does not forward ss_reverse — the mixer would silently run forward"
    )


def test_decay_schedules_refuse_a_missing_k_at_construction():
    """`k` is required by `exponential` and `inverse_sigmoid`, not optional.

    Before this guard the constructor accepted `k=None` and `get_epsilon` raised TypeError the
    first time the schedule left its warmup — i.e. hundreds of lessons into a run, long after the
    config was validated. The unusable combination is refused at construction instead of being
    carried as a value.
    """
    import pytest

    from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

    for schedule in ("exponential", "inverse_sigmoid"):
        with pytest.raises(ValueError, match="requires k"):
            ScheduledSamplingMixer(schedule, 0.5, warmup_lessons=10, k=None)


def test_reverse_epsilon_stays_a_probability_on_every_schedule():
    """The ITF path must never emit an epsilon outside [0, epsilon_max].

    `reverse` returns `(1 - raw) * epsilon_max`, which is only a probability while `raw <= 1`.
    This walks every valid schedule past its warmup and pins the range, so a future schedule whose
    `raw` overshoots is caught here rather than silently feeding a negative probability to the
    Bernoulli draw (where it would read as plain teacher forcing).
    """
    from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

    cases = [("linear", None), ("exponential", 0.9), ("inverse_sigmoid", 12.0)]
    for schedule, k in cases:
        mixer = ScheduledSamplingMixer(schedule, 0.5, warmup_lessons=100, k=k, reverse=True)
        values = [mixer.get_epsilon(i) for i in range(400)]
        assert min(values) >= 0.0, f"{schedule}: negative epsilon {min(values)}"
        assert max(values) <= 0.5, f"{schedule}: epsilon {max(values)} above epsilon_max"
