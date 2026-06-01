"""
Scheduled Sampling Mixer (Bengio et al. 2015, ADR-056).

Computes epsilon schedules for gradually replacing ground-truth input
with model predictions during training. Closes the train/inference
distribution gap (exposure bias) for autoregressive models.
"""

import logging
import math

logger = logging.getLogger(__name__)

VALID_SCHEDULES = ("linear", "inverse_sigmoid", "exponential")


class ScheduledSamplingMixer:
    """
    Epsilon schedule for binary scheduled sampling.

    At each training timestep, the model's own prediction replaces the
    ground-truth input with probability epsilon. epsilon=0 is pure
    teacher forcing; epsilon=epsilon_max approaches free-running.
    """

    def __init__(
        self,
        schedule: str,
        epsilon_max: float,
        warmup_lessons: int | None = None,
        k: float | None = None,
    ):
        if schedule not in VALID_SCHEDULES:
            raise ValueError(f"Invalid schedule '{schedule}'. Must be one of: {VALID_SCHEDULES}")
        if epsilon_max < 0 or epsilon_max > 1:
            raise ValueError(f"epsilon_max must be in [0, 1], got {epsilon_max}")
        if schedule == "exponential" and k is not None and k >= 1.0:
            raise ValueError(f"exponential schedule requires k < 1.0, got {k}")
        self.schedule = schedule
        self.epsilon_max = epsilon_max
        self.warmup_lessons = warmup_lessons or 0
        self.k = k
        logger.info(
            f"ScheduledSamplingMixer: schedule={schedule}, "
            f"epsilon_max={epsilon_max}, warmup={self.warmup_lessons}, k={k}"
        )

    def get_epsilon(self, lesson_idx: int) -> float:
        if lesson_idx < self.warmup_lessons:
            if self.schedule == "linear":
                raw = lesson_idx / max(self.warmup_lessons, 1)
            else:
                raw = 0.0
        elif self.schedule == "linear":
            raw = 1.0
        elif self.schedule == "inverse_sigmoid":
            k = self.k
            shifted = lesson_idx - self.warmup_lessons
            raw = 1.0 - k / (k + math.exp(shifted / k))
        elif self.schedule == "exponential":
            shifted = lesson_idx - self.warmup_lessons
            raw = 1.0 - self.k**shifted
        else:
            raw = 0.0

        return min(raw * self.epsilon_max, self.epsilon_max)
