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
        reverse: bool = False,
    ):
        if schedule not in VALID_SCHEDULES:
            raise ValueError(f"Invalid schedule '{schedule}'. Must be one of: {VALID_SCHEDULES}")
        if epsilon_max < 0 or epsilon_max > 1:
            raise ValueError(f"epsilon_max must be in [0, 1], got {epsilon_max}")
        # `k` is not optional for the two decay schedules: get_epsilon divides by it and
        # exponentiates it, so a None here does not fall back to anything — it raises TypeError
        # deep inside the training loop, hundreds of lessons after the config was accepted.
        # Refuse the unusable combination at construction rather than carrying it as a value.
        if schedule in ("exponential", "inverse_sigmoid") and k is None:
            raise ValueError(f"{schedule} schedule requires k; got None.")
        if schedule == "exponential" and k is not None and k >= 1.0:
            raise ValueError(f"exponential schedule requires k < 1.0, got {k}")
        # Bengio 2015 inverse-sigmoid decay requires k >= 1 (k<1 is a wrong schedule shape; k=0
        # would divide by zero inside get_epsilon). Symmetric to the exponential k<1 guard above.
        if schedule == "inverse_sigmoid" and k is not None and k < 1.0:
            raise ValueError(f"inverse_sigmoid schedule requires k >= 1.0, got {k}")
        self.schedule = schedule
        self.epsilon_max = epsilon_max
        self.warmup_lessons = warmup_lessons or 0
        self.k = k
        self.reverse = reverse
        logger.info(
            f"ScheduledSamplingMixer: schedule={schedule}, epsilon_max={epsilon_max}, "
            f"warmup={self.warmup_lessons}, k={k}, reverse={reverse}"
            + (" (INCREASING teacher forcing — Teutsch 2022 ITF, #287)" if reverse else "")
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

        if self.reverse:
            # INCREASING teacher forcing (Teutsch et al. 2022, #287): epsilon starts at
            # `epsilon_max` and DECAYS to 0, so the model begins near free-running and is given
            # progressively more ground truth. The forward direction — epsilon rising from 0 — is
            # the decreasing-TF curriculum that Teutsch reports failing on time series and that
            # our own SS sweep measured as harmful (M30-M33).
            #
            # `warmup_lessons` is the linear ramp length, so an ITF arm sets it to the TOTAL
            # lesson count: the paper's method ramps across training, not over a short warmup.
            # See `2026-08-23_itf_pilot_dossier/05_analysis_plan.md` AMENDMENT 1 for why a strict
            # mirror of our constant-dose SS arm was rejected.
            return min((1.0 - raw) * self.epsilon_max, self.epsilon_max)
        return min(raw * self.epsilon_max, self.epsilon_max)
