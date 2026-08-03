"""
CurriculumLearner: The Strategic Planner for the HydraNet Training Trajectory.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class CurriculumLearner:
    """
    Strategic Planner responsible for scheduling training difficulty (Cooling)
    and rotating subject targets (Oscillation).

    Implements Target-Relative Thresholding to ensure Signal Anchorage
    across tasks of varying sparsity.
    """

    def __init__(self, config: Dict[str, Any], handler: VolumeHandler):
        """
        Initializes the planner with the authoritative Ledger and calculates
        subject-specific intensity maxima.
        """
        self.config = config
        self.handler = handler

        # 1. Pre-calculate trajectory parameters (Zero-Magic)
        # All these keys MUST exist per ADR 008 schema validation.
        self.total_steps = config["total_lessons"] * config["windows_per_lesson"]

        # In the relative world, min/max are ratios (0.0 to 1.0)
        self.min_ratio = config["min_ratio"]
        self.max_ratio = config["max_ratio"]

        self.slope_ratio = config["slope_ratio"]
        self.roof_ratio = config["roof_ratio"]
        self.min_events = config["min_events"]

        # 2. Extract targets from Ledger and Record Maxima
        # ADR-062 / C-158: input-only static channels (e.g. CoordConv row/col) are appended to
        # feature_cols but must NEVER be rotated in as trainable subjects. Exclude them, sourced
        # from config['static_channels'] (the same source from_df uses to populate the statics).
        # No statics ⇒ this is list(feature_cols) verbatim, so the no-coords trajectory is
        # byte-identical.
        static_channels = config.get("static_channels", [])
        self.subjects = [c for c in handler.feature_cols if c not in static_channels]
        if not self.subjects:
            err_msg = "CurriculumLearner: Ledger has no trainable (non-static) columns to target."

            logger.error(err_msg)

            raise ValueError(err_msg)

        self.subject_maxima = self._calculate_subject_maxima()
        logger.info(f"CurriculumLearner: Established subject maxima: {self.subject_maxima}")

    def _calculate_subject_maxima(self) -> Dict[str, float]:
        """
        Scans the global volume to find the maximum activity for each subject.
        Uses spatiotemporal count non-zero as the activity metric.
        """
        maxima = {}
        data = self.handler.data

        for name in self.subjects:
            idx = self.handler.channel_map.index(name)
            # Find max activity in any spatial cell across time
            # Activity = sum of events in the T dimension
            activity = np.count_nonzero(data[..., idx], axis=0)
            maxima[name] = float(np.max(activity))

        return maxima

    def get_intensity_ratio(self, global_step_idx: int) -> float:
        """
        Calculates the current global intensity ratio (The Cooling).
        Returns a float between min_ratio and max_ratio.
        """
        # b = rate of change per step
        b = (-self.max_ratio + self.min_ratio) / (self.total_steps * self.slope_ratio)

        # Linear progression
        ratio = self.max_ratio + b * global_step_idx

        # Contract Enforcement
        ratio = min(ratio, self.max_ratio * self.roof_ratio)
        ratio = max(ratio, self.min_ratio)

        return ratio

    def get_lesson(self, global_step_idx: int) -> Tuple[str, int]:
        """
        Returns the specific (target, absolute_threshold) for the current training step.
        """
        ratio = self.get_intensity_ratio(global_step_idx)

        # Subject Oscillation
        subject = self.subjects[global_step_idx % len(self.subjects)]

        # Convert relative ratio to absolute event threshold
        subject_max = self.subject_maxima[subject]
        threshold = int(ratio * subject_max)

        # Floor safety: Always require at least 1 event if ratio > 0
        if ratio > 0 and threshold == 0 and subject_max > 0:
            threshold = 1

        return subject, threshold
