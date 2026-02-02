"""
CurriculumLearner: The Strategic Planner for the HydraNet Training Trajectory.
"""
import logging
from typing import Any, Dict, Tuple
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class CurriculumLearner:
    """
    Strategic Planner responsible for scheduling training difficulty (Cooling)
    and rotating subject targets (Oscillation).
    """

    def __init__(self, config: Dict[str, Any], handler: VolumeHandler):
        """
        Initializes the planner with the authoritative Ledger.
        """
        self.config = config
        self.handler = handler
        
        # 1. Pre-calculate trajectory parameters (Zero-Magic)
        self.samples = config["samples"]
        self.min_events = config["min_events"]
        self.max_events = config.get("max_events", 100) # Fallback if missing
        self.slope_ratio = config.get("slope_ratio", 0.75)
        self.roof_ratio = config.get("roof_ratio", 0.7)
        
        # 2. Extract targets from Ledger (Handshake)
        # In this implementation, we assume all features are valid targets for oscillation
        self.subjects = list(handler._metadata.feature_cols)
        if not self.subjects:
             raise ValueError("CurriculumLearner: Ledger has no feature columns to target.")

    def get_threshold(self, sample_idx: int) -> int:
        """
        Calculates the current 'min_events' threshold (The Cooling).
        """
        # Linear decay logic derived from previous 'my_decay'
        # b = rate of change per sample
        b = ((-self.max_events + self.min_events) / (self.samples * self.slope_ratio))
        
        # Linear progression
        threshold = self.max_events + b * sample_idx
        
        # Contract Enforcement: Cap and Floor
        threshold = min(threshold, self.max_events * self.roof_ratio)
        threshold = max(threshold, self.min_events)
        
        return int(threshold)

    def get_lesson(self, sample_idx: int) -> Tuple[str, int]:
        """
        Returns the specific (target, threshold) for the current training sample.
        """
        threshold = self.get_threshold(sample_idx)
        
        # Subject Oscillation: Rotate through features
        subject = self.subjects[sample_idx % len(self.subjects)]
        
        return subject, threshold
