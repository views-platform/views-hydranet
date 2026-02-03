"""
ConfigInitializer: Canonical Entry Point for HydraNet Configuration.
"""
import logging
from typing import Any, Dict

from views_hydranet.utils.utils_config import HydraNetConfig

logger = logging.getLogger(__name__)

class ConfigInitializer:
    """
    Handles the initialization, normalization, and validation of 
    HydraNet run-time configurations.
    """

    def __init__(self, raw_config: Dict[str, Any]) -> None:
        """
        Store the raw configuration from the pipeline core.
        """
        self._raw = raw_config

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the processed and strictly validated configuration dictionary.
        This is the single 'Handshake' point for the whole pipeline.
        """
        # 1. Strict Validation via Pydantic (ADR 008)
        # Any missing fields or legacy keys will trigger a loud ValidationError here.
        config_obj = HydraNetConfig(**self._raw)
        
        # 2. Return as a dictionary for component consumption
        return config_obj.model_dump()
