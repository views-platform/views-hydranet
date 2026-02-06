"""
DEPRECATED: Old Configuration Handshake and Validation Logic.
Moved here to clear the HydranetManager for a more principled approach.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def _perform_strict_handshake_LEGACY(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Original strict handshake logic using Pydantic.
    """
    from pydantic import ValidationError

    from views_hydranet.utils.utils_config import HydraNetConfig

    try:
        validated = HydraNetConfig(**raw_config)
        logger.info("LEGACY Handshake: Success.")
        return validated.model_dump(exclude_none=True)
    except ValidationError as e:
        logger.error(f"LEGACY Handshake: Failed. {e}")
        raise ValueError(f"Handshake failure: {e}")
