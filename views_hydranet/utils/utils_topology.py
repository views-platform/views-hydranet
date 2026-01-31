from enum import Enum


class SpatialLayout(Enum):
    """
    Explicit enumeration of geographic orientations in the HydraNet pipeline.
    """
    SOUTH_UP = "SOUTH_UP"  # Natural DataFrame order (min row at top index)
    NORTH_UP = "NORTH_UP"  # Legacy CNN order (flipped, min row at bottom index)

# Canonical Invariants
DF_VOL_OUTPUT_LAYOUT = SpatialLayout.NORTH_UP
MODEL_INPUT_LAYOUT = SpatialLayout.NORTH_UP
CONTRACT_DF_INPUT_LAYOUT = SpatialLayout.SOUTH_UP
