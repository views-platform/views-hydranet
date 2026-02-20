"""
DataSniffer Offset Drift Detection Tests

CIC Reference: DataSniffer.md §6 Failure Modes:
  "Anchor Violation: Raises a critical error if geographic offsets
  (row_offset, col_offset) drift between data partitions."

The existing sniff_forecast_alignment() checks whether the incoming
DataFrame's min-coordinates are below the handler's anchor. It does
NOT check whether the handler's anchor matches the config's intended
row_offset/col_offset. If a handler was built with one offset and the
config was later changed (or vice versa), spatial scrambling occurs
silently. This file closes that blind spot.
"""
import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.volume_handler import VolumeHandler


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

DRIFT_CFG = {
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["value"],
    "height": 4,
    "width": 4,
    "row_offset": 87,    # <-- the config's INTENDED anchor
    "col_offset": 310,
}


def _make_handler(row_offset: int, col_offset: int, month: int = 100) -> VolumeHandler:
    """
    Build a minimal VolumeHandler with month_id and one feature channel.
    All month_id voxels are filled with `month` so the temporal continuity
    check in sniff_forecast_alignment passes.
    """
    data = np.zeros((1, 4, 4, 2), dtype=np.float32)
    data[..., 0] = float(month)   # channel 0 = month_id
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "value"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        spatial_offset=(row_offset, col_offset),
    )


def _make_df(row_min: int, col_min: int, month: int = 100) -> pd.DataFrame:
    """
    Build a minimal DataFrame whose coordinates start at (row_min, col_min)
    and whose month_id matches `month`.
    """
    return pd.DataFrame({
        "month_id": [month, month],
        "priogrid_gid": [1, 2],
        "row": [row_min, row_min + 1],
        "col": [col_min, col_min + 1],
        "value": [1.0, 2.0],
    })


# ---------------------------------------------------------------------------
# RED GATE: drift on row_offset must be detected
# ---------------------------------------------------------------------------

def test_sniffer_detects_row_offset_drift():
    """
    RED GATE: sniff_forecast_alignment must raise when the handler's
    row_offset (90) differs from the config's row_offset (87).

    The handler passes the existing min-coordinate check because the
    DataFrame rows start at 90 (≥ handler's anchor of 90). Only the
    new anchor-drift check can catch this mismatch.

    Without the fix: sniff_forecast_alignment silently returns.
    After the fix: raises ValueError with 'drift' or 'anchor' in the message.
    """
    # Handler was accidentally built with row_offset=90 instead of config's 87
    handler = _make_handler(row_offset=90, col_offset=310)

    # DataFrame coordinates start at the handler's anchor (90), so the
    # existing geographic anchor check (df.min < r_off) passes.
    df = _make_df(row_min=90, col_min=310)

    sniffer = DataSniffer(DRIFT_CFG)

    with pytest.raises(ValueError, match=r"[Dd]rift|[Aa]nchor"):
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)


def test_sniffer_detects_col_offset_drift():
    """
    RED GATE: sniff_forecast_alignment must raise when the handler's
    col_offset (315) differs from the config's col_offset (310).
    """
    handler = _make_handler(row_offset=87, col_offset=315)  # wrong col
    df = _make_df(row_min=87, col_min=315)  # df starts at handler's anchor

    sniffer = DataSniffer(DRIFT_CFG)

    with pytest.raises(ValueError, match=r"[Dd]rift|[Aa]nchor"):
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)


# ---------------------------------------------------------------------------
# GREEN GATE: correct offsets must NOT raise
# ---------------------------------------------------------------------------

def test_sniffer_accepts_correct_offsets():
    """
    GREEN GATE: When handler.spatial_offset == (config row_offset, col_offset),
    sniff_forecast_alignment must return without raising.
    """
    # Handler built with the correct offsets as specified in config
    handler = _make_handler(row_offset=87, col_offset=310)
    df = _make_df(row_min=87, col_min=310)

    sniffer = DataSniffer(DRIFT_CFG)
    # Must not raise
    sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)


def test_sniffer_skips_drift_check_when_offsets_absent_from_config():
    """
    BACKWARDS-COMPAT GATE: If config does not contain row_offset/col_offset,
    the drift check is skipped and sniff_forecast_alignment still passes for a
    handler whose offset does not match any implicit expectation.

    This ensures that existing DataSniffer usages without explicit offset config
    keys are not broken by the new check.
    """
    cfg_no_offsets = {k: v for k, v in DRIFT_CFG.items()
                      if k not in ("row_offset", "col_offset")}
    handler = _make_handler(row_offset=90, col_offset=315)  # any offset
    df = _make_df(row_min=90, col_min=315)

    sniffer = DataSniffer(cfg_no_offsets)
    # No offset keys in config → drift check skipped → must not raise
    sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
