"""
DataSniffer ingestion / forecast-alignment RED gates (ADR-005 taxonomy).

These exercise EXISTING fail-loud guards in views_hydranet/utils/data_sniffer.py
whose red paths were previously only run on happy input:

  1. _check_spatiotemporal_uniqueness  (data_sniffer.py:277-297)  — "Duplicate Entries"
  2. _check_finiteness                  (data_sniffer.py:178-186)  — "Non-finite values ...
     mandatory column" (reached via _check_identity_values before _check_non_finite)
  3. sniff_forecast_alignment continuity (data_sniffer.py:99-110) — "Forecast Continuity Broken"

df-builder / config style mirrors tests/test_datasniffer_offset_drift.py.
"""

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.volume_handler import VolumeHandler

# ---------------------------------------------------------------------------
# Shared config (no row_offset/col_offset → anchor-alignment/drift checks skip,
# isolating the guard under test).
# ---------------------------------------------------------------------------

INGEST_CFG = {
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["value"],
    "height": 180,
    "width": 180,
}


def _make_ingest_df(
    month_ids: list[int],
    gids: list[int],
    rows: list[int],
    cols: list[int],
    values: list[float],
) -> pd.DataFrame:
    """Build a DataFrame carrying every obligatory ingestion column."""
    return pd.DataFrame(
        {
            "month_id": month_ids,
            "priogrid_gid": gids,
            "row": rows,
            "col": cols,
            "value": values,
        }
    )


# ---------------------------------------------------------------------------
# 1. RED: duplicate (month_id, priogrid_gid) spatiotemporal key
# ---------------------------------------------------------------------------


def test_ingestion_rejects_duplicate_spatiotemporal_key():
    """
    RED GATE: sniff_ingestion() must raise when a (time_col, id_col) pair repeats.

    Fires _check_spatiotemporal_uniqueness (data_sniffer.py:277-297).
    """
    sniffer = DataSniffer(INGEST_CFG)
    # (month_id=100, priogrid_gid=1) appears twice → duplicate key.
    df = _make_ingest_df(
        month_ids=[100, 100],
        gids=[1, 1],
        rows=[10, 11],
        cols=[20, 21],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError, match="Duplicate Entries Detected"):
        sniffer.sniff_ingestion(df)


# ---------------------------------------------------------------------------
# 2. RED: non-finite value in an identity column (month_id NaN)
# ---------------------------------------------------------------------------


def test_ingestion_rejects_non_finite_identity_column():
    """
    RED GATE: sniff_ingestion() must raise when a mandatory identity column
    (month_id) carries a non-finite value (NaN).

    Fires _check_finiteness (data_sniffer.py:178-186) via _check_identity_values,
    reached before the broader _check_non_finite scan.
    """
    sniffer = DataSniffer(INGEST_CFG)
    # Distinct gids → uniqueness passes; NaN in month_id → finiteness fails.
    df = _make_ingest_df(
        month_ids=[100, np.nan],
        gids=[1, 2],
        rows=[10, 11],
        cols=[20, 21],
        values=[1.0, 2.0],
    )

    with pytest.raises(
        ValueError,
        match=r"Non-finite values detected in mandatory column 'month_id'",
    ):
        sniffer.sniff_ingestion(df)


# ---------------------------------------------------------------------------
# 3. RED: forecast temporal discontinuity (forecast min != history max + 1)
# ---------------------------------------------------------------------------


def _make_month_handler(month: int) -> VolumeHandler:
    """
    Minimal VolumeHandler whose month_id channel is filled with `month`,
    mirroring the fixture style in test_datasniffer_offset_drift.py.
    """
    data = np.zeros((1, 4, 4, 2), dtype=np.float32)
    data[..., 0] = float(month)  # channel 0 = month_id
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "value"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
    )


def test_forecast_alignment_rejects_temporal_discontinuity():
    """
    RED GATE: sniff_forecast_alignment(..., is_forecast=True) must raise when the
    forecast volume's min month != history-df max month + 1.

    Fires the continuity guard (data_sniffer.py:99-110). Mirrors the
    is_forecast=False alignment tests but breaks continuity: history ends at 50,
    forecast volume starts at 100 (expected 51).
    """
    sniffer = DataSniffer(INGEST_CFG)
    handler = _make_month_handler(month=100)  # forecast volume anchored at month 100
    df = _make_ingest_df(
        month_ids=[50, 50],  # history ends at 50 → expected forecast start 51
        gids=[1, 2],
        rows=[0, 1],
        cols=[0, 1],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError, match="Forecast Continuity Broken"):
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
