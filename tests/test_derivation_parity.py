"""
Derivation parity guard: verifies that the DataFrame-level binary derivation
logic and VolumeHandler._execute_derivations() produce identical thresholding
results on the same input data.

These two paths implement the same logic independently (DataFetcher.apply_blueprint
operates on DataFrames; VolumeHandler._execute_derivations operates on volumes).
This test catches divergence if one path is modified without the other.

Note: DataFetcher cannot be imported without views_pipeline_core, so we
replicate its binary derivation logic inline (it's a single line:
``(df[src] > threshold).astype(float)``). The source of truth is
data_fetcher.py:apply_blueprint().
"""

import numpy as np
import pandas as pd

from views_hydranet.utils.volume_handler import VolumeHandler


def _make_parity_config():
    """Config with a single binary derivation for parity testing."""
    return {
        "height": 2,
        "width": 2,
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "identity_cols": ["c_id", "row", "col"],
        "features": ["src_feat"],
        "derivations": {
            "binary": [{"from": "src_feat", "to": "dst_binary", "threshold": 0.5}],
        },
    }


def _make_parity_df():
    """DataFrame with known values spanning the threshold (0.5)."""
    rows = []
    for t in range(2):
        for r in range(2):
            for c in range(2):
                rows.append({
                    "month_id": 500 + t,
                    "priogrid_gid": r * 2 + c + 1,
                    "c_id": 1,
                    "row": float(r),
                    "col": float(c),
                    "src_feat": 0.1 * (t * 4 + r * 2 + c),  # 0.0, 0.1, ..., 0.7
                })
    return pd.DataFrame(rows)


def _apply_blueprint_binary(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Replicates the binary derivation from DataFetcher.apply_blueprint().
    Source of truth: data_fetcher.py lines 166-176.
    """
    df_out = df.copy()
    for instr in config["derivations"].get("binary", []):
        src = instr["from"]
        dst = instr["to"]
        threshold = instr["threshold"]
        df_out[dst] = (df_out[src] > threshold).astype(float)
    return df_out


def test_binary_derivation_produces_identical_results():
    """
    The DataFrame path (binary thresholding) and the Volume path
    (VolumeHandler._execute_derivations) must produce identical results.
    """
    config = _make_parity_config()
    df = _make_parity_df()

    # --- DataFrame path ---
    df_result = _apply_blueprint_binary(df, config)
    df_derived = df_result["dst_binary"].values  # flat array, ordered by df rows

    # --- Volume path ---
    # Build VolumeHandler WITHOUT derivations, then apply manually
    config_no_deriv = dict(config)
    config_no_deriv.pop("derivations")
    vh = VolumeHandler.from_df(df, config_no_deriv)

    # Execute derivations on the volume
    vh._execute_derivations(config["derivations"])

    # Extract the derived channel
    dst_idx = vh.channel_map.index("dst_binary")
    c_idx = vh.get_axis_idx("C")
    slc = [slice(None)] * vh.data.ndim
    slc[c_idx] = dst_idx
    vol_derived_raw = vh.data[tuple(slc)]  # shape [T, H, W]

    # Reconstruct the same flat ordering as the DataFrame
    # Volume is [T, H, W, C] with North-Up flip applied in from_df
    # Undo the flip on H axis to get back to original row ordering
    vol_flipped_back = np.flip(vol_derived_raw, axis=1)
    # Transpose to [H, W, T] to match (row, col, time) iteration order
    vol_hwt = np.transpose(vol_flipped_back, (1, 2, 0))

    # Flatten in the same order as the DataFrame (time-major, then row, then col)
    vol_flat = []
    for t in range(2):
        for r in range(2):
            for c in range(2):
                vol_flat.append(vol_hwt[r, c, t])
    vol_flat = np.array(vol_flat)

    np.testing.assert_array_equal(
        df_derived,
        vol_flat,
        err_msg=(
            "DataFrame binary derivation and VolumeHandler._execute_derivations() "
            "produced different results on the same input. "
            "Check data_fetcher.py:apply_blueprint() vs volume_handler.py:_execute_derivations()."
        ),
    )
