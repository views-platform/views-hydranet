import numpy as np
import pandas as pd
import pytest
import torch
from views_hydranet.utils.utils_df_to_vol_conversion import df_to_vol

from views_hydranet.utils.utils import get_full_tensor
from views_hydranet.utils.utils_contract_converters import zstack_to_contract_df


def test_topological_integrity_full_circuit():
    """
    GEOGRAPHIC INTEGRITY TEST:
    Verifies that spatial and temporal coordinates are preserved bit-for-bit.
    """
    print("\n--- TOPOLOGICAL INTEGRITY START ---")

    marker_val = 9.99
    marker_month = 101
    marker_row = 2
    marker_col = 3

    data = []
    for m in [100, 101, 102]:
        for r in range(1, 5):
            for c in range(1, 5):
                val = marker_val if (m == marker_month and r == marker_row and c == marker_col) else 0.0
                data.append({
                    "month_id": m,
                    "priogrid_gid": (r-1)*4 + c,
                    "row": r,
                    "col": c,
                    "c_id": 1,
                    "lr_sb_best": val,
                    "lr_ns_best": 0.0,
                    "lr_os_best": 0.0
                })

    df_raw = pd.DataFrame(data)
    config = {
        "transform": "identity",
        "evalution_mode": "stochastic",
        "input_channels": 3,
        "first_feature_idx": 5,
        "steps": [1, 2, 3],
        "test_samples": 1
    }
    columns = ["priogrid_gid", "col", "row", "month_id", "c_id", "lr_sb_best", "lr_ns_best", "lr_os_best"]

    # 1. DF -> Volume
    raw_features = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    vol = df_to_vol(df_raw, height=4, width=4, forecast_features=raw_features)

    # Trace 1: In Volume [T, H, W, C]
    # NOTE: df_to_vol now produces SOUTH_UP (Natural coordinates).
    # Original Row 2 (Index 1) should remain at Index 1.
    v_idx = np.argwhere(vol == marker_val)
    print(f"Trace 1 (Volume - Natural):  {v_idx.tolist()}")

    # 2. Volume -> Tensor
    full_tensor, meta_tensor = get_full_tensor(vol, config=config, columns=columns)
    t_idx = torch.where(full_tensor == marker_val)
    print(f"Trace 2 (Tensor):  {[[t.item() for t in t_list] for t_list in t_idx]}")

    # 3. Tensor -> Z-Stack
    # full_tensor is [1, T, C, H, W]
    # We want [T, H, W, C, S] (where S=1)
    # 1. squeeze(0) -> [T, C, H, W]
    # 2. unsqueeze(-1) -> [T, C, H, W, 1]
    # 3. permute(0, 2, 3, 1, 4) -> [T, H, W, C, 1]
    posterior_zstack = full_tensor.squeeze(0).unsqueeze(-1).permute(0, 2, 3, 1, 4).detach().cpu().numpy()
    meta_zstack = meta_tensor.squeeze(0).unsqueeze(-1).permute(0, 2, 3, 1, 4).detach().cpu().numpy()
    z_idx = np.argwhere(posterior_zstack == marker_val)
    print(f"Trace 3 (Z-Stack): {z_idx.tolist()}")

    # 4. ZStack -> Contract DF
    pg_map = meta_zstack[1, :, :, 0, 0]
    print("\nMetadata Map (PGIDs) for Month 101:")
    print(pg_map)

    results = zstack_to_contract_df(posterior_zstack, meta_zstack, "sb", config=config)
    df_recovered = results[0]

    # 5. Verification
    # Use np.isclose to handle float32/64 precision differences during list search
    recovered_marker_row = df_recovered[df_recovered["sb"].apply(lambda x: any(np.isclose(v, marker_val) for v in x))]
    if recovered_marker_row.empty:
        print("\n[!] FAILURE: Marker value 9.99 was LOST in the pipeline.")
        pytest.fail("Data was corrupted during transformation.")

    m_idx, pg_idx = recovered_marker_row.index[0]
    print(f"Marker found at: Month {m_idx}, PrioGrid {pg_idx}")

    expected_pg_id = (marker_row-1)*4 + marker_col
    assert m_idx == marker_month, f"Temporal shift! Expected {marker_month}, got {m_idx}"
    assert pg_idx == expected_pg_id, f"Spatial shift! Expected PGID {expected_pg_id}, got {pg_idx}"

    print("\n[v] TOPOLOGICAL INTEGRITY CONFIRMED.")
