
import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.volume_handler import VolumeHandler

# SHARED PHYSICS CONFIG
PHYSICS_CFG = {
    'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': [ 'lr_feature_a'],
    'row_offset': 0,
    'col_offset': 0,
    'height': 4,
    'width': 4
}

# --- GREEN TEAM: THE PROOF OF ACCURACY ---

def test_bridge_point_accuracy():
    """Prove bit-perfect accuracy for Point (4D) reconstruction."""
    df_in = pd.DataFrame({
        'month_id': [1, 1], 'priogrid_gid': [1, 2],
        'row': [0, 0], 'col': [0, 1],
         'lr_feature_a': [10.5, 20.5]
    })
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)

    posterior = np.zeros((1, 4, 4, 2)) # 1 Head (Signal + Prob)
    posterior[0, 3, 0, 0] = 100.5
    posterior[0, 3, 1, 0] = 200.5

    pred_handler = handler.wrap_predictions(posterior, base_names=[ 'lr_feature_a'])
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)

    assert df_out.loc[(1, 1), "pred_lr_feature_a"] == 100.5
    assert df_out.loc[(1, 2), "pred_lr_feature_a"] == 200.5
    assert df_out.loc[(1, 1), "lr_feature_a"] == 10.5

def test_bridge_stochastic_accuracy():
    """Prove bit-perfect accuracy for Stochastic (5D) reconstruction."""
    df_in = pd.DataFrame({'month_id': [1], 'priogrid_gid': [1], 'row': [0], 'col': [0],  'lr_feature_a': [10.0]})
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)

    posterior = np.zeros((1, 4, 4, 2, 3))
    posterior[0, 3, 0, 0, :] = [1.0, 2.0, 3.0]

    pred_handler = handler.wrap_predictions(posterior, base_names=[ 'lr_feature_a'])
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)

    val = df_out.loc[(1, 1), "pred_lr_feature_a"]


    assert isinstance(val, list), (
        f"Expected a list of stochastic samples for 5D posterior, got {type(val)}. "
        f"to_evaluation_df() must aggregate sample dimension S into a Python list per cell."
    )
    assert val == [1.0, 2.0, 3.0], (
        f"Expected [1.0, 2.0, 3.0] (the 3 posterior samples in order), got {val}. "
        f"Sample values must be preserved exactly; check collapse_to_point() and to_evaluation_df()."
    )

# --- BEIGE TEAM: THE PROOF OF ROBUSTNESS ---

def test_bridge_contract_violation():
    """Prove that duration mismatches fail loud and proud."""
    df_in = pd.DataFrame({'month_id': [1], 'priogrid_gid': [1], 'row': [0], 'col': [0],  'lr_feature_a': [1.0]})
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)

    posterior = np.zeros((2, 4, 4, 2)) # 2 months

    with pytest.raises(ValueError, match=r"Signal duration \(2\) does not match Handler duration \(1\)"):
        handler.wrap_predictions(posterior, base_names=[ 'lr_feature_a'])

def test_bridge_naming_suffixes():
    """Verify that both _raw and _prob suffixes are correctly applied."""
    df_in = pd.DataFrame({'month_id': [1], 'priogrid_gid': [1], 'row': [0], 'col': [0],  'lr_feature_a': [1.0]})
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)

    posterior = np.zeros((1, 4, 4, 2))
    pred_handler = handler.wrap_predictions(posterior, base_names=[ 'lr_feature_a'])
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)

    assert "pred_lr_feature_a" in df_out.columns, (
        f"Expected 'pred_lr_feature_a' in evaluation df columns, got: {list(df_out.columns)}. "
        f"wrap_predictions() must prepend 'pred_' to base_names for regression output."
    )
    assert "pred_by_feature_a" in df_out.columns, (
        f"Expected 'pred_by_feature_a' in evaluation df columns, got: {list(df_out.columns)}. "
        f"wrap_predictions() must generate 'pred_by_' prefix for binary classification output."
    )
    # --- RED TEAM: THE PROOF OF INVINCIBILITY ---

def test_bridge_vader_alignment_shuffle():
    """
    RED TEAM ATTACK: Verify that explicit joins fix topographic shuffling.
    """
    df = pd.DataFrame({
        'month_id': [1, 1, 1, 1],
        'priogrid_gid': [1, 2, 3, 4],
        'row': [0, 0, 1, 1],
        'col': [0, 1, 0, 1],
         'lr_feature_a': [10.0, 20.0, 30.0, 40.0]
    })
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)

    posterior = np.zeros((1, 4, 4, 2))
    # Index derivation: PHYSICS_CFG row_offset=0, col_offset=0, H=4.
    # North-Up flip: flipped_r = (H-1) - r_idx = 3 - r_idx.
    #   PGID 1: global_row=0, global_col=0 → r_idx=0, c_idx=0 → flipped_r=3 → posterior[0, 3, 0, 0]
    #   PGID 2: global_row=0, global_col=1 → r_idx=0, c_idx=1 → flipped_r=3 → posterior[0, 3, 1, 0]
    #   PGID 3: global_row=1, global_col=0 → r_idx=1, c_idx=0 → flipped_r=2 → posterior[0, 2, 0, 0]
    #   PGID 4: global_row=1, global_col=1 → r_idx=1, c_idx=1 → flipped_r=2 → posterior[0, 2, 1, 0]
    posterior[0, 3, 0, 0] = 10.0 # PGID 1
    posterior[0, 3, 1, 0] = 20.0 # PGID 2
    posterior[0, 2, 0, 0] = 30.0 # PGID 3
    posterior[0, 2, 1, 0] = 40.0 # PGID 4

    # ATTACK: Shuffle the data via API flip
    handler.flip("H")
    posterior = np.flip(posterior, axis=1)

    pred_handler = handler.wrap_predictions(posterior, base_names=[ 'lr_feature_a'])
    df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)

    # AUDIT: Re-alignment must be perfect
    assert df_res.loc[(1, 1), "pred_lr_feature_a"] == 10.0
    assert df_res.loc[(1, 2), "pred_lr_feature_a"] == 20.0
    assert df_res.loc[(1, 3), "pred_lr_feature_a"] == 30.0
    assert df_res.loc[(1, 4), "pred_lr_feature_a"] == 40.0

# --- MATHEMATICAL AUDIT: THE PROOF OF SEQUENCE ---

def test_jensens_inequality_sequence():
    """
    Formal Proof: Verifies that 'Inverse -> Collapse' (Arithmetic Mean) 
    yields a different (and scientifically correct) result than 
    'Collapse -> Inverse' (Geometric Mean).    
    Data: Two samples, 10 and 100 (Raw Count Space).
    Transformation: Log1p (Natural Log(x + 1)).
    """
    raw_samples = np.array([10.0, 100.0])
    model_output_log_space = np.log1p(raw_samples)
    data_5d = model_output_log_space.reshape(1, 1, 1, 1, 2)

    vh = VolumeHandler(
        data=data_5d, axes=("T", "H", "W", "C", "S"),
        channel_map=["pred_sb_INTERNAL_SIGNAL"],
        time_col="month_id", id_col="pg_id",
        spatial_cols=("row", "col"), spatial_offset=(0,0)
    )

    # PATH A: The HydraNet Manager Path (Correct)
    data_raw = np.expm1(vh.data)
    vh_raw = VolumeHandler(
        data=data_raw, axes=vh.axes, channel_map=vh.channel_map,
        time_col=vh.time_col, id_col=vh.id_col,
        spatial_cols=vh.spatial_cols, spatial_offset=vh.spatial_offset
    )
    vh_collapsed_A = vh_raw.collapse_to_point(method="mean")
    result_A = vh_collapsed_A.data.flatten()[0]
    assert np.allclose(result_A, 55.0, rtol=1e-5), (
        f"Path A (Inverse then Collapse): expected arithmetic mean of [10.0, 100.0] = 55.0, "
        f"got {result_A:.6f}. "
        f"expm1(log1p([10, 100])) → back to [10, 100] → mean = 55.0 exactly."
    )

    # PATH B: The Wrong Path (Geometric Mean)
    vh_collapsed_B_log = vh.collapse_to_point(method="mean")
    result_B = np.expm1(vh_collapsed_B_log.data.flatten()[0])

    # Conclusion: Results must differ significantly (Jensen's Inequality)
    assert abs(result_A - result_B) > 10.0, (
        f"Jensen's Inequality test failed: Path A (expm1 then mean) = {result_A:.4f}, "
        f"Path B (mean then expm1) = {result_B:.4f}. "
        f"For inputs [10, 100], the arithmetic mean in raw space (55.0) must differ by >10 "
        f"from the geometric-mean equivalent in log space (~31.6). "
        f"If this fails, the two paths have collapsed to the same value."
    )
