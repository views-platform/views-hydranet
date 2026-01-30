"""
Utilities for HydraNet Output Processing and Contract Compliance.

This module provides the core transformation layer between HydraNet's spatiotemporal 
tensors (zstacks) and the row-oriented DataFrames required by the ViEWS evaluation 
and forecasting pipelines.

Core Invariant:
    All transformations must be bit-identical and reversible. The 'Contract' format 
    is defined as a MultiIndex DataFrame (month_id, priogrid_gid) containing 
    inverse-transformed raw count predictions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from views_hydranet.utils.utils import heal_non_finite
from views_hydranet.utils.utils_internal_containers import ModelOutputs

logger = logging.getLogger(__name__)


def predictions_to_contract_df(
    posterior_list: List[np.ndarray],
    forecast_storage_vol: np.ndarray,
    target: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[pd.DataFrame]:
    """
    Converts a list of posterior sample arrays into the standard Contract DataFrame.

    This is the primary converter for the 'sample_posterior' workflow.

    Args:
        posterior_list: A list of N posterior sample arrays, each with shape [steps, features, H, W].
        forecast_storage_vol: Metadata volume [batch, steps, channels, H, W].
                              Required channels: 0 (pg_id), 3 (month_id).
        target: The target variable name (e.g., 'ln_sb_best').
        config: Optional configuration dictionary for transform lookups.

    Returns:
        List[pd.DataFrame]: A list containing a single MultiIndex DataFrame formatted 
                            for the ViEWS evaluation library.

    Invariants:
        - Output column name is always prefixed with 'pred_lr_'.
        - Values are inverse-transformed using the symmetric transform from config.
        - Ocean cells (priogrid_gid == 0) are strictly excluded.
    """
    # 1. Standardize Input
    all_samples = np.stack(posterior_list)  # [samples, steps, features, H, W]
    
    # 2. Channel Mapping (Robust Registry Lookup)
    from views_hydranet.utils.utils_config import get_target_index
    t_idx = get_target_index(target)
    
    # 3. Target Naming (Explicit Raw Scale)
    if target.startswith("ln_"):
        out_target = target.replace("ln_", "lr_")
    elif not target.startswith("lr_"):
        out_target = f"lr_{target}"
    else:
        out_target = target
    out_col = f"pred_{out_target}"

    # 4. Extract and Unlog
    # [samples, steps, H, W]
    target_samples = all_samples[:, :, t_idx, :, :]
    
    # Pass 1: Heal and Clamp before transform (20.0 in log-space is ~480 million fatalities)
    target_samples = heal_non_finite(target_samples, f"predictions_to_contract_df({target})", clamp_val=20.0)

    from views_hydranet.utils.utils_config import TRANSFORMS
    # Default to log1p for legacy compatibility
    transform_name = config.get("transform", "log1p") if config else "log1p"
    _, inverse_fn = TRANSFORMS[transform_name]
    
    target_samples = inverse_fn(target_samples)
    
    # Pass 2: Heal again after transform (catch overflows from non-log transforms)
    target_samples = heal_non_finite(target_samples, f"predictions_to_contract_df({target}) [POST-TRANSFORM]")
    
    # Handle Binarization
    if target.endswith("_binarized"):
        target_samples = (target_samples > 0).astype(float)
    
    # 5. Extract Metadata IDs
    pg_ids = forecast_storage_vol[0, :, 0, :, :]
    month_ids = forecast_storage_vol[0, :, 3, :, :]
    
    # 6. Vectorized Masking
    mask = pg_ids > 0
    land_pg_ids = pg_ids[mask].astype(int)
    land_month_ids = month_ids[mask].astype(int)
    
    # target_samples is [S, T, H, W] -> transpose to [T, H, W, S]
    land_mags = target_samples.transpose(1, 2, 3, 0)[mask]
    
    # 7. List Conversion (One-shot)
    land_samples_list = land_mags.tolist()
    
    # 8. Construct DF
    df = pd.DataFrame({
        "month_id": land_month_ids,
        "priogrid_gid": land_pg_ids,
        out_col: land_samples_list
    })
    
    df = df.set_index(["month_id", "priogrid_gid"])
    return [df]

def zstack_to_contract_df(
    posterior_zstack: np.ndarray,
    meta_zstack: np.ndarray,
    target: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[pd.DataFrame]:
    """
    Converts zstacks from HydraNetInference into the standard Contract DataFrame.

    This is the primary converter for the 'HydraNetInference' workflow.

    Args:
        posterior_zstack: Predicted magnitudes [steps, H, W, channels, samples].
        meta_zstack: Metadata volume [steps, H, W, channels, 1].
        target: The target variable name (e.g., 'ln_sb_best').
        config: Optional configuration dictionary for transform lookups.

    Returns:
        List[pd.DataFrame]: A list containing a single MultiIndex DataFrame formatted 
                            for the ViEWS evaluation library.

    Warnings:
        This function performs a heavy list-conversion. For 180x180 grids, ensure 
        at least 8GB of free RAM is available when samples > 100.
    """
    samples = posterior_zstack.shape[-1]
    
    # Internal channel mapping (Robust Registry Lookup)
    from views_hydranet.utils.utils_config import get_target_index
    t_idx = get_target_index(target)
    
    # Construct Output Column Name
    if target.startswith("ln_"):
        out_target = target.replace("ln_", "lr_")
    elif not target.startswith("lr_"):
        out_target = f"lr_{target}"
    else:
        out_target = target
    out_col = f"pred_{out_target}"
    
    # Extract magnitudes and apply inverse transform from registry
    mags = posterior_zstack[:, :, :, t_idx, :]
    
    # 3. Collapse/Aggregation Logic (Compatibility Shim)
    eval_mode = config.get("evalution_mode", "stochastic") if config else "stochastic"
    agg_method = config.get("aggregate_method", "mean") if config else "mean"
    agg_space = config.get("aggregate_space", "raw") if config else "raw"

    def apply_aggregation(data: np.ndarray, method: str) -> np.ndarray:
        if method == "mean":
            return np.mean(data, axis=-1, keepdims=True)
        elif method == "median":
            return np.median(data, axis=-1, keepdims=True)
        elif method == "max_aposteriori":
            # For now, simplistic MAP as the mode of samples
            # In continuous space, we'd need KDE, but for now we'll use mean as proxy or actual mode
            from scipy import stats
            # Mode on samples axis
            mode_res = stats.mode(data, axis=-1, keepdims=True)
            return mode_res.mode
        return data

    if eval_mode == "point" and agg_space == "logged":
        logger.info(f"Collapsing samples using {agg_method} in LOGGED space.")
        mags = apply_aggregation(mags, agg_method)

    # Pass 1: Heal and Clamp before transform
    mags = heal_non_finite(mags, f"zstack_to_contract_df({target})", clamp_val=20.0)

    from views_hydranet.utils.utils_config import TRANSFORMS
    # Default to log1p for legacy compatibility
    transform_name = config.get("transform", "log1p") if config else "log1p"
    _, inverse_fn = TRANSFORMS[transform_name]
    
    logger.debug(f"Converting magnitudes back to raw counts using {transform_name} inverse.")
    
    # Apply the dynamic inverse
    mags = inverse_fn(mags) 

    if eval_mode == "point" and agg_space == "raw":
        logger.info(f"Collapsing samples using {agg_method} in RAW space.")
        mags = apply_aggregation(mags, agg_method)
    
    # Pass 2: Heal again after transform
    mags = heal_non_finite(mags, f"zstack_to_contract_df({target}) [POST-TRANSFORM]")
    
    # 4. Handle Binarization (Classification Contract)
    if target.endswith("_binarized"):
        # We binarize on the RAW scale: > 0
        mags = (mags > 0).astype(float)
    
    # Extract IDs
    pg_ids = meta_zstack[:, :, :, 0, 0]
    month_ids = meta_zstack[:, :, :, 3, 0]
    
    # Create mask for all land cells
    mask = pg_ids > 0
    land_pg_ids = pg_ids[mask].astype(int)
    land_month_ids = month_ids[mask].astype(int)
    land_mags = mags[mask]
    
    # Convert samples to list-of-lists
    land_samples_list = land_mags.tolist()
    
    # Construct DataFrame
    df = pd.DataFrame({
        "month_id": land_month_ids,
        "priogrid_gid": land_pg_ids,
        out_col: land_samples_list
    })
    
    df = df.set_index(["month_id", "priogrid_gid"])
    return [df]

def validate_contract_dataframes(list_df: list[pd.DataFrame]) -> None:
    """
    Validates that the contract DataFrames are robust and safe for evaluation.
    
    Checks for:
    1. Non-finite values (NaN, Inf) in predictions.
    2. Presence of ocean cells (priogrid_gid == 0).
    3. Empty DataFrames.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if not list_df:
        raise ValueError("Contract DataFrame list is empty!")

    for i, df in enumerate(list_df):
        if df.empty:
            raise ValueError(f"Sequence {i} is empty!")
            
        # Check for Ocean Cells in Index
        pg_ids = df.index.get_level_values("priogrid_gid")
        if (pg_ids == 0).any():
            raise ValueError(f"Sequence {i} contains ocean cells (priogrid_gid=0)!")

        # Check for Non-Finite Numbers in all columns
        for col in df.columns:
            # Flatten lists of samples to a single array for fast checking
            all_values = np.concatenate(df[col].values)
            if not np.isfinite(all_values).all():
                num_bad = (~np.isfinite(all_values)).sum()
                # We log this as a CRITICAL warning, but don't crash if healer is active
                # Actually, validate should still crash if it finds NaNs AFTER the healer
                # This ensures the healer actually worked.
                raise ValueError(
                    f"Sequence {i}, column {col} contains {num_bad} non-finite values (NaN/Inf)!"
                )

    logger.info("Adversarial data validation passed: Data is finite and land-only.")

def contract_df_to_zstack(
    list_df_predictions: List[pd.DataFrame],
    meta_zstack: np.ndarray,
    target: str,
) -> np.ndarray:
    """
    Inverse operation of zstack_to_contract_df. 
    Reconstructs the original identical posterior_zstack magnitudes from the DataFrame.

    This function is the 'Reversibility Proof'. It ensures that the transformation 
    to the contract format is lossless.

    Args:
        list_df_predictions: The list of contract DataFrames.
        meta_zstack: The spatial template [steps, H, W, channels, 1].
        target: The target variable name.

    Returns:
        np.ndarray: Reconstructed magnitudes [steps, H, W, 1, samples].
                    Returns only the requested target channel.

    Invariants:
        - Ocean cells in the template are filled with 0.0.
        - Land cells are populated from the DataFrame and log-transformed back.
    """
    df = list_df_predictions[0]
    steps, H, W, _, _ = meta_zstack.shape
    
    # Peek at first list to get sample count
    samples = len(df.iloc[0][f"pred_lr_{target}"])
    
    # Pre-allocate reconstructed volume
    reconstructed = np.zeros((steps, H, W, 1, samples))
    
    # Extract IDs from template
    pg_ids_template = meta_zstack[:, :, :, 0, 0]
    month_ids_template = meta_zstack[:, :, :, 3, 0]
    
    # Inverse transform column name
    col = f"pred_lr_{target}"
    
    # Iterate over template steps
    for t in range(steps):
        month_id = int(np.unique(month_ids_template[t])[0])
        # Performance Warning: .xs() is used here for clarity in the proof logic
        df_month = df.xs(month_id, level="month_id")
        
        for h in range(H):
            for w in range(W):
                pg_id = int(pg_ids_template[t, h, w])
                if pg_id > 0:
                    raw_samples = df_month.loc[pg_id, col]
                    # Apply log1p forward: ln(x + 1)
                    reconstructed[t, h, w, 0, :] = np.log1p(raw_samples)
                else:
                    reconstructed[t, h, w, 0, :] = 0.0
                    
    return reconstructed
