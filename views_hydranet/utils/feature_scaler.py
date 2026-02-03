"""
Declarative and Stateful Feature Scaling for HydraNet.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from views_hydranet.utils.utils_config import TRANSFORMS

logger = logging.getLogger(__name__)

class FeatureScaler:
    """
    A one-shot stateful gateway for DataFrame feature transformations.
    
    Identifies and reverses non-linear scaling with bit-perfect precision.
    Strictly follows ADR 019: Fail Loud and Proud on missing columns.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with a config containing transformation keys (e.g., 'log1p', 'asinh').
        """
        self._config = {}
        # Only track methods recognized by the global registry
        for method in TRANSFORMS.keys():
            self._config[method] = config.get(method, [])

        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler (locks state) and applies transformations.
        """
        if self._is_fitted:
            raise RuntimeError("FeatureScaler is a one-shot gate and is already fitted.")

        df_out = df.copy()
        
        total_scaled = sum(len(cols) for cols in self._config.values())
        logger.info(f"FeatureScaler: Starting FIT-TRANSFORM ({total_scaled} columns)")

        for method, columns in self._config.items():
            forward_func, _ = TRANSFORMS[method]

            for col in columns:
                if col not in df_out.columns:
                    raise ValueError(
                        f"[CRITICAL DATA ERROR] FeatureScaler Fit Failure!\n"
                        f"Requested column '{col}' missing from Raw DataFrame."
                    )
                
                df_out[col] = forward_func(df_out[col])

        self._is_fitted = True
        self._log_data_state(df_out)

        return df_out

    def _log_data_state(self, df: pd.DataFrame) -> None:
        """Internal: Generates a diagnostic report of the Semantic Space."""
        stats = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                stats.append(f"  - {col:.<40} [{c_min:>12.4f}, {c_max:>12.4f}]")
        
        report = "\n" + "="*80 + "\n"
        report += " FEATURE SCALER: SEMANTIC SPACE REPORT\n"
        report += "-"*80 + "\n"
        report += "\n".join(stats)
        report += "\n" + "="*80
        logger.info(report)

    @property
    def configured_columns(self) -> List[str]:
        """Returns a flat list of all columns explicitly configured for scaling."""
        all_cols = []
        for cols in self._config.values():
            all_cols.extend(cols)
        return all_cols

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformations to restore the Raw count space.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler Contract Violation: Must be FITTED before inverse pass.")

        df_out = df.copy()
        logger.info("FeatureScaler: Starting INVERSE-TRANSFORM (Raw Exit)")

        for method, columns in self._config.items():
            _, inverse_func = TRANSFORMS[method]

            for col in columns:
                if col not in df_out.columns:
                    raise ValueError(
                        f"[CRITICAL DATA ERROR] FeatureScaler Inverse Failure!\n"
                        f"Transformed column '{col}' missing from Semantic DataFrame."
                    )

                df_out[col] = inverse_func(df_out[col])

        return df_out
