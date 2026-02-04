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
    
    Matches the pipeline configuration 1-to-1: consumes the 'transform' dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with the explicit 'transform' dictionary.
        """
        self._transform_config = config.get("transform", {})
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler (locks state) and applies transformations.
        """
        if self._is_fitted:
            raise RuntimeError("FeatureScaler is a one-shot gate and is already fitted.")

        df_out = df.copy()

        total_scaled = sum(len(cols) for cols in self._transform_config.values())
        logger.info(f"🚀 FeatureScaler: Entering Semantic Space ({total_scaled} features to transform)")

        for method, columns in self._transform_config.items():
            if not columns or method not in TRANSFORMS:
                continue

            forward_func, _ = TRANSFORMS[method]
            logger.info(f"  → Applying [{method:.10}] to {len(columns)} features")

            for col in columns:
                if col not in df_out.columns:
                    raise ValueError(
                        f"[CRITICAL DATA ERROR] FeatureScaler Fit Failure!\n"
                        f"Requested feature '{col}' missing from Raw DataFrame."
                    )

                df_out[col] = forward_func(df_out[col])

        self._is_fitted = True
        self._log_data_state(df_out, space="SEMANTIC")

        return df_out

    def _log_data_state(self, df: pd.DataFrame, space: str = "SEMANTIC") -> None:
        """Internal: Generates a beautiful diagnostic report."""
        stats = []

        method_lookup = {}
        for method, cols in self._transform_config.items():
            for col in cols:
                method_lookup[col] = method

        for col in self.configured_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                method = method_lookup.get(col, "unknown")
                stats.append(f"  [{method:^10}] {col:.<30} min: {c_min:>10.4f} | max: {c_max:>10.4f}")

        report = "\n" + "💠" + "="*78 + "\n"
        report += f"  FEATURE SCALER: {space} SPACE REPORT\n"
        report += "  " + "-"*76 + "\n"
        if stats:
            report += "\n".join(stats)
        else:
            report += "  (No transformed features found)"
        report += "\n" + "💠" + "="*78 + "\n"
        logger.info(report)

    @property
    def configured_columns(self) -> List[str]:
        """Returns a flat list of all columns configured for scaling."""
        all_cols = []
        for cols in self._transform_config.values():
            all_cols.extend(cols)
        return all_cols

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformations to restore the Raw count space.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler Contract Violation: Must be FITTED before inverse pass.")

        df_out = df.copy()
        logger.info(f"🔙 FeatureScaler: Returning to Raw Space ({len(self.configured_columns)} features to invert)")

        for method, columns in self._transform_config.items():
            if not columns:
                continue

            _, inverse_func = TRANSFORMS[method]
            logger.info(f"  ← Reversing [{method:.<10}]")

            for col in columns:
                if col not in df_out.columns:
                    continue

                df_out[col] = inverse_func(df_out[col])

        self._log_data_state(df_out, space="RAW")
        return df_out
