"""
Declarative and Stateful Feature Scaling for HydraNet.
"""

import logging
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureScaler:
    """
    A one-shot stateful gateway for DataFrame feature transformations.
    
    This class enforces the boundary between Raw and Semantic data spaces.
    It uses a declarative config to apply and reverse math on specific columns.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with a config containing 'log1p', 'asinh', 'identity' keys.
        """
        self._config = {}
        for method in ["log1p", "asinh", "identity"]:
            self._config[method] = config.get(method, [])

        self._is_fitted = False

        # Math Registry: (Forward, Inverse)
        self._methods: Dict[str, tuple[Callable, Callable]] = {
            "log1p": (np.log1p, np.expm1),
            "asinh": (np.arcsinh, np.sinh),
            "identity": (lambda x: x, lambda x: x)
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler (locks state) and applies transformations.
        Returns a DataFrame with all original columns intact.
        """
        if self._is_fitted:
            raise RuntimeError("FeatureScaler is one-shot and already fitted.")

        df_out = df.copy()
        
        # Calculate how many features we are actually scaling
        total_scaled = sum(len(cols) for cols in self._config.values())
        logger.info(f"FeatureScaler: FIT-TRANSFORM ({total_scaled} features)")

        for method, columns in self._config.items():
            if method not in self._methods:
                raise ValueError(f"FeatureScaler: Unknown method '{method}'")

            forward_func, _ = self._methods[method]

            for col in columns:
                if col not in df_out.columns:
                    raise ValueError(f"FeatureScaler Fit Error: Column '{col}' missing from DataFrame.")
                
                # Apply transformation to the existing column (preserving other columns)
                df_out[col] = forward_func(df_out[col])

        self._is_fitted = True

        # AUDIT LOG: Comprehensive view of the resulting DataFrame
        stats = []
        for col in df_out.columns:
            if pd.api.types.is_numeric_dtype(df_out[col]):
                c_min = df_out[col].min()
                c_max = df_out[col].max()
                stats.append(f"  - {col:.<40} [{c_min:>12.4f}, {c_max:>12.4f}]")
            else:
                stats.append(f"  - {col:.<40} [Non-numeric]")
        
        report = "\n" + "="*80 + "\n"
        report += " FEATURE SCALER: DATA STATE REPORT (Semantic Space)\n"
        report += "-"*80 + "\n"
        report += "\n".join(stats)
        report += "\n" + "="*80
        
        logger.info(report)

        return df_out

    @property
    def configured_columns(self) -> List[str]:
        """Returns a flat list of all columns configured for scaling."""
        all_cols = []
        for cols in self._config.values():
            all_cols.extend(cols)
        return all_cols

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformations using the initial configuration.
        Fails if columns are missing.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before inverse_transform.")

        df_out = df.copy()
        logger.info("FeatureScaler: INVERSE-TRANSFORM (Raw Exit)")

        for method, columns in self._config.items():
            _, inverse_func = self._methods[method]

            for col in columns:
                if col not in df_out.columns:
                    raise ValueError(f"FeatureScaler Inverse Error: Column '{col}' missing.")

                logger.info(f"FeatureScaler: {col} <- {method} (Inverse)")
                df_out[col] = inverse_func(df_out[col])

        return df_out