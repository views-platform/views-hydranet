"""
Declarative and Stateful Feature Scaling for HydraNet.
"""

import logging
from typing import Dict, List, Callable
import numpy as np
import pandas as pd
from typing import Any
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
        Fails if columns are missing.
        """
        if self._is_fitted:
            raise RuntimeError("FeatureScaler is one-shot and already fitted.")

        df_out = df.copy()
        logger.info("FeatureScaler: FIT-TRANSFORM (Semantic Entry)")

        for method, columns in self._config.items():
            if method not in self._methods:
                raise ValueError(f"FeatureScaler: Unknown method '{method}'")

            forward_func, _ = self._methods[method]

            for col in columns:
                if col not in df_out.columns:
                    raise ValueError(f"FeatureScaler Fit Error: Column '{col}' missing from DataFrame.")
                
                logger.info(f"FeatureScaler: {col} -> {method}")
                df_out[col] = forward_func(df_out[col])

        self._is_fitted = True
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
