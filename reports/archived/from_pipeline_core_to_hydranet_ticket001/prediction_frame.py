import numpy as np
from typing import Dict, Set
import pandas as pd

class PredictionFrame:
    """
    The canonical, framework-agnostic representation of a model's inference output.
    
    Encapsulates predictions and their associated spatiotemporal metadata,
    serving as the universal transport object between Models and the Pipeline Core.
    
    Attributes:
        y_pred (np.ndarray): 2D array of predictions of shape (N, S).
        identifiers (Dict[str, np.ndarray]): Mapping of dimension names to 1D arrays of shape (N,).
    """

    REQUIRED_IDENTIFIERS: Set[str] = {"time", "unit"}

    def __init__(self, y_pred: np.ndarray, identifiers: Dict[str, np.ndarray]):
        """
        Initialize PredictionFrame with predictions and metadata.
        
        Args:
            y_pred: NumPy array of shape (N, S).
            identifiers: Dictionary mapping keys (e.g., 'time', 'unit') to 1D arrays of length N.
            
        Raises:
            ValueError: If shapes are inconsistent, y_pred is not 2D, or required 
                        identifiers are missing or contain NaNs.
        """
        self._validate_input(y_pred, identifiers)
        self.y_pred = y_pred
        self.identifiers = identifiers

    def _validate_input(self, y_pred: np.ndarray, identifiers: Dict[str, np.ndarray]) -> None:
        # 1. Check Dimensions
        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must be a 2D array of shape (N, S). Got ndim={y_pred.ndim}")

        n_rows = y_pred.shape[0]
        n_samples = y_pred.shape[1]
        if n_rows == 0:
            raise ValueError("y_pred must have at least one row. Got shape (0, ...).")
        if n_samples == 0:
            raise ValueError("y_pred must have at least one sample column. Got shape (..., 0).")

        # 2. Check Required Keys
        for req in self.REQUIRED_IDENTIFIERS:
            if req not in identifiers:
                raise ValueError(f"Missing required identifier: '{req}'")

        # 3. Check Shape Consistency and NaNs
        for key, arr in identifiers.items():
            if len(arr) != n_rows:
                raise ValueError(
                    f"Shape mismatch: identifier '{key}' has length {len(arr)} "
                    f"but y_pred has {n_rows} rows."
                )
            if np.any(pd.isna(arr)):
                raise ValueError(f"NaN detected in identifier '{key}'. Identifiers must be complete.")

    @property
    def n_rows(self) -> int:
        """Return the number of observation rows (N)."""
        return self.y_pred.shape[0]

    @property
    def sample_count(self) -> int:
        """Return the number of samples per observation (S)."""
        return self.y_pred.shape[1]

    @property
    def identifier_keys(self) -> Set[str]:
        """Return the set of available identifier keys."""
        return set(self.identifiers.keys())

    def __repr__(self) -> str:
        return (
            f"PredictionFrame(n_rows={self.n_rows}, "
            f"sample_count={self.sample_count}, "
            f"identifiers={list(self.identifier_keys)})"
        )
