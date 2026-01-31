import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

class ScalingEngine:
    """
    Formal Numerical Transformation Layer for HydraNet.
    
    Ensures mathematical symmetry between Training (Scaling) and 
    Evaluation (Unscaling).
    """

    # Registry of symmetric pairs: (Forward, Inverse)
    TRANSFORMS: Dict[str, Tuple[Callable, Callable]] = {
        "log1p": (np.log1p, np.expm1),
        "asinh": (np.arcsinh, np.sinh),
        "identity": (lambda x: x, lambda x: x)
    }

    def __init__(self, transform_name: str = "log1p"):
        if transform_name not in self.TRANSFORMS:
            raise ValueError(f"Unsupported transform: {transform_name}. Available: {list(self.TRANSFORMS.keys())}")

        self.name = transform_name
        self.forward_fn, self.inverse_fn = self.TRANSFORMS[transform_name]

    def scale(self, data: np.ndarray | torch.Tensor, context: str) -> np.ndarray:
        """
        Applies forward transformation (Raw -> Scaled).
        """
        # Convert Torch to NumPy if necessary
        is_torch = isinstance(data, torch.Tensor)
        data_np = data.detach().cpu().numpy() if is_torch else data.copy()

        if data_np.size == 0:
            return data_np

        # Pure Transform
        scaled = self.forward_fn(data_np)

        # Audit log (Sample check - NO INTERVENTION)
        logger.info(f"SCALING AUDIT [{context}]: Applied {self.name}. Max {np.max(data_np):.4f} -> {np.max(scaled):.4f}")

        return scaled

    def unscale(self, data: np.ndarray | torch.Tensor, context: str) -> np.ndarray:
        """
        Applies inverse transformation (Scaled -> Raw).
        """
        # Convert Torch to NumPy if necessary
        is_torch = isinstance(data, torch.Tensor)
        data_np = data.detach().cpu().numpy() if is_torch else data.copy()

        # Pure Inverse Transform
        unscaled = self.inverse_fn(data_np)

        return unscaled

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "ScalingEngine":
        """Factory method to build engine from HydraNetConfig. Handles None."""
        if config is None:
            return cls()

        name = config.get("transform", "log1p")
        return cls(transform_name=name)
