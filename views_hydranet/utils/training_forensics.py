
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

class TrainingForensics:
    """
    Independent Forensic Auditor for HydraNet training performance.
    Calculates and stores metrics trajectory decoupled from the optimization logic.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.reg_targets = config.get("regression_targets", [])
        self.cls_targets = config.get("classification_targets", [])
        self.reg_metrics = config.get("regression_metrics", ["mse"])
        self.cls_metrics = config.get("classification_metrics", ["ap"])
        
        # 1. Threshold Integrity Guard
        forbidden = ["f1", "accuracy", "recall", "precision"]
        for m in self.reg_metrics + self.cls_metrics:
            if m.lower() in forbidden:
                err_msg = (f"Metric '{m}' requires a threshold to be specified in the config; "
                           f"this is currently not supported by TrainingForensics.")
                logger.error(err_msg)
                raise ValueError(err_msg)

        # 2. Initialize Histories
        # {target: {metric: [values]}}
        self.history = {}
        all_targets = self.reg_targets + self.cls_targets
        for target in all_targets:
            self.history[target] = {
                "bias_instant": [],
                "bias_running": [],
                "y_bar": [],
                "y_hat_bar": []
            }
            metrics = self.reg_metrics if target in self.reg_targets else self.cls_metrics
            for m in metrics:
                self.history[target][m] = []

        # 3. Running Totals for Global Bias
        self.running_sum_y = {target: 0.0 for target in all_targets}
        self.running_sum_yh = {target: 0.0 for target in all_targets}

        # 4. Lesson Accumulators (Reset every lesson)
        self._reset_accumulators()

    def _reset_accumulators(self):
        """Prepares empty buffers for a new lesson."""
        all_targets = self.reg_targets + self.cls_targets
        self.lesson_y = {target: [] for target in all_targets}
        self.lesson_yh = {target: [] for target in all_targets}

    def record(self, target_name: str, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        """
        Records a single window pass. 
        Stores raw values in lesson buffers for end-of-lesson reduction.
        """
        if target_name not in self.lesson_y:
            raise KeyError(f"TrainingForensics: Target '{target_name}' not initialized.")

        # Detach and move to CPU/NumPy immediately to free GPU and avoid graph interference
        # We flatten to [N] to simplify aggregation
        self.lesson_y[target_name].append(y.detach().cpu().numpy().flatten())
        self.lesson_yh[target_name].append(y_hat.detach().cpu().numpy().flatten())

    def finalize_lesson(self) -> None:
        """
        Reduces lesson buffers into final metrics and updates history.
        """
        for target in self.history.keys():
            if not self.lesson_y[target]:
                # Skip targets that were not seen this lesson (e.g. inactive classification)
                # Ensure we append a nan or last value to keep history length aligned?
                # Actually, ADR 014 says we only optimize what we see. 
                # To keep plots aligned, we'll append the last value if history exists, else 0.
                for m in self.history[target].keys():
                     last_val = self.history[target][m][-1] if self.history[target][m] else 0.0
                     self.history[target][m].append(last_val)
                continue

            y_all = np.concatenate(self.lesson_y[target])
            yh_all = np.concatenate(self.lesson_yh[target])
            
            # 1. Calculate Metrics
            if target in self.reg_targets:
                for m in self.reg_metrics:
                    val = self._calculate_reg_metric(m, y_all, yh_all)
                    self.history[target][m].append(val)
            else:
                for m in self.cls_metrics:
                    val = self._calculate_cls_metric(m, y_all, yh_all)
                    self.history[target][m].append(val)

            # 2. Calculate Bias (Dual Mode)
            sum_y = np.sum(y_all)
            sum_yh = np.sum(yh_all)
            
            # Store raw means
            self.history[target]["y_bar"].append(np.mean(y_all))
            self.history[target]["y_hat_bar"].append(np.mean(yh_all))
            
            # Instantaneous
            instant_bias = sum_yh / sum_y if sum_y > 0 else 1.0
            self.history[target]["bias_instant"].append(instant_bias)
            
            # Running
            self.running_sum_y[target] += sum_y
            self.running_sum_yh[target] += sum_yh
            running_bias = self.running_sum_yh[target] / self.running_sum_y[target] if self.running_sum_y[target] > 0 else 1.0
            self.history[target]["bias_running"].append(running_bias)

        self._reset_accumulators()

    def get_dossier(self, target_name: str) -> Dict[str, List[float]]:
        """Returns the historical record for a specific feature."""
        return self.history.get(target_name, {})

    def _calculate_reg_metric(self, name: str, y: np.ndarray, yh: np.ndarray) -> float:
        if name.lower() == "mse":
            return np.mean((y - yh)**2)
        if name.lower() == "mae":
            return np.mean(np.abs(y - yh))
        return 0.0

    def _calculate_cls_metric(self, name: str, y: np.ndarray, yh: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        # Ensure binary y for sklearn
        y_bin = (y > 0).astype(int)
        
        if name.lower() == "ap":
            try:
                return average_precision_score(y_bin, yh)
            except ValueError: return 0.0 # No positive samples
        if name.lower() == "auc":
            try:
                return roc_auc_score(y_bin, yh)
            except ValueError: return 0.5 # No positive samples
        return 0.0
