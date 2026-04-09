import logging
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrainingForensics:
    """
    Independent Forensic Auditor for HydraNet training performance.
    Calculates and stores metrics trajectory decoupled from the optimization logic.
    Namespaces internal storage to prevent collisions between Reg and Cls targets.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.reg_targets = config.get("regression_targets", [])
        self.cls_targets = config.get("classification_targets", [])

        # Pull and filter metrics (strip redundant calibration pulses)
        raw_reg_metrics = config.get("regression_metrics", ["mse"])
        self.reg_metrics = [m for m in raw_reg_metrics if m.lower() != "y_hat_bar"]

        self.cls_metrics = config.get("classification_metrics", ["ap"])

        # 1. Threshold Integrity Guard
        forbidden = ["f1", "accuracy", "recall", "precision"]
        for m in self.reg_metrics + self.cls_metrics:
            if m.lower() in forbidden:
                err_msg = (
                    f"Metric '{m}' requires a threshold to be specified in the config; "
                    f"this is currently not supported by TrainingForensics."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)

        # 2. Initialize Histories with NAMESPACED Keys (ADR 003 Explicit)
        # We use "REG:name" and "CLS:name" to prevent collisions
        self.history: Dict[str, Dict[str, List[float]]] = {}
        self.target_map: Dict[str, Dict[str, Any]] = {}  # Maps namespaced key to metadata
        for target in self.reg_targets:
            key = f"REG:{target}"
            self.target_map[key] = {"type": "REG", "name": target, "metrics": self.reg_metrics}
            self._init_target_history(key, self.reg_metrics)

        for target in self.cls_targets:
            key = f"CLS:{target}"
            self.target_map[key] = {"type": "CLS", "name": target, "metrics": self.cls_metrics}
            self._init_target_history(key, self.cls_metrics)

        # 3. Running Totals for Global Bias
        all_keys = list(self.target_map.keys())
        self.running_sum_y = {key: 0.0 for key in all_keys}
        self.running_sum_yh = {key: 0.0 for key in all_keys}

        # 4. Lesson Accumulators (Reset every lesson)
        self._reset_accumulators()

    def _init_target_history(self, key: str, metrics: List[str]):
        """Helper to initialize history for a namespaced key."""
        self.history[key] = {"bias_instant": [], "bias_running": [], "y_bar": [], "y_hat_bar": []}
        for m in metrics:
            self.history[key][m] = []

    def _reset_accumulators(self):
        """Prepares empty buffers for a new lesson."""
        all_keys = list(self.target_map.keys())
        self.lesson_y = {key: [] for key in all_keys}
        self.lesson_yh = {key: [] for key in all_keys}

    def record(self, namespaced_key: str, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        """
        Records a single window pass.
        namespaced_key: e.g. 'REG:lr_sb_best'
        """
        if namespaced_key not in self.lesson_y:
            err_msg = f"TrainingForensics: Key '{namespaced_key}' not initialized."

            logger.error(err_msg)

            raise KeyError(err_msg)

        self.lesson_y[namespaced_key].append(y.detach().cpu().numpy().flatten())
        self.lesson_yh[namespaced_key].append(y_hat.detach().cpu().numpy().flatten())

    def finalize_lesson(self) -> None:
        """
        Reduces lesson buffers into final metrics and updates history.
        """
        for key, meta in self.target_map.items():
            if not self.lesson_y[key]:
                for m in self.history[key].keys():
                    last_val = self.history[key][m][-1] if self.history[key][m] else 0.0
                    self.history[key][m].append(last_val)
                continue

            y_all = np.concatenate(self.lesson_y[key])
            yh_all = np.concatenate(self.lesson_yh[key])

            # 1. Calculate Metrics
            if meta["type"] == "REG":
                for m in self.reg_metrics:
                    val = self._calculate_reg_metric(m, y_all, yh_all)
                    self.history[key][m].append(val)
            else:
                for m in self.cls_metrics:
                    val = self._calculate_cls_metric(m, y_all, yh_all)
                    self.history[key][m].append(val)

            # 2. Calculate Bias (Dual Mode)
            sum_y = np.sum(y_all)
            sum_yh = np.sum(yh_all)

            self.history[key]["y_bar"].append(np.mean(y_all))
            self.history[key]["y_hat_bar"].append(np.mean(yh_all))

            instant_bias = sum_yh / sum_y if sum_y > 0 else 1.0
            self.history[key]["bias_instant"].append(instant_bias)

            self.running_sum_y[key] += sum_y
            self.running_sum_yh[key] += sum_yh
            running_bias = (
                self.running_sum_yh[key] / self.running_sum_y[key]
                if self.running_sum_y[key] > 0
                else 1.0
            )
            self.history[key]["bias_running"].append(running_bias)

        self._reset_accumulators()

    def get_dossier(self, namespaced_key: str) -> Dict[str, List[float]]:
        """Returns the historical record for a namespaced key."""
        return self.history.get(namespaced_key, {})

    def _calculate_reg_metric(self, name: str, y: np.ndarray, yh: np.ndarray) -> float:
        if name.lower() == "mse":
            return np.mean((y - yh) ** 2)
        if name.lower() == "mae":
            return np.mean(np.abs(y - yh))
        # Handle RMSLE, MSLE, CRPS placeholder
        if name.lower() == "rmsle":
            return np.sqrt(np.mean((np.log1p(y) - np.log1p(yh)) ** 2))
        if name.lower() == "msle":
            return np.mean((np.log1p(y) - np.log1p(yh)) ** 2)

        err_msg = f"Unknown regression metric: '{name}'"

        logger.error(err_msg)

        raise ValueError(err_msg)

    def _calculate_cls_metric(self, name: str, y: np.ndarray, yh: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score, roc_auc_score

        # Ensure binary y for sklearn
        y_bin = (y > 0).astype(int)

        # Guard: sklearn metrics fail if only one class is present
        if len(np.unique(y_bin)) < 2:
            if name.lower() == "auc":
                return 0.5
            if name.lower() == "ap":
                return 0.0
            return 0.0

        if name.lower() == "ap":
            try:
                return average_precision_score(y_bin, yh)
            except ValueError:
                return 0.0
        if name.lower() == "auc":
            try:
                return roc_auc_score(y_bin, yh)
            except ValueError:
                return 0.5
        err_msg = f"Unknown classification metric: '{name}'"

        logger.error(err_msg)

        raise ValueError(err_msg)
