import logging
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Distribution-family parameter-health thresholds (pre-registered F1 falsifier; 05_analysis_plan).
# The cross-cell coefficient of variation (std/mean) of per-cell θ over ACTIVE cells:
THETA_HEALTH_COV = 0.10  # pred-1: a healthy heteroscedastic head has θ std/mean > 0.10
THETA_COLLAPSE_COV = 0.02  # F1 falsifier: θ (or π) collapsing to ~constant, std/mean < 0.02
_PARAM_EPS = 1e-8


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

        # 3b. Distribution-family parameter-health (ADR-067 / C-213). Lazily populated: a target's
        # param-health stat names are registered the first time `record_params` is called for it
        # (so n_params is known — nb has no π). Persistent across lessons (NOT reset).
        self._param_health_keys: Dict[str, List[str]] = {}

        # 4. Lesson Accumulators (Reset every lesson)
        self._reset_accumulators()

    # Horizon-step slices for the per-lesson dossier columns (t=0 / early / late vs the full-window
    # aggregate). Lets the dossier show that t=0 (first forecast step) is calibrated while later
    # rollout steps inflate — hidden in the full-window average.
    HORIZON_SLICES = ("t0", "early", "late")

    def _init_target_history(self, key: str, metrics: List[str]):
        """Helper to initialize history for a namespaced key."""
        self.history[key] = {"bias_instant": [], "bias_running": [], "y_bar": [], "y_hat_bar": []}
        for m in metrics:
            self.history[key][m] = []
        # Per-horizon-slice series (t0/early/late): magnitude + calibration + metrics over lessons.
        for s in self.HORIZON_SLICES:
            self.history[key][f"{s}_y_bar"] = []
            self.history[key][f"{s}_y_hat_bar"] = []
            self.history[key][f"{s}_bias"] = []
            for m in metrics:
                self.history[key][f"{s}_{m}"] = []

    def _reset_accumulators(self):
        """Prepares empty buffers for a new lesson."""
        all_keys = list(self.target_map.keys())
        self.lesson_y = {key: [] for key in all_keys}
        self.lesson_yh = {key: [] for key in all_keys}
        # Per-window-timestep buffers: step_idx -> list of flat arrays (the horizon split).
        self.lesson_y_step = {key: {} for key in all_keys}
        self.lesson_yh_step = {key: {} for key in all_keys}
        # Distribution-family param-health buffers (activated params + truth for active-cell mask).
        self.lesson_params = {key: [] for key in all_keys}
        self.lesson_params_y = {key: [] for key in all_keys}
        self.lesson_params_step = {key: {} for key in all_keys}
        self.lesson_params_y_step = {key: {} for key in all_keys}

    def record(
        self,
        namespaced_key: str,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        step_idx: int | None = None,
    ) -> None:
        """
        Records a single window-timestep pass.
        namespaced_key: e.g. 'REG:lr_sb_best'
        step_idx: the timestep index within the window (0 = first forecast step). When provided,
            the dossier horizon columns (t0/early/late) are populated; None ⇒ full-window only.
        """
        if namespaced_key not in self.lesson_y:
            err_msg = f"TrainingForensics: Key '{namespaced_key}' not initialized."

            logger.error(err_msg)

            raise KeyError(err_msg)

        y_flat = y.detach().cpu().numpy().flatten()
        yh_flat = y_hat.detach().cpu().numpy().flatten()
        self.lesson_y[namespaced_key].append(y_flat)
        self.lesson_yh[namespaced_key].append(yh_flat)
        if step_idx is not None:
            self.lesson_y_step[namespaced_key].setdefault(step_idx, []).append(y_flat)
            self.lesson_yh_step[namespaced_key].setdefault(step_idx, []).append(yh_flat)

    @staticmethod
    def _param_health_stat_names(n_params: int) -> List[str]:
        """Which param-health series a family with ``n_params`` channels has (θ=idx1, π=idx2)."""
        names = ["mu_bar", "theta_cov"]
        if n_params >= 3:  # zinb: structural π present
            names += ["pi_bar", "pi_min", "pi_max"]
        return names

    def record_params(
        self,
        namespaced_key: str,
        params: torch.Tensor,
        y: torch.Tensor,
        step_idx: int | None = None,
    ) -> None:
        """Record a family head's per-cell **activated** params for the parameter-health frame.

        ``params``: activated ``[..., n_params]`` (μ=idx0, θ=idx1, π=idx2 — the locked
        ``activate()`` order). ``y``: the log1p-space truth for the same cells (drives the
        active-cell θ mask). Only distribution families (nb/zinb) call this; point heads never do
        (so no param-health rows). Stat names are registered lazily on the first call (n_params).
        """
        if namespaced_key not in self.lesson_y:
            err_msg = f"TrainingForensics: Key '{namespaced_key}' not initialized."
            logger.error(err_msg)
            raise KeyError(err_msg)

        p = params.detach().cpu().numpy()
        p = p.reshape(-1, p.shape[-1])  # [N, n_params]
        yv = y.detach().cpu().numpy().reshape(-1)  # [N]

        if namespaced_key not in self._param_health_keys:
            stat_names = self._param_health_stat_names(p.shape[1])
            self._param_health_keys[namespaced_key] = stat_names
            for sn in stat_names:
                self.history[namespaced_key].setdefault(sn, [])
                for s in self.HORIZON_SLICES:
                    self.history[namespaced_key].setdefault(f"{s}_{sn}", [])

        self.lesson_params[namespaced_key].append(p)
        self.lesson_params_y[namespaced_key].append(yv)
        if step_idx is not None:
            self.lesson_params_step[namespaced_key].setdefault(step_idx, []).append(p)
            self.lesson_params_y_step[namespaced_key].setdefault(step_idx, []).append(yv)

    def _reduce_param_health(self, params: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Reduce per-cell activated params → the param-health stats. θ CoV is over ACTIVE
        (truth>0) cells (where θ is identified; pred-1/F1); π stats over all cells (degeneracy)."""
        out: Dict[str, float] = {"mu_bar": float(np.mean(params[:, 0]))}
        theta = params[:, 1]
        active = y > 0
        th = theta[active] if bool(active.any()) else theta
        m = float(np.mean(th))
        out["theta_cov"] = float(np.std(th) / m) if m > _PARAM_EPS else 0.0
        if params.shape[1] >= 3:
            pi = params[:, 2]
            out["pi_bar"] = float(np.mean(pi))
            out["pi_min"] = float(np.min(pi))
            out["pi_max"] = float(np.max(pi))
        return out

    def finalize_lesson(self) -> None:
        """
        Reduces lesson buffers into final metrics and updates history.
        """
        for key, meta in self.target_map.items():
            if not self.lesson_y[key]:
                # Param-health keys (record_params) are carried/computed by the dedicated §4 block
                # below — exclude them here so they are not double-appended.
                ph_keys = {
                    sk
                    for sn in self._param_health_keys.get(key, [])
                    for sk in [sn, *(f"{s}_{sn}" for s in self.HORIZON_SLICES)]
                }
                for m in self.history[key].keys():
                    if m in ph_keys:
                        continue
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

            # 3. Per-horizon-slice reduction (t0/early/late) for the dossier columns. Pick
            # representative timestep indices from those recorded this lesson; if none were tagged
            # with a step_idx, carry the last value so the per-slice series stay lesson-aligned.
            # "early" uses the first-third index (steps[len//3]) to match the eval autoregressive
            # forensic's T//3 convention, so "early" means the same horizon across both plots.
            steps = sorted(self.lesson_y_step[key].keys())
            chosen = (
                {"t0": steps[0], "early": steps[len(steps) // 3], "late": steps[-1]}
                if steps
                else {}
            )
            for sname in self.HORIZON_SLICES:
                stat_keys = [f"{sname}_y_bar", f"{sname}_y_hat_bar", f"{sname}_bias"] + [
                    f"{sname}_{m}" for m in meta["metrics"]
                ]
                if sname not in chosen:
                    for sk in stat_keys:
                        self.history[key][sk].append(
                            self.history[key][sk][-1] if self.history[key][sk] else 0.0
                        )
                    continue
                ys = np.concatenate(self.lesson_y_step[key][chosen[sname]])
                yhs = np.concatenate(self.lesson_yh_step[key][chosen[sname]])
                self.history[key][f"{sname}_y_bar"].append(float(np.mean(ys)))
                self.history[key][f"{sname}_y_hat_bar"].append(float(np.mean(yhs)))
                s_y = float(np.sum(ys))
                self.history[key][f"{sname}_bias"].append(
                    float(np.sum(yhs)) / s_y if s_y > 0 else 1.0
                )
                for m in meta["metrics"]:
                    if meta["type"] == "REG":
                        self.history[key][f"{sname}_{m}"].append(
                            self._calculate_reg_metric(m, ys, yhs)
                        )
                    else:
                        self.history[key][f"{sname}_{m}"].append(
                            self._calculate_cls_metric(m, ys, yhs)
                        )

        # 4. Family parameter-health reduction (μ̄, θ-CoV, π stats) — C-213 upgrade. Lesson-aligned
        # with the metric/bias history; carry-forward on an empty lesson (as the §above block).
        for key, stat_names in self._param_health_keys.items():
            if not self.lesson_params[key]:
                for sn in stat_names:
                    for sk in [sn] + [f"{s}_{sn}" for s in self.HORIZON_SLICES]:
                        self.history[key][sk].append(
                            self.history[key][sk][-1] if self.history[key][sk] else 0.0
                        )
                continue
            full = self._reduce_param_health(
                np.concatenate(self.lesson_params[key]), np.concatenate(self.lesson_params_y[key])
            )
            for sn in stat_names:
                self.history[key][sn].append(full[sn])
            p_steps = sorted(self.lesson_params_step[key].keys())
            chosen = (
                {"t0": p_steps[0], "early": p_steps[len(p_steps) // 3], "late": p_steps[-1]}
                if p_steps
                else {}
            )
            for sname in self.HORIZON_SLICES:
                if sname not in chosen:
                    for sn in stat_names:
                        sk = f"{sname}_{sn}"
                        self.history[key][sk].append(
                            self.history[key][sk][-1] if self.history[key][sk] else 0.0
                        )
                    continue
                sl = self._reduce_param_health(
                    np.concatenate(self.lesson_params_step[key][chosen[sname]]),
                    np.concatenate(self.lesson_params_y_step[key][chosen[sname]]),
                )
                for sn in stat_names:
                    self.history[key][f"{sname}_{sn}"].append(sl[sn])

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
