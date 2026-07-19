"Shared Utilities for the HydraNet Pipeline."

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.count_mean_loss import CountMeanMSELoss
from views_hydranet.utils.dense_nb_loss import DenseNBLoss
from views_hydranet.utils.focal_loss import FocalLoss
from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss
from views_hydranet.utils.mtloss import MultiTaskLoss
from views_hydranet.utils.pareto_loss import ParetoLoss
from views_hydranet.utils.quantile_head import QuantileLoss, midpoint_levels
from views_hydranet.utils.shrinkage_loss import ShrinkageLoss
from views_hydranet.utils.tobit_loss import TobitLoss
from views_hydranet.utils.truncated_nb_loss import TruncatedNBLoss
from views_hydranet.utils.warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler
from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

logger = logging.getLogger(__name__)


def choose_model(config: dict, device: torch.device) -> nn.Module:
    """Factory for model instantiation."""
    if config["model"] == "HydraBNUNet06_LSTM4":
        model = HydraBNUNet06_LSTM4(
            config["input_channels"],
            config["total_hidden_channels"],
            config["output_channels"],
            config["dropout_rate"],
            output_distribution=config.get("output_distribution", "standard"),
            n_static_channels=len(config.get("static_channels", [])),  # ADR-061 top-skip
            reg_activation=config.get("reg_activation"),  # Exp B: decouple emit activation
            n_quantiles=config.get("n_quantiles"),  # quantile head: K reg channels per target
        ).to(device)
    else:
        err_msg = f"Unknown model type: {config['model']}"

        logger.error(err_msg)

        raise ValueError(err_msg)
    return model


# Loss registries: add new losses here, not in choose_loss().
# "params" lists are declared for future ReproducibilityGate genome audit (C-43).
LOSS_REG_REGISTRY: dict[str, Any] = {
    "mse": {
        "cls": nn.MSELoss,
        "params": [],
        "factory": lambda config, device: nn.MSELoss().to(device),
    },
    # count_mean: MSE in COUNT space (expm1) — minimizer E[y|x], the count MEAN, vs mse's
    # log-median
    # under-fit of the heavy tail (port of views-lstm-lab EXP-07). Pair with
    # output_distribution='standard' (E[y]=expm1(body)) and NO hurdle mask (unconditional mean).
    "count_mean": {
        "cls": CountMeanMSELoss,
        "params": [],
        "factory": lambda config, device: CountMeanMSELoss().to(device),
    },
    # Non-negative point bodies for the hurdle (trained on positive cells, log1p space; emit via
    # the point compose, output_distribution='hurdle_shrinkage'). MAE -> conditional median
    # (robust, under-fires the heavy-tailed mean); Huber -> robust mean, delta in log1p units
    # (default 1.0; 'loss_reg_huber_delta' is an optional forward-compat hook, not a config field).
    "mae": {
        "cls": nn.L1Loss,
        "params": [],
        "factory": lambda config, device: nn.L1Loss().to(device),
    },
    "huber": {
        "cls": nn.HuberLoss,
        "params": [],
        "factory": lambda config, device: nn.HuberLoss(
            delta=config.get("loss_reg_huber_delta", 1.0)
        ).to(device),
    },
    "shrinkage": {
        "cls": ShrinkageLoss,
        "params": ["loss_reg_a", "loss_reg_c"],
        "factory": lambda config, device: ShrinkageLoss(
            a=config["loss_reg_a"],
            c=config["loss_reg_c"],
            size_average=True,
        ).to(device),
    },
    "lognormal_nll": {
        "cls": LogNormalFixedSigmaLoss,
        "params": ["loss_reg_sigma"],
        "factory": lambda config, device: LogNormalFixedSigmaLoss(
            sigma=config["loss_reg_sigma"],
        ).to(device),
    },
    "pareto": {
        "cls": ParetoLoss,
        "params": ["loss_reg_pareto_alpha"],
        "factory": lambda config, device: ParetoLoss(
            alpha=config["loss_reg_pareto_alpha"],
        ).to(device),
    },
    "tobit": {
        "cls": TobitLoss,
        "params": ["loss_reg_sigma"],
        "factory": lambda config, device: TobitLoss(
            sigma=config["loss_reg_sigma"],
        ).to(device),
    },
    # Hurdle-NB body (D2, #99): zero-truncated NB on positive cells; learnable per-target θ.
    # Per-target instances are built in choose_loss() so each θ reaches the optimizer.
    "hurdle_nb": {
        "cls": TruncatedNBLoss,
        "params": ["loss_reg_theta_init"],
        "factory": lambda config, device: TruncatedNBLoss(
            theta_init=config["loss_reg_theta_init"],
            learnable=config.get("learnable_theta", True),
        ).to(device),
    },
    # Quantile head (2026-07 build): K monotone quantiles/target, multi-tau pinball (Riemann-CRPS).
    # A single shared instance is built in choose_loss(); the training loop slices K
    # channels/target.
    "quantile": {
        "cls": QuantileLoss,
        "params": ["n_quantiles"],
        "factory": lambda config, device: QuantileLoss(midpoint_levels(config["n_quantiles"])).to(
            device
        ),
    },
    # Dense NB body (C-168 sharpness experiment): plain NB NLL on ALL cells (zeros supervised).
    # Per-target instances built in choose_loss() so each θ reaches the optimizer.
    "dense_nb": {
        "cls": DenseNBLoss,
        "params": ["loss_reg_theta_init"],
        "factory": lambda config, device: DenseNBLoss(
            theta_init=config["loss_reg_theta_init"],
            learnable=config.get("learnable_theta", True),
        ).to(device),
    },
}

LOSS_CLASS_REGISTRY: dict[str, Any] = {
    "bce": {
        "cls": nn.BCELoss,
        "params": [],
        "factory": lambda config, device: nn.BCELoss().to(device),
    },
    "focal": {
        "cls": FocalLoss,
        "params": ["loss_class_alpha", "loss_class_gamma"],
        "factory": lambda config, device: FocalLoss(
            alpha=config["loss_class_alpha"],
            gamma=config["loss_class_gamma"],
        ).to(device),
    },
    # Hurdle-NB gate (D2, #99): proper class-weighted BCE on logits (replaces focal; C-147).
    "weighted_bce": {
        "cls": WeightedBCEWithLogitsLoss,
        "params": [],
        "factory": lambda config, device: WeightedBCEWithLogitsLoss(
            pos_weight=config.get("loss_class_pos_weight"),
        ).to(device),
    },
}


def choose_loss(
    config: dict[str, Any], device: torch.device
) -> tuple[nn.Module | dict[str, nn.Module], nn.Module, "MultiTaskLoss"]:
    """Factory for loss function instances.

    Loss functions are selected by name via the LOSS_REG_REGISTRY and
    LOSS_CLASS_REGISTRY dictionaries. Adding a new loss requires only
    adding an entry to the appropriate registry — no modification of
    this function (OCP).
    """
    loss_reg_sigma = config.get("loss_reg_sigma")
    learnable = config.get("learnable_sigma", False)
    if config["loss_reg"] in ("hurdle_nb", "dense_nb"):
        # Per-target NB body: one loss instance per regression target, each with its own learnable
        # θ (reaches the optimizer via the dict-of-losses path in training_engine). hurdle_nb =
        # zero-truncated (TruncatedNBLoss); dense_nb = NB on all cells (DenseNBLoss, C-168 expt).
        theta_init = config.get("loss_reg_theta_init") or 1.0
        learnable_theta = config.get("learnable_theta", True)
        body_cls = TruncatedNBLoss if config["loss_reg"] == "hurdle_nb" else DenseNBLoss
        criterion_reg = {
            target: body_cls(theta_init=theta_init, learnable=learnable_theta).to(device)
            for target in config.get("regression_targets", [])
        }
    elif config["loss_reg"] == "quantile":
        # Quantile head: a single multi-tau pinball instance shared across targets (the head emits
        # K
        # monotone quantiles/target; the training loop slices K channels per target). K =
        # n_quantiles.
        k = config.get("n_quantiles")
        if not k or k < 2:
            raise ValueError("loss_reg='quantile' requires config['n_quantiles'] >= 2")
        criterion_reg = QuantileLoss(midpoint_levels(k)).to(device)
    elif isinstance(loss_reg_sigma, dict) and config["loss_reg"] == "tobit":
        criterion_reg = {
            target: TobitLoss(sigma=s, learnable=learnable).to(device)
            for target, s in loss_reg_sigma.items()
        }
    else:
        try:
            criterion_reg = LOSS_REG_REGISTRY[config["loss_reg"]]["factory"](config, device)
        except KeyError:
            raise ValueError(
                f"Unknown regression loss: '{config['loss_reg']}'. "
                f"Available: {list(LOSS_REG_REGISTRY.keys())}"
            ) from None
    _pw = config.get("loss_class_pos_weight")
    if config["loss_class"] == "weighted_bce" and isinstance(_pw, (list, tuple)):
        # per-target gate: one WeightedBCEWithLogitsLoss per classification target (sb/ns/os), each
        # with
        # its own pos_weight. Applied per-target in the training loop (mirrors the per-target reg
        # dict).
        criterion_class = [WeightedBCEWithLogitsLoss(pos_weight=float(p)).to(device) for p in _pw]
    else:
        try:
            criterion_class = LOSS_CLASS_REGISTRY[config["loss_class"]]["factory"](config, device)
        except KeyError:
            raise ValueError(
                f"Unknown classification loss: '{config['loss_class']}'. "
                f"Available: {list(LOSS_CLASS_REGISTRY.keys())}"
            ) from None

    logger.info(f"Regression loss: {criterion_reg}\n classification loss: {criterion_class}")

    # Dynamic Loss Mask (ADR 020 Multi-Task)
    # We construct the boolean mask based on the configured targets
    n_reg = len(config.get("regression_targets", []))
    n_cls = len(config.get("classification_targets", []))

    # Mask: True for Regression, False for Classification
    mask_list = [True] * n_reg + [False] * n_cls
    is_regression = torch.Tensor(mask_list)

    multitaskloss_instance = MultiTaskLoss(is_regression, reduction="sum")
    return (criterion_reg, criterion_class, multitaskloss_instance)


def choose_scheduler(config: dict[str, Any], unet: nn.Module) -> tuple[torch.optim.Optimizer, Any]:
    """Factory for learning rate schedulers."""
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"],
    )

    if config["scheduler"] == "WarmupDecay":
        d = config["window_dim"] * config["window_dim"] * config["input_channels"]
        scheduler = WarmupDecayLearningRateScheduler(
            optimizer, d=d, warmup_steps=config["warmup_steps"]
        )
    elif config["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        scheduler = None

    return (optimizer, scheduler)


def init_weights(m: nn.Module, config: dict[str, Any]) -> None:
    """Weight initialization gate."""
    if config["weight_init"] == "xavier_uni":
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    elif config["weight_init"] == "kaiming_uni":
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
    elif config["weight_init"] == "xavier_norm":
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    else:
        err_msg = (
            f"Unknown weight_init scheme: '{config['weight_init']}'. "
            "Valid: 'xavier_uni', 'xavier_norm', 'kaiming_uni'."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def train_log(
    avg_loss_list: list[float],
    avg_loss_reg_list: list[float],
    avg_loss_class_list: list[float],
) -> None:
    """Metric logging gate for W&B."""
    import wandb

    if wandb.run is not None:
        wandb.log(
            {
                "avg_loss": np.mean(avg_loss_list),
                "avg_loss_reg": np.mean(avg_loss_reg_list),
                "avg_loss_class": np.mean(avg_loss_class_list),
            }
        )
