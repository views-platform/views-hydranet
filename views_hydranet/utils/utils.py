"Shared Utilities for the HydraNet Pipeline."

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.basu_loss import BasuDPDLoss
from views_hydranet.utils.focal_loss import FocalLoss
from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss
from views_hydranet.utils.mtloss import MultiTaskLoss
from views_hydranet.utils.pareto_loss import ParetoLoss
from views_hydranet.utils.shrinkage_loss import ShrinkageLoss
from views_hydranet.utils.warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler

logger = logging.getLogger(__name__)


def choose_model(config: dict, device: torch.device) -> nn.Module:
    """Factory for model instantiation."""
    if config["model"] == "HydraBNUNet06_LSTM4":
        model = HydraBNUNet06_LSTM4(
            config["input_channels"],
            config["total_hidden_channels"],
            config["output_channels"],
            config["dropout_rate"],
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
    "shrinkage": {
        "cls": ShrinkageLoss,
        "params": ["loss_reg_a", "loss_reg_c"],
        "factory": lambda config, device: ShrinkageLoss(
            a=config["loss_reg_a"],
            c=config["loss_reg_c"],
            size_average=True,
        ).to(device),
    },
    "basu_dpd": {
        "cls": BasuDPDLoss,
        "params": ["loss_reg_alpha", "loss_reg_sigma"],
        "factory": lambda config, device: BasuDPDLoss(
            alpha=config["loss_reg_alpha"],
            sigma=config["loss_reg_sigma"],
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
}


def choose_loss(
    config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, nn.Module, "MultiTaskLoss"]:
    """Factory for loss function instances.

    Loss functions are selected by name via the LOSS_REG_REGISTRY and
    LOSS_CLASS_REGISTRY dictionaries. Adding a new loss requires only
    adding an entry to the appropriate registry — no modification of
    this function (OCP).
    """
    try:
        criterion_reg = LOSS_REG_REGISTRY[config["loss_reg"]]["factory"](config, device)
    except KeyError:
        raise ValueError(
            f"Unknown regression loss: '{config['loss_reg']}'. "
            f"Available: {list(LOSS_REG_REGISTRY.keys())}"
        ) from None
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
