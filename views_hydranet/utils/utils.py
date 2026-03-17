"Shared Utilities for the HydraNet Pipeline."

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.focal_loss import FocalLoss
from views_hydranet.utils.mtloss import MultiTaskLoss
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


def choose_loss(
    config: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, nn.Module, "MultiTaskLoss"]:
    """Factory for loss function instances."""
    if config["loss_reg"] == "a":
        criterion_reg = nn.MSELoss().to(device)
    elif config["loss_reg"] == "b":
        criterion_reg = ShrinkageLoss(
            a=config["loss_reg_a"], c=config["loss_reg_c"], size_average=True
        ).to(device)
    else:
        err_msg = f"Unknown regression loss type: {config['loss_reg']}"

        logger.error(err_msg)

        raise ValueError(err_msg)

    if config["loss_class"] == "a":
        criterion_class = nn.BCELoss().to(device)
    elif config["loss_class"] == "b":
        criterion_class = FocalLoss(
            alpha=config["loss_class_alpha"], gamma=config["loss_class_gamma"]
        ).to(device)
    else:
        err_msg = f"Unknown classification loss type: {config['loss_class']}"

        logger.error(err_msg)

        raise ValueError(err_msg)

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


def choose_scheduler(
    config: Dict[str, Any], unet: nn.Module
) -> Tuple[torch.optim.Optimizer, Any]:
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


def init_weights(m: nn.Module, config: Dict[str, Any]) -> None:
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
    avg_loss_list: List[float],
    avg_loss_reg_list: List[float],
    avg_loss_class_list: List[float],
) -> None:
    """Metric logging gate for W&B."""
    if wandb.run is not None:
        wandb.log(
            {
                "avg_loss": np.mean(avg_loss_list),
                "avg_loss_reg": np.mean(avg_loss_reg_list),
                "avg_loss_class": np.mean(avg_loss_class_list),
            }
        )
