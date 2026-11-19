"Shared Utilities for the HydraNet Pipeline."

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from views_hydranet.architectures.registry import get_architecture
from views_hydranet.utils.focal_loss import FocalLoss
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss
from views_hydranet.utils.mtloss import MultiTaskLoss
from views_hydranet.utils.warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler
from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

logger = logging.getLogger(__name__)


def choose_model(config: dict, device: torch.device) -> nn.Module:
    """Factory for model instantiation."""
    # ADR-061 top-skip / C-228 encoder-only / Exp B emit activation are all
    # passed to EVERY architecture, so the registry stays a name lookup rather than a per-model
    # argument table. get_architecture raises on an unknown name, naming what is registered.
    model_cls = get_architecture(config["model"])
    # Every model build passes here: training, and the reload of a saved model from its sidecar.
    # A body the package does not know stops here; otherwise the head would fall back to one ReLU
    # channel and a saved model would fail obscurely or forecast wrong numbers (issue #628).
    from views_hydranet.utils.config_initializer import check_output_distribution

    distribution = config.get("output_distribution", "standard")
    check_output_distribution(distribution)
    return model_cls(
        config["input_channels"],
        config["total_hidden_channels"],
        config["output_channels"],
        config["dropout_rate"],
        output_distribution=distribution,
        n_static_channels=len(config.get("static_channels", [])),  # ADR-061 top-skip
        static_top_skip=config.get("static_top_skip", True),  # C-228: False=encoder-only
        # Only softplus is implemented (ADR-079 Rule 1, #632); None gives softplus, and the model
        # refuses anything else. A saved ReLU model is refused before this, by the reload.
        reg_activation=config.get("reg_activation"),
    ).to(device)


# Loss registries: add new losses here, not in choose_loss().
# "params" lists are declared for future ReproducibilityGate genome audit (C-43).
LOSS_REG_REGISTRY: dict[str, Any] = {
    "mse": {
        "cls": nn.MSELoss,
        "params": [],
        "factory": lambda config, device: nn.MSELoss().to(device),
    },
    # Ranking body (#578, 2026-11-10_ranking_head_dossier 04 phase 1): the FIRST listwise entry
    # here — its value depends on the relative order of the 1,024-cell patch-month, not on any
    # cell alone. Level-indexed (Zhu & Klabjan 2020 Eq. 2) on dyadic rating levels
    # r = ceil(log2(1+y)); `loss_reg_y_max` declares the level set a priori (C-423).
    # ⛔ Nothing has been trained with it. Reachable only by naming it in `loss_reg`.
    "listwise_level": {
        "cls": ListwiseLevelLoss,
        # Required parameters only. `loss_reg_listwise_transform` has a default and is read with
        # `.get` below, so it is not listed: every reader of this list treats it as "must be set".
        "params": ["loss_reg_y_max"],
        "factory": lambda config, device: ListwiseLevelLoss(
            y_max=config["loss_reg_y_max"],
            # .get with the schema's own default, unlike y_max above: this field HAS a default, so
            # a pre-schema raw dict that omits it must get the shipped objective, not a KeyError.
            transform=config.get("loss_reg_listwise_transform", "exp"),
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
) -> tuple[nn.Module, nn.Module, "MultiTaskLoss"]:
    """Factory for loss function instances.

    Loss functions are selected by name via the LOSS_REG_REGISTRY and
    LOSS_CLASS_REGISTRY dictionaries. Adding a new loss requires only
    adding an entry to the appropriate registry — no modification of
    this function (OCP).
    """
    # ADR-067 strangler-fig: if output_distribution names a registered distribution family
    # (nb, truncated_nb, mixture_nb), the reg loss is the family NLL (FamilyLoss over family.nll)
    # — loss_reg is irrelevant for a family. resolve_family returns None for every legacy value,
    # so those keep their loss_reg path below (byte-identical). A single shared instance (the
    # parameters are head-emitted).
    from views_hydranet.distributions import resolve_family
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.utils.config_initializer import (
        check_loss_reg,
        validate_body_nll_denominator_pair,
    )

    # body_nll_denominator is read with NO fallback default — the schema owns it (P7 convention).
    # `.get()` with no second argument yields None for a caller that bypassed `HydraNetConfig`
    # (a raw dict from `get_hp_config()`, a measurement driver, a pre-schema config). Such a
    # caller never ran `reject_ignored_body_nll_denominator`, so the family/denominator rule is
    # re-checked HERE — the same pattern `HydraNetInference.__init__` uses to re-reject the inert
    # clamp for pydantic-bypassing drivers. Without this, a dict setting 'all_cells' on `nb` was
    # neither rejected nor forwarded: silently ignored, which is the exact failure the validator
    # exists to prevent. Found by /code-review on PR #398.
    _dist = config.get("output_distribution", "standard")
    _den = config.get("body_nll_denominator")
    validate_body_nll_denominator_pair(_den, _dist, config.get("body_supervision", "all"))
    _family = (
        resolve_family(_dist, body_nll_denominator=_den)
        if _dist == "truncated_nb" and _den is not None
        else resolve_family(_dist)
    )
    if _family is not None:
        # The family's likelihood trains and `loss_reg` is not read. A raw dict that still names a
        # loss is checked all the same, so a removed one is refused here too.
        if config.get("loss_reg") is not None:
            check_loss_reg(config["loss_reg"])
        criterion_reg = FamilyLoss(_family).to(device)
    else:
        # Membership is checked first, so a KeyError raised inside a factory (a missing parameter
        # such as `loss_reg_y_max`) is not reported as an unknown loss.
        check_loss_reg(config.get("loss_reg"))
        criterion_reg = LOSS_REG_REGISTRY[config["loss_reg"]]["factory"](config, device)
    _pw = config.get("loss_class_pos_weight")
    if config["loss_class"] == "weighted_bce" and isinstance(_pw, (list, tuple)):
        # per-target gate: one WeightedBCEWithLogitsLoss per classification target (sb/ns/os), each
        # with
        # its own pos_weight. Applied per-target in the training loop.
        criterion_class = [WeightedBCEWithLogitsLoss(pos_weight=float(p)).to(device) for p in _pw]
    else:
        try:
            criterion_class = LOSS_CLASS_REGISTRY[config["loss_class"]]["factory"](config, device)
        except KeyError:
            err_msg = (
                f"Unknown classification loss: '{config['loss_class']}'. "
                f"Available: {list(LOSS_CLASS_REGISTRY.keys())}"
            )
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
