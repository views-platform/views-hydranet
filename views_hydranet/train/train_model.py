import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from views_pipeline_core.managers.model import ModelPathManager

from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.utils import (
    choose_loss,
    choose_model,
    choose_scheduler,
    init_weights,
    train_log,
)
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logger = logging.getLogger(__name__)

def make(config: dict, device: torch.device):
    model = choose_model(config, device)

    # Create a partial function with the initialization function and the config parameter
    import functools
    init_fn = functools.partial(init_weights, config=config)

    # Apply the initialization function to the model
    model.apply(init_fn)

    # choose loss function
    criterion = choose_loss(
        config, device
    )  # this is a tuple of the reg and the class criteria

    # choose scheduler - the optimizer is always AdamW right now
    optimizer, scheduler = choose_scheduler(config, model)

    return (model, criterion, optimizer, scheduler)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion_reg: nn.Module,
    criterion_class: nn.Module,
    multitaskloss_instance: nn.Module,
    sample_handler: VolumeHandler,
    config: dict,
    device: torch.device,
    pbar: tqdm,
) -> None:

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []
    total_loss = torch.tensor(0.0).to(device) 

    model.train()
    multitaskloss_instance.train()

    # 1. Stochastic Data Augmentation (Tube-Level)
    # We flip the entire spatial-temporal tube together to maintain consistency.
    if config.get("random_flips", True):
        if np.random.rand() < 0.5:
            sample_handler.flip("H")
        if np.random.rand() < 0.5:
            sample_handler.flip("W")

    # 1. Model Entry Gate: Transform to PyTorch [B, T, C, H, W]
    # We strip identity channels here so the model only sees features.
    train_tensor = sample_handler.to_pytorch(device, include_identities=False)

    seq_len = train_tensor.shape[1]
    window_dim = train_tensor.shape[-1]
    
    # initialize a hidden state
    h = model.init_h(hidden_channels=model.base, dim=window_dim).float().to(device)

    # Sequence loop rnn style
    for i in range(seq_len - 1):
            t0 = train_tensor[:, i, :, :, :]
            t1 = train_tensor[:, i + 1, :, :, :]
            t1_binary = (t1.clone().detach().requires_grad_(True) > 0) * 1.0

            # Forward pass (Data is already North-Up via VolumeHandler)
            t1_pred, t1_pred_class, h = model(t0, h.detach())

            losses_list = []
            for j in range(t1_pred.shape[1]):
                losses_list.append(criterion_reg(t1_pred[:, j, :, :], t1[:, j, :, :]))

            for j in range(t1_pred_class.shape[1]):
                losses_list.append(
                    criterion_class(t1_pred_class[:, j, :, :], t1_binary[:, j, :, :])
                )

            losses = torch.stack(losses_list)
            loss = multitaskloss_instance(losses)
            total_loss += loss

            loss_reg = losses[:t1_pred.shape[1]].sum()
            loss_class = losses[-t1_pred.shape[1]:].sum()

            avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
            avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
            avg_loss_list.append(loss.detach().cpu().numpy().item())

            # Update pbar for each month
            pbar.update(1)

    # log each sequence/timeline/batch
    train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list)

    # Backpropagation and optimization
    if total_loss > 0:
        optimizer.zero_grad()
        total_loss.backward()

        # NUMERICAL AUDIT: Hard stop on explosion
        IntegrityGuardian.monitor(model, t1_pred, total_loss, context="Sequence End")

        # Gradient Clipping
        if config.get("clip_grad_norm", False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # optimize
        optimizer.step()

    scheduler.step()


def training_loop(
    config: dict,
    model: nn.Module,
    criterion: tuple,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    handler: VolumeHandler,
    device: torch.device,
    columns: list[str] | None = None,
) -> None:
    """
    Orchestrates the training process over multiple stochastic samples.
    """
    criterion_reg, criterion_class, multitaskloss_instance = criterion

    np.random.seed(config["np_seed"])
    torch.manual_seed(config["torch_seed"])
    logger.info("🚀 Training initiated...")

    # Initialize the Sampler Lens
    sampler = VolumeSampler(handler, config)

    # 1. Determine total steps upfront for a unified progress bar
    train_vol_shape = sampler.get_train_volume().shape
    seq_len = train_vol_shape[0]
    total_iterations = config["samples"] * config["batch_size"] * (seq_len - 1)

    with tqdm(
        total=total_iterations,
        desc="👾 Training HydraNet",
        unit="month",
        leave=True
    ) as pbar:
        for sample_idx in range(config["samples"]):
            pbar.set_description(f"👾 Training Sample {sample_idx + 1}/{config['samples']}")

            # The Sampler now owns the full batch extraction
            batch = sampler.get_next_batch(sample_idx)

            for sample_handler in batch:
                train(
                    model,
                    optimizer,
                    scheduler,
                    criterion_reg,
                    criterion_class,
                    multitaskloss_instance,
                    sample_handler,
                    config,
                    device,
                    pbar
                )

    logger.info("✅ Training complete!")


def train_model_artifact(
    model_path: ModelPathManager,
    config: dict,
    device: torch.device,
    handler: VolumeHandler,
    columns: list[str] | None = None,
) -> None:
    """Creates, trains, and saves a model artifact."""

    # Create the model, criterion, optimizer and scheduler
    model, criterion, optimizer, scheduler = make(config, device)

    # Train the model
    training_loop(
        config, model, criterion, optimizer, scheduler, handler, device, columns=columns
    )
    logger.info("Done training")

    # just in case the artifacts folder does not exist
    os.makedirs(model_path.artifacts, exist_ok=True)

    # Define the path for the artifacts with a timestamp and a run type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{config['run_type']}_model_{timestamp}.pt"
    # save the model
    torch.save(model, model_path.artifacts / model_filename)

    # done
    logger.info(f"Model saved as: {model_path.artifacts / model_filename}")
