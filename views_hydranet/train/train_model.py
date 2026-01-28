import numpy as np
import pickle
import time
import os
import functools
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

import sys
from pathlib import Path
import logging

from views_pipeline_core.managers.model import ModelPathManager
from views_hydranet.utils.utils import (
    choose_model,
    choose_loss,
    choose_scheduler,
    get_train_tensors,
    get_full_tensor,
    apply_dropout,
    execute_freeze_h_option,
    train_log,
    init_weights,
    get_data,
)

logger = logging.getLogger(__name__)

def make(config: dict, device: torch.device):
    model = choose_model(config, device)

    # Create a partial function with the initialization function and the config parameter
    init_fn = functools.partial(init_weights, config=config)

    # Apply the initialization function to the modelrawi
    model.apply(init_fn)

    # choose loss function
    criterion = choose_loss(
        config, device
    )  # this is a touple of the reg and the class criteria

    # choose scheduler - the optimizer is always AdamW right now
    optimizer, scheduler = choose_scheduler(config, model)

    return (model, criterion, optimizer, scheduler)  # , dataloaders, dataset_sizes)


def train(model, optimizer, scheduler, criterion_reg, criterion_class, multitaskloss_instance, views_vol, sample, config, device): # views vol and sample

    wandb.watch(model, [criterion_reg, criterion_class], log= None, log_freq=2048)

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []
    total_loss = 0

    model.train()  # train mode
    multitaskloss_instance.train() # meybe another place...


    # Batch loops:
    for batch in range(config["batch_size"]):

        # Getting the train_tensor
        train_tensor = get_train_tensors(views_vol, sample, config, device)
        seq_len = train_tensor.shape[1]
        window_dim = train_tensor.shape[-1] # the last dim should always be a spatial dim (H or W)

        # initialize a hidden state
        h = model.init_h(hidden_channels = model.base, dim = window_dim).float().to(device)

        # Sequens loop rnn style
        for i in range(seq_len-1): # so your sequnce is the full time len - last month.
            print(f'\t\t\t\t month: {i+1}/{seq_len}...', end='\r')

            t0 = train_tensor[:, i, :, :, :]

            t1 = train_tensor[:, i+1, :, :, :]
            t1_binary = (t1.clone().detach().requires_grad_(True) > 0) * 1.0 # 1.0 to ensure float. Should avoid cloning warning now.

            # forward-pass
            t1_pred, t1_pred_class, h = model(t0, h.detach())
        
            losses_list = []

            for j in range(t1_pred.shape[1]): # first each reggression loss. Should be 1 channel, as I conccat the reg heads on dim = 1

                losses_list.append(criterion_reg(t1_pred[:,j,:,:], t1[:,j,:,:])) # index 0 is batch dim, 1 is channel dim (here pred), 2 is H dim, 3 is W dim

            for j in range(t1_pred_class.shape[1]): # then each classification loss. Should be 1 channel, as I conccat the class heads on dim = 1

                losses_list.append(criterion_class(t1_pred_class[:,j,:,:], t1_binary[:,j,:,:])) # index 0 is batch dim, 1 is channel dim (here pred), 2 is H dim, 3 is W dim

            losses = torch.stack(losses_list)
            loss = multitaskloss_instance(losses)

            total_loss += loss

            # traning output
            loss_reg = losses[:t1_pred.shape[1]].sum() # sum the reg losses
            loss_class = losses[-t1_pred.shape[1]:].sum() # assuming 

            avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
            avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
            avg_loss_list.append(loss.detach().cpu().numpy().item())


    # log each sequence/timeline/batch
    train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list) # FIX!!!

    # Backpropagation and optimization - after a full sequence... 
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient Clipping
    if config["clip_grad_norm"] == True:
        clip_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

    else:
        pass

    # optimize
    optimizer.step()

    # Adjust learning rate based on the loss
    scheduler.step()


def training_loop(
    config: dict,
    model: nn.Module,
    criterion: tuple,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    views_vol: np.ndarray,
    device: torch.device,
) -> None:
    # # add spatail transformer

    criterion_reg, criterion_class, multitaskloss_instance = criterion

    np.random.seed(config["np_seed"])
    torch.manual_seed(config["torch_seed"])
    logger.info("🚀 Training initiated...")

    for sample in range(config["samples"]):
        progress_msg = f'📡 Training Sample {sample + 1}/{config["samples"]}'
        print(progress_msg, end="\r")  # Live updating print

        train(
            model,
            optimizer,
            scheduler,
            criterion_reg,
            criterion_class,
            multitaskloss_instance,
            views_vol,
            sample,
            config,
            device,
        )

    logger.info("✅ Training complete!")


def train_model_artifact(
    model_path: ModelPathManager,
    config: dict,
    device: torch.device,
    views_vol: np.ndarray,
) -> None:
    """Creates, trains, and saves a model artifact.

    This function creates the model, criterion, optimizer, and scheduler. It then trains the model
    using the provided training loop and saves the trained model with a timestamp and run type as an artifact
    in the specified artifacts path.

    Args:
        model_path: The ModelPathManager instance for path resolution.
        config: Configuration dictionary containing parameters and settings.
        device: The device (torch.device) to run the model on (CPU or GPU).
        views_vol: The array containing the input data for training.
    """

    # Create the model, criterion, optimizer and scheduler
    model, criterion, optimizer, scheduler = make(config, device)

    # Train the model
    training_loop(config, model, criterion, optimizer, scheduler, views_vol, device)
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
