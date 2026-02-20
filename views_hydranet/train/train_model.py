import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from views_pipeline_core.managers.model import ModelPathManager

from views_hydranet.utils.curriculum import CurriculumLearner
from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.utils_logging import log_curriculum_report
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.utils import (
    choose_loss,
    choose_model,
    choose_scheduler,
    init_weights,
    train_log,
)
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler
from views_hydranet.utils.training_forensics import TrainingForensics

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
    viz: VisualDiagnostics = None,
    stage_label: str = "",
    forensics: TrainingForensics = None,
) -> Dict[str, torch.Tensor]: # Returns window losses

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

    mem_allocated = torch.cuda.memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
    logger.debug(f"🚀 Training: Entered Gate with Tensor {train_tensor.shape} | GPU Mem: {mem_allocated:.2f} MB")

    # initialize a hidden state
    h = model.init_h(hidden_channels=model.base, dim=window_dim).float().to(device)

    # STAGE 5 DIAGNOSTIC: Accumulators
    acc_y_reg, acc_yh_reg = [], []
    acc_y_cls, acc_yh_cls = [], []
    acc_months = []
    
    time_idx = -1
    if viz and stage_label:
         try:
              time_idx = sample_handler.channel_map.index(sample_handler.time_col)
         except Exception:
              pass

    # Sequence loop rnn style
    for i in range(seq_len - 1):
            t0 = train_tensor[:, i, :, :, :]
            t1 = train_tensor[:, i + 1, :, :, :]
            t1_binary = (t1.clone().detach() > 0) * 1.0

            # Forward pass (Data is already North-Up via VolumeHandler)
            # We remove h.detach() to enable Backpropagation Through Time (BPTT).
            # This allows gradients to flow back across the entire temporal sequence.
            t1_pred, t1_pred_class, h = model(t0, h)
            
            # --- FORENSIC RECORDING (ADR 001 Custodian) ---
            if forensics:
                 reg_targets = config.get("regression_targets", [])
                 cls_targets = config.get("classification_targets", [])
                 
                 # Record Regression Targets
                 for idx, target_name in enumerate(reg_targets):
                      forensics.record(f"REG:{target_name}", t1[:, idx:idx+1], t1_pred[:, idx:idx+1])
                 
                 # Record Classification Targets
                 for idx, target_name in enumerate(cls_targets):
                      forensics.record(f"CLS:{target_name}", t1_binary[:, idx:idx+1], torch.sigmoid(t1_pred_class[:, idx:idx+1]))

            # STAGE 5 DIAGNOSTIC: Accumulate middle steps
            if viz and stage_label:
                 # We want 6 steps.
                 start_idx = max(0, (seq_len // 2) - 3)
                 if i >= start_idx and len(acc_y_reg) < 6:
                      # [B, C, H, W] -> [H, W, C]
                      acc_y_reg.append(t1[0].permute(1, 2, 0).detach().cpu().numpy())
                      acc_yh_reg.append(t1_pred[0].permute(1, 2, 0).detach().cpu().numpy())
                      acc_y_cls.append(t1_binary[0].permute(1, 2, 0).detach().cpu().numpy())
                      acc_yh_cls.append(torch.sigmoid(t1_pred_class[0]).permute(1, 2, 0).detach().cpu().numpy())
                      
                      if time_idx >= 0:
                           # Extract month_id from sample_handler.data [T, H, W, C]
                           m_id = sample_handler.data[i+1, 0, 0, time_idx]
                           acc_months.append(m_id)

            losses_list = []
            n_reg = t1.shape[1] # Actual regression features in data
            for j in range(n_reg):
                losses_list.append(criterion_reg(t1_pred[:, j, :, :], t1[:, j, :, :]))

            for j in range(n_reg):
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

    # STAGE 5 DIAGNOSTIC: Finalize Biopsy
    if viz and stage_label and acc_y_reg:
         viz.biopsy_training_performance(
             np.stack(acc_y_reg), np.stack(acc_yh_reg),
             np.stack(acc_y_cls), np.stack(acc_yh_cls),
             stage_label,
             time_indices=acc_months
         )

    # log each sequence/timeline/batch
    train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list)

    # RETURN LOSS COMPONENTS
    return {
        "total": total_loss,
        "reg": torch.tensor(np.sum(avg_loss_reg_list)).to(device),
        "cls": torch.tensor(np.sum(avg_loss_class_list)).to(device)
    }


def training_loop(
    config: dict,
    model: nn.Module,
    criterion: tuple,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    handler: VolumeHandler,
    device: torch.device,
    columns: list[str] | None = None,
    run_timestamp: str | None = None,
) -> dict:
    """
    Orchestrates the training process over multiple stochastic samples.
    Returns a diagnostic summary dictionary.
    """
    criterion_reg, criterion_class, multitaskloss_instance = criterion

    np.random.seed(config["np_seed"])
    torch.manual_seed(config["torch_seed"])
    logger.info("🚀 Training initiated...")
    
    # Initialize Visual Truth Engine with Authoritative Timestamp
    viz = VisualDiagnostics(config, run_timestamp=run_timestamp)
    
    # Initialize Forensic Auditor (ADR 001 Custodian)
    forensics = TrainingForensics(config)

    # Initialize the Sampler Components
    # 1. The Lens (Mechanical)
    sampler = VolumeSampler(handler, config)
    # 2. The Planner (Strategic)
    planner = CurriculumLearner(config, handler)
    log_curriculum_report(planner.subjects, planner.subject_maxima, config)

    # 1. Determine total steps upfront for a unified progress bar
    train_vol_shape = sampler.get_train_volume().shape
    seq_len = train_vol_shape[0]
    total_iterations = config["total_lessons"] * config["windows_per_lesson"] * (seq_len - 1)

    loss_history = []
    loss_history_reg = []
    loss_history_cls = []
    max_raw_grad_norm = 0.0
    
    with tqdm(
        total=total_iterations,
        desc="👾 Training HydraNet",
        unit="month",
        leave=True
    ) as pbar:
        # Loop over Strategic Lessons
        for lesson_idx in range(config["total_lessons"]):

            optimizer.zero_grad() # Reset gradients at start of Lesson
            lesson_loss = torch.tensor(0.0).to(device)
            lesson_reg = 0.0
            lesson_cls = 0.0

            # Pull one lesson per window in the batch (The Mixed Salad)
            for window_idx in range(config["windows_per_lesson"]):
                # 1. Handshake with Planner
                global_step_idx = lesson_idx * config["windows_per_lesson"] + window_idx
                target, threshold = planner.get_lesson(global_step_idx)

                # 2. Handshake with Lens
                batch, qualified_cells = sampler.get_batch(target, threshold, batch_size=1)
                sample_handler = batch[0]
                
                # DIAGNOSTIC: Stage 4 (Sampling)
                # We biopsy every window to verify geometry and variety
                viz.biopsy_sample(sample_handler, handler, f"Stage 4: Training Window {window_idx+1} (Lesson {lesson_idx+1} Target {target})")

                # Update progress bar
                pbar.set_description(
                    f"👾 Training | Lesson {lesson_idx + 1}/{config['total_lessons']} | "
                    f"Window {window_idx + 1}/{config['windows_per_lesson']} | "
                    f"Target: {target} | Threshold: {threshold}"
                )

                # 3. Process Window (Accumulate Loss)
                # Pass viz to capture training dynamics (Stage 5) for all windows
                slbl = f"Stage 5: Training Forensic (Lesson {lesson_idx + 1}_Win {window_idx + 1})"
                losses = train(
                    model,
                    optimizer,
                    scheduler,
                    criterion_reg,
                    criterion_class,
                    multitaskloss_instance,
                    sample_handler,
                    config,
                    device,
                    pbar,
                    viz=viz,
                    stage_label=slbl,
                    forensics=forensics
                )

                # --- MEMORY-SAFE ACCUMULATION (ADR 014 Hardening) ---
                w_loss = losses["total"]
                if w_loss > 0:
                    w_loss.backward()

                lesson_loss += w_loss.detach() # Keep track of magnitude for logging
                lesson_reg += losses["reg"].item()
                lesson_cls += losses["cls"].item()

            # --- THE OPTIMIZATION GATE (ADR 014) ---
            if lesson_loss > 0:
                # NUMERICAL AUDIT: Hard stop on explosion
                IntegrityGuardian.monitor(model, torch.tensor([0.0]), lesson_loss, context=f"Lesson {lesson_idx}")
                
                loss_history.append(lesson_loss.item())
                loss_history_reg.append(lesson_reg / config["windows_per_lesson"])
                loss_history_cls.append(lesson_cls / config["windows_per_lesson"])
                
                # DIAGNOSTIC: Update Dynamic Loss Curves
                viz.biopsy_loss_curves(loss_history_reg, loss_history_cls, loss_history, f"Lesson {lesson_idx+1}")
                
                # DIAGNOSTIC: Finalize Forensic Auditor and Trigger Dossiers
                forensics.finalize_lesson()
                logger.info(f"📊 Training: Finalized Forensic Lesson {lesson_idx+1}. Generating {len(forensics.history)} dossiers...")
                for key, meta in forensics.target_map.items():
                     dossier = forensics.get_dossier(key)
                     viz.biopsy_feature_dossier(meta["name"], dossier, f"Lesson {lesson_idx+1}", target_type=meta["type"])

                # --- 1. Audit Raw Gradient Energy BEFORE Clipping ---
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                raw_grad_norm = total_norm ** 0.5
                max_raw_grad_norm = max(max_raw_grad_norm, raw_grad_norm)

                # Gradient Clipping
                if config.get("clip_grad_norm", False):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimize (Update Weights)
                optimizer.step()

            # Step scheduler at the end of the lesson if it exists
            if scheduler and not isinstance(scheduler, list):
                scheduler.step()

    logger.info("✅ Training complete!")
    
    # 4. Final weight audit
    weight_norms = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weight_norms[name] = param.data.norm().item()

    return {
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "min_loss": min(loss_history) if loss_history else 0.0,
        "max_loss": max(loss_history) if loss_history else 0.0,
        "max_raw_grad_norm": max_raw_grad_norm,
        "loss_history": loss_history,
        "weight_norms": weight_norms,
        "learning_rate": optimizer.param_groups[0]['lr']
    }


def train_model_artifact(
    model_path: ModelPathManager,
    config: dict,
    device: torch.device,
    handler: VolumeHandler,
    columns: list[str] | None = None,
    run_timestamp: str | None = None,
) -> dict:
    """Creates, trains, and saves a model artifact."""

    # Create the model, criterion, optimizer and scheduler
    model, criterion, optimizer, scheduler = make(config, device)

    # Train the model
    summary = training_loop(
        config, model, criterion, optimizer, scheduler, handler, device, columns=columns, run_timestamp=run_timestamp
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
    return summary
