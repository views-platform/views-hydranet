"""
Training Engine: Pure training logic with no framework dependencies.

This module contains the core training functions extracted from train_model.py
per Uncle Bob's Dependency Rule: Entity-layer logic (gradient math, sequence
processing, curriculum orchestration) must not depend on Framework-layer
types (ModelPathManager, artifact I/O).

All functions here are importable and testable without views_pipeline_core.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from views_hydranet.utils.curriculum import CurriculumLearner
from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.training_forensics import TrainingForensics
from views_hydranet.utils.utils import (
    choose_loss,
    choose_model,
    choose_scheduler,
    init_weights,
    train_log,
)
from views_hydranet.utils.utils_logging import log_curriculum_report
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logger = logging.getLogger(__name__)


def make(config: dict, device: torch.device):
    model = choose_model(config, device)

    # Create a partial function with the initialization function and the config parameter
    init_fn = functools.partial(init_weights, config=config)

    # Apply the initialization function to the model
    model.apply(init_fn)

    # choose loss function
    criterion = choose_loss(config, device)  # this is a tuple of the reg and the class criteria

    # choose scheduler - the optimizer is always AdamW right now
    optimizer, scheduler = choose_scheduler(config, model)

    return (model, criterion, optimizer, scheduler)


class _SequenceIndices:
    """Pre-computed channel indices for the sequence loop (Zero Magic ADR 003)."""

    __slots__ = ("reg", "cls", "feat", "n_reg", "n_cls", "reg_names", "cls_names")

    def __init__(self, feature_names: list[str], config: dict) -> None:
        reg_targets = config.get("regression_targets", [])
        cls_targets = config.get("classification_targets", [])
        input_features = config.get("features", [])

        self.reg = [feature_names.index(t) for t in reg_targets]
        self.cls = [feature_names.index(t) for t in cls_targets]
        self.feat = [feature_names.index(f) for f in input_features]
        self.n_reg = len(reg_targets)
        self.n_cls = len(cls_targets)
        self.reg_names = reg_targets
        self.cls_names = cls_targets


def _process_sequence(
    train_tensor: torch.Tensor,
    model: nn.Module,
    h: torch.Tensor,
    criterion_reg: nn.Module,
    criterion_class: nn.Module,
    multitaskloss_instance: nn.Module,
    idx: "_SequenceIndices",
    device: torch.device,
    pbar: Optional[tqdm] = None,
    forensics: Optional[TrainingForensics] = None,
) -> Dict[str, Any]:
    """
    Pure sequence processing: forward pass over [B, T, C, H, W] tensor.

    Runs the recurrent model step-by-step through the temporal sequence,
    computes multi-task losses, and optionally records forensic data.
    No optimizer, no diagnostics, no data augmentation — just forward + loss.

    Returns dict with keys: total, reg, cls, h, per_step_losses.
    """
    seq_len = train_tensor.shape[1]
    total_loss = torch.tensor(0.0).to(device)
    step_reg: list[float] = []
    step_cls: list[float] = []
    step_total: list[float] = []

    for i in range(seq_len - 1):
        t0 = train_tensor[:, i, :, :, :]
        t1 = train_tensor[:, i + 1, :, :, :]

        # ADR 046: Separate ground truth signals explicitly
        y_reg = t1[:, idx.reg, :, :]
        y_cls = t1[:, idx.cls, :, :]

        # Forward pass: Feed ONLY the input features (Zero Magic)
        t0_input = t0[:, idx.feat, :, :]
        t1_pred, t1_pred_class, h = cast(Any, model)(t0_input, h)

        # --- FORENSIC RECORDING (ADR 001 Custodian) ---
        if forensics:
            for j, target_name in enumerate(idx.reg_names):
                forensics.record(
                    f"REG:{target_name}", y_reg[:, j : j + 1], t1_pred[:, j : j + 1]
                )
            for j, target_name in enumerate(idx.cls_names):
                forensics.record(
                    f"CLS:{target_name}",
                    y_cls[:, j : j + 1],
                    torch.sigmoid(t1_pred_class[:, j : j + 1]),
                )

        # Loss computation
        losses_list = []
        for j in range(idx.n_reg):
            losses_list.append(criterion_reg(t1_pred[:, j, :, :], y_reg[:, j, :, :]))
        for j in range(idx.n_cls):
            losses_list.append(criterion_class(t1_pred_class[:, j, :, :], y_cls[:, j, :, :]))

        losses = torch.stack(losses_list)
        loss = cast(Any, multitaskloss_instance)(losses)
        total_loss += loss

        loss_reg = losses[: idx.n_reg].sum()
        loss_class = losses[idx.n_reg :].sum()

        step_reg.append(loss_reg.detach().cpu().numpy().item())
        step_cls.append(loss_class.detach().cpu().numpy().item())
        step_total.append(loss.detach().cpu().numpy().item())

        if pbar is not None:
            pbar.update(1)

    return {
        "total": total_loss,
        "reg": torch.tensor(np.sum(step_reg)).to(device),
        "cls": torch.tensor(np.sum(step_cls)).to(device),
        "h": h,
        "per_step_losses": (step_total, step_reg, step_cls),
    }


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion_reg: nn.Module,
    criterion_class: nn.Module,
    multitaskloss_instance: nn.Module,
    sample_handler: VolumeHandler,
    config: dict,
    device: torch.device,
    pbar: tqdm,
    viz: Optional[VisualDiagnostics] = None,
    stage_label: str = "",
    forensics: Optional[TrainingForensics] = None,
) -> Dict[str, torch.Tensor]:  # Returns window losses

    model.train()
    multitaskloss_instance.train()

    # 1. Stochastic Data Augmentation (Tube-Level)
    # We flip the entire spatial-temporal tube together to maintain consistency.
    if config.get("random_flips", True):
        if np.random.rand() < 0.5:
            sample_handler.flip("H")
        if np.random.rand() < 0.5:
            sample_handler.flip("W")

    # 2. Model Entry Gate: Transform to PyTorch [B, T, C, H, W]
    # We strip identity channels here so the model only sees features.
    train_tensor = sample_handler.to_pytorch(device, include_identities=False)

    # 3. Pre-compute channel indices (Zero Magic ADR 003)
    feature_names = [
        n for n in sample_handler.channel_map if n in sample_handler._metadata.feature_cols
    ]
    idx = _SequenceIndices(feature_names, config)

    seq_len = train_tensor.shape[1]
    window_H = train_tensor.shape[-2]
    window_W = train_tensor.shape[-1]

    mem_allocated = torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else 0
    logger.debug(
        f"🚀 Training: Entered Gate with Tensor {train_tensor.shape} | "
        f"GPU Mem: {mem_allocated:.2f} MB"
    )

    # 4. Initialize hidden state (float32 from init_hTtime)
    h = (
        cast(Any, model)
        .init_hTtime(hidden_channels=model.base, H=window_H, W=window_W)
        .to(device)
    )

    # 5. STAGE 5 DIAGNOSTIC: Accumulate visual biopsy data around midpoint
    acc_y_reg: list[np.ndarray] = []
    acc_yh_reg: list[np.ndarray] = []
    acc_y_cls: list[np.ndarray] = []
    acc_yh_cls: list[np.ndarray] = []
    acc_months = []
    time_idx = -1
    if viz and stage_label:
        try:
            time_idx = sample_handler.channel_map.index(sample_handler.time_col)
        except Exception:
            pass

    # --- CORE SEQUENCE PROCESSING ---
    result = _process_sequence(
        train_tensor, model, h,
        criterion_reg, criterion_class, multitaskloss_instance,
        idx, device, pbar=pbar, forensics=forensics,
    )
    step_total, step_reg, step_cls = result["per_step_losses"]

    # --- STAGE 5 DIAGNOSTIC: Lightweight midpoint biopsy ---
    if viz and stage_label:
        biopsy_start = max(0, (seq_len // 2) - 3)
        biopsy_end = min(seq_len - 1, biopsy_start + 6)

        # Re-run the midpoint steps in eval mode for diagnostic capture only
        model.eval()
        h_diag = (
            cast(Any, model)
            .init_hTtime(hidden_channels=model.base, H=window_H, W=window_W)
            .to(device)
        )
        with torch.no_grad():
            for i in range(seq_len - 1):
                t0 = train_tensor[:, i, :, :, :]
                t1 = train_tensor[:, i + 1, :, :, :]
                t0_input = t0[:, idx.feat, :, :]
                t1_pred, t1_pred_class, h_diag = cast(Any, model)(t0_input, h_diag)

                if biopsy_start <= i < biopsy_end:
                    y_reg = t1[:, idx.reg, :, :]
                    y_cls = t1[:, idx.cls, :, :]
                    acc_y_reg.append(y_reg[0].permute(1, 2, 0).cpu().numpy())
                    acc_yh_reg.append(t1_pred[0].permute(1, 2, 0).cpu().numpy())
                    acc_y_cls.append(y_cls[0].permute(1, 2, 0).cpu().numpy())
                    acc_yh_cls.append(
                        torch.sigmoid(t1_pred_class[0]).permute(1, 2, 0).cpu().numpy()
                    )
                    if time_idx >= 0:
                        m_id = sample_handler.data[i + 1, 0, 0, time_idx]
                        acc_months.append(m_id)
        model.train()

        if acc_y_reg:
            viz.biopsy_training_performance(
                np.stack(acc_y_reg),
                np.stack(acc_yh_reg),
                np.stack(acc_y_cls),
                np.stack(acc_yh_cls),
                stage_label,
                time_indices=acc_months,
            )

    # log each sequence/timeline/batch
    train_log(step_total, step_reg, step_cls)

    # RETURN LOSS COMPONENTS
    return {
        "total": result["total"],
        "reg": result["reg"],
        "cls": result["cls"],
    }


def training_loop(
    config: dict,
    model: nn.Module,
    criterion: tuple,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    handler: VolumeHandler,
    device: torch.device,
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
        total=total_iterations, desc="👾 Training HydraNet", unit="month", leave=True
    ) as pbar:
        # Loop over Strategic Lessons
        for lesson_idx in range(config["total_lessons"]):
            optimizer.zero_grad()  # Reset gradients at start of Lesson
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
                viz.biopsy_sample(
                    sample_handler,
                    handler,
                    f"Stage 4: Training Window {window_idx + 1} "
                    f"(Lesson {lesson_idx + 1} Target {target})",
                )

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
                    forensics=forensics,
                )

                # --- MEMORY-SAFE ACCUMULATION (ADR 014 Hardening) ---
                w_loss = losses["total"]
                if w_loss > 0:
                    w_loss.backward()

                lesson_loss += w_loss.detach()  # Keep track of magnitude for logging
                lesson_reg += losses["reg"].item()
                lesson_cls += losses["cls"].item()

            # --- THE OPTIMIZATION GATE (ADR 014) ---
            if lesson_loss > 0:
                # NUMERICAL AUDIT: Hard stop on explosion
                IntegrityGuardian.monitor(
                    model, torch.tensor([0.0]), lesson_loss, context=f"Lesson {lesson_idx}"
                )

                loss_history.append(lesson_loss.item())
                loss_history_reg.append(lesson_reg / config["windows_per_lesson"])
                loss_history_cls.append(lesson_cls / config["windows_per_lesson"])

                # DIAGNOSTIC: Update Dynamic Loss Curves
                viz.biopsy_loss_curves(
                    loss_history_reg, loss_history_cls, loss_history, f"Lesson {lesson_idx + 1}"
                )

                # DIAGNOSTIC: Finalize Forensic Auditor and Trigger Dossiers
                forensics.finalize_lesson()
                logger.info(
                    f"📊 Training: Finalized Forensic Lesson {lesson_idx + 1}. "
                    f"Generating {len(forensics.history)} dossiers..."
                )
                for key, meta in forensics.target_map.items():
                    dossier = forensics.get_dossier(key)
                    viz.biopsy_feature_dossier(
                        meta["name"], dossier, f"Lesson {lesson_idx + 1}", target_type=meta["type"]
                    )

                # --- 1. Audit Raw Gradient Energy BEFORE Clipping ---
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                raw_grad_norm = total_norm**0.5
                max_raw_grad_norm = max(max_raw_grad_norm, raw_grad_norm)

                # Gradient Clipping
                if config.get("clip_grad_norm", False):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimize (Update Weights)
                optimizer.step()

            # Step scheduler at the end of the lesson if it exists
            if scheduler is not None:
                scheduler.step()

    logger.info("✅ Training complete!")

    # 4. Final weight audit
    weight_norms = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weight_norms[name] = param.data.norm().item()

    # ADR-037: Health Constellation radar plot of L2 norms per functional block
    viz.biopsy_health_constellation(weight_norms, "End of Training")

    return {
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "min_loss": min(loss_history) if loss_history else 0.0,
        "max_loss": max(loss_history) if loss_history else 0.0,
        "max_raw_grad_norm": max_raw_grad_norm,
        "loss_history": loss_history,
        "weight_norms": weight_norms,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
