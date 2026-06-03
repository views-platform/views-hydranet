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
import math
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate
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


def _init_classification_head_bias(model: nn.Module, bias_value: float) -> None:
    """
    Initialize classification decoder head biases to logit(event_rate).

    Autoresearch Finding F1: sigmoid(0) = 0.50 is a 25-71x overestimate of
    the true event rate on zero-inflated PRIO-GRID data. Initializing to
    logit(event_rate) provides 98.5% metric improvement.

    Targets layers named 'dec_conv4_head{N}_class' — the final output
    Conv2d of each classification decoder branch.
    """
    count = 0
    for name, module in model.named_modules():
        is_class_head = "dec_conv4" in name and "_class" in name
        has_bias = hasattr(module, "bias") and module.bias is not None
        if is_class_head and has_bias:
            nn.init.constant_(module.bias, bias_value)
            count += 1
            logger.debug(f"Classification head '{name}' bias → {bias_value:.2f}")

    sigmoid_val = 1.0 / (1.0 + math.exp(-bias_value))
    logger.info(
        f"Onset bias initialization: {count} classification heads set to {bias_value:.2f} "
        f"(sigmoid = {sigmoid_val:.4f}, i.e. {sigmoid_val * 100:.2f}% prior event probability)"
    )


def make(config: dict, device: torch.device):
    model = choose_model(config, device)

    init_fn = functools.partial(init_weights, config=config)
    model.apply(init_fn)

    onset_bias = config.get("onset_bias_init")
    if onset_bias is not None:
        _init_classification_head_bias(model, onset_bias)

    criterion = choose_loss(config, device)
    optimizer, scheduler = choose_scheduler(config, model)

    # ADR-055: add learnable sigma parameters to the optimizer.
    # weight_decay=0.0: like the MultiTaskLoss log_vars below, the learnable
    # per-target log_sigma values are uncertainty estimates, not model weights —
    # weight decay would pull them back toward their initialization (same drag
    # that froze the balancer in C-111).
    criterion_reg = criterion[0]
    if isinstance(criterion_reg, dict):
        for loss_instance in criterion_reg.values():
            sigma_params = list(loss_instance.parameters())
            if sigma_params:
                optimizer.add_param_group({"params": sigma_params, "weight_decay": 0.0})

    # C-111: add the MultiTaskLoss balancer's log_vars to the optimizer so the
    # Kendall et al. (2018) homoscedastic uncertainty weighting can actually
    # learn. Without this, log_vars accumulate gradients but are never stepped,
    # so they stay frozen at their zero initialization and the balancer is inert.
    # weight_decay=0.0: log_vars are uncertainty estimates, not model weights —
    # decaying them toward zero defeats their purpose.
    multitaskloss_instance = criterion[2]
    log_var_params = list(multitaskloss_instance.parameters())
    if log_var_params:
        optimizer.add_param_group({"params": log_var_params, "weight_decay": 0.0})

    return (model, criterion, optimizer, scheduler)


class _SequenceIndices:
    """Pre-computed channel indices for the sequence loop (Zero Magic ADR 003)."""

    __slots__ = ("reg", "cls", "feat", "n_reg", "n_cls", "reg_names", "cls_names")

    def __init__(self, feature_names: list[str], config: dict) -> None:
        reg_targets = config.get("regression_targets")
        cls_targets = config.get("classification_targets")
        input_features = config.get("features")

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
    criterion_reg: nn.Module | dict[str, nn.Module],
    criterion_class: nn.Module,
    multitaskloss_instance: nn.Module,
    idx: "_SequenceIndices",
    device: torch.device,
    pbar: tqdm | None = None,
    forensics: TrainingForensics | None = None,
    hurdle_threshold: float | None = None,
    qs99_weight: float | None = None,
    qs99_tau: float | None = None,
    target_weights: dict[str, float] | None = None,
    ss_epsilon: float = 0.0,
) -> dict[str, Any]:
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

    # Censored losses (TobitLoss) need pre-ReLU latent mu — resolve once
    if isinstance(criterion_reg, dict):
        use_latent = any(getattr(v, "needs_latent", False) for v in criterion_reg.values())
    else:
        use_latent = getattr(criterion_reg, "needs_latent", False) is True

    prev_pred: torch.Tensor | None = None

    for i in range(seq_len - 1):
        t0 = train_tensor[:, i, :, :, :]
        t1 = train_tensor[:, i + 1, :, :, :]

        # ADR 046: Separate ground truth signals explicitly
        y_reg = t1[:, idx.reg, :, :]
        y_cls = t1[:, idx.cls, :, :]

        # Forward pass: Feed ONLY the input features (Zero Magic)
        t0_gt = t0[:, idx.feat, :, :]

        # ADR-056: scheduled sampling — may replace ground truth with model prediction
        if ss_epsilon > 0.0 and prev_pred is not None:
            mask = torch.rand(t0_gt.shape[0], 1, 1, 1, device=device) < ss_epsilon
            t0_input = torch.where(mask, prev_pred, t0_gt)
        else:
            t0_input = t0_gt

        output = model(t0_input, h)
        t1_pred, t1_pred_class, h = output.reg, output.cls, output.h_next
        prev_pred = t1_pred.detach()

        t1_pred_for_loss = output.reg_latent if use_latent else t1_pred

        # --- FORENSIC RECORDING (ADR 001 Custodian) ---
        if forensics:
            for j, target_name in enumerate(idx.reg_names):
                forensics.record(f"REG:{target_name}", y_reg[:, j : j + 1], t1_pred[:, j : j + 1])
            for j, target_name in enumerate(idx.cls_names):
                forensics.record(
                    f"CLS:{target_name}",
                    y_cls[:, j : j + 1],
                    torch.sigmoid(t1_pred_class[:, j : j + 1]),
                )

        # Loss computation
        losses_list = []
        qs99_loss = torch.tensor(0.0, device=device)
        for j in range(idx.n_reg):
            pred_j = t1_pred_for_loss[:, j, :, :]
            target_j = y_reg[:, j, :, :]

            # Issue #44: per-target loss instance (or shared single instance)
            loss_fn_j = (
                criterion_reg[idx.reg_names[j]]
                if isinstance(criterion_reg, dict)
                else criterion_reg
            )

            # C-87: per-target loss weight (1.0 if not configured)
            tw = 1.0
            if target_weights is not None:
                tw = target_weights[idx.reg_names[j]]

            if hurdle_threshold is not None and not use_latent:
                # C-45: Regression loss on positive observations only
                mask = target_j > hurdle_threshold
                if mask.any():
                    losses_list.append(tw * loss_fn_j(pred_j[mask], target_j[mask]))
                else:
                    losses_list.append(torch.tensor(0.0, device=device))

                # C-48: QS99 regularizer (distribution-free pinball on mu)
                # Only active when hurdle is enabled and weight > 0
                qs99_active = (
                    qs99_weight is not None
                    and qs99_weight > 0
                    and qs99_tau is not None
                    and mask.any()
                )
                if qs99_active:
                    error = target_j[mask] - pred_j[mask]
                    pinball = torch.where(
                        error >= 0,
                        qs99_tau * error,
                        (qs99_tau - 1.0) * error,
                    )
                    qs99_loss = qs99_loss + tw * pinball.mean()
            else:
                losses_list.append(tw * loss_fn_j(pred_j, target_j))

        for j in range(idx.n_cls):
            losses_list.append(criterion_class(t1_pred_class[:, j, :, :], y_cls[:, j, :, :]))

        losses = torch.stack(losses_list)
        loss = cast(Any, multitaskloss_instance)(losses)
        if qs99_weight is not None and qs99_weight > 0:
            loss = loss + qs99_weight * qs99_loss
        total_loss += loss

        loss_reg = losses[: idx.n_reg].sum()
        loss_class = losses[idx.n_reg :].sum()

        step_reg.append(loss_reg.detach().item())
        step_cls.append(loss_class.detach().item())
        step_total.append(loss.detach().item())

        if pbar is not None:
            pbar.update(1)

    return {
        "total": total_loss,
        "reg": torch.tensor(np.sum(step_reg)).to(device),
        "cls": torch.tensor(np.sum(step_cls)).to(device),
        "h": h,
        "per_step_losses": (step_total, step_reg, step_cls),
    }


class TrainingContext:
    """Bundles the 'wired once' training components (C-17).

    Reduces train() from 13 parameters to 4: ctx, sample_handler, pbar, stage_label.
    Created once in training_loop(), passed to every train() call.
    """

    __slots__ = (
        "model",
        "optimizer",
        "scheduler",
        "criterion_reg",
        "criterion_class",
        "multitaskloss_instance",
        "config",
        "device",
        "viz",
        "forensics",
    )

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        criterion_reg: nn.Module | dict[str, nn.Module],
        criterion_class: nn.Module,
        multitaskloss_instance: nn.Module,
        config: dict,
        device: torch.device,
        viz: VisualDiagnostics | None = None,
        forensics: TrainingForensics | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion_reg = criterion_reg
        self.criterion_class = criterion_class
        self.multitaskloss_instance = multitaskloss_instance
        self.config = config
        self.device = device
        self.viz = viz
        self.forensics = forensics


def train(
    ctx: TrainingContext,
    sample_handler: VolumeHandler,
    pbar: tqdm,
    stage_label: str = "",
    ss_epsilon: float = 0.0,
) -> dict[str, torch.Tensor]:
    ctx.model.train()
    ctx.multitaskloss_instance.train()

    config = ctx.config
    model = ctx.model
    device = ctx.device
    viz = ctx.viz
    forensics = ctx.forensics

    # 1. Stochastic Data Augmentation (Tube-Level)
    if config.get("random_flips"):
        if np.random.rand() < 0.5:
            sample_handler = sample_handler.flip("H")
        if np.random.rand() < 0.5:
            sample_handler = sample_handler.flip("W")

    # 2. Model Entry Gate: Transform to PyTorch [B, T, C, H, W]
    train_tensor = sample_handler.to_pytorch(device, include_identities=False)

    # 3. Pre-compute channel indices (Zero Magic ADR 003)
    feature_names = [n for n in sample_handler.channel_map if n in sample_handler.feature_cols]
    idx = _SequenceIndices(feature_names, config)

    seq_len = train_tensor.shape[1]
    window_H = train_tensor.shape[-2]
    window_W = train_tensor.shape[-1]

    mem_allocated = torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else 0
    logger.debug(
        f"🚀 Training: Entered Gate with Tensor {train_tensor.shape} | "
        f"GPU Mem: {mem_allocated:.2f} MB"
    )

    # 4. Initialize hidden state
    h = model.init_hTtime(hidden_channels=model.base, H=window_H, W=window_W).to(device)

    # 5. STAGE 5 DIAGNOSTIC
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
            logger.error(
                "Training: Failed to extract time index for diagnostic biopsy — skipping.",
                exc_info=True,
            )

    # --- CORE SEQUENCE PROCESSING ---
    result = _process_sequence(
        train_tensor,
        model,
        h,
        ctx.criterion_reg,
        ctx.criterion_class,
        ctx.multitaskloss_instance,
        idx,
        device,
        pbar=pbar,
        forensics=forensics,
        hurdle_threshold=config.get("hurdle_threshold"),
        qs99_weight=config.get("qs99_weight"),
        qs99_tau=config.get("qs99_tau"),
        target_weights=config.get("target_weights"),
        ss_epsilon=ss_epsilon,
    )
    step_total, step_reg, step_cls = result["per_step_losses"]

    # --- STAGE 5 DIAGNOSTIC: Lightweight midpoint biopsy ---
    if viz and stage_label:
        biopsy_start = max(0, (seq_len // 2) - 3)
        biopsy_end = min(seq_len - 1, biopsy_start + 6)

        # Re-run the midpoint steps in eval mode for diagnostic capture only
        model.eval()
        h_diag = model.init_hTtime(hidden_channels=model.base, H=window_H, W=window_W).to(device)
        with torch.no_grad():
            for i in range(seq_len - 1):
                t0 = train_tensor[:, i, :, :, :]
                t1 = train_tensor[:, i + 1, :, :, :]
                t0_input = t0[:, idx.feat, :, :]
                output_diag = model(t0_input, h_diag)
                t1_pred, t1_pred_class, h_diag = (
                    output_diag.reg,
                    output_diag.cls,
                    output_diag.h_next,
                )

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
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    handler: VolumeHandler,
    device: torch.device,
    run_timestamp: str | None = None,
) -> dict:
    """
    Orchestrates the training process over multiple stochastic samples.
    Returns a diagnostic summary dictionary.
    """
    criterion_reg, criterion_class, multitaskloss_instance = criterion

    ReproducibilityGate.lock_entropy(np_seed=config["np_seed"], torch_seed=config["torch_seed"])
    logger.info("🚀 Training initiated...")

    # Initialize Visual Truth Engine with Authoritative Timestamp
    viz = VisualDiagnostics(config, run_timestamp=run_timestamp)

    # Initialize Forensic Auditor (ADR 001 Custodian)
    forensics = TrainingForensics(config)

    # C-17: Bundle training components into context
    ctx = TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_reg=criterion_reg,
        criterion_class=criterion_class,
        multitaskloss_instance=multitaskloss_instance,
        config=config,
        device=device,
        viz=viz,
        forensics=forensics,
    )

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

    # ADR-056: scheduled sampling mixer (disabled when ss_schedule is None)
    ss_mixer = None
    if config.get("ss_schedule") is not None:
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        ss_mixer = ScheduledSamplingMixer(
            schedule=config["ss_schedule"],
            epsilon_max=config.get("ss_epsilon_max", 1.0),
            warmup_lessons=config.get("ss_warmup_lessons"),
            k=config.get("ss_k"),
        )

    with tqdm(
        total=total_iterations, desc="👾 Training HydraNet", unit="month", leave=True
    ) as pbar:
        # Loop over Strategic Lessons
        for lesson_idx in range(config["total_lessons"]):
            optimizer.zero_grad()  # Reset gradients at start of Lesson
            lesson_loss = torch.tensor(0.0).to(device)
            lesson_reg = 0.0
            lesson_cls = 0.0

            # ADR-056: compute epsilon once per lesson
            ss_epsilon = ss_mixer.get_epsilon(lesson_idx) if ss_mixer is not None else 0.0

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
                losses = train(ctx, sample_handler, pbar, stage_label=slbl, ss_epsilon=ss_epsilon)

                # --- MEMORY-SAFE ACCUMULATION (ADR 014 Hardening) ---
                w_loss = losses["total"]
                if w_loss > 0:
                    w_loss.backward()

                lesson_loss += w_loss.detach()  # Keep track of magnitude for logging
                lesson_reg += losses["reg"].item()
                lesson_cls += losses["cls"].item()

                # --- PER-WINDOW MEMORY RELEASE (C-07) ---
                del sample_handler, losses, w_loss

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
                if config.get("clip_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimize (Update Weights)
                optimizer.step()

                # ADR-055: log per-target sigma once per lesson (after optimizer step)
                if isinstance(criterion_reg, dict):
                    import wandb

                    if wandb.run is not None:
                        wandb.log(
                            {
                                f"sigma/{name}": loss_fn.sigma
                                for name, loss_fn in criterion_reg.items()
                            }
                        )

                # Issue #48: log multi-task loss weights once per lesson
                import wandb

                if wandb.run is not None:
                    mtl_names = config.get("regression_targets", []) + config.get(
                        "classification_targets", []
                    )
                    mtl_log_vars = multitaskloss_instance.log_vars.detach()
                    wandb.log(
                        {f"mtl_log_var/{n}": lv.item() for n, lv in zip(mtl_names, mtl_log_vars)}
                    )

            # ADR-056: log scheduled sampling epsilon once per lesson (outside loss gate)
            if ss_mixer is not None:
                import wandb

                if wandb.run is not None:
                    wandb.log({"ss_epsilon": ss_epsilon})

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
