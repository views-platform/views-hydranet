"""
Training Engine: Pure training logic with no framework dependencies.

This module contains the core training functions extracted from train_model.py
per Uncle Bob's Dependency Rule: Entity-layer logic (gradient math, sequence
processing, curriculum orchestration) must not depend on Framework-layer
types (ModelPathManager, artifact I/O).

All functions here are importable and testable without views_pipeline_core.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from views_hydranet.distributions import resolve_family
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate
from views_hydranet.utils.body_supervision import (
    _active_window_mask as _resolve_active_window_mask,
)
from views_hydranet.utils.body_supervision import (
    event_threshold_from_config,
    resolve_body_supervision,
)
from views_hydranet.utils.curriculum import CurriculumLearner
from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.quantile_head import QuantileLoss
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
    # C-119: re-seed immediately before construction/init so weight initialization draws from a
    # fixed RNG state — independent of any RNG consumed earlier in the pipeline between the
    # manager's lock and here (the cause of run-to-run init drift). Guarded so seedless callers
    # (some unit tests) are unaffected; production/determinism configs always carry the seeds.
    if config.get("np_seed") is not None:
        ReproducibilityGate.lock_entropy(
            np_seed=config["np_seed"], torch_seed=config.get("torch_seed")
        )
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
    #
    # C-113 bisect: freeze_multitask_balancer (default False) restores the
    # pre-C-111 behavior — log_vars stay frozen at zero (equal weighting), the
    # regime under which the model was stable for years. Used to test whether
    # C-111's active balancer is what destabilised training. See
    # reports/preanalysis_balancer_bisect.md.
    multitaskloss_instance = criterion[2]
    if config.get("freeze_multitask_balancer", False):
        logger.info(
            "C-113 bisect: MultiTaskLoss balancer FROZEN — log_vars excluded from "
            "the optimizer (pre-C-111 equal-weighting regime)."
        )
    else:
        log_var_params = list(multitaskloss_instance.parameters())
        if log_var_params:
            optimizer.add_param_group({"params": log_var_params, "weight_decay": 0.0})

    return (model, criterion, optimizer, scheduler)


class _SequenceIndices:
    """Pre-computed channel indices for the sequence loop (Zero Magic ADR 003)."""

    __slots__ = ("reg", "cls", "feat", "static", "n_reg", "n_cls", "reg_names", "cls_names")

    def __init__(self, feature_names: list[str], config: dict) -> None:
        reg_targets = config.get("regression_targets")
        cls_targets = config.get("classification_targets")
        input_features = config.get("features")
        static_channels = config.get("static_channels") or []  # ADR-060: input-only channels

        self.reg = [feature_names.index(t) for t in reg_targets]
        self.cls = [feature_names.index(t) for t in cls_targets]
        self.feat = [feature_names.index(f) for f in input_features]
        self.static = [feature_names.index(s) for s in static_channels]
        self.n_reg = len(reg_targets)
        self.n_cls = len(cls_targets)
        self.reg_names = reg_targets
        self.cls_names = cls_targets


@contextlib.contextmanager
def _batchnorm_eval(model: nn.Module):
    """Run a forward without letting it touch BatchNorm running statistics.

    ``model.eval()`` on the whole model would also disable dropout and change the forward; this
    flips only the ``_BatchNorm`` modules that are currently training, and restores exactly those.
    ``torch.no_grad()`` is NOT a substitute — it stops gradients, not buffer updates.

    Used by the pushforward auxiliary step (#289), whose extra forward would otherwise write
    self-fed, off-distribution activations into buffers that are saved with the artifact and
    recomputed by the C-184 BN recalibration.
    """
    flipped = [
        m
        for m in model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
    ]
    for m in flipped:
        m.eval()
    try:
        yield
    finally:
        for m in flipped:
            m.train()


def _attach_static_channels(
    dyn_input: torch.Tensor, t0_step: torch.Tensor, idx: "_SequenceIndices"
) -> torch.Tensor:
    """ADR-060 I3: re-attach input-only static channels to the model input as [dynamic ⧺ static].

    Statics (e.g. CoordConv row/col) are geometry-constant — always the true values, never sampled
    and never fed from the output. The model was widened to expect them, so every forward (main
    training AND the Stage-5 diagnostic biopsy) must include them. Empty static ⇒ returns dyn_input
    unchanged, so the pre-seam path is byte-identical (I5)."""
    if idx.static:
        return torch.cat([dyn_input, t0_step[:, idx.static, :, :]], dim=1)
    return dyn_input


# ADR-065 (Epic #158): the point-body mask now lives in views_hydranet.utils.body_supervision.
# Re-exported here for the decay-gate penalty below and for existing importers
# (tests/test_active_window_mask).
_active_window_mask = _resolve_active_window_mask


def _decay_gate_penalty(
    gate_logits: torch.Tensor, cls_target: torch.Tensor, decay_active: torch.Tensor
) -> torch.Tensor:
    """Mean gate firing probability on DECAY cells (active in window AND zero now).

    The step-1 leak is the gate hedging (~0.38) on recently-active-now-zero cells (it ranks them
    below true conflict, AUC ~0.75, but does not commit). Penalising this mean sigmoid pushes them
    toward 0 — suppressing the leak — while leaving true-positive cells (label != 0) untouched.

    Args:
        gate_logits: ``[B, H, W]`` pre-sigmoid gate output for one classification target.
        cls_target: ``[B, H, W]`` binary label (a cell is a decay cell only where this is 0).
        decay_active: ``[B, H, W]`` bool — cell active at any window step (active_window mask).

    Returns:
        Scalar penalty = mean sigmoid over (decay_active & target==0); 0.0 if there are none.
    """
    decay = decay_active & (cls_target == 0)
    if not decay.any():
        return gate_logits.new_zeros(())
    return torch.sigmoid(gate_logits)[decay].mean()


def _family_target_log1p_mean(reg: torch.Tensor, family) -> torch.Tensor:
    """Collapse a family head's per-target activated params ``[B, n_reg*n_params, H, W]``
    (target-major) to per-target ``log1p(E[y])`` ``[B, n_reg, H, W]`` — the self-zeroed mean
    ``(1-π)μ`` for zinb, ``μ`` for nb. Shared by the training biopsy and the forensic dossier so
    both plot the honest per-target forecast (C-213), not a raw channel. Mirrors the
    ``hydranet_inference._emit_magnitude`` family branch.
    """
    npar = family.n_params
    means = [
        family.mean(reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1))
        for j in range(reg.shape[1] // npar)
    ]
    return torch.log1p(torch.stack(means, dim=1))  # [B, n_reg, H, W]


def _family_feedback_log1p(reg, family, mode, gate, composition, threshold, generator=None):
    """EXP-4 (GTF): the per-target log1p feedback for scheduled sampling with a family head.

    'mean' => log1p(E[y]) (fixes the legacy ss path, which fed raw n_params channels — shape-
    mismatched to the n_reg dynamic inputs). 'sample' => one composition-aware family DRAW per
    target (mirrors inference `_sample_feedback`), so training exposure == deployment exposure.
    Returns ``[B, n_reg, H, W]`` in log1p space, matching the dynamic input channels.

    ``generator`` (C-261): seeds the family draw + composition Bernoulli so a parity test can
    assert byte-equality against ``hydranet_inference._sample_feedback`` (seeded). NOTE: the
    production call site passes ``None`` (global RNG), so SS training feedback is NOT
    byte-reproducible today — the seeded path is exercised only by the parity test (C-261).
    """
    if mode != "sample":
        return _family_target_log1p_mean(reg, family)
    npar = family.n_params
    n_reg = reg.shape[1] // npar
    draws = torch.stack(
        [
            family.sample(
                reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1), 1, generator
            ).squeeze(-1)
            for j in range(n_reg)
        ],
        dim=1,
    )  # [B, n_reg, H, W] count space
    if composition != "self_zeroed":
        from views_hydranet.distributions.composition import compose_samples

        cube = draws.permute(0, 2, 3, 1).unsqueeze(-1)  # [B,H,W,n_reg,1]
        g = gate[:, :n_reg].permute(0, 2, 3, 1)  # [B,H,W,n_reg]
        cube = compose_samples(cube, g, composition, threshold, generator)
        draws = cube.squeeze(-1).permute(0, 3, 1, 2)  # -> [B, n_reg, H, W]
    return torch.log1p(draws.clamp(min=0.0))


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
    qs99_weight: float | None = None,
    qs99_tau: float | None = None,
    target_weights: dict[str, float] | None = None,
    ss_epsilon: float = 0.0,
    cls_valid_mask: torch.Tensor | None = None,
    decay_gate_weight: float = 0.0,
    body_supervision: str = "all",
    onset_lead: int = 0,
    cessation_lag: int = 0,
    event_threshold: float = 0.0,
    pi_penalty_weight: float | None = None,
    pi_penalty_prior_logit: float = 0.0,
    family=None,
    ss_feedback: str = "mean",
    forecast_composition: str = "self_zeroed",
    gate_threshold: float | None = None,
    pushforward_weight: float = 0.0,
    pushforward_detach_state: bool = False,
    ss_backprop_through_feedback: bool = False,
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

    # Point-body supervision window (ADR-065 + amend. 2026-07-28): resolve the validated
    # body_supervision config to a concrete cell-timestep set ONCE, here — no `if mode ==` ladder
    # in the loss loop. It applies only to a POINT body: a latent loss owns its own zero handling
    # (validated by validate_body_supervision_latent), so it is a no-op there and we keep the
    # all-cell path (byte-identical for `all`+latent). `event_threshold` from derivation (C-195).
    apply_body_mask = body_supervision == "active" and not use_latent
    body_mask_full: torch.Tensor | None = (
        resolve_body_supervision(onset_lead, cessation_lag, event_threshold)(
            train_tensor[:, 1:, idx.reg, :, :]
        )
        if apply_body_mask
        else None
    )

    # Opt-in gate-side decay penalty (2026-06-28 gate_resolution): suppress the GATE on cells
    # active in the window but zero now. Keyed only on the true targets (threshold 0), so — unlike
    # body active_window mask (C-180) — it applies even under a latent body (production hurdle_nb).
    decay_active: torch.Tensor | None = None
    if decay_gate_weight > 0:
        decay_active = _active_window_mask(train_tensor[:, 1:, idx.reg, :, :], 0.0)

    prev_pred: torch.Tensor | None = None

    for i in range(seq_len - 1):
        t0 = train_tensor[:, i, :, :, :]
        t1 = train_tensor[:, i + 1, :, :, :]

        # ADR 046: Separate ground truth signals explicitly
        y_reg = t1[:, idx.reg, :, :]
        y_cls = t1[:, idx.cls, :, :]

        # Forward pass: Feed ONLY the input features (Zero Magic)
        t0_gt = t0[:, idx.feat, :, :]

        # ADR-056: scheduled sampling may replace the ground-truth DYNAMIC features with prediction
        if ss_epsilon > 0.0 and prev_pred is not None:
            mask = torch.rand(t0_gt.shape[0], 1, 1, 1, device=device) < ss_epsilon
            dyn_input = torch.where(mask, prev_pred, t0_gt)
        else:
            dyn_input = t0_gt

        # ADR-060 I3: re-attach static (input-only) channels [dynamic ⧺ static]. See helper.
        t0_input = _attach_static_channels(dyn_input, t0, idx)

        output = model(t0_input, h)
        t1_pred, t1_pred_class, h = output.reg, output.cls, output.h_next
        # EXP-4 (GTF): only compute the fed-back copy when scheduled sampling is active (ε>0), so
        # ε=0 stays byte-identical (branch never runs). A family emits n_params channels/target, so
        # the raw `t1_pred` is shape-mismatched to the n_reg dynamic inputs — collapse it to the
        # per-target log1p mean ('mean', fixes the legacy family path) or a composition-aware DRAW
        # ('sample', EXP-4: train exposure == deploy exposure). Legacy point head: raw pred.
        if ss_epsilon > 0.0:
            # BPTT-SA (#308, Vlachas2023). The `.detach()` below is the ONLY cut in the training
            # graph's feedback path: the recurrent state `h` already flows across steps undetached
            # (~383-step graph, one backward() -- 2026-08-26 audit), so with it the model is told
            # "step i was wrong" but never "and step i-1 produced the input that made it wrong".
            # That is standard scheduled sampling, and it is a mechanical account of why M26-M33
            # failed: the fed-back prediction is a constant, so no credit reaches its producer.
            # Leaving it attached lets the gradient reach back through the handoff.
            # Default False keeps every existing arm byte-identical.
            if family is not None:
                fed = _family_feedback_log1p(
                    t1_pred,
                    family,
                    ss_feedback,
                    torch.sigmoid(t1_pred_class),
                    forecast_composition,
                    gate_threshold,
                )
            else:
                fed = t1_pred
            prev_pred = fed if ss_backprop_through_feedback else fed.detach()

        t1_pred_for_loss = output.reg_latent if use_latent else t1_pred

        # --- FORENSIC RECORDING (ADR 001 Custodian) ---
        # step_idx=i tags the within-window timestep so the dossier can split t=0 / early / late
        # from the full-window aggregate (horizon-split columns).
        if forensics:
            # ADR-067 (C-213): a distribution-family reg head emits n_params channels PER target
            # (target-major). Collapse to per-target log1p(E[y]) so the forensic plots each
            # target's OWN self-zeroed forecast (not a raw channel of target-0), and record the
            # activated params for the param-health frame. Point/legacy heads keep the raw path.
            _fl = next(
                (
                    lf
                    for lf in (
                        criterion_reg.values()
                        if isinstance(criterion_reg, dict)
                        else [criterion_reg]
                    )
                    if isinstance(lf, FamilyLoss)
                ),
                None,
            )
            _yh_ey = _family_target_log1p_mean(t1_pred, _fl.family) if _fl is not None else None
            for j, target_name in enumerate(idx.reg_names):
                if _fl is not None:
                    npar = _fl.n_params
                    forensics.record(
                        f"REG:{target_name}", y_reg[:, j : j + 1], _yh_ey[:, j : j + 1], step_idx=i
                    )
                    forensics.record_params(
                        f"REG:{target_name}",
                        t1_pred[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1),
                        y_reg[:, j],
                        step_idx=i,
                    )
                else:
                    forensics.record(
                        f"REG:{target_name}",
                        y_reg[:, j : j + 1],
                        t1_pred[:, j : j + 1],
                        step_idx=i,
                    )
            for j, target_name in enumerate(idx.cls_names):
                forensics.record(
                    f"CLS:{target_name}",
                    y_cls[:, j : j + 1],
                    torch.sigmoid(t1_pred_class[:, j : j + 1]),
                    step_idx=i,
                )

        # Loss computation
        losses_list = []
        qs99_loss = torch.tensor(0.0, device=device)
        # C-200: family parameter-prior ridge (e.g. ZINB π/μ-ridge), accumulated across targets and
        # scaled by pi_penalty_weight at the reduction site (qs99/decay additive precedent).
        pi_pen = torch.tensor(0.0, device=device)
        for j in range(idx.n_reg):
            # Issue #44: per-target loss instance (or shared single instance)
            loss_fn_j = (
                criterion_reg[idx.reg_names[j]]
                if isinstance(criterion_reg, dict)
                else criterion_reg
            )

            # ADR-067 (C-207): a distribution family emits n_params channels per target; slice this
            # target's n_params and move them to the last dim ([B,H,W,n_params]) for family.nll.
            # Quantile head: K channels per target (multi-tau pinball). Every other head keeps one
            # channel per target. Registry-first via FamilyLoss, so legacy dispatch is untouched.
            if isinstance(loss_fn_j, FamilyLoss):
                npar = loss_fn_j.n_params
                pred_j = t1_pred_for_loss[:, j * npar : (j + 1) * npar, :, :].permute(0, 2, 3, 1)
            elif isinstance(loss_fn_j, QuantileLoss):
                k = loss_fn_j.taus.numel()
                pred_j = t1_pred_for_loss[:, j * k : (j + 1) * k, :, :].permute(0, 2, 3, 1)
            else:
                pred_j = t1_pred_for_loss[:, j, :, :]
            target_j = y_reg[:, j, :, :]

            # C-87: per-target loss weight (1.0 if not configured)
            tw = 1.0
            if target_weights is not None:
                tw = target_weights[idx.reg_names[j]]

            if apply_body_mask:
                # ADR-065: the pre-resolved point-body mask for target j at this step. `pos_cells`
                # = per-step positives; `pos_timelines` = the full timeline of ever-active cells
                # (the decay signal, dossier 15). Resolved once above; here we only index it.
                mask = body_mask_full[:, i, j]
                if isinstance(loss_fn_j, FamilyLoss):
                    # C-199: pass the mask as a broadcast weight into family.nll instead of
                    # boolean-indexing pred_j[mask] — numerically identical for a 0/1 mask, but a
                    # graph-connected 0 on an all-zero-active batch (not a detached tensor(0.0)).
                    losses_list.append(
                        tw * loss_fn_j(pred_j, target_j, weight=mask.to(pred_j.dtype))
                    )
                elif mask.any():
                    losses_list.append(tw * loss_fn_j(pred_j[mask], target_j[mask]))
                else:
                    losses_list.append(torch.tensor(0.0, device=device))

                # C-48: QS99 regularizer (distribution-free pinball on mu). Undefined over a family
                # param vector [.,n_params] (F-1) → skip for FamilyLoss.
                # Only active when the body is masked and weight > 0.
                qs99_active = (
                    qs99_weight is not None
                    and qs99_weight > 0
                    and qs99_tau is not None
                    and mask.any()
                    and not isinstance(loss_fn_j, FamilyLoss)
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

            # C-200/C-205: family parameter-prior ridge (family-agnostic hook — nb returns a
            # graph-safe 0, zinb the π/μ-ridge). Accumulated on the full-grid params; scaled by
            # pi_penalty_weight at the reduction site. Byte-identical no-op when weight is None/0.
            if pi_penalty_weight and isinstance(loss_fn_j, FamilyLoss):
                pi_pen = pi_pen + tw * loss_fn_j.family.parameter_penalty(
                    pred_j, prior_logit=pi_penalty_prior_logit, scale=1.0
                )

        decay_pen = torch.tensor(0.0, device=device)
        for j in range(idx.n_cls):
            pred_cj = t1_pred_class[:, j, :, :]
            targ_cj = y_cls[:, j, :, :]
            # Gate-side decay penalty (opt-in): accumulate on the FULL grid BEFORE any valid-mask.
            if decay_active is not None and j < decay_active.shape[1]:
                decay_pen = decay_pen + _decay_gate_penalty(pred_cj, targ_cj, decay_active[:, j])
            # C-181 (opt-in A/B): restrict the gate loss to valid (land) cells, matching the
            # hurdle-masked reg loss and the priogrid-masked eval. None ⇒ full grid (default).
            if cls_valid_mask is not None:
                pred_cj = pred_cj[:, cls_valid_mask]
                targ_cj = targ_cj[:, cls_valid_mask]
            # per-target gate loss (list) OR a shared instance (mirrors the reg per-target dict)
            crit_cj = (
                criterion_class[j]
                if isinstance(criterion_class, (list, tuple))
                else criterion_class
            )
            losses_list.append(crit_cj(pred_cj, targ_cj))

        # ── PUSHFORWARD (Brandstetter et al. 2022, #289) ────────────────────────────────────
        # An AUXILIARY loss. It does not change the recurrent trajectory — `h` is threaded by the
        # main teacher-forced loop only — and at `pushforward_weight == 0.0` the branch is never
        # entered, so that setting is byte-identical to the pre-flag model.
        #
        # It is NOT side-effect-free, and two side effects have to be suppressed explicitly:
        #
        # 1. **BatchNorm.** The model is in `train()` mode, so an extra `model(...)` call updates
        #    `running_mean`/`running_var`/`num_batches_tracked` on all 15 BN layers — measured at
        #    `num_batches_tracked` 5 -> 9 on a T=6 window, i.e. roughly HALF the saved BN
        #    statistics would come from self-fed, off-distribution inputs. Those buffers are model
        #    state: they go into the artifact and are used at eval. Worse, `train()` is also the
        #    workhorse of `_recalibrate_bn` (the C-184 fix, `bn_recalibrate: True` by default),
        #    which resets BN and recomputes with `momentum=None` — a cumulative average, so the
        #    pushforward forwards would carry EQUAL weight rather than an EMA discount. An arm with
        #    `pushforward_weight > 0` would then differ from its control at the BN layer for
        #    reasons unrelated to the auxiliary loss, confounding the A/B. Suppressed by running
        #    the extra forward with BN in eval mode.
        # 2. **The recalibration pass itself.** `_recalibrate_bn` calls this under `no_grad`, where
        #    the pushforward loss is computed and discarded — pure cost. Skipped via
        #    `torch.is_grad_enabled()`.
        #
        # Dropout still advances the global RNG stream on the extra forward. Harmless (it cannot
        # reach the `pushforward_weight == 0.0` control, which never enters this branch), but two
        # arms differing only in `pushforward_weight` do not share a dropout stream.
        #
        # Take one extra step from the model's OWN emitted field and score it against ground truth
        # at t+2 — "unroll 2 steps, cut the gradient after the first, compute the loss at the
        # pushforward time". The gradient cut is `.detach()` on the fed field, which on the
        # `sample` path is already unavoidable (`torch.poisson` severs it; measured as
        # exactly 0.0),
        # so on that path that half of the paper's method is a NO-OP and the SECOND UNROLL is the
        # entire intervention. Recorded because implementing only the detach would have done
        # nothing while looking correct.
        #
        # `pushforward_detach_state` is a fork the paper cannot settle: its solver is stateless, so
        # cutting the input cuts every path. Ours is recurrent and `h` carries gradient, so False
        # (default) lets the pushforward loss train the RECURRENCE to produce states that survive
        # one step of self-feeding. True reproduces the stateless reading.
        pf_loss = None
        if (
            pushforward_weight > 0.0
            and family is not None
            and i + 2 < seq_len
            and torch.is_grad_enabled()  # see (2) above: no-op under _recalibrate_bn's no_grad
        ):
            fed = _family_feedback_log1p(
                t1_pred,
                family,
                ss_feedback,
                torch.sigmoid(t1_pred_class),
                forecast_composition,
                gate_threshold,
            ).detach()
            # The fed field stands in for t+1, so its static channels come from t1, matching how
            # the main path pairs `dyn_input` with its own `t0`. A no-op while statics are
            # geometry-constant, but the inconsistent pairing would become a real bug the moment a
            # time-varying "static" channel is added.
            pf_in = _attach_static_channels(fed, t1, idx)
            pf_h = h.detach() if pushforward_detach_state else h
            with _batchnorm_eval(model):  # see (1) above
                pf_out = model(pf_in, pf_h)
            y2_reg = train_tensor[:, i + 2, idx.reg, :, :]
            pf_terms = []
            for j, name in enumerate(idx.reg_names):
                loss_fn_j = (
                    criterion_reg[name] if isinstance(criterion_reg, dict) else criterion_reg
                )
                # `family is not None` guarantees FamilyLoss, which always exposes n_params.
                # Indexed rather than getattr-with-default: a default of 1 would silently
                # mis-slice a multi-channel head instead of failing.
                npar = loss_fn_j.n_params
                # C-87 parity with the main reg loop: the same per-target weight, or the auxiliary
                # term silently re-weights the targets relative to the primary objective.
                tw_pf = 1.0 if target_weights is None else target_weights[name]
                pf_terms.append(
                    tw_pf
                    * loss_fn_j(
                        pf_out.reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1),
                        y2_reg[:, j],
                    )
                )
            pf_loss = torch.stack(pf_terms).sum()

        losses = torch.stack(losses_list)
        loss = cast(Any, multitaskloss_instance)(losses)
        if pf_loss is not None:
            loss = loss + pushforward_weight * pf_loss
        if qs99_weight is not None and qs99_weight > 0:
            loss = loss + qs99_weight * qs99_loss
        if decay_gate_weight > 0:
            loss = loss + decay_gate_weight * decay_pen
        if pi_penalty_weight:
            loss = loss + pi_penalty_weight * pi_pen
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

    Reduces train() from 13 parameters to 5: ctx, sample_handler, pbar, stage_label, ss_epsilon.
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
    # ADR-062: indices are read off the kept model tensor (inputs ⧺ targets) = tensor_cols,
    # the de-overloaded feature_cols (== old feature_cols value → byte-identical).
    feature_names = [n for n in sample_handler.channel_map if n in sample_handler.tensor_cols]
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

    # C-181 (opt-in A/B): land mask for the gate loss, read from the priogrid (id_col) channel
    # via include_identities=True so it is North-Up aligned with the model tensor. None ⇒ off
    # (gate trained on the full zero-filled grid, the default).
    cls_valid_mask = None
    if config.get("cls_valid_mask"):
        _full = sample_handler.to_pytorch(device, include_identities=True)  # [B, T, C_full, H, W]
        _pg = sample_handler.channel_map.index(sample_handler.id_col)
        cls_valid_mask = _full[0, 0, _pg] > 0  # [H, W] bool — land cells

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
        qs99_weight=config.get("qs99_weight"),
        qs99_tau=config.get("qs99_tau"),
        target_weights=config.get("target_weights"),
        ss_epsilon=ss_epsilon,
        cls_valid_mask=cls_valid_mask,
        decay_gate_weight=config.get("decay_gate_weight", 0.0),
        # ADR-065 amend.: the point-body supervision window is driven ONLY by the validated
        # body_supervision + onset_lead/cessation_lag (C-194). Event threshold comes from the
        # binary-target derivation — the sole authority for "what is an event" (C-195).
        body_supervision=config.get("body_supervision", "all"),
        onset_lead=config.get("onset_lead", 0),
        cessation_lag=config.get("cessation_lag", 0),
        event_threshold=event_threshold_from_config(config),
        # C-200 (ADR-067): family parameter-prior ridge weight + prior logit (None ⇒ no-op).
        pi_penalty_weight=config.get("pi_penalty_weight"),
        pi_penalty_prior_logit=config.get("pi_penalty_prior_logit", 0.0),
        # EXP-4 (GTF): family + composition drive the scheduled-sampling feedback (only when ε>0).
        family=resolve_family(config.get("output_distribution", "standard")),
        ss_feedback=config.get("ss_feedback", "mean"),
        forecast_composition=config.get("forecast_composition", "self_zeroed"),
        gate_threshold=config.get("gate_threshold"),
        pushforward_weight=config.get("pushforward_weight", 0.0),
        pushforward_detach_state=config.get("pushforward_detach_state", False),
        ss_backprop_through_feedback=config.get("ss_backprop_through_feedback", False),
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
                # ADR-060 I3 / C-157: the model was widened for statics — the diagnostic biopsy
                # must re-attach them too, exactly like the main forward, or this forward crashes
                # (N dynamic channels into an N+static-channel model).
                t0_input = _attach_static_channels(t0[:, idx.feat, :, :], t0, idx)
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
                    # A family head emits n_params channels/target; collapse to per-target
                    # log1p(E[y]) so the biopsy's body + E[y] rows are 3-channel like the truth
                    # and gate (else y_hat_cls * y_hat_reg mis-broadcasts and the forensic plot is
                    # skipped). Mirrors _emit_magnitude's family branch. Legacy = unchanged.
                    _yh_reg = t1_pred
                    _fam = resolve_family(config.get("output_distribution", "standard"))
                    if _fam is not None:
                        _yh_reg = _family_target_log1p_mean(t1_pred, _fam)  # [B, n_reg, H, W]
                    acc_yh_reg.append(_yh_reg[0].permute(1, 2, 0).cpu().numpy())
                    acc_y_cls.append(y_cls[0].permute(1, 2, 0).cpu().numpy())
                    acc_yh_cls.append(
                        torch.sigmoid(t1_pred_class[0]).permute(1, 2, 0).cpu().numpy()
                    )
                    if time_idx >= 0:
                        m_id = sample_handler.data[i + 1, 0, 0, time_idx]
                        acc_months.append(m_id)
        model.train()

        if acc_y_reg:
            # self-zeroed families (zinb) forecast the self-zeroed body directly (no ×gate); NB
            # (and legacy) forecast gate × body. Tell the biopsy which, so its forecast is honest.
            _bfam = resolve_family(config.get("output_distribution", "standard"))
            viz.biopsy_training_performance(
                np.stack(acc_y_reg),
                np.stack(acc_yh_reg),
                np.stack(acc_y_cls),
                np.stack(acc_yh_cls),
                stage_label,
                time_indices=acc_months,
                self_zeroed=bool(_bfam is not None and _bfam.self_zeroed),
            )

    # log each sequence/timeline/batch
    train_log(step_total, step_reg, step_cls)

    # RETURN LOSS COMPONENTS
    return {
        "total": result["total"],
        "reg": result["reg"],
        "cls": result["cls"],
    }


def _reset_bn_stats(model: nn.Module) -> int:
    """Reset every BatchNorm's running stats and set momentum=None (cumulative average over the
    subsequent forwards). Returns the number of BN layers reset."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
            n += 1
    return n


def _recalibrate_bn(ctx: "TrainingContext", sampler, planner, config: dict) -> None:
    """C-184 fix: recompute BatchNorm running statistics post-training.

    The recurrent BN accumulates running stats over T-correlated timesteps during training, biasing
    them so eval-mode over-amplifies — the seed-bimodal eval collapse (~40% of trained seeds
    saturate the gate at eval; confirmed root cause + universal fix, register C-184). This resets
    the running stats and re-accumulates them FORWARD-ONLY (no weight change) over real curriculum
    windows, so eval-mode matches the true activation distribution. Validated 2026-06-27: flips
    6/6 bad seeds GOOD and preserves 2/2 good. Disable with config['bn_recalibrate'] = False.
    """
    model = ctx.model
    n_bn = _reset_bn_stats(model)
    n_windows = config.get("bn_recal_windows", config["windows_per_lesson"] * 10)
    logger.info(
        f"🔧 C-184: BN-recalibrating {n_bn} layers over {n_windows} windows "
        f"(forward-only, no weight change)"
    )
    with torch.no_grad():
        for w in range(n_windows):
            target, threshold = planner.get_lesson(w)
            batch, _ = sampler.get_batch(target, threshold, batch_size=1)
            train(ctx, batch[0], None, stage_label="")  # empty label ⇒ forward only, no biopsy
            del batch
    model.eval()


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

    # Fail-loud (C-134): per-lesson metrics below are guarded by `if wandb.run is not None`,
    # which silently no-ops when no run is active. Warn so a missing train run is visible
    # rather than an empty dashboard. wandb is imported lazily here (C-12 pattern), matching
    # the per-lesson logging blocks below.
    import wandb

    if wandb.run is None:
        logger.warning(
            "⚠️ wandb.run is None at training start — per-lesson training metrics will NOT be "
            "logged to wandb. In a real run this means the training phase did not open a wandb "
            "run (see C-132/C-134); only expected under unit tests."
        )

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

    # OPT-IN BN-recalibration (C-184 test, 2026-06-27): load a trained artifact into `model`, reset
    # BN running stats, then run the lesson loop FORWARD-ONLY (skip backward + optimizer.step) to
    # re-accumulate BN stats on the real scaled windows. config.bn_recal_from=None (default) ⇒
    # normal training, byte-unchanged.
    _bn_recal = config.get("bn_recal_from")
    if _bn_recal:
        model.load_state_dict(torch.load(_bn_recal, map_location=device, weights_only=True))
        _reset_bn_stats(model)
        model.train()
        logger.info(f"🔧 BN-recal: loaded {_bn_recal}, reset BN, forward-only recompute")

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
            reverse=config.get("ss_reverse", False),
        )

    # OPT-IN trajectory diagnostic (bifurcation hunt): per-lesson grad-norm + gate-logit-mean +
    # losses → CSV. config.trajectory_log_path=None (default) ⇒ no hooks, no file, byte-unchanged.
    _traj_path = config.get("trajectory_log_path")
    _traj_file = _traj_writer = None
    _traj_acc = {"gate_sum": 0.0, "gate_n": 0}
    if _traj_path:
        import csv as _csv

        def _gate_hook(_m, _i, _o):
            if _m.training:  # training forwards only — skip forensic/eval biopsies
                _traj_acc["gate_sum"] += _o.detach().mean().item()
                _traj_acc["gate_n"] += 1

        for _hn in ("dec_conv4_head1_class", "dec_conv4_head2_class", "dec_conv4_head3_class"):
            dict(model.named_modules())[_hn].register_forward_hook(_gate_hook)
        _traj_file = open(_traj_path, "w", newline="")
        _traj_writer = _csv.writer(_traj_file)
        _traj_writer.writerow(
            ["lesson", "raw_grad_norm", "loss_reg", "loss_cls", "gate_logit_mean"]
        )
        logger.info(f"📈 trajectory-log ON → {_traj_path}")

    # The optional trajectory CSV must close on ANY exit from the lesson loop, not just the happy
    # one. `IntegrityGuardian.monitor` has always been able to raise from inside this loop, and the
    # C-312 non-finite check adds a second such exit; neither reached the `close()` below.
    _traj_cleanup = contextlib.ExitStack()
    if _traj_file is not None:
        _traj_cleanup.callback(_traj_file.close)

    with (
        _traj_cleanup,
        tqdm(
            total=total_iterations, desc="👾 Training HydraNet", unit="month", leave=True
        ) as pbar,
    ):
        # Loop over Strategic Lessons
        for lesson_idx in range(config["total_lessons"]):
            optimizer.zero_grad()  # Reset gradients at start of Lesson
            lesson_loss = torch.tensor(0.0).to(device)
            lesson_reg = 0.0
            lesson_cls = 0.0
            windows_trained = 0  # C-312: how many windows actually backpropagated

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
                # C-312: this guard used to read `if w_loss > 0`, which is NOT the question
                # ADR-014 means to ask. MultiTaskLoss adds `log(stds)`, negative once a task fits
                # well (per task, the term is negative for `L < (r+1)/e`), so with the balancer
                # unfrozen a perfectly healthy window scores below zero and the whole backward was
                # skipped in silence — measured at 2 updates across 6 lessons. The question is
                # "did this window have anything to learn from", i.e. is the loss exactly zero,
                # which is what an all-masked / empty window produces. Under the production config
                # (`freeze_multitask_balancer=True` pins log_vars at 0, so every term is >= 0)
                # `!= 0` and `> 0` select identically. NOTE "every term is >= 0" holds in float32,
                # where mtloss's `log(stds + eps)` = `log(1 + 1e-8)` rounds to exactly 0.0; in
                # float64 it is +1e-8 per task, still >= 0, so the equivalence survives either way.
                # The test does not re-derive this — it observes every window's actual total. See
                # tests/train/test_backward_gate.py
                # ::test_the_new_guard_is_byte_identical_when_frozen.
                w_loss = losses["total"]
                if _bn_recal:
                    # BN-recal is forward-only: no backward, and no reason to abort the pass over
                    # a non-finite loss nobody consumes. Kept OUT of the finiteness check below so
                    # a `bn_recal_from` run behaves exactly as it did before this fix.
                    pass
                elif not torch.isfinite(w_loss):
                    # RuntimeError, not ValueError: IntegrityGuardian.monitor raises RuntimeError
                    # on the same condition a few lines below, and one fail-loud channel is worth
                    # more than a more precise exception type nobody catches.
                    raise RuntimeError(
                        f"Lesson {lesson_idx + 1} window {window_idx + 1}: loss is "
                        f"{w_loss.item()}, not a finite number. Refusing to backpropagate NaN/inf "
                        "into the model — this is a bug upstream in the loss (cf. C-212), not a "
                        "skippable window. The old `if w_loss > 0` guard swallowed it silently."
                    )
                elif w_loss.item() == 0.0:
                    # Reachable only when EVERY term is exactly zero. Under the production config
                    # the classification terms are scored on the full grid with no mask, so an
                    # all-zero target still yields a small non-zero BCE and this branch does not
                    # fire — i.e. in production every window backpropagates, which is the intent.
                    # It stays because a masked / genuinely empty window must not cost a backward.
                    logger.debug(
                        f"Lesson {lesson_idx + 1} window {window_idx + 1}: loss is exactly 0 "
                        "(nothing supervised) — no backward, as ADR-014 intends."
                    )
                else:
                    w_loss.backward()
                    windows_trained += 1

                lesson_loss += w_loss.detach()  # Keep track of magnitude for logging
                lesson_reg += losses["reg"].item()
                lesson_cls += losses["cls"].item()

                # --- PER-WINDOW MEMORY RELEASE (C-07) ---
                del sample_handler, losses, w_loss

            # --- THE OPTIMIZATION GATE (ADR 014) ---
            # C-312: gated on whether gradient was actually produced, not on the SIGN of the
            # accumulated loss. `or _bn_recal` preserves the pre-fix behaviour of the recal pass,
            # which enters this block for its diagnostics but is separately barred from stepping.
            if windows_trained > 0 or _bn_recal:
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

                # OPT-IN trajectory row (pre-clip grad-norm + lesson gate-logit-mean + losses)
                if _traj_writer is not None:
                    _gm = _traj_acc["gate_sum"] / max(_traj_acc["gate_n"], 1)
                    _traj_writer.writerow(
                        [
                            lesson_idx,
                            raw_grad_norm,
                            lesson_reg / config["windows_per_lesson"],
                            lesson_cls / config["windows_per_lesson"],
                            _gm,
                        ]
                    )
                    _traj_file.flush()
                    _traj_acc["gate_sum"] = 0.0
                    _traj_acc["gate_n"] = 0

                # Gradient Clipping. C-314: the threshold used to be the literal 1.0 while the
                # only config field was the on/off bool, so it could not be tuned from a config at
                # all. `clip_grad_max_norm` defaults to 1.0, so every existing config is unchanged.
                # NOTE (still open, C-314): this covers `model.parameters()` only — the balancer's
                # `log_vars` are a separate optimizer param group and are clipped by nothing.
                if config.get("clip_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.get("clip_grad_max_norm", 1.0)
                    )

                # Optimize (Update Weights) — skipped in BN-recal (weights frozen, BN only)
                if not _bn_recal:
                    optimizer.step()

                # ADR-055 / #99: log per-target dispersion once per lesson — sigma for
                # tobit/lognormal, theta for hurdle-NB (TruncatedNBLoss) — whichever the loss
                # exposes. (getattr-guarded so a loss without one doesn't crash the logger.)
                if isinstance(criterion_reg, dict):
                    import wandb

                    if wandb.run is not None:
                        disp_log = {}
                        for name, loss_fn in criterion_reg.items():
                            if hasattr(loss_fn, "sigma"):
                                disp_log[f"sigma/{name}"] = loss_fn.sigma
                            if hasattr(loss_fn, "theta"):
                                disp_log[f"theta/{name}"] = loss_fn.theta
                        if disp_log:
                            wandb.log(disp_log)

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

    # (the trajectory CSV is closed by _traj_cleanup on every exit path, including exceptions)

    # C-184 FIX: post-training BN recalibration (default ON). Mutates `model` in place so the saved
    # artifact has corrected BN running stats (fixes the seed-bimodal eval collapse). Skipped for a
    # bn_recal_from-only experiment run (its lesson loop already recalibrated). Guarded: a recal
    # failure must NEVER lose a completed training run — snapshot the BN buffers first and restore
    # them on any error (so a half-reset model is never saved), then proceed to save as-is.
    if config.get("bn_recalibrate", True) and not _bn_recal:
        _bn_snapshot = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if "running_" in k or "num_batches" in k
        }
        try:
            _recalibrate_bn(ctx, sampler, planner, config)
        except Exception:
            logger.error(
                "C-184 BN-recalibration FAILED — restoring pre-recal BN stats and saving the "
                "trained model as-is (training NOT lost).",
                exc_info=True,
            )
            model.load_state_dict(_bn_snapshot, strict=False)
            model.eval()

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
