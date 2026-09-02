import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from views_hydranet.distributions import resolve_family
from views_hydranet.distributions.composition import compose_mean
from views_hydranet.distributions.sampling import to_cube_samples
from views_hydranet.utils.correlated_bernoulli import correlated_bernoulli
from views_hydranet.utils.disk_guard import assert_cube_fits
from views_hydranet.utils.feedback_field_transforms import (
    inject,
    magnitude_perturb,
    parse_feedback_transform,
    spatial_scramble,
    splice_occurrence_magnitude,
    thin,
)
from views_hydranet.utils.gate_field_structure import gate_structure_stats
from views_hydranet.utils.hurdle_nb import (
    hurdle_lognormal_expected_log1p,
    hurdle_nb_expected_log1p,
    hurdle_point_expected_log1p,
)
from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

if TYPE_CHECKING:
    from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

# Data-informed emit ceiling for the quantile head (log1p count). ~13 ≈ 442k counts, ~4× the sb max
# (113k): keeps the 36-step autoregressive rollout FINITE (rollout only; not scored) and caps a
# residual
# top-quantile over-shoot. A well-trained T=0 0.99 quantile sits well below it, so the scored T=0
# distribution is untouched; this is a rollout/robustness guard, not a scored quantity.
QUANTILE_EMIT_CEIL = 13.0

# The ConvLSTM packs its recurrent state into ONE tensor of 8 equal channel groups:
# hs_1..hs_4 (short-term / hidden) then hl_1..hl_4 (long-term / cell) — see
# HydraBNrecurrentUnet_06_LSTM4.forward's `torch.split(h, split_h, dim=1)`. Freezing a memory type
# is therefore a channel slice, and the halves are contiguous.
# Offset so the feedback-transform RNG cannot share a stream with the sample-feedback RNG.
_FB_TRANSFORM_SEED_NAMESPACE = 10_000_019
# A THIRD stream, for the correlated-feedback copula. It must not share with fb_gen (which drives
# family.sample and compose_samples) nor with the transform RNG, for the same reason those two are
# separated: an intervention drawn from the stream it perturbs correlates with what it measures.
_FB_CORRELATED_SEED_NAMESPACE = 20_000_033

_STATE_GROUPS = 8
FREEZE_RECURRENT_MODES = ("hidden", "cell", "all")


def blend_recurrent_state(
    new: torch.Tensor, anchor: torch.Tensor, mode: str, weight: float = 1.0
) -> torch.Tensor:
    """Return the evolved state with the ``mode`` memory type pulled ``weight`` toward ``anchor``.

    **A diagnostic, not a mechanism.** ``freeze_h`` was retired (ADR-027, 2026-06-05) and stays
    retired: this is reachable only from an explicit ``HydraNetInference`` argument, never from a
    model config, so no production run can enable it.

    It exists because the question ``freeze_h``'s ablation answered is not the question now being
    asked. `reports/results_freezeh_ablation.md` (2026-06-04) measured **regression CRPS** on a
    pre-ADR-070 artifact that exploded at ~1e17 in every arm, and concluded freezing was inert
    against the C-113 runaway. It never measured **classification**, and activation-aware metrics
    did not exist until 2026-08. Whether holding the state preserves *gate* skill across the
    horizon is untested, not refuted (C-222).

    Args:
        new: the state the model just produced, ``[B, C, H, W]``.
        anchor: the state to hold, same shape — in the rollout, the state at the end of the seed
            step, i.e. everything learned from real observations.
        mode: ``"hidden"`` (hold the short-term half), ``"cell"`` (hold the long-term half), or
            ``"all"`` (hold both — the full hard prior).
        weight: how far to pull the selected half back toward the anchor at each step, as a convex
            blend ``weight * anchor + (1 - weight) * new``. **1.0 (default) is a hard freeze and is
            byte-identical to the pre-dial behaviour**; 0.0 is a no-op. Values in between apply an
            exponential pull, so the free evolution decays toward the anchor rather than being
            replaced by it — which is the shape the L=300 result argues for: freezing the cell
            recovers 23% of the oracle gap (M38/M39), leaving 77% open, and a hard freeze is the
            most extreme setting of a dial nobody had turned.

    Returns:
        A new tensor. Neither input is mutated.

    Raises:
        ValueError: unknown ``mode``, mismatched shapes, a channel count not divisible by 8 (the
            split would silently mis-assign memory types), or a ``weight`` outside [0, 1] — an
            extrapolating blend is not a decay and would leave the state off the segment entirely.
    """
    if mode not in FREEZE_RECURRENT_MODES:
        raise ValueError(
            f"blend_recurrent_state: mode must be one of {FREEZE_RECURRENT_MODES}; got {mode!r}."
        )
    if new.shape != anchor.shape:
        raise ValueError(
            f"blend_recurrent_state: shape mismatch, new={tuple(new.shape)} vs "
            f"anchor={tuple(anchor.shape)}. Both must be the same recurrent state tensor."
        )
    channels = new.shape[1]
    if channels % _STATE_GROUPS != 0:
        raise ValueError(
            f"blend_recurrent_state: {channels} channels is not divisible by {_STATE_GROUPS}. "
            "The ConvLSTM state is 4 short-term + 4 long-term groups; an uneven split would "
            "silently hold the wrong memory type."
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            f"blend_recurrent_state: weight must be in [0, 1]; got {weight!r}. Outside that range "
            "the result is an extrapolation, not a blend, and the state leaves the segment "
            "between what the model produced and what it learned from real observations."
        )
    # weight == 1.0 takes the original branches verbatim, so the hard-freeze arms already measured
    # (M38/M39) stay byte-identical rather than passing through new float arithmetic.
    if weight == 1.0:
        if mode == "all":
            return anchor.clone()
        half = channels // 2  # hs_1..hs_4 | hl_1..hl_4
        if mode == "hidden":
            return torch.cat([anchor[:, :half], new[:, half:]], dim=1)
        return torch.cat([new[:, :half], anchor[:, half:]], dim=1)

    # `torch.lerp(a, b, w) == a + w * (b - a)`, i.e. the convex blend, as ONE fused kernel and one
    # allocation, on ONLY the half that survives. The first version wrote
    # `weight * anchor + (1 - weight) * new` across the FULL state and then sliced half of it away.
    # Benchmarked at the real state shape: 36.2 ms/call full-blend vs 12.3 ms/call here, against
    # 7.1 ms for the weight==1.0 `cat` above — at ~36 rollout steps that is seconds per origin,
    # not minutes.
    # (The wall-clock investigation behind those numbers is in the state-freeze-l300 dossier, not
    # here.)
    if mode == "all":
        return torch.lerp(new, anchor, weight)
    half = channels // 2
    if mode == "hidden":
        head = torch.lerp(new[:, :half], anchor[:, :half], weight)
        return torch.cat([head, new[:, half:]], dim=1)
    tail = torch.lerp(new[:, half:], anchor[:, half:], weight)
    return torch.cat([new[:, :half], tail], dim=1)


class HydraNetInference:
    """Handles inference with the HydraNet model.

    Includes model loading, inference execution, and posterior sampling using
    Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(
        self,
        model: Module,
        config: dict,
        device: Optional[str] = None,
        visualizer: Optional["VisualDiagnostics"] = None,
        freeze_recurrent: Optional[str] = None,
        freeze_recurrent_weight: float = 1.0,
        feedback_transform: Optional[str] = None,
        feedback_length_scale: Optional[float] = None,
        record_gate_probe: bool = False,
        body_mean_dump_dir: Optional[str] = None,
        freeze_anchor_roll: Optional[int] = None,
    ) -> None:
        """Initializes the inference pipeline for HydraNet.

        Args:
            model: The trained PyTorch model for inference.
            config: Configuration settings for inference.
            device: The device to run inference on ('cuda' or 'cpu').
                If not specified, it is automatically detected.
            visualizer: Optional VisualDiagnostics observer.
            freeze_recurrent: **Diagnostic only.** ``None`` (default) evolves the full ConvLSTM
                state — the only production behaviour, byte-identical to before this argument
                existed. ``"hidden"`` / ``"cell"`` / ``"all"`` hold that memory type at its
                end-of-seed-step value for the whole free-running rollout; see
                :func:`blend_recurrent_state`. Deliberately **not** a config key, so no model
                config can enable it and ADR-027's retirement of ``freeze_h`` is untouched.
            feedback_transform: **Diagnostic only.** ``None`` (default) feeds back exactly what
                ``rollout_feedback`` produces — byte-identical to before this argument existed. A
                spec string (e.g. ``"thin:0.25"``, ``"use_real"``, ``"wrong_month:-60"``) replaces
                the fed-back **dynamic** channels; statics are never touched. Also not a config
                key. See :mod:`views_hydranet.utils.feedback_field_transforms`.

        Raises:
            TypeError: If model or config are of incorrect types.
            ValueError: If ``freeze_recurrent`` is not None or a known mode.
        """
        # Step 1: Determine the best available device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Step 2: Validate inputs
        if not isinstance(model, Module):
            err_msg = "Expected 'model' to be an instance of torch.nn.Module."

            logger.error(err_msg)

            raise TypeError(err_msg)
        if not isinstance(config, dict):
            err_msg = "Expected 'config' to be a dictionary."

            logger.error(err_msg)

            raise TypeError(err_msg)

        self.model: Module = model
        self.config = config
        # C-113: optional in-domain feedback clamp (per-target log1p ceiling).
        # Bounds ONLY the autoregressive feedback copy, never an emitted prediction.
        # None (default) => no behavior change. See reports/preanalysis_feedback_clamp.md.
        self.feedback_clamp = self._parse_feedback_clamp(config.get("feedback_clamp_log1p"))
        # #101: hurdle-NB inference compose. The model carries output_distribution (#100); the
        # learned per-target theta is attached to the model at load time (sidecar -> fetcher).
        self.output_distribution = getattr(model, "output_distribution", "standard")
        # ADR-067 strangler-fig: a registered family (nb/zinb) owns emit (family.mean) + sampling
        # (family.sample). resolve_family returns None for every legacy value, so those keep their
        # exact emit/sampler path below (byte-identical). Head emits activated params per cell.
        self._family = resolve_family(self.output_distribution)
        # H-SAMPLE / ADR-070: the autoregressive FEEDBACK copy — the C-113 bloom mitigation.
        # None (default) = AUTO: resolve to 'sample' for a registered family head (the mitigation;
        # T=0-neutral so the scored T=0 product is byte-unchanged), 'mean' for a legacy head (raw
        # pred — byte-identical to history). Explicit values override: 'sample' (composition-aware
        # family draw), 'mean' (emit-mean E[y]), 'teacher_forced' (EXP-3 oracle, real input).
        rf = config.get("rollout_feedback")
        if rf is None:
            rf = "sample" if self._family is not None else "mean"
        self.rollout_feedback = rf
        if self.rollout_feedback not in ("mean", "sample", "teacher_forced"):
            raise ValueError(
                "rollout_feedback must be None (auto), 'mean', 'sample', or 'teacher_forced'; "
                f"got {self.rollout_feedback!r}."
            )
        if self.rollout_feedback == "sample" and self._family is None:
            raise ValueError(
                "rollout_feedback='sample' needs a registered distribution family "
                f"(output_distribution={self.output_distribution!r} has none)."
            )
        # Diagnostic recurrent-state freeze (see blend_recurrent_state). Validated here rather
        # than at the call site so a typo fails loud instead of silently running the control arm —
        # a silent no-op would make an experiment report "no effect" when it never ran.
        if freeze_recurrent is not None and freeze_recurrent not in FREEZE_RECURRENT_MODES:
            raise ValueError(
                f"freeze_recurrent must be None or one of {FREEZE_RECURRENT_MODES}; "
                f"got {freeze_recurrent!r}."
            )
        if not 0.0 <= freeze_recurrent_weight <= 1.0:
            raise ValueError(
                f"freeze_recurrent_weight must be in [0, 1]; got {freeze_recurrent_weight!r}."
            )
        self.freeze_recurrent = freeze_recurrent
        self.freeze_recurrent_weight = freeze_recurrent_weight

        # DIAGNOSTIC (silence-vs-fade EXP-3): spatially roll the anchor before holding it. The
        # clamp pins the state to its last real-observation value, which tangles two things — the
        # state's SCALE and its MAP of which cells are hot. A roll is a permutation: it preserves
        # every scalar property of the anchor exactly (norm, mean, variance, per-channel
        # distribution, internal spatial smoothness) and destroys only its correspondence to the
        # geography. So the arm clamps just as hard, to a state just as well-behaved, that is
        # simply about the wrong places — which is what separates "the clamp preserves placement"
        # from "the clamp steadies the state's scale".
        if freeze_anchor_roll is not None:
            if not isinstance(freeze_anchor_roll, int) or isinstance(freeze_anchor_roll, bool):
                raise ValueError(
                    f"freeze_anchor_roll must be an int or None; got {freeze_anchor_roll!r}."
                )
            if freeze_anchor_roll == 0:
                # A zero roll is the plain clamp wearing a rolled arm's label — it would write a
                # duplicate of the control into a file named for the treatment.
                raise ValueError(
                    "freeze_anchor_roll=0 is the identity and reproduces the plain clamp arm; "
                    "run freeze_recurrent alone for that control instead."
                )
            if freeze_recurrent is None:
                # Rolling an anchor nobody holds changes nothing. A silent no-op here would make
                # the arm report "no effect" when it never ran.
                raise ValueError(
                    "freeze_anchor_roll needs freeze_recurrent set — the anchor is only read when "
                    "a memory half is held, so rolling it without a clamp is a no-op."
                )
        self.freeze_anchor_roll = freeze_anchor_roll

        # Diagnostic body-mean dump (silence-vs-fade dossier, 2026-09-02). Same contract as
        # freeze_recurrent: explicit argument, no config key, default None = byte-identical
        # production path. When set, writes the family's count-space body mean E[Y|body] and the
        # gate P(y>0) as raw fields, for the MC-dropout pass 0 only. It reads tensors the family
        # path already computes and adds NO forward pass and NO train()-mode work, so it cannot
        # perturb the run it measures (the BatchNorm scar: an extra train()-mode forward silently
        # wrote running stats). Every statistic is derived OFFLINE from these fields, so no
        # analysis logic sits in the inference path.
        self.body_mean_dump_dir = body_mean_dump_dir
        if body_mean_dump_dir is not None and self._family is None:
            raise ValueError(
                "body_mean_dump_dir needs a registered distribution family "
                f"(output_distribution={self.output_distribution!r} has none)."
            )

        # Diagnostic feedback-field transform (#258/#262 — measuring `the feedback realism gap`).
        # Same contract as freeze_recurrent: explicit argument, no config key, default None =
        # byte-identical production path. Parsed here so an unknown arm raises before any GPU time.
        self.feedback_transform = feedback_transform
        self._feedback_arm = (
            parse_feedback_transform(feedback_transform) if feedback_transform else None
        )
        if self._feedback_arm:
            # The transforms convert with expm1/log1p, which is the round-trip ONLY if every
            # dynamic feature is log1p-transformed. `validate_family_requires_log1p_targets` covers
            # family heads' regression_targets and nothing else, so an asinh/identity feature would
            # run every arm on mis-scaled counts and emit plausible, wrong dose-response numbers —
            # the one failure the transforms module declares must be impossible.
            log1p_cols = set((config.get("transformations") or {}).get("log1p", []))
            not_log1p = [f for f in config.get("features", []) if f not in log1p_cols]
            if not_log1p:
                raise ValueError(
                    f"feedback_transform={feedback_transform!r} needs every dynamic feature in "
                    f"log1p space (the transforms round-trip with expm1/log1p), but {not_log1p} "
                    "are not in transformations['log1p']. Every arm would be run on mis-scaled "
                    "counts and would look plausible."
                )
            # The E4 splice arms read the MODEL's field out of `t0_autoreg`, which is assembled
            # after both rollout_feedback branches. Under 'teacher_forced' that tensor holds the
            # REAL field, so "real occurrence x model magnitude" would splice real with real and
            # report an E4 decomposition in which the model never appeared. Nothing downstream can
            # detect it: the arm runs, scores, and looks like a result.
            if (
                self._feedback_arm[0]
                in (
                    "occurrence_real_magnitude_model",
                    "occurrence_model_magnitude_real",
                )
                and self.rollout_feedback == "teacher_forced"
            ):
                raise ValueError(
                    f"feedback_transform={feedback_transform!r} splices the model's field with "
                    "the real one, but rollout_feedback='teacher_forced' already feeds the real "
                    "— the arm would splice real with real and report an E4 result that never "
                    "involved the model. Use rollout_feedback='sample' or 'mean'."
                )
        # Every arm self-reports the field it actually fed, per (sample, step). This is not
        # instrumentation for one experiment — it is how we check on REAL data that a transform did
        # what its fixture tests say it does. A `thin` arm whose active fraction did not fall is a
        # silent no-op, and would otherwise be published as "this axis does not matter".
        self.feedback_field_stats: List[dict] = []
        # Does the GATE still carry the spatial structure that `compose_samples`' independent
        # Bernoulli then discards? Two fixes with nothing in common hang on the answer — a
        # correlated sampler (no retraining) vs training-side work. See
        # views_hydranet/utils/gate_field_structure.py.
        #
        # OPT-IN, not implied by an arm. Each record runs a randperm over the grid, a topk, and (on
        # sample 0) five correlated draws whose kernels reach 49x49 at the calibration length
        # scales — per origin x step x target, on every arm including ones with nothing to do with
        # the gate. The first probe run was SIGKILLed (rc=137, `gateprobe_manifest.txt`), which is
        # consistent with the instrumentation being the resource problem rather than the model.
        self.record_gate_probe = bool(record_gate_probe)
        self._record_gate_probe = self.record_gate_probe
        self.gate_structure_stats: List[dict] = []
        # DIAGNOSTIC: correlation length for the fed-back gate draw. None = production's
        # independent Bernoulli. Applies to the FEEDBACK path only; the scored cube is untouched.
        if feedback_length_scale is not None and feedback_length_scale <= 0:
            raise ValueError(
                f"feedback_length_scale must be > 0 or None, got {feedback_length_scale}."
            )
        if feedback_length_scale is not None:
            # The copula is reachable only from _sample_feedback, and only on the soft_gate branch.
            # Under any other configuration a run launched with a correlation length executes the
            # CONTROL end to end and reports "correlated sampling is NULL" — a null manufactured by
            # the harness rather than measured. Fail here, as freeze_recurrent and
            # parse_feedback_transform already do, so a diagnostic can never silently no-op.
            comp = self.config.get("forecast_composition", "self_zeroed")
            if self.rollout_feedback != "sample" or comp != "soft_gate":
                raise ValueError(
                    "feedback_length_scale needs rollout_feedback='sample' and "
                    f"forecast_composition='soft_gate' to have any effect; got "
                    f"rollout_feedback={self.rollout_feedback!r}, forecast_composition={comp!r}. "
                    "Under this configuration the correlated sampler is never reached and the run "
                    "would silently produce the independent-Bernoulli control."
                )
        self._feedback_length_scale = feedback_length_scale
        # Set per rollout in `predict`; declared here so the attribute always exists.
        self._fb_correlated_gen: torch.Generator | None = None

        self.hurdle_theta = self._parse_hurdle_theta(getattr(model, "hurdle_nb_theta", None))
        # generic hurdle bodies: lognormal needs sigma (fixed, from sidecar); point needs nothing.
        self.lognormal_sigma = self._parse_lognormal_sigma(
            getattr(model, "hurdle_lognormal_sigma", None)
        )
        self.viz = visualizer or VisualDiagnostics({"diagnostic_visualizations": False})

        # Step 3: Move model to device and configure for inference.
        self.model.to(self.device)
        self.model.eval()
        # ADR-057: enable MC-Dropout with a *locked* (consistent) mask. The model
        # owns its stochastic-dropout state; inference just asks for it. The mask
        # is then refreshed per posterior sample by reset_locked_dropout() at the
        # top of predict(), so it is held fixed across each sample's 36-step
        # autoregressive roll-forward — preventing per-step dropout noise from
        # compounding into runaway predictions (C-113). hasattr-guarded so bare
        # mock models (used in tests) skip cleanly.
        if hasattr(self.model, "set_locked_dropout"):
            self.model.set_locked_dropout(True)

        logger.info("HydraNetInference initialized successfully.")

    def _parse_feedback_clamp(self, raw):
        """Validate the per-target log1p feedback ceiling (C-113). None => disabled.

        Fail-loud (no silent correction): must be a non-empty list of positive
        floats whose length matches regression_targets. Returns a broadcastable
        [1, C, 1, 1] float32 tensor, or None.
        """
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            err_msg = (
                "feedback_clamp_log1p must be a non-empty list of positive floats "
                f"(one per regression target) or None; got {raw!r}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        vals = [float(v) for v in raw]
        if any(v <= 0 for v in vals):
            err_msg = f"feedback_clamp_log1p values must be positive (log1p space); got {vals}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        n_targets = len(self.config.get("regression_targets", []))
        if n_targets and len(vals) != n_targets:
            err_msg = (
                f"feedback_clamp_log1p has {len(vals)} entries but there are "
                f"{n_targets} regression_targets; provide one ceiling per target."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return torch.tensor(vals, dtype=torch.float32).view(1, len(vals), 1, 1)

    def _clamp_feedback(self, t0_autoreg: torch.Tensor) -> torch.Tensor:
        """Bound the fed-back prediction to the per-target in-domain ceiling (C-113).

        Clamps ONLY the autoregressive feedback copy — never an emitted prediction —
        to keep the next-step input within the log1p training range and break the
        runaway ratchet (violet's free-running map settles at log ~40 -> expm1 ~1e17;
        see reports/results_io_gain_diagnostic.md). Only the upper bound is applied
        (ReLU already provides the >=0 floor). Identity when the clamp is unset.
        """
        if self.feedback_clamp is None:
            return t0_autoreg
        ceiling = self.feedback_clamp.to(device=t0_autoreg.device, dtype=t0_autoreg.dtype)
        return torch.minimum(t0_autoreg, ceiling)

    def _real_dynamic(self, full_tensor, model_in_indices, n_dyn: int, step: int):
        """The REAL dynamic channels at ``step``, log1p space — the same expression the
        ``teacher_forced`` branch feeds, sliced to the dynamic prefix.

        Reusing that expression rather than re-deriving a month offset is deliberate: an
        off-by-one here would compare the wrong month and every arm built on it would be
        confidently wrong while looking healthy.
        """
        n_months = full_tensor.shape[1]
        if not 0 <= step < n_months:
            raise ValueError(
                f"feedback transform asked for month index {step}, outside the loaded window "
                f"[0, {n_months - 1}]. Refusing to clamp — a silently substituted month would "
                "make the arm uninterpretable."
            )
        return full_tensor[:, step, model_in_indices, :, :][:, :n_dyn]

    def _apply_feedback_transform(
        self, t0_autoreg, full_tensor, model_in_indices, n_static: int, step: int
    ):
        """Replace the fed-back DYNAMIC channels per the diagnostic arm. Statics are untouched.

        All field manipulation happens in **count space** (``expm1`` in, ``log1p`` out) because
        the model's dynamic inputs are ``log1p(counts)`` with no standardisation. See
        :mod:`views_hydranet.utils.feedback_field_transforms`.
        """
        name, param = self._feedback_arm
        n_dyn = t0_autoreg.shape[1] - n_static
        model_log1p = t0_autoreg[:, :n_dyn]

        # --- step remappings: choose WHICH month's real field, change nothing about it ---------
        if name == "identity":
            return t0_autoreg
        if name in ("use_real", "wrong_month", "shuffle_months"):
            src = {
                "use_real": step,
                "wrong_month": step + int(param or 0),
                "shuffle_months": self._month_shuffle.get(step, step),
            }[name]
            if name == "use_real":
                # The FULL slice, exactly what the teacher_forced branch feeds — including its
                # statics from month `t`. The feedback branch attaches statics from month `origin`
                # instead, so reusing those would make F1's byte-identity depend on the statics
                # being time-invariant, which nothing here asserts.
                return full_tensor[:, src, model_in_indices, :, :]
            real = self._real_dynamic(full_tensor, model_in_indices, n_dyn, src)
            return torch.cat([real, t0_autoreg[:, n_dyn:]], dim=1)

        # --- field transforms: degrade the REAL field (E2) or splice the two (E4) -------------
        real_counts = torch.expm1(
            self._real_dynamic(full_tensor, model_in_indices, n_dyn, step)
        ).clamp(min=0.0)
        model_counts = torch.expm1(model_log1p).clamp(min=0.0)
        g = self._fb_transform_gen

        if name == "thin":
            out = thin(real_counts, p=float(param), generator=g)
        elif name == "inject":
            out = inject(real_counts, q=float(param), generator=g)
        elif name == "magnitude_perturb":
            out = magnitude_perturb(real_counts, sigma=float(param), generator=g)
        elif name == "spatial_scramble":
            out = spatial_scramble(real_counts, permutation=self._scramble_perm)
        elif name == "occurrence_real_magnitude_model":
            out = splice_occurrence_magnitude(
                real_counts, model_counts, generator=g, on_empty_donor="zeros"
            )
        elif name == "occurrence_model_magnitude_real":
            out = splice_occurrence_magnitude(
                model_counts, real_counts, generator=g, on_empty_donor="zeros"
            )
        else:  # pragma: no cover - parse_feedback_transform already rejects unknown names
            raise ValueError(f"unhandled feedback transform {name!r}")

        return torch.cat([torch.log1p(out), t0_autoreg[:, n_dyn:]], dim=1)

    def _record_feedback_stats(
        self, field_log1p, *, origin: int, sample_idx: int, step: int, prev_active
    ):
        """Summarise the field fed at this step, PER TARGET; return its mask for the next call.

        Recorded for EVERY arm, not just the observer one: a `thin` arm whose active fraction did
        not fall, or a `shuffle_months` arm whose persistence did not drop, is a silent no-op that
        would otherwise be published as "this axis does not matter".

        **Per target, not pooled.** An earlier version divided a channel-0 neighbour count by an
        active count summed over batch *and* all three targets — off by 1/(B*C) on the very
        statistic that verifies `spatial_scramble` destroyed clustering — and pooled sb/ns/os into
        one active fraction, hiding a per-target collapse. Both are computed per (batch, channel)
        now.
        """
        counts = torch.expm1(field_log1p).clamp(min=0.0)
        active = counts > 0
        for b in range(active.shape[0]):
            for c in range(active.shape[1]):
                a = active[b, c]
                n_active = int(a.sum())
                af = a.float()
                pairs = float((af[:, :-1] * af[:, 1:]).sum() + (af[:-1, :] * af[1:, :]).sum())
                if prev_active is None:
                    persistence, prev_n = -1.0, 0
                else:
                    prev = prev_active[b, c]
                    prev_n = int(prev.sum())
                    persistence = (float((a & prev).sum()) / prev_n) if prev_n else -1.0
                self.feedback_field_stats.append(
                    {
                        "origin": origin,
                        "sample_idx": sample_idx,
                        "step": step,
                        "target_idx": c,
                        "n_cells": int(a.numel()),
                        "n_active": n_active,
                        "active_fraction": n_active / a.numel(),
                        # -1 = UNDEFINED (no active cells), matching `persistence` below rather
                        # than colliding with a real measurement. 0.0 would be indistinguishable
                        # from "the field was scattered", and averaging the column would then mix
                        # empty fields with scattered ones — biasing the clustering statistic
                        # downward exactly in the collapse regime this study is about.
                        "mean_magnitude_on_active": (
                            float(counts[b, c][a].mean()) if n_active else -1.0
                        ),
                        # P(on | on at the previous step). -1 = no previous step.
                        "persistence": persistence,
                        "neighbour_pairs_per_active": (pairs / n_active) if n_active else -1.0,
                    }
                )
        return active

    def _record_gate_structure(self, gate, *, origin: int, sample_idx: int, step: int):
        """Record, per (origin, sample, step, target), what a coherent sampler COULD do with this
        gate versus what the independent Bernoulli in ``compose_samples`` actually does."""
        # Slice to n_reg, mirroring _sample_feedback's defensive `prob[..., :n_reg]`. The cls head
        # is not guaranteed to carry exactly n_reg channels; an extra one would be written as
        # target_idx=3 and silently corrupt any per-target aggregate over the column.
        n_reg = len(self.config.get("regression_targets", []) or [])
        g = gate.detach().cpu()
        if n_reg:
            g = g[:, :n_reg]
        for b in range(g.shape[0]):
            for c in range(g.shape[1]):
                rec = gate_structure_stats(
                    g[b, c],
                    generator=self._fb_transform_gen,
                    # The sweep is ~5 extra correlated draws per record; restricting it to the
                    # first posterior sample keeps the cost off every arm while still giving
                    # 13 origins x 35 steps of calibration data.
                    sweep_length_scales=(sample_idx == 0),
                )
                rec.update(origin=origin, sample_idx=sample_idx, step=step, target_idx=c)
                self.gate_structure_stats.append(rec)

    def _parse_hurdle_theta(self, theta):
        """Per-target NB dispersion theta for the hurdle-NB mean (#101). None unless hurdle_nb.

        Accepts the learned theta dict {target: value} (persisted in the artifact sidecar,
        attached to the model at load). Orders by regression_targets -> [1, C, 1, 1] tensor.
        Fail-loud if hurdle_nb is active but theta is missing/incomplete.
        """
        if self.output_distribution != "hurdle_nb":
            return None
        targets = list(self.config.get("regression_targets", []))
        if not theta or not targets:
            err_msg = (
                "output_distribution='hurdle_nb' requires per-target theta (from the artifact "
                f"sidecar) and regression_targets; got theta={theta!r}, targets={targets!r}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        try:
            vals = [float(theta[t]) for t in targets]
        except (KeyError, TypeError) as exc:
            err_msg = f"hurdle_nb theta missing a target entry: {theta!r} vs {targets!r}."
            logger.error(err_msg)
            raise ValueError(err_msg) from exc
        if any(v <= 0 for v in vals):
            raise ValueError(f"hurdle_nb theta values must be > 0; got {vals}.")
        return torch.tensor(vals, dtype=torch.float32).view(1, len(vals), 1, 1)

    def _parse_lognormal_sigma(self, sigma):
        """Fixed lognormal scale sigma for the hurdle-lognormal compose. None unless that body.

        Accepts a scalar (shared across targets) or a {target: value} dict (persisted in the
        artifact sidecar). Orders by regression_targets -> [1, C, 1, 1]. Fail-loud if missing.
        """
        if self.output_distribution != "hurdle_lognormal":
            return None
        targets = list(self.config.get("regression_targets", []))
        if sigma is None or not targets:
            err_msg = (
                "output_distribution='hurdle_lognormal' requires a fixed sigma (from the artifact "
                f"sidecar) and regression_targets; got sigma={sigma!r}, targets={targets!r}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if isinstance(sigma, dict):
            try:
                vals = [float(sigma[t]) for t in targets]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"hurdle_lognormal sigma missing a target: {sigma!r}.") from exc
        else:
            vals = [float(sigma)] * len(targets)
        if any(v <= 0 for v in vals):
            raise ValueError(f"hurdle_lognormal sigma must be > 0; got {vals}.")
        return torch.tensor(vals, dtype=torch.float32).view(1, len(vals), 1, 1)

    def _roll_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        """Spatially roll the clamp anchor — the EXP-3 dissociation arm.

        The clamp's benefit could come from the anchor's SCALE (holding the state's magnitudes
        steady) or from its MAP (which cells are hot). Rolling separates them: ``torch.roll`` is a
        permutation, so the returned anchor is byte-identical to the original as a multiset — same
        norm, same mean, same variance, same per-channel distribution, same internal spatial
        structure — and differs only in where that structure sits relative to the geography.

        Raises:
            ValueError: if the shift is a whole number of grid widths, which rolls the tensor back
                onto itself and would silently run the control arm under the treatment's label.
        """
        shift = self.freeze_anchor_roll
        h, w = anchor.shape[-2], anchor.shape[-1]
        if shift % h == 0 and shift % w == 0:
            raise ValueError(
                f"freeze_anchor_roll={shift} is a whole number of grid periods for a {h}x{w} "
                f"field, so torch.roll returns the anchor unchanged — this would run the plain "
                f"clamp under a rolled arm's label. Choose a shift that is not a multiple of both."
            )
        return torch.roll(anchor, shifts=(shift, shift), dims=(-2, -1))

    def _dump_body_mean(self, params_zstack, gate_thwc, origin: int) -> None:
        """Write the raw body mean ``E[Y|body]`` and the gate ``P(y>0)`` as fields (diagnostic).

        The silence-vs-fade question — does the free-running field lose CELLS or lose SIZE — needs
        the body mean **un-composed**, because every composed readout conflates the two. The cube
        cannot supply it: ``compose_samples`` applies a per-draw ``Bernoulli(gate)`` mask to family
        DRAWS, so ``expm1(cube)/gate`` is unbiased for ``mu`` but its variance is inflated by
        ``1/gate`` — unusable exactly in the collapsed-gate regime the question is about.

        So dump the two factors separately and derive everything offline. Writes MC-dropout pass 0
        only (the fields are per-pass; one pass is enough for a field-level ratio and keeps the
        artifact small). No forward pass is added and no ``train()``-mode work is done — these are
        tensors the family path already computed.
        """
        fam = self._family
        npar = fam.n_params
        params = torch.as_tensor(np.asarray(params_zstack), dtype=torch.float32)
        n_reg = params.shape[1] // npar
        # ADR-068: mirror _emit_magnitude's core switch, or the dumped body is not the one
        # actually emitted.
        mean_fn = fam.mean_core if self.config.get("emit_family_core", False) else fam.mean
        mu = torch.stack(
            [
                mean_fn(params[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1))
                for j in range(n_reg)
            ],
            dim=1,
        )  # [T, n_reg, H, W] count-space E[Y|body], NOT composed with the gate
        out_dir = Path(self.body_mean_dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gate = torch.as_tensor(np.asarray(gate_thwc), dtype=torch.float32)
        np.savez_compressed(
            out_dir / f"bodymean_origin{origin}.npz",
            mu=mu.detach().cpu().numpy().astype(np.float32),
            gate=gate.detach().cpu().numpy().astype(np.float32),
            origin=np.int64(origin),
            n_reg=np.int64(n_reg),
        )

    def _emit_magnitude(self, reg: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """Map the raw regression-head output to the emitted/fed-back magnitude.

        Standard: identity (reg is already the log1p-space point prediction).
        Hurdle-NB (#101): reg is the count-space NB mean mu; emit log1p(E[y]) where the EXACT
        zero-truncated hurdle mean is E[y] = P(y>0) * mu / (1 - NB0(mu, theta))
        (Cragg/Mullahy/Cameron&Trivedi). log1p so the downstream inverse_transform (expm1)
        recovers E[y] in count space — we never expm1 a free prediction (C-140).
        """
        # ADR-067 (A-S8): a registered family emits per-cell activated params [B, n_reg*n_params,
        # H, W]; E[y] = family.mean(params) per target, then log1p (emit space) so the downstream
        # expm1 recovers counts — same contract as the hurdle branches. getattr keeps the A-S2
        # parity stub (no _family) on the legacy path below, so those stay byte-identical.
        fam = getattr(self, "_family", None)
        if fam is not None:
            npar = fam.n_params
            n_reg = reg.shape[1] // npar
            # ADR-068 emit_family_core: feed back the π-stripped core mean (the LARGE body actually
            # emitted for {gated,th_gated}_ZINBcore), not the small (1-π)μ self-zeroed mean.
            mean_fn = fam.mean_core if self.config.get("emit_family_core", False) else fam.mean
            means = torch.stack(
                [
                    mean_fn(reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1))
                    for j in range(n_reg)
                ],
                dim=1,
            )  # [B, n_reg, H, W] count-space E[Y|body]
            # ADR-069 (#183): compose with the cls gate at emit time so the fed-back point forecast
            # is the composed arm. self_zeroed => passthrough (byte-identical: zinb (1-π)μ, nb μ).
            comp = self.config.get("forecast_composition", "self_zeroed")
            if comp != "self_zeroed":
                means = compose_mean(
                    means, prob[:, :n_reg], comp, self.config.get("gate_threshold")
                )
            return torch.log1p(means)  # [B, n_reg, H, W]

        # Single source of truth for each hurdle mean (shared with the explosion-check probe so it
        # feeds back exactly this — C-142). All return log1p(count-space E[y]).
        if self.output_distribution == "hurdle_nb":  # mu/(1-NB0) -> 1 as mu->0
            return hurdle_nb_expected_log1p(reg, prob, self.hurdle_theta)
        if self.output_distribution == "hurdle_lognormal":
            return hurdle_lognormal_expected_log1p(reg, prob, self.lognormal_sigma)
        if self.output_distribution == "hurdle_shrinkage":
            return hurdle_point_expected_log1p(reg, prob)
        if self.output_distribution == "dense_nb":  # C-168: dense (non-truncated, NO-gate) NB body
            # reg is the count-space NB mean mu (softplus emit), so E[y]=mu directly — no P(y>0)
            # factor, no zero-truncation normalizer. log1p so downstream expm1 recovers mu.
            return torch.log1p(reg.clamp(min=0.0))
        if self.output_distribution == "quantile":
            # reg is the K monotone log1p-space quantiles/target (already the emit space). Bound to
            # [0, QUANTILE_EMIT_CEIL] — a DATA-informed ceiling (log1p count ~13 ≈ 442k, ~4× the sb
            # max 113k) so (a) the 36-step rollout stays finite (C-113 bloom) and (b) an
            # over-inflated cumulative-softplus fan can't peg predictions at 1e13. A well-trained
            # head's 0.99 quantile (~log1p 12) sits below the ceiling, so the scored T=0 tail is
            # untouched.
            return reg.clamp(min=0.0, max=QUANTILE_EMIT_CEIL)
        return reg  # standard: identity (reg is already the log1p-space point prediction)

    def _sample_feedback(
        self, reg: torch.Tensor, prob: torch.Tensor, generator: "torch.Generator"
    ) -> torch.Tensor:
        """H-SAMPLE (EXP-2): one seeded, gate-COMPOSED family draw/cell → log1p, for AR feedback.

        Mirrors `_emit_magnitude`'s family branch but SAMPLES (k=1) not the mean, and applies
        the SAME `forecast_composition` as the mean path (so a mean/sample A/B isolates the one
        variable — feedback content — not gated-vs-ungated): self_zeroed => the family's own draw
        (zinb self-zeroes natively, nb is plain); soft_gate / threshold_gate compose the draw with
        the cls gate `prob`. Under `emit_family_core` it draws the π-stripped CORE (`sample_core`),
        mirroring `_emit_magnitude`'s `mean_core` switch, so the fed-back history matches the
        emitted core (C-234). Drawn on CPU with the caller's seeded generator (S2 #121). Only the
        fed-back copy uses this; the scored cube is unchanged.
        """
        fam = self._family
        npar = fam.n_params
        n_reg = reg.shape[1] // npar
        reg_cpu = reg.detach().to("cpu")
        # C-234 (S1): mirror _emit_magnitude — under emit_family_core the AR feedback must draw the
        # π-stripped CORE (the large body actually emitted for {gated,th_gated}_ZINBcore), NOT the
        # small self-zeroed draw. Feeding back the self-zeroed body while emitting the core makes
        # the rollout incoherent (h≥2 on a history the model never emitted). nb: core==sample.
        draw_fn = fam.sample_core if self.config.get("emit_family_core", False) else fam.sample
        draws = torch.stack(
            [
                draw_fn(
                    reg_cpu[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1), 1, generator
                ).squeeze(-1)
                for j in range(n_reg)
            ],
            dim=1,
        )  # [B, n_reg, H, W] count space
        comp = self.config.get("forecast_composition", "self_zeroed")
        if comp != "self_zeroed":
            from views_hydranet.distributions.composition import compose_samples

            cube = draws.permute(0, 2, 3, 1).unsqueeze(-1)  # [B,H,W,n_reg,1]
            gate = prob.detach().to("cpu")[:, :n_reg].permute(0, 2, 3, 1)  # [B,H,W,n_reg]
            if self._feedback_length_scale is not None and comp == "soft_gate":
                # DIAGNOSTIC: replace the independent Bernoulli with a spatially-correlated draw
                # of the SAME marginals (Gaussian copula). Applied here and ONLY here — the scored
                # cube keeps independent sampling via `to_cube_samples`, so any effect is the model
                # behaving differently, not the metric being handed a prettier cube.
                #
                # Advance `generator` exactly as the control does, then DISCARD the result. The
                # copula consumes a different number of variates than compose_samples' bernoulli,
                # so without this every LATER step's body draw (`draw_fn`, same generator) would
                # come from a different stream than the control's — and the arm would confound
                # "coherent placement" with "different body noise". The copula itself draws from a
                # third, namespaced stream. Same defect class as the C-113 generator coupling, and
                # as the fb_gen/transform separation documented in `predict`.
                compose_samples(cube, gate, comp, self.config.get("gate_threshold"), generator)
                mask = correlated_bernoulli(
                    gate.permute(0, 3, 1, 2),  # [B,H,W,n_reg] -> [B,n_reg,H,W] for the (H,W) tail
                    length_scale=self._feedback_length_scale,
                    generator=self._fb_correlated_gen,
                ).permute(0, 2, 3, 1)  # back to [B,H,W,n_reg]
                cube = cube * mask.unsqueeze(-1)
            else:
                cube = compose_samples(
                    cube, gate, comp, self.config.get("gate_threshold"), generator
                )
            draws = cube.squeeze(-1).permute(0, 3, 1, 2)  # -> [B, n_reg, H, W]
        # log1p emit space, on the model device
        return torch.log1p(draws.clamp(min=0.0)).to(reg.device)

    def _finalize_ar_forensic(
        self,
        truth_accumulator: list,
        pred_accumulator: list,
        stage_label: str,
        time_indices: Optional[List[float]],
    ) -> None:
        """Render the Stage-5 autoregressive forensic from the rollout accumulators.

        Called once per `predict()` (guarded by `sample_idx == 0 and self.viz.active`) **before**
        the `return_params` early-return, so it fires for family (nb/zinb) runs too — C-214 (it was
        previously positioned after that return, i.e. dead for every family run). The accumulators
        already hold per-target `log1p(E[y])` (emit-mean) vs truth, so the plot is family-honest.
        """
        if not truth_accumulator:
            logger.warning(f"🧬 Stage 5: no forensic-biopsy data in {stage_label}")
            return
        logger.info(
            f"Stage 5: Finalizing Autoregressive Forensic for {stage_label} "
            f"({len(truth_accumulator)} steps captured)"
        )
        # Ensure we have exactly 6 frames (padding if the model exploded early).
        while len(truth_accumulator) < 6:
            truth_accumulator.append(np.zeros_like(truth_accumulator[0]))
            pred_accumulator.append(np.zeros_like(pred_accumulator[0]))
        self.viz.biopsy_autoregressive(
            truth_accumulator,
            pred_accumulator,
            stage_label,
            channel_names=self.config["regression_targets"],
            time_indices=time_indices if time_indices else [],
        )

    def predict(
        self,
        full_tensor: torch.Tensor,
        origin: int,
        sample_idx: int,
        feature_names: List[str],
        pbar: Optional[tqdm] = None,
        stage_label: str = "Stage 5",
        time_indices: Optional[List[float]] = None,
        return_params: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts a sequence using the HydraNet model.

        Args:
            full_tensor: Input tensor (batch, time, channels, H, W).
            sample_idx: Current sample index for posterior sampling.
            feature_names: Names of channels in full_tensor.
            pbar: Optional progress bar to update.
            stage_label: Label for visual diagnostics.
            return_params: ADR-067 (A-S8) — for a distribution family, return the per-step
                **activated params** ``[T, n_reg*n_params, H, W]`` (pre-emit) in place of the
                emit-magnitude, so the D×K sampler can draw ``family.sample`` from them. The AR
                feedback still uses the emit-mean, so the rollout trajectory is identical. Legacy
                callers omit it (``False``) → the unchanged emit-magnitude 2-tuple.

        Returns:
            A tuple: magnitudes (or params if ``return_params``) and probabilities zstacks.
        """
        # ADR-057: refresh locked dropout masks once per posterior sample, so the
        # mask is held fixed across this sample's 36-step autoregressive
        # roll-forward and drawn fresh for the next sample. No-op for the
        # standard (unlocked) dropout path.
        if hasattr(self.model, "reset_locked_dropout"):
            self.model.reset_locked_dropout()

        _, seq_len, _, H, W = full_tensor.shape

        # ADR 046: Identify input features by name
        input_features = self.config.get("features", [])
        feat_indices = [feature_names.index(f) for f in input_features]

        # ADR-060: static (input-only) channels — appended after the dynamic features in the
        # model input [dynamic ⧺ static], re-attached unchanged to the AR feedback (I3).
        static_indices = [feature_names.index(s) for s in self.config.get("static_channels", [])]
        model_in_indices = feat_indices + static_indices

        reg_targets = self.config.get("regression_targets", [])
        reg_indices = [feature_names.index(t) for t in reg_targets]

        # Initialize hidden state
        h_tt = (
            self.model.init_hTtime(hidden_channels=self.model.base, H=H, W=W)
            .float()
            .to(self.device)
        )

        # BOUNDARY ANCHORING (ADR 015)
        # History ends at 'origin'. So there are 'origin + 1' months of history.
        time_steps = self.config["time_steps"]

        # GPU Accumulators for sequence steps
        acc_magnitudes = []
        acc_probabilities = []
        # ADR-067 (A-S8): activated family params per emitted step (pre-emit), for the D×K sampler.
        acc_params = []

        # STAGE 5 DIAGNOSTIC: Accumulators
        truth_accumulator = []
        pred_accumulator = []

        # THE UNIFIED CAUSAL LOOP (ADR 015)
        # Total iterations: Digest History (origin) + Autoregression (time_steps)
        t1_pred = None
        # H-SAMPLE (EXP-2): the value fed to the next step. 'mean' => the emit-mean (== t1_pred, so
        # byte-identical to before); 'sample' => a seeded family draw. Generator seeded per dropout
        # pass (sample_idx) so each path is distinct yet deterministic (S2 #121). None when off.
        feedback_mag = None
        fb_gen = (
            torch.Generator(device="cpu").manual_seed(int(self.config["torch_seed"]) + sample_idx)
            if self.rollout_feedback == "sample"
            else None
        )
        # Diagnostic state freeze: the state at the END of the seed step (t == origin) — everything
        # digested from REAL observations. Held for t > origin only, so h=1 is untouched by
        # construction and must come out byte-identical across every mode.
        state_anchor = None
        # Diagnostic feedback transform: per-rollout RNG, plus the two rollout-constant objects.
        # Both are seeded from torch_seed ALONE (not sample_idx) because "which permutation" and
        # "which month order" are properties of the experimental ARM, not of a posterior draw —
        # every sample must see the same scrambling or the arm is not one intervention.
        fb_prev_active = None
        # The copula stream is seeded independently of `_feedback_arm`: a correlation length can be
        # set on its own (the sweep's control is `identity`, but nothing requires an arm), and the
        # generator must exist before the first step either way.
        if self._feedback_length_scale is not None:
            self._fb_correlated_gen = torch.Generator(device="cpu").manual_seed(
                int(self.config["torch_seed"]) + _FB_CORRELATED_SEED_NAMESPACE + sample_idx
            )
        # The gate probe draws from the transform RNG too, and is now independent of
        # `_feedback_arm` — so seed it whenever either is active, not only when an arm is set.
        if self._record_gate_probe and not self._feedback_arm:
            self._fb_transform_gen = torch.Generator(device="cpu").manual_seed(
                int(self.config["torch_seed"]) + _FB_TRANSFORM_SEED_NAMESPACE + sample_idx
            )
        if self._feedback_arm:
            seed = int(self.config["torch_seed"])
            # NAMESPACED away from fb_gen (line ~648), which is seeded `torch_seed +
            # sample_idx`. Two Generators with the same seed emit the same stream, so an
            # un-namespaced transform would draw the SAME uniforms that drive family.sample
            # and compose_samples' bernoulli — correlating the intervention with the quantity
            # it measures. Same defect class as the C-113 shared-generator coupling.
            self._fb_transform_gen = torch.Generator(device="cpu").manual_seed(
                seed + _FB_TRANSFORM_SEED_NAMESPACE + sample_idx
            )
            arm_gen = torch.Generator(device="cpu").manual_seed(seed)
            _, _, hh, ww = full_tensor.shape[0], full_tensor.shape[1], H, W
            self._scramble_perm = torch.randperm(hh * ww, generator=arm_gen)
            steps = list(range(origin + 1, origin + time_steps))
            # A plain randperm leaves fixed points (~1 expected over 35 steps): those steps would
            # feed the TRUE month while being scored as "persistence destroyed" — a silent control
            # inside the treatment arm. Resample until deranged.
            for _ in range(1000):
                order = torch.randperm(len(steps), generator=arm_gen).tolist()
                if all(i != j for i, j in enumerate(order)):
                    break
            else:  # pragma: no cover - astronomically unlikely
                raise RuntimeError("could not draw a derangement for shuffle_months")
            self._month_shuffle = dict(zip(steps, [steps[i] for i in order]))
            # Pre-flight the month range this arm will need. The per-step check would otherwise
            # fire ~30 autoregressive steps into the first origin, wasting GPU on an arm that was
            # mis-specified before it started.
            name, param = self._feedback_arm
            if name in ("use_real", "wrong_month", "shuffle_months") or name not in ("identity",):
                offset = int(param) if name == "wrong_month" else 0
                needed = [s + offset for s in steps] + [origin + offset]
                oob = [m for m in needed if not 0 <= m < full_tensor.shape[1]]
                if oob:
                    raise ValueError(
                        f"feedback arm {self.feedback_transform!r} needs month indices "
                        f"{min(needed)}..{max(needed)}, outside the loaded window "
                        f"[0, {full_tensor.shape[1] - 1}] (e.g. {oob[:3]}). Refusing to start."
                    )
        for t in range(origin + time_steps):
            if t < origin:
                # 1. HISTORY DIGESTION: Update hidden state only
                t0_input = full_tensor[:, t, model_in_indices, :, :]
                h_tt = self.model(t0_input, h_tt).h_next

            elif t == origin:
                # 2. SEED STEP: Month Origin -> Month Origin + 1 (Step 1)
                t0_input = full_tensor[:, t, model_in_indices, :, :]
                output = self.model(t0_input, h_tt)
                t1_pred, t1_pred_class, h_tt = output.reg, output.cls, output.h_next
                if self.freeze_recurrent:
                    state_anchor = h_tt.clone()  # the last state built from real observations
                    if self.freeze_anchor_roll is not None:
                        state_anchor = self._roll_anchor(state_anchor)
                t1_pred_class = torch.sigmoid(t1_pred_class)
                if return_params:
                    acc_params.append(t1_pred)  # activated params, pre-emit
                t1_pred = self._emit_magnitude(t1_pred, t1_pred_class)  # #101: hurdle-NB E[y]
                # H-SAMPLE: what feeds the next step — a sample (from params) or the mean.
                feedback_mag = (
                    self._sample_feedback(output.reg, t1_pred_class, fb_gen)
                    if self.rollout_feedback == "sample"
                    else t1_pred
                )

                acc_magnitudes.append(t1_pred)
                acc_probabilities.append(t1_pred_class)

                if sample_idx == 0 and self.viz.active:
                    # Seed frame for biopsy
                    y_seed = (
                        full_tensor[0, t, reg_indices, :, :]
                        .permute(1, 2, 0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    truth_accumulator.append(y_seed)
                    pred_accumulator.append(y_seed)

                    # Step 1 truth
                    try:
                        y_truth = (
                            full_tensor[0, t + 1, reg_indices, :, :]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        truth_accumulator.append(y_truth)
                    except IndexError:
                        truth_accumulator.append(np.zeros_like(y_seed))

                    y_pred = t1_pred[0].permute(1, 2, 0).detach().cpu().numpy()
                    pred_accumulator.append(y_pred)

            else:
                # 3. AUTOREGRESSION: Pred[k] -> Pred[k+1]
                # C-113: clamp ONLY the fed-back copy to the in-domain ceiling; the
                # emitted prediction (appended below) is never capped.
                # Quantile head emits K channels/target; feed back only the median quantile
                # (1/target)
                # so the AR input keeps the [3 dynamic ⧺ static] width. (Step-1 is unaffected — no
                # feedback at the seed step; rollout quality is an M2 concern.)
                if self.rollout_feedback == "teacher_forced":
                    # EXP-3 ORACLE: feed the REAL month-t input (zero exposure bias). The calib
                    # window is historical, so full_tensor holds real values; this is the
                    # one-step-conditioned ceiling. No feedback/clamp/static re-attach — the real
                    # input already carries [dynamic ⧺ static] in model_in_indices order.
                    t0_autoreg = full_tensor[:, t, model_in_indices, :, :]
                else:
                    # H-SAMPLE: feed back the previous step's chosen copy (mean default / sample).
                    fb = feedback_mag
                    if self.output_distribution == "quantile":
                        k = self.config["n_quantiles"]
                        b, c, hh, ww = fb.shape
                        fb = fb.view(b, c // k, k, hh, ww)[:, :, k // 2]  # median quantile/target
                    t0_autoreg = self._clamp_feedback(fb.detach())
                    # ADR-060 I3: re-attach the geometry-constant static channels to the feedback,
                    # matching the [dynamic ⧺ static] model-input order. The clamp bounds only the
                    # 3 dynamic prediction channels; statics are never clamped. Empty => unchanged.
                    if static_indices:
                        t0_autoreg = torch.cat(
                            [t0_autoreg, full_tensor[:, origin, static_indices, :, :]], dim=1
                        )
                # Diagnostic feedback transform (#258/#262). Applied AFTER both branches so it
                # composes with any rollout_feedback, and it replaces only the dynamic prefix —
                # statics stay exactly as attached above.
                if self._feedback_arm:
                    t0_autoreg = self._apply_feedback_transform(
                        t0_autoreg, full_tensor, model_in_indices, len(static_indices), t
                    )
                    fb_prev_active = self._record_feedback_stats(
                        t0_autoreg[:, : t0_autoreg.shape[1] - len(static_indices)],
                        origin=origin,
                        sample_idx=sample_idx,
                        step=t - origin,
                        prev_active=fb_prev_active,
                    )
                # freeze_h retired (2026-06-05): production always evolves the full ConvLSTM
                # state (the former "none" behaviour) — the only mode that was not a
                # train/inference mismatch, and the freeze was inert vs the C-113 runaway
                # (which rides the prediction→input feedback path, not the state).
                # Durable fix: Axis-B rollout training (rollout_training_dossier, ADR-058).
                #
                # `freeze_recurrent` below does NOT reinstate it: it is an explicit diagnostic
                # argument with no config key, default None (this line then runs unchanged). It
                # exists because the 2026-06 ablation scored regression CRPS and never measured
                # whether holding the state preserves GATE skill (C-222, #258/#262).
                output = self.model(t0_autoreg, h_tt)
                t1_pred, t1_pred_class, h_tt = output.reg, output.cls, output.h_next
                if self.freeze_recurrent:
                    h_tt = blend_recurrent_state(
                        h_tt, state_anchor, self.freeze_recurrent, self.freeze_recurrent_weight
                    )
                t1_pred_class = torch.sigmoid(t1_pred_class)
                if self._record_gate_probe:
                    self._record_gate_structure(
                        t1_pred_class, origin=origin, sample_idx=sample_idx, step=t - origin
                    )
                if return_params:
                    acc_params.append(t1_pred)  # activated params, pre-emit
                t1_pred = self._emit_magnitude(t1_pred, t1_pred_class)  # #101: hurdle-NB E[y]
                # H-SAMPLE: update the fed-back copy for the NEXT step (sample or mean).
                feedback_mag = (
                    self._sample_feedback(output.reg, t1_pred_class, fb_gen)
                    if self.rollout_feedback == "sample"
                    else t1_pred
                )

                # C-20: Soft magnitude guard — detect gradual drift
                # C-51: three-tier escalation (100 → 500 → 1000)
                max_pred = t1_pred.abs().max().item()
                if max_pred > 500.0:
                    logger.error(
                        f"Autoregressive drift SEVERE: step {t}, max |pred| = {max_pred:.1f}. "
                        f"Predictions are almost certainly diverging — "
                        f"IntegrityGuardian will halt at "
                        f"{IntegrityGuardian.PREDICTION_MAGNITUDE_CEILING}."
                    )
                elif max_pred > 100.0:
                    logger.warning(
                        f"Autoregressive drift: step {t}, max |pred| = {max_pred:.1f}. "
                        f"Predictions may be diverging."
                    )

                acc_magnitudes.append(t1_pred)
                acc_probabilities.append(t1_pred_class)

                if sample_idx == 0 and self.viz.active and len(truth_accumulator) < 6:
                    try:
                        y_truth = (
                            full_tensor[0, t + 1, reg_indices, :, :]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        truth_accumulator.append(y_truth)
                    except IndexError:
                        truth_accumulator.append(np.zeros_like(truth_accumulator[0]))

                    y_pred = t1_pred[0].permute(1, 2, 0).detach().cpu().numpy()
                    pred_accumulator.append(y_pred)

            if pbar:
                pbar.update(1)

        # STAGE 5 DIAGNOSTIC: finalize the AR forensic HERE — before the return_params early-return
        # below — so it renders for family (nb/zinb) runs too (C-214: the return skipped it).
        if sample_idx == 0 and self.viz.active:
            self._finalize_ar_forensic(
                truth_accumulator, pred_accumulator, stage_label, time_indices
            )

        # --- BATCH TRANSFERS (Speed Hardening) ---
        # ADR-067 (A-S8): the family sampler wants the pre-emit params, not the emit-mean. The
        # rollout above still fed back the emit-mean, so the trajectory is identical either way.
        if return_params:
            full_params = torch.cat(acc_params, dim=0)  # [T_steps, n_reg*n_params, H, W]
            full_probabilities = torch.cat(acc_probabilities, dim=0)
            del acc_magnitudes, acc_params, acc_probabilities
            if not torch.isfinite(full_params).all():
                err_msg = (
                    f"Model produced non-finite params during sample {sample_idx}. "
                    f"Aborting inference (ADR-003: Fail Loud)."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            return (
                full_params.detach().cpu().numpy(),
                full_probabilities.detach().cpu().numpy(),
            )

        full_magnitudes = torch.cat(acc_magnitudes, dim=0)  # [T_steps, C, H, W]
        del acc_magnitudes  # step tensors no longer needed; free before full+numpy coexist
        full_probabilities = torch.cat(acc_probabilities, dim=0)
        del acc_probabilities

        if not torch.isfinite(full_magnitudes).all():
            err_msg = (
                f"Model produced non-finite predictions during sample {sample_idx}. "
                f"Aborting inference (ADR-003: Fail Loud)."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        pred_magnitudes_zstack = full_magnitudes.detach().cpu().numpy()
        del full_magnitudes  # tensor no longer needed after numpy copy
        pred_probabilities_zstack = full_probabilities.detach().cpu().numpy()
        del full_probabilities

        return pred_magnitudes_zstack, pred_probabilities_zstack

    def generate_posterior_samples(
        self,
        handler: "VolumeHandler",
        origin: Optional[int] = None,
        window_info: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates posterior samples from the model.

        Args:
            handler: VolumeHandler carrier [Months, H, W, Channels].
            window_info: Text for progress reporting.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (posterior_magnitudes_zstack, posterior_probabilities_zstack)
        """

        # 1. Model Entry Gate: Standardized PyTorch Layout
        # We strip identity channels here for the model input
        # HARDENING: Move to GPU ONCE before the loop
        full_tensor = handler.to_pytorch(self.device, include_identities=False).to(self.device)
        _, seq_len, _, H, W = full_tensor.shape

        # ADR 046: Map channel names for consistent indexing in predict()
        feature_names = [n for n in handler.channel_map if n in handler.feature_cols]

        # 2. Extract Time Indices for Forensic Biopsy (Stage 5)
        # month_id is in the channel_map.
        time_indices = None
        if self.viz.active:
            try:
                t_idx = handler.channel_map.index(handler.time_col)
                # handler.data is [T, H, W, C]
                time_indices = handler.data[:, 0, 0, t_idx].tolist()
            except Exception:
                logger.error(
                    "HydraNetInference: Failed to extract time indices "
                    "for diagnostic biopsy — skipping.",
                    exc_info=True,
                )

        # Resolve Origin
        if origin is None:
            # Default to using all available history
            origin = seq_len - 1

        time_steps = len(self.config["steps"])
        n_reg = len(self.config["regression_targets"])
        n_cls = len(self.config["classification_targets"])

        # ADR-067 (A-S8): the posterior width S = D×K. D = n_posterior_samples (MC-dropout, model
        # uncertainty); K = n_head_samples (per-cell family draws, outcome uncertainty). Legacy
        # heads keep K=1, so posterior_S == n_posterior_samples (byte-identical). Single source for
        # every S-width read below.
        posterior_D = self.config["n_posterior_samples"]
        posterior_K = self.config.get("n_head_samples", 1)
        posterior_S = posterior_D * posterior_K

        # RAM preflight: reject an oversize D×K cube BEFORE allocation (auto RAM guard + optional
        # max_posterior_cube_gb cap) — the 37GB-cache-scar class must fail loud, not OOM-kill.
        mag_shape = (time_steps, H, W, n_reg, posterior_S)
        prob_shape = (time_steps, H, W, n_cls, posterior_S)
        assert_cube_fits(
            mag_shape,
            prob_shape,
            dtype=np.float32,
            budget_gb=self.config.get("max_posterior_cube_gb"),
        )

        # Pre-allocate memory
        posterior_magnitudes_zstack = np.zeros(mag_shape, dtype=np.float32)
        posterior_probabilities_zstack = np.zeros(prob_shape, dtype=np.float32)

        # Progress bar logic
        # Digest (origin) + Seed (1) + Autoreg (time_steps - 1) = origin + time_steps
        steps_per_sample = origin + time_steps
        total_inference_steps = posterior_D * steps_per_sample

        desc_prefix = f"[{window_info}] " if window_info else ""

        with tqdm(
            total=total_inference_steps,
            desc=f"{desc_prefix}🎲 Drawing Posterior Samples",
            unit="step",
            leave=False,  # Don't clutter the terminal, the manager has the main bar
        ) as pbar:
            # HARDENING: Explicitly wrap the whole loop in no_grad
            with torch.no_grad():
                if self.output_distribution == "quantile":
                    # Path A: the quantile head's OWN distribution replaces MC-dropout as the
                    # sample
                    # source. Run one pass (the head emits K monotone log1p quantiles/target), then
                    # inverse-CDF resample K -> n_posterior_samples to fill the (T,H,W,n_reg,S)
                    # cube —
                    # byte-compatible with the MC-dropout carrier, so Wrap->Invert->CRPS is
                    # untouched.
                    from views_hydranet.utils.quantile_head import (
                        hurdle_quantiles_to_samples,
                        midpoint_levels,
                    )

                    pred_mag, pred_prob = self.predict(
                        full_tensor,
                        origin,
                        0,
                        feature_names=feature_names,
                        pbar=pbar,
                        stage_label=window_info,
                        time_indices=time_indices,
                    )
                    k = self.config["n_quantiles"]
                    t_n = pred_mag.shape[0]
                    # [T, n_reg*K, H, W] -> [T, H, W, n_reg, K] (channels are target-major)
                    q = pred_mag.reshape(t_n, n_reg, k, H, W).transpose(0, 3, 4, 1, 2)
                    s_n = posterior_S  # quantile is legacy (K=1) => == n_posterior_samples
                    # [T, H, W, n_cls] = P(y>0), gate order = reg
                    prob = pred_prob.transpose(0, 2, 3, 1)
                    # Hurdle compose: gate P(y>0) × positive-magnitude quantiles (mass 1-p at 0).
                    # The
                    # magnitude head is trained on positive cells only, so its quantiles fit a
                    # smooth
                    # positive distribution (no 99.7%-zero cliff) and the gate supplies occurrence.
                    samples = hurdle_quantiles_to_samples(
                        q, midpoint_levels(k), prob[..., :n_reg], s_n
                    )
                    posterior_magnitudes_zstack[:] = samples.astype(np.float32)
                    posterior_probabilities_zstack[:] = np.repeat(prob[..., None], s_n, axis=-1)
                    del pred_mag, pred_prob
                elif self._family is not None:
                    # Path B': D×K family posterior. Keep the D MC-dropout passes (model
                    # uncertainty); each pass returns per-cell activated params, from which we draw
                    # K samples (outcome uncertainty) via family.sample. A single seeded generator
                    # (from torch_seed) makes the draws deterministic run-to-run (S2 #121 gate) —
                    # this is the first non-dropout randomness in inference. Fills S = D×K columns.
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(self.config["torch_seed"])
                    )
                    for d in range(posterior_D):
                        params_zstack, prob_zstack = self.predict(
                            full_tensor,
                            origin,
                            d,
                            feature_names=feature_names,
                            pbar=pbar,
                            stage_label=window_info,
                            time_indices=time_indices,
                            return_params=True,
                        )
                        cols = slice(d * posterior_K, (d + 1) * posterior_K)
                        # gate = classifier P(y>0) (sigmoid(cls)); reused for the composed body and
                        # the gate cube below.
                        prob_thwc = prob_zstack.transpose(0, 2, 3, 1)  # [T,H,W,n_cls]
                        if self.body_mean_dump_dir is not None and d == 0:
                            self._dump_body_mean(params_zstack, prob_thwc, origin)
                        # [T,H,W,n_reg,K] log1p draws for this pass, composed with the gate at emit
                        # time (ADR-069 #183): self_zeroed => the family's own sample unchanged;
                        # soft_gate / threshold_gate mask the draws by the cls gate.
                        posterior_magnitudes_zstack[:, :, :, :, cols] = to_cube_samples(
                            params_zstack,
                            self._family,
                            posterior_K,
                            generator,
                            n_reg,
                            gate=prob_thwc,
                            composition=self.config.get("forecast_composition", "self_zeroed"),
                            threshold=self.config.get("gate_threshold"),
                            pass_index=d,  # ADR-070: per-(pass,step) seed → T=0-invariant cube
                            core=self.config.get("emit_family_core", False),  # ADR-068 ZINBcore
                        )
                        # gate cube: the classifier P(y>0), repeated across the K cols
                        posterior_probabilities_zstack[:, :, :, :, cols] = np.repeat(
                            prob_thwc[..., None], posterior_K, axis=-1
                        )
                        del params_zstack, prob_zstack
                else:
                    for sample_idx in range(posterior_D):
                        pred_magnitudes_zstack, pred_probabilities_zstack = self.predict(
                            full_tensor,
                            origin,
                            sample_idx,
                            feature_names=feature_names,
                            pbar=pbar,
                            stage_label=window_info,
                            time_indices=time_indices,
                        )

                        # Store slices directly without concatenation
                        posterior_magnitudes_zstack[:, :, :, :, sample_idx] = (
                            pred_magnitudes_zstack.transpose(0, 2, 3, 1)
                        )
                        posterior_probabilities_zstack[:, :, :, :, sample_idx] = (
                            pred_probabilities_zstack.transpose(0, 2, 3, 1)
                        )
                        del pred_magnitudes_zstack
                        del pred_probabilities_zstack

            # Explicit release of the input tensor before returning.
            # del + gc.collect() ensures the PyTorch allocator pool receives the
            # memory BEFORE the next origin allocates its own full_tensor.
            del full_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            else:
                gc.collect()  # on CPU, prompt PyTorch allocator to coalesce its pool

        return posterior_magnitudes_zstack, posterior_probabilities_zstack
