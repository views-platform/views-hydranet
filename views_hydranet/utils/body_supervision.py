"""Point-body **supervision window** resolver (ADR-065 + amendment 2026-07-28).

The single place that turns the validated ``body_supervision`` config (``all`` / ``active`` +
``onset_lead`` + ``cessation_lag``) into the concrete set of cell-timesteps the POINT body is
supervised on. Validated at the config boundary (``config_initializer``); this module is the pure
mechanism. Reused by ``training_engine._process_sequence``. (Renamed from retired ``body_mask``.)

Semantics (ADR-065 amend. §A2): given a target window ``[B, T, n_reg, H, W]`` and an event
threshold ``thr``, ``active`` supervises a timestep ``t`` iff an active month ``t'`` (``y > thr``)
exists with

    ``t − cessation_lag ≤ t' ≤ t + onset_lead``

i.e. an **asymmetric temporal dilation** of the per-step-positive mask: ``onset_lead`` reaches to
FUTURE active months (supervises the run-up), ``cessation_lag`` reaches to PAST active months
(supervises the decay). Endpoints (byte-identical to the retired keywords):
  * ``onset_lead = cessation_lag = 0``  → per-step positives  (old ``pos_cells``)
  * ``onset_lead, cessation_lag ≥ T−1`` → active-anywhere      (old ``pos_timelines``)
  * config ``body_supervision='all'``   → every cell           (old ``none``; loop's own branch)

``event_threshold`` is the SINGLE authority for "what counts as an event" — sourced from the binary
target derivation config (``event_threshold_from_config``), never a literal baked here (C-195).
"""

from collections.abc import Callable

import torch


def _active_window_mask(reg_target_window: torch.Tensor, threshold: float) -> torch.Tensor:
    """Cells active (> ``threshold``) at ANY timestep in the window → ``[B, n_reg, H, W]`` bool.

    The saturated-radii endpoint (old ``pos_timelines``): supervise the FULL timeline of every
    ever-active cell; never-active (structural-zero) cells stay excluded.
    """
    return (reg_target_window > threshold).any(dim=1)


def resolve_body_supervision(
    onset_lead: int, cessation_lag: int, event_threshold: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the pure supervision-window function for the validated ``active`` config.

    Args:
        onset_lead: months before onset to supervise (reach to future active months). ``≥ 0``.
        cessation_lag: months after cessation to supervise (reach to past active months). ``≥ 0``.
        event_threshold: the event threshold (single authority: the binary derivation config).

    Returns:
        ``window[B, T, n_reg, H, W] -> BoolTensor[same]`` — True where the body is supervised.
    """

    def _supervise(window: torch.Tensor) -> torch.Tensor:
        active = window > event_threshold  # [B, T, n_reg, H, W] bool
        t_len = active.shape[1]
        # Saturated → active-anywhere broadcast (byte-identical to the old pos_timelines path).
        if onset_lead >= t_len - 1 and cessation_lag >= t_len - 1:
            return _active_window_mask(window, event_threshold).unsqueeze(1).expand_as(window)
        out = active.clone()
        # onset_lead: sup[t] |= active[t + d] for d in 1..onset_lead  (reach to future active).
        for d in range(1, onset_lead + 1):
            shifted = torch.zeros_like(active)
            shifted[:, : t_len - d] = active[:, d:]
            out |= shifted
        # cessation_lag: sup[t] |= active[t - d] for d in 1..cessation_lag  (reach to past active).
        for d in range(1, cessation_lag + 1):
            shifted = torch.zeros_like(active)
            shifted[:, d:] = active[:, : t_len - d]
            out |= shifted
        return out

    return _supervise


def event_threshold_from_config(config: dict) -> float:
    """Source the event threshold from the binary-target derivation config (sole authority, C-195).

    ADR-046 makes the derivation the one owner of "what counts as an event". The supervision window
    reuses that threshold rather than restating a literal ``0`` in the loop. Requires the binary
    derivations to agree on one threshold (the mechanism is a single scalar); disagreement is a
    config contradiction and fails loud (ADR-008). No binary derivation ⇒ ``0.0`` (legacy default).
    """
    derivations = config.get("derivations") or {}
    binary = derivations.get("binary") or []
    thresholds = {float(d.get("threshold", 0)) for d in binary}
    if not thresholds:
        return 0.0
    if len(thresholds) > 1:
        raise ValueError(
            "Ambiguous event threshold for the body supervision window: the binary derivations "
            f"declare multiple thresholds {sorted(thresholds)}. The window is a single scalar — "
            "align the binary derivation thresholds or extend to per-target thresholds."
        )
    return next(iter(thresholds))
