"""Point-body training-mask resolver (ADR-065, Epic #158).

The single place that turns a validated ``body_mask`` keyword into the concrete set of cells the
POINT body is supervised on — the Ousterhout "resolve once, no ``if mode ==`` ladder in the loop".
The keyword is validated at the config boundary (``config_initializer.validate_body_mask``); this
module is the pure mechanism it maps to. Reused by ``training_engine._process_sequence``.

Semantics (see ADR-065 §2 taxonomy), given a target window ``[B, T, n_reg, H, W]``:
  * ``none``          → every cell (all-True). The all-cell foundation; the loop takes its own
                        unmasked branch here, so the callable exists only for completeness/tests.
  * ``pos_cells``     → per-step positives: ``y > event_threshold`` at that timestep.
  * ``pos_timelines`` → cells active anywhere in the window (``_active_window_mask``), broadcast to
                        every step so a cell's post-conflict decay-zero steps are also supervised.

``event_threshold`` is the SINGLE authority for "what counts as an event" — sourced from the binary
target derivation config (``event_threshold_from_config``), never a literal baked in here (C-195).
"""

from collections.abc import Callable

import torch


def _active_window_mask(reg_target_window: torch.Tensor, threshold: float) -> torch.Tensor:
    """Cells active (> ``threshold``) at ANY timestep in the window.

    Args:
        reg_target_window: regression targets ``[B, T, n_reg, H, W]`` over the window.
        threshold: event threshold (a cell counts as active where ``y > threshold``).

    Returns:
        ``[B, n_reg, H, W]`` bool — True for a cell active at any timestep. Used by the
        ``pos_timelines`` mask to supervise the FULL timeline of ever-active cells (incl. their
        post-conflict zero steps — the decay signal the per-step mask drops; dossier 15), while
        never-active (structural-zero) cells stay excluded.
    """
    return (reg_target_window > threshold).any(dim=1)


def resolve_body_mask(name: str, event_threshold: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the pure mask function for a validated ``body_mask`` keyword.

    Args:
        name: one of ``none`` / ``pos_cells`` / ``pos_timelines`` (validated upstream).
        event_threshold: the event threshold (single authority: the binary derivation config).

    Returns:
        ``window[B, T, n_reg, H, W] -> BoolTensor[B, T, n_reg, H, W]`` — True where the body is
        supervised. The per-step boolean at ``[:, i, j]`` is the cell set for target ``j`` at step
        ``i``, matching the training loop's indexing.
    """
    if name == "none":
        return lambda window: torch.ones_like(window, dtype=torch.bool)
    if name == "pos_cells":
        return lambda window: window > event_threshold
    if name == "pos_timelines":

        def _timelines(window: torch.Tensor) -> torch.Tensor:
            active = _active_window_mask(window, event_threshold)  # [B, n_reg, H, W]
            return active.unsqueeze(1).expand_as(window)  # broadcast across T

        return _timelines
    # Defensive: config validation (validate_body_mask) already rejects unknown values, so this is
    # a programming error, not a user-config error — fail loud (ADR-008), never silently all-cell.
    raise ValueError(f"resolve_body_mask: unknown body_mask '{name}'.")


def event_threshold_from_config(config: dict) -> float:
    """Source the event threshold from the binary-target derivation config (sole authority, C-195).

    ADR-046 makes the derivation the one owner of "what counts as an event". The point-body mask
    reuses that threshold rather than restating a literal ``0`` in the loop. Requires the binary
    derivations to agree on one threshold (the mask mechanism is a single scalar); disagreement is
    a config contradiction and fails loud (ADR-008). No binary derivation ⇒ ``0.0`` (the legacy
    default), so a config without derivations is unchanged.
    """
    derivations = config.get("derivations") or {}
    binary = derivations.get("binary") or []
    thresholds = {float(d.get("threshold", 0)) for d in binary}
    if not thresholds:
        return 0.0
    if len(thresholds) > 1:
        raise ValueError(
            "Ambiguous event threshold for the body mask: the binary derivations declare "
            f"multiple thresholds {sorted(thresholds)}. The point-body mask is a single scalar — "
            "align the binary derivation thresholds or extend the mask to per-target thresholds."
        )
    return next(iter(thresholds))
