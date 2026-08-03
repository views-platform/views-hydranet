"""Free-running autoregressive-rollout diagnostics (C-113 / C-121).

Single-responsibility helpers for the C-113 runaway: given a recurrent
``(x, h) -> ModelOutput`` model, run a *free-running* rollout (each step's
prediction feeds the next input) and report the log-space level the prediction
settles at. The out-of-range attractor is the C-113 signature — a level that
``expm1`` amplifies into astronomical raw counts.

Consumed by:
- ``tests/test_rollout_stability_guard.py`` — the fast regression guard (C-121).
- ``scripts/diagnose_io_gain.py`` — the retrain-free attractor probe (Part B).

Kept dependency-light (only torch + duck-typed ``out.reg``/``out.h_next``) so it
works on the real ``HydraBNUNet06_LSTM4`` and on minimal test models alike.
"""

from __future__ import annotations

import torch

# log1p of the largest observed count; a rollout settling above this is out of the
# data range and expm1-amplifies to catastrophe (the C-113 failure mode).
DATA_LOG_MAX = 12.1
_OUT_OF_RANGE_MARGIN = 1.0


def is_out_of_range(
    level: float,
    data_log_max: float = DATA_LOG_MAX,
    margin: float = _OUT_OF_RANGE_MARGIN,
) -> bool:
    """True if a settled log-space ``level`` is out of the data range — the C-113
    signature the regression guard must catch. NaN/inf count as out-of-range."""
    if not (level == level) or level == float("inf"):  # NaN or +inf
        return True
    return level > data_log_max + margin


@torch.no_grad()
def free_running_attractor(
    model,
    x0: torch.Tensor,
    h0: torch.Tensor,
    steps: int = 48,
    update_state: bool = True,
    emit_fn=None,
) -> tuple[float, list[float]]:
    """Run ``steps`` free-running autoregressive steps; return ``(final_level, trajectory)``.

    Each step: ``out = model(x, h)``; the fed-back next input is ``out.reg`` (the standard
    log1p point head) unless ``emit_fn`` is given, in which case it is ``emit_fn(out)``.
    ``h <- out.h_next`` when ``update_state`` (the standard rollout; pass ``False`` to hold the
    recurrent state fixed — the retired ``freeze_h='all'`` probe variant).

    ``emit_fn`` exists so the probe measures the SAME quantity inference feeds back when the
    feedback is not the identity (C-142): the hurdle-NB head emits ``log1p(E[y])`` (not the
    count-space ``out.reg``), so feeding ``out.reg`` raw would compare count-space ``mu`` against
    the log-space :data:`DATA_LOG_MAX` bound — a category mismatch. ``emit_fn=None`` (default)
    is byte-identical to the pre-#106 standard path.

    Returns the final-step ``max|·|`` of the fed-back tensor (the attractor's log-space magnitude)
    and the per-step trajectory. Pair with :func:`is_out_of_range` to gate.

    Raises:
        ValueError: if ``steps < 1`` (fail loud rather than return an empty trajectory).
    """
    if steps < 1:
        raise ValueError(f"free_running_attractor: steps must be >= 1, got {steps}")
    x, h = x0, h0
    trajectory: list[float] = []
    for _ in range(steps):
        out = model(x, h)
        x = out.reg if emit_fn is None else emit_fn(out)
        if update_state:
            h = out.h_next
        trajectory.append(float(x.abs().max()))
    return trajectory[-1], trajectory
