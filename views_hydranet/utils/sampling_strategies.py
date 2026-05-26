"""
Sampling strategy registry for VolumeSampler anchor selection.

EXPERIMENTAL — added as part of the structural sensitivity investigation.
Each strategy is a pure function: (activity, threshold, min_events, rng, config) -> (r, c).
Default "threshold" preserves current production behaviour exactly.

Mirrors the loss registry pattern in utils.py (LOSS_REG_REGISTRY).
"""

from __future__ import annotations

import numpy as np


def select_anchor_threshold(
    activity: np.ndarray,
    threshold: int,
    min_events: int,
    rng: np.random.Generator,
    config: dict,
) -> tuple[int, int]:
    """Hard threshold: uniform over cells with activity >= threshold."""
    busy_cells = np.argwhere(activity >= threshold)
    if busy_cells.size > 0:
        r, c = busy_cells[rng.choice(len(busy_cells))]
        return int(r), int(c)
    return int(rng.integers(0, activity.shape[0])), int(rng.integers(0, activity.shape[1]))


def select_anchor_power_law(
    activity: np.ndarray,
    threshold: int,
    min_events: int,
    rng: np.random.Generator,
    config: dict,
) -> tuple[int, int]:
    """Power-law weighting: p proportional to activity^alpha, floor at min_events."""
    alpha = config["sampling_alpha"]
    flat = activity.ravel().astype(np.float64)
    eligible = flat >= min_events
    if not eligible.any():
        return int(rng.integers(0, activity.shape[0])), int(rng.integers(0, activity.shape[1]))
    log_w = np.full_like(flat, -np.inf)
    log_w[eligible] = alpha * np.log(flat[eligible])
    log_w -= log_w[eligible].max()
    weights = np.where(eligible, np.exp(log_w), 0.0)
    total = weights.sum()
    idx = rng.choice(len(weights), p=weights / total)
    r, c = np.unravel_index(idx, activity.shape)
    return int(r), int(c)


def select_anchor_boltzmann(
    activity: np.ndarray,
    threshold: int,
    min_events: int,
    rng: np.random.Generator,
    config: dict,
) -> tuple[int, int]:
    """Boltzmann weighting: p proportional to exp(activity / tau), floor at min_events."""
    tau = config["sampling_temperature"]
    flat = activity.ravel().astype(np.float64)
    eligible = flat >= min_events
    if not eligible.any():
        return int(rng.integers(0, activity.shape[0])), int(rng.integers(0, activity.shape[1]))
    log_w = np.full_like(flat, -np.inf)
    log_w[eligible] = flat[eligible] / tau
    log_w -= log_w[eligible].max()
    weights = np.where(eligible, np.exp(log_w), 0.0)
    total = weights.sum()
    idx = rng.choice(len(weights), p=weights / total)
    r, c = np.unravel_index(idx, activity.shape)
    return int(r), int(c)


def select_anchor_sigmoid(
    activity: np.ndarray,
    threshold: int,
    min_events: int,
    rng: np.random.Generator,
    config: dict,
) -> tuple[int, int]:
    """Sigmoid soft threshold: p = sigmoid(k * (activity - threshold)), zero below min_events."""
    steepness = config["sampling_steepness"]
    flat = activity.ravel().astype(np.float64)
    eligible = flat >= min_events
    if not eligible.any():
        return int(rng.integers(0, activity.shape[0])), int(rng.integers(0, activity.shape[1]))
    x = steepness * (flat - threshold)
    x = np.clip(x, -500, 500)
    weights = np.where(eligible, 1.0 / (1.0 + np.exp(-x)), 0.0)
    total = weights.sum()
    if total == 0:
        return int(rng.integers(0, activity.shape[0])), int(rng.integers(0, activity.shape[1]))
    idx = rng.choice(len(weights), p=weights / total)
    r, c = np.unravel_index(idx, activity.shape)
    return int(r), int(c)


SAMPLING_STRATEGY_REGISTRY = {
    "threshold": {"fn": select_anchor_threshold, "params": []},
    "power_law": {"fn": select_anchor_power_law, "params": ["sampling_alpha"]},
    "boltzmann": {"fn": select_anchor_boltzmann, "params": ["sampling_temperature"]},
    "sigmoid": {"fn": select_anchor_sigmoid, "params": ["sampling_steepness"]},
}
