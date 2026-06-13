"""C-113 autoregressive-runaway regression guard (closes C-121).

A fast, deterministic guard that fails if a model's free-running rollout leaves the
in-range attractor — so a future retrain (e.g. rollout-training, #77/#78) cannot
silently re-introduce the C-113 runaway. It tests the detection MECHANISM on
controllable tiny models (contractive → bounded; expansive → flagged), via the shared
`free_running_attractor` helper (also consumed by `scripts/diagnose_io_gain.py`).

Register: C-113 (the runaway), C-121 (this guard). ADR-005 Green/Red taxonomy.
"""

import pytest
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hurdle_nb import hurdle_nb_expected_log1p
from views_hydranet.utils.rollout_diagnostics import (
    DATA_LOG_MAX,
    free_running_attractor,
    is_out_of_range,
)


class _GainModel(nn.Module):
    """Minimal (x, h) -> ModelOutput map with a controllable linear gain on the
    fed-back prediction. gain < 1 contracts (stable); gain > 1 expands (runaway).
    n_reg == n_in == 1 so the prediction feeds straight back as the next input."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain
        self.base = 1

    def forward(self, x, h):
        return ModelOutput(reg=self.gain * x, cls=torch.zeros_like(x), h_next=h)


def _seed_inputs(level: float = 2.0, hw: int = 4):
    x0 = torch.full((1, 1, hw, hw), float(level))
    h0 = torch.zeros((1, 1, hw, hw))
    return x0, h0


def test_green_contractive_rollout_stays_in_range():
    """Green: a contractive map (gain < 1) settles in-range over >=12 steps."""
    x0, h0 = _seed_inputs()
    level, traj = free_running_attractor(_GainModel(0.5), x0, h0, steps=12)
    assert len(traj) == 12
    assert not is_out_of_range(level), f"contractive rollout wrongly flagged: {level}"
    assert traj[-1] < traj[0], "contractive trajectory should shrink across steps"


def test_red_expansive_rollout_is_flagged():
    """Red-catcher: an expansive map (gain > 1) — the C-113 signature — MUST be flagged
    out-of-range. If this ever passes silently, the guard has gone blind (C-121)."""
    x0, h0 = _seed_inputs()
    level, traj = free_running_attractor(_GainModel(1.5), x0, h0, steps=12)
    assert is_out_of_range(level), f"expansive runaway NOT flagged (guard blind): {level}"
    assert traj[-1] > traj[0], "expansive trajectory should grow across steps"


def test_guard_is_deterministic():
    """The guard must be deterministic (no per-step RNG) so it is a stable gate."""
    x0, h0 = _seed_inputs()
    a, _ = free_running_attractor(_GainModel(0.9), x0, h0, steps=10)
    b, _ = free_running_attractor(_GainModel(0.9), x0, h0, steps=10)
    assert a == b


def test_is_out_of_range_boundary_and_nonfinite():
    """The detector's key paths: in-range boundary, and the runaway → inf/nan cases
    (a real C-113 blowup yields inf/nan from max|reg|; the guard must catch those)."""
    assert not is_out_of_range(DATA_LOG_MAX)  # in range
    assert not is_out_of_range(DATA_LOG_MAX + 1.0)  # at the inclusive margin boundary
    assert is_out_of_range(DATA_LOG_MAX + 1.01)  # just past the margin
    assert is_out_of_range(float("inf"))  # runaway diverged to +inf
    assert is_out_of_range(float("nan"))  # runaway produced nan


def test_steps_must_be_positive():
    """free_running_attractor rejects steps < 1 (fail loud, not a silent IndexError)."""
    x0, h0 = _seed_inputs()
    with pytest.raises(ValueError):
        free_running_attractor(_GainModel(0.5), x0, h0, steps=0)


# --- C-142: the probe must measure what hurdle-NB inference actually feeds back ---
# Standard head feeds back `out.reg` (already log1p). Hurdle-NB inference feeds back
# `log1p(E[y])` (E[y]=P(y>0)*mu/(1-NB0); C-140), where `out.reg` is count-space mu. So the probe
# must compose `log1p(E[y])` via `emit_fn` — else it compares mu against the log-space DATA_LOG_MAX
# bound (mismatch → wrong verdict). These cases validate that.


class _HurdleSettleModel(nn.Module):
    """Hurdle-NB synthetic that settles at a fixed count-space mean `mu_star` (p~1, large theta
    => E[y]~mu_star). With the emit compose the fed-back quantity is log1p(E[y])~log1p(mu_star);
    without it the raw probe feeds back `out.reg`=mu_star (count space)."""

    def __init__(self, mu_star: float):
        super().__init__()
        self.mu_star = mu_star
        self.base = 1

    def forward(self, x, h):
        reg = torch.full_like(x, float(self.mu_star))  # count-space mean mu
        cls = torch.full_like(x, 20.0)  # sigmoid(20) ~ 1 => P(y>0) ~ 1
        return ModelOutput(reg=reg, cls=cls, h_next=h)


class _HurdleExpandModel(nn.Module):
    """Hurdle-NB synthetic whose composed E[y] grows by `gain` each step: reads the fed-back
    log1p(E[y]_prev), reconstructs E[y]_prev, scales by gain -> mu (p~1, large theta => E[y]~mu).
    gain>1 is a genuine count-space runaway the validated probe must flag."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain
        self.base = 1

    def forward(self, x, h):
        e_prev = torch.expm1(x.clamp_min(0.0))
        reg = (self.gain * e_prev).clamp_min(1e-6)
        cls = torch.full_like(x, 20.0)
        return ModelOutput(reg=reg, cls=cls, h_next=h)


def _hurdle_emit(theta_val: float = 50.0):
    theta = torch.tensor(float(theta_val))
    return lambda out: hurdle_nb_expected_log1p(out.reg, torch.sigmoid(out.cls), theta)


def test_hurdle_nb_expected_log1p_known_value():
    """E[y]=P(y>0)*mu/(1-NB0); mu=2, p=0.5, theta=1 -> NB0=1/3 -> E[y]=1.5 -> log1p(1.5)."""
    out = hurdle_nb_expected_log1p(
        torch.tensor(2.0), torch.tensor(0.5), torch.tensor(1.0)
    )
    assert torch.allclose(out, torch.log1p(torch.tensor(1.5)), atol=1e-6)


def test_hurdle_inrange_count_not_falsely_flagged():
    """C-142: a healthy in-range COUNT (mu~50 => log1p(E[y])~3.9) reads IN-RANGE with the
    count-space-aware emit compose — and WOULD be mis-flagged by the raw `out.reg` probe.
    This is the false-verdict the compose fixes; the regression catch is the raw-path assertion."""
    x0, h0 = _seed_inputs(level=3.9)
    model = _HurdleSettleModel(mu_star=50.0)
    composed_level, _ = free_running_attractor(model, x0, h0, steps=24, emit_fn=_hurdle_emit())
    raw_level, _ = free_running_attractor(model, x0, h0, steps=24)  # emit_fn=None -> raw out.reg
    assert not is_out_of_range(composed_level), (
        f"composed log1p(E[y]) of a healthy count wrongly flagged: {composed_level}"
    )
    assert is_out_of_range(raw_level), (
        "C-142 regression catch: the raw out.reg probe should mis-flag the healthy count "
        f"(category mismatch the emit compose fixes); raw_level={raw_level}"
    )


def test_hurdle_expansive_eymean_is_flagged():
    """Red-catcher: a genuine count-space runaway (composed E[y] grows) MUST be flagged
    by the validated probe — the hurdle-NB analogue of the C-113 signature."""
    x0, h0 = _seed_inputs(level=2.0)
    level, traj = free_running_attractor(
        _HurdleExpandModel(2.0), x0, h0, steps=48, emit_fn=_hurdle_emit()
    )
    assert is_out_of_range(level), f"hurdle-NB E[y] runaway NOT flagged (probe blind): {level}"
    assert traj[-1] > traj[0], "expansive composed trajectory should grow across steps"


def test_emit_fn_none_is_byte_identical_to_raw_reg():
    """Parity: emit_fn=None (default) feeds back `out.reg` unchanged — the standard path is
    bit-identical to pre-#106 behaviour."""
    x0, h0 = _seed_inputs(level=2.0)
    a, ta = free_running_attractor(_GainModel(0.9), x0, h0, steps=10)
    b, tb = free_running_attractor(_GainModel(0.9), x0, h0, steps=10, emit_fn=None)
    assert a == b and ta == tb
