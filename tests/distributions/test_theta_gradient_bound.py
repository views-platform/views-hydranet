"""A-S9 (#176): C-202 — the θ head-channel gradient is bounded by the softplus link.

The register feared a `1/θ` (digamma) explosion driving ~1e6 into the θ head channel as a cell's θ
is pushed to the `NBCore` floor. But the head emits θ via `softplus`, and the chain rule
`dθ/d(raw) = sigmoid(raw) → 0` as θ → 0 CANCELS the `1/θ` term at the head channel: the register's
1e6 is `dNLL/dθ` (θ-space), not the raw-channel gradient the optimizer actually sees.

Scope precisely: in the **floor regime** the register feared (raw_θ ≤ -10 ⇒ θ ≲ 5e-5, near the
`_EPS=1e-6` floor), the head-channel gradient stays ≤ ~1 for ANY target — the explosion is fully
cancelled, so no forward-distorting θ-floor is needed (which would harm the heavy tail this epic
requires). (At *moderate* θ with an *extreme* count the gradient is legitimately larger — that is
real heavy-tail signal, not the `1/θ` floor pathology, and the existing global `clip_grad_norm`
bounds it. That regime is out of C-202's scope.)

TWO CAVEATS ADDED BY THE 2026-08-26 GRADIENT AUDIT (C-313), neither of which invalidates the
result above:

1. The `raw_theta=-16.0` row is **below** the clamp edge (`softplus(raw) == _EPS` at
   `raw ≈ -13.8155`), so at that point the bound is met by a gradient of **exactly 0.0** — the
   channel is dead, not cancelled. The `-10.0` and `-13.0` rows are above the edge and are the
   ones that actually demonstrate the cancellation this test is named for. The row is kept
   rather than deleted because it is the evidence that the dead zone exists;
   `tests/distributions/test_gradient_dead_zones.py` asserts that fact directly.

2. This test, C-202 and its resolution are all about **theta**. The identical clamp applies to
   **mu**, where `dtheta/d(raw) -> 0` has no counterpart and the head-channel gradient near the
   floor is approximately the observed count. That is not a defect in this test — it is simply
   outside its scope, and it is registered separately as C-313.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402

_FLOOR_BOUND = 1.5  # O(1) — softplus cancels the 1/θ explosion in the near-floor regime


@pytest.mark.parametrize("fam_name", ["nb", "zinb"])
@pytest.mark.parametrize("raw_theta", [-10.0, -13.0, -16.0])  # theta near the _EPS floor
@pytest.mark.parametrize("y", [0.0, 5.0, 500.0, 5000.0])
def test_theta_floor_head_channel_gradient_is_bounded(fam_name, raw_theta, y):
    fam = resolve_family(fam_name)
    raw = torch.zeros(1, fam.n_params)
    raw[0, 0] = 1.0  # mu channel
    raw[0, 1] = raw_theta  # theta channel driven toward the floor
    raw.requires_grad_(True)
    params = fam.activate(raw)
    target = torch.log1p(torch.tensor([float(y)]))
    fam.nll(params, target).backward()
    assert torch.isfinite(raw.grad).all()
    g_theta = raw.grad[0, 1].abs().item()
    assert g_theta <= _FLOOR_BOUND, (
        f"{fam_name}: near-floor |dNLL/d(raw_theta)|={g_theta:.3e} exceeds {_FLOOR_BOUND} at "
        f"raw_theta={raw_theta}, y={y} — the softplus link should cancel the 1/θ blow-up (C-202)."
    )
