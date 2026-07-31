"""D×K posterior-cube sampling helper (ADR-067, A-S8).

`to_cube_samples` turns one MC-dropout pass's activated family params into the K per-cell posterior
draws that fill that pass's slice of the `[T,H,W,n_reg,S]` cube. It bridges the space boundary: a
family samples in **count space** (`family.sample`), but the cube — like `_emit_magnitude` — lives
in **log1p space** so the downstream `inverse_transform` (`expm1`) recovers counts. Determinism
rides on the caller-supplied seeded `torch.Generator` (S2 #121 gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from views_hydranet.distributions.composition import compose_samples

if TYPE_CHECKING:
    from views_hydranet.distributions.base import DistributionFamily


def to_cube_samples(
    params_zstack,
    family: "DistributionFamily",
    k: int,
    generator: "torch.Generator | None",
    n_reg: int,
    gate=None,
    composition: str = "self_zeroed",
    threshold: "float | None" = None,
    pass_index: int = 0,
    core: bool = False,
) -> np.ndarray:
    """Draw K per-cell samples/target from activated family params → log1p-space cube slice.

    Args:
        params_zstack: activated params ``[T, n_reg*n_params, H, W]`` (torch/numpy), target-major.
        family: the resolved ``DistributionFamily`` (owns ``n_params``, ``sample``).
        k: per-cell head draws (K).
        generator: seeded ``torch.Generator`` for deterministic sampling (may be ``None``).
        n_reg: number of regression targets (the ``n_reg`` axis width).
        gate: per-cell ``P(y>0)`` ``[T, H, W, n_cls]`` (torch/numpy); the first ``n_reg`` channels
            are used. Required when ``composition`` gates the body; ignored for ``self_zeroed``.
        composition: forecast-composition arm (ADR-069) — ``self_zeroed`` (passthrough, default),
            ``soft_gate`` (per-draw ``Bernoulli(gate)``), or ``threshold_gate`` (keep cell if
            ``gate >= threshold``).
        threshold: τ ∈ (0,1) for ``threshold_gate``.
        pass_index: MC-dropout pass index (D axis). Combined with the run seed + step index to
            seed a per-``(pass, step)`` sub-generator (ADR-070 T=0-neutrality — see below).

    Returns:
        ``np.ndarray`` ``[T, H, W, n_reg, k]`` float32, in **log1p space** (non-negative).
    """
    params = torch.as_tensor(np.asarray(params_zstack), dtype=torch.float32)
    npar = family.n_params
    t, c, h, w = params.shape
    if c != n_reg * npar:
        raise ValueError(
            f"to_cube_samples: params channel dim {c} != n_reg*n_params "
            f"({n_reg}*{npar}={n_reg * npar})."
        )
    # C-240: the core (π-stripped body) has NO zero mechanism of its own — it MUST be externally
    # gated. core=True + self_zeroed would draw the ungated NB core (zeros nowhere) → a dense
    # ~85%-nonzero forecast on a ~99.7%-zero field. Fail loud rather than silently over-forecast
    # (HydraNetConfig rejects this upstream; this guards a direct/ad-hoc to_cube_samples call).
    if core and composition == "self_zeroed":
        raise ValueError(
            "to_cube_samples: core=True requires an external gate (composition='soft_gate' or "
            "'threshold_gate'); core + 'self_zeroed' has no zero mechanism and over-forecasts."
        )
    # ADR-068 emit_family_core: draw the BULK body (π-stripped for zinb) instead of the self-zeroed
    # sample, for the externally-gated {gated,th_gated}_ZINBcore arms. nb: sample_core == sample.
    draw = family.sample_core if core else family.sample
    out = torch.zeros((t, h, w, n_reg, k), dtype=torch.float32)
    gate_t = None
    # ADR-069 (#183): compose the sample cube with the gate at emit time. self_zeroed =>
    # passthrough (byte-identical); soft_gate / threshold_gate mask the log1p draws.
    if composition != "self_zeroed":
        if gate is None:
            raise ValueError(
                f"to_cube_samples: composition '{composition}' needs a gate, got None."
            )
        # first n_reg channels of the gate -> [T,H,W,n_reg]
        gate_t = torch.as_tensor(np.asarray(gate), dtype=torch.float32)[..., :n_reg]
    # ADR-070 (T=0-neutrality): draw EACH timestep from its own sub-generator, seeded from the run
    # seed + (pass, step). A single generator streamed across the 36-step trajectory couples a
    # step's draws to LATER steps' params — torch's batched Gamma rejection re-draws across the
    # whole tensor — so rollout_feedback (which only changes h>=2 params) would perturb the SCORED
    # h=1 cube. Independent per-(pass, step) streams make each step's draw depend only on its own
    # params, so h=1 is byte-invariant to feedback. Deterministic + reproducible (S2 #121): same
    # seed, same pass_index => byte-identical cube. For a single step (T=1) this reproduces the
    # pre-fix draw exactly (LOCKED golden anchors preserved).
    base = int(generator.initial_seed()) if generator is not None else 0
    for tt in range(t):
        seed_t = (base + pass_index * 1_000_003 + tt * 10_007) & 0x7FFF_FFFF_FFFF_FFFF
        gen_t = torch.Generator().manual_seed(seed_t)
        for j in range(n_reg):
            pj = params[tt, j * npar : (j + 1) * npar].permute(1, 2, 0)  # [H,W,n_params]
            out[tt, :, :, j, :] = torch.log1p(draw(pj, k, gen_t))  # count -> log1p (emit) space
        if gate_t is not None:
            out[tt] = compose_samples(out[tt], gate_t[tt], composition, threshold, gen_t)
    return out.numpy()
