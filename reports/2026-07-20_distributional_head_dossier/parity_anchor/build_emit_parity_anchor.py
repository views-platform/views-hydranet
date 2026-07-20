"""Generate the legacy emit-math parity anchor (A-S2 / ADR-067).

Captures the byte-exact output of the EXISTING ``HydraNetInference._emit_magnitude`` for every
``output_distribution`` value, on a fixed synthetic ``(reg, prob)`` input. The strangler-fig
wiring (A-S6..A-S8) delegates these same branches to distribution families;
``test_parity_anchor.py`` re-runs the real method and asserts byte-identity against this golden,
so any drift in an emit path fails loud.

We call the REAL unbound method through a minimal stand-in that carries only the three attributes
it reads (``output_distribution``, ``hurdle_theta``, ``lognormal_sigma``) — no heavy ``__init__``.
This anchors the actual production math, not a re-derivation of it.

Run: ``conda run -n views-hydranet-env python
reports/2026-07-20_distributional_head_dossier/parity_anchor/build_emit_parity_anchor.py``
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from views_hydranet.utils.hydranet_inference import HydraNetInference

_HERE = Path(__file__).resolve().parent
GOLDEN = _HERE / "emit_parity_golden.json"

# Fixed, seed-free synthetic input. Three targets on a 2x2 grid; values span 0 -> large so every
# branch (zero limit, small, medium, tail) is exercised. Shapes match the [B, C, H, W] the method.
_C, _H, _W = 3, 2, 2
_N = _C * _H * _W
REG = torch.linspace(0.0, 6.0, _N, dtype=torch.float32).view(1, _C, _H, _W)
PROB = torch.linspace(0.05, 0.95, _N, dtype=torch.float32).view(1, _C, _H, _W)
THETA = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32).view(1, _C, 1, 1)  # per-target NB theta
SIGMA = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32).view(1, _C, 1, 1)  # per-target ln sigma

# output_distribution -> extra sidecar params its branch reads (None where the branch ignores them)
CASES = {
    "standard": (None, None),
    "hurdle_nb": (THETA, None),
    "hurdle_lognormal": (None, SIGMA),
    "hurdle_shrinkage": (None, None),
    "dense_nb": (None, None),
    "quantile": (None, None),
}


def emit(name: str, theta, sigma) -> torch.Tensor:
    """Call the real ``_emit_magnitude`` via a stand-in carrying only the attrs it reads."""
    stub = SimpleNamespace(output_distribution=name, hurdle_theta=theta, lognormal_sigma=sigma)
    return HydraNetInference._emit_magnitude(stub, REG, PROB)


def build() -> dict:
    outputs = {name: emit(name, theta, sigma).tolist() for name, (theta, sigma) in CASES.items()}
    return {
        "description": "Legacy _emit_magnitude parity anchor (ADR-067 A-S2). Do not edit by hand.",
        "shape": [1, _C, _H, _W],
        "inputs": {
            "reg": REG.tolist(),
            "prob": PROB.tolist(),
            "theta": THETA.tolist(),
            "sigma": SIGMA.tolist(),
        },
        "outputs": outputs,
    }


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(build(), indent=2))
    print(f"wrote {GOLDEN} ({len(CASES)} distributions)")
