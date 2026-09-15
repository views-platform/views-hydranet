"""Launch precondition for the BPTT-SA screen: prove the knob can act, on the arm's own config.

C-324's mitigation made mechanical. #308's first run trained two arms for 276 minutes to
byte-identical weights because the treatment was inert on the family + sampled-feedback path that
C-259 forces production to use. This runs the same contrast in two seconds and refuses to let the
launcher start if the knob cannot move a gradient.

It reads the ARM'S OWN config rather than a fixture, because the whole failure class is
"verified on a path production never takes".
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

_HYD = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HYD))
sys.path.insert(0, str(_HYD / "scripts"))

from potency_check import PotencyError, assert_potent  # noqa: E402
from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _family_composed_mean_log1p,
    _family_feedback_log1p,
)

_MODELS = _HYD.parent / "views-models" / "models"


def _cfg(model: str) -> dict:
    p = _MODELS / model / "configs" / "config_hyperparameters.py"
    ns: dict = {}
    text = p.read_text()
    ast.parse(text)
    exec(compile(text, str(p), "exec"), ns)  # noqa: S102 - repo-local config
    return ns["get_hp_config"]()


def check(model: str) -> dict:
    """Run the potency contrast using this arm's actual family, feedback mode and composition."""
    c = _cfg(model)
    fam = resolve_family(c["output_distribution"])
    feedback, comp = c["ss_feedback"], c["forecast_composition"]
    n_reg, npar = len(c["regression_targets"]), fam.n_params

    torch.manual_seed(0)
    raw = torch.randn(1, n_reg * npar, 8, 8)
    act = torch.cat(
        [
            fam.activate(raw[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            for j in range(n_reg)
        ],
        dim=1,
    ).requires_grad_(True)
    gate = torch.rand(1, n_reg, 8, 8)

    def grad_through(backprop: bool) -> float:
        torch.manual_seed(3)
        fed = _family_feedback_log1p(act, fam, feedback, gate, comp, c.get("gate_threshold"))
        if backprop:
            if feedback == "sample":
                s = _family_composed_mean_log1p(act, fam, gate, comp, c.get("gate_threshold"))
                fed = s + (fed - s).detach()
        else:
            fed = fed.detach()
        if not fed.requires_grad:
            return 0.0
        g = torch.autograd.grad(fed.sum(), act, retain_graph=True, allow_unused=True)[0]
        return 0.0 if g is None else float(g.abs().sum())

    return assert_potent(
        grad_through,
        off=False,
        on=True,
        name=f"{model}: ss_backprop_through_feedback "
        f"(family={c['output_distribution']}, ss_feedback={feedback!r}, "
        f"eps={c['ss_epsilon_max']}, composition={comp!r})",
    )


def main() -> int:
    failed = False
    for model in sys.argv[1:] or ["ssattached_fortytwo"]:
        try:
            r = check(model)
            print(f"POTENT   {model}: gradient off={r['off']:.4g} on={r['on']:.4g}")
        except PotencyError as exc:
            print(f"INERT    {exc}")
            failed = True
    if failed:
        print(
            "\nREFUSING TO LAUNCH — the treatment cannot act on this configuration, so any "
            "result would be a fact about the harness rather than the hypothesis (C-324)."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
