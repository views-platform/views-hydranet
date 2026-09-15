#!/usr/bin/env python3
"""gradient_probe.py — Check C (#288): is the fed-back count draw differentiable?

Criterion is in `05_analysis_plan.md` (`6ec3c3c`, 22:14:04): this is a **fact about code**, not a
measurement, so there is no threshold. It EXECUTES the decisive operations rather than reasoning
about them, because the answer turns on whether an op has a registered backward and what that
backward returns — which is a property of the installed torch, not of the source.

The finding that makes this worth a script rather than a paragraph: `torch.poisson` and
`torch.bernoulli` ARE graph-connected and their backward returns **exactly zero**. So removing the
two `.detach()` calls at `training_engine.py:363,365` would not crash, would not warn, would pay
the full memory cost of a retained 36-step feedback graph, and would return a numerically
identical model. Silent no-op — the worst failure mode an experiment can have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HN))


def probe_op(name, fn, arg):
    x = arg.clone().detach().requires_grad_(True)
    out = fn(x)
    connected = out.requires_grad
    gname = type(out.grad_fn).__name__ if out.grad_fn else None
    grad = None
    if connected:
        out.sum().backward()
        grad = x.grad.abs().sum().item()
    return {"op": name, "graph_connected": connected, "grad_fn": gname, "grad_abs_sum": grad}


def main() -> int:
    results = {"torch": torch.__version__, "ops": [], "family_path": {}}

    results["ops"].append(probe_op("torch.poisson", torch.poisson, torch.tensor([2.0, 5.0, 9.0])))
    results["ops"].append(probe_op("torch.bernoulli", torch.bernoulli, torch.tensor([0.3, 0.7])))

    # the reparameterised Gamma one line ABOVE the sever point (nb_core.py:163)
    from views_hydranet.distributions.nb_core import _standard_gamma

    conc = torch.tensor([[2.0, 3.0, 7.0]], requires_grad=True)
    g = _standard_gamma(conc.contiguous(), None)
    g.sum().backward()
    results["family_path"]["_standard_gamma"] = {
        "graph_connected": bool(g.requires_grad),
        "grad_fn": type(g.grad_fn).__name__ if g.grad_fn else None,
        "grad_abs_sum": conc.grad.abs().sum().item(),
    }

    print(f"torch {results['torch']}")
    print()
    print("The two operations on the feedback path:")
    for r in results["ops"]:
        print(
            f"  {r['op']:<18} connected={str(r['graph_connected']):<5} "
            f"grad_fn={str(r['grad_fn']):<20} sum|grad| = {r['grad_abs_sum']}"
        )
    r = results["family_path"]["_standard_gamma"]
    print()
    print("One line ABOVE the sever point (nb_core.py:163, the Marsaglia-Tsang sampler):")
    print(
        f"  {'_standard_gamma':<18} connected={str(r['graph_connected']):<5} "
        f"grad_fn={str(r['grad_fn']):<20} sum|grad| = {r['grad_abs_sum']:.4f}"
    )
    print()

    severed = all(x["grad_abs_sum"] == 0.0 for x in results["ops"])
    silent = severed and all(x["graph_connected"] for x in results["ops"])
    reparam = r["grad_abs_sum"] > 0
    results["severed"] = severed
    results["fails_silently"] = silent
    results["reparameterised_rate_available"] = reparam

    print("VERDICT")
    print(f"  gradient severed by the draw ............ {severed}")
    print(f"  ...and it fails SILENTLY (no error) ..... {silent}")
    print(f"  differentiable rate exists one line up .. {reparam}")
    print()
    if silent:
        print("  => un-detaching alone is a SILENT NO-OP for ss_feedback='sample'.")
        print("     Any future arm MUST carry a graph assertion; test_feedback_parity.py")
        print("     pins values only and would pass a zero-gradient no-op unchanged.")
    if reparam:
        print("  => log1p(lam) instead of log1p(poisson(lam)) is stochastic AND differentiable.")

    out = _HN / "reports/2026-08-23_falsifier_checks/results/gradient_probe.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
