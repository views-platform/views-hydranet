#!/usr/bin/env python3
"""step1_decompose.py — split the step-1 occurrence shortfall into gate vs body-zeros.

See `05_analysis_plan.md` (LOCKED). One forward pass at the PRODUCTION origin; nothing is trained.

`soft_gate` emits `body_sample * bernoulli(gate)`, and the NB body has its own mass at zero, so
expected occurrence is `E[gate * P(NB>0)]`, not `E[gate]`. This measures both factors:

    G = mean(gate)             / truth   how much of true occurrence the GATE alone accounts for
    C = mean(gate * p_nonzero)  / truth      what the composition actually emits

Dropout is OFF, as in the state-range dossier: the quantity is a deterministic property of the
composition, and locked dropout would add mask noise to both factors for nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_hydranet.distributions import resolve_family
from views_hydranet.manager.hydranet_manager import HydranetManager

_CAPTURED: dict = {}


class _GotContext(RuntimeError):
    pass


class ContextManager(HydranetManager):
    def _setup_evaluation(self, *args, **kwargs):
        _CAPTURED["ctx"] = super()._setup_evaluation(*args, **kwargs)
        raise _GotContext("captured")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-label", required=True)
    ap.add_argument("--origin", type=int, required=True, help="F1: production origin")
    ap.add_argument("--period", type=int, required=True, help="F1: measured sample period")
    ap.add_argument("--k-draws", type=int, default=200, help="AMENDMENT 1: draws to average F2 over")
    args, _ = ap.parse_known_args()

    mgr = ContextManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    run_args = ForecastingModelArgs.from_namespace(
        ForecastingModelArgs._create_parser().parse_args(
            ["--run_type", "calibration", "--evaluate",
             "--saved", "--artifact_name", args.artifact]
        )
    )
    try:
        mgr.execute_single_run(run_args)
    except Exception:
        if "ctx" not in _CAPTURED:
            raise
    ctx = _CAPTURED["ctx"]
    orch = ctx.orchestrator
    model, handler, config, device = orch.model, ctx.handler, orch.config, orch.device
    model.eval()

    full = handler.to_pytorch(device, include_identities=False).to(device)
    _, seq_len, _, H, W = full.shape
    names = [n for n in handler.channel_map if n in handler.feature_cols]
    idxs = [names.index(f) for f in config.get("features", [])] + [
        names.index(s) for s in config.get("static_channels", [])
    ]
    origin = args.origin
    # F1: the origin is an ARGUMENT, never `seq_len - 1`. That fallback is C-308's 2nd occurrence.
    if origin + args.period - origin != args.period or origin >= seq_len - 1:
        raise SystemExit(f"origin {origin} invalid against seq_len {seq_len}")

    # Resolve the family the way `HydraNetInference.__init__` does — off the MODEL's declared
    # `output_distribution`, not off the config, so this cannot silently disagree with what the
    # artifact actually emits (`hydranet_inference.py:218-222`).
    fam = resolve_family(getattr(model, "output_distribution", "standard"))
    if fam is None:
        raise SystemExit("no distribution family on this artifact — the decomposition is undefined")
    npar = fam.n_params
    target = config["regression_targets"][0]
    t_ch = names.index(target)

    h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).float().to(device)
    with torch.no_grad():
        for t in range(origin):  # history digestion, stopping BEFORE the seed step
            h = model(full[:, t, idxs, :, :], h).h_next
        out = model(full[:, origin, idxs, :, :], h)  # the seed step: input is entirely REAL data
        gate = torch.sigmoid(out.cls)[:, 0]                        # [B,H,W] P(y>0), target 0
        params = out.reg[:, 0:npar].permute(0, 2, 3, 1)   # [B,H,W,npar] activated (mu, theta)
        p_nz = fam.prob_positive(params)                           # [B,H,W] P(NB draw > 0)

    truth = float((full[:, origin + 1, t_ch] > 0).float().mean())
    g_mean, c_mean = float(gate.mean()), float((gate * p_nz).mean())

    # F2: the analytic decomposition must describe the SAMPLER that builds the fed-back field.
    # Draw it the way `_sample_feedback` does — a body draw composed with a Bernoulli gate.
    # AMENDMENT 1: average over k draws. ONE realisation has a Monte-Carlo standard error of
    # ~15% here (~45 active cells in 32,400), which is 7x F2's 2% tolerance — the original check
    # was measuring the sampler's variance, not the decomposition. k=200 brings it to ~1.1%.
    gen = torch.Generator(device="cpu").manual_seed(20260823)
    pc, gc = params.detach().cpu(), gate.detach().cpu()
    k = args.k_draws
    per_draw = []
    for _ in range(k):
        d = fam.sample(pc, 1, gen).squeeze(-1)
        m = torch.bernoulli(gc, generator=gen)
        per_draw.append(float(((d * m) > 0).float().mean()))
    sampled = sum(per_draw) / k
    sampled_sd = (sum((x - sampled) ** 2 for x in per_draw) / max(k - 1, 1)) ** 0.5

    G, C = g_mean / truth, c_mean / truth
    f2_rel = abs(c_mean - sampled) / (sampled or 1.0)
    res = {
        "seed": args.seed_label,
        "origin": origin,
        "period": args.period,
        "seq_len": seq_len,
        "truth_active_fraction": truth,
        "mean_gate": g_mean,
        "mean_gate_x_pnonzero": c_mean,
        "mean_p_nonzero": float(p_nz.mean()),
        "sampled_emitted_active_fraction": sampled,
        "sampled_per_draw_sd": sampled_sd,
        "sampled_se_of_mean": sampled_sd / (args.k_draws ** 0.5),
        "k_draws": args.k_draws,
        "G_gate_over_truth": G,
        "C_composed_over_truth": C,
        "F2_analytic_vs_sampled_rel": f2_rel,
        "F2": "PASS" if f2_rel <= 0.02 else "FAIL",
        "F4": "PASS" if bool(torch.isfinite(gate).all() & torch.isfinite(p_nz).all()) else "FAIL",
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, f"step1_{args.seed_label}.json").write_text(json.dumps(res, indent=1))
    print(f"{args.seed_label}: truth={truth:.5f} G={G:.3f} C={C:.3f} "
          f"p_nz={res['mean_p_nonzero']:.3f} sampled={sampled:.5f} F2={res['F2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
