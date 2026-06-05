#!/usr/bin/env python
"""Axis-0 diagnostic for C-113 — input->output gain of the autoregressive map.

Corrected per the freeze_h ablation (reports/results_freezeh_ablation.md): the
divergence rides the prediction->input feedback loop, not the recurrent state.
So the quantity to measure is the operator gain of ONE autoregressive step's
prediction w.r.t. its fed-back input:  J = d(reg)/d(x),  ||J||_2 (top singular
value). In the rollout, x_{t+1} = model(x_t, h).reg directly (no remapping), so
||J||_2 > 1 at the visited operating points => local amplification => runaway.

Two parts, both retrain-free (loads saved artifacts):
  A) power-iteration ||J||_2 across a sweep of in-range inputs (h fixed), and
  B) a free-running rollout from synthetic seeds, recording ||prediction||.

Compare pink_pirate (bounded in eval) vs violet_visitor (exploded). Pre-registered
expectation: violet's gain > 1 (and/or grows with input magnitude); pink's < 1.
HONESTY: operating points are synthetic (no data pipeline). Part A is a weight-level
property (broad input sweep); Part B corroborates. Real-data confirmation would need
the full input pipeline.
"""

import argparse
import json
from pathlib import Path

import torch

from views_hydranet.utils.rollout_diagnostics import (
    DATA_LOG_MAX,
    free_running_attractor,
    is_out_of_range,
)
from views_hydranet.utils.utils import choose_model


def load_model(artifact: Path, device):
    cfg = json.loads(artifact.with_suffix(".pt.config.json").read_text())
    model = choose_model(cfg, device)
    state = torch.load(artifact, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()  # dropout off (identity), BN uses running stats -> deterministic map
    return model, cfg


@torch.no_grad()
def _zero_state(model, H, W, device):
    return model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)


def op_norm(model, x, h, iters=25):
    """Top singular value of J = d(reg)/d(x) at operating point x (h fixed)."""

    def f(z):
        return model(z, h).reg

    v = torch.randn_like(x)
    v = v / (v.norm() + 1e-12)
    sigma = torch.tensor(0.0)
    for _ in range(iters):
        _, jv = torch.autograd.functional.jvp(f, x, v)  # J v
        _, jtjv = torch.autograd.functional.vjp(f, x, jv)  # J^T (J v)
        n = jtjv.norm()
        if n < 1e-20:
            return 0.0  # output locally constant (all-ReLU-clamped) -> zero gain
        v = jtjv / n
        sigma = n.sqrt()  # sqrt of top eigenvalue of J^T J = top singular value of J
    return float(sigma)


def part_A(model, H, W, device, label):
    print(f"\n=== Part A — input->output operator gain ||dreg/dx||_2  [{label}] ===")
    print(f"{'input (>=0, log1p-space)':<34}{'||J||_2':>12}   amplifies?")
    h = _zero_state(model, H, W, device)
    base_ch = model_in_ch
    # constant-fill sweep across the in-range magnitude (data log1p ~ 0..12)
    results = {}
    for c in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]:
        x = torch.full((1, base_ch, H, W), float(c), device=device)
        s = op_norm(model, x, h)
        results[f"const={c}"] = s
        print(f"{('constant fill = %.2f' % c):<34}{s:>12.4f}   {'YES >1' if s > 1 else 'no'}")
    # random in-range inputs (uniform[0,c]); report mean+max over several draws
    g = torch.Generator(device="cpu").manual_seed(0)
    for c in [1.0, 4.0, 12.0]:
        vals = []
        for _ in range(5):
            x = (torch.rand((1, base_ch, H, W), generator=g) * c).to(device)
            vals.append(op_norm(model, x, h))
        mx = max(vals)
        results[f"rand<={c}"] = mx
        print(
            f"{('random U[0,%.0f]  max/mean' % c):<34}{mx:>12.4f}/{sum(vals) / len(vals):.3f}"
            f"   {'YES >1' if mx > 1 else 'no'}"
        )
    return results


@torch.no_grad()
def part_B(model, H, W, device, label, freeze_h="none", steps=48):
    print(f"\n=== Part B — free-running rollout (freeze_h='{freeze_h}')  [{label}] ===")
    print(
        "    log-space level the prediction settles at; healthy = within data "
        f"range (<= log {DATA_LOG_MAX}); expm1(level) = raw-count equivalent"
    )
    print(
        f"{'seed U[0,s]':<13}{'step1':>8}{'step12':>9}{'step24':>9}{'step48':>9}"
        f"{'expm1(final)':>15}   verdict"
    )
    g = torch.Generator(device="cpu").manual_seed(1)
    update_state = freeze_h != "all"  # 'none' evolves the recurrent state; 'all' freezes it
    for s in [0.5, 1.0, 2.0]:
        h = _zero_state(model, H, W, device)
        x = (torch.rand((1, model_in_ch, H, W), generator=g) * s).to(device)
        final, traj = free_running_attractor(model, x, h, steps=steps, update_state=update_state)
        raw = torch.expm1(torch.tensor(final)).item()
        verdict = "PATHOLOGICAL (out-of-range)" if is_out_of_range(final) else "healthy (in-range)"
        print(
            f"{('U[0,%.1f]' % s):<13}{traj[0]:>8.2f}{traj[11]:>9.2f}{traj[23]:>9.2f}"
            f"{final:>9.2f}{raw:>15.2e}   {verdict}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--hw", type=int, default=32)  # training window_dim
    args = ap.parse_args()

    device = torch.device("cpu")  # small map; avoid GPU contention
    model, cfg = load_model(args.artifact, device)
    model_in_ch = cfg["input_channels"]
    label = args.label or args.artifact.parent.parent.name
    print(
        f"\n################ {label}  ({args.artifact.name}, {args.hw}x{args.hw}, "
        f"dropout={cfg['dropout_rate']}) ################"
    )
    part_A(model, args.hw, args.hw, device, label)
    part_B(model, args.hw, args.hw, device, label, freeze_h="none")
    part_B(model, args.hw, args.hw, device, label, freeze_h="all")
    print()
