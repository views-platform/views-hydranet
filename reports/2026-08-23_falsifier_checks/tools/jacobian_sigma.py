#!/usr/bin/env python3
"""jacobian_sigma.py — Check D (#294): the TRUE recurrent Jacobian's spectral norm.

Threshold registered in GitHub issue #294 (2026-08-22); estimator in `05_analysis_plan.md`
AMENDMENT 1 (`e631f74`), which states that the analytic bound licenses nothing above 1 and that a
verdict requires *"power iteration on the true Jacobian at states from a real rollout"*. This is
that measurement.

sigma_max = sup over rollout states of || d h_next / d h ||_2, computed matrix-free by power
iteration on J^T J: one jvp and one vjp per step. `h_next` is assembled purely from the LSTM block
(`HydraBNrecurrentUnet_06_LSTM4.py:555`), so this is exactly the recurrent map's Jacobian and the
U-Net decoder does not enter it.

States come from `capture_states.py` — a forward hook on a live free-running rollout, with NO
diagnostic flag set, so they are production-path states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HN))


def power_iterate(f, h: torch.Tensor, iters: int, seed: int) -> tuple[float, float]:
    """Largest singular value of J = df/dh at h, and the relative change over the last 10 steps."""
    if iters < 11:
        # `prev` is sampled at iters-11; below that it would stay 0.0, the drift would compute as
        # 0.0, and the registered convergence falsifier would pass VACUOUSLY — closing #294 on an
        # iteration that never ran. Refuse rather than return a sentinel that reads as a
        # measurement (the "unmeasured input absorbed as a value" class the register tracks).
        raise ValueError(f"iters must be >= 11 to measure drift; got {iters}")
    g = torch.Generator(device=h.device).manual_seed(seed)
    v = torch.randn(h.shape, generator=g, device=h.device, dtype=h.dtype)
    v /= v.norm()
    sigma, prev = 0.0, 0.0
    for i in range(iters):
        _, jv = torch.func.jvp(f, (h,), (v,))
        sig = jv.norm()
        _, vjp_fn = torch.func.vjp(f, h)
        (jtjv,) = vjp_fn(jv)
        n = jtjv.norm()
        if n == 0:
            # A zero J^T J v means the iterate fell in the null space — the iteration has not
            # measured sigma, it has died. Returning (0.0, 0.0) here would read as "sigma < 1,
            # converged" and close the issue.
            raise ValueError(
                f"power iteration collapsed to zero at step {i} — J^T J v vanished, so no "
                "spectral norm was measured"
            )
        v = jtjv / n
        if i == iters - 11:
            prev = float(sig)
        sigma = float(sig)
    if not sigma or not prev:
        raise ValueError("power iteration produced a zero singular value — nothing was measured")
    return sigma, abs(sigma - prev) / sigma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Build the module through the repo's own factory rather than reconstructing it here — the
    # artifact is a bare state_dict, and inferring constructor args from tensor shapes would be a
    # second, drifting definition of the architecture.
    import importlib.util

    from views_hydranet.utils.utils import choose_model

    cfg_path = Path(a.model_dir) / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location("cfg_hp", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = mod.get_hp_config()

    model = choose_model(config, torch.device("cpu"))
    art_path = Path(a.model_dir) / "artifacts" / a.artifact
    sd = torch.load(art_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch — missing {list(missing)[:3]}, "
            f"unexpected {list(unexpected)[:3]}. "
            "Refusing rather than probing a partially-loaded model."
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    files = sorted(Path(a.states).glob("state_*.pt"))
    if not files:
        raise SystemExit(f"no captured states in {a.states}")

    results = []
    for f in files:
        st = torch.load(f, map_location="cpu", weights_only=False)
        # Provenance: measuring a real Jacobian at states from a DIFFERENT model is silent and
        # plausible. This model dir holds several artifacts, which is why `freeze_arm_entry.py`
        # makes --artifact required for the same reason.
        src = st.get("artifact")
        if src is not None and Path(src).name != art_path.name:
            raise SystemExit(
                f"{f.name} was captured from {Path(src).name} but this run loads "
                f"{art_path.name} — refusing to measure one model's Jacobian at another's states"
            )
        x, h = st["x"], st["h"]

        def step(hh, _x=x):
            return model(_x, hh).h_next

        sigma, drift = power_iterate(step, h, a.iters, seed=0)
        results.append(
            {
                "state": f.name,
                "sigma": sigma,
                "drift_last10": drift,
                "h_absmax": float(h.abs().max()),
            }
        )
        print(
            f"  {f.name}  sigma = {sigma:.4f}   (drift over last 10 iters {drift:.2%}, "
            f"max|h| = {float(h.abs().max()):.3f})"
        )

    sigmas = [r["sigma"] for r in results]
    sup = max(sigmas)
    worst_drift = max(r["drift_last10"] for r in results)
    converged = worst_drift <= 0.01

    # the registered threshold (issue #294)
    if not converged:
        verdict = "VOID — power iteration did not converge"
    elif sup < 1.0:
        verdict = "CLOSE #294 — sigma_max < 1, so GTF's alpha = 1 - 1/sigma_max is undefined here"
    elif 1.05 <= sup <= 1.20:
        verdict = "STRIKING — sigma_max in [1.05, 1.20]; the derived alpha lands on M41's w~0.1"
    else:
        verdict = "CORRESPONDENCE SUPERFICIAL — sigma_max outside [1.05, 1.20] and >= 1"

    alpha = 1.0 - 1.0 / sup if sup >= 1.0 else None
    print()
    print(f"n states = {len(sigmas)}   sigma range [{min(sigmas):.4f}, {sup:.4f}]")
    print(f"sigma_max (sup over states) = {sup:.4f}")
    if alpha is not None:
        print(f"GTF alpha = 1 - 1/sigma_max = {alpha:.4f}   (M41 measured w ~ 0.10)")
    else:
        print("GTF alpha undefined: the formula requires sigma_max >= 1")
    print(f"\nVERDICT: {verdict}")

    Path(a.out).write_text(
        json.dumps(
            {
                "per_state": results,
                "sigma_max": sup,
                "converged": converged,
                "gtf_alpha": alpha,
                "m41_w": 0.10,
                "verdict": verdict,
                # Self-describing. A first run at iters=60 on the WRONG rollout phase was VOID,
                # and a later one at 250 on the autoregressive phase was accepted; without these
                # fields the two artifacts are indistinguishable on disk (C-308). `states_dir` is
                # what identifies the PHASE the Jacobian was measured in.
                "iters": a.iters,
                "n_states": len(sigmas),
                "artifact": str(art_path),
                "states_dir": str(a.states),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
