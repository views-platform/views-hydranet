#!/usr/bin/env python3
"""capture_regimes.py — recurrent-state distributions for R1 / R2 / F3 (state-range dossier).

Answers: does the state the model must hold at the rollout origin lie inside the range of states it
produces on training-DISTRIBUTION input? See `05_analysis_plan.md` (LOCKED, + AMENDMENT 1, 2).

Three regimes, all **history digestion only** — no autoregression, so nothing here depends on the
emit/composition path:

  R2  full grid, `origin + 1` real steps      -> the state free-running inherits (`state_anchor`)
  R1  32x32 patches from the PRODUCTION anchor strategy at three curriculum ratios, same months
  F3  32x32 patches at FIXED locations, compared against the same cells read out of R2  [HARD STOP]

R3 (the free-running tail) is NOT computed here: it needs the production autoregressive path, so
it is measured by the already-proven `2026-08-23_falsifier_checks/tools/capture_states.py`.

**Dropout is OFF** (AMENDMENT 2): `hydranet_inference` calls `model.eval()` *and*
`set_locked_dropout(True)`, making the production rollout MC-dropout stochastic. Leaving locked
dropout untouched keeps every regime deterministic so the one variable is the INPUT, not the mask.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.volume_sampler import SAMPLING_STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


#: The pipeline does not necessarily evaluate on the same manager instance the caller constructed,
#: so the captured context is stashed module-side rather than on `self`. Verified the hard way: an
#: instance attribute set inside `_setup_evaluation` read back as None on the caller's object.
_CAPTURED: dict = {}


class _GotContext(RuntimeError):
    """Raised to stop before the production rollout — we want the context, not the forecast."""


class ContextManager(HydranetManager):
    """`HydranetManager` that grabs the evaluation context and stops.

    Same seam `capture_states.py` and `freeze_arm_entry.py` use. No diagnostic flag is set and no
    forward hook is attached: this tool drives its own forward passes so the regime each state
    comes from is decided by THIS file, not inferred from a call index. That is the structural fix
    for C-308 (a probe that captured the wrong phase while every downstream guard passed).
    """

    def _setup_evaluation(self, *args, **kwargs):
        _CAPTURED["ctx"] = super()._setup_evaluation(*args, **kwargs)
        raise _GotContext("evaluation context captured")


def _state_halves(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the recurrent state into its hidden and cell halves.

    `init_hTtime` returns `[1, hidden_channels, H, W]` packing BOTH halves as
    `hs_1..hs_4 | hl_1..hl_4` — first half short-term (hidden), second half long-term (cell), the
    same split `blend_recurrent_state` uses (`hydranet_inference.py:124-127`). M39 established the
    cell half carries the ENTIRE freeze effect, so the halves are never pooled: a difference
    confined to one of them would be invisible in a pooled statistic.

    The divisibility guard is `% 8`, not `% 2`, deliberately matching production's `_STATE_GROUPS`
    check. The state is 4 short-term + 4 long-term groups; an even-but-not-8 channel count would
    split without error while silently mis-assigning memory types, and this probe's whole output is
    keyed on which half is which.
    """
    c = h.shape[1]
    if c % 8:
        raise ValueError(
            f"state channel count {c} is not divisible by 8 (4 short-term + 4 long-term groups); "
            "splitting it would silently mis-assign hidden vs cell."
        )
    return {"hidden": h[:, : c // 2], "cell": h[:, c // 2 :]}


def _digest(model, tensor, idxs, n_steps, device):
    """Run `n_steps` of history digestion and return the resulting state.

    Deliberately NOT hooked: the caller states how many real steps to absorb, so the returned
    state's provenance is a function argument rather than something reconstructed from ordering.
    """
    _, _, _, H, W = tensor.shape
    h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).float().to(device)
    if float(h.abs().max()) != 0.0:
        raise RuntimeError("F1 VIOLATED: init_hTtime did not return a zero state")
    with torch.no_grad():
        for t in range(n_steps):
            h = model(tensor[:, t, idxs, :, :], h).h_next
    if not torch.isfinite(h).all():
        raise RuntimeError("F5 VIOLATED: non-finite value in captured state")
    return h


def _stats(h: torch.Tensor) -> dict:
    """Per-channel and pooled summary of a state tensor.

    Quantiles, not just `max|h|`: C-308 was a single plausible-looking summary number describing
    the wrong thing. Per-channel arrays are kept so §4's interval is built per channel.
    """
    out = {}
    for name, half in _state_halves(h).items():
        a = half[0].float().cpu()  # [C, H, W]
        flat = a.reshape(a.shape[0], -1)
        out[name] = {
            "n_channels": int(a.shape[0]),
            "n_cells": int(flat.shape[1]),
            "abs_max": float(a.abs().max()),
            "pooled_mean": float(a.mean()),
            "pooled_sd": float(a.std()),
            "per_channel_q01": torch.quantile(flat, 0.01, dim=1).tolist(),
            "per_channel_q99": torch.quantile(flat, 0.99, dim=1).tolist(),
            "per_channel_mean": flat.mean(dim=1).tolist(),
            "per_channel_sd": flat.std(dim=1).tolist(),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-label", required=True, help="e.g. fortytwo — tags the output")
    ap.add_argument("--n-patches", type=int, default=40, help="patches per curriculum ratio")
    ap.add_argument("--n-f3", type=int, default=8, help="fixed locations for the F3 control")
    ap.add_argument("--np-seed", type=int, default=4711, help="patch-anchor RNG (this tool's own)")
    ap.add_argument(
        "--border",
        type=int,
        default=13,
        help="F3 border width b, in cells. Read off the architecture (3x3 convs, two 2x2 pools, "
        "bottleneck, two transposed convs => receptive-field radius ~12-13), NOT "
        "tuned. See AMENDMENT 3 for the rule, AMENDMENT 4 for why it may fail anyway.",
    )
    ap.add_argument(
        "--origin",
        type=int,
        default=None,
        help="history index the state is built to; production uses ctx.origins, not seq_len-1",
    )
    ap.add_argument(
        "--f3-only",
        action="store_true",
        help="run R2 and the F3 control, then stop — F3 gates everything downstream",
    )
    args, _ = ap.parse_known_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    mgr = ContextManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    run_args = ForecastingModelArgs.from_namespace(
        ForecastingModelArgs._create_parser().parse_args(
            ["--run_type", "calibration", "--evaluate",
             "--saved", "--artifact_name", args.artifact]
        )
    )
    # The pipeline wraps whatever `_setup_evaluation` raises in a `ModelEvaluationException`, so
    # `_GotContext` never surfaces as itself. Catching the wrapper is therefore required — but it
    # is only ACCEPTED when the context was actually captured. Any other failure leaves it
    # unset and is re-raised with its traceback intact, so a real breakage cannot masquerade.
    try:
        mgr.execute_single_run(run_args)
    except Exception:
        if "ctx" not in _CAPTURED:
            raise
        logger.info("stopped after capturing the evaluation context (expected)")
    if "ctx" not in _CAPTURED:
        raise SystemExit("failed to capture the evaluation context")

    ctx = _CAPTURED["ctx"]
    orch = ctx.orchestrator
    model, handler, config = orch.model, ctx.handler, orch.config
    device = orch.device
    model.eval()  # locked dropout deliberately NOT enabled — AMENDMENT 2

    # Rebuild the model input exactly as `HydraNetInference.predict` does
    # (hydranet_inference:805-815, 1147-1171). Reusing its construction rather than
    # re-deriving it keeps this probe on the production input contract.
    full_tensor = handler.to_pytorch(device, include_identities=False).to(device)
    _, seq_len, _, H, W = full_tensor.shape
    feature_names = [n for n in handler.channel_map if n in handler.feature_cols]
    idxs = [feature_names.index(f) for f in config.get("features", [])] + [
        feature_names.index(s) for s in config.get("static_channels", [])
    ]
    # `predict()` falls back to `seq_len - 1` only when no origin is passed; PRODUCTION scoring
    # rolls over `ctx.origins`, and the real free-running phase begins at one of those. Measured on
    # this vehicle: the sample period is 371 with time_steps=36, so origin=335, not 383.
    # Using the fallback would have measured a state built from 48 extra months of history and
    # called it "the state free-running inherits".
    origin = args.origin if args.origin is not None else seq_len - 1
    dim = config["window_dim"]

    manifest = {
        "seed_label": args.seed_label,
        "artifact": args.artifact,
        "grid": [H, W],
        "seq_len": seq_len,
        "origin": origin,
        "origin_default_would_be": seq_len - 1,
        "production_origins": list(ctx.origins),
        "window_dim": dim,
        "n_model_in_channels": len(idxs),
        "dropout": "off (AMENDMENT 2)",
        "curriculum_config": {
            k: config[k]
            for k in ("max_ratio", "min_ratio", "slope_ratio", "roof_ratio", "min_events")
        },
        "sampling_strategy": config["sampling_strategy"],
    }

    # ---- R2: the deployment state, built from `origin + 1` real observations -------------------
    logger.info("R2: full grid %dx%d, %d real steps", H, W, origin + 1)
    h_r2 = _digest(model, full_tensor, idxs, origin + 1, device)
    results = {"manifest": manifest, "R2": _stats(h_r2)}
    # §4 needs RAW per-cell values, not summaries: the interval is a 1/99% quantile of the POOLED
    # R1 distribution and `f` counts R2 cells outside it. Quantiles-of-quantiles would be a
    # different statistic. Saved as tensors so analysis stays separable from capture.
    _r2h = _state_halves(h_r2)
    torch.save(
        {"cell": _r2h["cell"].cpu(), "hidden": _r2h["hidden"].cpu()},
        out / f"r2_state_{args.seed_label}.pt",
    )

    # ---- F3 (HARD STOP): does patch-vs-full-grid alone move the state? -------------------------
    # Holds LOCATION fixed and varies only the input extent. If this fires, R1-vs-R2 would be
    # measuring U-Net image size rather than data distribution, and §4 is not consulted at all.
    rng = np.random.default_rng(args.np_seed)
    b = args.border
    f3 = []
    for _ in range(args.n_f3):
        r0 = int(rng.integers(0, H - dim))
        c0 = int(rng.integers(0, W - dim))
        crop = full_tensor[:, :, :, r0 : r0 + dim, c0 : c0 + dim]
        h_patch = _digest(model, crop, idxs, origin + 1, device)
        for half, sub in _state_halves(h_patch).items():
            ref = _state_halves(h_r2)[half][:, :, r0 : r0 + dim, c0 : c0 + dim]
            diff = (sub - ref).abs()
            # AMENDMENT 3: judge the INTERIOR, because a patch and the full grid MUST disagree at
            # the patch edge (border cells have no neighbours the full grid has). Pooling the two
            # confounds an unavoidable boundary artifact with the geometry effect F3 tests for.
            # AMENDMENT 4 predicts the interior fails anyway: the receptive field compounds ~12
            # cells PER TIMESTEP over 384 steps, so no part of a 32x32 patch stays boundary-free.
            inner = (slice(None), slice(None), slice(b, dim - b), slice(b, dim - b))
            den_all = float(ref.abs().mean()) or 1.0
            den_in = float(ref[inner].abs().mean()) or 1.0
            f3.append(
                {
                    "loc": [r0, c0],
                    "half": half,
                    "rel_abs_diff_pooled": float(diff.mean()) / den_all,
                    "rel_abs_diff_interior": float(diff[inner].mean()) / den_in,
                    "interior_side": dim - 2 * b,
                    "patch_abs_mean": float(sub.abs().mean()),
                    "fullgrid_same_cells_abs_mean": float(ref.abs().mean()),
                }
            )
    worst = max(x["rel_abs_diff_interior"] for x in f3)
    verdict = "PASS" if worst <= 0.02 else "HARD STOP"
    results["F3"] = {
        "border": b,
        "worst_interior_rel_abs_diff": worst,
        "verdict": verdict,
        "patches": f3,
    }
    logger.info("F3: worst interior rel_abs_diff = %.4f -> %s", worst, verdict)
    if verdict != "PASS":
        # HARD STOP (AMENDMENT 3). R1 is not computed and no `f` is produced: a confounded headline
        # with a caveat attached is exactly how C-308 reached the ledger.
        results["R1"] = None
        results["hard_stop"] = (
            f"F3 interior rel_abs_diff {worst:.4f} > 0.02: patch-vs-full-grid geometry moves the "
            "state where the data is identical, so R1-vs-R2 cannot be attributed to selection. "
            "No verdict rendered; see AMENDMENT 4 for the successor design."
        )
        path = out / f"regimes_{args.seed_label}.json"
        path.write_text(json.dumps(results, indent=1))
        print(f"F3 HARD STOP — wrote {path}")
        return 0
    if args.f3_only:
        path = out / f"regimes_{args.seed_label}.json"
        path.write_text(json.dumps(results, indent=1))
        print(f"F3 PASS (f3-only) — wrote {path}")
        return 0

    # ---- R1: activity-selected patches, SAME months as R2 (AMENDMENT 1) ------------------------
    # Anchors come from the production strategy registry and the production `min_events`, so the
    # SELECTION RULE is the real one; only the months are shared with R2 rather than taken from the
    # training partition (AMENDMENT 1 — removes the partition confound).
    target = config["regression_targets"][0]
    t_idx = feature_names.index(target)
    activity = np.count_nonzero(full_tensor[0, :, t_idx].cpu().numpy(), axis=0)  # [H, W]
    act_max = float(activity.max())
    select = SAMPLING_STRATEGY_REGISTRY[config["sampling_strategy"]]["fn"]

    r1 = {}
    for ratio in (0.665, 0.35, 0.05):
        thr = int(ratio * act_max) or 1
        per_patch, densities, raw = [], [], []
        for _ in range(args.n_patches):
            r_anc, c_anc = select(activity, thr, config["min_events"], rng, config)
            r0 = int(np.clip(r_anc - rng.integers(0, dim), 0, H - dim))
            c0 = int(np.clip(c_anc - rng.integers(0, dim), 0, W - dim))
            crop = full_tensor[:, :, :, r0 : r0 + dim, c0 : c0 + dim]
            h_p = _digest(model, crop, idxs, origin + 1, device)
            per_patch.append({"loc": [r0, c0], **_stats(h_p)})
            densities.append(float(activity[r0 : r0 + dim, c0 : c0 + dim].mean()))
            halves = _state_halves(h_p)
            raw.append({k: v[0].reshape(v.shape[1], -1).cpu() for k, v in halves.items()})
        r1[f"ratio_{ratio}"] = {
            "threshold": thr,
            "mean_event_density": float(np.mean(densities)),  # F4
            "patches": per_patch,
        }
        torch.save(
            {k: torch.cat([r[k] for r in raw], dim=1) for k in ("hidden", "cell")},
            out / f"r1_state_{args.seed_label}_ratio{ratio}.pt",
        )
        logger.info("R1 ratio=%.3f thr=%d density=%.3f", ratio, thr, float(np.mean(densities)))
    results["R1"] = r1
    results["activity_max"] = act_max

    path = out / f"regimes_{args.seed_label}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
