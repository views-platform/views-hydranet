#!/usr/bin/env python3
"""st_bias_entry.py — EXP-BIAS (#308 Phase 1) for ONE arm, on ITS OWN real data.

Run with ``cwd`` inside the arm directory, mirroring ``realism_arm_entry.py``: the arm's own
manager builds the combined config and runs the production data pipeline, so the model sees the
distribution it was trained on. That matters — the trained gate sits near logit -6.7, and a
synthetic input would move it, making the Bernoulli score term unrepresentative of the training
regime. Measuring in a regime the phenomenon does not occupy is **C-325**, which is the mistake
this whole phase exists to avoid repeating.

The measurement and its verdict rule are pre-registered in ``05_analysis_plan.md`` amendment A3.

Usage (normally via run_st_bias.sh):
    python st_bias_entry.py --model-dir <dir> --artifact <name.pt> --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HYD = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HYD))
sys.path.insert(0, str(_HYD / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import st_bias  # noqa: E402
from diagnose_io_gain import load_model  # noqa: E402

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.manager.hydranet_manager import HydranetManager  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _attach_static_channels,
    _SequenceIndices,
)
from views_hydranet.utils.config_initializer import ConfigInitializer  # noqa: E402
from views_hydranet.utils.curriculum import CurriculumLearner  # noqa: E402
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics  # noqa: E402
from views_hydranet.utils.volume_sampler import VolumeSampler  # noqa: E402


class _WindowManager(HydranetManager):
    """Exposes the production training-data pipeline, nothing else.

    `_run_data_pipeline` is the same call `_train_model_artifact` makes, so the window this returns
    is drawn exactly as a training window is.
    """

    def real_window(self):
        # Mirrors `_train_model_artifact` steps 0 and 1 exactly: config handshake, preflight,
        # then the shared data pipeline. Anything less and the window would not be the window
        # training sees.
        # `run_type` normally arrives from the CLI args a production run is launched with; this
        # entry has no such args, and the config schema requires it. The setter merges at highest
        # priority. calibration is the partition every arm in this dossier trained and scored on.
        self.configs = {"run_type": "calibration"}
        self.configs = ConfigInitializer(self.configs).get_config()
        # DataFetcher's default filename is `<run_type>_viewser_df`, but this fleet is on
        # views-datafactory and its file is `calibration_datafactory_df.parquet`. Production
        # supplies the override through `_cached_data_path`; resolve it from disk instead of
        # hardcoding either name, and REFUSE if it is ambiguous rather than picking one.
        raw = Path(self._model_path.data_raw)
        found = sorted(raw.glob(f"{self.configs['run_type']}_*_df.parquet"))
        if len(found) != 1:
            raise SystemExit(
                f"st_bias_entry: expected exactly one {self.configs['run_type']} parquet in "
                f"{raw}, found {[f.name for f in found]}"
            )
        self._cached_data_path = found[0]
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)
        handler, _scaler, _sniffer = self._run_data_pipeline(viz)
        sampler = VolumeSampler(handler, self.configs)
        planner = CurriculumLearner(self.configs, handler)
        target, threshold = planner.get_lesson(0)
        batch, _ = sampler.get_batch(target, threshold, batch_size=1)
        return batch[0], self.configs


def rollout_loss(model, x, h, idx, step_i, fed, family):
    """The training objective one step past the handoff: family NLL + gate BCE.

    ONE step of lookahead, deliberately. A longer horizon would require feeding another DRAW,
    which changes the estimand from "is the credit at this handoff pointed correctly" into a
    different, multi-sample quantity. Robustness is taken across several handoff INDICES instead,
    which varies the conditions without moving the target.

    The multitask balancer is frozen in every arm here (`freeze_multitask_balancer: True`), so an
    unweighted sum is the objective the arms actually trained on.
    """
    i = step_i + 1
    t0 = x[:, i]
    out = model(_attach_static_channels(fed, t0, idx), h)
    t1 = x[:, i + 1]
    y_reg, y_cls = t1[:, idx.reg], t1[:, idx.cls]
    npar = family.n_params
    total = out.reg.new_zeros(())
    for j in range(idx.n_reg):
        params = out.reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)
        total = total + family.nll(params, y_reg[:, j])
    return total + F.binary_cross_entropy_with_logits(out.cls, (y_cls > 0).float())


def measure(model, family, x, idx, *, step_i, n_draws, seed, composition, device):
    """cos(d_ST, d_SF) at one handoff, plus the split-half self-agreement of d_SF."""
    params = [p for p in model.parameters() if p.requires_grad]
    H, W = x.shape[-2], x.shape[-1]
    full, half_a, half_b = st_bias._Accumulator(), st_bias._Accumulator(), st_bias._Accumulator()
    rand_a, rand_b = st_bias._Accumulator(), st_bias._Accumulator()
    # Two independent ways to split the samples. The parity split pairs consecutive draws, whose
    # seeds differ by 1 -- if seed parity interacted with the sampler at all, the parity split
    # would manufacture the very (anti)correlation it is meant to detect. The permuted split has
    # no such structure. Reporting both means a disagreement between them is visible instead of
    # being silently inherited by the verdict.
    assign = torch.randperm(n_draws, generator=torch.Generator().manual_seed(seed)) % 2
    curve, max_fwd_gap = [], 0.0

    for n in range(n_draws):
        torch.manual_seed(seed * 100003 + n)
        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)
        for i in range(step_i):
            t0 = x[:, i]
            h = model(_attach_static_channels(t0[:, idx.feat], t0, idx), h).h_next

        t0 = x[:, step_i]
        out = model(_attach_static_channels(t0[:, idx.feat], t0, idx), h)
        reg, gate, h_next = out.reg, torch.sigmoid(out.cls), out.h_next

        fed, counts, mask = st_bias.draw_feedback(reg, gate, family, idx.n_reg, composition)
        surrogate = st_bias.composed_mean_log1p(reg, gate, family, idx.n_reg, composition)

        loss_cut = rollout_loss(model, x, h_next, idx, step_i, fed.detach(), family)
        loss_on = rollout_loss(
            model, x, h_next, idx, step_i, surrogate + (fed - surrogate).detach(), family
        )
        # Straight-through changes ONLY the backward, so the forward values must be identical.
        # A free per-draw correctness check; a mismatch voids the measurement (A3 gate 6).
        max_fwd_gap = max(max_fwd_gap, abs(float(loss_cut) - float(loss_on)))

        d_st = st_bias.flat_grad(loss_on, params) - st_bias.flat_grad(loss_cut, params)
        g_logp = st_bias.flat_grad(
            st_bias.score_log_prob(reg, gate, counts, mask, family, idx.n_reg), params
        )

        full.add(float(loss_cut), g_logp, d_st)
        (half_a if n % 2 == 0 else half_b).add(float(loss_cut), g_logp, d_st)
        (rand_a if int(assign[n]) == 0 else rand_b).add(float(loss_cut), g_logp, d_st)
        if (n + 1) in (32, 64, 128, 256, 512, 1024, 2048, n_draws):
            curve.append({"n": n + 1, "cos": st_bias.cosine(full.d_st(), full.d_sf())})

    return {
        "step_i": step_i,
        "n_draws": n_draws,
        "cos_st_sf": st_bias.cosine(full.d_st(), full.d_sf()),
        "split_half_cos_sf": st_bias.cosine(half_a.d_sf(), half_b.d_sf()),
        "split_half_cos_sf_permuted": st_bias.cosine(rand_a.d_sf(), rand_b.d_sf()),
        # Two noisy measurements cannot correlate above their own reliability. Spearman-Brown
        # lifts the half-sample reliability to the full sample, and the attenuation correction
        # then says what the cosine would be if the reference were measured without error. Both
        # are reported; NEITHER is the pre-registered readout, which stays the raw cos.
        "reliability_full": (
            2 * st_bias.cosine(rand_a.d_sf(), rand_b.d_sf())
            / (1 + st_bias.cosine(rand_a.d_sf(), rand_b.d_sf()))
        ),
        "negative_control_cos": st_bias.cosine(full.d_st(), torch.randn_like(full.d_st())),
        "curve": curve,
        "plateaued": st_bias.plateaued([c["cos"] for c in curve]),
        "max_forward_gap": max_fwd_gap,
        "norm_d_st": float(full.d_st().norm()),
        "norm_d_sf": float(full.d_sf().norm()),
        "mean_loss": full.sum_l / full.n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=256)
    ap.add_argument("--steps", default="1,3,5", help="handoff indices to measure")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from views_pipeline_core.managers import ModelPathManager  # noqa: PLC0415

    mgr = _WindowManager(model_path=ModelPathManager(Path(a.model_dir) / "main.py"))
    sample_handler, cfg = mgr.real_window()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = sample_handler.to_pytorch(device, include_identities=False)
    feature_names = [n for n in sample_handler.channel_map if n in sample_handler.tensor_cols]
    idx = _SequenceIndices(feature_names, cfg)

    artifact = Path(a.model_dir) / "artifacts" / a.artifact
    # .eval() on purpose: BatchNorm on running stats and dropout off give a DETERMINISTIC map, so
    # the d_ST vs d_SF contrast isolates the estimator's geometry instead of mixing in BN batch
    # noise and dropout masks that differ between draws. This is a different regime from training's
    # forward, and the write-up must say so.
    model, _ = load_model(artifact, device)
    family = resolve_family(cfg["output_distribution"])

    results = [
        measure(
            model,
            family,
            x,
            idx,
            step_i=int(s),
            n_draws=a.n_draws,
            seed=a.seed,
            composition=cfg.get("forecast_composition", "soft_gate"),
            device=device,
        )
        for s in a.steps.split(",")
    ]
    out = {
        "model_dir": a.model_dir,
        "artifact": a.artifact,
        "composition": cfg.get("forecast_composition"),
        "family": cfg["output_distribution"],
        "window_shape": list(x.shape),
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "handoffs": results,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    for r in results:
        print(
            f"step {r['step_i']:>2}  cos(d_ST,d_SF) = {r['cos_st_sf']:+.4f}   "
            f"split-half(d_SF) = {r['split_half_cos_sf']:+.4f}   "
            f"neg-ctrl = {r['negative_control_cos']:+.4f}   "
            f"plateau = {r['plateaued']}   fwd-gap = {r['max_forward_gap']:.2e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
