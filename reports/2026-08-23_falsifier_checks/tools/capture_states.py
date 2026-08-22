#!/usr/bin/env python3
"""capture_states.py — save real (x, h) pairs from a live rollout, for the sigma_max Jacobian.

Check D (#294) needs the TRUE recurrent Jacobian, not the analytic bound — the bound floors at
~2.5 for this model regardless of cell-state magnitude and can never be informative
(`05_analysis_plan.md` AMENDMENT 1, and `results/sigma_max.json`). The true Jacobian's contracting
factors are the gates f, i, o, which depend on the actual input field, so real states are required.

Mirrors `freeze_arm_entry.py`'s manager override exactly — the one seam this repo already uses for
diagnostics. It sets NO diagnostic flag; it only attaches a forward hook that copies the first N
``(x, h)`` pairs to disk and then stops the run by raising. Nothing in `views_hydranet/` changes,
and `freeze_recurrent` stays None, so the rollout it observes is the production path.

Usage (via run_capture.sh):
    python capture_states.py --model-dir <views-models>/models/<m> --artifact <f>.pt --out <dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_hydranet.manager.hydranet_manager import HydranetManager

logger = logging.getLogger(__name__)


class _Enough(RuntimeError):
    """Raised to stop the rollout once enough states are captured — this is a probe, not a run."""


class CaptureManager(HydranetManager):
    """`HydranetManager` that hooks the model's forward to copy real (x, h) pairs.

    Same seam as `FreezeArmManager`: `_setup_evaluation` returns an immutable NamedTuple whose
    orchestrator is a normal object. No diagnostic flag is set — `freeze_recurrent` stays None —
    so what the hook observes is the production free-running path.
    """

    out_dir: Path = Path(".")
    n_states: int = 6
    _seen: int = 0

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        model = ctx.orchestrator.model

        def hook(_mod, inputs, _output):
            # forward(self, x, h) -> inputs == (x, h)
            if len(inputs) < 2:
                return
            x, h = inputs[0], inputs[1]
            torch.save(
                {"x": x.detach().cpu(), "h": h.detach().cpu(), "step": self._seen},
                self.out_dir / f"state_{self._seen:02d}.pt",
            )
            logger.info("captured state %d: x%s h%s", self._seen, tuple(x.shape), tuple(h.shape))
            self._seen += 1
            if self._seen >= self.n_states:
                raise _Enough(f"captured {self._seen} states — stopping the probe")

        model.register_forward_hook(hook)
        logger.info("CaptureManager: hook attached, will capture %d states", self.n_states)
        return ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6)
    args, _ = ap.parse_known_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    mgr = CaptureManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    mgr.out_dir = out
    mgr.n_states = args.n_states

    # `parse_args()` takes no argument list — it reads sys.argv, which this script has already
    # consumed. Build the namespace from the parser instead, exactly as `freeze_arm_entry.py`
    # does: keeps the pipeline's own validation without rewriting global state.
    run_args = ForecastingModelArgs.from_namespace(
        ForecastingModelArgs._create_parser().parse_args(
            [
                "--run_type",
                "calibration",
                "--evaluate",
                "--saved",
                "--artifact_name",
                args.artifact,
            ]
        )
    )
    try:
        mgr.execute_single_run(run_args)
    except _Enough as e:
        logger.info("%s", e)
    n = len(list(out.glob("state_*.pt")))
    print(f"captured {n} states into {out}")
    return 0 if n else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
