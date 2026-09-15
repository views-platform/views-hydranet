#!/usr/bin/env python3
"""preflight_input_noise.py — can the noise knob ACT? Measured on the arm's own config, at a
TRAINED checkpoint.

The launch precondition for Epic #311's S5 (#316). **C-324 is Tier 1**: a broken implementation must
not produce the same signature as a null result. #308's first run burned 276 minutes training two
arms to byte-identical weights because the treatment was inert on the path production takes.

**And C-325**, which is why this runs at a trained checkpoint rather than at initialisation: two of
#308's mechanism tests measured an untrained network, returned correct numbers, and were recorded as
ruling a mechanism out — the effect they were looking for takes ~33 lessons of training to appear.
An untrained network cannot exhibit what training creates. Here the noise acts on the *input*, so
its effect on the input is weight-independent — but its effect on the LOSS, which is what training
follows, is not.

Run with cwd inside the arm directory, like `st_bias_entry.py`. Exits non-zero on an inert knob.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HYD = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HYD))
sys.path.insert(0, str(_HYD / "scripts"))

from diagnose_io_gain import load_model  # noqa: E402
from potency_check import PotencyError, assert_control_fires, assert_potent  # noqa: E402
from views_hydranet.manager.hydranet_manager import HydranetManager  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _noisable_channels,
    _process_sequence,
    _SequenceIndices,
)
from views_hydranet.utils.config_initializer import ConfigInitializer  # noqa: E402
from views_hydranet.utils.curriculum import CurriculumLearner  # noqa: E402
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics  # noqa: E402
from views_hydranet.utils.volume_sampler import VolumeSampler  # noqa: E402


class _WindowManager(HydranetManager):
    """Exposes the production training-data pipeline. Duplicated from the bptt dossier's
    `st_bias_entry.py` rather than imported: dossiers are archived independently, and a
    cross-dossier import would make this one's launch gate depend on another's lifetime."""

    def real_window(self):
        self.configs = {"run_type": "calibration"}
        self.configs = ConfigInitializer(self.configs).get_config()
        raw = Path(self._model_path.data_raw)
        found = sorted(raw.glob(f"{self.configs['run_type']}_*_df.parquet"))
        if len(found) != 1:
            raise SystemExit(f"expected one calibration parquet in {raw}, found {len(found)}")
        self._cached_data_path = found[0]
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)
        handler, _s, _sn = self._run_data_pipeline(viz)
        sampler = VolumeSampler(handler, self.configs)
        planner = CurriculumLearner(self.configs, handler)
        target, threshold = planner.get_lesson(0)
        batch, _ = sampler.get_batch(target, threshold, batch_size=1)
        return batch[0], self.configs


def _loss(model, x, idx, cfg, dropout, seed=11):
    torch.manual_seed(seed)
    res = _process_sequence(
        train_tensor=x,
        model=model,
        h=model.init_hTtime(hidden_channels=model.base, H=x.shape[-2], W=x.shape[-1]).to(x.device),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=torch.nn.BCEWithLogitsLoss(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=x.device,
        event_threshold=0.0,
        input_noise_dropout=dropout,
        input_noise_segment=cfg["time_steps"] if dropout is not None else None,
        input_noise_channels=_noisable_channels(cfg) if dropout is not None else None,
    )
    return float(res["total"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact-path", required=True,
                    help="a TRAINED checkpoint — normally the floor's, since the arm "
                         "being gated has not been trained yet (C-325)")
    a = ap.parse_args()

    from views_pipeline_core.managers import ModelPathManager  # noqa: PLC0415

    mgr = _WindowManager(model_path=ModelPathManager(Path(a.model_dir) / "main.py"))
    handler, cfg = mgr.real_window()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = handler.to_pytorch(device, include_identities=False)
    names = [n for n in handler.channel_map if n in handler.tensor_cols]
    idx = _SequenceIndices(names, cfg)
    model, _ = load_model(Path(a.artifact_path), device)

    rate = cfg.get("input_noise_dropout")
    if rate is None:
        print(f"CONTROL  {Path(a.model_dir).name}: input_noise_dropout is None — nothing to gate")
        return 0

    try:
        # C-324: the knob must move a number, measured with the arm's OWN rate on a TRAINED model.
        r = assert_potent(
            lambda d: _loss(model, x, idx, cfg, d),
            off=None,
            on=rate,
            name=f"{Path(a.model_dir).name} input noise @ trained checkpoint",
            min_relative_change=1e-4,
        )
        # A null from a blind harness is not a null: prove the readout can see a KNOWN effect.
        assert_control_fires(
            lambda d: _loss(model, x, idx, cfg, d),
            baseline=None,
            known_effect=0.9,
            name="the loss readout responds to a large dropout",
        )
    except PotencyError as e:
        print(f"INERT    {e}")
        return 1
    print(
        f"POTENT   {Path(a.model_dir).name}: loss off={r['off']:.6f} on={r['on']:.6f} "
        f"rel={r['relative_change']:.6f} (rate={rate})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
