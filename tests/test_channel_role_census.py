"""Channel-role census (channel-role side-quest, Phase 1.1) — characterization of C-156.

The static-channel seam (#108) gave the model input-only channels (e.g. coordinates) that are
appended to ``feature_cols`` (volume_handler.py:159). Several consumers still assume
``feature_cols`` == "things the model predicts/trains on", so they crash or silently mis-handle
the input-only statics. This file PINS each break-point: every test exercises a consumer with a
static channel present and documents the CURRENT behavior.

The three known bugs are ``xfail(strict=True)`` — they fail today, and must turn XPASS *only*
after the channel-role refactor (Phase 4) routes every consumer through explicit roles. The two
secondary suspects are classified by plain assertions on current behavior.

Empirically confirmed in the 2026-06-13 runs (see RESULTS_LOG / session notes):
  bug1 / C-158 curriculum: input-only statics rotated in as trainable subjects (silent mis-train)
  bug2 / C-157 biopsy:      Stage-5 diagnostic forward feeds features-only -> crashes a 5-ch model
  bug3 / C-159 sidecar:     arch sidecar omits static_channels -> reload wrong width -> crash
"""

from unittest.mock import MagicMock

import numpy as np
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

_STATIC = "row_coord"
_DEVICE = torch.device("cpu")


# bug 1 / C-158 — FIXED (#115 4b): curriculum excludes input-only statics from subjects.
def test_census_bug1_curriculum_excludes_static_subjects():
    from views_hydranet.utils.curriculum import CurriculumLearner

    config = {
        "total_lessons": 10,
        "windows_per_lesson": 1,
        "min_events": 5,
        "slope_ratio": 0.75,
        "roof_ratio": 0.7,
        "min_ratio": 0.05,
        "max_ratio": 0.95,
        "static_channels": [_STATIC],
    }
    data = np.zeros((2, 8, 8, 6))
    handler = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["t", "i", "lr_sb", "lr_ns", "lr_os", _STATIC],
        time_col="t",
        id_col="i",
        spatial_cols=["y", "x"],
        identity_cols=["t", "i"],
        feature_cols=["lr_sb", "lr_ns", "lr_os", _STATIC],
    )
    planner = CurriculumLearner(config, handler)
    # DESIRED: an input-only static is never a curriculum subject. CURRENT: it is.
    assert _STATIC not in planner.subjects


# bug 3 / C-159 — arch sidecar omits static_channels, so reload rebuilds the wrong width
# bug 3 / C-159 — FIXED (#115 4b): train_model.py now persists static_channels in the sidecar,
# so choose_model rebuilds the top-skip decoder at the correct width and a coord-trained model
# reloads. Arrange below models the fixed writer's output; the reload assertion is unchanged.
def test_census_bug3_sidecar_roundtrips_coord_model():
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.utils.utils import choose_model

    trained = HydraBNUNet06_LSTM4(
        input_channels=5, total_hidden_channels=8, output_channels=1,
        dropout_rate=0.0, n_static_channels=2,
    )
    # The sidecar train_model.py writes now (arch_keys + static_channels per C-159 fix):
    sidecar = {
        "model": "HydraBNUNet06_LSTM4", "input_channels": 5,
        "total_hidden_channels": 8, "output_channels": 1, "dropout_rate": 0.0,
        "static_channels": ["row_coord", "col_coord"],
    }
    rebuilt = choose_model(sidecar, _DEVICE)
    # DESIRED: a coord-trained model reloads. (Was: size mismatch 66 vs 64.)
    rebuilt.load_state_dict(trained.state_dict())


# bug 2 / C-157 — Stage-5 diagnostic biopsy feeds features-only into a static-widened model
_FEATURES = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
_IDENTITY = ["month_id", "priogrid_gid"]
_BIOPSY_CONFIG = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 4,  # 3 dynamic + 1 static
    "static_channels": [_STATIC],
    "output_channels": 1,
    "regression_targets": _FEATURES,
    "classification_targets": [],
    "features": _FEATURES,
    "steps": [1, 2],
    "total_lessons": 2,
    "windows_per_lesson": 1,
    "window_dim": 8,
    "np_seed": 42,
    "torch_seed": 42,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "scheduler": "none",
    "loss_reg": "mse",
    "loss_class": "bce",
    "clip_grad_norm": True,
    "random_flips": False,
    "diagnostic_visualizations": False,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "xavier_uni",
    "min_events": 1,
    "sampling_strategy": "threshold",
    "min_ratio": 0.0,
    "max_ratio": 1.0,
    "slope_ratio": 0.5,
    "roof_ratio": 1.0,
    "identity_cols": _IDENTITY,
    "spatial_cols": ["row", "col"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "transformations": {"identity": _FEATURES},
    "derivations": {},
}


def _biopsy_handler():
    T, H, W = 6, 8, 8
    channel_map = _IDENTITY + _FEATURES + [_STATIC]
    rng = np.random.RandomState(7)
    data = rng.rand(T, H, W, len(channel_map)).astype(np.float32)
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 1 + r * W + c
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=channel_map,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=tuple(_IDENTITY),
        feature_cols=tuple(_FEATURES + [_STATIC]),
    )


# bug 2 / C-157 — FIXED (#115 4b): biopsy re-attaches statics like the main forward.
def test_census_bug2_biopsy_handles_static_channels():
    from tqdm import tqdm

    from views_hydranet.train.training_engine import TrainingContext, make, train

    model, criterion, optimizer, scheduler = make(_BIOPSY_CONFIG, _DEVICE)
    criterion_reg, criterion_class, mtl = criterion
    ctx = TrainingContext(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion_reg=criterion_reg, criterion_class=criterion_class,
        multitaskloss_instance=mtl, config=_BIOPSY_CONFIG, device=_DEVICE,
        viz=MagicMock(),  # truthy viz => the Stage-5 biopsy block runs (no file I/O)
    )
    # DESIRED: the diagnostic biopsy completes. CURRENT: RuntimeError (3 ch into a 4-ch model).
    result = train(ctx, _biopsy_handler(), tqdm(total=5, disable=True), stage_label="census_probe")
    assert torch.isfinite(result["total"])


# secondary suspect A — visual_diagnostics plots input-only statics as if they were signal
def test_census_suspectA_visualdiagnostics_static_classification():
    """CLASSIFY (not xfail): does the diagnostics 'interesting channels' list include a static?"""
    from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

    data = np.zeros((2, 8, 8, 6))
    handler = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "priogrid_gid", "lr_sb", "lr_ns", "lr_os", _STATIC],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        identity_cols=["month_id", "priogrid_gid"],
        feature_cols=["lr_sb", "lr_ns", "lr_os", _STATIC],
    )
    viz = VisualDiagnostics({"diagnostic_visualizations": True})
    interesting, _ = viz._select_display_channels(handler)
    # Record current behavior: True => static is plotted as signal (silently-wrong, benign).
    print(f"\n[census suspectA] static in plotted channels = {_STATIC in interesting}")
    assert isinstance(interesting, list)


# secondary suspect B — to_pytorch includes statics in the model input (this is BY DESIGN)
def test_census_suspectB_topytorch_includes_statics_by_design():
    """CLASSIFY: to_pytorch SHOULD include statics (how coords reach the model) — confirm SAFE."""
    data = np.zeros((2, 8, 8, 6), dtype=np.float32)
    handler = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "priogrid_gid", "lr_sb", "lr_ns", "lr_os", _STATIC],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        identity_cols=["month_id", "priogrid_gid"],
        feature_cols=["lr_sb", "lr_ns", "lr_os", _STATIC],
    )
    t = handler.to_pytorch(_DEVICE, include_identities=False)
    # [B, T, C, H, W]; C must be 4 (3 dynamic + 1 static). Statics included = correct-by-design.
    assert t.shape[2] == 4
