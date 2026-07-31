"""
VisualDiagnostics Test Suite

CIC Reference: VisualDiagnostics.md (docs/CICs/)
ADR References: ADR-005 (Testing), ADR-006 (CIC), ADR-008 (Failure Loudness),
               ADR-034 (Prediction Diagnostics), ADR-035 (Training Health Audit),
               ADR-037 (Geometric Health Visualization)

Coverage:
  BEIGE GATES — Null Object pattern: all 8 public methods, inactive → no files, no crash
  GREEN GATES — Active mode: save_dir created, PNG files produced for each array-based method
             — _calculate_stats: math correctness for known inputs
             — biopsy_tensor: 5D and 4D variants both save PNG
  RED GATES   — Adversarial inputs (all-NaN, empty sequences, empty histories, empty dossier,
              unsupported tensor ndim): active methods must NEVER propagate exceptions

Test isolation strategy:
  Active-mode tests use `monkeypatch.chdir(tmp_path)` so that the relative save_dir
  `reports/plots/diagnostics/test_ts/` resolves inside pytest's temp directory.
  The `run_timestamp="test_ts"` pin makes the save_dir path deterministic.
"""

import matplotlib

matplotlib.use("Agg")  # Must precede any pyplot import; avoids GUI windows in CI

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler

# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

INACTIVE_CFG = {"diagnostic_visualizations": False}

ACTIVE_CFG = {
    "diagnostic_visualizations": True,
    "spatial_cols": ["row", "col"],
    "time_col": "month_id",
    "height": 8,
    "width": 8,
    "row_offset": 0,
    "col_offset": 0,
    "regression_metrics": ["mse"],
    "classification_metrics": ["ap"],
    "loss_reg": "mse",
    "loss_class": "bce",
    # ADR 046 Compliance
    "features": ["lr_feat"],
    "regression_targets": ["lr_feat"],
    "classification_targets": ["by_feat"],
    "transformations": {"identity": ["lr_feat"]},
    "derivations": {"binary": [{"from": "lr_feat", "to": "by_feat", "threshold": 0}]},
}

TS = "test_ts"  # Fixed timestamp → deterministic save_dir


# ---------------------------------------------------------------------------
# Shared array helpers
# ---------------------------------------------------------------------------


def _seq(n: int = 6, h: int = 4, w: int = 4, c: int = 1, fill: float = 0.0):
    """Return a list of n numpy arrays shaped [H, W, C]."""
    return [np.full((h, w, c), fill, dtype=np.float32) for _ in range(n)]


def _arr(t: int = 6, h: int = 4, w: int = 4, c: int = 1, fill: float = 0.0):
    """Return a single numpy array shaped [T, H, W, C]."""
    return np.full((t, h, w, c), fill, dtype=np.float32)


def _minimal_dossier_reg():
    """Minimal regression dossier matching TrainingForensics output schema."""
    return {
        "mse": [2.0, 1.5, 1.0],
        "bias_instant": [1.2, 1.1, 1.0],
        "bias_running": [1.2, 1.15, 1.1],
        "y_bar": [3.0, 3.0, 3.0],
        "y_hat_bar": [3.6, 3.3, 3.0],
    }


def _minimal_dossier_cls():
    """Minimal classification dossier matching TrainingForensics output schema."""
    return {
        "ap": [0.7, 0.75, 0.8],
        "bias_instant": [0.9, 0.95, 1.0],
        "bias_running": [0.9, 0.92, 0.93],
        "y_bar": [0.1, 0.1, 0.1],
        "y_hat_bar": [0.09, 0.095, 0.1],
    }


@pytest.fixture
def vd_active(tmp_path, monkeypatch):
    """
    Provides an active VisualDiagnostics instance with all file I/O redirected
    to pytest's tmp_path via monkeypatch.chdir().
    Save dir will be: tmp_path/reports/plots/diagnostics/test_ts/
    """
    monkeypatch.chdir(tmp_path)
    return VisualDiagnostics(ACTIVE_CFG, run_timestamp=TS)


# ---------------------------------------------------------------------------
# BEIGE GATES — Null Object: ALL 8 public methods, inactive → no files, no crash
# ---------------------------------------------------------------------------


def test_vd_inactive_biopsy_autoregressive_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive VisualDiagnostics.biopsy_autoregressive must return
    silently without creating any files.
    CIC §3: "Null Object Contract — ALL 8 public methods return immediately."
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_autoregressive(_seq(), _seq(fill=0.5), "test_label", ["lr_feat"])
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_autoregressive must not write any PNG files."
    )


def test_vd_inactive_biopsy_training_performance_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_training_performance must produce no files.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_training_performance(_arr(), _arr(fill=0.5), _arr(), _arr(fill=0.3), "test")
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_training_performance must not write any PNG files."
    )


def test_vd_inactive_biopsy_loss_curves_no_files(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_loss_curves must produce no files.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_loss_curves([1.0, 0.8], [0.5, 0.4], [1.5, 1.2], "test")
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_loss_curves must not write any PNG files."
    )


def test_vd_inactive_biopsy_feature_dossier_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_feature_dossier must produce no files.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_feature_dossier("lr_feat_a", _minimal_dossier_reg(), "test", target_type="REG")
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_feature_dossier must not write any PNG files."
    )


def test_vd_inactive_biopsy_tensor_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_tensor must produce no files.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_tensor(torch.zeros(1, 6, 1, 4, 4), "test", ["lr_feat"])
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_tensor must not write any PNG files."
    )


def test_vd_inactive_biopsy_dataframe_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_dataframe must return before touching the DataFrame.
    Passing an empty DataFrame is safe — the method returns at `if not self.active: return`.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_dataframe(pd.DataFrame(), "test", features=[])
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_dataframe must not write any PNG files."
    )


def test_vd_inactive_biopsy_volume_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_volume must return before accessing its argument.
    Passing None is safe — the method returns at `if not self.active: return`.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_volume(None, "test")
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_volume must not write any PNG files."
    )


def test_vd_inactive_biopsy_sample_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_sample must return before accessing its arguments.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_sample(None, None, "test")
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_sample must not write any PNG files."
    )


def test_vd_inactive_biopsy_health_constellation_no_file(tmp_path, monkeypatch):
    """
    BEIGE GATE: Inactive biopsy_health_constellation must return silently with no files.
    CIC §3: Null Object Contract — ALL public methods return immediately when inactive.
    ADR-037: Null Object guard is mandatory for the radar plot path.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(INACTIVE_CFG)
    vd.biopsy_health_constellation({"encoder.weight": 1.0}, "test")
    assert list(tmp_path.rglob("*.png")) == [], (
        "Inactive biopsy_health_constellation must not write any PNG files."
    )


# ---------------------------------------------------------------------------
# GREEN GATE — Active constructor: save_dir is created
# ---------------------------------------------------------------------------


def test_vd_active_creates_save_dir(tmp_path, monkeypatch):
    """
    GREEN GATE: Active VisualDiagnostics must create its save_dir at construction time.
    The timestamp pin "test_ts" makes the path deterministic.
    CIC §5: "self.save_dir directory is created at construction (side effect of os.makedirs)."
    """
    monkeypatch.chdir(tmp_path)
    VisualDiagnostics(ACTIVE_CFG, run_timestamp=TS)
    expected = tmp_path / "reports" / "plots" / "diagnostics" / TS
    assert expected.is_dir(), (
        f"Active VisualDiagnostics must create save_dir at construction. "
        f"Expected directory: {expected}"
    )


# ---------------------------------------------------------------------------
# GREEN GATES — Active: array-based methods produce PNG files
# ---------------------------------------------------------------------------


def test_vd_biopsy_autoregressive_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_autoregressive must save exactly one PNG file.
    CIC §5: PNG file is saved to save_dir with sanitized stage_label as stem.
    """
    vd_active.biopsy_autoregressive(
        _seq(fill=1.0),
        _seq(fill=0.5),
        "autoregressive_stage",
        ["lr_feat_a"],
        time_indices=list(range(400, 406)),
    )
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_autoregressive must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "autoregressive" in pngs[0].name, (
        f"Saved PNG name must contain 'autoregressive'. Got: {pngs[0].name}"
    )


def test_vd_biopsy_training_performance_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_training_performance must save exactly one PNG file.
    """
    vd_active.biopsy_training_performance(
        _arr(fill=1.0),
        _arr(fill=0.5),
        _arr(fill=1.0),
        _arr(fill=0.3),
        "training_perf_stage",
        time_indices=list(range(400, 406)),
    )
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_training_performance must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )


def test_vd_biopsy_loss_curves_saves_two_pngs(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_loss_curves must save exactly 2 PNG files:
    'loss_evolution.png' (linear) and 'loss_evolution_log.png' (log scale).
    CIC §5: "biopsy_loss_curves saves two PNGs."
    """
    vd_active.biopsy_loss_curves([1.0, 0.8, 0.6], [0.5, 0.4, 0.3], [1.5, 1.2, 0.9], "lesson_1")
    pngs = {p.name for p in tmp_path.rglob("*.png")}
    assert "loss_evolution.png" in pngs, (
        f"biopsy_loss_curves must produce 'loss_evolution.png'. Found: {pngs}"
    )
    assert "loss_evolution_log.png" in pngs, (
        f"biopsy_loss_curves must produce 'loss_evolution_log.png'. Found: {pngs}"
    )
    assert len(pngs) == 2, f"biopsy_loss_curves must produce exactly 2 PNGs. Found: {pngs}"


def test_vd_biopsy_feature_dossier_regression_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_feature_dossier with target_type="REG" must save a PNG
    named 'forensic_reg_{target_name.lower()}.png'.
    CIC §5: forensic dossier PNG keyed by target name.
    """
    vd_active.biopsy_feature_dossier(
        "lr_feat_a", _minimal_dossier_reg(), "lesson_1", target_type="REG"
    )
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_feature_dossier (REG) must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "forensic_reg_lr_feat_a" in pngs[0].name, (
        f"PNG name must contain 'forensic_reg_lr_feat_a'. Got: {pngs[0].name}"
    )


def test_vd_biopsy_feature_dossier_classification_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_feature_dossier with target_type="CLS" must save a PNG
    named 'forensic_cls_{target_name.lower()}.png'.
    """
    vd_active.biopsy_feature_dossier(
        "by_feat_a", _minimal_dossier_cls(), "lesson_1", target_type="CLS"
    )
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_feature_dossier (CLS) must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "forensic_cls_by_feat_a" in pngs[0].name, (
        f"PNG name must contain 'forensic_cls_by_feat_a'. Got: {pngs[0].name}"
    )


# ---------------------------------------------------------------------------
# GREEN GATE — biopsy_tensor: 5D and 4D inputs both produce PNG
# ---------------------------------------------------------------------------


def test_vd_biopsy_tensor_5d_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_tensor with a 5D tensor [B, T, C, H, W] must save one PNG.
    The method extracts batch 0 and permutes to [T, H, W, C] internally.
    """
    t = torch.zeros(1, 6, 1, 4, 4)  # [B=1, T=6, C=1, H=4, W=4]
    vd_active.biopsy_tensor(t, "tensor_5d_stage", ["lr_feat_a"])
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_tensor (5D) must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )


def test_vd_biopsy_tensor_4d_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_tensor with a 4D tensor [T, C, H, W] must save one PNG.
    """
    t = torch.zeros(6, 1, 4, 4)  # [T=6, C=1, H=4, W=4]
    vd_active.biopsy_tensor(t, "tensor_4d_stage", ["lr_feat_a"])
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_tensor (4D) must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )


# ---------------------------------------------------------------------------
# GREEN GATE — biopsy_volume: active mode saves PNG
# ---------------------------------------------------------------------------


def test_vd_biopsy_volume_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_volume must save exactly one PNG in active mode.
    CIC §5: "biopsy_volume routes to pipeline/ (non-predict) or reasoning/ (predict)."
    Stage label without 'predict' → routed to 01_pipeline_health/.
    """
    data = np.zeros((6, 8, 8, 2), dtype=np.float32)
    vh = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["priogrid_gid", "lr_feat"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        identity_cols=["priogrid_gid"],
        feature_cols=["lr_feat"],
    )
    vd_active.biopsy_volume(vh, "volume_stage")
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_volume must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "volume_stage" in pngs[0].name, (
        f"PNG name must contain 'volume_stage'. Got: {pngs[0].name}"
    )


# ---------------------------------------------------------------------------
# GREEN GATE — biopsy_dataframe: active mode saves PNG
# ---------------------------------------------------------------------------


def test_vd_biopsy_dataframe_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_dataframe must save exactly one PNG in active mode.
    CIC §5: "biopsy_dataframe routes to 01_pipeline_health/."
    Constructs a minimal spatial DataFrame and calls biopsy_dataframe with one feature.
    """
    rows = []
    for month in range(400, 406):
        for r in range(4):
            for c in range(4):
                rows.append({"month_id": month, "row": r, "col": c, "lr_feat": float(r + c)})
    df = pd.DataFrame(rows)

    vd_active.biopsy_dataframe(df, "dataframe_stage", features=["lr_feat"])
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_dataframe must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "dataframe_stage" in pngs[0].name, (
        f"PNG name must contain 'dataframe_stage'. Got: {pngs[0].name}"
    )


# ---------------------------------------------------------------------------
# GREEN GATE — biopsy_sample: active mode saves PNG
# ---------------------------------------------------------------------------


def test_vd_biopsy_sample_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_sample must save exactly one PNG in active mode.
    CIC §5: "biopsy_sample routes to 02_training_dynamics/ and includes global context."
    Constructs a global (8×8) and a sample (4×4) VolumeHandler with matching channel maps
    so that _select_display_channels produces a non-empty signal feature.
    """
    global_data = np.ones((6, 8, 8, 2), dtype=np.float32)
    global_vh = VolumeHandler(
        data=global_data,
        axes=("T", "H", "W", "C"),
        channel_map=["priogrid_gid", "lr_feat"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        identity_cols=["priogrid_gid"],
        feature_cols=["lr_feat"],
        spatial_offset=(0, 0),
    )
    sample_data = np.ones((6, 4, 4, 2), dtype=np.float32)
    sample_vh = VolumeHandler(
        data=sample_data,
        axes=("T", "H", "W", "C"),
        channel_map=["priogrid_gid", "lr_feat"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        identity_cols=["priogrid_gid"],
        feature_cols=["lr_feat"],
        spatial_offset=(2, 2),
    )
    vd_active.biopsy_sample(sample_vh, global_vh, "sample_stage")
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_sample must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "sample_stage" in pngs[0].name, (
        f"PNG name must contain 'sample_stage'. Got: {pngs[0].name}"
    )


# ---------------------------------------------------------------------------
# GREEN GATE — _calculate_stats: math correctness
# ---------------------------------------------------------------------------


def test_vd_calculate_stats_known_array(tmp_path, monkeypatch):
    """
    GREEN GATE: _calculate_stats must produce correct μ, min, max for a known array.
    array = [0, 2, 4, 6] → μ=3.0, min=0.0, max=6.0
    The method is internal but forms the data substrate for ALL plot overlays.
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(ACTIVE_CFG, run_timestamp=TS)
    result = vd._calculate_stats(np.array([0.0, 2.0, 4.0, 6.0]))
    assert "μ:3.00" in result, f"_calculate_stats output must contain 'μ:3.00'. Got: {result!r}"
    assert "[0.00, 6.00]" in result, (
        f"_calculate_stats output must contain '[0.00, 6.00]'. Got: {result!r}"
    )


def test_vd_calculate_stats_all_nan_returns_empty(tmp_path, monkeypatch):
    """
    GREEN GATE: _calculate_stats must return 'EMPTY' for an all-NaN array
    without raising an exception.
    CIC §6: "all-NaN/empty data: _calculate_stats returns 'EMPTY' string."
    """
    monkeypatch.chdir(tmp_path)
    vd = VisualDiagnostics(ACTIVE_CFG, run_timestamp=TS)
    result = vd._calculate_stats(np.array([np.nan, np.nan]))
    assert result == "EMPTY", f"_calculate_stats on all-NaN must return 'EMPTY'. Got: {result!r}"


# ---------------------------------------------------------------------------
# GREEN GATE — biopsy_health_constellation: radar plot saved as PNG (ADR-037)
# ---------------------------------------------------------------------------


def test_vd_biopsy_health_constellation_saves_png(vd_active, tmp_path):
    """
    GREEN GATE: biopsy_health_constellation must save exactly one PNG in active mode.
    ADR-037: Radar plot of L2 weight norms per functional block (Encoder, Bottleneck,
    Decoder, MultiTaskHead). Saved to 02_training_dynamics/ as
    constellation_{safe_label}.png.
    """
    weight_norms = {
        "encoder_1.0.weight": 1.5,
        "encoder_1.1.weight": 0.8,
        "bottleneck.0.weight": 2.1,
        "decoder_final.0.weight": 0.9,
        "multi_task_head.reg.0.weight": 0.4,
        "multi_task_head.cls.0.weight": 0.6,
    }
    vd_active.biopsy_health_constellation(weight_norms, "end_of_training")
    pngs = list(tmp_path.rglob("*.png"))
    assert len(pngs) == 1, (
        f"biopsy_health_constellation must save exactly 1 PNG. Found: {[p.name for p in pngs]}"
    )
    assert "constellation" in pngs[0].name, (
        f"PNG name must contain 'constellation'. Got: {pngs[0].name}"
    )
    assert "end_of_training" in pngs[0].name, (
        f"PNG name must contain 'end_of_training'. Got: {pngs[0].name}"
    )


# ---------------------------------------------------------------------------
# RED GATES — Adversarial inputs: active methods must NOT propagate exceptions
# CIC §6: "NO exception propagation is permitted from any public method."
# ---------------------------------------------------------------------------


def test_vd_nan_arrays_biopsy_autoregressive_no_crash(vd_active):
    """
    RED GATE: biopsy_autoregressive with all-NaN truth_seq and pred_seq must not raise.
    CIC §6: all active methods wrap bodies in try/except — logs error, never raises.
    """
    nan_seq = [np.full((4, 4, 1), np.nan) for _ in range(6)]
    # Must not raise — the try/except in biopsy_autoregressive catches all exceptions
    vd_active.biopsy_autoregressive(nan_seq, nan_seq, "nan_stage", ["lr_feat"])


def test_vd_empty_sequences_biopsy_autoregressive_no_crash(vd_active):
    """
    RED GATE: biopsy_autoregressive with empty truth_seq and pred_seq must not raise.
    np.stack([]) raises ValueError — must be caught by try/except.
    """
    vd_active.biopsy_autoregressive([], [], "empty_stage", ["lr_feat"])


def test_vd_empty_histories_biopsy_loss_curves_no_crash(vd_active):
    """
    RED GATE: biopsy_loss_curves with all-empty histories must not raise.
    Matplotlib may issue warnings for empty data — must not propagate as exceptions.
    """
    vd_active.biopsy_loss_curves([], [], [], "empty_loss_stage")


def test_vd_empty_dossier_biopsy_feature_dossier_no_crash(vd_active):
    """
    RED GATE: biopsy_feature_dossier with an empty dossier ({}) must not raise.
    active_metrics will be empty → total_rows computed from zero metrics.
    CIC §6: active methods catch all exceptions internally.
    """
    vd_active.biopsy_feature_dossier("lr_feat_a", {}, "empty_dossier_stage", target_type="REG")


def test_vd_biopsy_tensor_bad_ndim_no_crash(vd_active):
    """
    RED GATE: biopsy_tensor with a 2D tensor must not raise.
    CIC §6: "Unsupported tensor ndim: logs a logger.warning and returns silently."
    """
    t = torch.zeros(4, 4)  # 2D — not supported, should log.warning and return
    vd_active.biopsy_tensor(t, "bad_ndim_stage", ["lr_feat"])


def test_vd_biopsy_dataframe_oob_indices_logs_error_no_crash(vd_active, tmp_path, caplog):
    """
    RED GATE: biopsy_dataframe with out-of-bounds spatial indices must log an
    error and return without crashing or producing a PNG.
    CIC §3: "Resilient Active Mode" — failures logged, never propagated.
    """
    import logging

    rows = []
    for month in range(400, 406):
        for r in range(20, 24):
            for c in range(4):
                rows.append({"month_id": month, "row": r, "col": c, "lr_feat": 1.0})
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_dataframe(df, "oob_stage", features=["lr_feat"])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_dataframe must log ERROR for out-of-bounds indices"
    assert any("row index" in r.message or "row indices" in r.message for r in errors)
    assert list(tmp_path.rglob("*.png")) == [], (
        "No PNG should be produced when indices are out of bounds."
    )


def test_vd_biopsy_dataframe_negative_indices_logs_error(vd_active, tmp_path, caplog):
    """
    RED GATE: biopsy_dataframe with negative spatial indices after offset
    subtraction must log an error and skip plotting.
    """
    import logging

    cfg = {**ACTIVE_CFG, "row_offset": 100}
    vd = VisualDiagnostics(cfg, run_timestamp=TS)

    rows = []
    for month in range(400, 406):
        for r in range(4):
            for c in range(4):
                rows.append({"month_id": month, "row": r, "col": c, "lr_feat": 1.0})
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.ERROR):
        vd.biopsy_dataframe(df, "neg_idx_stage", features=["lr_feat"])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_dataframe must log ERROR for negative indices"
    assert any("negative" in r.message for r in errors)


def test_vd_biopsy_dataframe_stochastic_list_in_cell_no_crash(vd_active, tmp_path):
    """
    RED GATE: biopsy_dataframe must handle list-in-cell prediction columns without crashing.

    In stochastic evaluation mode, InferenceOrchestrator produces DataFrames where each
    pred_<target> cell contains a Python list of S posterior samples.  biopsy_dataframe
    is called on this DataFrame before _to_pf_dict() collapses the lists.  It must
    silently collapse each list to its mean for plotting rather than raising
    "ValueError: setting an array element with a sequence."

    Regression guard for the stochastic-mode crash fixed in visual_diagnostics.py.
    """
    rows = []
    for month in range(400, 406):
        for r in range(4):
            for c in range(4):
                rows.append(
                    {
                        "month_id": month,
                        "row": r,
                        "col": c,
                        # list-in-cell: 5 posterior samples per cell
                        "pred_lr_feat": [float(r + c + s) for s in range(5)],
                    }
                )
    df = pd.DataFrame(rows)

    # Must complete without raising — the ValueError crash this guards against would propagate
    # out of biopsy_dataframe in DEBUG mode and was silently swallowed otherwise.
    vd_active.biopsy_dataframe(df, "stochastic_stage", features=["pred_lr_feat"])


# ---------------------------------------------------------------------------
# RED GATES — C-16: Error logging with exc_info (ADR-008 §4 Fail-Safe)
# ---------------------------------------------------------------------------


def test_vd_biopsy_volume_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_volume must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_volume(None, "broken")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_volume must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_tensor_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_tensor must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_tensor(None, "broken", channel_names=["a"])
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_tensor must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_sample_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_sample must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_sample(None, None, "broken")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_sample must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_health_constellation_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_health_constellation must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_health_constellation(None, "broken")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_health_constellation must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_autoregressive_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_autoregressive must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_autoregressive(None, None, "broken", channel_names=["a"])
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_autoregressive must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_training_performance_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_training_performance must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_training_performance(None, None, None, None, "broken")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_training_performance must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_loss_curves_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_loss_curves must log ERROR with exc_info on failure."""
    import logging

    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_loss_curves(None, None, None, "broken")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_loss_curves must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


def test_vd_biopsy_feature_dossier_logs_error_with_traceback(vd_active, caplog):
    """C-16: biopsy_feature_dossier must log ERROR with exc_info on failure."""
    import logging

    # A malformed value (y_bar an int, not a series) breaks the plotting aggregation inside the
    # guarded block — exercises the except path. (A missing-key dict no longer fails: the
    # horizon-split rewrite reads keys via .get() defaults.)
    with caplog.at_level(logging.ERROR):
        vd_active.biopsy_feature_dossier("broken", {"y_bar": 5}, "broken")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "biopsy_feature_dossier must log ERROR on failure"
    assert any(r.exc_info is not None and r.exc_info[0] is not None for r in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
