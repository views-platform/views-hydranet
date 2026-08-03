"""
TrainingForensics Test Suite

CIC Reference: TrainingForensics.md
ADR References: ADR-005 (Testing), ADR-008 (Failure Loudness), ADR-035 (Training Health Audit)

Coverage:
  RED GATES  — Init validation (forbidden metrics → ValueError, ADR 008 pattern)
             — record() on unknown target → KeyError
  GREEN GATES — Metric math: MSE, MAE, instantaneous bias, running bias, y_bar, ŷ_bar
             — Multi-lesson state: history length, cumulative running bias
  BEIGE GATES — Empty lessons (carry-forward), zero-sum y (sentinel 1.0), unknown get_dossier,
             — AP/AUC no-positive-samples edge cases
"""

import pytest
import torch

from views_hydranet.utils.training_forensics import TrainingForensics

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FORENSICS_CFG = {
    "regression_targets": ["lr_feat_a"],
    "classification_targets": ["by_feat_a"],
    "regression_metrics": ["mse", "mae"],
    "classification_metrics": ["ap"],
}

FORENSICS_AUC_CFG = {
    "regression_targets": ["lr_feat_a"],
    "classification_targets": ["by_feat_a"],
    "regression_metrics": ["mse"],
    "classification_metrics": ["auc"],
}


def _t(values: list) -> torch.Tensor:
    """Helper: wrap a Python list in a CPU float32 tensor."""
    return torch.tensor(values, dtype=torch.float32)


def _record_reg(tf: TrainingForensics, y: list, yh: list) -> None:
    tf.record("REG:lr_feat_a", _t(y), _t(yh))


def _record_cls(tf: TrainingForensics, y: list, yh: list) -> None:
    tf.record("CLS:by_feat_a", _t(y), _t(yh))


# ---------------------------------------------------------------------------
# RED GATES — Init: forbidden metrics raise ValueError (ADR 008 Narrative Failure)
# ---------------------------------------------------------------------------


def test_forensics_rejects_f1_in_reg_metrics():
    """
    RED GATE: TrainingForensics.__init__ must raise ValueError when 'f1' is
    listed as a regression_metric. F1 requires a threshold not supported here.
    """
    cfg = {**FORENSICS_CFG, "regression_metrics": ["f1"]}
    with pytest.raises(ValueError, match=r"[Ff]1|threshold"):
        TrainingForensics(cfg)


def test_forensics_rejects_accuracy_in_cls_metrics():
    """
    RED GATE: 'accuracy' in classification_metrics must raise ValueError.
    """
    cfg = {**FORENSICS_CFG, "classification_metrics": ["accuracy"]}
    with pytest.raises(ValueError, match=r"[Aa]ccuracy|threshold"):
        TrainingForensics(cfg)


def test_forensics_rejects_recall():
    """
    RED GATE: 'recall' in any metric list must raise ValueError.
    """
    cfg = {**FORENSICS_CFG, "classification_metrics": ["recall"]}
    with pytest.raises(ValueError, match=r"[Rr]ecall|threshold"):
        TrainingForensics(cfg)


def test_forensics_rejects_precision():
    """
    RED GATE: 'precision' in any metric list must raise ValueError.
    """
    cfg = {**FORENSICS_CFG, "regression_metrics": ["precision"]}
    with pytest.raises(ValueError, match=r"[Pp]recision|threshold"):
        TrainingForensics(cfg)


# ---------------------------------------------------------------------------
# RED GATE — record(): unknown target raises KeyError
# ---------------------------------------------------------------------------


def test_forensics_record_raises_on_unknown_target():
    """
    RED GATE: record() must raise KeyError if the target was not initialized.
    CIC §6: "Missing Targets: Raises KeyError if a recording call is made for
    a target not initialized in the config."
    """
    tf = TrainingForensics(FORENSICS_CFG)
    with pytest.raises(KeyError, match=r"ghost_target"):
        tf.record("ghost_target", _t([1.0]), _t([0.5]))  # Not a valid "REG:/CLS:" key


# ---------------------------------------------------------------------------
# GREEN GATES — Metric Mathematics (Independent Auditor Check)
# ---------------------------------------------------------------------------


def test_forensics_mse_correctness():
    """
    GREEN GATE: MSE = mean((y - ŷ)²)
    y=[0, 4], ŷ=[0, 2] → MSE = mean([0, 4]) = 2.0
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [0.0, 4.0], [0.0, 2.0])
    tf.finalize_lesson()

    mse_val = tf.history["REG:lr_feat_a"]["mse"][0]
    assert mse_val == pytest.approx(2.0), f"MSE calculation failed. Expected 2.0, got {mse_val}"


def test_forensics_mae_correctness():
    """
    GREEN GATE: MAE = mean(|y - ŷ|)
    y=[0, 4], ŷ=[0, 2] → MAE = mean([0, 2]) = 1.0
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [0.0, 4.0], [0.0, 2.0])
    tf.finalize_lesson()

    mae_val = tf.history["REG:lr_feat_a"]["mae"][0]
    assert mae_val == pytest.approx(1.0), f"MAE calculation failed. Expected 1.0, got {mae_val}"


def test_forensics_bias_instant_correctness():
    """
    GREEN GATE: bias_instant = Σŷ / Σy
    y=[3, 1], ŷ=[6, 2] → Σy=4, Σŷ=8 → bias_instant=2.0
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [3.0, 1.0], [6.0, 2.0])
    tf.finalize_lesson()

    bias_val = tf.history["REG:lr_feat_a"]["bias_instant"][0]
    assert bias_val == pytest.approx(2.0), f"Instant Bias failed. Expected 2.0, got {bias_val}"


def test_forensics_y_bar_and_yhat_bar_correctness():
    """
    GREEN GATE: y_bar = mean(y), yhat_bar = mean(ŷ)
    y=[0, 4], ŷ=[0, 2] → y_bar=2.0, yhat_bar=1.0
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [0.0, 4.0], [0.0, 2.0])
    tf.finalize_lesson()

    y_bar = tf.history["REG:lr_feat_a"]["y_bar"][0]
    yh_bar = tf.history["REG:lr_feat_a"]["y_hat_bar"][0]
    assert y_bar == 2.0
    assert yh_bar == 1.0


def test_forensics_bias_running_is_cumulative_across_lessons():
    """
    GREEN GATE: bias_running must accumulate across lessons, not reset.
    """
    tf = TrainingForensics(FORENSICS_CFG)

    # Lesson 1
    _record_reg(tf, [2.0], [4.0])
    tf.finalize_lesson()
    assert tf.history["REG:lr_feat_a"]["bias_running"][0] == 2.0

    # Lesson 2
    _record_reg(tf, [2.0], [0.0])
    tf.finalize_lesson()
    # Σy = 4, Σŷ = 4 → Running Bias = 1.0
    assert tf.history["REG:lr_feat_a"]["bias_running"][1] == 1.0


# ---------------------------------------------------------------------------
# BEIGE GATES — Edge Cases and Stability (Robust Auditor)
# ---------------------------------------------------------------------------


def test_forensics_empty_lesson_first_appends_zero():
    """
    BEIGE GATE: finalize_lesson() with no records must append 0.0 or carry forward.
    """
    tf = TrainingForensics(FORENSICS_CFG)
    tf.finalize_lesson()
    mse_history = tf.history["REG:lr_feat_a"]["mse"]
    assert len(mse_history) == 1
    assert mse_history[0] == 0.0


def test_forensics_empty_lesson_subsequent_carries_forward():
    """
    BEIGE GATE: finalize_lesson() must carry forward the last recorded value if empty.
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [0.0, 4.0], [0.0, 2.0])  # MSE = 2.0
    tf.finalize_lesson()

    tf.finalize_lesson()  # Empty lesson
    mse_history = tf.history["REG:lr_feat_a"]["mse"]
    assert len(mse_history) == 2
    assert mse_history[1] == 2.0


def test_forensics_zero_sum_y_bias_falls_back_to_one():
    """
    BEIGE GATE: When Σy=0, bias_instant falls back to 1.0 (sentinel).
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [0.0, 0.0], [0.5, 0.5])
    tf.finalize_lesson()
    assert tf.history["REG:lr_feat_a"]["bias_instant"][0] == 1.0


def test_forensics_get_dossier_returns_empty_dict_on_unknown():
    """
    BEIGE GATE: get_dossier() should not raise KeyError on unknown target.
    """
    tf = TrainingForensics(FORENSICS_CFG)
    assert tf.get_dossier("REG:non_existent") == {}


def test_forensics_ap_no_positive_samples_returns_zero():
    """
    BEIGE GATE: AP calculation when all y=0.
    """
    tf = TrainingForensics(FORENSICS_CFG)
    _record_cls(tf, [0.0, 0.0, 0.0], [0.3, 0.7, 0.1])
    tf.finalize_lesson()
    ap_val = tf.history["CLS:by_feat_a"]["ap"][0]
    assert ap_val == 0.0


def test_forensics_auc_no_positive_samples_returns_half():
    """
    BEIGE GATE: AUC calculation when all y=0.
    """
    tf = TrainingForensics(FORENSICS_AUC_CFG)
    _record_cls(tf, [0.0, 0.0, 0.0], [0.3, 0.7, 0.1])
    tf.finalize_lesson()
    auc_val = tf.history["CLS:by_feat_a"]["auc"][0]
    assert auc_val == 0.5


def test_forensics_history_length_tracks_finalize_calls():
    """
    GREEN GATE: History length must track finalize_lesson calls.
    """
    tf = TrainingForensics(FORENSICS_CFG)
    for _ in range(3):
        _record_reg(tf, [1.0], [1.5])
        tf.finalize_lesson()

    assert len(tf.history["REG:lr_feat_a"]["mse"]) == 3


def test_forensics_reg_and_cls_history_length_aligned():
    """
    GREEN GATE: Reg and Cls histories must stay in lockstep.
    """
    tf = TrainingForensics(FORENSICS_CFG)
    # Record Reg ONLY
    for _ in range(4):
        _record_reg(tf, [1.0], [1.5])
        tf.finalize_lesson()

    assert len(tf.history["REG:lr_feat_a"]["mse"]) == 4
    assert len(tf.history["CLS:by_feat_a"]["ap"]) == 4


# ---------------------------------------------------------------------------
# FAMILY-AWARE (C-213) — calibration collapse + parameter-health frame
# ---------------------------------------------------------------------------


def _params(rows: list) -> torch.Tensor:
    """[N, n_params] activated-params tensor (mu, theta[, pi])."""
    return torch.tensor(rows, dtype=torch.float32)


def test_family_reg_collapse_records_each_targets_own_ey():
    """C-213 regression: the family collapse must return each target's OWN self-zeroed E[y]=(1-π)μ,
    NOT a raw channel. A ZINB reg output is target-major [t0(μ,θ,π), t1(μ,θ,π)] — the old forensic
    sliced channel j, so target-1 got target-0's θ. The helper must slice j*npar:(j+1)*npar."""
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import _family_target_log1p_mean

    fam = resolve_family("zinb")
    # [B=1, n_reg*3=6, H=1, W=1], target-major: t0=(μ10,θ.5,π.9), t1=(μ4,θ2,π.5)
    reg = torch.tensor([10.0, 0.5, 0.9, 4.0, 2.0, 0.5]).view(1, 6, 1, 1)
    out = _family_target_log1p_mean(reg, fam)  # [1, 2, 1, 1] of log1p((1-π)μ) per target
    assert out.shape == (1, 2, 1, 1)
    # t0: (1-.9)*10 = 1.0 → log1p(1); t1: (1-.5)*4 = 2.0 → log1p(2). t1 must use its OWN μ,π.
    assert out[0, 0, 0, 0].item() == pytest.approx(torch.log1p(torch.tensor(1.0)).item(), abs=1e-5)
    assert out[0, 1, 0, 0].item() == pytest.approx(torch.log1p(torch.tensor(2.0)).item(), abs=1e-5)


def test_record_params_theta_cov_flags_collapse():
    """Param-health: constant θ over active cells → theta_cov ≈ 0 (< 0.02 F1 collapse)."""
    tf = TrainingForensics(FORENSICS_CFG)
    tf.record_params(
        "REG:lr_feat_a",
        _params([[5.0, 1.0, 0.9], [3.0, 1.0, 0.8], [8.0, 1.0, 0.95]]),
        _t([2.0, 3.0, 1.0]),
    )
    tf.finalize_lesson()
    assert tf.history["REG:lr_feat_a"]["theta_cov"][0] == pytest.approx(0.0, abs=1e-6)


def test_record_params_theta_cov_reflects_spread():
    """Varied θ over active cells → theta_cov > 0.02 (head learning heteroscedastic dispersion)."""
    tf = TrainingForensics(FORENSICS_CFG)
    tf.record_params(
        "REG:lr_feat_a",
        _params([[5.0, 0.5, 0.9], [3.0, 1.0, 0.8], [8.0, 2.0, 0.95]]),
        _t([2.0, 3.0, 1.0]),
    )
    tf.finalize_lesson()
    assert tf.history["REG:lr_feat_a"]["theta_cov"][0] > 0.02


def test_record_params_theta_cov_is_over_active_cells_only():
    """θ CoV must use ACTIVE (truth>0) cells only — zero-cell θ must not leak in (pred-1/F1)."""
    tf = TrainingForensics(FORENSICS_CFG)
    # active (y>0) cells have constant θ=1.0; zero cells have wild θ that must NOT affect the CoV
    tf.record_params(
        "REG:lr_feat_a",
        _params([[5.0, 1.0, 0.9], [1.0, 99.0, 0.1], [8.0, 1.0, 0.95], [1.0, 0.01, 0.1]]),
        _t([2.0, 0.0, 3.0, 0.0]),  # active = idx 0, 2 (θ = 1, 1)
    )
    tf.finalize_lesson()
    assert tf.history["REG:lr_feat_a"]["theta_cov"][0] == pytest.approx(0.0, abs=1e-6)


def test_record_params_pi_range_flags_degeneracy():
    """π stats: a degenerate π (all ≈ 1) → pi_min ≈ pi_max ≈ 1 (F1 π→1 dead-cell degeneracy)."""
    tf = TrainingForensics(FORENSICS_CFG)
    tf.record_params(
        "REG:lr_feat_a",
        _params([[5.0, 1.0, 0.995], [3.0, 1.0, 0.997], [8.0, 1.0, 0.996]]),
        _t([2.0, 3.0, 1.0]),
    )
    tf.finalize_lesson()
    h = tf.history["REG:lr_feat_a"]
    assert h["pi_min"][0] > 0.99 and h["pi_max"][0] > 0.99
    assert h["pi_bar"][0] == pytest.approx(0.996, abs=1e-3)


def test_record_params_mu_bar():
    """μ̄ = mean of the μ channel (the body's mean conditional magnitude)."""
    tf = TrainingForensics(FORENSICS_CFG)
    tf.record_params(
        "REG:lr_feat_a",
        _params([[2.0, 1.0, 0.9], [4.0, 1.0, 0.8], [6.0, 1.0, 0.95]]),
        _t([2.0, 3.0, 1.0]),
    )
    tf.finalize_lesson()
    assert tf.history["REG:lr_feat_a"]["mu_bar"][0] == pytest.approx(4.0)


def test_record_params_nb_has_no_pi_stats():
    """nb (n_params=2) → μ̄ + θ-CoV present, NO π stats (the π row is omitted for nb)."""
    tf = TrainingForensics(FORENSICS_CFG)
    tf.record_params(
        "REG:lr_feat_a", _params([[5.0, 1.0], [3.0, 1.0], [8.0, 1.0]]), _t([2.0, 3.0, 1.0])
    )
    tf.finalize_lesson()
    h = tf.history["REG:lr_feat_a"]
    assert "mu_bar" in h and "theta_cov" in h
    assert "pi_bar" not in h


def test_point_head_records_no_param_health_keys():
    """Parity: without record_params (a point head), NO param-health keys appear — unchanged."""
    tf = TrainingForensics(FORENSICS_CFG)
    _record_reg(tf, [0.0, 4.0], [0.0, 2.0])
    tf.finalize_lesson()
    h = tf.history["REG:lr_feat_a"]
    assert "theta_cov" not in h and "mu_bar" not in h
