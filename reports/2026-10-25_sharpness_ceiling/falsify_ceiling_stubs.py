"""Failing stubs for the four falsifications found by /falsify on 2026-10-25.

⛔ DELIBERATELY NOT NAMED `test_*.py`. There is no `[tool.pytest.ini_options]` in pyproject.toml,
so a plain `pytest` from the repo root collects by the default `test_*.py` / `*_test.py` globs and
would pick this up — and these assertions are RED BY DESIGN until the prereg is corrected. Run it
explicitly:

    pytest reports/2026-10-25_sharpness_ceiling/falsify_ceiling_stubs.py

Each test encodes one finding. A test turning GREEN means the corresponding defect has been fixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "2026-10-25_sharpness_ceiling_prereg.md"

# Measured by the probe run, 2026-10-25. Primary slot: sb, interleaved, eps=1e-6, E-W.
C_HALF_D1 = 0.4982
B1_ESTIMATOR_HALF_VS_HALF = (1.0089, 0.9957, 1.0091, 0.9861, 0.9833, 0.9866)  # d = 1,2,4,8,16,32
C_LOGGED_D1, C_RAW_D1 = 0.4982, 0.1509
BODY_B1_D1 = 0.391  # reports/2026-10-21_body_scoreboard.md, control, in-sample E-W


def _c_full(c_half: float) -> float:
    """`ceiling.py`'s half->full correction, copied so this file imports nothing heavy."""
    r = 2.0 * c_half / (1.0 - c_half)
    return r / (r + 1.0)


# ---------------------------------------------------------------- F2 (hard)
def test_F2_the_ceiling_is_commensurable_with_the_body_number_it_caps():
    """⛔⛔ THE ONE THAT MATTERS. §3's decision rule caps the arm's upside at C(1)=0.498 and
    compares it to B-1's 0.391. They are different estimators.

    B-1 divides out the ratio of the two fields' variances (the same-shape null). C(1) does not.
    Applying B-1's OWN estimator to two independent truth halves gives ~1.0 at every lag, because
    the variance normalisation cancels exactly the noise C(1) measures.

    => The ceiling against which 0.391 must be read is ~1.0, not 0.498, and §3 must be rewritten.
    """
    ceiling_under_b1 = float(np.mean(B1_ESTIMATOR_HALF_VS_HALF))
    assert abs(ceiling_under_b1 - C_HALF_D1) < 0.05, (
        f"the two estimators disagree: B-1 form on two truth halves = {ceiling_under_b1:.4f}, "
        f"C_half(1) = {C_HALF_D1:.4f}. 0.391 may not be read against 0.498."
    )


def test_F2b_the_prereg_no_longer_caps_the_arm_at_C1():
    """The §3 text that encodes the category error must be gone."""
    text = PREREG.read_text()
    assert "capped at" not in text.lower(), (
        "§3 still caps the arm's upside at C(1); the measured ceiling under B-1's estimator is ~1.0"
    )


# ---------------------------------------------------------------- F3 (hard)
def test_F3_c_full_is_not_applied_across_a_log_transform():
    """⛔ Spearman-Brown's premise is additive noise with variance proportional to 1/n -- a
    statement about COUNTS. `c_full` is applied to a field transformed by log(rate + eps).

    Own evidence: the same correction on the raw and logged fields disagrees by 3.3x. Both cannot
    describe the same latent noise process, so the premise fails on at least one scale -- and the
    scale it is defensible on (counts) is not the one the primary slot reports.
    """
    full_logged = _c_full(C_LOGGED_D1)
    full_raw = _c_full(C_RAW_D1)
    assert abs(full_logged - full_raw) < 0.10, (
        f"c_full on the logged field = {full_logged:.4f} but on raw counts = {full_raw:.4f}; "
        "the 1/n additive-noise premise cannot hold on both scales, so 0.665 is not derivable"
    )


# ---------------------------------------------------------------- F4 (soft)
def test_F4_the_void_rules_are_scoped_to_the_primary_slot():
    """⚠️ W1/W2 are drafted globally ('any C(d) < 0 or > 1') over a sweep whose degenerate arms the
    prereg itself PREDICTS in Q-4. They fired on `raw`/`contiguous` slots while the primary slot was
    clean and monotone, and the driver printed VOID and then reported a verdict anyway.
    """
    text = PREREG.read_text()
    w2 = next((ln for ln in text.splitlines() if "**W2**" in ln), "")
    assert w2 and "primary" in w2.lower(), (
        f"W2 is still global, so it voids the measurement on slots Q-4 predicts will be "
        f"degenerate: {w2.strip()!r}"
    )


# ---------------------------------------------------------------- F1 (hard)
def test_F1_the_control_run_artifact_is_on_disk():
    """⛔ 'the run is done and we have everything we need' -- the control artifact was not present.
    Newest checkpoint anywhere under views_platform was 2026-10-21 (silver_kestrel), four days old.
    """
    models = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
    if not models.is_dir():
        pytest.skip("views-models not present on this machine")
    fresh = [
        p
        for p in models.glob("*/data/generated/**/*.pt")
        if p.stat().st_mtime > 1792800000  # 2026-10-24T00:00Z
    ]
    assert fresh, "no control-run checkpoint newer than 2026-10-24 anywhere in views-models"
