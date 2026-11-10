"""``G-T1`` must FIRE: a disagreeing arm config has to be REFUSED, not reconciled.

Epic #579, story S4 #583. Same discipline as ``test_conditional_magnitude.py`` and
``test_e1_data.py``: every gate gets a correct input asserted to pass -- without which a gate that
rejects everything would look perfect -- and a deliberately broken input asserted to be refused.

⭐ The gate guards an AGREEMENT, so the only way to break it is to make an arm disagree. That
substitution is the test, and it is done by monkeypatching the per-arm reader rather than by
editing a config in views-models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-10_rank_coupling_verdict/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "arm_config.py").exists(),
    reason="the rank-coupling-verdict dossier tools are not present in this tree",
)

if (_TOOLS / "arm_config.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))


@pytest.fixture
def ac():
    import arm_config

    return arm_config


def test_tau_is_read_from_the_config_not_transcribed(ac):
    """⭐ The correct input, asserted to PASS. Also pins the value the epic depends on."""
    tau, composition = ac.read_arm_gate_convention("ctlfullzero_fortytwo")
    assert composition == "threshold_gate"
    assert tau == pytest.approx(0.20), (
        "every open-set-dependent number in epic #579 is wrong if this moves, with no error signal"
    )
    assert ac.resolve_tau() == pytest.approx(0.20)


def test_a_disagreeing_arm_is_refused(ac, monkeypatch):
    """⛔ The discriminating half: one arm with a different tau must RAISE, never be averaged."""
    real = ac.read_arm_gate_convention

    def fake(arm: str):
        return (0.26, "threshold_gate") if arm == "ctlfullzero_fortythree" else real(arm)

    monkeypatch.setattr(ac, "read_arm_gate_convention", fake)
    with pytest.raises(ValueError, match="G-T1 FAILED"):
        ac.resolve_tau()


def test_a_foreign_composition_is_refused(ac, monkeypatch):
    """⛔ The deterministic open set is a property of ``threshold_gate`` specifically."""
    monkeypatch.setattr(ac, "read_arm_gate_convention", lambda arm: (0.20, "soft_gate"))
    with pytest.raises(ValueError, match="G-T1 FAILED"):
        ac.resolve_tau()


def test_open_set_is_deterministic_and_matches_the_threshold(ac):
    """⭐ B's firing convention: ``gate >= tau``, no coin, boundary INCLUDED."""
    import numpy as np

    gate = np.array([0.0, 0.1999, 0.20, 0.2001, 1.0])
    got = ac.open_set(gate, 0.20)
    assert got.tolist() == [False, False, True, True, True]
    assert ac.open_set(gate, 0.20).tolist() == got.tolist(), "must not consume randomness"


def test_the_ghost_arm_is_excluded_from_the_control_set(ac):
    """⚠️ The ghost removes an ARM-form seed, never a control one => sigma-hat has n = 6."""
    assert "upctwofullzero_fortysix" in ac.GHOST_ARMS
    assert "upctwofullzero_fortysix" not in ac.CONTROL_ARMS
    assert len(ac.CONTROL_ARMS) == 6
    assert not set(ac.GHOST_ARMS) & set(ac.CONTROL_ARMS + ac.ARM_FORM_ARMS)
    # ⭐ pre-registered spellings, not directory names — the readout identifiers must match PREREG
    assert all(a.startswith("ctlfullzero_") for a in ac.CONTROL_ARMS)
    assert all(a.startswith("upctwofullzero_") for a in ac.ARM_FORM_ARMS)
