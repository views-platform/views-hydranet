"""S3 (#186, Epic #183) — `forecast_composition` config field + fail-loud validators (ADR-069).

The composition axis (how gate + body combine into the emitted forecast) becomes a first-class
config setting: ``forecast_composition`` ∈ {self_zeroed, soft_gate, threshold_gate} (+
``gate_threshold`` τ). Validators enforce the ADR-069 rules fail-loud; the single authority for
"is this family self-zeroed?" is the family's ``self_zeroed`` attribute, mirrored torch-free in the
registry (parity-guarded).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig


def _cfg(valid_config_dict, **overrides):
    cfg = dict(valid_config_dict)
    cfg.update(overrides)
    return cfg


# ── the three valid arms + inert legacy default ──────────────────────────────────────────────
def test_zinb_self_zeroed_ok(valid_config_dict):
    cfg = _cfg(valid_config_dict, output_distribution="zinb", forecast_composition="self_zeroed")
    assert HydraNetConfig(**cfg).forecast_composition == "self_zeroed"


def test_nb_soft_gate_ok(valid_config_dict):
    cfg = _cfg(valid_config_dict, output_distribution="nb", forecast_composition="soft_gate")
    assert HydraNetConfig(**cfg).forecast_composition == "soft_gate"


def test_nb_threshold_gate_ok(valid_config_dict):
    cfg = _cfg(
        valid_config_dict,
        output_distribution="nb",
        forecast_composition="threshold_gate",
        gate_threshold=0.5,
    )
    obj = HydraNetConfig(**cfg)
    assert obj.forecast_composition == "threshold_gate" and obj.gate_threshold == 0.5


def test_default_is_self_zeroed_and_legacy_inert(valid_config_dict):
    # a legacy (default "standard") config: field defaults to self_zeroed and is inert (composer
    # never runs on the legacy path).
    assert "forecast_composition" not in valid_config_dict
    obj = HydraNetConfig(**valid_config_dict)
    assert obj.forecast_composition == "self_zeroed" and obj.gate_threshold is None


# ── the intentional breaking change: nb must DECLARE a gate composition ────────────────────────
def test_nb_without_composition_raises(valid_config_dict):
    # nb defaults to self_zeroed -> RAISE. nb has no self-zeroing; it must declare soft/threshold.
    cfg = _cfg(valid_config_dict, output_distribution="nb")  # no forecast_composition => default
    with pytest.raises(ValidationError, match="forecast_composition|self.zeroed|gate"):
        HydraNetConfig(**cfg)


# ── double-count guard: a self-zeroed family may not be gated ─────────────────────────────────
@pytest.mark.parametrize("comp", ["soft_gate", "threshold_gate"])
def test_zinb_with_gate_raises(valid_config_dict, comp):
    cfg = _cfg(
        valid_config_dict,
        output_distribution="zinb",
        forecast_composition=comp,
        gate_threshold=0.5 if comp == "threshold_gate" else None,
    )
    with pytest.raises(ValidationError, match="self.zeroed|double|zinb|gate"):
        HydraNetConfig(**cfg)


# ── ADR-068 emit_family_core: a core-emitting zinb IS gated (π stripped) ───────────────────────
@pytest.mark.parametrize("comp,thr", [("soft_gate", None), ("threshold_gate", 0.5)])
def test_zinbcore_gated_ok(valid_config_dict, comp, thr):
    # emit_family_core strips π → the bare core MUST be externally gated (not double-counting).
    cfg = _cfg(
        valid_config_dict,
        output_distribution="zinb",
        emit_family_core=True,
        forecast_composition=comp,
        gate_threshold=thr,
    )
    obj = HydraNetConfig(**cfg)
    assert obj.emit_family_core is True and obj.forecast_composition == comp


def test_zinbcore_self_zeroed_raises(valid_config_dict):
    # core-emitting zinb has no self-zeroing left → self_zeroed has no zero mechanism → RAISE.
    cfg = _cfg(
        valid_config_dict,
        output_distribution="zinb",
        emit_family_core=True,
        forecast_composition="self_zeroed",
    )
    with pytest.raises(ValidationError, match="self.zeroed|gate|zero mechanism"):
        HydraNetConfig(**cfg)


def test_emit_family_core_on_nb_raises(valid_config_dict):
    # nb has no structural-π core to strip → emit_family_core is meaningless → RAISE.
    cfg = _cfg(
        valid_config_dict,
        output_distribution="nb",
        emit_family_core=True,
        forecast_composition="soft_gate",
    )
    with pytest.raises(ValidationError, match="emit_family_core|self-zeroed|core"):
        HydraNetConfig(**cfg)


# ── threshold τ rules ─────────────────────────────────────────────────────────────────────────
def test_threshold_gate_requires_gate_threshold(valid_config_dict):
    cfg = _cfg(valid_config_dict, output_distribution="nb", forecast_composition="threshold_gate")
    with pytest.raises(ValidationError, match="gate_threshold"):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("bad_tau", [0.0, 1.0, -0.1, 1.5])
def test_gate_threshold_out_of_open_unit_interval_raises(valid_config_dict, bad_tau):
    cfg = _cfg(
        valid_config_dict,
        output_distribution="nb",
        forecast_composition="threshold_gate",
        gate_threshold=bad_tau,
    )
    with pytest.raises(ValidationError, match="gate_threshold|0.*1"):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("comp", ["soft_gate", "self_zeroed"])
def test_gate_threshold_set_without_threshold_gate_raises(valid_config_dict, comp):
    # gate_threshold is only meaningful for threshold_gate; elsewhere it is a silent no-op -> RAISE
    od = "nb" if comp == "soft_gate" else "zinb"
    cfg = _cfg(
        valid_config_dict, output_distribution=od, forecast_composition=comp, gate_threshold=0.5
    )
    with pytest.raises(ValidationError, match="gate_threshold|threshold_gate"):
        HydraNetConfig(**cfg)


# ── unknown value + legacy-head misuse ────────────────────────────────────────────────────────
def test_unknown_forecast_composition_rejected(valid_config_dict):
    cfg = _cfg(valid_config_dict, output_distribution="nb", forecast_composition="not_a_thing")
    with pytest.raises(ValidationError, match="forecast_composition.*not valid|not_a_thing"):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("comp", ["soft_gate", "threshold_gate"])
def test_legacy_head_with_gate_composition_raises(valid_config_dict, comp):
    # composition applies only to nb/zinb families; a gate on a legacy head is meaningless
    cfg = _cfg(
        valid_config_dict,
        output_distribution="standard",
        forecast_composition=comp,
        gate_threshold=0.5 if comp == "threshold_gate" else None,
    )
    with pytest.raises(ValidationError, match="forecast_composition|family|legacy"):
        HydraNetConfig(**cfg)


def test_zinbcore_self_zeroed_message_is_core_aware(valid_config_dict):
    # C-242: the message must blame emit_family_core (π strip), NOT claim zinb "is not
    # self-zeroed" — zinb IS self-zeroed; misdirecting that wastes debugging time.
    import pytest
    from pydantic import ValidationError

    cfg = dict(valid_config_dict)
    cfg.update(
        output_distribution="zinb", forecast_composition="self_zeroed", emit_family_core=True
    )
    with pytest.raises(ValidationError) as ei:
        HydraNetConfig(**cfg)
    msg = str(ei.value)
    assert "emit_family_core" in msg
    assert "is not self-zeroed" not in msg
