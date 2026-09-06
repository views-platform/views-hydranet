"""ADR-027 §2.1: the cell clamp is a production config setting, and OFF must stay byte-identical.

The amendment that admits `freeze_recurrent` to production rests on one property: **a config that
omits the key behaves exactly as it did before the key existed.** Without that, promoting a
diagnostic to a config field silently changes every model in the fleet. That property is the first
test here; the rest are the ADR's Beige Team clauses.

This file must NOT be read as reinstating `freeze_h`. That mechanism stays retired and its guard
(`tests/test_inference_logic.py::test_freeze_h_option_retired`) stays green — see ADR-027 §2.1 for
why the June retirement and the September promotion are compatible rather than contradictory.
"""

from __future__ import annotations

import pytest

from views_hydranet.utils.config_initializer import HydraNetConfig


@pytest.fixture
def cfg(valid_config_dict):
    """The repo's canonical valid config (tests/conftest.py::valid_config_dict)."""
    return dict(valid_config_dict)


def _with(cfg, **over):
    out = dict(cfg)
    out.update(over)
    return out


class TestTheOffPathIsUnchanged:
    """The load-bearing property of the amendment."""

    def test_omitting_the_key_yields_none(self, cfg):
        """A config that never mentions the clamp must resolve to None — ADR-027 §2 behaviour."""
        cfg = HydraNetConfig(**cfg)
        assert cfg.freeze_recurrent is None, (
            "a config omitting freeze_recurrent resolved to "
            f"{cfg.freeze_recurrent!r}, not None — the amendment would change every model"
        )

    def test_the_orchestrator_does_not_clamp_when_the_key_is_absent(self, cfg):
        """Reading from config must not turn the clamp on by accident."""
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        cfg = HydraNetConfig(**cfg).model_dump()
        cfg.pop("freeze_recurrent", None)
        cfg.pop("freeze_recurrent_weight", None)
        orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
        InferenceOrchestrator.__init__(
            orch, config=cfg, model=_DummyModel(), device=_cpu(), visualizer=None
        )
        assert orch.freeze_recurrent is None


class TestTheClampActuallyReachesInference:
    """The promotion itself. Without this, the amendment is prose.

    A mutation audit found that reverting the orchestrator to `self.freeze_recurrent = None` —
    undoing the entire ADR-027 §2.1 change — left every other test in this file green. The config
    field validated, the CIC claimed the field existed, and nothing checked that a config asking
    for the clamp ever switched it on. That is **C-303** (prose asserting a check the code does not
    implement), which the register carries twelve times.
    """

    def test_a_config_asking_for_the_clamp_gets_it(self, cfg):
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        built = HydraNetConfig(**_with(cfg, freeze_recurrent="cell")).model_dump()
        orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
        InferenceOrchestrator.__init__(
            orch, config=built, model=_DummyModel(), device=_cpu(), visualizer=None
        )
        assert orch.freeze_recurrent == "cell", (
            "a config with freeze_recurrent='cell' produced an orchestrator with "
            f"{orch.freeze_recurrent!r} — the setting never reaches inference, so ADR-027 §2.1 "
            "is documentation only"
        )
        assert orch.freeze_recurrent_weight == 1.0

    def test_the_weight_reaches_inference_too(self, cfg):
        """A dial nobody can turn is not a dial."""
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        built = HydraNetConfig(
            **_with(cfg, freeze_recurrent="cell", freeze_recurrent_weight=0.25)
        ).model_dump()
        orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
        InferenceOrchestrator.__init__(
            orch, config=built, model=_DummyModel(), device=_cpu(), visualizer=None
        )
        assert orch.freeze_recurrent_weight == 0.25

    def test_the_clamp_without_a_weight_fails_loud(self, cfg):
        """No shadow default: a bare dict must not have the blend strength guessed for it."""
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        bare = _with(cfg, freeze_recurrent="cell")
        bare.pop("freeze_recurrent_weight", None)
        orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
        with pytest.raises(ValueError, match="freeze_recurrent_weight is missing"):
            InferenceOrchestrator.__init__(
                orch, config=bare, model=_DummyModel(), device=_cpu(), visualizer=None
            )


class TestTheModeIsValidated:
    """ADR-027 §2.1 Beige Team: an unknown mode fails loud, it does not silently no-op."""

    @pytest.mark.parametrize("bad", ["Cell", "CELL", "cel", "hidden_state", "true", ""])
    def test_unknown_modes_are_rejected(self, cfg, bad):
        with pytest.raises(ValueError, match="freeze_recurrent must be None or one of"):
            HydraNetConfig(**_with(cfg, freeze_recurrent=bad))

    @pytest.mark.parametrize("good", ["hidden", "cell", "all"])
    def test_the_three_known_modes_are_accepted(self, cfg, good):
        """Anti-vacuity for the test above: if nothing were accepted, it would prove nothing."""
        assert HydraNetConfig(**_with(cfg, freeze_recurrent=good)).freeze_recurrent == good

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_the_weight_is_bounded(self, cfg, bad):
        with pytest.raises(Exception):
            HydraNetConfig(**_with(cfg, freeze_recurrent="cell", freeze_recurrent_weight=bad))

    def test_the_weight_default_is_the_measured_value(self, cfg):
        """Every measurement behind ADR-027 §2.1 used a hard freeze; the default must be it."""
        assert HydraNetConfig(**cfg).freeze_recurrent_weight == 1.0


class TestFreezeHStaysRetired:
    """This amendment must not smuggle the retired mechanism back in."""

    def test_no_freeze_h_field_exists(self):
        assert "freeze_h" not in HydraNetConfig.model_fields, (
            "freeze_h was retired 2026-06-05 and ADR-027 §2.1 explicitly does not reinstate it"
        )


def _cpu():
    import torch

    return torch.device("cpu")


class _DummyModel:
    """The orchestrator's __init__ only stores the model; it is never called here."""
