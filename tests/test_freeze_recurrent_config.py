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


def _build_inference(config_dict, *, freeze_recurrent):
    """Construct a real `HydraNetInference` the way the orchestrator does, after any override.

    S5/#358: the clamp verdict is emitted by this constructor, because it is the last point at
    which `freeze_recurrent` can change. `HydraNetInference` type-checks the model, so this needs a
    real `nn.Module` rather than the orchestrator test's bare `_DummyModel`.
    """
    import torch.nn as nn

    from views_hydranet.utils.hydranet_inference import HydraNetInference

    return HydraNetInference(
        nn.Identity(),
        config_dict,
        device="cpu",
        visualizer=None,
        freeze_recurrent=freeze_recurrent,
        freeze_recurrent_weight=1.0,
    )


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

    @pytest.mark.parametrize(
        "entry", ["generate_prediction_frames", "generate_prediction_frames_streaming"]
    )
    def test_both_construction_sites_forward_the_clamp_to_inference(self, cfg, entry, monkeypatch):
        """The two tests above read the orchestrator's ATTRIBUTES. Nothing read what the
        orchestrator hands to `HydraNetInference`, and that is the clamp's only delivery path:
        `/code-review max` on #372 deleted both `freeze_recurrent=` forwarding kwargs at both
        construction sites and the FULL suite stayed green — every `freeze_recurrent: 'cell'`
        roster config would run unclamped under a log saying `from config`. Pinned here at both
        sites, by intercepting the constructor call itself."""
        import views_hydranet.utils.inference_orchestrator as io_mod
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        built = HydraNetConfig(
            **_with(cfg, freeze_recurrent="cell", freeze_recurrent_weight=0.25)
        ).model_dump()
        orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
        InferenceOrchestrator.__init__(
            orch, config=built, model=_DummyModel(), device=_cpu(), visualizer=None
        )

        received: dict = {}

        class _Intercept(Exception):
            pass

        def _fake_inference(*a, **kw):
            received.update(kw)
            raise _Intercept  # stop before anything downstream needs a real model

        monkeypatch.setattr(io_mod, "HydraNetInference", _fake_inference)
        kwargs = dict(handler=None, scaler=None, origins=[1], all_targets=[])
        if entry == "generate_prediction_frames_streaming":
            kwargs["origin_sink"] = lambda *_a, **_k: None
        with pytest.raises(_Intercept):
            getattr(orch, entry)(**kwargs)

        assert received.get("freeze_recurrent") == "cell", (
            f"{entry} built HydraNetInference with freeze_recurrent="
            f"{received.get('freeze_recurrent')!r}; the orchestrator's setting never reached "
            "inference, so the production clamp is documentation only"
        )
        assert received.get("freeze_recurrent_weight") == 0.25, (
            f"{entry} forwarded weight {received.get('freeze_recurrent_weight')!r}, not 0.25"
        )

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

    def test_a_clamp_with_a_zero_weight_is_rejected(self, cfg):
        """S1/#354: `freeze_recurrent='cell'` + `weight=0.0` claims the clamp and delivers none.

        `blend_recurrent_state` returns the freely-evolved state unchanged at weight 0.0, so this
        pair validated, logged `recurrent state CLAMPED`, and shipped the unclamped control — the
        C-324 inert-knob signature on a production setting (C-331, escalated from diagnostic arms).
        """
        with pytest.raises(ValueError, match="is inert"):
            HydraNetConfig(**_with(cfg, freeze_recurrent="cell", freeze_recurrent_weight=0.0))

    @pytest.mark.parametrize("mode", ["hidden", "cell", "all"])
    def test_the_rejection_covers_every_mode(self, cfg, mode):
        """Not just the production mode — a diagnostic arm can lie about itself too."""
        with pytest.raises(ValueError, match="is inert"):
            HydraNetConfig(**_with(cfg, freeze_recurrent=mode, freeze_recurrent_weight=0.0))

    @pytest.mark.parametrize("weight", [1.0, 0.5, 0.25, 0.1])
    def test_every_acting_weight_is_still_accepted(self, cfg, weight):
        """Anti-vacuity: if the guard rejected everything it would prove nothing.

        M41 swept w over {0, 0.1, 0.25, 0.5, 0.75, 1.0}; every non-zero point must stay reachable.
        """
        c = HydraNetConfig(**_with(cfg, freeze_recurrent="cell", freeze_recurrent_weight=weight))
        assert c.freeze_recurrent_weight == weight

    def test_a_zero_weight_without_a_clamp_is_still_legal(self, cfg):
        """The ADR governs the RANGE; the guard governs the PAIR.

        ADR-027 §2.1's Beige Team contract specifies the field as "outside [0, 1] is rejected", so
        `gt=0.0` on the field would contradict it and would delete M41's `w=0` reference. A weight
        of 0.0 with no clamp is meaningless but honest — nothing claims to be happening.
        """
        c = _with(cfg, freeze_recurrent_weight=0.0)
        c.pop("freeze_recurrent", None)
        assert HydraNetConfig(**c).freeze_recurrent_weight == 0.0

    def test_the_message_names_the_way_out(self, cfg):
        """A guard that says 'no' without saying 'do this instead' gets worked around."""
        with pytest.raises(ValueError) as e:
            HydraNetConfig(**_with(cfg, freeze_recurrent="cell", freeze_recurrent_weight=0.0))
        msg = str(e.value)
        assert "freeze_recurrent=None" in msg, msg
        assert "C-324" in msg or "C-331" in msg, msg


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


class TestTheConsumerRefusesTheInertPairToo:
    """S1 put the rule in HydraNetConfig. Research drivers bypass pydantic — they set the two
    attributes on the orchestrator after construction — so `cell@0.0` still reached
    `HydraNetInference`, which printed `CLAMPED — weight=0.0` over an identity blend (#372)."""

    @pytest.mark.parametrize("mode", ["cell", "hidden", "all"])
    def test_a_zero_weight_with_a_mode_is_rejected_at_inference(self, cfg, mode):
        import torch.nn as nn

        from views_hydranet.utils.hydranet_inference import HydraNetInference

        built = HydraNetConfig(**cfg).model_dump()
        with pytest.raises(ValueError, match="inert"):
            HydraNetInference(
                nn.Identity(),
                built,
                device="cpu",
                visualizer=None,
                freeze_recurrent=mode,
                freeze_recurrent_weight=0.0,
            )

    def test_a_zero_weight_with_no_mode_is_still_fine_at_inference(self, cfg):
        """The weight is unread without a mode; rejecting it here would break the control."""
        import torch.nn as nn

        from views_hydranet.utils.hydranet_inference import HydraNetInference

        built = HydraNetConfig(**cfg).model_dump()
        HydraNetInference(
            nn.Identity(),
            built,
            device="cpu",
            visualizer=None,
            freeze_recurrent=None,
            freeze_recurrent_weight=0.0,
        )


class TestTheClampIsVisibleInTheLog:
    """A production setting that changes the forecast must announce itself.

    ADR-027 §2.1 admitted the cell clamp to production. The artifact sidecar records 12 keys and
    `freeze_recurrent` is not among them, and nothing printed it — so a delivered forecast carried
    **no evidence of whether the clamp was on**. A run with a mistyped key would be
    indistinguishable in every log from a run with the clamp live: the **C-324** inert-knob
    signature, on the one setting whose only purpose is to change the output.

    Found while smoke-testing a 29-hour run whose entire premise is the clamp.
    """

    def test_a_clamped_run_says_so(self, cfg, caplog):
        """S5/#358: the verdict now comes from the layer that CONSUMES the value.

        It used to be emitted by `InferenceOrchestrator.__init__`, which runs before the
        post-construction override that file documents as supported — so a driver-set clamp logged
        "evolves freely" and then ran clamped.
        """
        import logging

        built = HydraNetConfig(**_with(cfg, freeze_recurrent="cell")).model_dump()
        with caplog.at_level(logging.INFO):
            _build_inference(built, freeze_recurrent="cell")
        assert "CLAMPED" in caplog.text and "'cell'" in caplog.text, (
            f"a clamped run did not announce the clamp; log was: {caplog.text!r}"
        )

    def test_a_driver_override_is_still_announced(self, cfg, caplog):
        """The defect itself: config omits the key, a driver sets it, the run IS clamped.

        This is the exact sequence `roster_arm_entry.py` performs — construct the orchestrator from
        a config without the key, then assign the attribute before inference is built.
        """
        import logging

        bare = HydraNetConfig(**cfg).model_dump()
        bare.pop("freeze_recurrent", None)
        with caplog.at_level(logging.INFO):
            _build_inference(bare, freeze_recurrent="cell")
        assert "CLAMPED" in caplog.text, (
            "a driver-overridden clamp was not recorded as clamped — the provenance mechanism "
            f"still lies about what ran. Log was: {caplog.text!r}"
        )
        assert "evolves freely" not in caplog.text

    def test_an_unclamped_run_also_says_so(self, cfg, caplog):
        """Anti-vacuity: silence must not be the signal for either state."""
        import logging

        built = HydraNetConfig(**cfg).model_dump()
        built.pop("freeze_recurrent", None)
        with caplog.at_level(logging.INFO):
            _build_inference(built, freeze_recurrent=None)
        assert "evolves freely" in caplog.text
        assert "CLAMPED" not in caplog.text

    def test_the_orchestrator_does_not_announce_a_verdict_it_cannot_know(self, cfg, caplog):
        """S5/#358: it may say what config asked for; it may not say what will be in effect."""
        import logging

        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        # With the clamp REQUESTED. The first draft popped the key, so the pre-PR premature
        # `CLAMPED` block — the S5 regression itself — could be re-added and pass (#372 review).
        built = HydraNetConfig(**_with(cfg, freeze_recurrent="cell")).model_dump()
        orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
        with caplog.at_level(logging.INFO):
            InferenceOrchestrator.__init__(
                orch, config=built, model=_DummyModel(), device=_cpu(), visualizer=None
            )
        assert "freeze_recurrent='cell' from config" in caplog.text, caplog.text
        assert "CLAMPED" not in caplog.text and "evolves freely" not in caplog.text, (
            "the orchestrator announced a verdict before the override it documents as supported"
        )
