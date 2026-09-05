"""Tests for #311's input-noise augmentation (S2, #314).

Design derived in `reports/2026-09-04_input_noise_dossier/02_design.md` from S1's measurement: the
model's free-running failure is near-total SILENCING (FN 0.9959 vs FP 0.000027 at h18), so the
augmentation removes events rather than adding noise.

Two properties are load-bearing and get the hardest tests: **off is genuinely off** (proved by
making the helper explode if the off path ever reaches it, not by comparing numbers), and
**geometry is never noised** — which no config in this fleet can exercise, so it is tested against
a synthetic one (**C-309**: a guard whose firing case has never been observed is not a guard).
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.train.training_engine import (  # noqa: E402
    _apply_input_noise,
    _noisable_channels,
    _process_sequence,
    _SequenceIndices,
)

_H, _W, _T = 3, 3, 6
_FEATS = ["reg0", "cls0", "feat0"]
_CFG = {
    "regression_targets": ["reg0"],
    "classification_targets": ["cls0"],
    "features": ["feat0"],
    "static_channels": [],
}


# ---------------------------------------------------------------------------
# _noisable_channels — the geometry guard, on a SYNTHETIC config (C-309)
# ---------------------------------------------------------------------------


def test_every_channel_is_noisable_when_there_are_no_statics():
    assert _noisable_channels({"features": ["a", "b", "c"], "static_channels": []}) == [0, 1, 2]


def test_a_static_INSIDE_features_is_excluded():
    """No arm in this fleet has statics (measured 2026-09-04: `static_channels` is empty), so this
    branch cannot be reached by any real config. Without this synthetic case the guard would be
    code that never runs — and geometry is "always the true values, never sampled"."""
    cfg = {"features": ["lr_sb", "row", "lr_ns", "col"], "static_channels": ["row", "col"]}
    assert _noisable_channels(cfg) == [0, 2]


def test_missing_keys_do_not_crash():
    assert _noisable_channels({}) == []


# ---------------------------------------------------------------------------
# _apply_input_noise — it only REMOVES, and it accumulates
# ---------------------------------------------------------------------------


def _ones(c=2):
    return torch.ones(1, c, 4, 4)


def test_dropout_of_zero_probability_keeps_everything():
    x = _ones()
    keep = torch.ones_like(x)
    out, new_keep = _apply_input_noise(x, keep, 0.0, [0, 1])
    assert torch.equal(out, x) and torch.equal(new_keep, keep)


def test_dropout_of_one_silences_everything():
    x = _ones()
    out, keep = _apply_input_noise(x, torch.ones_like(x), 1.0, [0, 1])
    assert float(out.abs().sum()) == 0.0
    assert float(keep.sum()) == 0.0


def test_it_only_ever_REMOVES_never_adds():
    """M45: AP loss scales with how much the model fires. A corruption that can manufacture
    occurrence would be aimed at the lever four other interventions already died on."""
    torch.manual_seed(0)
    x = torch.zeros(1, 2, 8, 8)
    x[0, 0, 1, 1] = 5.0
    out, _ = _apply_input_noise(x, torch.ones_like(x), 0.5, [0, 1])
    assert float(out[x == 0].abs().sum()) == 0.0, "a zero cell became non-zero"
    assert bool((out <= x + 1e-12).all()), "a value grew"


def test_a_dropped_cell_STAYS_dropped():
    """Accumulation is the whole point — rollout error accumulates, and the paper's own ablation
    found random-walk noise beat i.i.d. A non-accumulating mask would model the wrong process."""
    x = _ones()
    keep = torch.ones_like(x)
    keep[0, 0, 2, 2] = 0.0  # already dropped on an earlier step
    out, new_keep = _apply_input_noise(x, keep, 0.0, [0, 1])  # p=0: nothing NEW is dropped
    assert float(out[0, 0, 2, 2]) == 0.0
    assert float(new_keep[0, 0, 2, 2]) == 0.0


def test_channels_not_listed_are_left_completely_alone():
    x = _ones(c=3)
    out, keep = _apply_input_noise(x, torch.ones_like(x), 1.0, [0])
    assert float(out[:, 0].sum()) == 0.0
    assert torch.equal(out[:, 1:], x[:, 1:]), "an unlisted channel was noised"
    assert float(keep[:, 1:].min()) == 1.0


def test_an_empty_channel_list_is_a_no_op():
    x = _ones()
    out, keep = _apply_input_noise(x, torch.ones_like(x), 1.0, [])
    assert torch.equal(out, x)


def test_the_drop_rate_is_approximately_the_requested_one():
    """Potency with a number: the knob must move the measured quantity to roughly where it is set,
    not merely move it."""
    torch.manual_seed(3)
    x = torch.ones(1, 1, 700, 700)  # n=490k; sd of the estimate ~0.00058
    out, _ = _apply_input_noise(x, torch.ones_like(x), 0.204, [0])
    dropped = float((out == 0).float().mean())
    # abs=0.0025 is ~4.3 sd at n=490k. The history matters: abs=0.01 on n=40k tolerated a +-0.008
    # systematic bias (~4% of a pre-registered 0.204); round 2 tightened it to 0.004 on n=160k and
    # claimed that closed it; round 3 showed a +0.003 bias still passing, caught only by an
    # unseeded 32-element test with ~9% power, i.e. 1 run in 8. This bound is ~1.2% of the value.
    assert dropped == pytest.approx(0.204, abs=0.0025)


# ---------------------------------------------------------------------------
# The call site — off must be genuinely off
# ---------------------------------------------------------------------------


class _Model(torch.nn.Module):
    def __init__(self, n_reg, n_cls):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, n_reg + n_cls, kernel_size=1)
        self.n_reg = n_reg

    def forward(self, x, hidden):
        o = self.conv(x[:, :1])
        return types.SimpleNamespace(
            reg=o[:, : self.n_reg], cls=o[:, self.n_reg :], reg_latent=None, h_next=hidden
        )


def _run(dropout, monkeypatch=None, **kw):
    torch.manual_seed(7)
    idx = _SequenceIndices(_FEATS, _CFG)
    model = _Model(idx.n_reg, idx.n_cls)
    res = _process_sequence(
        train_tensor=torch.rand(1, _T, len(_FEATS), _H, _W),
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
        input_noise_dropout=dropout,
        input_noise_segment=kw.get("segment", 3),
        input_noise_channels=kw.get("channels", [0]),
    )
    return float(res["total"])


def test_OFF_never_reaches_the_noise_helper_at_all(monkeypatch):
    """Byte-identity proved by construction rather than by comparing numbers: if the default path
    ever calls the helper, this explodes. A numeric comparison could pass while the helper ran and
    happened to drop nothing."""
    import views_hydranet.train.training_engine as te

    def _boom(*a, **k):
        raise AssertionError("the noise helper ran with input_noise_dropout=None")

    monkeypatch.setattr(te, "_apply_input_noise", _boom)
    _run(None)  # must not raise


def test_ON_changes_the_loss():
    """Potency at the call site (C-324): the knob must act on the path production takes, not only
    in the helper's unit tests."""
    off, on = _run(None), _run(0.9)
    assert off != on, "the flag changed nothing at the call site — the C-324 inert signature"


def test_the_same_dropout_is_reproducible_under_the_same_seed():
    assert _run(0.5) == _run(0.5)


# ---------------------------------------------------------------------------
# The config field. Added after the 2026-09-04 audit found SEVEN survivors here:
# nothing instantiated HydraNetConfig with this field at all, so the bounds could
# be removed, inverted to accept negatives, or the default flipped to ON for the
# whole fleet — all with 1973 tests green.
# ---------------------------------------------------------------------------


def test_the_field_exists_under_exactly_this_name():
    """Survivor CI-05: renaming the field kept the field COUNT identical, so the only thing
    watching it (the CIC drift test) stayed green while every config silently lost the knob."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    assert "input_noise_dropout" in HydraNetConfig.model_fields


def test_the_default_is_OFF():
    """Survivors CI-03/CI-07: a default of 0.204 (or a non-optional 0.5) turns the noise ON for
    every arm in the fleet, and nothing failed."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    assert HydraNetConfig.model_fields["input_noise_dropout"].default is None


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1, -1.0])
def test_out_of_range_dropouts_are_REJECTED(bad, valid_config_dict):
    """Survivors CI-01/CI-02/CI-04/CI-08: the bounds could be removed outright or widened to accept
    negatives. `0.0` matters most — the comment beside the field says `gt` rather than `ge` was
    chosen because a 0.0 dropout is a no-op indistinguishable from off, which is the C-324
    inert-knob signature that cost 276 min of GPU on #308. That reasoning had no guard."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        HydraNetConfig(**{**valid_config_dict, "input_noise_dropout": bad})


def test_an_in_range_dropout_is_accepted(valid_config_dict):
    """The mirror: rejecting bad values must not reject good ones."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = HydraNetConfig(**{**valid_config_dict, "input_noise_dropout": 0.204})
    assert cfg.input_noise_dropout == pytest.approx(0.204)


def test_the_helper_returns_a_NEW_mask_and_does_not_mutate_the_callers(_=None):
    """Survivor AN-09: `keep.clone()` → `keep` was invisible because the single call site rebinds
    the result immediately — but the helper then mutates its caller's tensor in place, and any
    second caller would get silent corruption."""
    x = _ones()
    keep = torch.ones_like(x)
    before = keep.clone()
    _apply_input_noise(x, keep, 1.0, [0, 1])
    assert torch.equal(keep, before), "the helper mutated the caller's mask in place"


def test_the_rate_SURVIVES_the_config_to_dict_handshake(valid_config_dict):
    """The headline finding of the third audit, and the one that sits on the path a real run takes.

    `ConfigInitializer.get_config()` is *"the single Handshake point for the whole pipeline"* — it
    validates and then returns `model_dump()`. Adding `exclude=True` to the field makes it vanish
    from that dict: the field still exists, its default is still `None`, its bounds still reject
    0.0/1.0/−0.1, so **all three config tests pass** — and `train()`'s
    `config.get("input_noise_dropout")` returns `None`, so **no real run is ever noised**.

    That is the C-324 inert-knob signature, reached through the one layer none of the four test
    files exercised: the config tests read `model_fields`, and the seam and wiring tests use raw
    dicts that bypass pydantic entirely. Nothing asserted on the resolved dict.
    """
    from views_hydranet.utils.config_initializer import ConfigInitializer

    resolved = ConfigInitializer({**valid_config_dict, "input_noise_dropout": 0.204}).get_config()
    assert "input_noise_dropout" in resolved, (
        "the field did not survive into the resolved config; every run would be un-noised"
    )
    assert resolved["input_noise_dropout"] == pytest.approx(0.204)
    # And the off case must resolve to a value `train()` reads as off, not to a missing key.
    off = ConfigInitializer(dict(valid_config_dict)).get_config()
    assert off.get("input_noise_dropout") is None


def test_a_TYPO_in_the_config_key_is_not_silently_accepted_as_the_real_one(valid_config_dict):
    """`HydraNetConfig` sets `extra="allow"`, so `input_noise_droput: 0.204` validates cleanly
    and the arm trains with the augmentation OFF and no warning — the same inert signature,
    reached by a plausible authoring slip.

    This is NOT fixed here: changing `extra` is repo-wide and out of this epic's scope. The test
    pins the behaviour so it is visible rather than surprising, and the mitigation that does catch
    it is the S4 potency gate, which runs on the arm's own config and aborts on an inert knob.
    """
    from views_hydranet.utils.config_initializer import ConfigInitializer

    resolved = ConfigInitializer({**valid_config_dict, "input_noise_droput": 0.204}).get_config()
    assert resolved.get("input_noise_dropout") is None, (
        "a typo'd key was somehow bound to the real field"
    )
    assert resolved.get("input_noise_droput") == 0.204, (
        "extra='allow' no longer keeps unknown keys — if that changed, the typo now fails loud "
        "and this test should become an expectation of that, not of silent acceptance"
    )
