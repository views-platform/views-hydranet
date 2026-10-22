"""ADR-073: the family a model is EMITTED from, when it differs from the one it was TRAINED on.

Epic #428. The capability exists so the gate can own occurrence and the body only magnitude —
measured at −5.94% composed CRPS and an emitted positive fraction of 0.0037 against an observed
0.0034, with nothing retrained (`reports/2026-10-21_emit_time_split_prereg.md`).

⛔ The test that matters in this file is not any of the config ones. It is
`test_the_flag_reaches_the_emitted_cube`: a setting that validates and does nothing is C-324's
inert-knob signature, and `views-models#578` records the same lesson from the gate-temperature
track — *"prove the temperature reaches the emitted cube, not just that the config validates."*
A green suite is not evidence here; the test that would fail if the seam were reverted is.
"""

import numpy as np
import pytest
import torch
from torch import nn

from views_hydranet.utils.config_initializer import HydraNetConfig


def _cfg(valid_config_dict, **over):
    """A valid config with an `nb` family and a gate composition, plus overrides.

    `forecast_composition` is set explicitly because the shipped fixture carries `self_zeroed`,
    which is itself illegal on a non-self-zeroed family (ADR-069) — leaving it would let that
    unrelated rule decide every outcome in this file instead of the one under test.
    """
    cfg = dict(valid_config_dict)
    cfg.update(output_distribution="nb", forecast_composition="soft_gate")
    cfg.update(over)
    return cfg


# --------------------------------------------------------------- the default
def test_the_default_is_emit_as_trained(valid_config_dict):
    """⭐ Every roster model depends on this: unset means byte-identical to today."""
    assert HydraNetConfig(**valid_config_dict).emit_distribution is None


def test_an_existing_config_is_unaffected(valid_config_dict):
    """The field must not change validation for a config that never mentions it."""
    assert "emit_distribution" not in valid_config_dict
    HydraNetConfig(**valid_config_dict)  # must not raise


# ------------------------------------------------------- the three guards
def test_the_only_admissible_pair_today_is_accepted(valid_config_dict):
    """`nb` trained, `truncated_nb` emitted — the combination the epic exists for."""
    cfg = _cfg(valid_config_dict, emit_distribution="truncated_nb")
    assert HydraNetConfig(**cfg).emit_distribution == "truncated_nb"


def test_a_family_with_a_different_parameter_count_is_refused(valid_config_dict):
    """⭐ THE STRUCTURAL GUARD, and it is what keeps ADR-073's surface narrow.

    The head's channels were trained to mean the TRAINED family's parameters. `zinb` emits 3
    (mu, theta, pi) against `nb`'s 2, so reading an `nb` head as `zinb` would reinterpret channels
    as something they were never trained to be — silently, since the slice would still have a
    shape. Refused at config time, before any pod is paid for.
    """
    cfg = _cfg(valid_config_dict, emit_distribution="zinb")
    with pytest.raises(ValueError, match="parameters per cell"):
        HydraNetConfig(**cfg)


def test_the_error_names_both_parameter_counts(valid_config_dict):
    """A guard that refuses without saying WHICH layouts disagree sends the reader to the wrong
    line — the C-242 lesson, already learned once in this validator."""
    cfg = _cfg(valid_config_dict, emit_distribution="mixture_nb")
    with pytest.raises(ValueError) as exc:
        HydraNetConfig(**cfg)
    msg = str(exc.value)
    assert "5" in msg and "2" in msg
    assert "mixture_nb" in msg and "nb" in msg


def test_an_unregistered_family_is_refused(valid_config_dict):
    cfg = _cfg(valid_config_dict, emit_distribution="not_a_family")
    with pytest.raises(ValueError, match="not a registered distribution family"):
        HydraNetConfig(**cfg)


def test_setting_it_equal_to_the_trained_family_is_refused(valid_config_dict):
    """⭐ C-324: a setting that reads as an intervention and is none.

    `emit_distribution='nb'` on an `nb` model changes nothing, but it appears in a config diff and
    in a provenance record as though an arm were running. Refused for the same reason
    `reject_inert_struct_penalty` exists — an inert knob produces a clean, believable null.
    """
    cfg = _cfg(valid_config_dict, emit_distribution="nb")
    with pytest.raises(ValueError, match="equals"):
        HydraNetConfig(**cfg)


def test_the_inert_refusal_explains_that_None_already_means_that(valid_config_dict):
    """The message must say what to do instead, or the next person sets it again."""
    cfg = _cfg(valid_config_dict, emit_distribution="nb")
    with pytest.raises(ValueError) as exc:
        HydraNetConfig(**cfg)
    assert "None" in str(exc.value)


# ------------------------------------- ADR-069's rules follow the EMITTED family
@pytest.mark.parametrize("emit", [None, "truncated_nb"])
@pytest.mark.parametrize("comp", ["soft_gate", "threshold_gate", "self_zeroed"])
def test_composition_outcomes_are_unchanged_by_setting_emit_distribution(
    valid_config_dict, emit, comp
):
    """⭐ The byte-identical guarantee, at the composition seam.

    ⛔ Today **no admissible pair changes a composition outcome**, and that is worth stating
    plainly rather than letting this test imply otherwise: the only pairs the parameter-count
    guard admits are `nb <-> truncated_nb`, and neither is self-zeroed. So re-pointing ADR-069's
    rules at the emitted family is a correctness fix for a family that does not exist yet, not a
    live bug fix — and this test pins that it changed nothing meanwhile.
    """
    cfg = dict(valid_config_dict)
    cfg.update(output_distribution="nb", forecast_composition=comp)
    if emit is not None:
        cfg["emit_distribution"] = emit

    def _accepts(c):
        try:
            HydraNetConfig(**c)
            return True
        except ValueError:
            return False

    baseline = dict(cfg)
    baseline.pop("emit_distribution", None)
    assert _accepts(cfg) == _accepts(baseline)


def test_the_self_zeroed_rule_reads_the_EMITTED_family_not_the_trained_one(
    valid_config_dict, monkeypatch
):
    """⭐ THE POINT OF THIS CHANGE, and it needs a monkeypatch to be testable at all.

    `self_zeroed` is the property ADR-069's rules turn on, and no family pair the parameter-count
    guard admits differs on it — so the re-point cannot be observed with the families that exist.
    Rather than leave it unproven, declare `truncated_nb` self-zeroed for the duration of this
    test and check which family the rule consults.

    Gating a self-zeroed body is refused because π and a gate would double-count the zeros. If the
    validator read `output_distribution` it would see plain `nb`, allow the gate, and let a
    double-counted composition through.
    """
    import views_hydranet.distributions.registry as reg

    monkeypatch.setattr(reg, "SELF_ZEROED_FAMILIES", frozenset({"zinb", "truncated_nb"}))

    # emitted = truncated_nb (self-zeroed under the patch) + a gate => must be REFUSED
    gated = _cfg(valid_config_dict, emit_distribution="truncated_nb")
    with pytest.raises(ValueError, match="double-count"):
        HydraNetConfig(**gated)

    # the same trained family with NO emit override is plain nb => the gate is fine
    HydraNetConfig(**_cfg(valid_config_dict))


def test_a_refusal_names_the_EMITTED_family_when_the_two_differ(valid_config_dict, monkeypatch):
    """A message blaming `output_distribution` for a rule applied to the emitted family sends the
    reader to the wrong config line. C-242 is already in this validator for that reason."""
    import views_hydranet.distributions.registry as reg

    monkeypatch.setattr(reg, "SELF_ZEROED_FAMILIES", frozenset({"zinb", "truncated_nb"}))
    with pytest.raises(ValueError) as exc:
        HydraNetConfig(**_cfg(valid_config_dict, emit_distribution="truncated_nb"))
    msg = str(exc.value)
    assert "emit_distribution='truncated_nb'" in msg
    assert "output_distribution='truncated_nb'" not in msg


# -------------------------------------------- the seam, for callers that skip the schema
class _MockModel(nn.Module):
    """Minimal stand-in: `HydraNetInference` reads `output_distribution` off the MODEL, and
    type-checks that the model is an `nn.Module` — same shape as `test_composition_
    characterization.py`'s mock, reused rather than reinvented."""

    def __init__(self, output_distribution):
        super().__init__()
        self.output_distribution = output_distribution

    def forward(self, x, h):  # pragma: no cover - never called here
        raise NotImplementedError


def _inference(trained, emit=None):
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    cfg = {"regression_targets": ["lr_ged_sb"]}
    if emit is not None:
        cfg["emit_distribution"] = emit
    return HydraNetInference(_MockModel(trained), cfg, device="cpu")


def test_the_seam_resolves_the_EMITTED_family():
    """⭐ The one line this epic turns on: `_family` must be the family we DRAW FROM.

    Every other use of `self._family` is already emit-side — the magnitude emit, the feedback mean
    and draw, and the emitted cube — so if this resolves to the trained family the flag is inert
    everywhere, which is exactly the C-324 failure the axis exists to avoid.
    """
    from views_hydranet.distributions.truncated_negative_binomial import TruncatedNBFamily

    inf = _inference("nb", emit="truncated_nb")
    assert inf._emit_distribution == "truncated_nb"
    assert isinstance(inf._family, TruncatedNBFamily)
    # the TRAINED name must not move: the legacy branches key on the head's layout
    assert inf.output_distribution == "nb"


def test_unset_resolves_the_trained_family():
    from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily

    inf = _inference("nb")
    assert inf._emit_distribution == "nb"
    assert isinstance(inf._family, NegativeBinomialFamily)


def test_a_RAW_DICT_bypassing_pydantic_is_still_refused():
    """⛔ The guard that exists because the schema is not on this path.

    `get_hp_config()` returns a plain dict and every measurement driver builds one, so a caller
    here may never have run `validate_emit_distribution`. With `extra="allow"` upstream, an
    unchecked value would be neither rejected nor honoured — silently ignored, the precise
    failure this axis exists to prevent. `choose_loss` re-checks `body_nll_denominator` for the
    same reason, after a review found exactly that hole.
    """
    with pytest.raises(ValueError, match="parameters per cell"):
        _inference("nb", emit="zinb")


def test_the_raw_dict_refusal_names_both_families_and_both_counts():
    with pytest.raises(ValueError) as exc:
        _inference("nb", emit="mixture_nb")
    msg = str(exc.value)
    assert "mixture_nb" in msg and "'nb'" in msg
    assert "5" in msg and "2" in msg


def test_the_feedback_draw_follows_the_EMITTED_family():
    """⛔ PROVISIONAL, pinned deliberately — see #433, which is blocked on a vehicle.

    The autoregressive feedback must draw from the family the model EMITS, which is C-234's
    principle ("feeding back a body the model never emitted makes h>=2 a rollout over a history
    that does not exist") applied to a second emit-time axis.

    ⭐ THIS TEST WAS DECORATIVE AND A MUTATION CAUGHT IT. The first version asserted
    `isinstance(inf._family, TruncatedNBFamily)` and the resolved `rollout_feedback` — neither of
    which touches `_sample_feedback`. Replacing `fam = self._family` with the trained family in
    that method left the whole file green (mutation M6, SURVIVED). It now drives the real feedback
    draw and asserts the truncation mechanism in it, so flipping the wiring is a failing test and
    a conversation rather than a silent change.
    """
    inf_trained = _inference("nb")
    inf_emitted = _inference("nb", emit="truncated_nb")
    params, gate = _params_and_gate()
    # `_sample_feedback` wants reg `[T, n_reg*n_params, H, W]` and prob `[T, n_cls, H, W]`
    prob = torch.ones(params.shape[0], 1, *params.shape[-2:])

    def _draw(inf):
        return inf._sample_feedback(params, prob, torch.Generator().manual_seed(0))

    plain = _draw(inf_trained)
    truncated = _draw(inf_emitted)
    assert plain.shape == truncated.shape
    # the fixture must bite, or the assertion below proves nothing
    assert (plain == 0).any(), "fixture is useless — the plain NB feedback drew no zeros"
    # ⭐ a zero-truncated body cannot feed back a zero where the gate fired
    assert not (truncated == 0).any(), (
        "the fed-back draw still contains zeros, so the emitted family did not reach "
        "_sample_feedback — the rollout would run on a history the model never emitted"
    )


# ============================ ⭐ the test the epic turns on ============================
def _params_and_gate(mu=0.3, theta=1.0, t=2, h=8, w=8):
    """Activated `[T, n_reg*n_params, H, W]` params, target-major, plus an all-firing gate.

    `mu = 0.3` with `theta = 1.0` puts most of an NB's mass at zero — `P(Y=0) = (theta/(theta+mu))
    ** theta ≈ 0.77` — which is what makes the fixture able to discriminate. The gate is 1.0
    everywhere so `soft_gate`'s Bernoulli mask is all ones and the composed cube IS the body's
    draws: the comparison then isolates the emission family, with the gate contributing nothing.
    """
    params = torch.empty(t, 2, h, w)
    params[:, 0] = mu
    params[:, 1] = theta
    gate = torch.ones(t, h, w, 1)
    return params, gate


def _emit_cube(inference, k=8):
    """Draw the emitted cube through the PRODUCTION seam, with the resolved family."""
    from views_hydranet.distributions.sampling import to_cube_samples

    params, gate = _params_and_gate()
    return to_cube_samples(
        params,
        inference._family,
        k=k,
        generator=torch.Generator().manual_seed(0),
        n_reg=1,
        gate=gate,
        composition="soft_gate",
    )


def test_the_flag_reaches_the_emitted_cube():
    """⭐⭐ THE TEST THIS EPIC EXISTS FOR, and it is the one that would fail on a revert.

    ⛔ It asserts the truncation MECHANISM, not that two arrays differ. "The cubes are not equal"
    would also pass for a seeding bug, a shape change, or a different generator — none of which
    would mean the emission rule arrived. What must be true is specific: a zero-truncated body
    cannot emit a zero, so wherever the gate fired there must be NO zero draw.

    `views-models#578` records the same lesson from the gate-temperature track — proving a config
    validates is not proving it reaches the cube — and C-324 is the register entry for a setting
    that validates and does nothing.
    """
    trained = _emit_cube(_inference("nb"))
    emitted = _emit_cube(_inference("nb", emit="truncated_nb"))

    assert trained.shape == emitted.shape
    # the fixture must actually bite: a plain NB at mu=0.3 draws zeros often
    assert (trained == 0).any(), "fixture is useless — the NB body drew no zeros to remove"
    # ⭐ and the truncated emission must remove every one of them
    assert not (emitted == 0).any(), (
        "the emitted cube still contains zeros, so the truncated family did not reach it"
    )


def test_the_emitted_cube_is_byte_identical_when_the_flag_is_unset():
    """⭐ The guarantee every roster model depends on: unset changes nothing, bit for bit."""
    a = _emit_cube(_inference("nb"))
    b = _emit_cube(_inference("nb"))
    assert np.array_equal(a, b), "the fixture is not deterministic; the comparison below is void"

    from views_hydranet.distributions import resolve_family
    from views_hydranet.distributions.sampling import to_cube_samples

    params, gate = _params_and_gate()
    before_the_flag_existed = to_cube_samples(
        params,
        resolve_family("nb"),
        k=8,
        generator=torch.Generator().manual_seed(0),
        n_reg=1,
        gate=gate,
        composition="soft_gate",
    )
    assert np.array_equal(a, before_the_flag_existed)


def test_the_truncated_emission_raises_the_emitted_magnitude():
    """The direction the epic measured: truncation moves mass OFF zero, so the emitted field is
    larger. −5.94% composed CRPS and a positive fraction of 0.0037 against 0.0034 observed came
    from this, and a sign flip here would mean the arithmetic and the engine disagree."""
    trained = _emit_cube(_inference("nb"))
    emitted = _emit_cube(_inference("nb", emit="truncated_nb"))
    assert emitted.mean() > trained.mean()
