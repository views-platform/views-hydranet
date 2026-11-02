"""ADR-078 §2.1: the field that selects which horizons a run EMITS.

⛔ This file tests the configuration contract only. Whether the field reaches the emitted output
is a different question, asked in the story that honours it (#505) — and until that lands this
field is an inert knob, which is the defect pattern this repository registers as C-324. A test
that the configuration validates is not evidence that the flag does anything.

What *is* testable here is that the field is **declared**: an undeclared key rides on
`extra="allow"`, so a build that predates it would swallow the selection and emit everything while
looking clean — the W-22 trap, which has cost one confounded run already.
"""

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig


def _cfg(valid_config_dict, **overrides):
    cfg = dict(valid_config_dict)
    cfg.update(overrides)
    return cfg


# ─── The field exists and defaults to every horizon ──────────────────────────


def test_a_config_omitting_the_field_is_unchanged(valid_config_dict):
    """Default is all horizons, so no existing configuration is affected."""
    assert "emit_horizons" not in valid_config_dict
    assert HydraNetConfig(**valid_config_dict).emit_horizons is None


def test_the_field_is_declared_not_swallowed_by_extra_allow(valid_config_dict):
    """⭐ An undeclared key would be accepted and ignored, which is worse than being rejected.

    `extra="allow"` means a misspelling, or a build that predates this field, takes the selection
    silently and emits everything — a subset arm that ran as its own control and would read as a
    clean null (W-22). Declaration is what makes a pod preflight able to confirm the installed
    build supports it.
    """
    assert "emit_horizons" in HydraNetConfig.model_fields

    # The prefix is the only mechanically checkable half of ADR-078 §2.1; that the name reads as
    # *emission* rather than as a bare horizon setting is a review judgement, not a test.
    assert "emit_horizons".startswith("emit_")


def test_a_valid_subset_is_accepted_and_kept_as_declared(valid_config_dict):
    """The two horizons EXP-02's endpoints need (h1 and h36) are a legal selection."""
    cfg = _cfg(valid_config_dict, emit_horizons=[1, 36])
    assert HydraNetConfig(**cfg).emit_horizons == [1, 36]


# ─── Refusals, all at construction ───────────────────────────────────────────


@pytest.mark.parametrize("horizons", [[0, 1], [37], [1, 36, 37], [-1]])
def test_a_horizon_the_run_does_not_produce_is_refused(valid_config_dict, horizons):
    """*Red:* out of range is refused when the config is built, not discovered during analysis.

    The alternative leaves a missing directory as the only evidence, and that is
    indistinguishable from a partial failure.
    """
    cfg = _cfg(valid_config_dict, emit_horizons=horizons)
    with pytest.raises(ValidationError, match="does not produce"):
        HydraNetConfig(**cfg)


def test_the_refusal_names_the_offending_horizons_and_the_range(valid_config_dict):
    """A message that says only "invalid" sends the reader back to the source to find out why."""
    cfg = _cfg(valid_config_dict, emit_horizons=[1, 40])
    with pytest.raises(ValidationError) as exc:
        HydraNetConfig(**cfg)

    message = str(exc.value)
    assert "[40]" in message, "the offending horizon is not named"
    assert "1..36" in message, "the range the run does produce is not stated"


def test_an_empty_selection_is_refused(valid_config_dict):
    """Asking for a run that writes nothing is a mistake, not a configuration."""
    cfg = _cfg(valid_config_dict, emit_horizons=[])
    with pytest.raises(ValidationError, match="writes nothing"):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("horizons", [[36, 1], [1, 1, 2], [2, 1]])
def test_duplicates_and_descending_order_are_refused(valid_config_dict, horizons):
    """Both describe an emission the writer cannot carry out.

    Normalising them silently is the tempting alternative and the wrong one: the manifest would
    then record a selection the configuration does not contain (ADR-074).
    """
    cfg = _cfg(valid_config_dict, emit_horizons=horizons)
    with pytest.raises(ValidationError, match="ascending and without duplicates"):
        HydraNetConfig(**cfg)


# ─── What this field is NOT ──────────────────────────────────────────────────


def test_a_sparse_selection_is_legal_because_it_selects_output_not_computation(valid_config_dict):
    """⭐ ADR-078 §2.3: contiguity is `steps`' constraint, never this field's.

    `steps` is a count — only its length is read — so requiring it to be contiguous costs nothing.
    This field names the horizons an analysis reads, and those are routinely the first and the
    last. Refusing a gap here would reintroduce the very thing the field exists to provide.
    """
    cfg = _cfg(valid_config_dict, emit_horizons=[1, 18, 36])
    assert HydraNetConfig(**cfg).emit_horizons == [1, 18, 36]


# ─── `steps` is a count, not a selector (ADR-078 §2.3) ───────────────────────


def test_a_contiguous_step_list_validates_unchanged(valid_config_dict):
    """*Green:* every live configuration declares `1..36`, and must keep validating.

    Checked against views-models rather than assumed: all **134** declarations of `steps` across
    its model configs and sweeps are `range(1, 37)` or an exact equivalent. ✅ The trap this
    validator closes is latent, not firing, and no past result is affected.
    """
    assert valid_config_dict["steps"] == list(range(1, 37))
    assert HydraNetConfig(**valid_config_dict).steps == list(range(1, 37))


@pytest.mark.parametrize(
    "steps",
    [
        [1, 18, 36],  # the sparse set someone actually wants
        [2, 3, 4],  # contiguous but not from one
        [3, 2, 1],  # descending
        [1, 1, 2],  # duplicated
    ],
)
def test_a_step_list_that_is_not_a_contiguous_count_is_refused(valid_config_dict, steps):
    """*Red:* only the list's LENGTH is read, so a sparse list produces consecutive horizons.

    ⚠️ On an evaluation run pipeline-core builds its month-to-step mapping from the declared
    values and raises on a predicted month outside it, so a sparse list already fails there. On a
    forecasting run that check is not called. ⇒ The trap is real but narrower than it looks, and
    this validator closes the remaining half at construction.
    """
    cfg = _cfg(valid_config_dict, steps=steps, time_steps=len(steps))
    with pytest.raises(ValidationError, match="not contiguous and ascending from 1"):
        HydraNetConfig(**cfg)


def test_the_refusal_routes_the_reader_to_the_field_that_does_select(valid_config_dict):
    """⭐ Someone who wanted a sparse set should be routed, not merely blocked.

    They have a real need — the horizons an analysis reads are routinely the first and the last —
    and the field that serves it now exists. A refusal that does not name it sends them to read
    the source of a validator instead.
    """
    cfg = _cfg(valid_config_dict, steps=[1, 18, 36], time_steps=3)
    with pytest.raises(ValidationError) as exc:
        HydraNetConfig(**cfg)

    message = str(exc.value)
    assert "emit_horizons" in message, "the refusal does not name the field that does select"
    assert "[1, 2, 3]" in message, "the refusal does not show the contiguous list to declare"
    assert "COUNT" in message


def test_an_empty_step_list_is_refused(valid_config_dict):
    """A count of zero rollout steps has nothing to produce."""
    cfg = _cfg(valid_config_dict, steps=[], time_steps=0)
    with pytest.raises(ValidationError, match="declares no horizons"):
        HydraNetConfig(**cfg)
