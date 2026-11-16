"""The closed vocabulary of regression bodies (ADR-079). **Torch-free at import.**

One config field, `body`, names what trains the regression head, what activation that head applies,
and what the model emits forecasts from. Its value comes from the frozen mapping below and from
nowhere else: a combination that is not a key here cannot be written in a config, so there is no
pair of fields to cross-check against each other.

Why this module exists at all. Before ADR-079 the body was described by three fields — `loss_reg`,
`output_distribution` and `reg_activation` — and the objective that actually trained it was
*inferred* from the combination, by statement order inside `choose_loss`. The inference discarded
`loss_reg` whenever a distribution was named, silently, and 27 of the 37 live configs therefore
stated a loss they did not train (census 2026-11-13). See `legacy.py` for that rule, preserved
exactly and quarantined.

Adding a row is a deliberate act with an owner and a stated reason. It is not a config edit.
Membership is *currently in use*, not *ever tried*: other combinations have been trained here in
the past — `zinb` among them — and have no row, so no config can name them.
They remain loadable from a saved artifact, which never passes through the schema.

Torch-free, like `distributions/registry.py` and for the same reason: `config_initializer` imports
this, and the schema layer must not pull torch (CRP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from views_hydranet.distributions.registry import FAMILY_N_PARAMS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BodySpec:
    """What one declared `body` value means. Exactly one of `loss`/`family` is set.

    Attributes
    ----------
    activation : str
        The activation named by the value's middle segment — `softplus` on every declared row.
        It is what the head applies to its raw output **when the body has no emit family**. When
        the body has one, the family supplies the activation itself and this field is recorded but
        not consulted (`HydraBNrecurrentUnet_06_LSTM4.py:118-119`). ⚠️ On a family row the segment
        is therefore approximate: the NB and truncated-NB families do apply softplus to both
        channels, but the mixture applies `sigmoid` to its weight and `mu1 + softplus(delta)` to
        its second mean. ADR-079 §4 records that as the accepted price of the long spelling.
    loss : str | None
        A key of ``LOSS_REG_REGISTRY``, for a body trained by a named loss, not a likelihood.
    family : str | None
        A key of ``DISTRIBUTION_REGISTRY``, for a body trained by that family's likelihood and
        emitting draws from it.
    """

    activation: str
    loss: str | None = None
    family: str | None = None

    def __post_init__(self) -> None:
        if (self.loss is None) == (self.family is None):
            err_msg = (
                f"a body states exactly one objective, got loss={self.loss!r} and "
                f"family={self.family!r}. A loss-trained body names a LOSS_REG_REGISTRY key; a "
                "likelihood-trained body names a distribution family."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    @property
    def width(self) -> int:
        """Channels the head emits per target — the emit family's parameter count, or 1.

        ⛔ Derived, never declared. ADR-079 §2 refuses a config field for the width because the
        number is already fixed by the distribution, and two places recording one number can
        disagree — which is the defect this ADR removes, relocated to a new field.
        """
        return 1 if self.family is None else FAMILY_N_PARAMS[self.family]

    @property
    def emits_draws(self) -> bool:
        """Whether forecasts are drawn from a distribution, or the one number *is* the forecast."""
        return self.family is not None

    @property
    def gloss(self) -> str:
        """One line for the refusal message. Derived, so it cannot misdescribe the fields."""
        trains = f"the {self.family} likelihood" if self.family else f"{self.loss} loss"
        emits = f"{self.family} draws" if self.family else "a point, no draws"
        owner = " (applied by the family)" if self.family else ""
        return f"width {self.width}, {self.activation}{owner}, {trains}, emits {emits}"


#: The declared vocabulary: exactly the combinations live in the fleet today (ADR-079 §2).
BODIES: dict[str, BodySpec] = {
    "mse_softplus_point": BodySpec(activation="softplus", loss="mse"),
    "listwise_softplus_point": BodySpec(activation="softplus", loss="listwise_level"),
    "nll_softplus_nb": BodySpec(activation="softplus", family="nb"),
    "nll_softplus_truncated_nb": BodySpec(activation="softplus", family="truncated_nb"),
    "nll_softplus_mixture_nb": BodySpec(activation="softplus", family="mixture_nb"),
}


def declared_bodies() -> str:
    """The declared values with their glosses, one per line — built by iterating ``BODIES``.

    ⛔ Built, never written out. #602 requires the printed set to equal the set that runs, and the
    only way to guarantee that rather than test it is to have one set.
    """
    return "\n".join(f"  {name:<28}{spec.gloss}" for name, spec in BODIES.items())


def resolve_body(value: str) -> BodySpec:
    """The spec for a declared ``body`` value; fail-loud on anything else (#602)."""
    spec = BODIES.get(value)
    if spec is None:
        # ADR-008: log before raising.
        err_msg = (
            f"body={value!r} is not a declared body. The declared bodies are:\n"
            f"{declared_bodies()}\n"
            "A combination that is not listed cannot be written in a config. Adding a row is a "
            "deliberate act with an owner and a stated reason (ADR-079 §2), not a config edit."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    return spec
