"""The pre-ADR-079 three-field rule, preserved exactly and quarantined. **Torch-free at import.**

Before ADR-079 the body was described by `loss_reg`, `output_distribution` and `reg_activation`,
and what actually trained it was decided by statement order inside `choose_loss`: a registered
distribution family won, and `loss_reg` was then never read. That rule is reproduced here, once, as
a total function.

⛔ This is a door, not a path. A config can no longer spell a body this way — `HydraNetConfig`
refuses all three keys. What still arrives in the old spelling is a saved artifact's sidecar and a
driver that builds a raw dict without passing through the schema. Both are legitimate and both are
why the six orphan distribution values stay loadable.

The function is pinned by `tests/test_body_characterization.py`, which was committed before any of
ADR-079 moved, with pairs whose stated loss *disagrees* with what trains. Delete this file whole on
the day the last pre-ADR-079 artifact is retired.
"""

from __future__ import annotations

from views_hydranet.body.vocabulary import BodySpec
from views_hydranet.distributions.registry import family_names


def legacy_body_spec(
    loss_reg: str,
    output_distribution: str = "standard",
    reg_activation: str | None = None,
) -> BodySpec:
    """The body a pre-ADR-079 triple describes — including the six values with no declared row.

    The precedence, in one line: if `output_distribution` names a registered family, that family's
    likelihood trains the body and `loss_reg` is irrelevant. Otherwise `loss_reg` trains it.

    The activation mirrors `HydraBNrecurrentUnet_06_LSTM4.py:118-127`: an explicit
    `softplus`/`relu` wins, and the default keys off the `hurdle` prefix — which is why `dense_nb`
    trains relu while the three hurdle bodies train softplus. A family head is the one case where
    the field is not consulted at all, because the family applies its own map; the declared value
    for such a body names `softplus`, so that is what this returns.
    """
    is_family = output_distribution in family_names()
    if is_family:
        return BodySpec(activation="softplus", family=output_distribution)
    if reg_activation in ("softplus", "relu"):
        activation = reg_activation
    else:
        activation = "softplus" if output_distribution.startswith("hurdle") else "relu"
    return BodySpec(activation=activation, loss=loss_reg)
