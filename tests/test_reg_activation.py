"""The regression-head activation: softplus only (ADR-079 Rule 1, amended 2026-11-20, #632).

`reg_activation` stays as a setting so that a second activation can be added later. Today the only
implemented one is softplus; a model built without the setting gets softplus, and any other value
is refused by the model (and by the config schema, #650). ReLU was the default until 2026-11-20.
"""

import pytest
import torch.nn.functional as F

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.config_initializer import IMPLEMENTED_REG_ACTIVATIONS


def _act(output_distribution, reg_activation):
    m = HydraBNUNet06_LSTM4(
        3,
        8,
        1,
        0.0,
        output_distribution=output_distribution,
        reg_activation=reg_activation,
    )
    return m._reg_activation


def test_the_implemented_activations_are_softplus_alone():
    """A literal, so adding an activation is a deliberate change here, not a side effect."""
    assert IMPLEMENTED_REG_ACTIVATIONS == ("softplus",)


def test_default_standard_is_softplus():
    assert _act("standard", None) is F.softplus
    assert HydraBNUNet06_LSTM4(3, 8, 1, 0.0)._reg_activation is F.softplus  # the argument default


def test_softplus_on_standard_head():
    assert _act("standard", "softplus") is F.softplus


@pytest.mark.parametrize("value", ["relu", "leaky_relu", "Softplus", "softpuls", ""])
def test_anything_else_is_refused_and_names_what_is_implemented(value):
    with pytest.raises(ValueError) as excinfo:
        _act("standard", value)
    assert str(excinfo.value) == (
        f"reg_activation={value!r} is not implemented. Only ['softplus'] is implemented at "
        "present (ADR-079 Rule 1, views-hydranet issue #632)."
    )
