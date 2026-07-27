"""reg_activation flag (Exp B): decouple the emit activation from output_distribution.

Default (reg_activation=None) must be byte-identical to the pre-flag behavior (keyed off
output_distribution); an explicit value overrides it.
"""

import torch.nn.functional as F

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4


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


def test_default_hurdle_nb_is_softplus():
    assert _act("hurdle_nb", None) is F.softplus  # parity: unchanged when flag unset


def test_default_standard_is_relu():
    assert _act("standard", None) is F.relu  # parity


def test_override_softplus_on_standard_head():
    # Exp B: shrinkage + standard inference + softplus emit head
    assert _act("standard", "softplus") is F.softplus


def test_override_relu_on_hurdle_head():
    assert _act("hurdle_nb", "relu") is F.relu
