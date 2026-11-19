"""The model exposes its regression output from before the activation, in training mode only.

Moved here unchanged from tests/test_tobit_loss.py when the tobit loss was deleted (issue #631,
2026-11-18). `reg_latent` outlived tobit: the body objects read it (ADR-079 rule 2), and so does
the ranking loss on the older training path. With no `reg_activation` the head applies ReLU, so
`reg == relu(reg_latent)`.
"""

import torch


class TestGreenModelOutputRegLatent:
    """Verify the pre-ReLU latent is exposed by ModelOutput."""

    def test_model_output_has_reg_latent_field(self):
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            ModelOutput,
        )

        output = ModelOutput(
            reg=torch.zeros(1),
            cls=torch.zeros(1),
            h_next=torch.zeros(1),
        )
        assert output.reg_latent is None

        output_with_latent = ModelOutput(
            reg=torch.zeros(1),
            cls=torch.zeros(1),
            h_next=torch.zeros(1),
            reg_latent=torch.ones(1),
        )
        assert output_with_latent.reg_latent is not None

    def test_model_forward_returns_reg_latent(self):
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.1).float()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        assert output.reg_latent is not None
        assert output.reg_latent.shape == output.reg.shape

    def test_reg_latent_is_pre_relu(self):
        """reg_latent can be negative; reg (post-ReLU) cannot."""
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        torch.manual_seed(42)
        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        assert (output.reg >= 0).all(), "Post-ReLU should be non-negative"
        assert (output.reg_latent < 0).any(), "Pre-ReLU latent should contain negative values"

    def test_relu_relationship(self):
        """reg should equal relu(reg_latent)."""
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        torch.manual_seed(42)
        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        expected = torch.relu(output.reg_latent)
        assert torch.allclose(output.reg, expected, atol=1e-6), (
            "reg should equal F.relu(reg_latent)"
        )

    def test_reg_latent_none_in_eval_mode(self):
        """reg_latent should be None when model is in eval mode (saves memory)."""
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()
        model.eval()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        assert output.reg_latent is None
        assert output.reg is not None
