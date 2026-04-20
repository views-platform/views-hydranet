import numpy as np
import pytest
import torch

from views_hydranet.utils.volume_handler import VolumeHandler


class TestGreen:
    """Green: VolumeHandler geometric transformations preserve Ledger integrity."""

    @pytest.fixture
    def vh(self):
        # Create a simple 2x2 spatial volume
        # [T=1, H=2, W=2, C=1]
        # Data: Top row [1, 2], Bottom row [3, 4]
        data = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
        return VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["f1"],
            time_col="t",
            id_col="i",
            spatial_cols=["H", "W"],
        )

    def test_flip_spatial(self, vh):
        """Verify that flip('W') reverses the columns."""
        # Original: [1, 2], [3, 4]
        # Flip W: [2, 1], [4, 3]
        flipped = vh.flip("W")

        expected = np.array([[[[2], [1]], [[4], [3]]]], dtype=np.float32)
        assert np.array_equal(flipped.data, expected)

        # Check history
        assert flipped.history[-1] == ("flip", "W")

    def test_flip_temporal(self, vh):
        """Verify that flip('T') works (though unusual for this model)."""
        flipped = vh.flip("T")
        assert flipped.history[-1] == ("flip", "T")

    def test_permute_axes(self, vh):
        """Verify that permute reorders data and updates the axes ledger."""
        # Current: (T, H, W, C) -> Index (0, 1, 2, 3)
        # Target: (T, C, H, W) -> Index (0, 3, 1, 2)
        permuted = vh._permute((0, 3, 1, 2))

        assert permuted.axes == ("T", "C", "H", "W")
        assert permuted.data.shape == (1, 1, 2, 2)
        assert permuted.history[-1] == ("permute", (0, 3, 1, 2))

    def test_flip_returns_new_instance(self, vh):
        """flip() must return a new VolumeHandler, not mutate the original."""
        original_data = vh.data.copy()
        flipped = vh.flip("W")
        assert flipped is not vh
        np.testing.assert_array_equal(vh.data, original_data)

    def test_permute_returns_new_instance(self, vh):
        """_permute() must return a new VolumeHandler, not mutate the original."""
        original_data = vh.data.copy()
        permuted = vh._permute((0, 3, 1, 2))
        assert permuted is not vh
        np.testing.assert_array_equal(vh.data, original_data)

    def test_torch_geometric(self):
        """Verify that flip and permute work on torch tensors."""
        # [B=1, T=1, C=1, H=2, W=2]
        data = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
        vh = VolumeHandler(
            data=data,
            axes=("B", "T", "C", "H", "W"),
            channel_map=["v"],  # dummy
            time_col="t",
            id_col="i",
            spatial_cols=["H", "W"],
        )

        flipped = vh.flip("W")
        assert flipped.data[0, 0, 0, 0, 0].item() == 2.0

        permuted = flipped._permute((0, 1, 2, 4, 3))
        assert permuted.axes == ("B", "T", "C", "W", "H")
