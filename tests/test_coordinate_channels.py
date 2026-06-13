"""ADR-061 coordinate channels — the (row,col) derivation + the top-skip decoder injection.

Coordinates are the first real instance of the ADR-060 static-channel seam: derived in [-1, 1]
over the grid, injected at the input (the seam, #108) AND re-concatenated raw onto the
full-resolution skip (`e0s`) feeding each decoder head (#109). With `n_static_channels=0` the
model is byte-identical to the pre-coord network. ADR-005 Green/Beige/Red.
"""

import numpy as np
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.static_channels import GridGeometry, derive


def _model(input_channels: int, n_static: int) -> HydraBNUNet06_LSTM4:
    return HydraBNUNet06_LSTM4(
        input_channels=input_channels,
        total_hidden_channels=8,  # divisible by 8 (4 LSTM cells × 2)
        output_channels=1,
        dropout_rate=0.0,
        n_static_channels=n_static,
    ).eval()


# ── coord derivation: [-1, 1] range + corners ───────────────────────────────────────────────────
def test_coord_derivation_range_and_corners():
    geom = GridGeometry(height=10, width=20, row_offset=87, col_offset=310)
    row = derive("row_coord", geom)
    col = derive("col_coord", geom)
    assert row.shape == (10, 20) and col.shape == (10, 20)
    # canonical CoordConv range, corners at the extremes
    assert row.min() == -1.0 and row.max() == 1.0
    assert col.min() == -1.0 and col.max() == 1.0
    assert row[0, 0] == -1.0 and row[-1, 0] == 1.0  # row varies down the grid
    assert col[0, 0] == -1.0 and col[0, -1] == 1.0  # col varies across
    assert np.allclose(row[:, 0], row[:, -1])  # row is constant across columns (and vice-versa)
    assert np.allclose(col[0, :], col[-1, :])


def test_coord_single_cell_is_safe():
    """H or W == 1 must not divide by zero — degenerate grids derive to 0, not NaN."""
    out = derive("row_coord", GridGeometry(height=1, width=4, row_offset=0, col_offset=0))
    assert np.isfinite(out).all() and out.shape == (1, 4)


# ── top-skip: the model accepts 5-ch input, emits 3 reg + 3 cls, dec_conv1 widened by n_static ───
def test_model_forward_with_coordinate_channels():
    torch.manual_seed(0)
    model = _model(input_channels=5, n_static=2)
    x = torch.randn(1, 5, 8, 8)
    h = model.init_hTtime(model.base, 8, 8)
    out = model(x, h)
    assert out.reg.shape == (1, 3, 8, 8)  # 3 reg heads, output_channels=1 each
    assert out.cls.shape == (1, 3, 8, 8)
    # the top-skip widened every head's dec_conv1 by n_static (reg AND cls — the gate needs it)
    assert model.dec_conv1_head1_reg.in_channels == model.base * 2 + 2
    assert model.dec_conv1_head1_class.in_channels == model.base * 2 + 2
    assert model.dec_conv1_head3_class.in_channels == model.base * 2 + 2


# ── n_static=0 is byte-identical to the pre-coord network (no widening, no concat) ───────────────
def test_n_static_zero_is_unchanged():
    model = _model(input_channels=3, n_static=0)
    assert model.n_static == 0
    # dec_conv1 in-channels unchanged (base*2) — the only n_static-gated structure is inert at 0
    assert model.dec_conv1_head1_reg.in_channels == model.base * 2
    assert model.dec_conv1_head2_class.in_channels == model.base * 2
    x = torch.randn(1, 3, 8, 8)
    h = model.init_hTtime(model.base, 8, 8)
    out = model(x, h)  # the coords=None / e0s_topskip==e0s path runs the original forward
    assert out.reg.shape == (1, 3, 8, 8)
