"""`static_top_skip` flag (C-228 diagnosis/fix) — statics may stay in the ENCODER input but leave
the ADR-061 top-skip re-injection at the decoder heads' dec_conv1.

Default True = byte-identical to pre-flag (dec_conv1 sized base*2 + n_static, coords re-injected).
False = dec_conv1 sized base*2 (no re-injection), while the encoder still receives the statics via
input_channels. ADR-005 Green/Beige/Red.
"""

import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

BASE = 8  # total_hidden_channels (÷8)
N_STATIC = 1
_HEADS = ("head1_reg", "head1_class", "head2_reg", "head2_class", "head3_reg", "head3_class")


def _build(static_top_skip):
    return HydraBNUNet06_LSTM4(
        input_channels=3 + N_STATIC,  # 3 dynamic + 1 static in the ENCODER input
        total_hidden_channels=BASE,
        output_channels=1,
        dropout_rate=0.0,
        output_distribution="nb",
        n_static_channels=N_STATIC,
        static_top_skip=static_top_skip,
    )


def test_default_is_true_and_reinjects_static_at_all_heads():
    """Green: default True re-injects the static at every dec_conv1 (base*2 + n_static) — the
    pre-flag / byte-identical wiring, incl. the class (gate) heads."""
    m = _build(static_top_skip=True)
    assert m.static_top_skip is True
    for head in _HEADS:
        assert getattr(m, f"dec_conv1_{head}").in_channels == BASE * 2 + N_STATIC, head


def test_false_drops_static_from_the_topskip_but_keeps_it_in_the_encoder():
    """Red→green: False sizes every dec_conv1 at base*2 (no re-injection), yet the ENCODER input
    still carries the static channel (input_channels unchanged)."""
    m = _build(static_top_skip=False)
    assert m.static_top_skip is False
    for head in _HEADS:
        assert getattr(m, f"dec_conv1_{head}").in_channels == BASE * 2, head  # gate: no raw static
    # the encoder still receives the static: enc_conv0 input width includes the 4 input channels
    assert m.enc_conv0.in_channels == (3 + N_STATIC) + BASE // 2


def test_forward_runs_with_topskip_off():
    """The forward pass must run with the top-skip disabled (coords gated to None)."""
    m = _build(static_top_skip=False).eval()
    b, h, w = 1, 16, 16
    x = torch.zeros(b, 3 + N_STATIC, h, w)
    hidden = torch.zeros(b, BASE, h, w)
    with torch.no_grad():
        out = m(x, hidden)
    assert out.reg.shape[0] == b and out.cls.shape[0] == b


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
