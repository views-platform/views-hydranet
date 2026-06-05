"""
Tests for LockedDropout — variational (consistent-mask) dropout for
autoregressive stability (ADR-057; Gal & Ghahramani 2016, *A Theoretically
Grounded Application of Dropout in RNNs*).

Defect (C-113): standard ``nn.Dropout`` in train mode resamples a fresh mask on
every forward. In the 36-step autoregressive inference loop this injects fresh
white noise at each step, which the free-running recurrence amplifies (step-1
normal → step-12 catastrophic). ``LockedDropout`` holds the mask fixed across
forwards (one posterior sample's trajectory) and refreshes it only on
``reset()`` — eliminating the per-step noise injection. When ``locked`` is
False it behaves exactly as ``nn.Dropout`` (so the training path is unchanged,
ADR-057 Decision 2a).
"""

import pytest
import torch

from views_hydranet.architectures.locked_dropout import LockedDropout


def _make_real_model(in_ch=3, out_ch=1, hidden=16, dropout_rate=0.5):
    """Real HydraBNUNet06_LSTM4 with active dropout (TinyModel has none)."""
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
        HydraBNUNet06_LSTM4,
    )

    return HydraBNUNet06_LSTM4(in_ch, hidden, out_ch, dropout_rate).float()


class TestGreenLockedDropoutModule:
    """Module contract: a drop-in for nn.Dropout with a lockable consistent mask."""

    def test_rate_zero_is_identity(self):
        d = LockedDropout(0.0)
        d.train()
        d.locked = True
        x = torch.randn(2, 4, 8, 8)
        assert torch.equal(d(x), x)

    def test_eval_mode_is_identity(self):
        d = LockedDropout(0.5)
        d.eval()
        x = torch.randn(2, 4, 8, 8)
        assert torch.equal(d(x), x), "no dropout in eval mode (matches nn.Dropout)"

    def test_unlocked_resamples_like_standard_dropout(self):
        d = LockedDropout(0.5)
        d.train()  # locked defaults False
        x = torch.ones(1, 16, 8, 8)
        a, b = d(x), d(x)
        assert not torch.equal(a, b), "unlocked must resample per forward (== nn.Dropout)"

    def test_locked_mask_consistent_across_forwards(self):
        d = LockedDropout(0.5)
        d.train()
        d.locked = True
        x = torch.ones(1, 16, 8, 8)
        a, b = d(x), d(x)
        assert torch.equal(a, b), "locked mask must be identical across forwards"

    def test_reset_refreshes_mask(self):
        torch.manual_seed(0)
        d = LockedDropout(0.5)
        d.train()
        d.locked = True
        x = torch.ones(1, 64, 8, 8)
        a = d(x)
        d.reset()
        b = d(x)
        assert not torch.equal(a, b), "reset() must draw a fresh mask"

    def test_shape_keyed_independent_and_stable(self):
        d = LockedDropout(0.5)
        d.train()
        d.locked = True
        x1 = torch.ones(1, 8, 8, 8)
        x2 = torch.ones(1, 8, 4, 4)
        a1, a2 = d(x1), d(x2)
        assert torch.equal(d(x1), a1), "mask stable per shape across forwards"
        assert torch.equal(d(x2), a2), "second shape has its own stable mask"

    def test_inverted_scaling_matches_dropout(self):
        d = LockedDropout(0.5)
        d.train()
        d.locked = True
        x = torch.ones(1, 256, 8, 8)
        y = d(x)
        kept = y[y != 0]
        assert torch.allclose(kept, torch.full_like(kept, 2.0)), (
            "kept units scale by 1/(1-p) (inverted dropout, matches nn.Dropout)"
        )


class TestRedLockedDropoutValidation:
    """Fail-loud on invalid dropout probability (ADR-008)."""

    def test_p_one_raises(self):
        with pytest.raises(ValueError):
            LockedDropout(1.0)

    def test_p_negative_raises(self):
        with pytest.raises(ValueError):
            LockedDropout(-0.1)


class TestGreenConsistentMaskStopsPerStepNoise:
    """Mechanism (C-113): feeding a constant input across simulated steps yields
    a constant output under a locked mask (no per-step noise), but a varying
    output under standard per-step dropout — the noise the recurrence amplifies."""

    def test_locked_constant_input_gives_constant_output(self):
        d = LockedDropout(0.5)
        d.train()
        d.locked = True
        x = torch.ones(1, 64, 8, 8)
        outs = [d(x) for _ in range(12)]  # 12 autoregressive "steps"
        for o in outs[1:]:
            assert torch.equal(o, outs[0]), "locked: no per-step variation injected"

    def test_unlocked_constant_input_gives_varying_output(self):
        d = LockedDropout(0.5)
        d.train()  # locked False == per-step resampling (the bug)
        x = torch.ones(1, 64, 8, 8)
        outs = [d(x) for _ in range(12)]
        assert any(not torch.equal(o, outs[0]) for o in outs[1:]), (
            "per-step dropout injects fresh noise each step"
        )


class TestGreenLockedDropoutModelIntegration:
    """Real HydraBNUNet06_LSTM4 with LockedDropout swapped in: under the inference
    configuration (locked + train), repeated forwards on identical (x, h) are
    reproducible; reset() refreshes the mask."""

    def _setup(self):
        from views_hydranet.infrastructure.reproducibility_gate import (
            ReproducibilityGate,
        )

        ReproducibilityGate.lock_entropy(np_seed=4, torch_seed=4)
        model = _make_real_model(dropout_rate=0.5).eval()
        H = W = 8
        x = torch.ones(1, 3, H, W)
        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).float()
        return model, x, h

    def test_locked_forward_reproducible_within_sample(self):
        model, x, h = self._setup()
        model.set_locked_dropout(True)  # MC dropout active + mask locked
        out1 = model(x, h).reg
        out2 = model(x, h).reg
        assert torch.allclose(out1, out2), (
            "locked: same mask across a sample's forwards → reproducible"
        )

    def test_reset_between_samples_changes_output(self):
        model, x, h = self._setup()
        model.set_locked_dropout(True)
        out1 = model(x, h).reg
        model.reset_locked_dropout()  # next posterior sample
        out2 = model(x, h).reg
        assert not torch.allclose(out1, out2), (
            "reset() between samples → fresh mask → different draw"
        )

    def test_default_model_dropout_is_per_step(self):
        """Without enabling locked mode, the model dropout resamples per forward
        (status quo / training behavior is unchanged)."""
        model, x, h = self._setup()
        model.train()  # locked stays False
        out1 = model(x, h).reg
        out2 = model(x, h).reg
        assert not torch.allclose(out1, out2), (
            "unlocked model dropout resamples per forward (training unchanged)"
        )
