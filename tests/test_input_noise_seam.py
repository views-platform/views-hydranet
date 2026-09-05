"""The SEAM tests for #311's input noise — what the call site actually passes.

Written after the 2026-09-04 independent mutation audit, which found the helper well guarded
(17 of 22 arithmetic mutations caught) and the **wiring not guarded at all: 20 of 22 call-site,
`train()` and config mutations survived**. `tests/test_input_noise.py` proves the augmentation
computes correctly and proves *off is off*; it did not prove that **on is on the way the design
says**, and every one of these tests exists because a specific mutation slipped through.

The technique is a spy on `_apply_input_noise` that records its arguments. That is what makes the
wiring observable: the earlier call-site tests asserted only `off != on` and determinism, both of
which survive essentially any corruption of what gets passed.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import views_hydranet.train.training_engine as te  # noqa: E402
from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices  # noqa: E402

_H, _W, _T = 3, 3, 8
_FEATS = ["reg0", "reg1", "cls0", "row"]
_CFG = {
    "regression_targets": ["reg0", "reg1"],
    "classification_targets": ["cls0"],
    "features": ["reg0", "reg1"],
    "static_channels": [],
}
_CFG_STATIC = {**_CFG, "features": ["reg0", "reg1", "row"], "static_channels": ["row"]}


class _Model(torch.nn.Module):
    def __init__(self, n_reg, n_cls):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, n_reg + n_cls, kernel_size=1)
        self.n_reg = n_reg

    def forward(self, x, hidden):
        o = self.conv(x[:, :1])
        return types.SimpleNamespace(
            reg=o[:, : self.n_reg], cls=o[:, self.n_reg :], reg_latent=None, h_next=hidden
        )


def _spy(monkeypatch):
    calls = []
    real = te._apply_input_noise

    def _rec(dyn_input, keep, dropout, channels):
        calls.append(
            {
                "width": dyn_input.shape[1],
                "keep_width": keep.shape[1],
                "keep_all_ones": bool((keep == 1).all()),
                "dropout": dropout,
                "channels": list(channels),
                "input": dyn_input.detach().clone(),
            }
        )
        return real(dyn_input, keep, dropout, channels)

    monkeypatch.setattr(te, "_apply_input_noise", _rec)
    return calls


def _run(monkeypatch, *, dropout=0.5, segment=3, channels=(0, 1), cfg=None, ss_epsilon=0.0):
    cfg = cfg or _CFG
    calls = _spy(monkeypatch)
    torch.manual_seed(5)
    idx = _SequenceIndices(_FEATS, cfg)
    model = _Model(idx.n_reg, idx.n_cls)
    _process_sequence(
        train_tensor=torch.rand(1, _T, len(_FEATS), _H, _W),
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
        ss_epsilon=ss_epsilon,
        input_noise_dropout=dropout,
        input_noise_segment=segment,
        input_noise_channels=None if channels is None else list(channels),
    )
    return calls


# --- what gets passed (TR-01, TR-04, CS-04, CS-05, CS-08) ------------------


def test_the_channel_restriction_actually_reaches_the_helper(monkeypatch):
    """Mutation TR-01/CS-04: passing `None` (or falling back to every channel) makes the whole
    augmentation a no-op in production while every test stays green — the C-324 inert signature."""
    calls = _run(monkeypatch, channels=(1,))
    assert calls, "the helper was never called"
    assert all(c["channels"] == [1] for c in calls)


def test_an_empty_channel_list_is_not_silently_widened(monkeypatch):
    calls = _run(monkeypatch, channels=())
    assert calls, "the helper was never called — `all(...)` over an empty list proves nothing"
    assert all(c["channels"] == [] for c in calls)


def test_the_configured_dropout_reaches_the_helper_unchanged(monkeypatch):
    """Mutation TR-04/AN-15: a hardcoded or dropped rate."""
    calls = _run(monkeypatch, dropout=0.37)
    assert calls, "the helper was never called — `all(...)` over an empty list proves nothing"
    assert all(c["dropout"] == pytest.approx(0.37) for c in calls)


def test_a_dropout_of_zero_still_runs_rather_than_being_treated_as_off(monkeypatch):
    """Mutation CS-05: `is not None` weakened to truthiness would make 0.0 silently off — and the
    config's `gt=0.0` bound, which is the only thing making 0.0 unreachable, is itself mutable."""
    calls = _run(monkeypatch, dropout=0.0)
    assert calls and all(c["dropout"] == 0.0 for c in calls)


def test_the_arguments_are_not_transposed(monkeypatch):
    """Mutation CS-08: swapping `dyn_input` and the keep-mask type-checks and stays green.

    The earlier version of this test asserted `width == keep_width`, which can never fail — the two
    always have the same shape, transposed or not. Found by the second audit as a vacuous assertion
    whose docstring named a tell it did not check. The real tell is the CONTENT: the mask is all
    ones on a reset step, and the data is not.
    """
    calls = _run(monkeypatch, dropout=0.0, segment=100)
    assert calls[0]["keep_all_ones"], "the first argument is the mask, not the input"
    assert not bool((calls[0]["input"] == 1).all()), (
        "the input the helper received was all ones — the mask was passed in its place"
    )


# --- the segment and accumulation (CS-02, CS-03, CS-07, CS-11, CS-12, PS-01) ---


def test_the_mask_resets_exactly_on_segment_boundaries(monkeypatch):
    """Mutations CS-02 (never resets) and CS-03/CS-12 (resets every step). CS-03 turns the design's
    central claim — accumulating, because the paper's ablation found i.i.d. worse — into i.i.d."""
    calls = _run(monkeypatch, dropout=0.5, segment=3)
    fresh = [i for i, c in enumerate(calls) if c["keep_all_ones"]]
    assert fresh == [0, 3, 6], f"reset points were {fresh}, expected every 3 steps"


def test_the_mask_is_carried_forward_between_resets(monkeypatch):
    """Mutation CS-07/CS-12: discarding the returned mask silently removes accumulation."""
    calls = _run(monkeypatch, dropout=1.0, segment=100)
    assert calls[0]["keep_all_ones"]
    assert not any(c["keep_all_ones"] for c in calls[1:]), "the mask was not carried forward"


def test_the_fresh_mask_is_ones_not_zeros(monkeypatch):
    """Mutation CS-11: `zeros_like` blanks every input at every step; `off != on` still holds."""
    assert _run(monkeypatch, dropout=0.5, segment=3)[0]["keep_all_ones"]


def test_a_missing_segment_fails_loud_when_the_noise_is_on(monkeypatch):
    """Mutation PS-01: a literal default would silently differ from the caller's `time_steps`."""
    # `True` is in the list because isinstance(True, int) is True: `segment=True` gives 1, the mask
    # resets every step, and the accumulating design silently becomes the i.i.d. one the paper's
    # ablation found WORSE. The engine's comment argues for this guard at length and nothing tested
    # it — C-303, in a comment that says "the pattern was known here and not applied".
    for bad in (None, 0, -1, 2.5, True, False):
        with pytest.raises(ValueError, match="input_noise_segment"):
            _run(monkeypatch, dropout=0.5, segment=bad)


# --- ordering and the scheduled-sampling branch (CS-09, CS-10) --------------


def test_the_noise_runs_BEFORE_the_static_re_attach(monkeypatch):
    """Mutation CS-09: moving the noise after `_attach_static_channels` noises CoordConv geometry.

    The C-309 argument was only half-carried before this test: `_noisable_channels` excluding
    statics is meaningless if the ordering that makes it apply is unpinned. The tell is the tensor
    width — after the re-attach the input carries the static channel too.
    """
    calls = _run(monkeypatch, cfg=_CFG_STATIC, channels=(0, 1))
    assert calls
    n_dyn = len(_CFG_STATIC["features"])
    n_with_static = n_dyn + len(_CFG_STATIC["static_channels"])
    assert all(c["width"] == n_dyn for c in calls), (
        f"the helper saw {calls[0]['width']} channels; {n_with_static} means geometry was noised"
    )


def test_the_noise_sees_the_scheduled_sampling_SUBSTITUTION(monkeypatch):
    """Mutation CS-10: noising `t0_gt` discards the fed-back prediction, so the ε>0 arm trains on a
    different input from the one the comment claims and the arms stop being comparable."""
    gt = _run(monkeypatch, dropout=0.0, segment=100, ss_epsilon=0.0)
    ss = _run(monkeypatch, dropout=0.0, segment=100, ss_epsilon=1.0)
    later = [i for i in range(1, min(len(gt), len(ss)))]
    assert any(not torch.equal(gt[i]["input"], ss[i]["input"]) for i in later), (
        "the helper saw the same input with and without scheduled sampling — the noise is "
        "reading ground truth rather than the substituted input"
    )


def test_each_noised_channel_gets_its_OWN_random_mask(monkeypatch):
    """Survivors AN-12/AN-16: drawing one [B,1,H,W] mask and broadcasting it silences the SAME
    cells in every target at once. The marginal drop rate stays exactly right, so the rate test
    passes — but the augmentation's correlation structure is different, and the model is trained on
    a different thing. Independence has to be asserted, not inferred from the rate."""
    from views_hydranet.train.training_engine import _apply_input_noise

    torch.manual_seed(4)
    x = torch.ones(1, 3, 64, 64)
    out, _ = _apply_input_noise(x, torch.ones_like(x), 0.5, [0, 1, 2])
    dropped = (out == 0).float()
    agree01 = float((dropped[:, 0] == dropped[:, 1]).float().mean())
    agree02 = float((dropped[:, 0] == dropped[:, 2]).float().mean())
    # Independent Bernoulli(0.5) masks agree on ~50% of cells; a shared mask agrees on 100%.
    assert agree01 < 0.65, f"channels 0 and 1 share a mask (agreement {agree01:.3f})"
    assert agree02 < 0.65, f"channels 0 and 2 share a mask (agreement {agree02:.3f})"
