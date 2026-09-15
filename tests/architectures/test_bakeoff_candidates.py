"""Contract tests for every bake-off candidate (dossier `03` §D, G7).

A 12-arm queue is ~29 GPU-hours and there is **no mid-training checkpoint** — an architecture that
emits the wrong shape, breaks the state contract, or produces NaN would be discovered ~2.4 h into
its arm and the arm would be lost. Every candidate is therefore checked here, before the GPU.

The contract, from `architectures/registry.py`:
  * `reg` width == n_targets x n_params, `cls` width == n_targets, `h_next` shape == `h`
  * the state is divisible by 8 (`blend_recurrent_state` splits 4 short-term + 4 long-term)
  * outputs finite
  * gradients reach the parameters (a frozen or detached branch would train silently to nothing)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.architectures.registry import (  # noqa: E402
    architecture_names,
    get_architecture,
)

IN_CH, HID, H, W = 3, 32, 32, 32
CANDIDATES = sorted(architecture_names())


def _build(name, seed=7):
    torch.manual_seed(seed)
    return get_architecture(name)(IN_CH, HID, 1, 0.0, output_distribution="nb").float()


def _fwd(model, seed=11, train=False):
    model.train(train)
    torch.manual_seed(seed)
    x = torch.randn(1, IN_CH, H, W).float()
    h = model.init_hTtime(model.base, H, W).float()
    return model(x, h), h


@pytest.mark.parametrize("name", CANDIDATES)
def test_output_contract(name):
    model = _build(name)
    with torch.no_grad():
        out, h = _fwd(model)
    assert out.reg.shape == (1, 3 * 2, H, W), f"{name}: reg must be n_targets x n_params"
    assert out.cls.shape == (1, 3, H, W), f"{name}: cls must be n_targets wide"
    assert out.h_next.shape == h.shape, (
        f"{name}: h_next must match h, or the rollout desynchronises"
    )


@pytest.mark.parametrize("name", CANDIDATES)
def test_state_divisible_by_eight(name):
    """`blend_recurrent_state` splits the state 4 short-term + 4 long-term.

    An even-but-not-8 width would split without error while silently mis-assigning memory types —
    which would break the freeze diagnostics (M38/M39) invisibly rather than loudly.
    """
    model = _build(name)
    assert model.base % 8 == 0, f"{name}: state width {model.base} is not divisible by 8"


@pytest.mark.parametrize("name", CANDIDATES)
def test_outputs_finite(name):
    model = _build(name)
    with torch.no_grad():
        out, _ = _fwd(model)
    for field in ("reg", "cls", "h_next"):
        assert torch.isfinite(getattr(out, field)).all(), f"{name}: non-finite {field}"


@pytest.mark.parametrize("name", CANDIDATES)
def test_gradients_reach_every_parameter_group(name):
    """Every trainable parameter must receive gradient from the combined heads.

    Catches a branch that was built but never wired into `forward` — it would train to nothing and
    the arm would read as "this architecture does not help" rather than "this architecture is
    disconnected". Buffers (the blur filter) are excluded by construction: they are not parameters.
    """
    model = _build(name)
    out, _ = _fwd(model, train=True)
    (out.reg.sum() + out.cls.sum()).backward()
    dead = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not dead, f"{name}: no/invalid gradient at {dead[:5]}"


@pytest.mark.parametrize("name", CANDIDATES)
def test_two_steps_keep_the_state_stable(name):
    """Feed the model its own state twice — shape must persist and values stay finite.

    The rollout does this 36 times; an architecture whose state changes shape or blows up on the
    second step fails here in milliseconds instead of hours in.
    """
    model = _build(name)
    with torch.no_grad():
        out1, h = _fwd(model)
        torch.manual_seed(13)
        x2 = torch.randn(1, IN_CH, H, W).float()
        out2 = model(x2, out1.h_next)
    assert out2.h_next.shape == h.shape
    assert torch.isfinite(out2.h_next).all()


def test_antialiased_pool_adds_no_learnable_parameters():
    """Candidate 1's entire claim to being the clean arm: the blur is a BUFFER, not a Parameter."""
    incumbent = _build("HydraBNUNet06_LSTM4")
    aa = _build("AntiAliasedPool")
    n_i = sum(p.numel() for p in incumbent.parameters())
    n_a = sum(p.numel() for p in aa.parameters())
    assert n_a == n_i, f"AntiAliasedPool added {n_a - n_i} parameters; it must add exactly 0"
    assert any("blur" in n for n, _ in aa.named_buffers()), "the blur filter must be a buffer"


def test_filmskip_starts_as_the_identity():
    """FiLM is zero-initialised, so before training it must reproduce the incumbent exactly.

    Same default-off discipline the config flags use: the new path is provably inert at step 0.
    """
    inc = _build("HydraBNUNet06_LSTM4")
    film = _build("FiLMSkip")
    film.load_state_dict(inc.state_dict(), strict=False)
    with torch.no_grad():
        out_i, _ = _fwd(inc)
        out_f, _ = _fwd(film)
    assert torch.allclose(out_i.reg, out_f.reg, atol=1e-6), "zero-init FiLM must be the identity"
