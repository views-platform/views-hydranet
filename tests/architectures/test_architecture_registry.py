"""The architecture registry seam, and the invariant that the refactor changed nothing.

`choose_model` was a hardcoded `if/else` with one branch. Six bake-off candidates would have meant
six edits to a dispatcher, so it now resolves through `architectures.registry`. That refactor
touches the construction path EVERY existing run uses, which is why the byte-identity test
below is a blocker rather than a nicety (dossier `03` §D, G2).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.architectures.registry import (  # noqa: E402
    architecture_names,
    get_architecture,
)

IN_CH, HID_CH, H, W = 3, 32, 16, 16
INCUMBENT = "HydraBNUNet06_LSTM4"


def _build(cls, seed=7):
    torch.manual_seed(seed)
    return cls(IN_CH, HID_CH, 1, 0.0, output_distribution="nb").float().eval()


def _fwd(model, seed=11):
    torch.manual_seed(seed)
    x = torch.randn(1, IN_CH, H, W).float()
    h = model.init_hTtime(HID_CH, H, W).float()
    with torch.no_grad():
        return model(x, h)


# ── the seam ────────────────────────────────────────────────────────────────────────────────


def test_incumbent_is_registered():
    assert INCUMBENT in architecture_names()
    assert get_architecture(INCUMBENT) is HydraBNUNet06_LSTM4


def test_unknown_name_raises_and_names_what_is_available():
    """A typo in `config['model']` must stop at construction, not train a different network."""
    with pytest.raises(ValueError, match="Unknown model type"):
        get_architecture("NoSuchArchitecture")
    try:
        get_architecture("NoSuchArchitecture")
    except ValueError as exc:
        assert INCUMBENT in str(exc), "the error must name the registered set, not just complain"


def test_registry_import_does_not_import_every_architecture():
    """Lazy factories: importing the registry must not drag in every architecture module.

    Mirrors `distributions.registry`. Keeps `config_initializer` and other torch-free consumers
    cheap, and means one broken candidate cannot break every import.
    """
    import inspect

    from views_hydranet.architectures import registry

    src = inspect.getsource(registry)
    assert "importlib.import_module" in src
    assert "from views_hydranet.architectures.HydraBNrecurrentUnet" not in src


# ── G2: the blocker ─────────────────────────────────────────────────────────────────────────


def test_incumbent_is_byte_identical_through_the_registry():
    """The registry path must reproduce direct construction EXACTLY, not merely closely.

    Same seed for weight init, same seed for the input, `torch.equal` on all three outputs. If this
    ever fails, every result in the ledger measured before the refactor is on a different network
    than every result after it.
    """
    direct = _build(HydraBNUNet06_LSTM4)
    viareg = _build(get_architecture(INCUMBENT))

    for (n_a, p_a), (n_b, p_b) in zip(direct.named_parameters(), viareg.named_parameters()):
        assert n_a == n_b
        assert torch.equal(p_a, p_b), f"weight init diverged at {n_a}"

    out_a, out_b = _fwd(direct), _fwd(viareg)
    assert torch.equal(out_a.reg, out_b.reg), "reg differs through the registry"
    assert torch.equal(out_a.cls, out_b.cls), "cls differs through the registry"
    assert torch.equal(out_a.h_next, out_b.h_next), "h_next differs through the registry"


def test_choose_model_routes_through_the_registry(valid_config_dict):
    """`choose_model` must consult the registry, so an unknown name fails the same way."""
    from views_hydranet.utils.utils import choose_model

    cfg = {**valid_config_dict, "model": INCUMBENT, "output_distribution": "nb"}
    model = choose_model(cfg, torch.device("cpu"))
    assert isinstance(model, HydraBNUNet06_LSTM4)

    with pytest.raises(ValueError, match="Unknown model type"):
        choose_model({**cfg, "model": "NoSuchArchitecture"}, torch.device("cpu"))


@pytest.mark.parametrize("name", sorted(architecture_names()))
def test_choose_model_builds_every_registered_architecture(valid_config_dict, name):
    """The construction path a real run takes, for EVERY registered name.

    A `/falsify guard` audit reverted `choose_model` to its pre-registry hardcoded form and all
    1754 tests stayed green — because nothing exercised it with a non-incumbent name. That defect
    kills every bake-off arm at construction, ~2 minutes into its own 2-hour slot, twelve times.
    Parametrising over the registry means a newly registered architecture is covered automatically
    rather than when someone remembers.
    """
    from views_hydranet.utils.utils import choose_model

    cfg = {**valid_config_dict, "model": name, "output_distribution": "nb"}
    model = choose_model(cfg, torch.device("cpu"))
    assert type(model) is get_architecture(name), f"choose_model built the wrong class for {name}"


def test_choose_model_forwards_the_config_kwargs(valid_config_dict):
    """The kwargs `choose_model` passes are part of the contract, not incidental.

    Byte-identity compares two directly-constructed models and never touches this path, so
    dropping `reg_activation` or flipping the `static_top_skip` default was invisible. Both are
    behavioural: `static_top_skip=False` disables the ADR-061 re-injection (C-228).
    """
    import inspect

    from views_hydranet.utils.utils import choose_model

    src = inspect.getsource(choose_model)
    for kw in ("output_distribution", "n_static_channels", "static_top_skip", "reg_activation"):
        assert f"{kw}=" in src, f"choose_model no longer forwards {kw}"
    assert 'config.get("static_top_skip", True)' in src, (
        "the static_top_skip default changed; False disables the ADR-061 top-skip re-injection"
    )
