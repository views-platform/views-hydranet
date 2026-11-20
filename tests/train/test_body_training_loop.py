"""S3 (#619, epic #616): the real training loop with a body ends at the older route's weights.

`tests/test_body_golden.py` pins one `_process_sequence` call. This file goes one level up: the
real `make()` and `training_loop()`, optimizer steps included, for each body in use. Until #629 the
loop ran twice from the same seed, once with `build_body` forced to return ``None`` so the older
route ran, and every parameter had to be `torch.equal`. The older route is gone (#629): its final
weights were recorded on 2026-11-20 at the commit before the deletion, after checking there that
the body route reproduced them bit for bit, in `tests/golden/body_loop_golden.json`.

⚠️ The comparison is exact, and only possible, on the machine that recorded them. Measured on the
CI runner (PR #659, 2026-11-20): after the loop the weight sums differ from the recording by 0.2%
to 0.5% for every body, because Adam turns float32 rounding differences between CPUs into
different steps. No tolerance separates that from a real change. Elsewhere the test says so with a
warning and checks only that the right body trained the model. The one-step references in
`tests/test_body_golden.py` do carry across machines within their 1e-4 tolerance.

🟥 Red by the ADR-005 taxonomy: this is the claim "training is unchanged" at the level a run
actually executes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import views_hydranet.train.training_engine as engine  # noqa: E402
from tests.test_body_golden import _digest, _fingerprint, _same  # noqa: E402
from tests.train.conftest import loop_config  # noqa: E402
from views_hydranet.body.bodies import (  # noqa: E402
    DistributionBody,
    MeanSquaredErrorBody,
    RankingBody,
)
from views_hydranet.train.training_engine import make, training_loop  # noqa: E402
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402

GOLDEN_LOOP = Path(__file__).parents[1] / "golden" / "body_loop_golden.json"

LIVE = {
    "mse": ({"output_distribution": "standard", "loss_reg": "mse"}, MeanSquaredErrorBody),
    "ranking": (
        {"output_distribution": "standard", "loss_reg": "listwise_level", "loss_reg_y_max": 1e4},
        RankingBody,
    ),
    "nb": ({"output_distribution": "nb", "loss_reg": "mse"}, DistributionBody),
    "truncated_nb": ({"output_distribution": "truncated_nb", "loss_reg": "mse"}, DistributionBody),
    "mixture_nb": ({"output_distribution": "mixture_nb", "loss_reg": "mse"}, DistributionBody),
}


def _varied_handler(cfg):
    """Cells with different values. The shared `loop_handler` sets every cell to 1.0, which ties
    every cell on one level: the ranking loss then has a gradient of exactly zero and its head
    never moves, so equal weights would prove nothing for that body."""
    rng = np.random.default_rng(3)
    data = rng.random((5, 8, 8, 4))
    data[..., 1:] = (rng.random((5, 8, 8, 3)) < 0.4) * rng.random((5, 8, 8, 3)) * 4.0
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["t", "lr_f1", "lr_f2", "lr_f3"],
        time_col="t",
        id_col="i",
        spatial_cols=["y", "x"],
        identity_cols=["t", "i"],
        feature_cols=["lr_f1", "lr_f2", "lr_f3"],
        config=cfg,
    )


def _train(name, monkeypatch):
    overrides, _ = LIVE[name]
    # weight_decay=0: with decay the head moves even when no loss trains it, and the last
    # assertion of the test would pass for that reason (code review, 2026-11-16).
    cfg = loop_config(
        reg_activation="softplus", loss_class="weighted_bce", weight_decay=0.0, **overrides
    )
    seen: list = []
    real = engine.build_body

    def spy(config, criterion_reg):
        body = real(config, criterion_reg)
        seen.append(body)
        return body

    monkeypatch.setattr(engine, "build_body", spy)
    # Which body each training step actually received. Equal weights cannot show this, because
    # the two paths are bit-identical by design (guard audit, 2026-11-16).
    used: list = []
    real_step = engine._process_sequence

    def step_spy(*args, **kwargs):
        used.append(kwargs.get("body"))
        return real_step(*args, **kwargs)

    monkeypatch.setattr(engine, "_process_sequence", step_spy)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    training_loop(cfg, model, criterion, optimizer, scheduler, _varied_handler(cfg), device)
    assert used, "the loop never ran a training step"
    assert len(used) == len(seen) and all(u is b for u, b in zip(used, seen, strict=True)), (
        "a training step did not receive the body `train()` built"
    )
    return model, seen, initial


@pytest.mark.parametrize("name", list(LIVE))
def test_the_loop_reaches_the_older_routes_recorded_weights(name, monkeypatch):
    with_body, seen, initial = _train(name, monkeypatch)
    assert seen, "`train()` never asked for a body"
    assert all(type(b) is LIVE[name][1] for b in seen), (
        f"{name}: expected {LIVE[name][1].__name__} on every step, got "
        f"{sorted({type(b).__name__ for b in seen})}"
    )
    golden = json.loads(GOLDEN_LOOP.read_text())
    assert set(golden["cases"]) == set(LIVE), "the loop golden lost a body"
    flat = torch.cat(
        [v.detach().to(torch.float32).reshape(-1) for v in with_body.state_dict().values()]
    )
    got, want = _digest(flat), golden["cases"][name]
    assert torch.isfinite(flat).all(), f"{name}: the weights after the loop are not finite"
    if golden["recorded_on"] == _fingerprint():
        assert _same(got, want, exact=True), (
            f"{name}: the weights after the loop differ bit-for-bit from the older route's "
            f"recording; sums {got['sum']!r} vs {want['sum']!r}"
        )
    else:
        warnings.warn(
            f"{name}: the loop's recorded weights come from another machine "
            f"({golden['recorded_on']}); they are not comparable here (see the module "
            "docstring), so only the body and the training were checked.",
            stacklevel=1,
        )

    # The equality above would hold trivially if nothing trained.
    head = "dec_conv4_head1_reg.weight"
    assert not torch.equal(with_body.state_dict()[head], initial[head]), (
        f"{name}: the regression head did not move, so equal weights prove nothing"
    )


def test_the_training_step_has_no_default_body():
    """Since #629 there is one route through the training step: the body. `body` must be a
    required keyword argument, so a caller that forgets it gets a TypeError at the call, not a
    body guessed from the loss object."""
    import inspect

    from views_hydranet.train.training_engine import _process_sequence

    parameter = inspect.signature(_process_sequence).parameters["body"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
