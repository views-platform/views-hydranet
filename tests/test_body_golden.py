"""S1 (#617, epic #616): the numbers the five live bodies produce in one training step, pinned.

Epic #616 changes what the training loss reads: the outputs from before the activation instead of
after it (ADR-079 rule 2). It claims that no number changes. Before this file that claim could not
be checked — nothing pinned `out.reg` or a per-target loss through the real model and
`_process_sequence` for any live body.

🟥 Red by the ADR-005 taxonomy: it describes behaviour that must be preserved. Every case builds
the model and the loss from a **raw config dict** through `choose_model` and `choose_loss`, the way
a pod driver does, runs one short `_process_sequence` on CPU, and compares against
`tests/golden/body_golden.json`.

⛔ The test goes through whatever path `train()` uses. Once `views_hydranet.body.bodies` exists,
the body is built here exactly as `train()` builds it and passed in — otherwise these cases would
keep exercising the old path and stay green while proving nothing.

Exact or tolerant. Softplus and convolutions are not bit-stable across CPUs, so a golden recorded
on one machine cannot be demanded bit-for-bit on another. The golden stores the machine it was
recorded on. On that machine the comparison is exact (sha256 of the float32 bytes). Anywhere else
it falls back to `rtol=1e-5, atol=1e-7` on the stored values, and says so.

Regenerate (only ever on an unmodified tree, and say why in the commit):

    PYTHONPATH=. conda run --no-capture-output -n views-hydranet-env \
        python tests/test_body_golden.py --write

⚠️ `PYTHONPATH=.` is required. Without it a script run imports the copy of `views_hydranet`
installed in the environment, not this source tree. The guard at the bottom refuses that.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices  # noqa: E402
from views_hydranet.utils.utils import choose_loss, choose_model  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "body_golden.json"
CPU = torch.device("cpu")

REG = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
CLS = ["by_sb_best", "by_ns_best", "by_os_best"]

#: Only the keys `choose_model`, `choose_loss` and `_SequenceIndices` read. A raw dict on purpose.
BASE = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "output_channels": 1,
    "total_hidden_channels": 32,
    "dropout_rate": 0.0,
    "features": REG,
    "regression_targets": REG,
    "classification_targets": CLS,
    "static_channels": [],
    "loss_class": "weighted_bce",  # what every live config uses (logits in)
    "loss_class_gamma": 1.5,
    "loss_class_alpha": 0.75,
    "loss_reg_a": 258,
    "loss_reg_c": 0.001,
    "reg_activation": "softplus",
}

#: The five live bodies, by the config values that select them today (ADR-079 §2).
BODIES = {
    "mse": {"output_distribution": "standard", "loss_reg": "mse"},
    "ranking": {
        "output_distribution": "standard",
        "loss_reg": "listwise_level",
        "loss_reg_y_max": 10000.0,
    },
    "nb": {"output_distribution": "nb", "loss_reg": "mse"},
    "truncated_nb": {"output_distribution": "truncated_nb", "loss_reg": "mse"},
    "mixture_nb": {"output_distribution": "mixture_nb", "loss_reg": "mse"},
}

#: case name -> (body, extra keyword arguments for `_process_sequence`)
CASES = {
    "mse/all": ("mse", {}),
    "ranking/all": ("ranking", {}),
    "nb/all": ("nb", {"forecast_composition": "soft_gate"}),
    "truncated_nb/all": ("truncated_nb", {"forecast_composition": "soft_gate"}),
    "mixture_nb/all": ("mixture_nb", {"forecast_composition": "soft_gate"}),
    "mse/active": ("mse", {"body_supervision": "active"}),
    "nb/active": ("nb", {"body_supervision": "active", "forecast_composition": "soft_gate"}),
    "mse/active+qs99": (
        "mse",
        {"body_supervision": "active", "qs99_weight": 0.5, "qs99_tau": 0.99},
    ),
    "nb/pushforward": (
        "nb",
        {
            "forecast_composition": "soft_gate",
            "ss_feedback": "sample",
            "pushforward_weight": 0.5,
        },
    ),
}


class _Sum(nn.Module):
    """A plain sum in place of the task balancer, so the per-task terms are read directly."""

    def forward(self, losses):  # noqa: D102
        return losses.sum()


def _fingerprint() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        # Linux only; anywhere else the weaker `platform` value stands and the test is tolerant.
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"torch {torch.__version__} | {platform.machine()} | {cpu}"


def _digest(t: torch.Tensor) -> dict:
    flat = t.detach().to(torch.float32).contiguous().reshape(-1)
    return {
        "sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
        "n": int(flat.numel()),
        "sum": float(flat.double().sum()),
        "abs_sum": float(flat.double().abs().sum()),
        "head": [float(v) for v in flat[:24]],
    }


def _body_kwarg(cfg: dict, criterion_reg) -> dict:
    """Pass the body the way `train()` does, once the body module exists (S2/S3 of #616)."""
    try:
        from views_hydranet.body.bodies import build_body
    except ImportError:
        return {}
    import inspect

    if "body" not in inspect.signature(_process_sequence).parameters:
        return {}
    return {"body": build_body(cfg, criterion_reg)}


def _run(case: str, *, use_body: bool = True) -> dict:
    body_name, kwargs = CASES[case]
    cfg = {**BASE, **BODIES[body_name]}
    torch.manual_seed(1234)
    model = choose_model(cfg, CPU).float()
    model.train()
    criterion_reg, criterion_class, _ = choose_loss(cfg, CPU)
    idx = _SequenceIndices(REG + CLS, cfg)

    hw, steps = 8, 5
    reg_field = (torch.rand(1, steps, 3, hw, hw) < 0.08).float() * torch.rand(1, steps, 3, hw, hw)
    reg_field = reg_field * 4.0  # log1p space; counts stay far below y_max
    x = torch.cat([reg_field, (reg_field > 0).float()], dim=2)
    h = model.init_hTtime(model.base, hw, hw).float()

    probe = copy.deepcopy(model)
    first = probe(x[:, 0, :3], h)

    torch.manual_seed(4321)
    out = _process_sequence(
        x,
        model,
        h,
        criterion_reg=criterion_reg,
        criterion_class=criterion_class,
        multitaskloss_instance=_Sum(),
        idx=idx,
        device=CPU,
        family=resolve_family(cfg["output_distribution"]),
        **(_body_kwarg(cfg, criterion_reg) if use_body else {}),
        **kwargs,
    )
    out["total"].backward()
    return {
        "out_reg": _digest(first.reg),
        "per_task": _digest(out["per_task"]),
        "total": _digest(out["total"].reshape(1)),
        "grad_reg_head1": _digest(model.dec_conv4_head1_reg.weight.grad),
        "grad_reg_head3": _digest(model.dec_conv4_head3_reg.weight.grad),
        "grad_first_conv": _digest(next(model.parameters()).grad),
    }


def _same(got: dict, want: dict, *, exact: bool) -> bool:
    if got["n"] != want["n"]:
        return False
    if exact:
        return got["sha256"] == want["sha256"]
    a = torch.tensor([got["sum"], got["abs_sum"], *got["head"]], dtype=torch.float64)
    b = torch.tensor([want["sum"], want["abs_sum"], *want["head"]], dtype=torch.float64)
    return bool(torch.allclose(a, b, rtol=1e-5, atol=1e-7))


@pytest.mark.parametrize("case", list(CASES))
def test_one_training_step_reproduces_the_golden(case):
    golden = json.loads(GOLDEN.read_text())
    assert case in golden["cases"], f"golden has no case {case!r} — regenerate it"
    exact = golden["recorded_on"] == _fingerprint()
    got, want = _run(case), golden["cases"][case]
    for key in want:
        assert _same(got[key], want[key], exact=exact), (
            f"{case}: {key} differs from the golden "
            f"({'bit-for-bit' if exact else 'rtol 1e-5, another machine'}). "
            f"got sum {got[key]['sum']!r}, golden sum {want[key]['sum']!r}. "
            "Epic #616 claims training is numerically unchanged; this is the check of that claim."
        )


@pytest.mark.parametrize("body_name", list(BODIES))
def test_every_live_body_gets_an_object_so_the_golden_runs_the_new_path(body_name):
    """Without this the cases above could fall back to the older path and prove nothing."""
    cfg = {**BASE, **BODIES[body_name]}
    kwarg = _body_kwarg(cfg, choose_loss(cfg, CPU)[0])
    assert "body" in kwarg, "`_process_sequence` has no `body` argument"
    assert kwarg["body"] is not None, f"{body_name} got no body object"


@pytest.mark.parametrize("case", list(CASES))
def test_the_body_path_equals_the_older_path_in_one_process(case):
    """The same claim as the golden, without the golden: both paths, one process, any CPU."""
    with_body, without = _run(case, use_body=True), _run(case, use_body=False)
    for key in with_body:
        assert with_body[key]["sha256"] == without[key]["sha256"], (
            f"{case}: {key} differs between the body path and the older path "
            f"(sums {with_body[key]['sum']!r} vs {without[key]['sum']!r})"
        )


def test_the_golden_covers_all_five_live_bodies():
    golden = json.loads(GOLDEN.read_text())
    assert set(golden["cases"]) == set(CASES)
    assert {CASES[c][0] for c in CASES} == set(BODIES)
    assert len(BODIES) == 5


def test_the_comparison_can_fail():
    """A golden that cannot fail pins nothing: a 1e-6 relative change must be seen, both ways."""
    d = _digest(torch.tensor([1.0, 2.0, 3.0]))
    moved = _digest(torch.tensor([1.0, 2.0, 3.0]) * (1 + 1e-6))
    assert not _same(moved, d, exact=True)
    far = _digest(torch.tensor([1.0, 2.0, 3.0]) * (1 + 1e-3))
    assert not _same(far, d, exact=False)
    assert _same(d, d, exact=True) and _same(d, d, exact=False)


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("usage: python tests/test_body_golden.py --write")
    import views_hydranet

    _repo = Path(__file__).resolve().parent.parent
    if _repo not in Path(views_hydranet.__file__).resolve().parents:
        sys.exit(
            f"refusing to record: views_hydranet was imported from {views_hydranet.__file__}, "
            f"not from {_repo}. Run with PYTHONPATH=."
        )
    GOLDEN.parent.mkdir(exist_ok=True)
    payload = {
        "what": "One _process_sequence step per live body; see tests/test_body_golden.py (#617).",
        "recorded_on": _fingerprint(),
        "cases": {case: _run(case) for case in CASES},
    }
    GOLDEN.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {GOLDEN} ({len(payload['cases'])} cases) on {payload['recorded_on']}")
