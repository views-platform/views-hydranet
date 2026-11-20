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
it falls back to a tolerance of 1e-4 of each tensor's mean absolute value, and says so. (A first
version used `rtol=1e-5, atol=1e-7` per value and failed on the CI runner: 7 of 9 cases, all in
the one gradient that passes through the whole rollout, by about 2e-6.)

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

from views_hydranet.body.bodies import build_body  # noqa: E402
from views_hydranet.body.vocabulary import body_name  # noqa: E402
from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices  # noqa: E402
from views_hydranet.utils.utils import choose_loss, choose_model  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "body_golden.json"
#: Every parameter's gradient per case, recorded from the older route before #629 deleted it.
GOLDEN_GRAD_ALL = Path(__file__).parent / "golden" / "body_grad_all_golden.json"
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
}

#: The five live bodies, by the config values that select them today (ADR-079 §2).
BODIES = {
    "mse": {"body": "mse_softplus_point"},
    "ranking": {
        "body": "listwise_softplus_point",
        "loss_reg_y_max": 10000.0,
    },
    "nb": {"body": "nll_softplus_nb"},
    "truncated_nb": {"body": "nll_softplus_truncated_nb"},
    "mixture_nb": {"body": "nll_softplus_mixture_nb"},
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


#: The one body whose full gradient is compared with a tolerance against the older route's
#: recording, not bit for bit. Pinned by an assertion in the full-gradient test.
_GRAD_ALL_TOLERANT = {"mixture_nb"}

#: Tolerance on a machine other than the one the golden was recorded on; see `_same`.
_OTHER_MACHINE_RTOL = 1e-4


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
    # The thread count is part of it: the same CPU gives different float32 bits with a different
    # number of threads (measured 2026-11-16: 1, 2, 6 and 8 threads all differ from the default).
    return (
        f"torch {torch.__version__} | {platform.machine()} | {cpu} | "
        f"{torch.get_num_threads()} threads"
    )


def _digest(t: torch.Tensor) -> dict:
    flat = t.detach().to(torch.float32).contiguous().reshape(-1)
    return {
        "sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
        "n": int(flat.numel()),
        "sum": float(flat.double().sum()),
        "abs_sum": float(flat.double().abs().sum()),
        "head": [float(v) for v in flat[:24]],
    }


def _run(case: str) -> dict:
    case_body, kwargs = CASES[case]
    cfg = {**BASE, **BODIES[case_body]}
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
        family=resolve_family(body_name(cfg).distribution),
        body=build_body(cfg, criterion_reg),
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
        # Every parameter's gradient. In its own golden file, recorded from the older route.
        "grad_all": _digest(
            torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        ),
    }


def _same(got: dict, want: dict, *, exact: bool) -> bool:
    if got["n"] != want["n"]:
        return False
    if exact:
        return got["sha256"] == want["sha256"]
    # Another machine. Float32 noise is proportional to the size of the tensor's values, not to
    # each value: an entry near zero carries the same absolute noise as a large one. So the
    # tolerance is 1e-4 of the tensor's mean absolute value (and of the absolute sum, for the
    # signed sum, which cancels). Measured on the CI runner, 2026-11-16: up to 2e-6 relative, in
    # the gradient that has passed through the whole recurrent rollout.
    mean_abs = want["abs_sum"] / max(want["n"], 1)
    # Written as `not (difference <= tolerance)` so that NaN, which compares False, is a mismatch.
    if not abs(got["sum"] - want["sum"]) <= _OTHER_MACHINE_RTOL * want["abs_sum"]:
        return False
    if not abs(got["abs_sum"] - want["abs_sum"]) <= _OTHER_MACHINE_RTOL * want["abs_sum"]:
        return False
    return all(
        abs(g - w) <= _OTHER_MACHINE_RTOL * (abs(w) + mean_abs)
        for g, w in zip(got["head"], want["head"], strict=True)
    )


@pytest.mark.parametrize("case", list(CASES))
def test_one_training_step_reproduces_the_golden(case):
    golden = json.loads(GOLDEN.read_text())
    assert case in golden["cases"], f"golden has no case {case!r} — regenerate it"
    exact = golden["recorded_on"] == _fingerprint()
    got, want = _run(case), golden["cases"][case]
    for key in want:
        assert _same(got[key], want[key], exact=exact), (
            f"{case}: {key} differs from the golden "
            f"({'bit-for-bit' if exact else 'tolerance 1e-4, another machine'}). "
            f"got sum {got[key]['sum']!r}, golden sum {want[key]['sum']!r}. "
            "Epic #616 claims training is numerically unchanged; this is the check of that claim."
        )


@pytest.mark.parametrize("case", list(CASES))
def test_every_gradient_matches_the_older_path_as_recorded(case):
    """Every parameter's gradient against the older route's, as recorded before #629 deleted it.

    Until #629 this ran both routes in one process. The older route is gone, so its numbers were
    recorded on 2026-11-20 at the commit before the deletion, after checking there that the body
    route reproduced them (bit for bit; the mixture body within the tolerance below).
    """
    golden = json.loads(GOLDEN_GRAD_ALL.read_text())
    assert set(golden["cases"]) == set(CASES), "the full-gradient golden lost a case"
    assert _GRAD_ALL_TOLERANT == {"mixture_nb"}, (
        "widening the exception is a decision, not an edit"
    )
    exact = golden["recorded_on"] == _fingerprint()
    got, want = _run(case)["grad_all"], golden["cases"][case]
    if CASES[case][0] in _GRAD_ALL_TOLERANT:
        # The one measured exception (code review, 2026-11-16): on this 8 x 8 grid the mixture
        # body's three head-bias gradients differ by up to 3e-8 between the two routes.
        exact = False
    assert _same(got, want, exact=exact), (
        f"{case}: a parameter's gradient differs from the older route's recording "
        f"({'bit-for-bit' if exact else 'tolerance 1e-4'}); sums {got['sum']!r} vs "
        f"{want['sum']!r}"
    )


def test_the_ranking_body_is_trained_on_all_cells_even_under_active_supervision(monkeypatch):
    """`requires_all_cells` must be read by the training step, not only declared.

    A guard audit (2026-11-16) found that ignoring the flag in `_process_sequence` left every test
    green. With the flag read, `body_supervision='active'` changes nothing for the ranking body.
    """
    monkeypatch.setitem(CASES, "ranking/active", ("ranking", {"body_supervision": "active"}))
    active, all_cells = _run("ranking/active"), _run("ranking/all")
    for key in all_cells:
        assert active[key]["sha256"] == all_cells[key]["sha256"], (
            f"{key}: the ranking body was trained on a subset of the cells"
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
    assert _same(moved, d, exact=False), "float noise of 1e-6 must pass on another machine"
    # NaN anywhere must be a mismatch on another machine too. Beyond the first 24 values it is
    # seen only through the two sums, so each sum is tried alone as well.
    ones = _digest(torch.ones(31))
    late_nan = _digest(torch.cat([torch.ones(30), torch.tensor([float("nan")])]))
    assert late_nan["head"] == ones["head"], "the NaN must lie beyond the stored values"
    assert not _same(late_nan, ones, exact=False) and not _same(late_nan, ones, exact=True)
    for key in ("sum", "abs_sum"):
        assert not _same({**ones, key: float("nan")}, ones, exact=False), f"NaN in {key} passed"


def test_the_fingerprint_names_the_thread_count_and_changes_with_it():
    """The same CPU gives other float32 bits with another thread count (measured 2026-11-16)."""
    before = torch.get_num_threads()
    assert f"| {before} threads" in _fingerprint()
    try:
        torch.set_num_threads(before + 1)
        assert f"| {before + 1} threads" in _fingerprint()
    finally:
        torch.set_num_threads(before)


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
