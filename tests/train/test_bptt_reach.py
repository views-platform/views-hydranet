"""How far back in time gradient actually travels — and that nothing truncates it.

There is **no BPTT truncation anywhere in the package**. ``training_engine.py:350`` rebinds the
recurrent state with no ``detach``::

    t1_pred, t1_pred_class, h = output.reg, output.cls, output.h_next

``total_loss += loss`` accumulates every step, and there is exactly one ``.backward()`` in the
whole codebase (``:1016``), called once per *window* over all ``seq_len - 1`` steps. No
``retain_graph``, no ``torch.utils.checkpoint``, no chunking. ``VolumeSampler._generate_window``
slices space only, so ``seq_len`` is the full training time axis — ~383 steps in a production run.

That is a deliberate design with a real cost (the whole graph is resident until backward), and
nothing tested it. Worse, nothing measured whether the reach is *used*: ``max_raw_grad_norm``
(``:1053-1060``) only watches for explosion, never for vanishing.

**Measured 2026-08-26**, ``d||h_final||^2 / dx_i`` at T=120 — frame ``i`` can reach ``h_final``
only through memory, so this isolates the recurrent path from each frame's own forward step:

===================  ==================  =====================
steps back           random init         trained L=300 incumbent
===================  ==================  =====================
0                    1.14e+01            5.56e+01
20                   1.40e-03            3.94e+00
60                   2.98e-09            2.65e-01
118                  2.81e-17            1.56e-02
===================  ==================  =====================

Two things follow. The untrained recurrence is sharply contractive and geometric — the
``MillerHardt2019`` stable regime, where a truncated feed-forward model would approximate it and
the memory is decorative. **Training escapes that regime**: the trained decay is sub-exponential
and retains ~1e14 times more gradient at 118 steps back. So the full BPTT graph is *not*
decorative, and the M46 ``WideMemory`` null is **not** explained by gradient failing to reach the
recurrence. Whatever caps that arm, it is not vanishing gradient.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.train.training_engine import _process_sequence  # noqa: E402

from .conftest import SumLoss, make_sequence  # noqa: E402


def _recurrent_reach(T: int, hw: int = 12, seed: int = 3):
    """``d||h_final||^2 / dx_i`` per input frame — the pure memory path.

    Deliberately NOT ``d(total_loss)/dx_i``: every frame is also the direct input to its own step,
    so that quantity is dominated by the feed-forward path and says nothing about memory.
    """
    x, model, h0, idx, family = make_sequence(seed=seed, T=T, hw=hw)
    x = x.detach().requires_grad_(True)
    out = _process_sequence(
        x,
        model,
        h0,
        criterion_reg=FamilyLoss(family),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=SumLoss(),
        idx=idx,
        device=torch.device("cpu"),
        family=family,
        ss_feedback="mean",
        forecast_composition="soft_gate",
    )
    (grad,) = torch.autograd.grad((out["h"] ** 2).sum(), x)
    return grad[0].flatten(1).norm(dim=1)


def test_the_recurrence_is_never_truncated():
    """The FIRST frame must still influence the LAST hidden state.

    This is the guard on the untruncated design. Inserting a ``.detach()`` anywhere in the state
    carry — the single most plausible "optimisation" someone reaches for when this graph runs out
    of memory — zeroes the early frames and fails here.
    """
    norms = _recurrent_reach(T=24)
    assert norms[0].item() > 0.0, (
        "frame 0 has NO gradient path to the final hidden state: the recurrence is being "
        "truncated. If that was intentional, this test is the place to say so."
    )
    dead = (norms[:-1] == 0).nonzero().flatten().tolist()
    assert not dead, f"frames with a severed gradient path to h_final: {dead}"


def test_gradient_survives_far_enough_back_to_justify_the_full_graph():
    """The reach must be long, or the untruncated graph is paying memory for nothing.

    Threshold set two orders below the measured value at this length, so it catches a regime
    change (a return to the contractive random-init behaviour, which is ~1e-17 at 118 steps) and
    not ordinary seed-to-seed variation.
    """
    T = 40
    norms = _recurrent_reach(T=T)
    ratio = (norms[0] / norms[T - 2]).item()
    assert ratio > 1e-9, (
        f"gradient at {T - 2} steps back is {ratio:.3e} of the immediate one. The recurrence has "
        "collapsed into the strongly contractive regime, where truncated BPTT would be free and "
        "the memory is decorative. That is a finding, not a flaky test — investigate."
    )


def test_reach_decays_monotonically_with_distance():
    """Sanity on the measurement itself: further back must not carry MORE gradient.

    Compares block means rather than adjacent frames, because per-frame values are noisy in the
    sparse field; a violation here means the probe is measuring something other than reach.
    """
    norms = _recurrent_reach(T=40)
    near = norms[-6:-1].mean().item()  # closest to the end
    far = norms[:5].mean().item()  # the very start of the sequence
    assert far < near, (
        f"early frames ({far:.4e}) carry more gradient to h_final than late ones ({near:.4e}); "
        "the reach probe is not measuring the recurrent path"
    )


def test_the_whole_sequence_is_one_graph_not_per_step_backwards():
    """One graph, one backward. A per-step ``backward()`` would free the graph and raise here.

    ``_process_sequence`` returns a ``total`` that must still be differentiable across the entire
    sequence; calling backward twice without ``retain_graph`` is how we detect that the returned
    scalar is a live accumulation rather than a detached sum of already-freed pieces.
    """
    x, model, h0, idx, family = make_sequence(seed=5, T=8, hw=8)
    out = _process_sequence(
        x,
        model,
        h0,
        criterion_reg=FamilyLoss(family),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=SumLoss(),
        idx=idx,
        device=torch.device("cpu"),
        family=family,
        ss_feedback="mean",
        forecast_composition="soft_gate",
    )
    assert out["total"].requires_grad, "the accumulated total is detached — nothing can train"
    out["total"].backward()
    with pytest.raises(RuntimeError, match="second time|freed"):
        out["total"].backward()
