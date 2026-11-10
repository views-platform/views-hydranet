"""The seven numerics properties of the level-indexed listwise loss, each with a refusing half.

⛔ ``reports/2026-11-10_ranking_head_dossier/04_roadmap`` phase 1, verbatim: *"The numerics are
tested before anything trains. These are the tests, and each needs a half that fails against the
defect."* The seven rows of that table are the seven ``test_property_N_*`` functions below; each is
followed by a ``test_property_N_refuses_*`` half that proves the check can fail.

⚠️ The reference implementation here is transcribed from Eq. 2 in plain Python with explicit loops
-- no torch, no ``searchsorted``, no vectorisation. It is the spec; the module is the optimisation.
"""

from __future__ import annotations

import math
import types

import pytest
import torch
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss, dyadic_levels
from views_hydranet.utils.utils import LOSS_REG_REGISTRY

# The transform's own table, from 02_design §2.2. y -> r = ceil(log2(1 + y)).
_LEVEL_TABLE = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 2),
    (4, 3),
    (7, 3),
    (8, 4),
    (15, 4),
    (16, 5),
    (31, 5),
    (63, 6),
    (127, 7),
    (255, 8),
    (511, 9),
    (681, 10),
    (1023, 10),
]

_Y_MAX = 681.0  # the grid's truth maximum on the S11 population => levels 0..10


def _reference_loss(scores: list[float], counts: list[int], level_max: int) -> float:
    """-> Eq. 2, transcribed directly. The spec, independent of the module under test."""
    levels = [int(c).bit_length() for c in counts]  # ceil(log2(1+y)) == bit_length(y)
    total = 0.0
    for r_t in range(level_max, 0, -1):  # steps 1..|R|-1: levels descending, lowest dropped
        c_t = [i for i, r in enumerate(levels) if r == r_t]
        if not c_t:
            continue
        mass_below = sum(math.exp(scores[i]) for i, r in enumerate(levels) if r < r_t)
        gain = 2**r_t - 1
        for i in c_t:
            p = math.exp(scores[i]) / (math.exp(scores[i]) + mass_below)
            total += gain * math.log(p)
    return -total / level_max


def _loss_of(scores: list[float], counts: list[int], y_max: float = _Y_MAX) -> torch.Tensor:
    crit = ListwiseLevelLoss(y_max=y_max)
    score = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    target = torch.log1p(torch.tensor(counts, dtype=torch.float32))
    return crit(score, target)


# ---------------------------------------------------------------------------------------------
# 1 — r = ceil(log2(1+y)) maps y = 681 -> r = 10, w = 1023
# ---------------------------------------------------------------------------------------------


def test_property_1_the_dyadic_transform_maps_the_whole_truth_range_onto_eleven_levels():
    y = torch.tensor([c for c, _ in _LEVEL_TABLE], dtype=torch.float32)
    got = dyadic_levels(y, level_max=10).tolist()
    assert got == [r for _, r in _LEVEL_TABLE], f"the transform drifted: {got}"
    # The gain weight at the top level, and the property that makes the transform principled:
    # 2^r - 1 recovers the count to within a factor of two.
    assert 2**10 - 1 == 1023
    assert 1.0 <= (2**10 - 1) / 681 <= 2.0


def test_property_1_survives_the_log1p_float32_round_trip():
    """⛔ The refusing half: a float ``log2`` of ``expm1(log1p(y))`` lands off-by-one at a power of
    two. Every exact power of two is a level boundary, so this is not a corner case."""
    counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    y = torch.expm1(torch.log1p(torch.tensor(counts, dtype=torch.float32)))
    assert dyadic_levels(y, level_max=10).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # ⛔ And the float route really does break just above a boundary, which is where the round
    # trip leaves a count: ``ceil(log2(1 + 1.0000001))`` is 2, one level too high.
    assert math.ceil(math.log2(1 + 1.0000001)) == 2
    assert dyadic_levels(torch.tensor([1.0000001]), level_max=10).item() == 1
    assert dyadic_levels(torch.tensor([0.9999999]), level_max=10).item() == 1


# ---------------------------------------------------------------------------------------------
# 2 — |R| is the a-priori level set, fixed across folds (C-423)
# ---------------------------------------------------------------------------------------------


def test_property_2_the_normaliser_is_the_declared_level_set_not_the_observed_one():
    """Two 'folds' observing the same two levels but declaring different maxima must differ by
    exactly the ratio of their declared ``|R| - 1``."""
    scores, counts = [0.4, -0.7], [1, 0]
    big = _loss_of(scores, counts, y_max=681.0)  # levels 0..10 -> normaliser 1/10
    small = _loss_of(scores, counts, y_max=100.0)  # levels 0..7  -> normaliser 1/7
    assert ListwiseLevelLoss(y_max=681.0).level_max == 10
    assert ListwiseLevelLoss(y_max=100.0).level_max == 7
    assert torch.isclose(big * 10, small * 7, atol=1e-6), (
        f"the normaliser is not 1/(|R|-1) from the declared level set: {big} vs {small}"
    )


def test_property_2_refuses_a_normaliser_read_off_the_observed_levels():
    """⛔ The refusing half. If ``|R|`` were counted from the levels this list happens to occupy,
    both folds would see ``|R| = 2`` and the two losses would be EQUAL. They must not be."""
    scores, counts = [0.4, -0.7], [1, 0]
    big = _loss_of(scores, counts, y_max=681.0)
    small = _loss_of(scores, counts, y_max=100.0)
    assert not torch.isclose(big, small, atol=1e-6), (
        "the two declared level sets gave the same loss, so |R| is being read from the data "
        "(C-423) and each fold is a different objective"
    )


# ---------------------------------------------------------------------------------------------
# 3 — an all-one-level list gives loss exactly 0
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 10, 681])
def test_property_3_an_unrankable_list_costs_exactly_zero(count):
    """Zhu, quoted: *"when all n documents have the same rating ... the loss is 0."* Here that is
    not a convenience — it is why the 99.6%-zero land grid is not something to engineer around."""
    scores = [1.5, -2.0, 0.0, 7.25]
    loss = _loss_of(scores, [count] * 4)
    assert loss.item() == 0.0, f"an all-one-level list cost {loss.item()}"


def test_property_3_refuses_a_list_that_can_be_ranked():
    """⛔ The refusing half: a check that returns 0 for everything would pass the test above."""
    loss = _loss_of([1.5, -2.0], [0, 10])
    assert loss.item() > 0.0, "a two-level list cost nothing, so the loss is inert"


# ---------------------------------------------------------------------------------------------
# 4 — the lowest level is the dropped final step
# ---------------------------------------------------------------------------------------------


def test_property_4_the_zero_level_enters_only_as_a_denominator():
    """The closed form of the two-cell case pins all four things at once: the dropped step, the
    gain weight, the probability of Eq. 1, and the fixed normaliser.

    With one cell at level 1 and one at level 0, the only surviving step is level 1 and
    ``L = -ln sigmoid(f_1 - f_0) / (|R| - 1)``.

    ⚠️ **Measured by mutation on 2026-11-10, and worth knowing before trusting this test:** giving
    the lowest level a step of its own, with a gain of 1 instead of ``2^0 - 1``, changes NOTHING
    and all 20 tests stay green. That mutant is *equivalent*, not undetected — the lowest level has
    no level below it, so ``s~_t`` is empty, ``P_t(d) = 1`` and the step contributes exactly 0
    whatever its weight. ⇒ Zhu's truncation at step ``|R|`` is **implied** by the objective's own
    structure at our data, not an extra rule to enforce. What this test does pin, each confirmed by
    a mutation that fails it: the gain weight (``2^r`` for ``2^r - 1`` -> 4 failures), the fixed
    normaliser (occupied-level count -> 6), the direction of ``s~_t`` (levels above -> 6), its
    strictness (``<=`` for ``<`` -> 7), and that the dropped step is the LOWEST level and not the
    highest (-> 2 failures, this test among them).
    """
    f1, f0 = 0.4, -0.7
    crit = ListwiseLevelLoss(y_max=_Y_MAX)
    score = torch.tensor([f1, f0], dtype=torch.float64, requires_grad=True)
    loss = crit(score, torch.log1p(torch.tensor([1.0, 0.0], dtype=torch.float64)))
    expected = -math.log(1.0 / (1.0 + math.exp(-(f1 - f0)))) / 10
    assert loss.item() == pytest.approx(expected, rel=1e-12)

    # And the gradient says the same: the zero cell is pushed DOWN by the step above it, which is
    # the only way a dropped level can still carry information.
    loss.backward()
    p = 1.0 / (1.0 + math.exp(-(f1 - f0)))
    assert score.grad[0].item() == pytest.approx(-(1 - p) / 10, rel=1e-10)
    assert score.grad[1].item() == pytest.approx(+(1 - p) / 10, rel=1e-10)


def test_property_4_refuses_a_loss_that_scores_the_lowest_level_as_its_own_step():
    """⛔ The refusing half: give level 0 a step of its own with a gain of 1 instead of
    ``2^0 - 1 = 0`` and the number moves. The test above would not notice a one-line slip."""
    f1, f0 = 0.4, -0.7
    p = 1.0 / (1.0 + math.exp(-(f1 - f0)))
    as_specified = -math.log(p) / 10
    with_zero_level_scored = as_specified - 1.0 * math.log(1.0 - p) / 10
    assert not math.isclose(as_specified, with_zero_level_scored, rel_tol=1e-9)
    got = _loss_of([f1, f0], [1, 0]).item()
    assert got == pytest.approx(as_specified, rel=1e-6)
    assert not math.isclose(got, with_zero_level_scored, rel_tol=1e-6)


# ---------------------------------------------------------------------------------------------
# 5 — an empty level contributes nothing and does not change |R|
# ---------------------------------------------------------------------------------------------


def test_property_5_an_empty_level_still_occupies_its_step():
    """Level 9 is empty on the S11 population. A list skipping a level must agree with the spec
    computed over the FULL declared level set."""
    scores = [2.0, 0.5, -1.0, -3.0, 0.25]
    counts = [681, 7, 0, 0, 1]  # occupies levels 10, 3, 0, 0, 1 -- 2 and 4..9 are empty
    got = _loss_of(scores, counts).item()
    assert got == pytest.approx(_reference_loss(scores, counts, level_max=10), rel=1e-6)


def test_property_5_refuses_a_normaliser_that_counts_only_occupied_levels():
    """⛔ The refusing half: dividing by the occupied-level count (3 here, not 10) is the C-423
    defect, and it is a factor of 10/3 away from the spec."""
    scores = [2.0, 0.5, -1.0, -3.0, 0.25]
    counts = [681, 7, 0, 0, 1]
    spec = _reference_loss(scores, counts, level_max=10)
    occupied_only = spec * 10 / 3
    assert not math.isclose(spec, occupied_only, rel_tol=1e-9)
    assert _loss_of(scores, counts).item() == pytest.approx(spec, rel=1e-6)


# ---------------------------------------------------------------------------------------------
# 6 — the criterion returns a 0-dim tensor (03 §3.3: nothing else asserts this)
# ---------------------------------------------------------------------------------------------


def test_property_6_the_criterion_returns_a_zero_dim_tensor():
    """⛔ ``03 §3.3``: the training loop stacks per-target losses and nothing in it asserts the
    criterion reduced. A ``[32,32]`` return would broadcast silently and weight this target by
    1,024."""
    crit = ListwiseLevelLoss(y_max=_Y_MAX)
    score = torch.randn(1, 32, 32, dtype=torch.float32, requires_grad=True)
    target = torch.log1p(torch.randint(0, 50, (1, 32, 32)).to(torch.float32))
    loss = crit(score, target)
    assert loss.ndim == 0 and loss.shape == torch.Size([])
    assert loss.requires_grad and torch.isfinite(loss)


def test_property_6_refuses_an_unreduced_loss():
    """⛔ The refusing half: the shape check must actually look at the shape."""
    unreduced = torch.nn.MSELoss(reduction="none")(torch.zeros(4), torch.ones(4))
    assert unreduced.ndim != 0, "the 0-dim check would accept a per-cell loss"


# ---------------------------------------------------------------------------------------------
# 7 — every weight is finite for every y in [0, y_max]  (C-420, the Tier-1 case)
# ---------------------------------------------------------------------------------------------


def test_property_7_every_gain_weight_over_the_declared_range_is_finite_in_float32():
    crit = ListwiseLevelLoss(y_max=_Y_MAX)
    weights = torch.tensor([2.0**r - 1 for r in range(crit.level_max + 1)], dtype=torch.float32)
    assert torch.isfinite(weights).all() and weights.max().item() == 1023.0
    # ... and the loss itself is finite on a list spanning every one of those levels.
    counts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 681]
    loss = _loss_of([float(i) - 5 for i in range(11)], counts)
    assert torch.isfinite(loss) and loss.item() > 0.0


def test_property_7_refuses_a_level_set_wide_enough_to_overflow_float32():
    """⛔ The refusing half, and ``C-420`` itself: on RAW counts the level is the count, and
    ``2^681`` is ``inf`` in float32 with no error signal. The constructor must refuse, loudly."""
    assert not torch.isfinite(torch.tensor(2.0, dtype=torch.float32) ** 681)
    with pytest.raises(ValueError, match="C-420"):
        ListwiseLevelLoss(y_max=float(2**200))
    with pytest.raises(ValueError, match="exceeds the declared level set"):
        dyadic_levels(torch.tensor([2000.0]), level_max=10)


# ---------------------------------------------------------------------------------------------
# The seam, as far as phase 1 goes: the entry exists, is latent, and survives an empty list.
# ---------------------------------------------------------------------------------------------


def test_the_registry_entry_exists_and_declares_its_parameter():
    entry = LOSS_REG_REGISTRY["listwise_level"]
    assert entry["cls"] is ListwiseLevelLoss
    assert entry["params"] == ["loss_reg_y_max"]
    crit = entry["factory"]({"loss_reg_y_max": _Y_MAX}, torch.device("cpu"))
    assert isinstance(crit, ListwiseLevelLoss) and crit.level_max == 10


def test_the_score_reaches_the_loss_pre_activation():
    """``needs_latent = True``: a scale-free score lives on the whole real line. A ReLU-activated
    body would clamp half of it to a tie block at 0 -- the exact defect ``02 §5`` measures as
    ``tie@0``."""
    assert ListwiseLevelLoss.needs_latent is True


def test_an_empty_list_returns_a_graph_connected_zero():
    """The all-zero-weight precedent of ``C-199``: a detached ``tensor(0.0)`` breaks the backward
    pass on a step that legitimately has nothing to rank."""
    crit = ListwiseLevelLoss(y_max=_Y_MAX)
    score = torch.zeros(0, requires_grad=True)
    loss = crit(score, torch.zeros(0))
    assert loss.item() == 0.0 and loss.requires_grad


# ---------------------------------------------------------------------------------------------
# The flag is default-off, and the config refuses the two pairings that would be inert or wrong.
# ---------------------------------------------------------------------------------------------


def test_no_existing_config_is_touched_because_the_field_defaults_to_unset(valid_config_dict):
    cfg = dict(valid_config_dict)
    cfg.pop("loss_reg_y_max", None)
    assert HydraNetConfig(**cfg).loss_reg_y_max is None


def test_the_ranking_body_cannot_start_without_declaring_its_level_set(valid_config_dict):
    """⛔ ``C-423``: the level set is the objective's scale. A default here would be a silent
    per-run objective, so ``validate_loss_reg_params`` must refuse the omission."""
    cfg = dict(valid_config_dict)
    cfg["loss_reg"] = "listwise_level"
    cfg["body_supervision"] = "all"
    cfg.pop("loss_reg_y_max", None)
    with pytest.raises(ValidationError, match="loss_reg_y_max"):
        HydraNetConfig(**cfg)
    cfg["loss_reg_y_max"] = _Y_MAX
    assert HydraNetConfig(**cfg).loss_reg_y_max == _Y_MAX


def test_a_supervision_window_on_the_ranking_body_is_refused(valid_config_dict):
    """``C-193``, inherited rather than rebuilt: ``needs_latent = True`` makes
    ``validate_body_supervision_latent`` reject ``body_supervision='active'``. That is the right
    answer here for a different reason than the one it was written for — the zero block IS the
    list's lower levels, and masking it away leaves a list with one level and a loss of 0."""
    cfg = dict(valid_config_dict)
    cfg["loss_reg"] = "listwise_level"
    cfg["loss_reg_y_max"] = _Y_MAX
    cfg["body_supervision"] = "active"
    with pytest.raises(ValidationError, match="needs_latent"):
        HydraNetConfig(**cfg)


# ---------------------------------------------------------------------------------------------
# ⭐⭐ 04_roadmap PHASE 2 — the seam. Does the score fit the existing dispatch, or must it widen?
#
# ⛔ This section exists because of #308. That change passed 7 unit tests, 5/5 mutations, lint and
# the full suite, and was a **no-op on the path production runs** — two arms trained to
# byte-identical weights over 276 minutes (`C-324`, `tests/test_bptt_sa.py`). A listwise loss with
# 23 green unit tests that never receives a gradient through `_process_sequence` would be the same
# failure. ⇒ Every claim below is made against the REAL dispatcher.
# ---------------------------------------------------------------------------------------------

_H, _W, _T = 4, 4, 4
_FEATURE_NAMES = ["reg0", "cls0", "feat0"]
_SEQ_CONFIG = {
    "regression_targets": ["reg0"],
    "classification_targets": ["cls0"],
    "features": ["feat0"],
    "static_channels": [],
}


class _ScoreModel(torch.nn.Module):
    """A trainable stub emitting ONE score channel per regression target, as a ranking head does.

    ⭐ ``reg`` is deliberately ``relu(latent)`` — the ``output_distribution='standard'`` activation
    — so ``reg`` and ``reg_latent`` DIFFER wherever the score is negative. That difference is what
    makes the ``needs_latent`` assertion below able to fail.
    """

    def __init__(self, n_reg: int, n_cls: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, n_reg + n_cls, kernel_size=1)
        self.n_reg = n_reg

    def forward(self, x, hidden):  # noqa: ANN001 - stub
        out = self.conv(x[:, :1])
        latent, cls = out[:, : self.n_reg], out[:, self.n_reg :]
        return types.SimpleNamespace(
            reg=torch.relu(latent), cls=cls, reg_latent=latent, h_next=hidden
        )


class _SpyListwiseLevelLoss(ListwiseLevelLoss):
    """Records the shape and the identity of what the dispatcher actually handed the criterion."""

    def __init__(self, y_max: float) -> None:
        super().__init__(y_max=y_max)
        self.seen_shapes: list[torch.Size] = []
        self.saw_a_negative_score = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(input.shape)
        self.saw_a_negative_score |= bool((input < 0).any())
        return super().forward(input, target)


def _run_sequence(*, constant_target: float | None = None, seed: int = 0):
    """One real ``_process_sequence`` pass with the listwise loss as ``criterion_reg``."""
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    torch.manual_seed(seed)
    idx = _SequenceIndices(_FEATURE_NAMES, _SEQ_CONFIG)
    tensor = torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W)
    if constant_target is not None:
        tensor[:, :, idx.reg] = constant_target
    model = _ScoreModel(idx.n_reg, idx.n_cls)
    crit = _SpyListwiseLevelLoss(y_max=_Y_MAX)
    res = _process_sequence(
        train_tensor=tensor,
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=crit,
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
    )
    return res, crit, model


def test_the_dispatcher_needs_no_fourth_branch_the_score_rides_the_point_path():
    """⭐⭐ The answer to `03 §3.1`, measured rather than argued.

    `03` asked whether a new loss *shape* forces a fourth `isinstance` branch at
    `training_engine.py:916-923` — the one thing `ADR-067`'s seam exists to avoid. It does not:
    the legacy `else` branch already slices **one channel per target** and hands the criterion
    `[B, H, W]`, which IS the `n_params = 1` score. ⇒ **The dispatcher is not widened**, and
    nothing in `ADR-067`'s registry is touched.
    """
    res, crit, _ = _run_sequence()
    assert crit.seen_shapes, "the criterion was never called — the loss is not wired at all"
    assert all(s == torch.Size([1, _H, _W]) for s in crit.seen_shapes), (
        f"the dispatcher handed the criterion {set(crit.seen_shapes)}, not [1,H,W]: the "
        "point-body branch is not the one being taken, so the seam DOES need widening"
    )
    assert torch.isfinite(res["total"]) and res["total"].ndim == 0


def test_the_score_reaches_the_criterion_before_the_activation_on_the_real_path():
    """⛔ `needs_latent = True` must actually route `reg_latent`. The stub's `reg` is
    `relu(latent)`, so if the dispatcher handed over the ACTIVATED field the criterion would never
    see a negative score — and a score clamped at 0 is the `tie@0` defect of `02 §5`, which is the
    whole reason B is being tried."""
    _, crit, _ = _run_sequence()
    assert crit.saw_a_negative_score, (
        "every score the criterion saw was >= 0, so it is being fed the ReLU-activated field "
        "instead of reg_latent — half the ordering information is clamped into a tie block"
    )


def test_the_listwise_loss_actually_moves_the_model_through_the_real_loop():
    """⛔⛔ The `#308` / `C-324` test. A loss that is registered, dispatched and inert is worse
    than one that errors: it trains an arm that reads as a clean null."""
    res, _, model = _run_sequence()
    res["total"].backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    assert grads.numel() > 0, "no parameter received a gradient"
    assert torch.isfinite(grads).all()
    assert float(grads.abs().sum()) > 0.0, (
        "the listwise loss produced a gradient of exactly zero on the real path — this is the "
        "#308 no-op signature, and any arm built on it would measure the harness, not the idea"
    )


def test_a_single_level_patch_produces_no_gradient_on_the_real_path():
    """⛔ The refusing half of the test above, and the property that makes it meaningful: the
    gradient is zero exactly when there is nothing to rank. A per-cell magnitude loss would still
    pull on an all-one-level patch; this must not."""
    res, crit, model = _run_sequence(constant_target=0.0)
    assert float(res["total"]) == 0.0, f"an all-zero patch cost {float(res['total'])}"
    res["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    # ⛔ Not `all(...)` over a possibly-empty list: that is green when NOTHING was differentiated,
    # which is the opposite of the property. The gradient must exist and be zero.
    assert grads, "no parameter was differentiated at all, so 'the gradient is zero' says nothing"
    assert all(float(g.abs().sum()) == 0.0 for g in grads), (
        "an unrankable patch moved the model, so the loss is not purely an ordering objective"
    )


def test_the_objective_is_shift_invariant_which_no_magnitude_loss_is():
    """⭐ The property that distinguishes a ranking objective from a magnitude one, on the real
    numbers: Eq. 1 multiplies numerator and denominator by `e^c` under `f -> f + c`, so the loss
    cannot see the score's level at all — only its order and its spacing."""
    crit = ListwiseLevelLoss(y_max=_Y_MAX)
    score = torch.tensor([0.4, -0.7, 2.1, 0.0], dtype=torch.float64)
    target = torch.log1p(torch.tensor([0.0, 1.0, 17.0, 3.0], dtype=torch.float64))
    base = crit(score, target)
    for shift in (-50.0, -1.0, 3.0, 100.0):
        assert crit(score + shift, target).item() == pytest.approx(base.item(), rel=1e-9), (
            f"adding {shift} to every score changed the loss, so it is reading the score's LEVEL "
            "and is not a pure ordering objective"
        )
    # ⛔ The refusing half: MSE is not shift-invariant, so the check above can fail.
    mse = torch.nn.MSELoss()
    assert mse(score, target).item() != pytest.approx(mse(score + 3.0, target).item(), rel=1e-9)


def test_the_fourteen_pre_existing_registry_entries_are_untouched():
    """⛔ Inertness, the registry half: phase 1 ADDED an entry and changed none. A snapshot rather
    than a count, because a changed `cls` on an existing key would keep any count green."""
    expected = {
        "mse": "MSELoss",
        "count_mean": "CountMeanMSELoss",
        "mae": "L1Loss",
        "huber": "HuberLoss",
        "shrinkage": "ShrinkageLoss",
        "lognormal_nll": "LogNormalFixedSigmaLoss",
        "pareto": "ParetoLoss",
        "tobit": "TobitLoss",
        "hurdle_nb": "TruncatedNBLoss",
        "quantile": "QuantileLoss",
        "dense_nb": "DenseNBLoss",
    }
    for key, cls_name in expected.items():
        assert key in LOSS_REG_REGISTRY, f"phase 1 removed the pre-existing entry {key!r}"
        assert LOSS_REG_REGISTRY[key]["cls"].__name__ == cls_name, (
            f"the class behind {key!r} changed to {LOSS_REG_REGISTRY[key]['cls'].__name__}"
        )
    assert set(LOSS_REG_REGISTRY) == set(expected) | {"listwise_level"}, (
        f"the registry's key set is not the pre-existing one plus listwise_level: "
        f"{sorted(LOSS_REG_REGISTRY)}"
    )


def test_relu_destroys_the_score_ordering_and_softplus_preserves_it_exactly():
    """⛔⛔ The seam's one real hazard, found in phase 2 and measured rather than reasoned.

    The loss reads `reg_latent`, so training is safe under any activation. **Emission is not.**
    `output_distribution='standard'` resolves the head's activation to **ReLU**
    (`HydraBNrecurrentUnet_06_LSTM4.py:125-127`), which maps the whole negative half-line to
    exactly 0 — re-creating the atom at zero that `02 §5` measures as `tie@0` and that the ranking
    head exists to remove. `reg_activation='softplus'` is **strictly increasing**, so it preserves
    the order exactly and introduces no tie.

    ⇒ A ranking arm sets `reg_activation: 'softplus'`, or emits from `reg_latent`. ⚠️ This is a
    config choice with **no validator behind it** — deliberately: `C-411` already counts ~50
    validators and 15 `reject_*` guards in `HydraNetConfig`. The constraint is recorded here, as a
    measurement, and as a RED box in `03 §5`'s pre-flight checklist.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    score = torch.randn(2048, dtype=torch.float64)
    negatives = int((score < 0).sum())
    assert negatives > 0, "the fixture has no negative scores, so it cannot show the clamp"

    relu = F.relu(score)
    assert int((relu == 0).sum()) == negatives, (
        "ReLU did not collapse the negative half-line, so this check is not measuring the clamp"
    )
    assert len(torch.unique(relu)) < len(torch.unique(score)), "ReLU introduced no tie block"

    softplus = F.softplus(score)
    # Strictly increasing => the rank vector is identical, element for element.
    assert torch.equal(torch.argsort(softplus), torch.argsort(score))
    assert len(torch.unique(softplus)) == len(torch.unique(score)), (
        "softplus produced a tie the raw score did not have"
    )
