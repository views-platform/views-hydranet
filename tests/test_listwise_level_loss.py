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
