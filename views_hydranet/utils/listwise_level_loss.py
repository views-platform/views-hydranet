"""listwise_level_loss.py — the level-indexed listwise loss (Zhu & Klabjan 2020, Eq. 2).

⭐ **The first listwise loss in this repo.** Every other entry in ``LOSS_REG_REGISTRY`` reduces a
strict per-cell term by ``.mean()``; this one's value depends on the relative ORDER of a set of
cells and on no cell alone. The set is the **1,024-cell patch-month** (``window_dim: 32``,
``batch_size: 1``) — ⛔ *not* the global firing set the rank-coupled emission couples. That gap
is a pre-registered falsifier, not an assumption:
``reports/2026-11-10_ranking_head_dossier/02_design`` §4.3, in the units of ``rho_event``.

Phase 1 of ``04_roadmap``: the loss and only the loss. Nothing is trained by it yet, and it is
reachable only by naming it in ``loss_reg`` — which no config does.

The objective, transcribed from ``~/brain/9_library/papers/Zhu2020_ListwiseUniqueRatings.pdf``
(library ``Zhu2020_ListwiseUniqueRatings``, claims ``C-4769``–``C-4772``). ``R`` is the rating
levels sorted descending, ``c_t`` the cells at level ``r_t``, and ``s~_t`` the cells strictly
below it::

                    exp(f(d))
    P_t(d)  =  ---------------------------------
               exp(f(d)) + SUM_{d' in s~_t} exp(f(d'))

                       1       |R|-1
    L(f; R, D)  =  - -------  ·  SUM   (2^{r_t} - 1) ·  SUM       ln P_t(d)
                     |R| - 1    t=1                   d in c_t

⇒ **Steps are indexed by rating level, not by cell.** ~1,400 positives over 9 levels give 9 steps,
not 1,400 — and the zero level, 99.6% of the land grid, is the single dropped final step.

Two design decisions this module implements, both settled in ``02_design``:

  * **Labels are dyadic**: ``r = ceil(log2(1 + y))``. The exponential gain then *inverts* the
    transform — ``2^r - 1 ~= y``, median ratio 1.348, max 1.984 — so the loss weights each level
    by roughly the number of people killed in it. ⛔ On **raw** counts that weight is ``inf`` in
    float32 at ``r >= 128`` with no error signal (``C-420``, Tier 1).
  * ⛔ **The level set is declared a priori** from ``y_max``, never read off a fold's observed
    levels — which would drift the ``1/(|R|-1)`` normaliser and make each fold a different
    objective (``C-423``).

⚠️ **The level set's size depends entirely on the population, so size ``y_max`` from the data the
loss will actually see** (``C-351``: population + transform + estimator):

  * **S11** (``gate >= 0.05``, months 463-469, n = 6,689 cell-months): levels ``0..10``, gain
    weights ``0..1023``, 9 occupied levels.
  * **The calibration TRAIN range** (months 121-456, ``region='land'``, 21,778,848 rows, max over
    all three regression targets): ``y_max = 178,459`` — ``ged_os_best`` in pgid 125700 at month
    172, April 1994, Rwanda. ⇒ levels ``0..18``, and the top level's gain is ``2^18 - 1 =
    262,143`` against ``1,023`` at level 10. Occupancy above level 10 is 135 cell-months of
    21.8M; 3 above 65,536; **1** above 131,072.

⇒ Reading the S11 figures as the general case understates ``level_max`` by 8 and the top gain by
256×. Both are real measurements of different populations.

``needs_latent = True``: the score lives on the whole real line and must reach the loss
PRE-activation — under ``transform='exp'`` because the objective is then scale-free, and under
``transform='softplus'`` because the loss applies the activation itself. This also makes
``validate_body_supervision_latent`` (``C-193``) refuse ``body_supervision='active'`` for it, which
is correct — the zero block must be present.

⭐⭐ **THE NORMALISING TRANSFORMATION ``T``, and why it is not cosmetic.**
``Bai2023_RegressionCompatibleListwise`` (CIKM '23) Definition 1 generalises the softmax above to
any non-decreasing ``T : R -> R+``, and Prop. 2 locates the global minimum at::

    T(s_i) / SUM_j T(s_j)   ->   E[y_i | x_i] / SUM_j E[y_j | x_j]

⇒ **``T`` has to be the map the pointwise objective calibrates the score through**, or the two
objectives have different minima and fight. Their Prop. 1 is the special case here: ``T = exp``
reduces to ``SoftmaxCE``, which is **translation-invariant** — it pins score *differences* only, so
the scale is free to drift indefinitely in continuous training (their §1, citing Yan et al.).

For this repo the pointwise objective is the negative-binomial NLL, whose mean map is
**softplus** (``distributions/negative_binomial.py:42``: *"softplus both channels -> strictly
positive (mu, theta)"*). ⇒ ``transform='softplus'`` is the regression-compatible choice and
``transform='exp'`` is the incompatible one.

⛔ **``'exp'`` is the default and is bit-for-bit the shipped behaviour**, because
``log exp(s) == s`` and this loss reads ``log T(s)``: the exp branch is the identity and touches
nothing. Changing the default would alter a published objective silently.

⚠️ **What this does NOT claim.** Bai's Prop. 2 is stated for ``ListCE``, whose steps are indexed by
*document* and weighted by ``y_i``. Zhu's objective indexes steps by *level* and weights them by
``2^r - 1``. The two coincide in the transformation's role — ``T`` is what maps the score onto the
scale the normaliser compares — but the alignment is **not proven for the level-indexed form**. It
is an open avenue to test, not a theorem we inherit.

⛔⛔ **``'softplus'`` IS NOT USABLE ALONE, and that is measured, not argued.**
``log softplus(s) -> s`` as ``s -> -inf``, so in the deep negative tail this objective *becomes*
the ``'exp'`` objective and regains translation invariance. Measured 2026-11-12 on six cells,
same truth, a uniform ``+1`` shift::

    scores ~  0   transform='exp'       |dL| = 0.000e+00      (exactly)
    scores ~  0   transform='softplus'  |dL| = 4.561e+00
    scores ~ -20  transform='softplus'  |dL| = 3.079e-07      and L agrees with 'exp' to 7 decimals

⇒ A pure ``'softplus'`` listwise arm has a **degenerate descent direction**: drive every score
toward ``-inf`` and the loss relaxes into its scale-free limit. The transformation is meaningful
only in a mixture with a pointwise term that pins the level — ``(1-a)*NB_NLL + a*ListCE(softplus)``
— which is Bai's own construction (their Def. 2) and not an embellishment of it. Guarded by
``test_softplus_recovers_exp_in_the_deep_tail_so_it_cannot_pin_the_scale_alone`` in
``tests/test_listwise_level_loss.py``.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from views_hydranet.utils.count_target_bridge import to_raw_counts

logger = logging.getLogger(__name__)

#: ⛔ ``C-420`` (Tier 1): the gain ``2^r - 1`` is ``inf`` in float32 from ``r = 128`` up, silently.
_MAX_LEVEL_FLOAT32 = 127
#: The int64 edge table below holds ``2^level_max - 1``, so it overflows one level sooner than
#: float32 does. The binding bound is therefore this one; both are named in the refusal.
_MAX_LEVEL_INT64 = 62


#: The normalising transformations ``T`` of ``Bai2023_RegressionCompatibleListwise`` Definition 1
#: that this loss implements. ``'exp'`` is Zhu's own softmax form; ``'softplus'`` is the one
#: compatible with a negative-binomial pointwise objective. See the module docstring.
TRANSFORMS = ("exp", "softplus")
#: ⛔ Below this score ``log(log(1 + e^s))`` and ``s`` differ by less than ``e^s / 2`` = 5e-14,
#: while evaluating the log directly underflows to ``-inf`` near ``s = -745`` and poisons the
#: gradient with ``nan``. The linear branch is exact to float64 here, not an approximation.
_LOG_SOFTPLUS_LINEAR_BELOW = -30.0


def log_transform(score: torch.Tensor, transform: str) -> torch.Tensor:
    """-> ``log T(score)``, which is the only form of ``T`` this loss ever needs.

    Every Plackett-Luce step below is a log-sum-exp over ``log T``, so the transformation enters
    in log space and nothing is ever exponentiated. ⭐ ``log exp(s) == s``, so ``'exp'`` returns
    its input unchanged and the shipped objective is reproduced exactly, not approximately.
    """
    if transform == "exp":
        return score
    if transform != "softplus":
        err_msg = f"transform={transform!r} is not one of {TRANSFORMS}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    # The clamp guards only the DISCARDED branch of the where: for scores above the cut it is
    # inert, and below it the linear branch is selected, so the clamp never shapes a gradient.
    guarded = torch.nn.functional.softplus(torch.clamp(score, min=_LOG_SOFTPLUS_LINEAR_BELOW))
    return torch.where(score < _LOG_SOFTPLUS_LINEAR_BELOW, score, torch.log(guarded))


def dyadic_levels(y: torch.Tensor, level_max: int) -> torch.Tensor:
    """-> the rating level ``r = ceil(log2(1 + y))`` of each raw count, as int64.

    Computed by **exact integer comparison** against the level edges ``[2^0 - 1, ..., 2^r - 1]``
    = ``[0, 1, 3, 7, ...]``, never by a float ``log2``: at an exact power of two a float log is
    one ULP from the boundary and ``ceil`` would silently return the next level up.

    Args:
        y: raw non-negative counts (any shape). Rounded to the nearest integer first, which
            absorbs the ``expm1(log1p(y))`` float32 round-trip.
        level_max: the a-priori top level. The returned levels lie in ``[0, level_max]``.

    Raises:
        ValueError: if ``level_max`` is out of the representable range, or if a count exceeds
            ``2^level_max - 1`` — i.e. falls outside the declared level set (``C-423``).
    """
    if not 1 <= level_max <= _MAX_LEVEL_INT64:
        err_msg = (
            f"level_max={level_max} is outside the representable range "
            f"[1, {_MAX_LEVEL_INT64}]: the int64 level-edge table overflows above "
            f"{_MAX_LEVEL_INT64}, and the float32 gain weight 2^r - 1 becomes inf above "
            f"{_MAX_LEVEL_FLOAT32} with no error signal (C-420). A level set this large means "
            "raw counts were passed as levels — the dyadic transform is the fix, not a wider "
            "dtype."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    edges = (2 ** torch.arange(level_max + 1, dtype=torch.int64, device=y.device)) - 1
    counts = torch.round(y).to(torch.int64).reshape(-1)
    if bool((counts > edges[-1]).any()):
        worst = int(counts.max())
        err_msg = (
            f"a truth count of {worst} exceeds the declared level set 0..{level_max} "
            f"(top edge {int(edges[-1])}). The level set is declared a priori from y_max and held "
            "fixed across folds (C-423); widen y_max deliberately rather than per fold."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    return torch.searchsorted(edges, counts.contiguous()).reshape(y.shape)


class ListwiseLevelLoss(nn.Module):
    """Zhu & Klabjan 2020 Eq. 2 over the cells of one patch-month, on dyadic rating levels.

    Parameters
    ----------
    y_max : float
        The grid's truth maximum, in **raw counts**. It declares the level set
        ``R = {0, 1, ..., ceil(log2(1 + y_max))}`` once and for all folds, and with it the fixed
        normaliser ``|R| - 1``.
    transform : str
        The normalising transformation ``T``, one of :data:`TRANSFORMS`. ``'exp'`` (the default)
        is Zhu's softmax form and is translation-invariant; ``'softplus'`` is the form compatible
        with a negative-binomial pointwise objective. See the module docstring.
    """

    needs_latent = True

    def __init__(self, y_max: float, transform: str = "exp") -> None:
        super().__init__()
        if transform not in TRANSFORMS:
            err_msg = (
                f"transform={transform!r} is not one of {TRANSFORMS}. 'exp' is Zhu's own softmax "
                "form; 'softplus' is the one that shares a minimum with a negative-binomial "
                "pointwise objective (Bai 2023 Prop. 2)."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.transform = transform
        if not math.isfinite(y_max) or y_max < 1:
            err_msg = (
                f"y_max={y_max!r} must be a finite count >= 1. Below 1 the level set is {{0}}, a "
                "single level: there is nothing to rank and the 1/(|R|-1) normaliser is 1/0."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        #: ``ceil(log2(1 + y))`` for an integer ``y`` is exactly its bit length — no float log.
        self.level_max = int(math.ceil(y_max)).bit_length()
        # Raises here, at construction, rather than on the first batch.
        dyadic_levels(torch.zeros(1), self.level_max)
        logger.info(
            f"ListwiseLevelLoss initialized: transform={transform}, "
            f"y_max={y_max}, levels 0..{self.level_max} "
            f"(|R|={self.level_max + 1}, normaliser 1/{self.level_max}), "
            f"top gain weight {2**self.level_max - 1}"
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """-> the 0-dim listwise loss over every cell of ``input``.

        Args:
            input: the pre-activation **score** field, any shape. Under ``transform='exp'`` only
                its order is read, so the objective is scale-free; under ``'softplus'`` the
                score's scale is read through ``log softplus``. ⛔ Never interpreted as a count.
            target: log1p-space ground truth, same number of elements; recovered to raw counts
                internally and graded dyadically.

        Returns:
            A 0-dim tensor, ``>= 0``, **graph-connected in every case**. Numerically zero (as
            ``-0.0``, which compares equal) when every cell shares one level or the list is
            empty — there is then no ordering to learn, and the grad path stays live so a caller
            that backwards the regression term alone does not raise.
        """
        score = input.reshape(-1).to(torch.float64)
        if score.numel() == 0:
            return input.sum() * 0.0
        y = to_raw_counts(target).reshape(-1)
        if y.numel() != score.numel():
            err_msg = (
                f"score has {score.numel()} cells and target has {y.numel()}: a listwise loss "
                "ranks one set, so the two must be the same set."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        level = dyadic_levels(y, self.level_max)
        # ⛔ Every step below reads log T(score), never the score. For transform='exp' this IS the
        # score (log exp(s) == s), so that branch is unchanged to the bit.
        log_t = log_transform(score, self.transform)

        # One log-sum-exp per OCCUPIED level, keyed by level. At most |R| of them, so the loss is
        # O(|R|) in the number of reductions however many cells the patch holds.
        occupied = sorted(int(v) for v in torch.unique(level).tolist())
        log_mass = {lvl: torch.logsumexp(log_t[level == lvl], dim=0) for lvl in occupied}

        # ⛔ Graph-connected, NOT `score.new_zeros(())`. When no step contributes — every cell at
        # one level, which an all-zero patch-month hits constantly on a 99.6%-zero grid — this IS
        # the return value, and a detached zero makes `.backward()` raise for any caller that
        # backwards the regression term alone (a reg-only probe, a per-target dict path, a
        # measurement driver). Today it survives only because `torch.stack(losses_list)` at
        # `training_engine.py:1117` also carries the gate term. The empty-list branch above was
        # written `input.sum() * 0.0` for exactly this reason (`C-199`); this is the same case and
        # the far more common one.
        total = score.sum() * 0.0
        # Steps 1 .. |R|-1: the levels DESCENDING, with the lowest level dropped. Zeros therefore
        # never appear as a c_t numerator — and their gain weight 2^0 - 1 is 0 besides.
        for step in range(self.level_max, 0, -1):
            if step not in log_mass:
                continue  # an empty level is a step with an empty c_t: it contributes nothing
            lower = [log_mass[lvl] for lvl in occupied if lvl < step]
            if not lower:
                continue  # s~_t is empty => P_t(d) = 1 => ln P_t(d) = 0 for every d in c_t
            log_below = torch.logsumexp(torch.stack(lower), dim=0)
            f = log_t[level == step]
            gain = float(2**step - 1)
            total = total + gain * (f - torch.logaddexp(f, log_below)).sum()

        return (-total / self.level_max).to(input.dtype)
