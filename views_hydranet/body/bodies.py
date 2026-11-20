"""One object per regression body (ADR-079 §2, Option B; epic #616).

A body is one complete specification of what the regression heads compute and how they are trained.
Each class below states, in one place, what ADR-079 says a body fixes: its width, its activation,
its loss, and the yes-or-no facts other settings are checked against.

ADR-079 rule 2 lives here: a body always receives the heads' outputs from **before** the
activation and applies the activation itself, so the activation and the loss that depends on it
cannot be paired wrongly.

Deliberately small:

* Only the five bodies in use have a class, and since issue #629 they are the only bodies: every
  config the schema accepts gets one, and `build_body` raises for anything else. The older path
  through the training loop is gone.
* A body is a plain object, never an ``nn.Module``. It adds nothing to a model's ``state_dict``,
  so every saved model still loads.
* Each body wraps the criterion `choose_loss` already built. `y_max`, the ranking loss's own
  function and the truncated NB's denominator therefore have one source, not two.
* `FamilyLoss` and `DistributionFamily` are not changed. Callers that hand them activated values
  keep working.

⚠️ Bit-identity depends on memory layout, for the value and for the gradient. Two measurements
(torch 2.6, CPU) shaped this module:

* softplus over a strided slice does not give the same float32 bits as softplus over the
  contiguous tensor the head activates (up to 5e-7);
* activating one target's slice alone gives the same loss value but a gradient that differs by
  about 1e-7, in every negative-binomial trial. Activating every head and concatenating them, as
  the model's own ``forward`` does, gives the identical value in 240 of 240 trials, and the
  identical gradient for every parameter except one case: the mixture body's three head biases on
  a small grid (8 x 8: up to 3e-8; 32 x 32, the live window: identical). The body hands the head
  a contiguous gradient where the older path hands it a channels-last one, which changes the
  order in which the bias gradient is summed.

So `activate` takes the raw outputs of **all** targets, ``[B, n_targets * width, H, W]``, activates
them head by head on contiguous slices, and concatenates. `loss` then slices its target out of
that, exactly as the training loop slices the model's activated output.

This module imports torch. It is not imported by ``views_hydranet.body.__init__``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss

logger = logging.getLogger(__name__)

#: The distribution bodies in use, and whether each can itself say "no fatalities" (ADR-079 §2,
#: "own zeros"). A negative binomial puts probability on zero; the zero-truncated one puts none
#: there by construction, which is why it always needs the gate (Mullahy 1986).
_LIVE_DISTRIBUTIONS: dict[str, bool] = {
    "nb": True,
    "mixture_nb": True,
    "truncated_nb": False,
}


class Body(ABC):
    """What every body states. Subclasses set the facts, `_activate_head` and `loss`."""

    #: Channels each regression head emits per grid cell.
    width: int
    #: Whether the body can draw random samples of the count.
    can_sample: bool
    #: Whether the body can itself say "no fatalities". ``None`` means not decided.
    own_zeros: bool | None
    #: Whether the loss undoes the log1p transform of the target to get counts.
    converts_target_to_counts: bool
    #: Whether the loss needs every grid cell, the zero cells included.
    requires_all_cells: bool
    #: Config names of the data-derived numbers the body cannot be built without.
    required_numbers: tuple[str, ...] = ()

    @abstractmethod
    def _activate_head(self, raw_head: torch.Tensor) -> torch.Tensor:
        """One head's contiguous raw output ``[B, width, H, W]`` -> activated, same shape."""

    def activate(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw outputs of all targets ``[B, n_targets * width, H, W]`` -> activated, same shape.

        Head by head, then concatenated: the same sequence as the model's ``forward``.
        """
        w = self.width
        if raw.dim() != 4 or raw.shape[1] == 0 or raw.shape[1] % w != 0:
            err_msg = (
                f"{type(self).__name__} expects the raw output of all regression heads, shaped "
                f"[B, n_targets * {w}, H, W]; got {tuple(raw.shape)}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        heads = [
            self._activate_head(raw[:, k * w : (k + 1) * w].contiguous())
            for k in range(raw.shape[1] // w)
        ]
        return torch.cat(heads, dim=1)

    def read(self, raw: torch.Tensor) -> BodyStep:
        """Read one training step: the raw outputs of all heads, activated once."""
        return BodyStep(self, raw)

    def loss(
        self,
        raw: torch.Tensor,
        target: torch.Tensor,
        j: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The training loss for target ``j``, from the raw outputs alone.

        ``raw`` is the output of all regression heads from before the activation,
        ``[B, n_targets * width, H, W]``. ``target`` is target ``j`` as the data delivers it,
        ``[B, H, W]``. ``mask`` is an optional boolean ``[B, H, W]`` selecting the cells to train
        on. Shorthand for ``self.read(raw).loss(target, j, mask)``.
        """
        return self.read(raw).loss(target, j, mask)

    @abstractmethod
    def _loss(
        self,
        raw: torch.Tensor,
        activated: torch.Tensor,
        target: torch.Tensor,
        j: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """The loss for target ``j``, given the raw outputs and this body's activation of them."""


class BodyStep:
    """One training step as its body reads it: the raw outputs, and their activation, once.

    The training loop needs the activated values in several places besides the loss (feedback,
    penalties, forensics). If the loss activated the raw outputs a second time, the gradient would
    reach the head through two activation nodes instead of one. The sum is the same number, but
    not the same float32 bits (measured: about 1e-9 relative, in the one case where two terms
    share a target). So the activation is computed once here and everything reads this object.
    """

    def __init__(self, body: Body, raw: torch.Tensor) -> None:
        self.body = body
        self.raw = raw
        self.activated = body.activate(raw)

    def loss(self, target: torch.Tensor, j: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        """The training loss for target ``j``; ``mask`` optionally selects the cells."""
        return self.body._loss(self.raw, self.activated, target, j, mask)


class MeanSquaredErrorBody(Body):
    """One output per cell: a prediction on the scale of the transformed target."""

    width = 1
    can_sample = False
    own_zeros = True
    converts_target_to_counts = False
    requires_all_cells = False

    def __init__(self, criterion: nn.MSELoss) -> None:
        self.criterion = criterion

    def _activate_head(self, raw_head: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw_head)

    def _loss(
        self,
        raw: torch.Tensor,
        activated: torch.Tensor,
        target: torch.Tensor,
        j: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        prediction = activated[:, j, :, :]
        if mask is None:
            return self.criterion(prediction, target)
        if mask.any():
            return self.criterion(prediction[mask], target[mask])
        # No cell selected in this batch: the loss is zero and carries no gradient.
        return torch.tensor(0.0, device=raw.device)


class RankingBody(Body):
    """One output per cell: a score of which only the order is used. Exponential version.

    The ranking loss applies a function of its own to the raw score. With the exponential the
    objective is scale-free. The softplus version of that loss is a different body by ADR-079 §2
    and has no class here, because no config uses it.

    Whether this body makes its own zeros is not decided: its score is not a count, and its count
    forecast is not built.
    """

    width = 1
    can_sample = False
    own_zeros = None
    converts_target_to_counts = True
    requires_all_cells = True
    required_numbers = ("loss_reg_y_max",)

    def __init__(self, criterion: ListwiseLevelLoss) -> None:
        self.criterion = criterion

    def _activate_head(self, raw_head: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw_head)

    def _loss(
        self,
        raw: torch.Tensor,
        activated: torch.Tensor,
        target: torch.Tensor,
        j: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is not None:
            err_msg = (
                "RankingBody must be trained on all cells: the zero cells are the bottom of its "
                "ranking. A cell mask was passed."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        # The loss reads the raw score and applies its own function to it. The head's softplus
        # shapes only what the model reports.
        return self.criterion(raw[:, j, :, :], target)


class DistributionBody(Body):
    """A count distribution per cell. The distribution owns its activation and its likelihood."""

    can_sample = True
    converts_target_to_counts = True
    requires_all_cells = False

    def __init__(self, name: str, family) -> None:
        if name not in _LIVE_DISTRIBUTIONS:
            err_msg = (
                f"DistributionBody has no entry for {name!r}; "
                f"the distributions in use are {sorted(_LIVE_DISTRIBUTIONS)}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.name = name
        self.family = family
        self.width = family.n_params
        self.own_zeros = _LIVE_DISTRIBUTIONS[name]

    def _activate_head(self, raw_head: torch.Tensor) -> torch.Tensor:
        # The head's own sequence (`_family_activation`): parameters to the last dim, activate,
        # back to channel-first, contiguous.
        params = self.family.activate(raw_head.permute(0, 2, 3, 1))
        return params.permute(0, 3, 1, 2).contiguous()

    def _loss(
        self,
        raw: torch.Tensor,
        activated: torch.Tensor,
        target: torch.Tensor,
        j: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        w = self.width
        params = activated[:, j * w : (j + 1) * w, :, :].permute(0, 2, 3, 1)
        if mask is None:
            return self.family.nll(params, target, weight=None)
        # A 0/1 weight, not boolean indexing: an all-zero mask then gives a graph-connected zero.
        return self.family.nll(params, target, weight=mask.to(params.dtype))


def body_class_for(distribution: str, loss_name: str | None) -> type[Body] | None:
    """The body class for what a `body` value fixes (ADR-081), or None if it is none of the five.

    The one place that maps a body's distribution and loss to its class. `build_body` uses it, and
    so does the config check that refuses a supervision window on a body that trains on all
    cells (#630).
    """
    if distribution in _LIVE_DISTRIBUTIONS:
        return DistributionBody
    # A "standard" head always applies softplus (ADR-079 Rule 1, issue #632); the model refuses
    # any other activation, so the setting need not be read here.
    if distribution == "standard":
        if loss_name == "mse":
            return MeanSquaredErrorBody
        if loss_name == "listwise_level":
            return RankingBody
    return None


def build_body(config: dict, criterion_reg) -> Body:
    """The body for this config. Raises ``ValueError`` if it is not one of the five in use.

    ``criterion_reg`` is what `choose_loss` returned for the same config. The schema refuses every
    other combination, so the error means the loss was built for a different config (#629).
    """
    from views_hydranet.body.vocabulary import body_name

    named = body_name(config)
    distribution, loss_name = named.distribution, named.loss_name
    body_class = body_class_for(distribution, loss_name)
    if body_class is DistributionBody and isinstance(criterion_reg, FamilyLoss):
        return DistributionBody(distribution, criterion_reg.family)
    if body_class is MeanSquaredErrorBody and type(criterion_reg) is nn.MSELoss:
        return MeanSquaredErrorBody(criterion_reg)
    if (
        body_class is RankingBody
        and isinstance(criterion_reg, ListwiseLevelLoss)
        and criterion_reg.transform == "exp"
    ):
        return RankingBody(criterion_reg)
    err_msg = (
        f"No body for body={named.value!r} with the loss {type(criterion_reg).__name__}. The "
        "loss was probably built for a different config (ADR-079, ADR-081)."
    )
    logger.error(err_msg)
    raise ValueError(err_msg)
