"""DistributionFamily — the one abstraction every output-distribution family implements (ADR-067).

**Torch-free at runtime.** `from __future__ import annotations` turns the ``torch.Tensor``
annotations into strings and the only ``torch`` import is under ``TYPE_CHECKING``. So the import
chain ``config -> registry -> base`` never pulls torch; the torch-heavy families (nb/zinb) are
imported lazily by the registry factories, only when a distribution is actually instantiated.

Contract (ADR-067 §2/§3): distribution parameters live in their **natural space** via link
functions (softplus/sigmoid), never ``log1p``/``expm1``'d. The family owns the count<->log1p
boundary and never ``expm1``'s a prediction (the C-113 explosion direction).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DistributionFamily(ABC):
    """A per-cell output distribution: how the head emits, activates, scores, samples, summarises.

    Subclasses (A-S3 ``NegativeBinomialFamily``, A-S4 ``ZINBFamily``) implement all abstract
    members. Consumers depend on this ABC, never on a concrete family (DIP).
    """

    #: Does the training loss consume the pre-activation latent (``reg_latent``) instead of the
    #: activated reg head? Reuses the repo's existing ``needs_latent`` dispatch convention.
    needs_latent: bool = False

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Parameters emitted per cell, per target (e.g. nb = 2 [mu, theta]; zinb = 3 [+pi])."""

    @abstractmethod
    def activate(self, raw: "torch.Tensor") -> "torch.Tensor":
        """Map raw head channels ``[..., n_params]`` to constrained parameters (same shape)."""

    @abstractmethod
    def nll(
        self,
        params: "torch.Tensor",
        target: "torch.Tensor",
        *,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Mean negative log-likelihood (a scalar).

        ``params`` are the **activated** per-cell params ``[..., n_params]``; ``target`` is the
        log1p-space count target. ``weight`` optionally reweights cells (active-cell wt, C-199).
        """

    @abstractmethod
    def sample(
        self,
        params: "torch.Tensor",
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Draw ``k`` per-cell samples in **count space** -> ``[..., k]``, using ``generator`` for
        determinism (preserves the S2 #121 gate; C-3)."""

    @abstractmethod
    def mean(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``E[Y]`` (count space) -> ``[...]``. For AR feedback + point emit."""

    @abstractmethod
    def prob_positive(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``P(Y>0)`` -> ``[...]``. Scores gate metrics on self-zeroed nb/zinb (C-201)."""

    @abstractmethod
    def initial_raw_bias(self, *, priors: "dict[str, float] | None" = None) -> "torch.Tensor":
        """Pre-activation head bias, length ``n_params``, for informed init (C-199 / C-203).

        Each family reads the priors it needs from ``priors`` (e.g. nb: ``theta``; zinb: ``theta``,
        ``pi``) and falls back to a sensible default for any absent key. The A-S6 head calls this
        **family-agnostically** to seed each channel away from a saturated dead-zone, so the
        theta/pi gradients are live from step 0.
        """
