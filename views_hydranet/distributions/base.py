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

    The concrete families in this package (``NegativeBinomialFamily``, ``ZINBFamily``,
    ``MixtureNBFamily``, ``TruncatedNBFamily``) implement all abstract members. Consumers depend
    on this ABC, never on a concrete family (DIP).
    """

    #: Does the training loss consume the pre-activation latent (``reg_latent``) instead of the
    #: activated reg head? Reuses the repo's existing ``needs_latent`` dispatch convention.
    needs_latent: bool = False

    #: Does the family produce its zeros STRUCTURALLY (inside the distribution), so its ``mean`` is
    #: already the full self-zeroed forecast and must NOT be multiplied by the classification gate?
    #: ``True`` for ZINB (structural π); ``False`` for NB (zeros handled OUTSIDE, via a gate ->
    #: forecast is ``gate × body`` = gated_NB). Consumed by the training-forensic biopsy so the
    #: plotted forecast is honest per mode (self-zeroed body vs gate × body).
    self_zeroed: bool = False

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

    def sample_core(
        self,
        params: "torch.Tensor",
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Draw ``k`` samples from the family's BULK body, WITHOUT any structural self-zeroing.

        Default = :meth:`sample` (identical for a family with no structural zero, e.g. ``nb``). A
        self-zeroed family (``zinb``) overrides it to draw its bare NB core (π dropped) — for the
        **gated_ZINBcore** arm, where an EXTERNAL cls gate supplies the zeros instead of the
        structural π (self-zeroing here too would double-count; ADR-068).
        """
        return self.sample(params, k, generator)

    @abstractmethod
    def mean(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``E[Y]`` (count space) -> ``[...]``. For AR feedback + point emit."""

    def mean_core(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``E[Y]`` of the BULK body, WITHOUT structural self-zeroing — the AR-feedback /
        point-emit counterpart of :meth:`sample_core`.

        Default = :meth:`mean` (identical for a family with no structural zero, e.g. ``nb``). A
        self-zeroed family (``zinb``) overrides it to the bare NB core mean (π dropped), so a
        core-emitting rollout (``emit_family_core``) feeds back the LARGE core mean it actually
        emits, not the small ``(1-π)μ`` self-zeroed mean.
        """
        return self.mean(params)

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

    def parameter_penalty(
        self,
        params: "torch.Tensor",
        *,
        prior_logit: float = 0.0,
        scale: float = 0.0,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Optional additive per-cell parameter regularizer — **0 by default** (C-205).

        A concrete (non-abstract) hook so a family-agnostic loss can add a parameter prior without
        ``isinstance``-branching to a concrete family (DIP). The default is a graph-safe scalar 0;
        ``NegativeBinomialFamily`` inherits it. ``ZINBFamily`` overrides it with the C-200 π-ridge
        (``pi_penalty``), pulling ``logit(pi)`` toward ``prior_logit`` with strength ``scale`` on
        the cells selected by ``weight``. The training loop scales the returned term by its own
        config weight (qs99/decay additive-penalty precedent), so ``scale=0`` here is a full no-op.
        """
        return params.new_zeros(())
