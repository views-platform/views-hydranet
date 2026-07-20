"""NegativeBinomialFamily — the per-cell negative-binomial output family (ADR-067, A-S3).

The first real family behind the ``DistributionFamily`` abstraction: a per-cell NB with mean ``mu``
and dispersion ``theta``, both emitted by the head (``n_params = 2``) and activated into natural
space by softplus. It COMPOSES the shared ``NBCore`` (has-a, not is-a) for all count math.

Contract (ADR-067 §3): parameters live in natural space (never ``log1p``/``expm1``'d); the training
target arrives in log1p space and ``nll`` recovers raw counts via ``to_raw_counts`` (the log1p
inverse, fail-loud). The config-parameterized inverse (C-198) is a wiring concern deferred to
A-S7 — here the log1p inverse is exactly the pipeline's transform.
"""

from __future__ import annotations

import torch

from views_hydranet.distributions.base import DistributionFamily
from views_hydranet.distributions.nb_core import NBCore, inverse_softplus
from views_hydranet.utils.count_target_bridge import to_raw_counts

_WEIGHT_SUM_EPS = 1e-8  # divide-by-zero guard for an all-zero weight (no supervised cells)


class NegativeBinomialFamily(DistributionFamily):
    """Per-cell NB: params ``[..., 2]`` = ``(mu, theta)`` in natural space."""

    needs_latent = False

    @property
    def n_params(self) -> int:
        return 2

    def activate(self, raw: "torch.Tensor") -> "torch.Tensor":
        """softplus both channels -> strictly-positive ``(mu, theta)``, same shape."""
        return torch.nn.functional.softplus(raw)

    def _split(self, params: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        return params[..., 0], params[..., 1]

    def nll(
        self,
        params: "torch.Tensor",
        target: "torch.Tensor",
        *,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Mean NB negative log-likelihood.

        ``params`` are the activated ``(mu, theta)`` ``[..., 2]``; ``target`` is the log1p-space
        count target (recovered to raw counts via ``to_raw_counts``). ``weight`` optionally
        reweights cells — active-cell weighting (C-199; theta is identified by the ~1% positives).
        ``weight`` must be broadcastable to the per-cell loss; an all-zero weight (no supervised
        cells) yields a graph-connected ``0`` loss (zero gradient), not an error.
        """
        mu, theta = self._split(params)
        counts = to_raw_counts(target)
        if counts.shape != mu.shape:
            raise ValueError(
                f"nll target shape {tuple(counts.shape)} must match the per-cell param shape "
                f"{tuple(mu.shape)} (params[..., :2] = mu, theta)."
            )
        nll_per_cell = -NBCore.log_prob(mu, theta, counts)
        if weight is None:
            return nll_per_cell.mean()
        # Broadcast to the per-cell shape so the numerator and the normalizer sum over the SAME
        # elements (a per-target [1, C, 1, 1] weight otherwise mis-scales the mean); .to(tensor)
        # moves dtype AND device. broadcast_to fails loud on a truly-incompatible weight shape.
        weight = torch.broadcast_to(weight.to(nll_per_cell), nll_per_cell.shape)
        total = weight.sum()
        return (nll_per_cell * weight).sum() / total.clamp_min(_WEIGHT_SUM_EPS)

    def sample(
        self,
        params: "torch.Tensor",
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Draw ``k`` counts per cell -> ``[..., k]`` (deterministic under ``generator``)."""
        mu, theta = self._split(params)
        return NBCore.sample(mu, theta, k, generator)

    def mean(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``E[Y] = mu``."""
        mu, _ = self._split(params)
        return mu

    def prob_positive(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``P(Y>0) = 1 - NB(0; mu, theta)`` (C-201 — self-zeroed gate scoring).

        Computed as ``-expm1(log P(Y=0))`` to stay accurate for small ``mu`` / large ``theta``,
        where the direct ``1 - prob_zero`` cancels to a spurious 0.
        """
        mu, theta = self._split(params)
        return -torch.expm1(NBCore.log_prob_zero(mu, theta))

    def initial_raw_bias(self, theta_prior: float = 1.0) -> "torch.Tensor":
        """Raw-space head bias ``[mu, theta]`` for informed init (C-199).

        Returns the pre-activation biases so ``softplus(bias_theta) ~= theta_prior`` (the global
        theta baseline, ~1.0 = near-geometric) and ``mu`` starts small-positive. The A-S6 head
        inits its theta channel from this instead of zero/default, keeping the theta-channel
        gradient live from step 0 (theta is identified by few cells; it dies if it saturates).
        """
        mu_bias = inverse_softplus(0.5)  # softplus(bias) = 0.5 -> a small positive starting mean
        theta_bias = inverse_softplus(theta_prior)
        return torch.tensor([mu_bias, theta_bias], dtype=torch.float32)
