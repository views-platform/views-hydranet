"""Does the gate still KNOW the spatial structure, or has the sampler thrown it away?

`the feedback realism gap` was localised (2026-08-16) to **spatial structure**: scrambling only the
locations of an otherwise-perfect field reproduces essentially the whole rollout collapse
(gate AP 0.3008 -> 0.0097 against a free-running 0.0070), while magnitude and sparsity cost little.

That raises a question with two completely different fixes attached. ``compose_samples`` draws
``torch.bernoulli(gate)`` **independently per cell** — there is no spatial correlation in the
sampler at all. So either:

* the gate's probability field still carries the clustering and **independent sampling destroys
  it** — in which case the fix is a correlated sampler and needs *no retraining*; or
* the probability field has itself smeared out — in which case no sampler helps and the fix is
  training-side.

Three traps this module is built to avoid
-----------------------------------------
1. **Comparing a continuous field to a binary one.** The gate is a probability in [0, 1]; the
   sampled field is essentially binary. Measuring "clustering" on each with its natural metric
   compares two different things, and the threshold choice would decide the answer. So the
   comparison here is **binary vs binary**, both derived from the *same* gate and differing only
   in the sampling rule.
2. **Asking "is the gate clustered?"** — the wrong question. A diffuse gate (p ~ 0.3 across a
   region) can still yield perfectly clustered draws under a *correlated* sampler. The
   decision-relevant question is what the **best coherent sampler** could do with this gate, which
   is what ``topk`` operationalises.
3. **One operationalisation.** ``moran_i`` on the raw continuous gate is reported alongside, as a
   threshold-free second opinion. If the two disagree, that is a finding to report, not a number
   to pick between.
"""

from __future__ import annotations

import torch

from views_hydranet.utils.correlated_bernoulli import correlated_bernoulli

# Candidate correlation lengths swept on the ORACLE arm, so the length scale used in the
# treatment is fixed by matching the REAL field's clustering on the control — never chosen by
# looking at the treatment's outcome.
CALIBRATION_LENGTH_SCALES = (1.0, 2.0, 3.0, 5.0, 8.0)

__all__ = ["gate_structure_stats", "moran_i", "neighbour_pairs_per_active", "topk_mask"]


def neighbour_pairs_per_active(mask: torch.Tensor) -> float:
    """Right/down adjacent active pairs, per active cell — the clustering measure used throughout.

    Deliberately the SAME statistic the fed-field records use, so a gate-derived field and a fed
    field are directly comparable rather than merely analogous.
    """
    a = mask.to(torch.float32)
    n = float(a.sum())
    if n == 0:
        return 0.0
    pairs = float((a[:, :-1] * a[:, 1:]).sum() + (a[:-1, :] * a[1:, :]).sum())
    return pairs / n


def topk_mask(gate: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """The ``K`` highest-probability cells, where ``K = round(sum(gate))`` — the gate's own mass.

    This is the "best a coherent sampler could do" reference: it keeps the gate's expected active
    count but places every cell at the model's most-confident locations instead of scattering them
    at random. If these cells form clumps, the ranking carries spatial structure that independent
    Bernoulli sampling is discarding; if they do not, the structure is not in the gate to begin
    with.

    K comes from the gate's own total mass rather than from a chosen threshold, so no free
    parameter decides the outcome.

    **Ties are broken at random, and that is load-bearing.** ``torch.topk`` resolves ties by flat
    index order, so on any flat region it returns a CONTIGUOUS run of cells — a horizontal line,
    which this module's clustering statistic scores as highly structured. A uniform gate would then
    look clustered and the probe would report "the gate knows" from a pure artifact. Caught by
    ``test_the_probe_separates_the_two_cases_it_must_decide_between``, which measured 1.00 vs 1.06
    on a structured and a smeared gate before this fix.
    """
    k = int(round(float(gate.sum())))
    flat = gate.reshape(-1)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if k > 0:
        order = torch.randperm(flat.numel(), generator=generator)
        picked = order[torch.topk(flat[order], min(k, flat.numel())).indices]
        mask[picked] = True
    return mask.reshape(gate.shape)


def moran_i(field: torch.Tensor) -> float:
    """Moran's I with a rook-adjacency weight — spatial autocorrelation of a CONTINUOUS field.

    Threshold-free, so it cannot be steered by a cutoff choice. Reported beside the binary
    comparison as an independent operationalisation: agreement between the two is what makes the
    reading robust, and disagreement is itself the finding.
    """
    x = field.to(torch.float64)
    xb = x - x.mean()
    denom = float((xb * xb).sum())
    if denom == 0:
        return float("nan")
    cross = float((xb[:, :-1] * xb[:, 1:]).sum() + (xb[:-1, :] * xb[1:, :]).sum())
    n_pairs = (x.shape[0] * (x.shape[1] - 1)) + ((x.shape[0] - 1) * x.shape[1])
    if n_pairs == 0:
        return float("nan")
    # I = (N / W) * (sum_ij w_ij xb_i xb_j) / sum_i xb_i^2 ; each undirected pair counted twice.
    return (x.numel() / (2.0 * n_pairs)) * (2.0 * cross) / denom


def gate_structure_stats(
    gate: torch.Tensor, *, generator: torch.Generator, sweep_length_scales: bool = False
) -> dict:
    """Compare what the sampler DOES to a gate against what a coherent sampler COULD do.

    Args:
        gate: ``[H, W]`` per-cell P(y>0) in [0, 1], one posterior draw, one target.
        generator: seeds the independent Bernoulli draw (CPU-side, for reproducibility).

    Returns:
        ``gate_mass`` (expected active count), ``gate_moran_i`` (continuous autocorrelation), and
        the same clustering statistic under both sampling rules: ``indep_clustering`` (what
        ``compose_samples`` does) and ``topk_clustering`` (the coherent reference), plus their
        active counts so a degenerate comparison is visible rather than inferred.
    """
    if gate.dim() != 2:
        raise ValueError(
            f"gate_structure_stats expects a 2-D [H, W] gate, got {tuple(gate.shape)}"
        )
    g = gate.detach().cpu()
    indep = torch.bernoulli(g, generator=generator).to(torch.bool)
    top = topk_mask(g, generator=generator)
    sweep = {}
    if sweep_length_scales:
        for ls in CALIBRATION_LENGTH_SCALES:
            m = correlated_bernoulli(g, length_scale=ls, generator=generator).to(torch.bool)
            sweep[f"corr_ls{ls}_clustering"] = neighbour_pairs_per_active(m)
            sweep[f"corr_ls{ls}_n_active"] = int(m.sum())
    return {
        **sweep,
        "gate_mass": float(gate.sum()),
        "gate_mean": float(gate.mean()),
        "gate_moran_i": moran_i(gate.detach().cpu()),
        "indep_n_active": int(indep.sum()),
        "indep_clustering": neighbour_pairs_per_active(indep),
        "topk_n_active": int(top.sum()),
        "topk_clustering": neighbour_pairs_per_active(top),
    }
