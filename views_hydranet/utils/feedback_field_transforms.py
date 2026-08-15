"""Diagnostic transforms on the autoregressive feedback field (#258/#262).

**Measuring `the feedback realism gap`, not fixing it.** The glossary defines that gap as: the
model's emitted field is not distributed like real conflict history, so feeding it back moves
the model off its training input distribution and the error compounds. We have only ever
sampled the two endpoints — ``rollout_feedback='teacher_forced'`` (a real field: gate AP
0.30 -> 0.27 at h36) and fully generated (0.30 -> 0.01). These transforms populate the space
between, so the damage can be decomposed as::

    skill loss = sum_i [ drift in statistic i ] x [ skill's sensitivity to statistic i ]

Every proposed fix (distribution matching, Professor Forcing, K-step rollout training) needs
the second factor, and none of it is currently known.

Numeric space — the thing that would silently invalidate everything
-------------------------------------------------------------------
The model's dynamic input channels are ``log1p(counts)`` and nothing else: the models
configure ``transformations: {'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best']}``, and
``FeatureScaler`` applies a stateless elementwise transform — **no mean/std standardisation**.
The round-trip is exactly ``expm1`` / ``log1p``.

Every function here therefore takes and returns **count space**; the caller converts.
Manipulating a field while believing it is counts when it is log-counts would produce
plausible, wrong numbers on every arm — precisely the failure mode this programme exists to
avoid — so the boundary is stated once, here, and tested.

Two kinds of transform
----------------------
**Field transforms** (here) take a count-space field and return a modified one: ``thin``,
``inject``, ``spatial_scramble``, ``magnitude_perturb``, ``splice_occurrence_magnitude``.

**Step remappings** (handled by the caller, named in ``FEEDBACK_TRANSFORMS`` for one vocabulary)
choose *which month's real field* to feed: ``use_real`` (this step), ``wrong_month`` (a fixed
distant offset), ``shuffle_months`` (a rollout-constant permutation of the step order). They
change nothing about a field, so every field stays real, correctly geo-located and fully
realistic — only the temporal ordering moves.

Orthogonality — spatial structure vs temporal persistence
----------------------------------------------------------
The naive way to break persistence is to reshuffle active cells each step, but that destroys
spatial clustering at the same time and the two arms would measure one axis. They are separated:

* ``spatial_scramble`` applies **one fixed permutation for the whole rollout** — a cell active
  at t and t+1 is still active at its image at t and t+1, so persistence is preserved exactly
  while clustering is destroyed. The value multiset is preserved exactly.
* ``shuffle_months`` permutes **which step's real field is fed**, so each field keeps its own
  spatial structure and geography intact while temporal continuity is destroyed.

A torus roll was tried for the persistence axis and rejected: it preserves the statistics, but
the grid is a map of Africa, not a torus, so rolling a blob off the east edge lands it in a
different country while the static channels (coordinates, ADR-061) stay fixed. That confounds
"persistence broken" with "field decoupled from geography".

**Caveat that cannot be engineered away:** ``spatial_scramble`` necessarily breaks the field's
alignment with the static channels too, because plausible locations *are* the clustering. That
confound is intrinsic to the axis, not an implementation defect — read that arm as "spatial
structure and its geographic grounding", not "structure alone".
"""

from __future__ import annotations

import torch

__all__ = [
    "FEEDBACK_TRANSFORMS",
    "inject",
    "magnitude_perturb",
    "parse_feedback_transform",
    "spatial_scramble",
    "splice_occurrence_magnitude",
    "thin",
]

# Named transforms and whether each carries a numeric parameter. Parsed from an explicit string
# spec (e.g. "thin:0.25") so the arm is visible in the process table, the log and the run manifest
# rather than being an anonymous callable.
FEEDBACK_TRANSFORMS: dict[str, bool] = {
    "identity": False,  # E1 observer: the model's own field, unchanged
    "use_real": False,  # F1 self-test: must byte-reproduce rollout_feedback='teacher_forced'
    "wrong_month": True,  # E3: a REAL field from another month - realistic, uninformative
    "occurrence_real_magnitude_model": False,  # E4
    "occurrence_model_magnitude_real": False,  # E4
    "thin": True,  # E2 recall
    "inject": True,  # E2 precision
    "spatial_scramble": False,  # E2 spatial structure
    "shuffle_months": False,  # E2 temporal persistence (a step remapping, not a field op)
    "magnitude_perturb": True,  # E2 magnitude realism
}


def parse_feedback_transform(spec: str) -> tuple[str, float | None]:
    """Parse ``"thin:0.25"`` -> ``("thin", 0.25)``; ``"spatial_scramble"`` -> ``(name, None)``.

    Fails loud on an unknown name, a missing parameter, or a parameter given to a transform that
    takes none. A typo must not fall through to a no-op: a silent identity would be reported as
    "this axis does not matter" when the axis was never broken.
    """
    name, _, raw = spec.partition(":")
    if name not in FEEDBACK_TRANSFORMS:
        raise ValueError(
            f"unknown feedback transform {name!r}; valid: {sorted(FEEDBACK_TRANSFORMS)}. "
            "A misspelled arm would silently run the control."
        )
    takes_param = FEEDBACK_TRANSFORMS[name]
    if takes_param and not raw:
        raise ValueError(f"feedback transform {name!r} requires a parameter, e.g. '{name}:0.25'.")
    if raw and not takes_param:
        raise ValueError(f"feedback transform {name!r} takes no parameter, got {raw!r}.")
    if not takes_param:
        return name, None
    value = float(raw)
    if name == "wrong_month":
        # A fractional offset truncates and a zero offset IS `use_real` — either would run the
        # control while being scored as the treatment, which is the exact failure this parser
        # exists to prevent.
        if value != int(value) or int(value) == 0:
            raise ValueError(
                f"wrong_month needs a NON-ZERO INTEGER month offset, got {raw!r}. A fractional "
                "offset truncates and 0 is `use_real` — both silently run the control."
            )
    return name, value


def _rand_like(field: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Uniform noise shaped like ``field``, drawn on the generator's device then moved.

    Drawn on CPU rather than on the field's device on purpose: a CPU ``torch.Generator`` cannot
    drive ``torch.rand`` on a CUDA tensor at all, and CUDA's RNG stream differs from CPU's — so
    drawing device-side would make an arm irreproducible across machines and non-comparable to the
    CPU-run tests. Determinism beats the copy.
    """
    return torch.rand(field.shape, generator=generator).to(field.device)


def _randint_like(field: torch.Tensor, high: int, generator: torch.Generator) -> torch.Tensor:
    """Indices into a pool of size ``high``, shaped like ``field``.

    CPU-drawn; see ``_rand_like``.
    """
    return torch.randint(high, field.shape, generator=generator).to(field.device)


def _active_pool(field: torch.Tensor) -> torch.Tensor:
    """The field's own active (>0) values — the empirical magnitude distribution to draw from."""
    pool = field[field > 0]
    if pool.numel() == 0:
        raise ValueError(
            "field has no active cells, so its magnitude distribution is undefined. Refusing to "
            "invent values — an arm built on a fabricated distribution is uninterpretable."
        )
    return pool


def thin(field: torch.Tensor, *, p: float, generator: torch.Generator) -> torch.Tensor:
    """Zero a fraction ``p`` of the active cells. Breaks **recall** / active fraction.

    Survivors keep their exact values, so the magnitude distribution of what remains is unchanged
    and only occurrence is degraded.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"thin: p must be in [0, 1], got {p}.")
    active = field > 0
    drop = (_rand_like(field, generator) < p) & active
    return torch.where(drop, torch.zeros_like(field), field)


def inject(field: torch.Tensor, *, q: float, generator: torch.Generator) -> torch.Tensor:
    """Activate a fraction ``q`` of the *zero* cells, with values from the field's own actives.

    Breaks **precision** — it adds false positives without moving the true ones and without
    changing the magnitude distribution (values are resampled from the field's own actives).
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"inject: q must be in [0, 1], got {q}.")
    pool = _active_pool(field)
    empty = field == 0
    add = (_rand_like(field, generator) < q) & empty
    idx = _randint_like(field, pool.numel(), generator)
    return torch.where(add, pool[idx], field)


def spatial_scramble(field: torch.Tensor, *, permutation: torch.Tensor) -> torch.Tensor:
    """Permute grid positions with a **caller-supplied, rollout-constant** permutation.

    Destroys spatial clustering; preserves the value multiset exactly; and because the same
    permutation is applied at every step, preserves temporal persistence exactly. That
    orthogonality is the point — see the module docstring. The permutation is passed in rather than
    drawn here so the caller owns its constancy across the rollout.
    """
    b, c, h, w = field.shape
    if permutation.numel() != h * w:
        raise ValueError(
            f"spatial_scramble: permutation has {permutation.numel()} entries for a {h}x{w} grid."
        )
    flat = field.reshape(b, c, h * w)
    return flat[:, :, permutation.to(field.device)].reshape(b, c, h, w)


def magnitude_perturb(
    field: torch.Tensor, *, sigma: float, generator: torch.Generator
) -> torch.Tensor:
    """Multiply active values by lognormal(0, sigma) noise. Breaks **magnitude realism** only.

    The occurrence mask is byte-identical to the input's — this axis moves *how much*, never
    *where*.
    """
    if sigma < 0:
        raise ValueError(f"magnitude_perturb: sigma must be >= 0, got {sigma}.")
    active = field > 0
    # exp(sigma*Z) has mean exp(sigma^2/2) — at sigma=1.5 that is a x2.5 inflation, so a plain
    # lognormal would confound "magnitude realism" with "magnitude inflation". On a model with a
    # documented runaway-feedback sensitivity that is the worst available confound, so the noise
    # is mean-one by construction.
    z = torch.randn(field.shape, generator=generator).to(field.device)
    noise = torch.exp(sigma * z - 0.5 * sigma * sigma)
    return torch.where(active, field * noise, field)


def splice_occurrence_magnitude(
    occurrence_field: torch.Tensor,
    magnitude_field: torch.Tensor,
    *,
    generator: torch.Generator,
    on_empty_donor: str = "raise",
) -> torch.Tensor:
    """Take *where* from one field and *how much* from another's magnitude distribution.

    E4's question: is the damage in where the model fires, or in what it emits there? Values are
    drawn from ``magnitude_field``'s active values rather than read cell-wise, because reading
    cell-wise would return 0 wherever the donor happens to be inactive — silently re-introducing
    the donor's occurrence pattern into an arm meant to isolate magnitude.
    """
    if on_empty_donor not in ("raise", "zeros"):
        raise ValueError(f"on_empty_donor must be 'raise' or 'zeros', got {on_empty_donor!r}.")
    if on_empty_donor == "zeros" and not (magnitude_field > 0).any():
        # The rollout arm feeds the MODEL's field as donor, and the phenomenon under study is that
        # the model goes quiet. Raising here would abort the run at exactly the moment the effect
        # appears — hours into an unattended batch. Degrade to an empty field; the caller's
        # fed-field statistics record the zero so it is visible, not silent.
        return torch.zeros_like(occurrence_field)
    pool = _active_pool(magnitude_field)
    mask = occurrence_field > 0
    idx = _randint_like(occurrence_field, pool.numel(), generator)
    return torch.where(mask, pool[idx], torch.zeros_like(occurrence_field))
