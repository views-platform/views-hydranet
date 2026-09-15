"""Potency gate — prove an experimental knob MOVES something before spending on it.

**The failure this exists for.** On 2026-09-03 a training change (#308) was implemented, unit
tested, mutation tested 5/5, lint clean, full suite green — and was a **no-op on the path
production
uses**. Two arms trained for 276 minutes and produced byte-identical weights. It was caught only
because identical-to-the-last-bit is *impossible*; a half-connected knob would have produced a
small
number, the pre-registered rule would have read it as "hypothesis dead", and a wrong conclusion
about the *idea* would have entered the ledger as a fact about the world.

**The principle.** A broken implementation must not produce the same signature as a null result. If
"the knob does nothing" and "the knob does nothing useful" look the same in the readout, every null
is ambiguous.

**What this does that tests do not.** Unit tests check the code against the paths the author
thought
to test. A potency check asks a different question: *in the exact configuration this experiment
will
run, does flipping the knob change a number?* It is run on the arm's own config, never a fixture,
because the whole class of error is "verified on a path production never takes" (**C-323**).

A potency check is NOT a claim the knob is beneficial, or even correct. It is the weaker,
prior claim that the experiment is capable of measuring anything at all. Failing it voids the
experiment before the GPU starts; passing it says only that a null result will be informative.
"""

from __future__ import annotations

import math
from typing import Any, Callable


class PotencyError(AssertionError):
    """Raised when a knob provably cannot affect the quantity the experiment will read."""


def assert_potent(
    measure: Callable[[Any], float],
    *,
    off: Any,
    on: Any,
    name: str,
    min_relative_change: float = 1e-6,
) -> dict[str, float]:
    """Assert that flipping a knob changes a measured quantity. Returns the two readings.

    Args:
        measure: called with the knob setting; returns ONE number from the arm's own config.
        off / on: the two settings the experiment contrasts.
        name: what the knob is, for the failure message.
        min_relative_change: below this the knob is treated as inert. The default is
            effectively "any change at all" — deliberately, because this gate answers
            *can it act*, not *does it act enough*.

    Raises:
        PotencyError: the readings are equal, or both non-finite. **The experiment is void.**
    """
    a, b = float(measure(off)), float(measure(on))
    if not (math.isfinite(a) and math.isfinite(b)):
        raise PotencyError(f"{name}: non-finite readings ({a}, {b}) — the measure is broken")
    denom = max(abs(a), abs(b))
    rel = 0.0 if denom == 0 else abs(b - a) / denom
    if rel < min_relative_change:
        raise PotencyError(
            f"{name}: flipping the knob changed the measured quantity by {rel:.3g} "
            f"(off={a!r}, on={b!r}). The knob is INERT on this configuration, so any result "
            f"would be a fact about the harness, not about the hypothesis. Refusing to proceed."
        )
    return {"off": a, "on": b, "relative_change": rel}


def assert_control_fires(
    measure: Callable[[Any], float],
    *,
    baseline: Any,
    known_effect: Any,
    name: str,
    min_relative_change: float = 1e-3,
) -> dict[str, float]:
    """Assert a POSITIVE CONTROL moves the readout — the harness can detect a real effect.

    The mirror of `assert_potent`. That one asks "can the treatment act?"; this asks "could the
    *readout* see it if it did?" A null from a harness whose positive control does not fire is not
    a null, and this repo has no way to tell those apart after the fact.

    `known_effect` should be a manipulation whose effect is not in question — a large, crude change
    that must move the number if the measurement pipeline works at all.
    """
    return assert_potent(
        measure,
        off=baseline,
        on=known_effect,
        name=f"positive control: {name}",
        min_relative_change=min_relative_change,
    )
