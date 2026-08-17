#!/usr/bin/env python3
"""floor_gate.py — refuse to run an experiment whose control cannot show an effect (C-299).

On 2026-08-14 a correctly-designed experiment — one variable, seed fixed, four doses, six GPU-hours
—
produced nothing, because its control sat **below random ranking**: gate AP 0.0070 against a
prevalence
of 0.009077 at h18. Three of its four arms scored at or below chance. It could not have
discriminated
between any hypothesis and any other, and the condition was visible in the control's own score CSV
**5 h 47 min before the last arm finished**.

Nothing caught it. Every falsifier in that plan (F1, F2, F-DEGEN) assumes the readout is valid, and
**no falsifier in this repo fires on an uninformative measurement**. A pre-registration can be
perfect
about what would refute the hypothesis and silent about whether the instrument can see anything at
all.

This module is that missing check. It is deliberately in ``scripts/`` beside
``rollout_ruler_core.py``
rather than in a dossier, because it is meant to gate every future intervention experiment.

The three clauses
-----------------
``FG-A`` (range)     ``AP_ctrl(h*) >= R * prevalence(h*)`` — the ranker must beat chance
                     at the readout horizon. Self-contained in the control's own score row, which
                     is
                     what makes it checkable before any reference exists (on 2026-08-14 the
                     climatology reference had not been built yet).
``FG-B`` (reference) ``AP_ctrl(h1) >= B * AP_clim(h1)`` — advisory: it adds a dependency
                     that may not exist yet.
``FG-C`` (power)     ``(1 - theta) * AP_ctrl(h*) >= K * MDE_AP(h*)`` — the pre-registered effect,
                     applied to *this* control, must exceed the measurement's resolution.

**FG-C is the clause that names the real failure.** It is not "the model is bad"; it is "the effect
we
came to measure is smaller than what this setup can resolve."

Validated on committed data (``tests/test_floor_gate.py``): ``truncated_smoke`` 0.77x → FAIL,
``violet_visitor`` 28.30x → PASS. A 36x separation, at zero GPU cost beyond the control arm.

Threshold discipline
--------------------
``threshold_md5`` hashes the thresholds actually used. A pre-registration records that hash; a
driver
refuses to launch treatment arms unless the token's hash matches. **Relaxing a threshold after
seeing
the control therefore invalidates the token** rather than quietly rescuing the run.
"""

from __future__ import annotations

import hashlib
import json
import math

__all__ = ["FloorGateResult", "floor_gate", "threshold_md5"]

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_PROVISIONAL = "PROVISIONAL"


def threshold_md5(*, horizon: int, target: str, theta: float, r: float, b: float, k: float) -> str:
    """Stable hash of the threshold block, for the pre-registration ↔ driver handshake."""
    blob = json.dumps(
        {"horizon": horizon, "target": target, "theta": theta, "R": r, "B": b, "K": k},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.md5(blob.encode()).hexdigest()


class FloorGateResult(dict):
    """A dict with a verdict. ``bool(result)`` is True only on an unqualified PASS."""

    def __bool__(self) -> bool:  # noqa: D105
        return self["verdict"] == VERDICT_PASS

    def report(self) -> str:
        lines = [f"FLOOR GATE: {self['verdict']}", f"  thresholds md5: {self['threshold_md5']}"]
        for clause in ("FG-A", "FG-B", "FG-C"):
            c = self["clauses"][clause]
            lines.append(f"  {clause} [{c['verdict']:>13}] {c['detail']}")
        if self["reasons"]:
            lines.append("  binding failures: " + "; ".join(self["reasons"]))
        return "\n".join(lines)


def floor_gate(
    *,
    ap_control: float,
    n_cells: int,
    n_event: int,
    horizon: int,
    target: str,
    ap_control_h1: float | None = None,
    ap_clim_h1: float | None = None,
    mde_ap: float | None = None,
    theta: float = 0.30,
    r: float = 5.0,
    b: float = 1.2,
    k: float = 3.0,
) -> FloorGateResult:
    """Evaluate the floor gate on a control arm.

    Args:
        ap_control: the control's gate AP at ``horizon``.
        n_cells: the scored support size (``N`` in the score CSV).
        n_event: the number of positive cells at ``horizon`` (``n_event``).
        horizon: the readout horizon, ``h*``. Pre-registered, not chosen after the fact.
        target: the scored target, for the record.
        ap_control_h1: the control's AP at h=1 — required for FG-B.
        ap_clim_h1: the climatology reference's AP at h=1 — required for FG-B.
        mde_ap: minimum detectable effect on AP at ``horizon`` — required for FG-C.
        theta: the pre-registered effect size of interest, as a fraction of the control.
        r, b, k: the FG-A / FG-B / FG-C multipliers.

    Returns:
        A :class:`FloorGateResult`. ``PASS`` only when every *evaluable binding* clause passes and
        none is missing; ``PROVISIONAL`` when a binding clause could not be evaluated (its inputs
        were not supplied); ``FAIL`` when any binding clause fails.

    Raises:
        ValueError: on inputs that cannot yield a meaningful verdict.

    Note:
        A missing input never silently passes. FG-C in particular is binding, so omitting
        ``mde_ap``
        yields ``PROVISIONAL`` — the state 2026-08-14 was actually in, had anyone asked.
    """
    if n_cells <= 0 or n_event <= 0:
        raise ValueError(f"floor_gate: need n_cells>0 and n_event>0, got {n_cells}, {n_event}")
    if n_event > n_cells:
        raise ValueError(f"floor_gate: n_event {n_event} exceeds n_cells {n_cells}")
    if not math.isfinite(ap_control):
        raise ValueError(f"floor_gate: ap_control is {ap_control}")
    if not 0.0 < theta < 1.0:
        raise ValueError(f"floor_gate: theta must be in (0,1), got {theta}")

    prevalence = n_event / n_cells
    clauses: dict[str, dict] = {}
    reasons: list[str] = []
    missing_binding = False

    # FG-A — range. Binding, self-contained.
    ratio = ap_control / prevalence
    ok_a = ratio >= r
    clauses["FG-A"] = {
        "verdict": VERDICT_PASS if ok_a else VERDICT_FAIL,
        "detail": (
            f"AP {ap_control:.5f} / prevalence {prevalence:.6f} = {ratio:.2f}x (need >= {r:.1f}x)"
        ),
        "ratio": ratio,
        "prevalence": prevalence,
    }
    if not ok_a:
        reasons.append(
            f"FG-A: the control ranks at {ratio:.2f}x chance at h{horizon}"
            + (" — BELOW RANDOM RANKING" if ratio < 1.0 else "")
        )

    # FG-B — reference. Advisory: the dependency may not exist yet.
    if ap_control_h1 is None or ap_clim_h1 is None:
        clauses["FG-B"] = {
            "verdict": "not evaluated",
            "detail": "no climatology reference supplied (advisory clause)",
        }
    else:
        if ap_clim_h1 <= 0:
            raise ValueError(f"floor_gate: ap_clim_h1 must be > 0, got {ap_clim_h1}")
        rb = ap_control_h1 / ap_clim_h1
        clauses["FG-B"] = {
            "verdict": VERDICT_PASS if rb >= b else VERDICT_FAIL,
            "detail": (
                f"AP(h1) {ap_control_h1:.5f} / clim {ap_clim_h1:.5f} = {rb:.3f}x (want >= {b})"
            ),
            "ratio": rb,
        }

    # FG-C — power. Binding. This is the clause that names the real failure.
    if mde_ap is None:
        clauses["FG-C"] = {
            "verdict": "not evaluated",
            "detail": "no MDE supplied — BINDING, so the verdict is PROVISIONAL",
        }
        missing_binding = True
    else:
        if not math.isfinite(mde_ap) or mde_ap <= 0:
            raise ValueError(f"floor_gate: mde_ap must be finite and > 0, got {mde_ap}")
        detectable = (1.0 - theta) * ap_control
        need = k * mde_ap
        ok_c = detectable >= need
        clauses["FG-C"] = {
            "verdict": VERDICT_PASS if ok_c else VERDICT_FAIL,
            "detail": (f"a {theta:.0%} effect is {detectable:.5f} AP; {k:.0f}x MDE is {need:.5f}"),
            "detectable": detectable,
            "needed": need,
        }
        if not ok_c:
            reasons.append(
                f"FG-C: the pre-registered {theta:.0%} effect ({detectable:.5f} AP) is "
                f"smaller than {k:.0f}x the resolution ({need:.5f}) — cannot see it"
            )

    if reasons:
        verdict = VERDICT_FAIL
    elif missing_binding:
        verdict = VERDICT_PROVISIONAL
    else:
        verdict = VERDICT_PASS

    return FloorGateResult(
        verdict=verdict,
        clauses=clauses,
        reasons=reasons,
        horizon=horizon,
        target=target,
        ap_control=ap_control,
        prevalence=prevalence,
        threshold_md5=threshold_md5(horizon=horizon, target=target, theta=theta, r=r, b=b, k=k),
    )
