"""The S1 -> S2 design-selection rule for Epic #311, and STOP-gate (a).

Lives in ``scripts/`` rather than in the dossier because a decision rule is governance: it belongs
beside ``lesson_curve_gate`` and ``ss_sweep_gate``, is imported by a tracked test that does not
depend on ``reports/``, and survives the dossier being archived. The plumbing that feeds it stays
dossier-local in ``reports/2026-09-04_input_noise_dossier/tools/error_profile.py``; the two have
different reasons to change (this one with the pre-registration, that one with the cube layout).

⚠️ The sibling gates justify the split by saying a tracked test *cannot* load the gitignored
``reports/`` tree. That is too strong — a force-added dossier tool is tracked and importable, and
``tests/test_input_noise_plumbing.py`` does exactly that. The real reason is the one above:
governance artifacts should not depend on a dossier's lifetime.

Pre-registered in ``reports/2026-09-04_input_noise_dossier/05_analysis_plan.md`` §5, committed in
``47d66af`` before S1 (#313) produced a single number.
"""

from __future__ import annotations

import hashlib
import json
import math

#: The larger error rate must exceed the smaller by this factor to "dominate".
DOMINANCE_FACTOR = 2.0
#: STOP-gate (a): above this coefficient of variation across origins the measured distribution is
#: not a stable target and S2 does not proceed. A judgement call, made in advance and marked so.
MAX_CV = 0.5


def rule_md5(*, dominance_factor: float = DOMINANCE_FACTOR, max_cv: float = MAX_CV) -> str:
    """Lock hash over the rule's thresholds — relaxing one after seeing the data invalidates it."""
    blob = json.dumps({"dominance_factor": dominance_factor, "max_cv": max_cv}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()  # noqa: S324 - provenance token, not a secret


def cv(values) -> float:
    """Coefficient of variation across origins. NaN whenever the spread is not measurable.

    Returns NaN — never 0.0 — for fewer than two values, a zero/non-finite mean, or **any**
    unmeasurable entry. Returning a number there would read as "stable", pass STOP-gate (a), and
    convert a measurement that did not happen into permission to spend GPU.

    The audit of 2026-09-04 found the earlier version *silently dropped* NaN and non-numeric
    entries, so ``cv([0.4, nan, nan, nan, 0.41])`` returned **0.0175** — three unmeasurable origins
    out of five reading as exquisitely stable. That is the "unmeasured input absorbed as a value"
    pattern, and the remedy this repo records for it is to **delete the representation**: an
    unmeasurable origin makes the spread unmeasurable, full stop. ``bool`` is rejected explicitly
    because ``isinstance(True, int)`` is True and a boolean is not a rate.
    """
    vals = list(values)
    if len(vals) < 2:
        return float("nan")
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return float("nan")
    mean = sum(vals) / len(vals)
    if mean == 0.0:
        return float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var) / abs(mean)


def select_design(fn_rate: float, fp_rate: float, cv_dominant: float) -> dict:
    """Apply the pre-registered selection rule. Returns design, stop flag, and the reason.

    ``fn_rate`` — expected fraction of TRUE events the model silences.
    ``fp_rate`` — expected fraction of TRUE zeros the model fires on.
    ``cv_dominant`` — coefficient of variation of the dominant rate across origins.
    """
    if any(x is None or math.isnan(x) for x in (fn_rate, fp_rate)):
        return {
            "design": None,
            "stop": True,
            "reason": "an error rate is undefined",
            "why": "an error rate is undefined",
        }

    if fn_rate >= DOMINANCE_FACTOR * fp_rate:
        design = "occurrence_dropout"
        why = f"FN {fn_rate:.4g} >= {DOMINANCE_FACTOR}x FP {fp_rate:.4g} — the model goes SILENT"
    elif fp_rate >= DOMINANCE_FACTOR * fn_rate:
        design = "occurrence_injection"
        why = f"FP {fp_rate:.4g} >= {DOMINANCE_FACTOR}x FN {fn_rate:.4g} — the model OVER-fires"
    else:
        design = "magnitude_only"
        why = (
            "neither rate dominates; magnitude-only is the only option that cannot manufacture "
            "occurrence, which is skepticism-ledger item 1 (M45: AP loss scales with firing)"
        )

    if math.isnan(cv_dominant) or cv_dominant > MAX_CV:
        return {
            "design": design,
            "stop": True,
            "reason": (
                f"STOP-gate (a): CV of the dominant rate is {cv_dominant:.4g} > {MAX_CV} or "
                "undefined — the distribution is not a stable target"
            ),
            "why": why,
        }
    return {"design": design, "stop": False, "reason": why, "why": why}
