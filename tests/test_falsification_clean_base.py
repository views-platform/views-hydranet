"""Falsification stubs — audit of the claim "we have a clean base to work from,
no more cleanup is needed before we proceed" (/falsify, 2026-06-06).

VERDICT: CONTESTED. The *repo-hygiene* reading SURVIVED (clean synced trees,
faithful config restoration, no leftover scaffolding/flags, all gates green on
development HEAD 77a3e2e). But the *broad* reading ("ready to barrel ahead")
is SOFT-falsified: the base still carries OPEN Tier-2 risks — chiefly C-113
(autoregressive runaway, UNFIXED) with NO automated regression guard (C-121).
The next step (rollout-training) involves RETRAINS — exactly the trigger under
which C-113 can silently recur. RESOLVED 2026-06-06: the guard landed (#76 -> tests/test_rollout_stability_guard.py
+ views_hydranet/utils/rollout_diagnostics.py). This test is now GREEN, retained as a
meta-guard that fails if the C-113 runaway guard is ever deleted (C-121).

Observation (P4, FIXED 2026-06-06): ADR-057's `**Branch:**` field still names the now-
deleted `fix/variational-dropout-autoregressive-stability` branch — stale but
historical (commits live in development history); cosmetic, non-blocking.
"""

import pathlib


def test_c113_runaway_regression_guard_exists():
    """RED (P6 / soft falsification). The 'clean base' is repo-clean but carries an
    open Tier-2 runaway (C-113) with no automated guard (C-121): the magnitude guard
    is blind to a log-space ~40 output (expm1 → ~1e17, below the 100/500/1000 log-space
    thresholds). Before relying on rollout-training retrains, a guard should fail when a
    ≥12-step free-running rollout leaves the in-range attractor. FAILS until C-121 is closed.
    """
    tests_dir = pathlib.Path(__file__).parent
    guard_tests = (
        list(tests_dir.glob("test_*runaway*"))
        + list(tests_dir.glob("test_*autoregress*guard*"))
        + list(tests_dir.glob("test_*rollout*stab*"))
    )
    assert guard_tests, (
        "No automated C-113 autoregressive-runaway regression guard found (C-121). "
        "Add a test that drives a >=12-step free-running rollout and asserts the "
        "prediction stays within the in-range attractor (log1p <= data max), so a "
        "future retrain (e.g. for rollout training) cannot silently re-introduce the "
        "runaway. See register C-113/C-121 and reports/2026-06-05_rollout_training_dossier/."
    )
