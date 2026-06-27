"""Falsification stub — curriculum/scheduled-sampling: missing inverse_sigmoid k>=1 validation.

/falsify "the curriculum learning is 100% correctly implemented" (2026-06-26) found that
ScheduledSamplingMixer (and HydraNetConfig.validate_scheduled_sampling_params) validate k<1 as
INVALID for 'exponential' but are SILENT on the symmetric Bengio-2015 requirement k>=1 for
'inverse_sigmoid'. Consequences of the gap:
  - k in (0,1): runs a silently WRONG schedule shape — epsilon starts HIGH (~0.77 at k=0.3)
    instead of ~0, so there is no teacher-forcing warmup and the "curriculum" is defeated.
  - k == 0: raises a bare ZeroDivisionError deep in get_epsilon mid-training (no upfront guard).
SOFT falsification: missing validation on a supported schedule. RED until the guard is added to
ScheduledSamplingMixer.__init__ AND HydraNetConfig.validate_scheduled_sampling_params.
"""

import inspect

import pytest

from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer


def test_inverse_sigmoid_rejects_k_below_1():
    """Bengio 2015 inverse-sigmoid decay requires k>=1; the mixer must reject k<1 at construction
    (symmetric to the existing exponential k<1 guard), not run a wrong schedule shape."""
    with pytest.raises(ValueError):
        ScheduledSamplingMixer("inverse_sigmoid", epsilon_max=1.0, warmup_lessons=0, k=0.3)


def test_inverse_sigmoid_k_zero_rejected_not_zerodivision():
    """k==0 for inverse_sigmoid currently raises ZeroDivisionError deep in get_epsilon; it should
    be rejected up-front with a clear ValueError at construction instead."""
    with pytest.raises(ValueError):
        m = ScheduledSamplingMixer("inverse_sigmoid", epsilon_max=1.0, warmup_lessons=0, k=0.0)
        m.get_epsilon(5)


def test_config_validator_guards_inverse_sigmoid_k():
    """The HydraNetConfig scheduled-sampling validator guards exponential k<1 (`ss_k >= 1.0`)
    but not the symmetric inverse_sigmoid k>=1 — encode that the guard must exist."""
    from views_hydranet.utils import config_initializer

    src = inspect.getsource(config_initializer.HydraNetConfig.validate_scheduled_sampling_params)
    # The existing exponential guard uses `== "exponential"`; the symmetric inverse_sigmoid k<1
    # reject must add a dedicated `== "inverse_sigmoid"` branch. Today inverse_sigmoid appears only
    # in the `in ("inverse_sigmoid", "exponential")` k-is-None check, never as a k-bound guard.
    assert '== "inverse_sigmoid"' in src, (
        "validate_scheduled_sampling_params must reject inverse_sigmoid with k<1 (Bengio k>=1)"
    )
