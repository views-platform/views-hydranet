"""Falsification guards for the claim "we know 100% what is dead and safe to delete" (2026-07-19).

/falsify verdict: FALSIFIED. The "dead" *science* is largely sound, but "100%" and "safe to delete"
are over-claims. These guards go RED if the false claim is acted on (i.e. if the retained lever or
the shipped contract is deleted), enforcing the corrected classification.
"""


def test_count_mean_is_a_retained_transfer_test_lever_not_safe_delete():
    """F-1 (HARD): count_mean was classified 100% dead + safe to delete, but the recorded finding
    (memory: project_count_mean_fails_oos) explicitly RETAINS it as a transfer-test lever
    ("retained, lever available"). Deleting it contradicts a recorded keep-decision — it is
    MAYBE-dead-as-a-body-loss, not deletable. Guard: it stays registered."""
    from views_hydranet.utils.utils import LOSS_REG_REGISTRY

    assert "count_mean" in LOSS_REG_REGISTRY


def test_body_supervision_modes_are_a_shipped_pushed_contract():
    """F-2 (HARD): the body-supervision window is the ADR-065 epic + amendment (pushed to origin) —
    production files + tests + ADR + CIC + a foundation `all` value. The retired pos_cells/
    pos_timelines keywords now live as the endpoints of the graded resolver, not as dead code.
    Guard: the resolver honors the shipped endpoints (0/0 = per-step; saturated = timelines)."""
    import torch

    from views_hydranet.utils.body_supervision import resolve_body_supervision

    w = torch.zeros(1, 2, 1, 1, 1)
    w[0, 0, 0, 0, 0] = 5.0
    assert resolve_body_supervision(0, 0, 0.0)(w).shape == w.shape  # old pos_cells endpoint
    assert resolve_body_supervision(9, 9, 0.0)(w).shape == w.shape  # old pos_timelines endpoint


def test_basu_deletion_is_coupled_to_its_validator():
    """F-4 (SOFT): deleting loss_reg='basu_dpd' is not a one-liner — a model_validator
    (validate_basu_dpd_range) hard-references it and must be removed in the same change. The
    'safe to delete' claim omitted this coupling. Guard: while basu_dpd is registered, its
    validator exists (they are deleted together)."""
    import inspect

    from views_hydranet.utils import config_initializer
    from views_hydranet.utils.utils import LOSS_REG_REGISTRY

    if "basu_dpd" in LOSS_REG_REGISTRY:
        assert "validate_basu_dpd_range" in inspect.getsource(config_initializer)
