"""Channel-role ACCESSOR contract (channel-role side-quest, #114 — Phase 3 characterization net).

TDD contract for the ADR-062 §2.1 role accessors that #115 will add to `VolumeHandler`.
These pin the *target API* before the 451-edge Custodian is touched; they are **red today**
(the accessors don't exist) and turn **XPASS** when #115 lands — at which point the
`xfail(strict)` markers flip and must be removed (that XPASS is the green signal for #115).

Contract (ADR-062 §2.1), all derived from one authoritative classification, never re-derived
by consumers:
  - static_cols       — input-only static channels (ADR-060 Static); () when none.
  - target_cols       — regression_targets + classification_targets; NEVER a static (C-158).
  - model_input_cols  — dynamic features ⧺ static_cols, in feed order (what reaches the model).
  - feature_cols      — retained, demoted to a derived ALIAS of model_input_cols (back-compat).
  - I5 collapse       — static_cols == () ⇒ model_input_cols == feature_cols (byte-identical).

Construction uses the real config-driven `VolumeHandler.from_df(df, config)` path (the config
already carries every role), so these tests do not presume #115's internal signature — #115 only
has to expose accessors over the metadata `from_df` already populates. Sibling: the *symptom*
census in `test_channel_role_census.py` (C-156/157/158/159); this file is the *contract* spec.
"""

import pytest

pytest.importorskip("views_pipeline_core")

# Reuse the validated 2×2-grid config + df builder (no replication).
from tests.test_prediction_frame_suite import PF_BASE_CFG, _make_history_df
from views_hydranet.utils.volume_handler import VolumeHandler

_STATICS = ["row_coord", "col_coord"]
_XFAIL = pytest.mark.xfail(
    strict=True, reason="#115: VolumeHandler role accessors (ADR-062 §2.1) not yet implemented"
)


@pytest.fixture
def vh_coords():
    """No-coords config + 2 CoordConv static channels (from_df derives them from geometry)."""
    cfg = {**PF_BASE_CFG, "static_channels": list(_STATICS), "input_channels": 5}
    return VolumeHandler.from_df(_make_history_df(), cfg)


@pytest.fixture
def vh_no_coords():
    """Bounded no-coords baseline (static_channels absent ⇒ [])."""
    return VolumeHandler.from_df(_make_history_df(), PF_BASE_CFG)


# ── Green: the three roles, partitioned correctly (coords on) ────────────────────────────────
@_XFAIL
def test_static_cols_returns_only_the_statics(vh_coords):
    assert tuple(vh_coords.static_cols) == tuple(_STATICS)


@_XFAIL
def test_model_input_cols_is_features_then_statics_feed_order(vh_coords):
    expected = tuple(PF_BASE_CFG["features"]) + tuple(_STATICS)
    assert tuple(vh_coords.model_input_cols) == expected


@_XFAIL
def test_target_cols_is_reg_plus_cls(vh_coords):
    expected = tuple(PF_BASE_CFG["regression_targets"]) + tuple(
        PF_BASE_CFG["classification_targets"]
    )
    assert tuple(vh_coords.target_cols) == expected


@_XFAIL
def test_classification_targets_present_in_target_cols(vh_coords):
    for cls_target in PF_BASE_CFG["classification_targets"]:
        assert cls_target in vh_coords.target_cols


@_XFAIL
def test_feature_cols_is_alias_of_model_input_cols(vh_coords):
    assert tuple(vh_coords.feature_cols) == tuple(vh_coords.model_input_cols)


# ── Red: the partition guarantees (a static is never a target — the C-158 root) ──────────────
@_XFAIL
def test_target_cols_excludes_every_static(vh_coords):
    """C-158: an input-only static must NEVER be a target (and thus never a curriculum subject)."""
    assert set(vh_coords.static_cols).isdisjoint(set(vh_coords.target_cols))


@_XFAIL
def test_roles_partition_cleanly(vh_coords):
    static = set(vh_coords.static_cols)
    assert static.issubset(set(vh_coords.model_input_cols))  # statics ARE model inputs
    assert static.isdisjoint(set(vh_coords.target_cols))     # statics are NOT targets


# ── Beige: the I5 collapse (no statics ⇒ byte-identical to today's feature_cols) ─────────────
@_XFAIL
def test_collapse_when_no_statics(vh_no_coords):
    """static_cols == () ⇒ model_input_cols == feature_cols (the back-compat invariant).
    (NOT '== target_cols': PF_BASE_CFG has classification_targets, so target_cols ⊋ features.)"""
    assert tuple(vh_no_coords.static_cols) == ()
    assert tuple(vh_no_coords.model_input_cols) == tuple(vh_no_coords.feature_cols)
