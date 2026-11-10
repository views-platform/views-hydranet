"""⛔ The guards on the TEACHER-FORCED ρ_event table — the instrument phase 3 reads.

⭐⭐ Built to the same recipe as `test_rho_event_readout.py`, which exists because
`reports/2026-09-30_evaluation_learning_report.md` §9c names three things that made
`ordering_by_index.py` the first instrument of its dossier whose headline did not have to be
withdrawn. The one this file carries is the first: **the estimator is imported, never copied**, and
that is asserted on the *function objects* so a near-copy cannot drift in.

⚠️ And one thing this file exists for specifically: on 2026-11-10 the fed→forecast month map was
written **inverted**, and it was caught by a population-coverage refusal rather than by review.
Both this dossier's alignment tool and `mu_ordering.py` carried a *sentence* with the sign the
wrong way round while mu_ordering's code was correct. ⇒ The map is now a named function with its
own test and an explicit discriminating half.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-10_ranking_head_dossier/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "rho_event_teacher_forced.py").is_file(),
    reason="the teacher-forced readout has not been built yet",
)

if _TOOLS.is_dir():
    sys.path.insert(0, str(_TOOLS))


@pytest.fixture(scope="module")
def tf():
    return pytest.importorskip("rho_event_teacher_forced")


# ----------------------------------------------------------------------------------------------
# ⭐⭐ §9c rule 1 — the estimator and the thresholds are IMPORTED, never copied.
# ----------------------------------------------------------------------------------------------


def test_the_estimator_and_thresholds_are_the_same_objects_not_copies(tf):
    """⭐⭐ Same function objects as the delivered readout and the alignment gate.

    Both of §9c's withdrawn HOLDs were about a baseline that had been re-implemented beside the
    thing it was baselining. An `is` check is the only form of this assertion that a near-copy
    cannot pass.
    """
    readout = pytest.importorskip("rho_event_readout")
    align = pytest.importorskip("gate_alignment")
    assert tf.rho_event is readout.rho_event, "ρ_event has been re-implemented, not imported"
    assert tf.distinct_share is readout.distinct_share, "distinct_share is a copy"
    assert tf._fold_inputs is readout._fold_inputs, "the population is built twice, not once"
    assert tf.anomaly_r is align.anomaly_r, "the alignment statistic is a copy"
    assert tf.OFFSETS is align.OFFSETS and tf.ANOM_R == align.ANOM_R, (
        "the alignment thresholds are restated rather than imported, so the two instruments can "
        "drift apart — which is exactly what the pre-registration forbade"
    )


# ----------------------------------------------------------------------------------------------
# ⛔⛔ The fed -> forecast month map. The defect this file was written for.
# ----------------------------------------------------------------------------------------------


def test_the_offset_map_reads_the_forecast_for_month_m_at_fed_index_m_plus_off(tf):
    """⛔⛔ `forecast_index` must agree with `mu_ordering.py`'s `src = fed_idx[month + best]`.

    At the measured offset `-1`, the forecast for month 463 is read at fed index of month **462**.
    """
    assert tf.forecast_index([462, 463, 464], -1) == {463: 0, 464: 1, 465: 2}


def test_the_offset_map_refuses_the_inverted_direction(tf):
    """⛔ The discriminating half: the inverted map must NOT produce this design's months.

    The inverse (`m + off`) sends fed 462 to forecast 461. With the dump's fed range 421..469 it
    covers forecast months 420..468 and so cannot score month **469** at all — which is precisely
    how the real inversion was caught, by a coverage refusal rather than by review.
    """
    inverted = {m + (-1): i for i, m in enumerate([462, 463, 464])}
    assert inverted == {461: 0, 462: 1, 463: 2}
    assert 465 not in inverted, "the two directions are indistinguishable on this input"
    fed = list(range(421, 470))
    assert max(tf.forecast_index(fed, -1)) == 470, "the correct map must reach forecast month 469"
    assert max({m + (-1): i for i, m in enumerate(fed)}) == 468, (
        "the inverted map must fall one month short of 469 — if it does not, this test cannot "
        "tell the two directions apart and the assertion above is vacuous"
    )


# ----------------------------------------------------------------------------------------------
# ⛔ The mirror of `G-TF`, and `G-CELLS`. Both fire BEFORE any data is loaded.
# ----------------------------------------------------------------------------------------------


def _fabricate(tmp_path: Path, *, mode: str, cells: np.ndarray, gate: bool = True) -> Path:
    p = tmp_path / f"field_{mode}_{gate}.npz"
    payload = {
        "score": np.zeros((2, cells.size), dtype=np.float32),
        "months": np.asarray([462, 463], dtype=np.int32),
        "land_cells": cells,
        "input_mode": np.asarray(mode),
        "channel": np.asarray("mu"),
    }
    if gate:
        payload["gate"] = np.zeros((2, cells.size), dtype=np.float32)
    np.savez_compressed(p, **payload)
    return p


def test_a_delivered_field_is_refused_by_the_mirror_of_G_TF(tf, tmp_path):
    """⛔ `C-399` in the other direction: a delivered field may not join a teacher-forced table."""
    cells = np.arange(3, dtype=np.int64)
    with pytest.raises(SystemExit, match="G-TF REFUSED"):
        tf.load_tf_field(_fabricate(tmp_path, mode="delivered", cells=cells), [463], cells)


def test_the_mirror_refusal_lets_a_teacher_forced_field_through_to_the_next_gate(tf, tmp_path):
    """⛔ The discriminating half: a refusal that rejects everything proves nothing.

    A teacher-forced field on PERMUTED cells must get past `G-TF` and be stopped by `G-CELLS` —
    which proves the first gate reads the mode and the second reads the cell ORDER, not the length.
    """
    cells = np.arange(3, dtype=np.int64)
    permuted = np.asarray([2, 0, 1], dtype=np.int64)
    with pytest.raises(SystemExit, match="G-CELLS REFUSED"):
        tf.load_tf_field(_fabricate(tmp_path, mode="teacher_forced", cells=permuted), [463], cells)


def test_a_field_without_its_gate_is_refused_because_the_offset_cannot_be_resolved(tf, tmp_path):
    """⛔ The offset is re-resolved per field from that field's own gate, so it is required."""
    cells = np.arange(3, dtype=np.int64)
    path = _fabricate(tmp_path, mode="teacher_forced", cells=cells, gate=False)
    with pytest.raises(SystemExit, match="G-0 REFUSED"):
        tf.load_tf_field(path, [463], cells)


# ----------------------------------------------------------------------------------------------
# ⭐⭐ The control pin. §9c rule 2 — the control is pinned by arithmetic, not by belief.
# ----------------------------------------------------------------------------------------------


def test_the_teacher_forced_control_is_pinned_and_is_not_the_delivered_anchor(tf):
    """⭐⭐ The pin must be set, and must NOT be the delivered figure.

    ⚠️ `+0.3127` is the body's **delivered autoregressive** `ρ_event`. The teacher-forced control
    measures `+0.5189` on the same weights, and the `+0.2062` difference is the `C-399` carry and
    the `C-317` checkpoint identity together. Pasting the delivered number in here would silently
    lower an arm's bar by that whole amount, which is the single most expensive mistake available
    in this table.
    """
    readout = pytest.importorskip("rho_event_readout")
    assert tf._ANCHOR_TF_MU is not None, (
        "the pin is unset, so the tool will print a control it has not verified and refuse arms"
    )
    assert tf._ANCHOR_TF_MU == pytest.approx(0.5189), (
        "the pinned teacher-forced control moved; it was measured at +0.5189 on "
        "score_grid_CONTROL_MU_v20261012.npz and may only change WITH a re-measurement"
    )
    assert tf._ANCHOR_TF_MU != readout._ANCHOR_BODY, (
        "the teacher-forced pin equals the DELIVERED anchor +0.3127 — the two are different "
        "populations and an arm scored against the wrong one gets a free +0.2062"
    )
