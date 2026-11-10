"""The ρ_event instrument's own guards — above all that its estimator is the PUBLISHED one.

⭐⭐ Why this file is short and why it exists: `2026-09-30_evaluation_learning_report.md` §9c
records that `ordering_by_index.py` was the fifth instrument of its dossier and the **first whose
headline did not have to be withdrawn**, and that the first of the three reasons was *"the
estimator is IMPORTED from the instrument it is compared against, not copied, and a test asserts
the same function objects."* ⛔ Both of that dossier's HOLDs were about a baseline that moved.

This is that test, for `reports/2026-11-10_ranking_head_dossier/tools/rho_event_readout.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-10_ranking_head_dossier/tools"
_PUBLISHED = _REPO / "reports/2026-11-10_rank_coupling_verdict/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "rho_event_readout.py").is_file(),
    reason="the ρ_event readout has not been built yet",
)


@pytest.fixture(scope="module")
def mods():
    """-> (the readout, the module that published +0.3127). Imported, not reimplemented."""
    pytest.importorskip("numpy")
    for p in (_TOOLS, _PUBLISHED, _REPO / "reports/2026-11-04_delivered_gate_lookup/tools"):
        sys.path.insert(0, str(p))
    import rank_coupling_verdict
    import rho_event_readout

    return rho_event_readout, rank_coupling_verdict


def test_the_estimator_is_the_same_function_object_as_the_published_one(mods):
    """⛔⛔ The whole point. A near-copy that drifts is how a baseline moves under a comparison."""
    readout, published = mods
    assert readout.spearman is published.spearman, (
        "the readout's `spearman` is NOT the same function object as the one that published "
        "+0.3127 — it is a copy, and a copy can drift without either file changing visibly"
    )


def test_the_published_estimator_still_reduces_on_midranks(mods):
    """⛔ Importing the right function is not enough if its tie handling is swapped out.

    `C-289`/`C-419`: on tied data `argsort(argsort(x))` is a function of **cell order**, which on
    this grid is **geography**. Midranks are the reason `ρ_event` is a statistic and not a raster
    sweep, so the property is asserted rather than assumed from the import.
    """
    import numpy as np

    _readout, published = mods
    # A block of four ties must take the average of the positions it occupies: (2+3+4+5)/4 = 3.5.
    got = published.midrank(np.array([1.0, 7.0, 7.0, 7.0, 7.0, 9.0]))
    assert got.tolist() == [1.0, 3.5, 3.5, 3.5, 3.5, 6.0]
    # ⛔ The discriminating half: the banned form gives the tied block four DIFFERENT ranks, in
    # input order, which is what makes it a function of geography.
    banned = np.argsort(np.argsort(np.array([1.0, 7.0, 7.0, 7.0, 7.0, 9.0]))) + 1
    assert banned.tolist() != got.tolist()


def test_the_control_anchors_are_the_published_figures(mods):
    """⛔ The anchors are the control. If they are edited the pin is meaningless, so they are
    asserted against the finish line in `00_README` rather than left as bare constants."""
    readout, _published = mods
    assert (readout._ANCHOR_BODY, readout._ANCHOR_BODY_SD) == (0.3127, 0.0713)
    assert (readout._ANCHOR_GATE, readout._ANCHOR_GATE_SD) == (0.4418, 0.0583)
    readme = (_REPO / "reports/2026-11-10_ranking_head_dossier/00_README.md").read_text(
        encoding="utf-8"
    )
    assert f"+{readout._ANCHOR_BODY}" in readme, (
        "the readout's body anchor is not the number 00_README names as the finish line"
    )


def test_a_degenerate_score_field_is_refused_not_scored(mods):
    """⛔⛔ §0.2 of the evaluation report: a delivery served **0.0 for all 2,330,712 cells** on
    global land, in all three targets, and **every structural check passed** — valid manifest,
    right coverage, file size 98.1% of a real run's. No metric in the inventory flagged it.

    ⇒ `distinct_share` is what G-2 reads, and an all-constant field must land at the floor.
    """
    import numpy as np

    readout, _published = mods
    assert readout.distinct_share(np.zeros(2_330_712)) == pytest.approx(1 / 2_330_712)
    assert readout.distinct_share(np.zeros(2_330_712)) < readout._DISTINCT_FLOOR
    # ⛔ The discriminating half: a continuous field must clear the same floor comfortably.
    rng = np.random.default_rng(0)
    assert readout.distinct_share(rng.standard_normal(10_000)) > readout._DISTINCT_FLOOR
