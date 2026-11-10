"""Midranks, and the defect they replace: ``argsort(argsort(x))`` is a function of CELL ORDER.

Epic #579, story S8 #587. The failure this guards was caught by a gate, not by inspection: ``G-L3``
reported that the nu dial moved AUC by 1.571e-03 while holding the key's value multiset fixed --
which is impossible. The dial was right; the rank estimator was wrong.

⭐ ``test_argsort_ranks_are_refused`` is the discriminating half: it reimplements the OLD rank and
asserts the property FAILS under it. Without that half, a guard that accepts everything would look
perfect (the discipline of ``test_conditional_magnitude.py`` and ``test_e1_data.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-10_rank_coupling_verdict/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "rank_coupling_verdict.py").exists(),
    reason="the rank-coupling-verdict dossier tools are not present in this tree",
)

if (_TOOLS / "rank_coupling_verdict.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_rank_coupled_body/tools"))


@pytest.fixture
def rcv():
    import rank_coupling_verdict

    return rank_coupling_verdict


def _old_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """⛔ THE DEFECT, verbatim: Mann-Whitney on ``argsort(argsort(...))`` instead of midranks."""
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    r = np.argsort(np.argsort(score)) + 1.0
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _quantised_case():
    """-> (score, positive). ⭐ Quantised at 1/16, like a mean of 16 draws: heavily tied."""
    rng = np.random.default_rng(3)
    score = rng.integers(0, 6, 400).astype(np.float64) / 16.0
    positive = rng.random(400) < 0.45
    return score, positive


def test_midrank_matches_the_definition_on_ties(rcv):
    """⭐ The correct input, asserted to PASS: tied values must share the AVERAGE rank."""
    x = np.array([5.0, 1.0, 5.0, 3.0, 5.0])
    # sorted: 1(rank1), 3(rank2), 5,5,5 (ranks 3,4,5 -> midrank 4)
    assert rcv.midrank(x).tolist() == [4.0, 1.0, 4.0, 2.0, 4.0]


def test_spearman_is_exactly_one_on_a_monotone_map_with_ties(rcv):
    """⭐ The property the headline depends on: a PERFECT ordering must read 1.0, not 0.9963."""
    truth = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 7.0, 7.0, 90.0])
    key = truth * 3.0 + 1.0  # strictly monotone, same tie structure
    assert rcv.spearman(key, truth) == pytest.approx(1.0)


def test_auc_is_invariant_to_permuting_values_among_positives(rcv):
    """⭐⭐ G-L3's invariant: reassigning the positives' OWN value multiset cannot move AUC."""
    score, positive = _quantised_case()
    rng = np.random.default_rng(5)
    moved = score.copy()
    idx = np.flatnonzero(positive)
    moved[idx] = score[idx][rng.permutation(idx.size)]

    before, after = rcv.auc(score, positive), rcv.auc(moved, positive)
    assert before == pytest.approx(after, abs=1e-15), (
        f"AUC moved by {abs(before - after):.3e} under a permutation that cannot change it"
    )


def test_argsort_ranks_are_refused(rcv):
    """⛔ The discriminating half: the old estimator must VIOLATE the same invariant."""
    score, positive = _quantised_case()
    rng = np.random.default_rng(5)
    moved = score.copy()
    idx = np.flatnonzero(positive)
    moved[idx] = score[idx][rng.permutation(idx.size)]

    drift = abs(_old_auc(score, positive) - _old_auc(moved, positive))
    assert drift > 1e-6, (
        f"the old argsort-based AUC drifted only {drift:.3e} on this input, so this test does "
        "not reproduce the defect and the guard above is vacuous"
    )

    # ⭐ and the old rank is not even a function of x alone: shuffling the ARRAY ORDER of a tied
    # input changes the ranks it reports.
    x = np.array([1.0, 1.0, 1.0, 2.0])
    assert np.argsort(np.argsort(x)).tolist() != np.argsort(np.argsort(x[::-1])).tolist()[::-1]
    assert rcv.midrank(x).tolist() == rcv.midrank(x[::-1]).tolist()[::-1]
