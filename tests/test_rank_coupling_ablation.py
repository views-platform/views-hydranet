"""The ``ranks FIXED`` ablation must HOLD the rank map, not re-sort the key.

Epic #579, story S3 #582. ⛔ The defect this guards: ``sim_rank_coupling.simulate`` used to do
``key = np.sort(key)`` inside the ablation, which makes ``np.argsort`` the identity and therefore
assigns magnitudes in **cell-index order**. That decouples the assignment from the ordering
entirely instead of holding it, and it printed the largest spread in the Q-C table as an artefact.

⭐ ``test_old_form_is_refused`` is the discriminating half: it reimplements the OLD line, then
asserts
the guard rejects it. Without that half, a guard that accepts everything would look perfect
(``test_conditional_magnitude.py``'s discipline: every gate gets a correct input asserted to pass
AND a deliberately broken input asserted to be refused).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-04_rank_coupled_body/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "sim_rank_coupling.py").exists(),
    reason="the rank-coupled-body dossier tools are not present in this tree",
)

if (_TOOLS / "sim_rank_coupling.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))


N_CELLS = 400
N_SAMPLES = 24


def _inputs(seed: int = 7):
    """-> (pi, pool, order_key). A gate that fires a varying subset, so the ablation has work."""
    rng = np.random.default_rng(seed)
    pi = rng.uniform(0.2, 0.8, N_CELLS)
    pool = np.sort(rng.integers(1, 500, 900).astype(np.float64))
    order_key = rng.standard_normal(N_CELLS)
    return pi, pool, order_key


def _simulate_ranks_fixed(pi, pool, order_key, *, old_form: bool, r: int = N_SAMPLES):
    """-> y[r, N], fired[r, N]. ``old_form`` reimplements the DEFECT for the refusal half."""
    rng = np.random.default_rng(0)
    n_cells = pi.size
    y = np.zeros((r, n_cells))
    fired = np.zeros((r, n_cells), bool)
    q_fixed = np.argsort(np.argsort(order_key, kind="stable")) / max(n_cells - 1, 1)
    for i in range(r):
        f = rng.random(n_cells) < pi
        idx = np.flatnonzero(f)
        n = idx.size
        if n == 0:
            continue
        fired[i, idx] = True
        d = np.sort(rng.choice(pool, n, replace=True))
        if old_form:
            key = np.sort(order_key[idx])  # ⛔ THE DEFECT, verbatim
            y[i, idx[np.argsort(key, kind="stable")]] = d
        else:
            y[i, idx] = d[np.rint(q_fixed[idx] * (n - 1)).astype(np.int64)]
    return y, fired


def _pairwise_violation_rate(y, fired, order_key):
    """-> the fraction of (cell pair, draw) comparisons in which the magnitude CONTRADICTS the key.

    ⭐ This is the property the ablation claims, stated so that it is EXACTLY held rather than held
    up to sampling noise. If a cell's rank is a property of the cell, then for any two cells whose
    keys are ordered, their assigned magnitudes are ordered the same way in **every** draw in which
    both fired. ⇒ 0.0 under a held rank map; ~0.5 when the assignment is made in cell-index order.

    ⚠️ Deliberately not a quantile-invariance check. A cell's quantile AMONG THE FIRED CELLS cannot
    be draw-invariant while the firing set varies -- it fluctuates at ~sqrt(q(1-q)/n) -- so such a
    check measures the gate's randomness, not the rank map. That mistake cost a test iteration.
    """
    rng = np.random.default_rng(11)
    a, b = rng.integers(0, y.shape[1], 4000), rng.integers(0, y.shape[1], 4000)
    keep = order_key[a] != order_key[b]
    a, b = a[keep], b[keep]
    bad = total = 0
    for i in range(y.shape[0]):
        m = fired[i][a] & fired[i][b]
        if not m.any():
            continue
        ya, yb = y[i, a[m]], y[i, b[m]]
        ka, kb = order_key[a[m]], order_key[b[m]]
        comparable = ya != yb
        bad += int((np.sign(ya - yb)[comparable] != np.sign(ka - kb)[comparable]).sum())
        total += int(comparable.sum())
    assert total > 500, f"too few comparable pairs ({total}) for this test to mean anything"
    return bad / total


def test_corrected_form_holds_the_rank_map():
    """⭐ The correct input, asserted to PASS — without which the guard proves nothing."""
    pi, pool, order_key = _inputs()
    y, fired = _simulate_ranks_fixed(pi, pool, order_key, old_form=False)

    v = _pairwise_violation_rate(y, fired, order_key)
    assert v == 0.0, f"a held rank map admits NO order violations, measured {v:.4f}"


def test_old_form_is_refused():
    """⛔ The discriminating half: the defect, reimplemented, must FAIL both assertions."""
    pi, pool, order_key = _inputs()
    y, fired = _simulate_ranks_fixed(pi, pool, order_key, old_form=True)

    v = _pairwise_violation_rate(y, fired, order_key)
    assert v > 0.3, (
        f"the old form assigns in CELL-INDEX order, so it must violate the key on roughly half "
        f"of all comparisons; measured {v:.4f} means this test does not discriminate and the "
        "guard above is vacuous"
    )


def test_the_shipped_simulate_uses_the_corrected_form():
    """⭐ Guards the FILE, not a copy of it — a reimplementation-only test can drift away."""
    import sim_rank_coupling as src

    pi, pool, order_key = _inputs()
    truth = np.zeros(N_CELLS)
    original_r = src.R
    try:
        src.R = N_SAMPLES
        y, fired, _ = src.simulate(
            pi, pool, truth, order_key, np.random.default_rng(0), fix_ranks=True
        )
    finally:
        src.R = original_r

    v = _pairwise_violation_rate(y, fired, order_key)
    assert v == 0.0, f"the shipped simulate() does not hold the rank map, violation rate {v:.4f}"
