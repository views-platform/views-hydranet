"""Failing stubs from the /falsify audit of "we have what we need to plan the next experiments".

⛔ DELIBERATELY NOT UNDER `tests/`. These encode findings about a PLAN, not defects in shipped
code, and the repo gates on a green suite (`test_falsification_repo_clean.py`). Put here so they
are durable and runnable without turning the suite red — the C-106 lesson about load-bearing
things with no home, applied to a falsification result rather than an estimator.

Run: pytest reports/2026-10-26_body_optimisation_gap/test_falsify_next_experiments.py

Each test fails TODAY and is the acceptance criterion for the corresponding finding.
"""

import numpy as np
import pytest

GRID_LAND_CELLS = 64_818


def test_H1_vs_p_weight_matrix_is_truncated_so_the_diagnostic_is_computable():
    """H-1. VS-p (Scheuerer & Hamill 2015 Eq. 3) sums over ALL d^2 ordered pairs.

    The paper's own weight recommendation is inverse distance, which is POSITIVE FOR EVERY PAIR
    and so offers no truncation. At the land grid that is 4.2e9 pairs before any draw axis.
    The plan must declare a cutoff radius; propriety survives it because the paper grants
    properness "for any choice of nonnegative weights" and zero is nonnegative.
    """
    pytest.fail(
        f"No cutoff radius is declared. All-pairs VS-p on d={GRID_LAND_CELLS:,} is "
        f"{GRID_LAND_CELLS**2:,} pairs (x128 draws = {GRID_LAND_CELLS**2 * 128:,} terms). "
        "Declare a radius (r=4 -> 3.1M pairs, r=8 -> 12.7M) AND record it as a deviation from "
        "the paper's inverse-distance recommendation."
    )


def test_H2_vs_p_has_a_falsifier_for_the_variance_inflation_loophole():
    """H-2. The new term's own cheap-satisfaction route, with no test designed for it.

    For p=2 with conditionally independent heads,
        E|X_i - X_j|^2 = (m_i - m_j)^2 + v_i + v_j.
    The model can raise the left-hand side at EVERY lag by the same amount by inflating per-cell
    variance v (lowering theta), with the arrangement untouched. That is structurally the same
    defect as the amplification loophole proved in `_structure_penalty` on 2026-10-26.

    The acceptance criterion is a transform in the battery -- "inflate v, hold m" -- plus a
    pre-committed requirement on how VS-p must respond to it.
    """
    pytest.fail(
        "The battery has amplify / shuffle / smooth / displace and NO variance-inflation "
        "transform. Until VS-p is measured against one, the term is proposed as the fix for a "
        "loophole while carrying an un-probed loophole of the same shape."
    )


def test_H2b_vs_p_nugget_cancels_or_is_declared_irreducible():
    """H-2, second half: does a lag-weighted VS-p see the nugget as signal?

    v_i + v_j is CONSTANT across lags. A term that reads only the lag PROFILE would be immune;
    a term that reads the level is not. Which one VS-p is, under our truncated weights, decides
    whether H-2 is fatal or merely a cost.
    """
    pytest.fail("Unmeasured. Decide by measurement, not by assuming the weights handle it.")


def test_S1_every_concrete_family_implements_variance():
    """S-1. `variance()` does not exist; `base.py` has `mean` and no `variance`.

    There are FOUR concrete families, not one, and two of them are not one-liners: truncated_nb
    needs the zero-truncated second moment, mixture_nb the mixture-variance formula. Adding
    `variance` as an @abstractmethod breaks all four until each is implemented.
    """
    from views_hydranet.distributions.registry import resolve_family

    missing = [n for n in ("nb", "truncated_nb", "zinb", "mixture_nb")
               if (f := resolve_family(n)) is not None and not hasattr(f, "variance")]
    assert not missing, f"families without variance(): {missing}"


def test_S2_head_draws_come_from_the_stored_parameters_not_from_a_new_pod_run():
    """S-2. The K=128 / D=2 draws asked of Simon do not require any model run.

    `d4_result_hf0125.npz` (control) and `d4_result_ARM_hf0126.npz` (arm) already store per-cell
    mu, theta and gate for the full grid. Head draws are samples from THOSE parameters: no
    forward pass, no pod, no cost. The ask implied otherwise.
    """
    from pathlib import Path

    d = Path("reports/2026-10-12_underfire_diagnosis")
    for f in ("d4_result_hf0125.npz", "d4_result_ARM_hf0126.npz"):
        z = np.load(d / f)
        assert {"mu", "theta"} <= set(z.files), f"{f} lacks mu/theta: {z.files}"


def test_S3_sigma_hat_exists_before_any_arm_is_pre_registered():
    """S-3. The claim is TRUE for the diagnostic and FALSE for the arm.

    Outcome report §8 limit 1: "Nothing here bounds run-to-run variance. sigma-hat is still
    unmeasured (views-models #572/#566)." An arm pre-registration needs an MDE; without
    sigma-hat there is none, and C-307 makes an n=1 null INCONCLUSIVE rather than negative.
    """
    pytest.fail(
        "sigma-hat unmeasured (views-models #572/#566). Plan the DIAGNOSTIC now; the ARM "
        "cannot carry a decision rule until run-to-run variance is bounded."
    )
