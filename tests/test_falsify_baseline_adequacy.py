"""⛔ FALSIFICATION STUBS — "we know all we need to move on with this" is FALSIFIED.

Produced by ``/falsify`` on 2026-11-07 against the adequacy claim for ``M100``
(``reports/2026-11-07_baseline_scorecard/``). Four hard and four soft falsifications.

⭐ Every stub is ``xfail(strict=True)``, deliberately:
  - the suite stays green, so this file does not block unrelated work;
  - a strict xfail that starts PASSING is reported as a FAILURE, so the moment someone closes the
    defect the suite goes red and the marker has to be removed. That is the enforcement.

⛔ These are not tests of the instrument's arithmetic — ``tests/test_baseline_scorecard.py`` covers
that and its gates reproduce every published figure. These are tests of what the baseline DOES NOT
CONTAIN, which is the only thing an adequacy claim can fail on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOSSIER = _REPO / "reports/2026-11-07_baseline_scorecard"
_PROGRAMME = _REPO / "reports/2026-11-04_rank_coupled_body_programme.md"

pytestmark = pytest.mark.skipif(
    not _DOSSIER.exists(), reason="the baseline-scorecard dossier is not present in this tree"
)


# ── HARD H1 · the quantity v1 orders by has no baseline ──────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="HARD: M100 baselines mean_draw and the gate, not mu")
def test_the_baseline_contains_the_ordering_source_v1_actually_uses() -> None:
    """§2.3 of the programme: ``rho_t`` comes from the REGRESSION OUTPUT, not the composed draws
    and not the gate. Its stated measurement is ``DSC_ISO`` on event cells per candidate order --
    *"the cheapest measurement in the whole programme"*, deadline *"before the first fit"*.

    M100's axes 3 and 4 rank by ``mean_draw`` (the delivered composed emission) and by ``gate``.
    Neither is the body's ``mu``. ⇒ The baseline measures the NEIGHBOURS of the quantity v1
    changes, never the quantity itself.
    """
    out = (_DOSSIER / "results/baseline_scorecard.out").read_text()
    assert "mu" in out or "DSC_ISO" in out


# ── HARD H2 · a committed ledger row restates a blocker the record already retired ───────────


# ⭐ CLOSED 2026-11-07, hours after this stub was written. It XPASSed under `xfail(strict=True)`,
# which is a FAILURE, so the marker came off and this is now a live regression test: the dossier
# not reacquire the retired blocker.
def test_the_dossier_does_not_claim_mu_requires_a_re_emit() -> None:
    """⛔⛔ THE SAME ERROR, THIRD TIME, AND IT IS IN THE RECORD AS MINE.

    ``reports/2026-11-04_rank_coupled_body_programme.md`` line 1271 lists, in a table of errors
    caught in that session: *"claimed `mu` 'is not on disk (verified)' and that recovering it
    needed a GPU"* -- caught by ``/falsify``, *"one CPU forward, 0.75 s, 1.06 GB"*. §2.8 then
    records #568 as *"No longer a prerequisite for anything ... Convenience only"*.

    M99 says the confound *"needs `mu` ... which is NOT ON DISK for any delivered run -- that is
    `S8` (#568) and it needs a re-emit"*. M100 and ``00_README.md`` repeat the attribution.

    ⇒ The confound those rows call blocked is closeable by one CPU forward. The literal words "not
    on disk" are true; the inference that it therefore needs #568 or a re-emit is false, and it is
    the part a reader acts on.

    ⭐ CLOSED: the README, `M99`, `M100` and `ordering_by_index.py`'s docstring now all carry the
    correction, and `N10` of the evaluation learning report records the repetition.
    """
    readme = (_DOSSIER / "00_README.md").read_text()
    assert "#568" not in readme or "one CPU forward" in readme
    # ⛔ and the positive form, so this cannot pass by the mention being deleted
    assert "one CPU forward" in readme and "0.75 s" in readme


# ── HARD H3 · the published ± is the wrong uncertainty, and the points are not central ───────


@pytest.mark.xfail(
    strict=True, reason="HARD: bootstrap sd is 1.6x-3.1x the published across-arm sd"
)
def test_the_quoted_uncertainty_covers_sampling_not_only_reproducibility() -> None:
    """Measured by this audit, 300 bootstrap replicates resampling the 2,905 event cells on
    ``ctlfullzero_fortytwo`` (the owed measurement, outstanding since ``M98``):

        mean_draw  point 9.2727x  95% CI [9.0698, 12.4286]  sd 1.2246  vs published +-0.7638
        gate       point 6.0400x  95% CI [6.0400,  9.2727]  sd 1.1974  vs published +-0.3972

    ⇒ 1.60x and 3.01x wider. ``8.61x +- 0.76`` and ``6.32x +- 0.40`` state REPRODUCIBILITY and are
    read as precision.
    """
    out = (_DOSSIER / "results/baseline_scorecard.out").read_text()
    # ⛔ NOT `"bootstrap" in out` -- the dossier names the bootstrap in order to say it is MISSING,
    # so that assertion passed on prose. The property is that it is no longer owed.
    assert "OWED" not in out.upper()


@pytest.mark.xfail(strict=True, reason="HARD: P(bootstrap <= point) = 0.19 and 0.043")
def test_the_published_crossings_are_not_at_the_centre_of_their_own_distribution() -> None:
    """⭐⭐ NEW, and it was not on the author's list of weaknesses.

    P(bootstrap <= point) is **0.190** for ``mean_draw`` and **0.043** for the gate; the bootstrap
    medians are 9.5224x and 7.0833x against points of 9.2727x and 6.0400x. Both published figures
    sit near the OPTIMISTIC EDGE of their own sampling distribution -- the signature of a
    minimum-type statistic (``crossing`` takes the smallest ratio reaching the target) read off an
    in-sample isotonic fit.

    ⛔ This stub does NOT assert a corrected value: the direction of the bias is indicated, its
    SIZE is not established by one arm and 300 replicates. What is established is that the
    interval is strongly asymmetric and a symmetric +- misreports it.
    """
    out = (_DOSSIER / "results/baseline_scorecard.out").read_text()
    assert "asymmetric" in out.lower() or "optimistic" in out.lower()


# ── HARD H4 · the axis the experiment will be JUDGED on is not baselined ─────────────────────


@pytest.mark.xfail(strict=True, reason="HARD: none of 2.6's four-part battery is in M100")
def test_the_baseline_covers_the_score_the_verdict_will_be_read_on() -> None:
    """§2.6's default battery is: unweighted CRPS on differences with an origin-blocked interval
    (⭐ *the gate*), the ``Gneiting2011`` threshold decomposition, the variogram score, and the
    firing-union difference as a paired contrast -- plus the non-randomised PIT for counts and the
    Dawid-Sebastiani score.

    M100 contains AP, pooled obs/pred, and two isotonic concordance crossings. ⇒ **Not one member
    of the battery has a pre-change value here.** ⚠️ §2.6 is marked ⬜ unsettled, so the battery is
    a default rather than a decision -- which makes this a reason to settle it before running, not
    a reason the baseline is excused.
    """
    out = (_DOSSIER / "results/baseline_scorecard.out").read_text()
    assert "CRPS" in out or "crps" in out


# ── SOFT S1 · power: the baseline can only see large changes ─────────────────────────────────


@pytest.mark.xfail(strict=True, reason="SOFT: MDE 15.9% at 0.70 and 46.2% at 0.80, 5 v 5")
def test_the_baseline_can_resolve_a_plausible_sized_improvement() -> None:
    """From M100's own across-arm spreads, a 5-arm vs 5-arm contrast:

        mean_draw @0.70   MDE 1.373x = 15.9% of baseline   (80% power: 1.726x = 20.1%)
        gate      @0.70   MDE 0.578x =  9.2%               (80% power: 0.727x = 11.5%)
        mean_draw @0.80   MDE 12.350x = 46.2%              (80% power: 15.531x = 58.1%)

    ⇒ The 0.80 target cannot adjudicate anything below a 46% change and should not be used as a
    comparator. ⚠️ And these use the across-arm sd, which H3 shows is 1.6x-3.0x too small, so the
    true MDE is LARGER than printed here.
    """
    pytest.fail("no power statement is recorded in the dossier")


# ── SOFT S2 · axis 2's population and index are both abolished by v1 ────────────────────────


@pytest.mark.xfail(strict=True, reason="SOFT: the open set is defined by the emission v1 replaces")
def test_the_calibration_figure_has_a_successor_it_can_be_compared_with() -> None:
    """Axis 2's population is ``(draws > 0).any(axis=1)`` over 16 NB draws, and its index is their
    mean. v1 emits the ``k``-th order statistic of ``n_t`` draws from an empirical pool, so there
    are no 16 NB draws to be open or closed; and §2.4's live options include *"draw `n_t` from a
    corrected rate"* and *"recalibrate the gate"*, both of which change WHICH cells fire.

    ⇒ ``1.5048`` is a figure about a population and an index that the intervention removes. ⛔ Not
    a defect in the number -- a limit on what it can be a control FOR.
    """
    pytest.fail("no emission-invariant calibration axis exists in the baseline")


# ── SOFT S3 · h1 only, against an intervention that lives in the rollout ────────────────────


@pytest.mark.xfail(strict=True, reason="SOFT: baseline is h1; v1 acts at h1..h36")
def test_the_baseline_covers_the_horizons_the_intervention_acts_on() -> None:
    """``config_initializer.py`` (C-105) records that autoregressive inference feeds the
    regression output back as input, which is why the empirical draw has to happen INSIDE the
    rollout. ⇒ v1 acts at every horizon, and compounding is the mechanism by which a magnitude
    change becomes a forecast change. All four of M100's axes are h1.
    """
    out = (_DOSSIER / "results/baseline_scorecard.out").read_text()
    assert "h36" in out or "horizon" in out.lower()


# ── SOFT S4 · no pre-committed decision rule ────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="SOFT: no success/failure threshold is pre-registered")
def test_a_decision_rule_is_pre_registered_before_the_experiment_runs() -> None:
    """A baseline with no pre-committed threshold is a measurement, not a control: after the run,
    any observed change can be called a win or noise. §2.6 ends *"Any promotion claim deferred
    until the amendment question is answered separately"*, so the gap is known and open.

    ⛔ The repo's own history is the argument: three claims were withdrawn in one week because the
    criterion was chosen after seeing the readout (``P-66``, ``M96``'s hump, ``M98``'s factor).
    """
    text = (_DOSSIER / "00_README.md").read_text() + (
        _DOSSIER / "results/baseline_scorecard.out"
    ).read_text()
    assert "decision rule" in text.lower() or "counts as success" in text.lower()


# ── OBSERVATION · the gate slide, and the record names it as a pattern of mine ──────────────


@pytest.mark.xfail(strict=True, reason="OBSERVATION: both M100 headlines are about the gate")
def test_the_headline_findings_are_about_the_body_not_the_gate() -> None:
    """``reports/2026-11-04_rank_coupled_body_programme.md`` F6 tabulates errors from one session,
    first row: *"slid to the **gate** rather than the body, five times -- caught by Simon, each
    time"*. Memory carries the same rule (``feedback_the_body_is_the_target_not_the_gate``).

    M100's two headline findings are *the gate ranks magnitude better on 10 of 10 arms* and *the
    architecture decides nothing*. The first is about the gate. ⇒ Sixth instance, and it is the
    mechanism behind H1: the baseline measured what was cheap to index rather than what v1 uses.
    """
    assert _PROGRAMME.exists()
    pytest.fail("the body's own ordering is absent from the baseline")
