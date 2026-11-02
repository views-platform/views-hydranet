"""ADR-074: the three refusals inside `_record_emitted_origin`, and the reset above it.

⛔⛔ **Every guard in this file was DECORATIVE until it was written.** A mutation audit of commit
`3fad6e4` deleted or inverted each of the recorder's three refusal branches, widened its exception
tuple, and removed the streaming loop's origin reset — and the full suite stayed green at *2504
passed* in every case.

The reason is structural and worth stating, because it will recur. `_record_emitted_origin` is
documented to **skip rather than raise**: it must never abort a finished multi-hour run over its
own bookkeeping. So the observable effect of *any* defect in it is an **empty `emitted_origins`** —
and every test that checked a record's contents did so by iterating that list, which passes
vacuously when it is empty. The contents were pinned thoroughly; the existence of contents was not.

⇒ These tests drive the branches directly, with the two helpers stubbed, because the refusals fire
on input shapes the 2×2 fixture grid cannot produce: a non-uniform row count per horizon, and a
horizon that emitted nothing. Stubbing is what makes those reachable at all — and the branch under
test is the real one, called on a real instance.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("views_pipeline_core")

import torch  # noqa: E402

from tests.test_emission_manifest_wiring import CFG, TARGETS, S, _df  # noqa: E402
from views_hydranet.utils.feature_scaler import FeatureScaler  # noqa: E402
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator  # noqa: E402
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402

MONTHS = (458, 459, 460)


def _orchestrator(emit_horizons):
    """A real orchestrator, configured through the config dict as a pod run would."""
    return InferenceOrchestrator(
        {**CFG, "emit_horizons": emit_horizons},
        MagicMock(spec=torch.nn.Module),
        torch.device("cpu"),
    )


def _record(monkeypatch, *, emit_horizons, per_horizon, months=MONTHS):
    """Call the real `_record_emitted_origin` with both helpers stubbed, and return the instance.

    The window and the frame are never touched: the stubs stand exactly where the helpers that read
    them would, so what runs is the branch logic under test and nothing else.
    """
    orch = _orchestrator(emit_horizons)
    monkeypatch.setattr(InferenceOrchestrator, "_months_of", staticmethod(lambda _w: months))
    monkeypatch.setattr(
        InferenceOrchestrator, "_rows_per_horizon_of", staticmethod(lambda *_a: per_horizon)
    )
    orch._record_emitted_origin(0, 457, object(), {"lr_ged_sb": object()})
    return orch


# ─── Refusal 1: a horizon that emitted nothing, with no selection in force ───


def test_a_short_emitted_set_with_no_selection_is_refused(monkeypatch, caplog):
    """⛔ A gap month must not be recorded as a deliberate subset.

    `from_df` builds a dense month range and zero-fills a missing month, so that step's validity
    mask is all-False and the horizon emits no rows — it is simply absent from `per_horizon`. With
    no selection in force, recording the short set would make a **partial failure**
    indistinguishable from a **declared subset**, which is the single confusion ADR-074 exists to
    prevent.

    ⚠️ Deleting this branch left the suite green: no fixture anywhere produces a gap.
    """
    with caplog.at_level("ERROR"):
        orch = _record(monkeypatch, emit_horizons=None, per_horizon={1: 4, 3: 4})

    assert orch.emitted_origins == [], (
        "an origin that emitted 2 of 3 horizons with no selection was recorded anyway; a reader "
        "cannot tell that from a subset the run asked for"
    )
    assert "the run selected none" in caplog.text


def test_a_declared_subset_is_still_recorded(monkeypatch):
    """*Green, and it is the half that catches the INVERSION.*

    Flipping the condition above to `is not None` makes every legitimate subset run record nothing
    — `origins: []` beside real `origin_*` directories, under an INFO line saying the manifest was
    written. Without this test that mutation is invisible, because the refusal test alone passes
    either way.
    """
    orch = _record(monkeypatch, emit_horizons=[1, 3], per_horizon={1: 4, 3: 4})

    assert len(orch.emitted_origins) == 1, "a declared subset was refused"
    assert orch.emitted_origins[0].emitted_horizons == (1, 3)
    assert orch.emitted_origins[0].horizon_months == MONTHS, (
        "the horizon map was filtered by the selection; it describes what was COMPUTED"
    )


# ─── Refusal 2: a per-horizon row count that is not one number ───────────────


def test_a_non_uniform_row_count_per_horizon_is_refused(monkeypatch, caplog):
    """⛔ `rows_per_horizon` is one integer, so it must be true of every horizon.

    The field it would otherwise hold is `counts.pop()` — an **arbitrary** horizon's count,
    recorded as the exact per-horizon count of a run where no horizon had it. The branch this
    replaced tested divisibility, which 9/8/7 cells over three horizons passes while recording 8.

    ⚠️ Unreachable through the fixture grid, which is uniform at every horizon by construction —
    which is why deleting the check cost nothing.
    """
    with caplog.at_level("ERROR"):
        orch = _record(monkeypatch, emit_horizons=None, per_horizon={1: 4, 2: 3, 3: 4})

    assert orch.emitted_origins == [], (
        "a run emitting 4/3/4 rows per horizon recorded a single exact count, which no horizon had"
    )
    assert "different number of rows per horizon" in caplog.text


def test_a_uniform_row_count_is_recorded_as_that_number(monkeypatch):
    """*Green:* the uniform case still records, and records the count itself."""
    orch = _record(monkeypatch, emit_horizons=None, per_horizon={1: 4, 2: 4, 3: 4})

    assert len(orch.emitted_origins) == 1
    assert orch.emitted_origins[0].rows_per_horizon == 4


# ─── Refusal 3: the exception tuple is narrow on purpose ─────────────────────


def test_a_programming_error_in_the_recorder_propagates(monkeypatch):
    """⭐ The `except` is three named exceptions, NOT `Exception`, and that is load-bearing.

    `(ValueError, KeyError, StopIteration)` are the failures the two helpers raise for data they
    cannot describe — those are skipped, by design. An `AttributeError` is a defect in this method,
    and absorbing it would report a bug in the same shape as unrecordable data: the run completes,
    the manifest is short, and nothing says which of the two happened.

    ⚠️ Widening the tuple to `Exception` left the suite green.
    """
    orch = _orchestrator(None)

    def boom(_w):
        raise AttributeError("a frame without an index")

    monkeypatch.setattr(InferenceOrchestrator, "_months_of", staticmethod(boom))

    with pytest.raises(AttributeError, match="a frame without an index"):
        orch._record_emitted_origin(0, 457, object(), {"lr_ged_sb": object()})


def test_unrecordable_data_is_skipped_rather_than_raised(monkeypatch, caplog):
    """*Green, and the other side of the same coin.*

    A `ValueError` from the month reader must NOT escape: raising here would abort a finished run
    between assembling an origin's frames and handing them to the sink, leaving the earlier origins
    on disk and this one written nowhere.
    """
    orch = _orchestrator(None)

    def refuse(_w):
        raise ValueError("not constant across the spatial axes")

    monkeypatch.setattr(InferenceOrchestrator, "_months_of", staticmethod(refuse))

    with caplog.at_level("ERROR"):
        orch._record_emitted_origin(0, 457, object(), {"lr_ged_sb": object()})

    assert orch.emitted_origins == []
    assert "could not be recorded" in caplog.text
    assert "origin_* directories" in caplog.text, (
        "the log must say the gap is detectable only by comparing counts; nothing announces it"
    )


# ─── The reset above the loop — on the LIVE path ─────────────────────────────


@pytest.fixture(scope="module")
def env():
    handler = VolumeHandler.from_df(_df(), CFG)
    scaler = FeatureScaler(CFG)
    scaler._is_fitted = True
    return handler, scaler, MagicMock(spec=torch.nn.Module)


def test_a_second_streaming_run_does_not_inherit_the_first_run_s_origins(env):
    """⛔ The same dead-path/live-path split that this epic was already caught in.

    `test_a_second_run_does_not_inherit_the_first_run_s_origins` drives the BATCH loop, which
    pipeline-core only calls on the DataFrame branch — and every HydraNet declares
    `prediction_format: prediction_frame`, so production always takes the streaming override.
    Deleting the streaming loop's `self.emitted_origins = []` therefore cost nothing: the reset was
    guarded on the copy nobody runs. A reused orchestrator would then list MORE origins than there
    are directories, mixing two runs' records in one manifest.
    """
    handler, scaler, model = env
    orch = InferenceOrchestrator({**CFG, "emit_horizons": None}, model, torch.device("cpu"))
    posterior = np.random.default_rng(20261102).random((CFG["time_steps"], 2, 2, 6, S))
    origins = [handler.shape[0] - 4, handler.shape[0] - 3]

    def stream(which):
        with patch("views_hydranet.utils.inference_orchestrator.HydraNetInference") as mock_cls:
            mock_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
            orch.generate_prediction_frames_streaming(
                handler,
                scaler,
                origins=which,
                all_targets=TARGETS,
                origin_sink=lambda _i, _pf: None,
            )

    stream(origins)
    assert len(orch.emitted_origins) == 2, "the first streaming run recorded nothing"

    stream(origins[:1])
    assert len(orch.emitted_origins) == 1, (
        "origins accumulated across two streaming runs; the manifest would list two runs' records "
        "beside one run's directories"
    )


# ─── `_rows_per_horizon_of`'s own refusal, which nothing reached ─────────────
#
# ⛔⛔ A second audit, of commit `568a749`, deleted this refusal and left the full suite green at
# *2532 passed*. `test_a_horizon_the_window_does_not_contain_is_refused` looks like its cover and
# is not: that one matches `"does not contain"`, which is the **assembler's** refusal, while this
# one says `"are absent from the window it was forecast over"`. Two different guards, one message
# apart.
#
# ⚠️ And the consequence is not merely a worse diagnostic. With the refusal gone, a mutation that
# skips unknown months (`if int(month) in month_to_horizon`) stops being dead code: the counter
# then returns a uniform count over the months it recognised, every refusal above passes, and the
# manifest records a `rows_per_horizon × emitted_horizons` that **undercounts the rows on disk**.
# That is the trusted-and-wrong manifest ADR-074 §4 calls worse than no manifest at all.


def _frame(months):
    """The smallest stand-in `_rows_per_horizon_of` reads: one labelled time column."""
    return SimpleNamespace(index=SimpleNamespace(time=np.asarray(months)))


def test_a_month_the_frame_emitted_but_the_window_never_held_is_refused():
    """⛔ The frame and the window disagree about what this origin is, so neither is the truth."""
    with pytest.raises(ValueError, match="absent from the window it was forecast over"):
        InferenceOrchestrator._rows_per_horizon_of(_frame([458, 458, 999]), MONTHS, 457)


def test_the_counter_numbers_horizons_from_the_window_s_order():
    """*Green, and the other half:* the count is per 1-based horizon, in the window's own order."""
    counts = InferenceOrchestrator._rows_per_horizon_of(
        _frame([460, 460, 458, 459, 459, 459]), MONTHS, 457
    )

    assert counts == {1: 1, 2: 3, 3: 2}, (
        f"counted {counts}; horizon 1 is the window's FIRST month, and each count is the "
        f"number of rows the frame actually carries for that horizon"
    )


# ─── When two refusals could both fire, the log must name the right one ──────


def test_an_origin_that_is_both_ragged_and_short_is_refused_as_ragged(monkeypatch, caplog):
    """⛔ Reordering the two refusal branches left the full suite green.

    The tests above reach exactly one branch each by construction — `{1: 4, 3: 4}` is uniform but
    short, `{1: 4, 2: 3, 3: 4}` is complete but ragged. A gap month in a ragged panel is **both**,
    and nothing pinned which cause gets reported. Swapping the order would then name the missing
    horizon on a run whose real fault was an incomplete panel, sending a reader to the wrong file.

    ⇒ Uniformity is checked first on purpose: a row count that is not uniform means no single
    per-horizon number is true of the origin at all, so the completeness question cannot be
    answered yet.
    """
    with caplog.at_level("ERROR"):
        orch = _record(monkeypatch, emit_horizons=None, per_horizon={1: 4, 3: 3})

    assert orch.emitted_origins == []
    assert "a different number of rows per horizon" in caplog.text, (
        "an origin that is both ragged and short was reported as short; the raggedness is the "
        "finding that makes every other number about it meaningless"
    )
    assert "the run selected none" not in caplog.text, (
        "both refusals reported on one origin, so the log states two causes for one skip"
    )


# ─── The exception tuple's third member ──────────────────────────────────────


def test_an_origin_with_no_frames_is_skipped_rather_than_aborting_the_run(monkeypatch, caplog):
    """⛔ Dropping `StopIteration` from the `except` tuple left the full suite green.

    `next(iter(pf_dict.values()))` raises it on an empty frame dict, and because this method sits
    between assembling an origin and handing it to the sink, an escaping exception would abort a
    finished multi-hour run **before** that origin was persisted — the one thing the docstring says
    it must never do. Unreachable through today's assembler, which is why the arm was credited and
    untested; the arm is cheaper to keep than the run is to lose.
    """
    orch = _orchestrator(None)
    monkeypatch.setattr(InferenceOrchestrator, "_months_of", staticmethod(lambda _w: MONTHS))

    with caplog.at_level("ERROR"):
        orch._record_emitted_origin(0, 457, object(), {})

    assert orch.emitted_origins == []
    assert "could not be recorded" in caplog.text, (
        "an origin with no frames was not reported as unrecordable, so the manifest is short by "
        "one with nothing in the log to say why"
    )
