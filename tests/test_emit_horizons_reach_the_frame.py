"""ADR-078 §2.2: the horizon selection reaches the emitted frame — and nothing else.

⛔ **The test that matters here is the bit-identity one.** A selection is only safe if it is a
pure projection of a full run: the rows that survive must be the same rows, bit for bit. If
someone ever "optimises" the field by shortening the rollout, that assertion is what fails —
documentation is not, and ADR-078 §4 names exactly that misreading as a cost with no other
defence.

⚠️ **What a mocked model can and cannot establish.** The rollout runs inside the model, which is
mocked here, so these tests cannot prove that a real rollout computed every month. What they do
prove is that the selection is applied *below* the rollout: the volume handed to the assembler
still carries every horizon on a subset run, and the rows that are written are untouched. A real
run's byte-identity is S7's measurement, on hardware, and is not claimed here.

The configuration, grid fixture and mocked-model harness are the wiring story's — same seam, same
run shape, so they are imported rather than restated.
"""

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("views_pipeline_core")

from unittest.mock import MagicMock  # noqa: E402

import torch  # noqa: E402

from tests.test_emission_manifest_wiring import CFG, TARGETS, S, _df  # noqa: E402
from views_hydranet.utils.feature_scaler import FeatureScaler  # noqa: E402
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator  # noqa: E402
from views_hydranet.utils.prediction_frame_assembler import (  # noqa: E402
    PredictionFrameAssembler,
)
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402


@pytest.fixture(scope="module")
def env():
    handler = VolumeHandler.from_df(_df(), CFG)
    scaler = FeatureScaler(CFG)
    scaler._is_fitted = True
    return handler, scaler, MagicMock(spec=torch.nn.Module)


def _run(env, origins, emit_horizons, assembler_patch=None):
    """Run the real orchestrator with a fixed posterior, and return (frames, orchestrator).

    The posterior is seeded and identical between calls, so any difference between a full run and
    a subset run is the selection's doing and nothing else's.

    ⭐ The selection is passed **through the configuration**, never by setting the attribute. An
    earlier version of this file set `orchestrator.emit_horizons` directly, and a mutant that
    replaced the configuration read with a hardcoded `None` left all seven tests green — the
    field would have validated, reached nothing, and been tested as if it worked. That is
    C-324's signature exactly, and driving every test through the config is what closes it.
    """
    handler, scaler, model = env
    rng = np.random.default_rng(20261102)
    posterior = rng.random((CFG["time_steps"], 2, 2, 6, S))

    orchestrator = InferenceOrchestrator(
        {**CFG, "emit_horizons": emit_horizons}, model, torch.device("cpu")
    )
    with patch("views_hydranet.utils.inference_orchestrator.HydraNetInference") as mock_cls:
        mock_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
        if assembler_patch is not None:
            with patch.object(PredictionFrameAssembler, "assemble_evaluation", assembler_patch):
                frames = orchestrator.generate_prediction_frames(
                    handler, scaler, origins=origins, all_targets=TARGETS
                )
        else:
            frames = orchestrator.generate_prediction_frames(
                handler, scaler, origins=origins, all_targets=TARGETS
            )
    return frames, orchestrator


def _origins(env):
    handler, _, _ = env
    return [handler.shape[0] - 4, handler.shape[0] - 3]


# ─── The one that matters ────────────────────────────────────────────────────


def test_a_subset_is_bit_identical_to_the_corresponding_rows_of_a_full_run(env):
    """⭐⭐ The rows a subset writes are the full run's rows, unchanged, bit for bit.

    Not a convenience check. This is the assertion that a selection is a projection rather than a
    different computation — and the only one that would notice if the selection ever started
    reaching the rollout.
    """
    origins = _origins(env)
    full, _ = _run(env, origins, None)
    subset, _ = _run(env, origins, [1])

    assert len(full) == len(subset), "a selection must not change how many origins are emitted"

    for full_dict, subset_dict in zip(full, subset):
        kept_months = {int(m) for m in np.asarray(subset_dict[TARGETS[0]].index.time)}
        assert len(kept_months) == 1, "the fixture must emit a strict subset of the horizons"

        for target in TARGETS:
            f, s = full_dict[target], subset_dict[target]
            keep = np.isin(np.asarray(f.index.time), sorted(kept_months))

            assert np.array_equal(np.asarray(f.index.time)[keep], np.asarray(s.index.time))
            assert np.array_equal(np.asarray(f.index.unit)[keep], np.asarray(s.index.unit))
            # `equal_nan=False` is deliberate: a NaN that appeared only on one side would be a
            # real difference, and `array_equal`'s default would hide it as "both NaN".
            assert np.array_equal(np.asarray(f.values)[keep], np.asarray(s.values)), (
                f"{target}: a selected horizon's values differ from the full run's — the "
                f"selection is not a pure projection"
            )


def test_the_volume_handed_to_the_assembler_still_carries_every_horizon(env):
    """⭐ The selection is applied below the rollout, not above it (ADR-078 §2.2).

    If an optimisation ever truncated the rollout to the requested horizons, the frames would
    still look right and this is the assertion that would not.
    """
    seen = []
    original = PredictionFrameAssembler.assemble_evaluation

    def spy(self, *, signal, history, start_idx, all_targets, emit_horizons=None):
        seen.append(signal.data.shape[signal.get_axis_idx("T")])
        return original(
            self,
            signal=signal,
            history=history,
            start_idx=start_idx,
            all_targets=all_targets,
            emit_horizons=emit_horizons,
        )

    _run(env, _origins(env), [1], assembler_patch=spy)

    assert seen, "the spy never ran, so this test proves nothing"
    assert all(depth == CFG["time_steps"] for depth in seen), (
        f"the assembler was handed {seen} horizons with a 1-horizon selection; the rollout was "
        f"shortened, which ADR-078 §2.2 forbids"
    )


# ─── Only which artifacts exist changes ──────────────────────────────────────


def test_only_the_selected_horizons_appear(env):
    """A two-horizon selection writes exactly those two horizons' months."""
    handler, _, _ = env
    origins = _origins(env)
    frames, orch = _run(env, origins, [1, 2])

    for frame_dict, recorded in zip(frames, orch.emitted_origins):
        months = sorted({int(m) for m in np.asarray(frame_dict[TARGETS[0]].index.time)})
        assert months == list(recorded.horizon_months[:2])


def test_the_default_emits_every_horizon(env):
    """*Green:* a run that sets nothing is unchanged — the whole-window row count is unaffected."""
    origins = _origins(env)
    frames, orch = _run(env, origins, None)

    for frame_dict, recorded in zip(frames, orch.emitted_origins):
        assert recorded.emitted_horizons == tuple(range(1, CFG["time_steps"] + 1))
        expected_rows = recorded.rows_per_horizon * CFG["time_steps"]
        assert int(frame_dict[TARGETS[0]].n_rows) == expected_rows


# ─── The manifest declares the subset ────────────────────────────────────────


def test_the_record_declares_the_subset_and_keeps_the_full_horizon_map(env):
    """ADR-074: a subset must be declared, or it cannot be told from a partial failure.

    The horizon-to-month map stays complete because every horizon in it was computed; only the
    emitted set shrinks.
    """
    _, orch = _run(env, _origins(env), [2])

    for recorded in orch.emitted_origins:
        assert recorded.emitted_horizons == (2,)
        assert len(recorded.horizon_months) == CFG["time_steps"], (
            "the horizon map was filtered by the selection; it describes what was computed"
        )


def test_the_recorded_subset_is_measured_from_the_frame_not_from_the_config(env):
    """⭐⭐ The record follows what was written, even when that disagrees with what was asked.

    The assembler is made to ignore the selection and emit everything, while the configuration
    still asks for one horizon. A record assembled from the configuration would report `(2,)` —
    the request — and would be wrong about the artifact beside it. ADR-074 §2.3 exists to keep
    exactly this comparison possible, and a manifest that re-read the request could never fail
    it.
    """
    original = PredictionFrameAssembler.assemble_evaluation

    def ignore_the_selection(self, *, signal, history, start_idx, all_targets, emit_horizons=None):
        return original(
            self,
            signal=signal,
            history=history,
            start_idx=start_idx,
            all_targets=all_targets,
            emit_horizons=None,
        )

    _, orch = _run(env, _origins(env), [2], assembler_patch=ignore_the_selection)

    for recorded in orch.emitted_origins:
        assert recorded.emitted_horizons == tuple(range(1, CFG["time_steps"] + 1)), (
            "the record reported the requested subset while the run emitted every horizon — it "
            "is reading the configuration, not the artifact"
        )


def test_the_selection_travels_from_the_config_and_not_from_an_attribute(env):
    """⭐⭐ The configuration key is what a pod run can set, so it is what must be read.

    `pod_run_model.sh` runs `main.py` from a model configuration and has no hook for setting an
    orchestrator attribute, so a selection reachable only from a driver script would be unusable
    exactly where output volume costs. This test asserts the read, not the plumbing below it.
    """
    handler, _, model = env
    orch = InferenceOrchestrator({**CFG, "emit_horizons": [1, 3]}, model, torch.device("cpu"))
    assert orch.emit_horizons == [1, 3]

    bare = InferenceOrchestrator(CFG, model, torch.device("cpu"))
    assert bare.emit_horizons is None, "a config without the key must emit everything"


# ─── Refusals ────────────────────────────────────────────────────────────────


def test_a_horizon_the_window_does_not_contain_is_refused(env):
    """*Red:* refusing beats writing fewer artifacts than requested.

    On disk the two outcomes are the same thing: a directory that is not there.
    """
    handler, scaler, model = env
    rng = np.random.default_rng(1)
    posterior = rng.random((CFG["time_steps"], 2, 2, 6, S))
    orchestrator = InferenceOrchestrator(
        {**CFG, "emit_horizons": [CFG["time_steps"] + 5]}, model, torch.device("cpu")
    )

    with patch("views_hydranet.utils.inference_orchestrator.HydraNetInference") as mock_cls:
        mock_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
        with pytest.raises(ValueError, match="does not contain"):
            orchestrator.generate_prediction_frames(
                handler, scaler, origins=[handler.shape[0] - 4], all_targets=TARGETS
            )


# ─── The STREAMING loop — the path production actually takes ─────────────────


def _run_streaming(env, origins, emit_horizons):
    """Drive `generate_prediction_frames_streaming` and return (sunk frames, orchestrator).

    ⛔⛔ This exists because every other test in this file drives `generate_prediction_frames` —
    the BATCH loop — and pipeline-core only calls that on the DataFrame branch, which no HydraNet
    takes. A reviewer deleted both `emit_horizons=` and the recorder call from the STREAMING loop
    and 85 tests still passed. The feature was fully guarded on the dead copy and unguarded on the
    live one, which is C-324's signature a third time.
    """
    handler, scaler, model = env
    rng = np.random.default_rng(20261102)
    posterior = rng.random((CFG["time_steps"], 2, 2, 6, S))

    orchestrator = InferenceOrchestrator(
        {**CFG, "emit_horizons": emit_horizons}, model, torch.device("cpu")
    )
    sunk = []
    with patch("views_hydranet.utils.inference_orchestrator.HydraNetInference") as mock_cls:
        mock_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
        orchestrator.generate_prediction_frames_streaming(
            handler,
            scaler,
            origins=origins,
            all_targets=TARGETS,
            origin_sink=lambda i, pf: sunk.append((i, {t: pf[t] for t in TARGETS})),
        )
    return sunk, orchestrator


def test_the_streaming_loop_honours_the_horizon_selection(env):
    """⭐⭐ The live path, not the batch copy. Deleting the kwarg there must fail a test."""
    origins = _origins(env)
    sunk, _ = _run_streaming(env, origins, [1])

    assert len(sunk) == len(origins), "the sink was not called once per origin"
    for _, pf_dict in sunk:
        months = {int(m) for m in np.asarray(pf_dict[TARGETS[0]].index.time)}
        assert len(months) == 1, (
            f"the streaming loop emitted {len(months)} horizons for a 1-horizon selection — the "
            f"selection does not reach the path production takes"
        )


def test_the_streaming_loop_records_what_it_emitted(env):
    """⭐⭐ The recorder on the live path. Deleting the call there must fail a test.

    Checked before the sink frees anything, which is why the recorder sits where it does: the
    pipeline's real sink pops frames out of `pf_dict`, so a record taken afterwards would read an
    empty dict.
    """
    origins = _origins(env)
    _, orch = _run_streaming(env, origins, None)

    assert len(orch.emitted_origins) == len(origins), (
        "the streaming loop recorded no origins — the manifest would show `origins: []` beside "
        "real prediction directories, under an INFO line saying it was written"
    )
    assert [o.origin_index for o in orch.emitted_origins] == origins
    for record in orch.emitted_origins:
        assert record.emitted_horizons == tuple(range(1, CFG["time_steps"] + 1))
        assert record.rows_per_horizon > 0


def test_the_streaming_and_batch_loops_agree(env):
    """The two loops are near-duplicates, so they must not be able to drift apart silently.

    ⚠️ This is a guard, not an endorsement: the duplication is the underlying problem and is noted
    rather than fixed here. Until one of them goes, this is what notices if only one is updated.
    """
    origins = _origins(env)
    batch, batch_orch = _run(env, origins, [1])
    sunk, stream_orch = _run_streaming(env, origins, [1])

    assert len(batch) == len(sunk)
    for (_, stream_pf), batch_pf in zip(sunk, batch):
        for target in TARGETS:
            assert np.array_equal(
                np.asarray(batch_pf[target].values), np.asarray(stream_pf[target].values)
            ), f"{target}: the two loops emit different values for the same selection"

    assert [o.emitted_horizons for o in batch_orch.emitted_origins] == [
        o.emitted_horizons for o in stream_orch.emitted_origins
    ]
