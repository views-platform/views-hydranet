"""`train()`-level wiring for #311's input noise, through the REAL training loop.

Added after the 2026-09-04 independent mutation audit. The augmentation's arguments are assembled
in `train()`, and **every mutation there survived**: passing `input_noise_channels=None` made the
whole feature a silent no-op in production, with 1973 tests green — the C-324 inert-knob signature;
hardcoding the segment to 36, or reading it via `config.get("time_steps", 36)`, both passed, while
the block comment argues at length that it must be read explicitly and fail loud rather than fall
back to a shadow default (C-85). Prose without a guard is C-303, this register's most habitual
defect at ten occurrences.

These tests drive the production `training_loop`, so they see what production actually assembles.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import views_hydranet.train.training_engine as te  # noqa: E402
from views_hydranet.train.training_engine import make, training_loop  # noqa: E402

from .conftest import loop_config, loop_handler  # noqa: E402


def _run(monkeypatch, **overrides):
    """Run the real loop, recording every argument the call site passes to the noise helper."""
    calls = []
    real = te._apply_input_noise

    def _rec(dyn_input, keep, dropout, channels):
        calls.append({"dropout": dropout, "channels": list(channels), "width": dyn_input.shape[1]})
        return real(dyn_input, keep, dropout, channels)

    monkeypatch.setattr(te, "_apply_input_noise", _rec)

    seen_segments = []
    real_seq = te._process_sequence

    def _seq(*a, **kw):
        seen_segments.append(kw.get("input_noise_segment"))
        return real_seq(*a, **kw)

    monkeypatch.setattr(te, "_process_sequence", _seq)

    cfg = loop_config(**overrides)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)
    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    return cfg, calls, seen_segments


def test_the_segment_comes_from_time_steps_and_is_not_hardcoded(monkeypatch):
    """Survivors TR-02/TR-03. `time_steps` is set to a value no literal would guess, so a hardcoded
    36 — or a `config.get("time_steps", 36)` shadow default — is visible."""
    cfg, _, segments = _run(monkeypatch, input_noise_dropout=0.5, time_steps=7)
    assert cfg["time_steps"] == 7
    assert segments and all(s == 7 for s in segments), f"segments were {set(segments)}, expected 7"


def test_the_configured_dropout_reaches_the_loop(monkeypatch):
    """Survivors TR-04/TR-05: the rate could be replaced with None, or read from a wrong key."""
    _, calls, _ = _run(monkeypatch, input_noise_dropout=0.31)
    assert calls, "the noise helper was never reached from the real training loop"
    assert all(c["dropout"] == pytest.approx(0.31) for c in calls)


def test_the_channel_list_is_assembled_and_not_passed_as_None(monkeypatch):
    """Survivor TR-01 — the worst of them. `input_noise_channels=None` collapses to `or []` in
    `_process_sequence`, the helper returns immediately on its empty-channels guard, and the entire
    augmentation becomes a no-op for every real run while the whole suite stays green."""
    cfg, calls, _ = _run(monkeypatch, input_noise_dropout=0.5)
    expected = list(range(len(cfg["features"])))
    assert calls
    assert all(c["channels"] == expected for c in calls), (
        f"channels were {calls[0]['channels']}, expected {expected}; an empty list means the "
        "augmentation is inert in production"
    )


def test_with_the_noise_OFF_the_loop_never_reaches_the_helper(monkeypatch):
    """Byte-identity at the production layer, not just in `_process_sequence`."""
    _, calls, segments = _run(monkeypatch)
    assert calls == []
    assert all(s is None for s in segments), "a segment was demanded while the noise was off"
