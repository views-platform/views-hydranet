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
    """Run the real loop, recording every argument the call site passes to the noise helper.

    `time_steps` defaults to 2 here, not `loop_config`'s 1: the segment start is left clean, so at
    `time_steps=1` every step is a start and the augmentation can never apply. A config validator
    now rejects that combination outright; these tests need a horizon where it CAN apply.
    """
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

    drop = overrides.pop("_drop_keys", ())
    if overrides.get("input_noise_dropout") is not None:
        overrides.setdefault("time_steps", 2)
        overrides.setdefault("steps", [1, 2])
    cfg = loop_config(**overrides)
    for k in drop:
        cfg.pop(k, None)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)
    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    return cfg, calls, seen_segments


def test_the_segment_comes_from_time_steps_and_is_not_hardcoded(monkeypatch):
    """Survivors TR-02/TR-03. `time_steps` is set to a value no literal would guess, so a hardcoded
    36 — or a `config.get("time_steps", 36)` shadow default — is visible."""
    # `steps` is set alongside `time_steps`: HydraNetConfig enforces time_steps == len(steps), and
    # the second audit noted the earlier fixture pinned the segment against a config shape
    # production could not produce. 3 rather than 7 because the fixture's volume is 5 months long.
    cfg, _, segments = _run(monkeypatch, input_noise_dropout=0.5, time_steps=3, steps=[1, 2, 3])
    assert cfg["time_steps"] == 3 == len(cfg["steps"])
    # `None` entries are the C-184 BN-recalibration pass, which is deliberately un-noised (#311);
    # the training calls are the ones that must carry the config's value.
    training = [s for s in segments if s is not None]
    assert training, "no training call carried a segment"
    assert all(s == 3 for s in training), f"segments were {set(training)}, expected 3"


def test_the_configured_dropout_reaches_the_loop(monkeypatch):
    """Survivors TR-04/TR-05: the rate could be replaced with None, or read from a wrong key."""
    _, calls, _ = _run(monkeypatch, input_noise_dropout=0.31)
    assert calls, "the noise helper was never reached from the real training loop"
    assert all(c["dropout"] == pytest.approx(0.31) for c in calls)


def test_the_channel_list_is_assembled_and_not_passed_as_None(monkeypatch):
    """Survivor TR-01 — the worst of them. `input_noise_channels=None` collapses to `or []` in
    `_process_sequence`, the helper returns immediately on its empty-channels guard, and the entire
    augmentation becomes a no-op for every real run while the whole suite stays green."""
    _, calls, _ = _run(monkeypatch, input_noise_dropout=0.5)
    assert calls, "the noise helper was never reached"
    assert calls[0]["channels"], "an empty channel list means the augmentation is inert"


def test_the_seam_CALLS_noisable_channels_rather_than_rebuilding_the_list(monkeypatch):
    """Survivor TR-07 — the sharpest finding of the second audit.

    The earlier version of the test above computed its expectation as
    `list(range(len(cfg["features"])))` — *literally the mutant's body* — and `loop_config`
    declares no statics, so expectation and mutation agreed by construction. `_noisable_channels`
    had three
    good unit tests including the synthetic-statics case, and the only place it is CALLED could
    stop calling it with nothing failing: the chain "statics exist → excluded → exclusion reaches
    the model" was guarded at both ends and open in the middle.

    Patching the helper to return a sentinel tests the *call*, not the value — so it discriminates
    without needing a static channel in the fixture's data, and no expectation is derived from the
    thing under test.
    """
    sentinel = [1]
    monkeypatch.setattr(te, "_noisable_channels", lambda cfg: list(sentinel))
    _, calls, _ = _run(monkeypatch, input_noise_dropout=0.5)
    assert calls, "the noise helper was never reached"
    assert all(c["channels"] == sentinel for c in calls), (
        f"channels were {calls[0]['channels']}, not the sentinel {sentinel} — the seam rebuilt "
        "the list instead of calling _noisable_channels; the exclusion never reaches the model"
    )


def test_a_missing_time_steps_fails_loud_when_the_noise_is_on(monkeypatch):
    """Survivor TR-03. The docstring of the test above claimed to close
    `config.get("time_steps", 36)`; it did not — setting `time_steps=7` catches a *hardcoded* 36
    but not a `.get` fallback, because the key is present in the fixture. Claiming more than the
    assertion delivers is C-303, and this one was written inside the fix for C-303."""
    with pytest.raises(KeyError):
        _run(monkeypatch, input_noise_dropout=0.5, _drop_keys=("time_steps",))


def test_train_does_not_treat_a_configured_zero_as_OFF(monkeypatch):
    """Survivor EX-04. `config.get("input_noise_dropout") or None` in `train()` turns a configured
    0.0 into off. The seam has this guard (CS-01, caught) and `train()` did not.

    Reachability is honest: a *validated* config cannot carry 0.0 because the field is `gt=0.0`, so
    this is only live through a raw dict — which is exactly what `loop_config` is, and what several
    tools build."""
    _, calls, _ = _run(monkeypatch, input_noise_dropout=0.0)
    assert calls, "a configured dropout of 0.0 was silently treated as off"
    assert all(c["dropout"] == 0.0 for c in calls)


def test_with_the_noise_OFF_the_loop_never_reaches_the_helper(monkeypatch):
    """Byte-identity at the production layer, not just in `_process_sequence`."""
    _, calls, segments = _run(monkeypatch)
    assert calls == []
    assert all(s is None for s in segments), "a segment was demanded while the noise was off"


def test_EVERY_non_static_channel_is_noised_not_just_a_prefix(monkeypatch):
    """Survivor TR-11: `_noisable_channels(config)[:1]` noises only the first target and survives.
    The sentinel test above pins that the helper is CALLED; a 1-element sentinel happens to agree
    with a `[:1]` slice. Nothing pinned that the full list arrives."""
    cfg, calls, _ = _run(monkeypatch, input_noise_dropout=0.5)
    assert calls, "the noise helper was never reached"
    n_dynamic = len(cfg["features"])
    assert n_dynamic > 1, "this test cannot discriminate with a single dynamic channel"
    assert all(len(c["channels"]) == n_dynamic for c in calls), (
        f"only {len(calls[0]['channels'])} of {n_dynamic} dynamic channels were noised"
    )


def test_process_sequence_defaults_to_noise_OFF(monkeypatch):
    """Survivor PS-02: defaulting the signature to 0.204/36 makes every OTHER caller of
    `_process_sequence` — pushforward, BPTT, the gradient-health tests — train under noise, and
    they all still pass because each compares two runs that are both noised under a fixed seed.
    'Off unless asked' has to be pinned at the function's own boundary."""
    import inspect

    from views_hydranet.train.training_engine import _process_sequence

    sig = inspect.signature(_process_sequence).parameters
    assert sig["input_noise_dropout"].default is None
    assert sig["input_noise_segment"].default is None
    assert sig["input_noise_channels"].default is None


def test_the_diagnostic_biopsy_is_NEVER_noised(monkeypatch):
    """Survivor BP-01. `docs/CICs/TrainingEngine.md` states as an invariant that the Stage-5
    diagnostic biopsy is never noised — it is a *clean-performance* probe. Noising it left all
    tests green: the invariant was prose.

    ⚠️ The anti-vacuity guard here was itself wrong in an earlier version, and the review caught
    it. It proved "the biopsy ran" by observing a `_attach_static_channels` call under `no_grad` —
    but `_recalibrate_bn` also runs under `no_grad`, so the guard stayed satisfied by BN-recal and
    would have gone vacuous with respect to the biopsy the moment the biopsy stopped running. That
    is C-329 (a test that cannot fail) inside the fix for it.

    The discriminator is the model's own mode: the Stage-5 biopsy runs under `model.eval()`, while
    the BN-recalibration loop runs in `train()` mode precisely so it can accumulate statistics. So
    `(no_grad, eval)` identifies the biopsy uniquely, and `(no_grad, train)` is BN-recal.
    """
    noise_grad_states: list[bool] = []
    attach_states: list[tuple[bool, bool]] = []
    real_noise = te._apply_input_noise
    real_attach = te._attach_static_channels

    cfg = loop_config(input_noise_dropout=0.5, time_steps=2, steps=[1, 2])
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)

    def _rec_noise(dyn_input, keep, dropout, channels):
        noise_grad_states.append(torch.is_grad_enabled())
        return real_noise(dyn_input, keep, dropout, channels)

    def _rec_attach(dyn_input, t0_step, idx):
        attach_states.append((torch.is_grad_enabled(), model.training))
        return real_attach(dyn_input, t0_step, idx)

    monkeypatch.setattr(te, "_apply_input_noise", _rec_noise)
    monkeypatch.setattr(te, "_attach_static_channels", _rec_attach)
    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)

    biopsy = [s for s in attach_states if s == (False, False)]
    assert biopsy, (
        "no forward ran under (no_grad, eval) — the Stage-5 biopsy did not execute, so this test "
        "would be vacuous. It must not be satisfied by the BN-recal pass, which runs (no_grad, "
        "train)"
    )
    assert noise_grad_states, "the noise helper never ran at all"
    assert all(noise_grad_states), (
        "input noise was applied under no_grad — it reached the diagnostic biopsy or a "
        "recalibration pass, both of which must stay clean"
    )
