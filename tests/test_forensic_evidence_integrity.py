"""The training biopsy must be readable, honest, and recoverable.

⛔ WHY THIS FILE EXISTS. The Stage-5 training forensic was the only evidence a 300-lesson run
left about the body's field, and three independent properties made it misleading:

1. the regression rows were labelled ``body E[y]`` while carrying ``log1p(E[y])``, so a reader in
   another session applied ``log1p`` a second time and shrank a true finding by an exponential;
2. every row was min-max stretched INDEPENDENTLY, so "bright body over dark truth" was the
   rendering a sparse panel always produces whatever the magnitudes are — and that reading was
   made and acted on. ⚠️ The fix then over-corrected by one row: the PREDICTED gate was pinned to
   the same fixed ``[0, 1]`` as the binary cls truth, and at this model's gate range (~0.003-0.05)
   that renders the row black and erases exactly the structure the row exists to show. Both
   directions are guarded below;
3. the forecast row was computed in the plot as ``gate × log1p(μ)`` under a docstring calling it
   "the ACTUAL scored forecast", while deployment emits ``log1p(compose_mean(μ, gate))``
   (`hydranet_inference._emit_magnitude`). Different functions of the same inputs; they agree
   only where the gate is 1.

and a fourth made it unrecoverable: the arrays were rendered to PNG and dropped, so any later
question cost a fresh run on rented hardware.

⚠️ A FIFTH, found on the v0.1.4 release surface: once the forecast row started composing, the
block ran real family + composition code with no enclosing ``try``, so a diagnostic could abort a
multi-hour training run — and it is not diagnostics-gated, because ``viz`` is a null object that
is truthy even when visual diagnostics are off.

Each test below fails if its fix is reverted. The composition test is the important one: it binds
to the DATA, not to a label, because a label assertion cannot see a wrong number.
"""

import numpy as np
import pytest

from views_hydranet.train.training_engine import _merge_fed_max
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

T, H, W, C = 6, 8, 8, 3


def _arr(fill: float = 1.0) -> np.ndarray:
    return np.full((T, H, W, C), fill, dtype=np.float32)


@pytest.fixture
def vd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return VisualDiagnostics({"diagnostic_visualizations": True, "run_type": "test"})


# ----------------------------------------------------------------- the arrays survive the plot


def test_the_biopsy_writes_an_npz_beside_the_png(vd, tmp_path):
    vd.biopsy_training_performance(
        _arr(2.0),
        _arr(1.0),
        _arr(1.0),
        _arr(0.5),
        "lesson_1",
        y_hat_forecast=_arr(0.5),
        forecast_label="FORECAST",
    )
    npzs = list(tmp_path.rglob("*.npz"))
    assert len(npzs) == 1, f"expected exactly one biopsy npz, found {[p.name for p in npzs]}"


def test_the_npz_holds_every_row_the_figure_draws(vd, tmp_path):
    vd.biopsy_training_performance(
        _arr(2.0),
        _arr(1.0),
        _arr(1.0),
        _arr(0.5),
        "lesson_1",
        y_hat_forecast=_arr(0.5),
        forecast_label="FORECAST",
    )
    with np.load(list(tmp_path.rglob("*.npz"))[0], allow_pickle=False) as z:
        for key in ("y_reg", "y_hat_reg", "y_cls", "y_hat_cls", "y_hat_forecast"):
            assert key in z, f"{key} is drawn in the figure but not persisted"
            assert z[key].shape == (T, H, W, C)
        assert "units" in z, (
            "the log1p-vs-counts ambiguity is what made these arrays hard to read; "
            "the units string must travel with them"
        )
        assert "log1p" in str(z["units"])


def test_the_arrays_are_written_even_when_the_figure_cannot_be_drawn(vd, tmp_path, monkeypatch):
    """The data is irrecoverable; the plot is not. So the write must precede the drawing.

    Mutating `_load_mpl` to raise simulates matplotlib being unavailable. The error propagates
    because `_load_mpl` sits outside the method's try/except — pre-existing behaviour, not
    changed here and not endorsed by this test. What is asserted is the ORDERING: the npz must
    already be on disk by the time the drawing is attempted.
    """
    import views_hydranet.utils.visual_diagnostics as vdmod

    def _boom():
        raise RuntimeError("matplotlib is unavailable")

    monkeypatch.setattr(vdmod, "_load_mpl", _boom)
    with pytest.raises(RuntimeError):
        vd.biopsy_training_performance(
            _arr(2.0),
            _arr(1.0),
            _arr(1.0),
            _arr(0.5),
            "lesson_1",
            y_hat_forecast=_arr(0.5),
            forecast_label="FORECAST",
        )
    assert list(tmp_path.rglob("*.npz")), (
        "the npz must be written BEFORE matplotlib is touched — otherwise a plotting failure "
        "destroys the only copy of the field"
    )


# ------------------------------------------------------- the plot does not invent a composition


def test_the_forecast_row_is_exactly_what_the_caller_passed(vd, tmp_path):
    """⛔ The guard against the plot re-deriving `gate × body`.

    `gate · log1p(mu)` and `log1p(gate · mu)` are different numbers, and the second is what is
    deployed. With gate 0.5 and body 2.0 the in-plot composition would give 1.0; the caller's
    honest value here is 3.0 and must survive untouched.
    """
    honest = _arr(3.0)
    vd.biopsy_training_performance(
        _arr(2.0),
        _arr(2.0),
        _arr(1.0),
        _arr(0.5),
        "lesson_1",
        y_hat_forecast=honest,
        forecast_label="FORECAST",
    )
    with np.load(list(tmp_path.rglob("*.npz"))[0], allow_pickle=False) as z:
        np.testing.assert_array_equal(
            z["y_hat_forecast"],
            honest,
            "the forecast row must be the field the caller composed, not one the plot re-derived",
        )
        assert not np.allclose(z["y_hat_forecast"], z["y_hat_cls"] * z["y_hat_reg"]), (
            "the forecast row equals gate × body — the plot is composing again"
        )


def test_the_forecast_field_is_required_not_defaulted(vd):
    """C-85: a shadow default would silently restore the wrong composition."""
    with pytest.raises(TypeError):
        vd.biopsy_training_performance(
            _arr(), _arr(), _arr(), _arr(), "lesson_1", forecast_label="FORECAST"
        )


# --------------------------------------------------------------- one colour scale per unit


def test_the_count_rows_share_one_scale_spanning_all_of_them():
    truth, body, forecast = _arr(0.0), _arr(4.0), _arr(1.0)
    assert VisualDiagnostics._shared_count_scale([truth, body, forecast]) == (0.0, 4.0)


def test_a_row_of_zeros_cannot_collapse_the_scale():
    """`vmin == vmax` makes imshow render every panel identically — the original failure."""
    vmin, vmax = VisualDiagnostics._shared_count_scale([_arr(0.0), _arr(0.0)])
    assert vmin != vmax, "an all-zero window must still produce a usable, monotone scale"
    assert (vmin, vmax) == (0.0, 1.0)


def test_the_scale_floor_stays_pinned_at_zero():
    """A floating vmin would reintroduce the per-row stretch these are counts, not z-scores."""
    assert VisualDiagnostics._shared_count_scale([_arr(3.0), _arr(5.0)])[0] == 0.0


def test_non_finite_values_do_not_poison_the_scale():
    dirty = _arr(2.0)
    dirty[0, 0, 0, 0] = np.nan
    dirty[0, 0, 1, 0] = np.inf
    assert VisualDiagnostics._shared_count_scale([dirty]) == (0.0, 2.0)


# ------------------------------------------------- fed_grad_max: unmeasured is not the same as 0


def test_an_empty_sink_leaves_the_value_unmeasured():
    assert _merge_fed_max(None, []) is None
    assert _merge_fed_max(None, None) is None


def test_a_measured_zero_is_kept_as_zero_not_blanked():
    """⛔ THE distinction this fix exists to make.

    The wire being off (nothing measured) and the wire being on with a dead gradient (measured
    0.0) were the same three characters in the CSV. A measured zero is a real observation.
    """
    assert _merge_fed_max(None, [0.0]) == 0.0


def test_the_running_max_is_a_max_in_both_directions():
    assert _merge_fed_max(1.0, [2.0]) == 2.0
    assert _merge_fed_max(3.0, [2.0]) == 3.0


# ------------------------------------------------- the shared scale must not black out the gate


def _render_and_capture(vd, monkeypatch, *, gate_fill):
    """Render one biopsy, recording every ``imshow``'s (vmin, vmax) and every row's y-label.

    The scales are chosen inside ``biopsy_training_performance`` and then exist only as pixels, so
    the only honest observable is what the plotting calls were handed. ``set_ylabel`` fires once
    per row (``t_idx == 0``), so the two lists index by row in the same order.
    """
    import matplotlib.axes

    scales, labels = [], []
    real_imshow = matplotlib.axes.Axes.imshow
    real_ylabel = matplotlib.axes.Axes.set_ylabel

    def spy_imshow(self, *a, **kw):
        scales.append((kw.get("vmin"), kw.get("vmax")))
        return real_imshow(self, *a, **kw)

    def spy_ylabel(self, label, *a, **kw):
        labels.append(label)
        return real_ylabel(self, label, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy_imshow)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", spy_ylabel)
    vd.biopsy_training_performance(
        _arr(2.0),
        _arr(1.0),
        _arr(1.0),
        _arr(gate_fill),
        "lesson_1",
        y_hat_forecast=_arr(0.5),
        forecast_label="FORECAST",
    )
    assert len(scales) == 30, f"expected 5 rows x 6 times of imshow, got {len(scales)}"
    assert len(labels) == 5, f"expected one y-label per row, got {len(labels)}"
    # 6 panels per row, in row order
    return [scales[r * 6] for r in range(5)], labels


def test_the_gate_row_is_not_pinned_to_the_full_probability_range(vd, monkeypatch):
    """⛔ The gate row shared row 2's fixed [0, 1]. This model's gate spans ~0.003-0.05, so the
    whole row rendered black and the spatial structure the row exists to show was erased — the
    same loss of information the shared scale was added to prevent, over-corrected by one row.
    """
    rows, _ = _render_and_capture(vd, monkeypatch, gate_fill=0.05)
    assert rows[3][0] == 0.0 and rows[3][1] == pytest.approx(0.05, abs=1e-6), (
        f"the gate row must stretch to its own max, not [0, 1]; got {rows[3]}"
    )


def test_the_binary_cls_truth_row_is_still_pinned_to_zero_one(vd, monkeypatch):
    """The anti-over-reach half. Row 2 is a 0/1 indicator that already spans [0, 1]; stretching it
    too would make an all-ones truth panel indistinguishable from an all-0.05 one."""
    rows, _ = _render_and_capture(vd, monkeypatch, gate_fill=0.05)
    assert rows[2] == (0.0, 1.0), f"the binary cls truth row must stay on [0, 1]; got {rows[2]}"


def test_the_gate_rows_label_prints_the_range_it_was_actually_drawn_on(vd, monkeypatch):
    """A row with its own stretch and a hardcoded "[0, 1]" label is worse than the black row: it
    is a wrong number. The label must carry the scale, like the three count rows already do."""
    _, labels = _render_and_capture(vd, monkeypatch, gate_fill=0.05)
    assert "0.05" in labels[3], f"the gate row label must print its own vmax; got {labels[3]!r}"
    assert "[0, 1]" not in labels[3], f"the gate row label still claims [0, 1]: {labels[3]!r}"


def test_an_all_zero_gate_row_does_not_collapse_to_a_degenerate_scale(vd, monkeypatch):
    """`vmin == vmax` makes imshow render every panel identically — the original failure, and the
    reason `_shared_count_scale` falls back to 1.0. The gate's own scale needs the same floor."""
    rows, _ = _render_and_capture(vd, monkeypatch, gate_fill=0.0)
    assert rows[3] == (0.0, 1.0), f"a degenerate gate row must fall back to [0, 1]; got {rows[3]}"


# ------------------------------------------- the biopsy must not be able to abort a 3-hour run


def _tiny_handler():
    """The canonical tiny synthetic world. ``TINY_CONFIG`` is imported (it is data, and
    `test_balancer_freeze` sets the precedent); the handler is built here rather than reaching for
    another module's fixture."""
    from tests.test_training_engine import CHANNEL_MAP, FEATURES, IDENTITY, H, T, W
    from views_hydranet.utils.volume_handler import VolumeHandler

    rng = np.random.RandomState(42)
    data = rng.rand(T, H, W, len(CHANNEL_MAP)).astype(np.float32)
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 1 + r * W + c
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=CHANNEL_MAP,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=tuple(IDENTITY),
        feature_cols=tuple(FEATURES),
    )


def test_a_failing_biopsy_neither_aborts_the_run_nor_leaves_the_model_in_eval_mode(monkeypatch):
    """⛔ The biopsy block runs real family + composition code on live tensors, and the nearest
    enclosing `try` was ~80 lines above it, guarding the time-index lookup only. So a raise in a
    DIAGNOSTIC aborted `train()` — and the block is not diagnostics-gated: `viz` is a null object
    that is truthy even with `diagnostic_visualizations: False`, so every run with a stage label
    executes it.

    `finally: model.train()` is the other half, and the more dangerous one: an escape that skipped
    it would leave the remaining lessons training in eval mode — frozen BatchNorm statistics and
    no dropout — which is far worse than a lost plot and would not raise anything.

    The synthetic failure is the biopsy's own forward, raising only when the module is in eval
    mode. That is the one forward `train()` performs outside train mode, so it fires inside the
    block and nowhere else.
    """
    from unittest.mock import MagicMock

    import torch
    from tqdm import tqdm

    from tests.test_training_engine import TINY_CONFIG
    from views_hydranet.train.training_engine import TrainingContext, make, train

    device = torch.device("cpu")
    model, criterion, optimizer, scheduler = make(TINY_CONFIG, device)
    criterion_reg, criterion_class, mtl = criterion

    real_forward = type(model).forward

    def raise_only_in_eval(self, *a, **kw):
        if not self.training:
            raise RuntimeError("synthetic biopsy failure")
        return real_forward(self, *a, **kw)

    monkeypatch.setattr(type(model), "forward", raise_only_in_eval)

    ctx = TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_reg=criterion_reg,
        criterion_class=criterion_class,
        multitaskloss_instance=mtl,
        config=TINY_CONFIG,
        device=device,
        viz=MagicMock(),  # truthy => the Stage-5 block runs, and no file I/O
    )
    result = train(
        ctx, _tiny_handler(), tqdm(total=5, disable=True), stage_label="containment_probe"
    )
    assert torch.isfinite(result["total"]), "a failed diagnostic took the training run with it"
    assert model.training, "the run was left in EVAL mode — `finally: model.train()` is missing"
    ctx.viz.biopsy_training_performance.assert_not_called()
