"""Guard tests for the rollout-ruler-trust dossier's drivers (Epic #263).

These load code from the gitignored `reports/` tree, so they are skip-guarded — the same C-247
pattern as `test_gw_stratified.py` / `test_score_v2_horizons.py`. The *pure* half of the ruler
lives in tracked `scripts/rollout_ruler_core.py` and is tested by `test_rollout_ruler_core.py`,
which never skips.

Repo root is resolved with `parents[1]` — **never** a hardcoded machine path (C-247).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_HN = Path(__file__).resolve().parents[1]
_DOSSIER = _HN / "reports" / "2026-08-15_rollout_ruler_trust_dossier"
_TOOL = _DOSSIER / "tools" / "partition_audit.py"

if not _TOOL.exists():
    pytest.skip(
        "partition_audit.py is gitignored (absent in a clone/CI); needs reports/.",
        allow_module_level=True,
    )


def _mod():
    spec = importlib.util.spec_from_file_location("partition_audit", _TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------- fixture builders


def _make_model(tmp_path: Path, *, train=(121, 456), feedback="sample") -> Path:
    """A minimal views-models-shaped model dir: configs/{partitions,hyperparameters}.py."""
    model = tmp_path / "fake_arm"
    (model / "configs").mkdir(parents=True)
    (model / "configs" / "config_partitions.py").write_text(
        "def generate(steps: int = 36) -> dict:\n"
        "    return {\n"
        f"        'calibration': {{'train': {train}, 'test': (457, 504)}},\n"
        "        'validation': {'train': (121, 504), 'test': (505, 552)},\n"
        "    }\n"
    )
    (model / "configs" / "config_hyperparameters.py").write_text(
        f"def get_hp_config():\n    return {{'rollout_feedback': {feedback!r}}}\n"
    )
    # resolve_artifact now RAISES when provenance cannot be established, so a usable model
    # fixture must carry the sha sidecar its prediction dir's timestamp points at.
    arts = model / "artifacts"
    arts.mkdir(exist_ok=True)
    (arts / "calibration_model_20260812_191742.pt.sha256").write_text("0" * 64)
    return model


def _make_preds(tmp_path: Path, *, first_m0=457, n_origins=3, n_samples=16, name=None) -> Path:
    """A prediction dir: origin_<i>/<target>/{y_pred.npy,identifiers.npz}."""
    pred = tmp_path / (name or "predictions_calibration_20260812_191742")
    for i in range(n_origins):
        m0 = first_m0 + i
        months = np.repeat(np.arange(m0, m0 + 36, dtype=np.int32), 4)
        units = np.tile(np.arange(1, 5, dtype=np.int32), 36)
        for tgt in ("lr_sb_best", "by_sb_best"):
            d = pred / f"origin_{i}" / tgt
            d.mkdir(parents=True)
            np.save(d / "y_pred.npy", np.zeros((months.size, n_samples), dtype=np.float32))
            np.savez(d / "identifiers.npz", time=months, unit=units)
    return pred


# ------------------------------------------------------------------- C-217 leakage


def test_partition_audit_rejects_leaking_origins(tmp_path):
    """An origin emitting at or before train_max is in-sample — must raise, not warn.

    This is the whole point of C-217: the clearance holds only for calibration-partition
    artifacts. Pointed at a validation-partition artifact (train 121-504), all 13 origins
    would be in-sample and every T>0 number optimistic.
    """
    m = _mod()
    model = _make_model(tmp_path, train=(121, 456))
    pred = _make_preds(tmp_path, first_m0=456)  # one month too early
    with pytest.raises(ValueError, match="C-217 LEAK"):
        m.audit_arm(pred, model, truth_parquet=None)


def test_partition_audit_rejects_a_validation_partition_artifact(tmp_path):
    """The realistic failure: correct origins, but the model was trained through 504."""
    m = _mod()
    model = _make_model(tmp_path, train=(121, 504))
    pred = _make_preds(tmp_path, first_m0=457)
    with pytest.raises(ValueError, match="C-217 LEAK"):
        m.audit_arm(pred, model, truth_parquet=None)


def test_partition_audit_accepts_457_origin(tmp_path):
    """The real configuration: train ends 456, origins start 457 ⇒ no leak."""
    m = _mod()
    rec = m.audit_arm(
        _make_preds(tmp_path, first_m0=457), _make_model(tmp_path), truth_parquet=None
    )
    assert rec["leak"] is False
    assert rec["min_emitted_month"] == 457
    assert rec["partition"]["train"] == [121, 456]
    assert rec["n_origins"] == 3


# ------------------------------------------------------------------ C-218 feedback


def test_partition_audit_rejects_mean_feedback(tmp_path):
    """A mean-feedback rollout is broken by construction; refuse to score it as deployed skill."""
    m = _mod()
    model = _make_model(tmp_path, feedback="mean")
    pred = _make_preds(tmp_path, first_m0=457)
    with pytest.raises(ValueError, match="C-218"):
        m.audit_arm(pred, model, truth_parquet=None)


def test_mean_feedback_allowed_only_as_an_explicit_diagnostic(tmp_path):
    """`diagnostic_only=True` is the sole escape, and it is recorded in the audit record."""
    m = _mod()
    model = _make_model(tmp_path, feedback="mean")
    pred = _make_preds(tmp_path, first_m0=457)
    rec = m.audit_arm(pred, model, diagnostic_only=True, truth_parquet=None)
    assert rec["rollout_feedback"] == "mean"
    assert rec["diagnostic_only"] is True


# ---------------------------------------------------------------------- C-220 cube


def test_partition_audit_rejects_point_forecast_cubes(tmp_path):
    """The driver must apply the pure C-220 guard to every cube it finds."""
    m = _mod()
    model = _make_model(tmp_path)
    pred = _make_preds(tmp_path, first_m0=457, n_samples=1)
    with pytest.raises(ValueError, match="C-220"):
        m.audit_arm(pred, model, truth_parquet=None)


# --------------------------------------------------------- provenance & real arms


def test_audit_reads_headers_not_cubes(tmp_path):
    """Cheap-by-design: the audit must not materialise a cube (it runs before any 2.5 GB load)."""
    m = _mod()
    pred = _make_preds(tmp_path, first_m0=457)
    shapes = m.cube_shapes(pred)
    assert shapes and all(len(s) == 2 and s[1] == 16 for s in shapes.values())
    # mmap_mode='r' returns a memmap, never an in-RAM copy
    arr = np.load(next(pred.glob("origin_*/*/y_pred.npy")), mmap_mode="r")
    assert isinstance(arr, np.memmap)


def test_truth_sha_pin_matches():
    """The pinned v2 truth must still hash to V2_TRUTH_SHA256 — the substrate has not moved."""
    v2 = _HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_ruler.py"
    if not v2.exists():
        pytest.skip("v2_ruler.py absent (reports/ gitignored)")
    spec = importlib.util.spec_from_file_location("v2_ruler", v2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not Path(mod.V2_TRUTH).exists():
        pytest.skip("v2 truth parquet absent")
    assert _mod()._sha256(Path(mod.V2_TRUTH)) == mod.V2_TRUTH_SHA256


def test_real_audit_record_is_clean_for_every_scored_arm():
    """The committed audit must show no leak and sample feedback for every arm S6 will score."""
    rec_file = _DOSSIER / "results" / "partition_audit.json"
    if not rec_file.exists():
        pytest.skip("partition_audit.json not generated yet (run tools/partition_audit.py)")
    records = json.loads(rec_file.read_text())
    assert records, "audit ran but recorded no arms"
    for r in records:
        assert r["leak"] is False, f"{r['arm']} leaks"
        assert r["rollout_feedback"] == "sample", f"{r['arm']} is not sample-feedback"
        assert r["n_origins"] == 13, f"{r['arm']} has {r['n_origins']} origins, expected 13"
        assert r["sample_widths"] == [16], f"{r['arm']} has widths {r['sample_widths']}"
        assert r["truth_matches_pin"] is True, f"{r['arm']} truth sha does not match the pin"


# ------------------------------- S3: the climatology drops into the existing ruler unchanged


def _v2_metric_row():
    """Load score_v2_horizons._metric_row — the frozen 21-column metric row."""
    v2 = _HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "tools" / "score_v2_horizons.py"
    if not v2.exists():
        pytest.skip("score_v2_horizons.py absent (reports/ gitignored)")
    spec = importlib.util.spec_from_file_location("score_v2_horizons", v2)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_climatology_passes_through_metric_row_unchanged():
    """The design constraint: the climatology arm must be scoreable by the EXISTING ruler.

    `climatology_resample` returns the same `{(m0, h, u): (samples, gate)}` gathered-dict shape
    `gather_all_horizons` produces, so it drops into `score_v2_horizons._metric_row` and
    `gw_stratified.score_gw_v2` with **zero changes to either**. If this breaks, the reference
    is not usable by the ruler and S6 cannot run.
    """
    import sys as _sys

    _sys.path.insert(0, str(_HN / "scripts"))
    from rollout_ruler_core import climatology_resample

    mod = _v2_metric_row()
    support = [(457, u) for u in range(1, 51)]
    tmap = {}
    rng = np.random.default_rng(0)
    for u in range(1, 51):
        for m in range(421, 494):
            tmap[(m, u)] = float(rng.gamma(2.0, 5.0)) if rng.random() < 0.2 else 0.0

    gathered = climatology_resample(tmap, support, (1,), window=36, n_samples=16, seed=0)
    row = mod._metric_row("climatology", gathered, support, tmap, "sb", 1, None)

    assert np.isfinite(row["crps_all"]), f"crps_all not finite: {row['crps_all']}"
    assert np.isfinite(row["crps_none"])
    assert row["N"] == len(support)
    assert row["model"] == "climatology"
    # 21-column contract intact — the frozen ruler was not modified to accommodate us
    for col in ("crps_all", "crps_events", "crps_none", "AP", "size_ratio", "act_ratio"):
        assert col in row, f"{col} missing from _metric_row output"


def test_climatology_crps_is_not_its_mae():
    """A real distribution, not a point: CRPS must differ from the absolute error of its mean.

    This is what `_persistence_gathered` fails — a 1-sample 'distribution' whose CRPS *is* MAE.
    """
    import sys as _sys

    _sys.path.insert(0, str(_HN / "scripts"))
    from rollout_ruler_core import climatology_resample

    tmap = {(m, 1): float(m % 7) for m in range(400, 500)}
    g = climatology_resample(tmap, [(457, 1)], (1,), n_samples=16, seed=0)
    draws = g[(457, 1, 1)][0]
    assert len(np.unique(draws)) > 1, "history was constant; pick a varying fixture"

    y = 3.0
    a = np.abs(draws - y).mean()
    b = np.abs(draws[:, None] - draws[None, :]).mean()
    crps = a - 0.5 * b
    mae_of_mean = abs(draws.mean() - y)
    assert crps != pytest.approx(mae_of_mean), "CRPS collapsed to MAE — reference is degenerate"
    assert crps < a, "CRPS must be below the mean absolute deviation for a spread ensemble"


# ------------------------------------------------ S4: inferential-honesty guards


def _gw():
    p = _HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "tools" / "gw_stratified.py"
    if not p.exists():
        pytest.skip("gw_stratified.py absent (reports/ gitignored)")
    spec = importlib.util.spec_from_file_location("gw_stratified", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gw_test_retains_only_vectors_not_cubes():
    """C-252, made explicit: the significance path must never allocate an (N, S) array.

    `score_horizons_v2` frees each arm's cube (`del g`) before gathering the next because two
    full cubes do not fit in RAM. A significance test that re-materialised them would defeat
    that guard and OOM the real run. The invariant was previously only *implicit* — the vector
    API was exercised, but nothing measured memory.
    """
    import tracemalloc

    gw = _gw()
    n_origins, per_origin = 13, 15_000
    n = n_origins * per_origin  # 195k, the real order of magnitude
    rng = np.random.default_rng(0)
    loss_a = rng.gamma(2.0, 1.0, n)
    loss_b = loss_a + rng.normal(0.0, 0.1, n)
    stratum = np.ones(n, dtype=bool)
    origins = np.repeat(np.arange(n_origins), per_origin)

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    gw.gw_stratified_test(loss_a, loss_b, stratum, origins, n_boot=50, seed=0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # An (N, 16) float64 cube at this N would be ~25 MB. The vector path must stay far below.
    grew_mb = (peak - base) / 1e6
    cube_mb = n * 16 * 8 / 1e6
    assert grew_mb < cube_mb / 2, (
        f"C-252: gw_stratified_test grew {grew_mb:.1f} MB; an (N,16) cube would be "
        f"{cube_mb:.1f} MB. The significance path must keep per-cell VECTORS only."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, deliberately pinned not fixed (Epic #263 S4, register C-277). "
        "block_bootstrap_crps computes single-arm support via _fixed_support({label: g}), "
        "while score_horizons/score_horizons_v2 use the cross-arm intersection — so a CI it "
        "produces is computed on a different (larger) cell set than the point estimate it "
        "annotates. Never fires today because the sets coincide on current arms. S6 uses "
        "score_gw_v2 (correct), so the function is unused here: fixing an unused function is "
        "scope creep, leaving it silently broken is a trap."
    ),
)
def test_block_bootstrap_uses_cross_arm_support():
    """The CI's support must equal the point estimate's cross-arm intersection, not one arm's."""
    roll = (
        _HN
        / "reports"
        / "2026-07-25_t0_rollout_skill_dossier"
        / "tools"
        / "rollout_skill_score.py"
    )
    if not roll.exists():
        pytest.skip("rollout_skill_score.py absent (reports/ gitignored)")
    text = roll.read_text()
    start = text.index("def block_bootstrap_crps")
    body = text[start : text.index("\ndef ", start + 1)]
    assert "_support_keys" in body and "_fixed_support" not in body, (
        "block_bootstrap_crps still builds support from a single arm (_fixed_support) instead "
        "of the cross-arm intersection (_support_keys), so its CI annotates a different cell "
        "set than the estimate. See register C-277."
    )


# ---------------------------- PR #273 review findings: the CI path and provenance


def _rescore():
    p = _DOSSIER / "tools" / "rescore_v2.py"
    if not p.exists():
        pytest.skip("rescore_v2.py absent (reports/ gitignored)")
    spec = importlib.util.spec_from_file_location("rescore_v2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ci_pass_uses_the_caller_support_not_per_arm_coverage():
    """C-277's shape, in the CI path — the guard that `block_bootstrap_crps` still fails.

    The point estimate is computed on the cross-arm intersection (G4). If the bootstrap builds
    its cell universe from one arm's own coverage instead, the interval describes a different
    population than the number it annotates. `_add_origin_block_ci` previously did exactly
    that; it now takes `support` from the caller.
    """
    import inspect

    src = inspect.getsource(_rescore()._add_origin_block_ci)
    assert "support" in inspect.signature(_rescore()._add_origin_block_ci).parameters, (
        "_add_origin_block_ci must receive the intersected support, not derive it (C-277)"
    )
    assert "for m0, (_c, _t, uu) in a.items() for u in uu" not in src, (
        "the per-arm support comprehension is back — that is C-277 reintroduced"
    )


def test_ci_pass_covers_every_horizon():
    """Restricting the CI to a subset of horizons is what starved the rule on 48 of 84 rows."""
    import inspect

    m = _rescore()
    src = inspect.getsource(m._add_origin_block_ci)
    assert "CI_HORIZONS" not in src, (
        "the CI pass is horizon-restricted again; every scored horizon needs an interval or "
        "its rows cannot reach REAL"
    )
    assert "for h in horizons" in src


def test_resolve_artifact_raises_when_provenance_cannot_be_established(tmp_path):
    """A missing sha sidecar must fail loud, not record None as if audited."""
    m = _mod()
    model = _make_model(tmp_path)
    for sidecar in (model / "artifacts").glob("*.sha256"):
        sidecar.unlink()  # the fixture supplies one; this test is about its absence
    good = tmp_path / "predictions_calibration_20260812_191742"
    good.mkdir()
    with pytest.raises(FileNotFoundError, match="no .sha256 sidecar"):
        m.resolve_artifact(model, good, "calibration")

    bad = tmp_path / "not_a_prediction_dir"
    bad.mkdir()
    with pytest.raises(ValueError, match="does not carry a run timestamp"):
        m.resolve_artifact(model, bad, "calibration")


def test_every_scored_row_carries_a_verdict_and_a_measured_ci():
    """The artifact must be self-contained: no row may rely on a downstream default."""
    f = _DOSSIER / "results" / "rescore.csv"
    if not f.exists():
        pytest.skip("rescore.csv not generated yet")
    import csv

    rows = list(csv.DictReader(f.open()))
    assert rows, "rescore.csv is empty"
    assert "verdict" in rows[0], (
        "the verdict is the epic's deliverable and must be persisted, not printed — otherwise "
        "the published tables are hand transcriptions with nothing to re-derive against"
    )
    for r in rows:
        assert r["verdict"] in ("REAL", "ARTIFACT", "UNDECIDABLE"), r
        assert r["ci_excludes_zero"] in ("True", "False"), (
            f"{r['model']} {r['target']} h={r['h']} has ci_excludes_zero="
            f"{r['ci_excludes_zero']!r} — an unmeasured CI silently forces non-REAL"
        )
