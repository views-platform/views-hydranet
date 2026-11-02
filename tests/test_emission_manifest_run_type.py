"""ADR-074: the manifest is written for the RUN TYPE, not for the CLI's `eval_type`.

⛔ This file exists because the manifest shipped **dead** and 2472 tests stayed green over it.

`_evaluate_model_artifact_streaming(eval_type, ...)` receives `self.args.eval_type` from
pipeline-core (`managers/model/model.py:1092,1486`), and that value is validated to
`{standard, complete, live}` (`cli/args.py:442`). It is a label for device logging. The manifest
needs the **partition** — `calibration` / `validation` / `forecasting` — because it resolves the
artifact and derives `predictions_{run_type}_{ts}` from it.

⭐ Passing `eval_type` therefore produced a manifest that silently never appeared: the artifact
resolved to nothing, the derived directory named a path that never exists, and the blanket
`except` that exists so a finished run is never discarded swallowed the difference. That is this
repository's C-324 — "inert but tested".

⚠️ **Why the existing tests could not catch it.** Every one of them passes
`eval_type="calibration"`, which happens to equal the run type, and the adversarial fixtures mock
`resolve_artifact_path` away entirely. ⇒ The two values were never distinguishable in a test. Here
they deliberately differ.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("views_pipeline_core")

from tests.test_manager_integration_local import (  # noqa: E402
    INTEGRATION_CFG,
    _make_history_df,
    _make_manager,
    _patch_data_and_model,
)
from views_hydranet.manager.hydranet_manager import HydranetManager  # noqa: E402


def _run_streaming(tmp_path, eval_type):
    """Drive the real streaming path and return the args the manifest writer was called with."""
    from tests.conftest import TinyModel

    manager = _make_manager(tmp_path)
    df = _make_history_df()
    model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)
    cfg_patches = _patch_data_and_model(manager, df, model, INTEGRATION_CFG)

    seen = {}

    def spy(self, ctx, run_type, artifact_name, **kw):
        seen["run_type"] = run_type
        seen["kw"] = kw

    with (
        cfg_patches[0],
        cfg_patches[1] as mock_fetch,
        cfg_patches[2] as mock_art,
        patch.object(HydranetManager, "_write_emission_manifest", spy),
    ):
        mock_fetch.return_value.fetch_df.return_value = df
        mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
        mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

        manager._evaluate_model_artifact_streaming(
            eval_type, None, origin_sink=lambda i, pf: pf.clear()
        )

    return seen


def test_the_manifest_gets_the_run_type_even_when_eval_type_differs(tmp_path):
    """⭐⭐ The one that matters. `eval_type="standard"` must NOT reach the manifest.

    `standard` is what a real run passes. The configured partition is `calibration`. A manifest
    built from the former describes a run that does not exist.
    """
    seen = _run_streaming(tmp_path, eval_type="standard")

    assert seen["run_type"] == "calibration", (
        "the manifest was handed the CLI's eval_type instead of the run's partition — it would "
        "resolve no artifact, derive a directory that never exists, and silently write nothing"
    )
    assert seen["run_type"] != "standard"


@pytest.mark.parametrize("eval_type", ["standard", "complete", "live"])
def test_every_legal_eval_type_still_yields_the_partition(tmp_path, eval_type):
    """All three legal `eval_type` values are wrong as a run type, so none may reach the manifest.

    Parametrised over the enum rather than one value, because picking only `standard` would pass
    on an implementation that special-cased it.
    """
    assert _run_streaming(tmp_path, eval_type)["run_type"] == "calibration"


def test_the_streaming_path_requires_the_directory_to_already_exist(tmp_path):
    """⚠️ The eval paths keep the moved-convention guard; only the forecast path waives it.

    On a streaming run the pipeline's sink has already created the per-origin directories, so an
    absent directory means the derivation is wrong and must be reported rather than created.
    """
    seen = _run_streaming(tmp_path, eval_type="standard")
    assert seen["kw"].get("directory_must_exist", True) is True


def test_the_unreachable_batch_path_writes_no_manifest(tmp_path):
    """⛔ `_evaluate_model_artifact` is the DataFrame branch, which no HydraNet takes.

    It also never creates `predictions_{run_type}_{ts}`, so a manifest written from there would
    have nowhere to go. The call was removed; a call that cannot run is worse than none, because
    it reads as wired. This test fails if someone adds it back.
    """
    from tests.conftest import TinyModel

    manager = _make_manager(tmp_path)
    df = _make_history_df()
    model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)
    cfg_patches = _patch_data_and_model(manager, df, model, INTEGRATION_CFG)
    writer = MagicMock()

    with (
        cfg_patches[0],
        cfg_patches[1] as mock_fetch,
        cfg_patches[2] as mock_art,
        patch.object(HydranetManager, "_write_emission_manifest", writer),
    ):
        mock_fetch.return_value.fetch_df.return_value = df
        mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
        mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")
        manager._evaluate_model_artifact(eval_type="calibration")

    writer.assert_not_called()
