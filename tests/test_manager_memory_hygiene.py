"""
Structural memory hygiene tests for HydranetManager.

Both _evaluate_model_artifact() and _forecast_model_artifact() follow the
same lifecycle:

  df = fetch → standardize → scale
  handler = VolumeHandler.from_df(df, ...)   # df fully consumed
  sniff_forecast_alignment(df, handler, ...) # df read-only, not stored
  del df; gc.collect()                       # ← required
  <inference loop>                           # handler only, df not needed

These tests verify the 'del df; gc.collect()' pattern is present in BOTH
methods — ensuring symmetric DataFrame lifecycle management across the
evaluate and forecast paths.
"""

import inspect

from views_hydranet.manager.hydranet_manager import HydranetManager


class TestManagerDataFrameLifecycle:
    """
    Structural tests: verify that the viewser DataFrame is explicitly released
    after sniff_forecast_alignment() in both artifact methods.

    These tests FAIL if 'del df' or 'gc.collect()' is absent from the method
    source, making any regression immediately visible.
    """

    def test_evaluate_model_artifact_deletes_df_after_sniff(self):
        """
        _setup_evaluation() must release the viewser DataFrame before the
        inference loop begins.

        Since _evaluate_model_artifact() delegates setup to _setup_evaluation(),
        the DataFrame lifecycle lives in _setup_evaluation().  The DataFrame
        (~1 GB at pgm scale) is fully consumed by VolumeHandler.from_df() and
        read (but not stored) by sniff_forecast_alignment(). It is not
        referenced after that point.
        """
        source = inspect.getsource(HydranetManager._setup_evaluation)
        assert "del df" in source, (
            "_setup_evaluation() must explicitly 'del df' after "
            "sniff_forecast_alignment(). The DataFrame is not needed during "
            "the inference loop; keeping it alive wastes ~1 GB at pgm scale."
        )
        assert "gc.collect()" in source, (
            "_setup_evaluation() must call gc.collect() after 'del df' "
            "to prompt the allocator to release the freed memory."
        )

    def test_forecast_model_artifact_deletes_df_after_sniff(self):
        """
        _forecast_model_artifact() must release the viewser DataFrame before
        the inference loop begins — symmetric with _evaluate_model_artifact().

        The DataFrame lifecycle is identical in both methods: fetch → scale →
        VolumeHandler.from_df() → sniff_forecast_alignment() → inference.
        The DataFrame is not referenced after sniff_forecast_alignment().
        """
        source = inspect.getsource(HydranetManager._forecast_model_artifact)
        assert "del df" in source, (
            "_forecast_model_artifact() must explicitly 'del df' after "
            "sniff_forecast_alignment(). Missing symmetry with evaluate path: "
            "the forecast DataFrame is equally large and equally redundant "
            "before the inference loop."
        )
        assert "gc.collect()" in source, (
            "_forecast_model_artifact() must call gc.collect() after 'del df' "
            "to match the evaluate-path pattern and ensure consistent memory "
            "behaviour across both lifecycle methods."
        )
