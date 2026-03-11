"""
Structural memory hygiene tests for HydranetManager.

The shared data pipeline (_run_data_pipeline) handles the DataFrame lifecycle:

  df = fetch → standardize → scale
  handler = VolumeHandler.from_df(df, ...)   # df fully consumed
  sniff_forecast_alignment(df, handler, ...) # df read-only, not stored
  del df; gc.collect()                       # ← required
  return handler, scaler, sniffer

All 3 artifact methods (train, evaluate, forecast) delegate to _run_data_pipeline,
so the memory hygiene gate is verified once on the shared method.
"""

import inspect

from views_hydranet.manager.hydranet_manager import HydranetManager


class TestManagerDataFrameLifecycle:
    """
    Structural tests: verify that the viewser DataFrame is explicitly released
    after sniff_forecast_alignment() in the shared data pipeline.

    These tests FAIL if 'del df' or 'gc.collect()' is absent from the method
    source, making any regression immediately visible.
    """

    def test_run_data_pipeline_deletes_df(self):
        """
        _run_data_pipeline() must release the viewser DataFrame after
        sniff_forecast_alignment(). The DataFrame (~1 GB at pgm scale)
        is fully consumed by VolumeHandler.from_df() and read (but not
        stored) by sniff_forecast_alignment(). It is not referenced
        after that point.
        """
        source = inspect.getsource(HydranetManager._run_data_pipeline)
        assert "del df" in source, (
            "_run_data_pipeline() must explicitly 'del df' after "
            "sniff_forecast_alignment(). The DataFrame is not needed "
            "after volume construction; keeping it alive wastes ~1 GB "
            "at pgm scale."
        )
        assert "gc.collect()" in source, (
            "_run_data_pipeline() must call gc.collect() after 'del df' "
            "to prompt the allocator to release the freed memory."
        )

    def test_train_delegates_to_shared_pipeline(self):
        """_train_model_artifact() must delegate to _run_data_pipeline()."""
        source = inspect.getsource(HydranetManager._train_model_artifact)
        assert "_run_data_pipeline" in source

    def test_setup_evaluation_delegates_to_shared_pipeline(self):
        """_setup_evaluation() must delegate to _run_data_pipeline()."""
        source = inspect.getsource(HydranetManager._setup_evaluation)
        assert "_run_data_pipeline" in source

    def test_forecast_delegates_to_shared_pipeline(self):
        """_forecast_model_artifact() must delegate to _run_data_pipeline()."""
        source = inspect.getsource(HydranetManager._forecast_model_artifact)
        assert "_run_data_pipeline" in source
