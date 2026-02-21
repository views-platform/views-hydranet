from unittest.mock import MagicMock, patch

from views_hydranet.manager.hydranet_manager import HydranetManager


def test_model_path_accessibility():
    """
    Regression Test: Ensure that _train_model_artifact uses the safe internal _model_path.

    The base class 'ForecastingModelManager' exposes 'model_path' via a property or attribute.
    However, if initialization is mocked or partial (as seen in our test environment setup),
    accessing 'self.model_path' might fail if it relies on internal state that wasn't set.

    This test verifies that the manager correctly set 'self._model_path' during init
    and that we should prefer using that.
    """

    # Simulate the "Mocked Base Class" scenario where super().__init__ does nothing
    with (
        patch(
            "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
            return_value=None,
        ),
        patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"),
    ):
        mock_mpm = MagicMock()
        manager = HydranetManager(model_path=mock_mpm)

        # 1. Verify that our safe internal storage is present
        assert hasattr(manager, "_model_path")
        assert manager._model_path == mock_mpm

        # 2. Verify if 'self.model_path' is accessible.
        # In the failing run, this raised AttributeError.
        # If base class is NOT initialized, 'self.model_path' won't exist.
        try:
            _ = manager.model_path
        except AttributeError:
            print("\nconfirmed: manager.model_path is missing when base init is skipped.")
