"""C-132 / C-133 guard — the wandb training-run lifecycle contract.

The base ForecastingModelManager uses a Template-Method shape: the *phase template*
`_execute_model_training` owns the wandb run lifecycle (`initialize_run(job_type="train")`
+ `TrainingStage.finalize_training` + `finish_run`), and calls the *hook*
`_train_model_artifact()` inside it. Subclasses must extend the HOOK, never override the
phase template (ADR-045/050; every other forecasting manager does this).

HydranetManager previously overrode the phase template with a bare `self._train_model_artifact()`,
silently dropping the wandb train run → training metrics logged into nothing (C-132). These tests
pin the contract so the divergence cannot return.

See reports/2026-06-07_wandb_falsification/.
"""

from views_pipeline_core.managers.model import ForecastingModelManager

from views_hydranet.manager.hydranet_manager import HydranetManager


def test_hydranet_does_not_override_training_phase_template():
    """Red (C-132/C-133): HydranetManager must NOT define its own `_execute_model_training`.
    Overriding the phase template bypasses the base's wandb `initialize_run('train')` wrapper and
    `finalize_training`, so training logs nothing to wandb. It must inherit the base facade."""
    assert (
        HydranetManager._execute_model_training is ForecastingModelManager._execute_model_training
    ), (
        "HydranetManager overrides _execute_model_training (the phase template) — this drops the "
        "base wandb train-run lifecycle (C-132). Customize training via the _train_model_artifact "
        "HOOK and let the base facade own initialize_run('train')/finalize_training/finish_run."
    )


def test_hydranet_customizes_training_via_the_hook():
    """Green-anchor: hydranet DOES implement training through the correct extension point —
    the `_train_model_artifact` hook — so inheriting the base phase template loses nothing."""
    assert (
        HydranetManager._train_model_artifact is not ForecastingModelManager._train_model_artifact
    ), "HydranetManager must implement the _train_model_artifact hook (the sanctioned point)."
