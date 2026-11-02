"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training,
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import gc
import logging
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, NamedTuple

import pandas as pd
from views_frames import PredictionFrame
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from views_hydranet.infrastructure.coverage import declared_coverage_of
from views_hydranet.infrastructure.emission_manifest import (
    SCHEMA_VERSION,
    EmissionManifest,
    RepoProvenance,
    write_manifest,
)
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate
from views_hydranet.train.train_model import train_model_artifact
from views_hydranet.utils.config_initializer import ConfigInitializer
from views_hydranet.utils.data_fetcher import DataFetcher
from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.disk_guard import assert_disk_headroom
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.grid_naming import GRID_ID_ALIASES, canonicalize_config_grid_name
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_logging import (
    log_device_report,
    log_training_summary,
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class _EvaluationContext(NamedTuple):
    """Typed return value of HydranetManager._setup_evaluation()."""

    handler: "VolumeHandler"
    scaler: "FeatureScaler"
    origins: list[int]
    all_targets: list[str]
    orchestrator: "InferenceOrchestrator"


def _quantities_of(all_targets: "list[str]") -> "dict[str, str]":
    """ADR-074: label each target this run emitted, so a reader need not know the convention.

    Maps the TARGET NAME to its quantity, because the target name is what a reader is actually
    holding — pipeline-core writes one directory per target, named by the target. An earlier
    version mapped the two quantity labels to their PREFIXES, which was wrong three ways: the
    prefixes match no directory on disk, the map was a hardcoded literal that asserted a gate even
    for an arm that declares no classification targets, and it would have gone on omitting any
    later quantity (ADR-075's diagnostic body) forever.

    The prefixes come from `volume_handler`, so renaming one cannot leave this map declaring the
    old spelling. A target matching neither is labelled rather than dropped: an unrecognised
    quantity is a finding, and silence about it is the thing ADR-074 exists to prevent.
    """
    from views_hydranet.utils.volume_handler import BINARY_PREFIX, LINEAR_PREFIX

    labels = {}
    for target in all_targets:
        if target.startswith(LINEAR_PREFIX):
            labels[target] = "forecast"
        elif target.startswith(BINARY_PREFIX):
            labels[target] = "gate"
        else:
            labels[target] = "unrecognised"
    return labels


class HydranetManager(ForecastingModelManager):
    """
    Orchestrator for HydraNet lifecycle tasks.

    Inherits from ForecastingModelManager to integrate with the ViEWS pipeline.
    Implements multi-task evaluation and rolling-origin orchestration.
    """

    configs: dict[str, Any]

    def _run_data_pipeline(
        self,
        viz: "VisualDiagnostics",
        partition_bound: int | None = None,
    ) -> tuple["VolumeHandler", "FeatureScaler", "DataSniffer"]:
        """
        Shared data ingestion pipeline (ADR 039 Steps 1-3).

        Fetch → Biopsy Stage 1 → Standardize → Sniff → Scale →
        Biopsy Stage 2 → (partition) → Volume → Biopsy Stage 3.

        Identical on every run type. The handler returned is always the **history** volume; the
        future volume is built later and per origin, by ``InferenceOrchestrator`` via
        ``VolumeHandler.extrapolate_time``. There is nothing here for a forecast flag to select,
        which is why this no longer takes one — see the sniffer call below.

        Parameters
        ----------
        viz : VisualDiagnostics
            Visual diagnostics engine for biopsy calls.
        partition_bound : int, optional
            If set, filters the DataFrame to ``time_col <= partition_bound``
            before constructing the VolumeHandler (evaluation path).

        Returns
        -------
        (VolumeHandler, FeatureScaler, DataSniffer)
        """
        # 1. Ingest

        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df(cached_path=getattr(self, "_cached_data_path", None))

        # GH #144: canonicalize the config's grid keys to the alias the DATA actually uses, so the
        # DataSniffer + VolumeHandler (which read config['id_col']/identity_cols/index_names
        # literally) accept the data even when the model config still says the legacy priogrid_gid.
        # This is the downstream-config seam complementing grid_naming.grid_id_col (the load-path
        # rule, #145): data_fetcher resolves its own contract from the data, but these consumers
        # read the config literally. `self.configs` is a property returning a fresh combined config
        # each access, so we mutate a copy and persist it back through the setter (same pattern as
        # :200/:278). Fail-SAFE: act only when exactly one grid alias is resolvable from the data;
        # otherwise skip and let the existing sniffer/contract fail loud. No-op (byte-identical)
        # when config already matches the data (today's priogrid_gid case).
        _grid_present = {n for n in (*df.index.names, *df.columns) if n in GRID_ID_ALIASES}
        if len(_grid_present) == 1:
            _cfg = self.configs
            canonicalize_config_grid_name(_cfg, next(iter(_grid_present)))
            self.configs = _cfg
            # C-243: viz was built with the pre-canonicalization config (before this method), so
            # refresh its config too — otherwise its biopsies read the stale grid alias on renamed
            # (priogrid_id) data. Diagnostic-only, but keeps the plots correct on migrated data.
            viz.config = _cfg

        # Diagnostic plot features
        plot_feats = (
            [
                self.configs.get("time_col", "month_id"),
                self.configs.get("id_col", "priogrid_gid"),
                "c_id",
            ]
            + self.configs.get("spatial_cols", [])
            + self.configs.get("regression_targets", [])
        )

        # DIAGNOSTIC: Stage 1 (Ingestion)
        viz.biopsy_dataframe(df, "Stage 1: Raw Ingestion", features=plot_feats)

        df = DataFetcher.standardize_raw_df(df, self.configs)

        # 2. Sniff

        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 3. Scale

        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # DIAGNOSTIC: Stage 2 (Transformation)
        viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=plot_feats)

        # Partition slicing (evaluation only)
        if partition_bound is not None:
            time_col = self.configs.get("time_col", "month_id")
            df = df[df[time_col] <= partition_bound]

        # 4. Transform: DataFrame -> Volume

        handler = VolumeHandler.from_df(df, self.configs)

        # DIAGNOSTIC: Stage 3 (Volume)
        viz.biopsy_volume(handler, "Stage 3: Global Volume")

        # `handler` is VolumeHandler.from_df(df) — the HISTORY volume, on every run type. So the
        # history contract is the one that applies, and `is_forecast` is a constant here.
        #
        # This used to pass `is_forecast=forecast`, closing C-58 ("forecast mode never validates
        # continuity"). The concern was real; the remedy was not. The forecast branch requires
        # `min(volume months) == max(df months) + 1` — a volume beginning one month AFTER the
        # frame — and the CIC (§8) names its second argument `forecast_handler` for that reason.
        # A handler built from `df` begins where `df` begins, so the condition was unsatisfiable
        # and every forecasting run died here, after a full training run. It went unseen because
        # the test covering it mocks DataSniffer and only asserts the flag was passed.
        #
        # The property C-58 wanted is not unguarded: `extrapolate_time` establishes it by
        # construction (it tiles the last frame and adds arange(1, steps+1) to the time channel),
        # and test_volume_handler_hard_gates.py::test_extrapolate_time_temporal_continuity (C-23)
        # pins it. Re-checking it here would be a guard that cannot fire.
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        del df
        gc.collect()

        return handler, scaler, sniffer

    def __init__(self, model_path: ModelPathManager, wandb_notification: bool = True) -> None:
        """
        Initializes the manager and setup device.
        """
        super().__init__(model_path, wandb_notification)
        self.device = setup_device()
        self.set_dataframe_format(format=".parquet")
        self._model_path = model_path

        # Authoritative Run Timestamp (ADR 026 / Visual Diagnostics)
        self.run_timestamp = datetime.now().strftime("%d%m%y_%H%M")
        logger.info(f"🕒 HydranetManager: Initialized run with timestamp {self.run_timestamp}")

    def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fulfills the handshake contract with views_pipeline_core.
        Augments the ground-truth DataFrame with signals manufactured
        via the Instructional Blueprint.
        """
        logger.info("🛠️ HydranetManager: Manufacturing derived signals for evaluation.")
        return DataFetcher.apply_blueprint(df, self.configs)

    def _run_preflight_check(self) -> None:
        """
        Hypothesis 3 Probe: Validates Config vs Architecture.
        """
        logger.info("🛡️  Running Pre-Flight Validation...")

        # 1. Check Head Alignment
        n_reg = len(self.configs.get("regression_targets", []))
        n_class = len(self.configs.get("classification_targets", []))

        # HydraBNUNet06_LSTM4 hardcodes 3 reg + 3 class
        if n_reg != 3 or n_class != 3:
            err_msg = (
                f"ARCHITECTURE MISMATCH: Model expects 3+3 heads, "
                f"Config has {n_reg}+{n_class}. Aborting."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info("✅ Architecture: Head Count Aligned (3+3)")

    # NOTE: _execute_model_training is intentionally NOT overridden. The base
    # ForecastingModelManager phase template owns the wandb run lifecycle
    # (initialize_run("train") + TrainingStage.finalize_training + finish_run) and
    # calls the _train_model_artifact() hook below. Overriding the phase template
    # silently drops the wandb train run (was C-132). Customize training via the hook.
    def _train_model_artifact(self) -> Any:
        """
        Executes the training lifecycle and returns the trained model object.
        This method acts as the 'Operational Core' for both standard runs and sweeps.
        """
        log_device_report(
            self.device, "training", require_cuda=bool(self.configs.get("require_cuda", False))
        )
        logger.info(f"Starting HydraNet training: {self.configs['run_type']}")

        # 0. Strict Config Handshake (ADR 008/015)
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()

        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)
        handler, scaler, sniffer = self._run_data_pipeline(viz)

        # 5. Train

        # Determine persistence (Sweep Safety)
        # In a sweep, we do NOT want to save every trial's artifact to disk locally.
        is_sweep = self.configs.get("sweep", False)
        save_artifact = not is_sweep

        # Pass visualizer to training loop for Stage 4 (Sampling) probes
        model, summary = train_model_artifact(
            self._model_path,
            self.configs,
            self.device,
            handler,
            run_timestamp=self.run_timestamp,
            save_artifact=save_artifact,
        )
        log_training_summary(summary)

        return model

    def _load_model_artifact(self, artifact_name: str | None = None) -> Any:
        """Load a trained model artifact from disk."""
        add_config_fn = (
            self._config_manager.add_config
            if hasattr(self, "_config_manager")
            else (lambda x: None)
        )
        model, _ = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            add_config_fn,
            self.device,
        ).fetch_model_artifact(model_artifact_name=artifact_name)
        return model

    def _setup_evaluation(
        self,
        eval_type: str,
        model: Any,
        *,
        forecast: bool = False,
    ) -> _EvaluationContext:
        """
        Shared setup for evaluation, streaming, forecasting, and sweeps.

        Loads and scales the data, creates the VolumeHandler, sniffs
        alignment, computes origin indices, and builds the
        InferenceOrchestrator with the provided model.

        Parameters
        ----------
        eval_type : str
            Label for device logging, and **only** that — it is never used to resolve a path, name
            a directory or pick a partition.
            ⚠️ Its real values are the CLI's `standard` / `complete` / `live`, not the partition.
            An earlier version of this line gave the examples *"calibration", "forecasting"*, which
            are **run types** and are what `self.configs["run_type"]` holds. That wrong example is
            what made it plausible to thread this value into the ADR-074 manifest, where it
            produced a record of a partition the run did not have — see
            `_write_emission_manifest`. On the forecast path the caller does pass `"forecasting"`,
            which is why the two were never obviously different.
        model : torch.nn.Module
            Trained model — loaded from disk via ``_load_model_artifact()``
            or received in-memory from ``_execute_model_sweeping()``.
        forecast : bool
            If True, skips partition lookup and uses the single latest
            time index as the sole origin. Data pipeline runs without
            partition_bound (uses all available data).

        Returns
        -------
        _EvaluationContext
            handler     : VolumeHandler — spatiotemporal data carrier
            scaler      : FeatureScaler — fitted, ready for inverse_transform_volume
            origins     : list[int]     — origin indices (rolling for eval, single for forecast)
            all_targets : list[str]     — regression_targets + classification_targets
            orchestrator: InferenceOrchestrator — wired with model, device, visualizer
        """
        log_device_report(
            self.device, eval_type, require_cuda=bool(self.configs.get("require_cuda", False))
        )
        self.configs = ConfigInitializer(self.configs).get_config()
        ReproducibilityGate.lock_entropy(
            np_seed=self.configs["np_seed"], torch_seed=self.configs["torch_seed"]
        )
        # C-154: abort loud before the ~2.5 GB/origin-set prediction writes if disk is short
        # (opt-in via `min_free_disk_gb`; None => no-op, unchanged behaviour).
        assert_disk_headroom(self.configs.get("min_free_disk_gb"))
        self._run_preflight_check()
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        # ── Data pipeline ──────────────────────────────────────────────────────

        if forecast:
            handler, scaler, sniffer = self._run_data_pipeline(viz)
            origins = [handler.shape[0] - 1]
        else:
            run_type = self.configs["run_type"]
            time_steps = len(self.configs["steps"])
            partition = getattr(self, "_partition_dict", {}).get(run_type)
            test_end = partition["test"][1] if partition is not None else None

            handler, scaler, sniffer = self._run_data_pipeline(viz, partition_bound=test_end)

            if partition is not None:
                test_start = partition["test"][0]
                num_windows = test_end - (test_start - 1) - time_steps + 1
            else:
                num_windows = 1
            origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

        # ── Targets and orchestrator ──────────────────────────────────────────

        all_targets = self.configs.get("regression_targets", []) + self.configs.get(
            "classification_targets", []
        )
        orchestrator = InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)

        return _EvaluationContext(
            handler=handler,
            scaler=scaler,
            origins=origins,
            all_targets=all_targets,
            orchestrator=orchestrator,
        )

    def _repo_provenance(self) -> tuple[RepoProvenance, ...]:
        """What version control says about each repository involved, with a dirty flag.

        A commit identifier taken from a modified working tree is a false provenance claim, so the
        flag is recorded beside it rather than folded into the string (ADR-074 section 2.1). A
        repository that cannot be read at all is recorded as unknown rather than omitted: a missing
        entry is indistinguishable from a repository that was never involved.

        ⚠️ **Two things here were measured wrong and are fixed; both read correctly under pytest
        and incorrectly in production, which is why neither was visible.**

        1. The views-hydranet anchor was the IMPORTED package directory. Under pytest the
           repository root is on ``sys.path`` first, so that is the working tree and git succeeds.
           On a real run the package is a non-editable wheel and the anchor is ``site-packages``,
           which is not a worktree — so every production manifest recorded unknown-and-dirty, the
           honest-degradation branch firing as the normal case for the most reproducible
           configuration there is. Worse in the other direction: ``git -C`` walks UP, so a
           ``site-packages`` tree nested under any unrelated repository would have recorded THAT
           repository's HEAD as ours. ⇒ The toplevel is now resolved and checked to be the
           repository the anchor claims.
        2. ``status --porcelain`` counts UNTRACKED files, and the run writes its own predictions
           into views-models before this is called — so ``dirty`` was structurally unable to be
           false for the repository that holds the configuration being run, and therefore carried
           no information. ⇒ ``-uno``: tracked modifications only.
        """
        import views_hydranet

        def _git(anchor: Path, *args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(anchor), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()

        repos: list[RepoProvenance] = []
        for name, anchor in (
            ("views-hydranet", Path(views_hydranet.__file__).resolve().parent),
            ("views-models", Path(self._model_path.models)),
        ):
            commit, dirty = None, True
            try:
                toplevel = Path(_git(anchor, "rev-parse", "--show-toplevel")).resolve()
                if toplevel.name == name:
                    commit = _git(anchor, "rev-parse", "HEAD")
                    # `-uno`: untracked files are not a modification of the committed state, and
                    # this run creates them itself under views-models.
                    dirty = bool(_git(anchor, "status", "--porcelain", "-uno"))
                else:
                    # The anchor sits inside SOME repository, but not the one it claims to be —
                    # a wheel in a site-packages tree nested under an unrelated checkout, say.
                    # Recording that repository's HEAD as ours would be a false provenance claim
                    # of exactly the kind the dirty flag exists to prevent, so the value is
                    # refused rather than qualified. Left as unknown-and-dirty, as below.
                    logger.warning(
                        "ADR-074: %s resolves to the git toplevel %s, which is not %s. Recording "
                        "it as unknown and dirty rather than claiming an unrelated repository's "
                        "revision as this one's.",
                        anchor,
                        toplevel,
                        name,
                    )
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning(
                    "ADR-074: could not read version control for %s at %s (%s). Recording it as "
                    "unknown and dirty, which is the honest reading of 'we do not know'.",
                    name,
                    anchor,
                    exc,
                )
            repos.append(RepoProvenance(name=name, commit=commit, dirty=dirty))
        return tuple(repos)

    def _write_emission_manifest(
        self, ctx: "_EvaluationContext", artifact_name: str | None
    ) -> None:
        """Write the ADR-074 manifest for the run that has just finished.

        ⛔ Every value comes from an object this run used — the configuration dict the orchestrator
        was constructed with, and the origin records the orchestrator built as it emitted. Nothing
        here re-reads a configuration file or re-derives a month, because a manifest assembled that
        way would agree with the configuration by construction and could not be compared against it
        (ADR-074 section 2.3).

        ⛔ **The run type is read HERE, inside the try, and is never a parameter.** It is the
        PARTITION — `calibration`, `validation`, `forecasting` — and emphatically not the CLI's
        ``eval_type`` (`standard`/`complete`/`live`), which is only a label for device logging.
        Threading the latter in was the first version of this method and it meant the artifact
        resolved to nothing, the derived directory named a path that never exists, and the manifest
        silently never appeared while every test stayed green — C-324, "inert but tested".

        ⇒ It is read inside the try for a second reason: ``self.configs`` is a pipeline-core
        property that re-merges five sources on every access, so a missing key or a failure during
        that merge would raise. Evaluated at a call site it would do so AFTER every origin had been
        streamed and persisted, and pipeline-core would convert it into a failed run — the one
        read in this feature able to discard a completed evaluation, from the component whose whole
        contract is that it never will.

        ⚠️ **The FORECAST path deliberately has no manifest.** The predictions directory is created
        by the pipeline's ``save_pf`` only after ``_forecast_model_artifact`` returns, so a write
        from there would either refuse (nothing to write beside) or create the directory and leave
        a complete-looking record of rows that a later ``save_pf`` failure means never existed —
        strictly worse than no manifest, by this ADR's own standard. There is no post-save hook in
        this repository to attach it to. Recorded as a gap in ADR-074 section 5 rather than papered
        over.

        A failure to write the manifest does not fail the run. The predictions are the deliverable
        and they are already on disk; losing their description is worse than losing nothing, but it
        is not worth discarding a completed run over.
        """
        try:
            run_type = self.configs["run_type"]
            artifact = self._model_path.resolve_artifact_path(run_type, artifact_name)
            directory = Path(
                self._model_path.data_generated / f"predictions_{run_type}_{artifact.stem[-15:]}"
            )
            # ⛔ `Path(...)` rather than the raw value, and `is_dir()` rather than `exists()`.
            # A `MagicMock`'s `.exists()` returns a MagicMock, which is TRUTHY, so under a mocked
            # path manager this guard passed and `write_manifest` then `mkdir`-ed the mock's
            # repr-derived path. That deposited 214 empty directories in the repository root over
            # three hours, invisible to `git status` because git does not track empty directories.
            # The guard has to be a statement about a real filesystem, not about an object.
            if not directory.is_dir():
                logger.error(
                    "ADR-074: expected the predictions directory at %s and it is not a directory. "
                    "The naming convention has moved; the manifest was NOT written. Fix the "
                    "derivation rather than writing it where a reader will not look.",
                    directory,
                )
                return

            sha_path = artifact.with_suffix(".pt.sha256")
            manifest = EmissionManifest(
                schema_version=SCHEMA_VERSION,
                run_type=run_type,
                model_name=self._model_path.model_name,
                package_version=version("views-hydranet"),
                # ⛔ `ctx.orchestrator.config`, NOT `self.configs`. The latter is a pipeline-core
                # PROPERTY that re-merges five sources on every access, so reading it here would be
                # the re-read ADR-074 section 2.3 forbids — it would agree with the configuration
                # by construction and could never contradict it. `ctx.orchestrator.config` is the
                # dict captured once and handed to the inference engine, so it is the object the
                # computation actually used, including any in-place mutation a driver made after
                # setup (`roster_arm_entry.py` mutates `forecast_composition` and `gate_threshold`
                # exactly that way, and those are the most load-bearing keys in these experiments).
                resolved_config=dict(ctx.orchestrator.config),
                # Derived from the targets this run emitted, grouped by the prefix constants the
                # rest of the package uses — not a hardcoded literal, which would agree with
                # nothing and would silently keep omitting any later quantity (ADR-075's body).
                quantities=_quantities_of(ctx.all_targets),
                origins=tuple(ctx.orchestrator.emitted_origins),
                # ADR-077 section 2.2: the extent this model declares, read without importing its
                # configuration. None if it does not state one — recorded as unknown rather than
                # assumed, because a manifest that filled the blank with the expected answer would
                # be the re-read ADR-074 section 2.3 forbids.
                declared_coverage=declared_coverage_of(self._model_path.model_dir),
                repos=self._repo_provenance(),
                artifact_filename=artifact.name,
                artifact_sha256=(sha_path.read_text().strip() if sha_path.exists() else None),
            )
            path = write_manifest(manifest, directory)
            logger.info("ADR-074: emission manifest written to %s", path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            # The expected failures: an unwritable path, a non-serialisable config value, a missing
            # key. The run is finished and its predictions are on disk, so losing the description
            # is not worth discarding it over.
            logger.error("ADR-074: failed to write the emission manifest (%s)", exc)
        except Exception:
            # ⚠️ Anything else is a PROGRAMMING error, and it must not be reported in the same
            # shape as an environmental one. Four tests used to pass while logging an
            # `AttributeError` through the old blanket clause, in a line indistinguishable from a
            # genuine serialisation failure and with no traceback — so the feature could have been
            # failing on every call and the suite would have stayed green.
            logger.error(
                "ADR-074: the emission manifest writer raised an UNEXPECTED exception — this is a "
                "defect in the writer, not an environmental failure. The run is not failed, but "
                "this must not be ignored.",
                exc_info=True,
            )

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> "dict[str, list[PredictionFrame]]":
        """Orchestrates rolling-origin evaluation via specialized component."""
        model = self._load_model_artifact(artifact_name)
        ctx = self._setup_evaluation(eval_type, model)
        handler, scaler, origins, all_targets, orchestrator = ctx

        list_pf_dicts = orchestrator.generate_prediction_frames(
            handler, scaler, origins=origins, all_targets=all_targets
        )
        result: dict[str, list[PredictionFrame]] = {
            t: [d[t] for d in list_pf_dicts] for t in all_targets
        }
        logger.info(
            f"✅ HydranetManager: Evaluation complete — "
            f"{len(list_pf_dicts)} origin(s), {len(result)} targets."
        )
        # ⛔ No ADR-074 manifest here, deliberately. This batch path is unreachable in production:
        # pipeline-core only calls it on the DataFrame branch, and every HydraNet declares
        # `prediction_format: prediction_frame`, so the streaming override below is always taken.
        # That branch also never creates `predictions_{run_type}_{ts}`, so a manifest written from
        # here would have nowhere to go. A call that cannot run is worse than no call: it looks
        # wired.
        return result

    def _evaluate_model_artifact_streaming(
        self,
        eval_type: str,
        artifact_name: str | None,
        origin_sink: Callable[[int, dict[str, "PredictionFrame"]], None],
    ) -> None:
        """
        Override of ForecastingModelManager._evaluate_model_artifact_streaming().

        Streams rolling-origin evaluation: calls origin_sink(i, pf_dict) exactly
        once per origin, in sequential 0-based order (i = 0, 1, ..., M-1).
        Each pf_dict maps every target name to its PredictionFrame for that origin.
        The sink is responsible for persisting and freeing pf_dict before returning.

        Memory advantage:
            Batch path: M × T PredictionFrames in RAM simultaneously
            Streaming:  1 × T PredictionFrames in RAM at any moment
            Reduction:  M× (e.g. 13× at pgm scale with S=32)
        """
        model = self._load_model_artifact(artifact_name)
        ctx = self._setup_evaluation(eval_type, model)
        ctx.orchestrator.generate_prediction_frames_streaming(
            ctx.handler,
            ctx.scaler,
            origins=ctx.origins,
            all_targets=ctx.all_targets,
            origin_sink=origin_sink,
        )
        # ADR-074. The writer reads the run type itself, inside its own try — see its docstring.
        self._write_emission_manifest(ctx, artifact_name)

    def _forecast_model_artifact(
        self, artifact_name: str | None = None
    ) -> "dict[str, PredictionFrame]":
        """Generates operational forecasts."""
        model = self._load_model_artifact(artifact_name)
        ctx = self._setup_evaluation("forecasting", model, forecast=True)

        list_pf_dicts = ctx.orchestrator.generate_prediction_frames(
            ctx.handler, ctx.scaler, origins=ctx.origins, all_targets=ctx.all_targets
        )
        result: dict[str, PredictionFrame] = {t: list_pf_dicts[0][t] for t in ctx.all_targets}
        logger.info(f"✅ HydranetManager: Forecast complete — {len(result)} targets.")
        # ⛔ No ADR-074 manifest on this path, and it is a GAP rather than an oversight — recorded
        # in ADR-074 section 5. `predictions_forecasting_{ts}` is created by the pipeline's
        # `save_pf` only after this method returns, so a write here would leave a complete-looking
        # provenance record of rows that a later failure means were never written. There is no
        # post-save hook to attach it to without changing pipeline-core.
        return result

    def _evaluate_sweep(self, eval_type: str, model: Any) -> "dict[str, list[PredictionFrame]]":
        """Evaluate a trained model during a WandB sweep iteration."""
        ctx = self._setup_evaluation(eval_type, model)
        list_pf_dicts = ctx.orchestrator.generate_prediction_frames(
            ctx.handler, ctx.scaler, origins=ctx.origins, all_targets=ctx.all_targets
        )
        return {t: [d[t] for d in list_pf_dicts] for t in ctx.all_targets}
