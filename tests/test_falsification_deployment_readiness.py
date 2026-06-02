"""
Falsification audit: deployment readiness claim (2026-04-19).

Claim: "views-hydranet is ready for real-world deployment in high-stakes settings."
Verdict: FALSIFIED (5 hard, 1 soft falsification).

Each test below encodes a specific falsifying observation. Tests are RED stubs —
they FAIL to document the gap. When the gap is fixed, flip the test to GREEN.
"""


# ── F2-01: HARD — Eval/forecast path never locks entropy ──────────────────


class TestF2_01_ReproducibilityGap:
    """lock_entropy() is training-only. Eval and forecast never seed."""

    def test_lock_entropy_called_in_eval_path(self):
        """
        F2-01a: _setup_evaluation() (called by both eval and forecast) must
        call lock_entropy() to ensure reproducible outputs.
        """
        import inspect

        from views_hydranet.manager import hydranet_manager

        source = inspect.getsource(hydranet_manager.HydranetManager._setup_evaluation)
        assert "lock_entropy" in source, (
            "HARD FALSIFICATION F2-01a: _setup_evaluation() never calls "
            "lock_entropy(). Stochastic eval is non-reproducible."
        )

    def test_lock_entropy_called_in_forecast_path(self):
        """
        F2-01b: _setup_evaluation() (shared by forecast path) must call
        lock_entropy() to ensure reproducible forecasts.
        """
        import inspect

        from views_hydranet.manager import hydranet_manager

        source = inspect.getsource(hydranet_manager.HydranetManager._setup_evaluation)
        assert "lock_entropy" in source, (
            "HARD FALSIFICATION F2-01b: _setup_evaluation() never calls "
            "lock_entropy(). Production forecasts are non-reproducible."
        )

    def test_deterministic_algorithms_never_set(self):
        """
        F2-01c: torch.use_deterministic_algorithms() is never called anywhere.
        GPU operations may use non-deterministic algorithms silently.
        """
        import importlib

        source_file = importlib.import_module(
            "views_hydranet.infrastructure.reproducibility_gate"
        ).__file__
        with open(source_file) as f:
            source = f.read()

        assert "use_deterministic_algorithms" in source, (
            "HARD FALSIFICATION F2-01c: ReproducibilityGate never calls "
            "torch.use_deterministic_algorithms(). GPU ops are non-deterministic."
        )


# ── F2-02: HARD — IntegrityGuardian absent from inference path ─────────────


class TestF2_02_IntegrityGuardianInferenceGap:
    """IntegrityGuardian monitors training only. Inference is unguarded."""

    def test_integrity_guardian_not_in_inference_orchestrator(self):
        """
        F2-02a: InferenceOrchestrator never calls IntegrityGuardian.monitor().
        A model producing predictions > 10K during eval/forecast goes undetected.
        Training has this guard (training_engine.py:499); inference does not.

        Falsifies: "ready for high-stakes deployment" requires numerical safety
        on the output path, not just the training path.
        """
        import inspect

        from views_hydranet.utils import inference_orchestrator

        source = inspect.getsource(inference_orchestrator.InferenceOrchestrator)
        assert "IntegrityGuardian" in source, (
            "HARD FALSIFICATION F2-02a: InferenceOrchestrator has zero "
            "IntegrityGuardian checks. Extreme predictions pass silently."
        )


# ── F2-04: HARD — No CI/CD pipeline ───────────────────────────────────────


class TestF2_04_NoCICDPipeline:
    """No automated quality gate exists between code and deployment."""

    def test_github_actions_workflow_exists(self):
        """
        F2-04: No .github/workflows/ directory exists. Tests, linting, and
        type checking run only when a developer remembers to invoke them.
        A system "ready for real-world deployment in high-stakes settings"
        must have automated gates preventing untested code from shipping.
        """
        from pathlib import Path

        workflows = Path(__file__).parent.parent / ".github" / "workflows"
        assert workflows.is_dir() and any(workflows.iterdir()), (
            "HARD FALSIFICATION F2-04: No CI/CD pipeline. Zero automated "
            "quality gates between code change and deployment."
        )


# ── F2-05: HARD — NaN propagation through forecast path ───────────────────


class TestF2_05_NaNPropagation:
    """Model explosion must raise, not propagate NaN silently to final output."""

    def test_predict_raises_on_nonfinite(self):
        """
        F2-05a: predict() must raise RuntimeError on non-finite predictions,
        not return NaN arrays that silently propagate to output.
        """
        import inspect

        from views_hydranet.utils import hydranet_inference

        source = inspect.getsource(hydranet_inference.HydraNetInference.predict)
        assert "raise RuntimeError" in source, (
            "HARD FALSIFICATION F2-05a: predict() does not raise on non-finite "
            "predictions. Model explosions propagate silently."
        )
        assert "np.full" not in source or "np.nan" not in source, (
            "HARD FALSIFICATION F2-05a: predict() still returns NaN arrays "
            "instead of raising on model explosion."
        )

    def test_integrity_guardian_in_inference_pipeline(self):
        """
        F2-05b: InferenceOrchestrator must call IntegrityGuardian.monitor_numpy()
        on posterior predictions before downstream processing.
        """
        import inspect

        from views_hydranet.utils import inference_orchestrator

        source = inspect.getsource(inference_orchestrator.InferenceOrchestrator)
        assert "IntegrityGuardian" in source, (
            "HARD FALSIFICATION F2-05b: InferenceOrchestrator has no "
            "IntegrityGuardian check on posterior predictions."
        )


# ── F2-06: SOFT — Unvalidated config fields used at runtime ───────────────


class TestF2_06_ConfigValidationGap:
    """Config fields used at runtime bypass HydraNetConfig validation."""

    def test_sweep_field_in_config_schema(self):
        """
        F2-06: The 'sweep' field controls whether model artifacts are saved
        (hydranet_manager.py: `is_sweep = self.configs.get("sweep", False)`).
        It is NOT in HydraNetConfig's schema — accepts any truthy value
        including strings, integers, or objects. A typo like sweep="true"
        (string) would be truthy and skip artifact saving silently.
        """
        from views_hydranet.utils.config_initializer import HydraNetConfig

        fields = HydraNetConfig.model_fields
        assert "sweep" in fields, (
            "SOFT FALSIFICATION F2-06: 'sweep' field controls artifact saving "
            "but is not in HydraNetConfig schema. Accepts any truthy value."
        )


# ── F2-07: HARD — Arbitrary code execution via torch.load ─────────────────


class TestF2_07_CheckpointIntegrity:
    """Model artifacts loaded with weights_only=False — pickle ACE risk."""

    def test_model_artifact_fetcher_uses_weights_only_true(self):
        """
        F2-07a: ModelArtifactFetcher must use weights_only=True for the
        primary (state_dict) loading path. The legacy fallback may use
        weights_only=False but must log a deprecation warning.

        C-09 resolution: dual-mode loader — new artifacts use
        weights_only=True, legacy full-object uses weights_only=False
        with deprecation warning and re-save guidance.
        """
        import inspect

        from views_hydranet.utils import model_artifact_fetcher

        source = inspect.getsource(
            model_artifact_fetcher.ModelArtifactFetcher.fetch_model_artifact
        )
        assert "weights_only=True" in source, (
            "F2-07a: Primary load path must use weights_only=True."
        )

    def test_model_artifact_has_checksum_verification(self):
        """
        F2-07b: No hash/checksum verification on loaded artifacts.
        A corrupted .pt file (disk error, truncated transfer) loads silently
        and produces garbage predictions.
        """
        import inspect

        from views_hydranet.utils import model_artifact_fetcher

        source = inspect.getsource(
            model_artifact_fetcher.ModelArtifactFetcher.fetch_model_artifact
        )
        has_integrity = any(
            kw in source
            for kw in [
                "hash",
                "checksum",
                "sha256",
                "md5",
                "digest",
                "verify",
            ]
        )
        assert has_integrity, (
            "HARD FALSIFICATION F2-07b: No integrity verification on model "
            "artifacts. Corrupted/tampered files load silently."
        )
