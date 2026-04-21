"""
Falsification audit: cradle-to-grave data correctness claim (2026-04-19).

Claim: "We have proven in strict and programmatical terms that data can
correctly pass through this model from cradle to grave for all partitions:
calibration, validation, and forecasting."

Verdict: FALSIFIED (4 hard, 2 soft falsifications).

Each test below encodes a specific falsifying observation. Tests are RED
stubs — they FAIL to document the gap. When the gap is fixed, flip to GREEN.
"""


# ── F3-01: HARD — Validation partition has no manager-level integration test ─


class TestF3_01_ValidationPartitionUntested:
    """No test exercises the HydraNet manager pipeline with run_type='validation'."""

    def test_validation_partition_exercised_in_integration_tests(self):
        """
        F3-01: All manager-level integration tests use run_type='calibration'.
        The validation partition — with its own partition boundaries and
        rolling-origin window calculations — is never exercised.

        The claim says 'all partitions', but validation is absent from
        every test that calls _evaluate_model_artifact or _setup_evaluation.
        """
        from pathlib import Path

        test_dir = Path(__file__).parent
        integration_files = [
            "test_pipeline_integration",
            "test_manager_integration_local",
            "test_audit_manager_eval_survival",
            "test_manager_memory_hygiene",
            "test_lifecycle_integration",
        ]

        found_validation = False
        for mod_name in integration_files:
            mod_path = test_dir / f"{mod_name}.py"
            if mod_path.exists():
                source = mod_path.read_text()
                if (
                    "run_type.*validation" in source
                    or '"validation"' in source
                    and "_evaluate_model_artifact" in source
                ):
                    found_validation = True
                    break

        assert found_validation, (
            "HARD FALSIFICATION F3-01: No manager integration test exercises "
            "run_type='validation'. The validation partition is untested."
        )


# ── F3-02: HARD — Primary E2E test has zero numeric value assertions ────────


class TestF3_02_AssertionQualityGap:
    """test_pipeline_integration.py checks structure, not correctness."""

    def test_pipeline_integration_has_value_assertions(self):
        """
        F3-02: test_pipeline_integration.py is the primary end-to-end test.
        It contains 17 assertions: type checks, key presence, shape checks,
        and len(dict)==6. Zero assertions verify that computed prediction
        values are numerically correct.

        'Proven in strict and programmatical terms' requires asserting that
        outputs are correct, not merely that the pipeline produces output.
        """
        from pathlib import Path

        source = (Path(__file__).parent / "test_pipeline_integration.py").read_text()

        value_keywords = [
            "assert_allclose",
            "assert_array_equal",
            "assert_almost_equal",
            "y_pred[",
            "isfinite",
        ]
        has_value_check = any(kw in source for kw in value_keywords)

        assert has_value_check, (
            "HARD FALSIFICATION F3-02: test_pipeline_integration.py has zero "
            "numeric value assertions on prediction outputs. Structure checks "
            "prove the pipeline runs, not that it produces correct results."
        )


# ── F3-03: HARD — No cradle-to-grave lifecycle test ─────────────────────────


class TestF3_03_NoCradleToGraveTest:
    """No single test trains a model then uses it for inference."""

    def test_single_test_covers_train_and_infer(self):
        """
        F3-03: 'Cradle to grave' means train → eval → forecast in one test.
        No test file imports both training functions (make, training_loop)
        and evaluation functions (_evaluate_model_artifact,
        generate_prediction_frames) as real implementations.

        The lifecycle is tested in disconnected fragments that never verify
        a trained model produces correct predictions.
        """
        from pathlib import Path

        test_dir = Path(__file__).parent
        train_imports = {"training_loop", "train_model_artifact", "make"}
        infer_imports = {
            "_evaluate_model_artifact",
            "_forecast_model_artifact",
            "generate_prediction_frames",
            "InferenceOrchestrator",
            "HydranetManager",
        }

        found_unified = False
        for test_file in test_dir.glob("test_*.py"):
            source = test_file.read_text()
            has_train = any(t in source for t in train_imports)
            has_infer = any(i in source for i in infer_imports)

            if has_train and has_infer:
                # Exclude files that only mock these (e.g. test_manager_memory_hygiene)
                lines = source.splitlines()
                real_train = any(
                    "from views_hydranet" in ln and any(t in ln for t in train_imports)
                    for ln in lines
                    if not ln.strip().startswith("#")
                    and "patch" not in ln
                    and "mock" not in ln.lower()
                )
                real_infer = any(
                    "from views_hydranet" in ln and any(i in ln for i in infer_imports)
                    for ln in lines
                    if not ln.strip().startswith("#")
                    and "patch" not in ln
                    and "mock" not in ln.lower()
                )
                if real_train and real_infer:
                    found_unified = True
                    break

        assert found_unified, (
            "HARD FALSIFICATION F3-03: No single test covers both training "
            "and inference. The 'cradle to grave' lifecycle is fragmented — "
            "no test verifies a trained model produces correct predictions."
        )


# ── F3-04: HARD — Partition boundary mechanism entirely untested ─────────────


class TestF3_04_PartitionBoundaryUntested:
    """_partition_dict lookup is never exercised in any test."""

    def test_partition_dict_tested_in_any_test(self):
        """
        F3-04: _setup_evaluation (hydranet_manager.py:286) reads
        _partition_dict[run_type] to get calibration/validation boundaries.
        No test ever sets _partition_dict on a manager instance.

        test_data_pipeline_extraction.py tests partition_bound=5 directly,
        bypassing the _partition_dict lookup. The mechanism that distinguishes
        calibration from validation partition slicing is untested.
        """
        from pathlib import Path

        test_dir = Path(__file__).parent
        found = False
        for test_file in test_dir.glob("test_*.py"):
            if test_file.name == Path(__file__).name:
                continue
            source = test_file.read_text()
            if "_partition_dict" in source:
                found = True
                break

        assert found, (
            "HARD FALSIFICATION F3-04: No test sets or mocks _partition_dict. "
            "The mechanism distinguishing calibration from validation partition "
            "slicing is entirely untested."
        )


# ── F3-05: SOFT — Identity column values almost never verified ───────────────


class TestF3_05_IdentityValueGap:
    """Only 1 of 19 identity assertions checks an actual value."""

    def test_unit_identity_values_verified_in_any_integration_test(self):
        """
        F3-05: 8 tests check key presence ('time' in pf.identifiers).
        Only 1 test checks an actual value (identifiers['time'][0] == 124).
        Zero tests verify identifiers['unit'] values anywhere.

        'Correctly pass through' requires verifying that priogrid_gid
        identifiers survive the pipeline with correct values, not just
        that the 'unit' key exists in the output dict.
        """
        from pathlib import Path

        test_dir = Path(__file__).parent
        found_unit_value = False
        for test_file in test_dir.glob("test_*.py"):
            if test_file.name == Path(__file__).name:
                continue
            source = test_file.read_text()
            has_unit_index = 'identifiers["unit"][' in source or "identifiers['unit'][" in source
            has_unit_value_check = (
                "unit_vals" in source
                and ("assert" in source)
                and ('identifiers["unit"]' in source or "identifiers['unit']" in source)
            )
            if has_unit_index or has_unit_value_check:
                found_unit_value = True
                break

        assert found_unit_value, (
            "SOFT FALSIFICATION F3-05: No test verifies actual values in "
            "identifiers['unit']. priogrid_gid identity preservation is "
            "asserted by key presence only."
        )


# ── F3-06: SOFT — No pipeline-level reproducibility test ─────────────────────


class TestF3_06_PipelineReproducibility:
    """Component seeding tested, but pipeline-level determinism is not."""

    def test_inference_pipeline_reproducibility_tested(self):
        """
        F3-06: ReproducibilityGate.lock_entropy() is tested for RNG reset.
        VolumeSampler seed reproducibility is tested. But no test runs
        the inference pipeline (generate_prediction_frames or
        _evaluate_model_artifact) twice with the same seeds and compares
        PredictionFrame outputs for equality.

        'Strict proof' of correct data flow requires demonstrating that
        the same inputs produce identical outputs — not just that seeds
        are set correctly at the component level.
        """
        import re
        from pathlib import Path

        test_dir = Path(__file__).parent
        found = False
        for test_file in test_dir.glob("test_*.py"):
            if test_file.name == Path(__file__).name:
                continue
            source = test_file.read_text()

            # Look for explicit two-run-and-compare pattern:
            # result_1 = ...; result_2 = ...; assert_allclose(result_1, result_2)
            has_repro_pattern = bool(
                re.search(
                    r"(run_1|result_1|first_run|pf_1|output_1).*"
                    r"(run_2|result_2|second_run|pf_2|output_2).*"
                    r"(assert_array_equal|assert_allclose|array_equal)",
                    source,
                    re.DOTALL,
                )
            )
            has_repro_keyword = "reproducib" in source.lower() and (
                "generate_prediction_frames" in source or "_evaluate_model_artifact" in source
            )
            if has_repro_pattern or has_repro_keyword:
                found = True
                break

        assert found, (
            "SOFT FALSIFICATION F3-06: No test runs the inference pipeline "
            "twice with same seeds and compares outputs. Pipeline-level "
            "reproducibility is assumed, not proven."
        )
