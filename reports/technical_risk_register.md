# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-04-08                           |
| Total Concerns    | 39                                   |
| Open Concerns     | 38                                   |
| Resolved Concerns | 0                                    |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Open Concerns

### C-01: Manager monolith orchestration

| Field | Value |
|-------|-------|
| ID | C-01 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Any change to component wiring or lifecycle ordering |
| Location | `manager/hydranet_manager.py` |

`hydranet_manager.py` imports 12 internal modules and wires all components manually. Any wiring change requires modifying this single 380-line file. Fan-out of 12 — highest in the codebase.

Per Martin (Clean Architecture Ch 26, p.228-232): the Manager correctly acts as the "Main Component" — the dirtiest component that creates everything and hands control to higher-level abstractions. High fan-out is expected for Main. The concern is not dirtiness but *size*: at 380 lines it exceeds a pure wiring role, mixing lifecycle orchestration with component construction. Martin: "Think of Main as a plugin to the application" — it should be replaceable without touching policy.

---

### C-02: Duplicated setup between eval and forecast

| Field | Value |
|-------|-------|
| ID | C-02 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Bug fix applied to one path but not the other |
| Location | `hydranet_manager.py:339-381` vs `218-290` |

`_forecast_model_artifact()` repeats model loading, config init, data pipeline, and orchestrator construction from `_setup_evaluation()` instead of reusing it. Nearly identical 40-line blocks in both methods.

---

### C-03: Architecture hardcodes 3+3 heads

| Field | Value |
|-------|-------|
| ID | C-03 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Adding or removing a regression/classification target |
| Location | `HydraBNrecurrentUnet_06_LSTM4.py:68-167`, `hydranet_manager.py:165-176` |

The `HydraBNUNet06_LSTM4` model has 6 decoder heads physically baked into the class definition. Adding a target requires duplicating ~50 lines of layer definitions + forward() code, plus updating the preflight check. Currently stable (no planned target changes).

Per Martin (Clean Architecture Ch 8, p.87-93): this violates OCP — the architecture is closed to extension. Adding a head requires modifying both `__init__()` and `forward()`. Martin's "hierarchy of protection" (p.91) says the model entity should be the most protected component — but here it's the component most exposed to change if target count evolves.

---

### C-04: Spatial offset arithmetic in VolumeSampler is untested

| Field | Value |
|-------|-------|
| ID | C-04 |
| Tier | 2 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Change to window_dim, spatial_offset, or VolumeHandler.from_df() coordinate logic |
| Location | `volume_sampler.py:91-98` |

`_generate_window()` computes `new_row_offset = p_row + (h_max - dim - r0)` to propagate geographic truth into sub-volumes. No test verifies this round-trip against the parent handler's coordinate system. A bug here would produce spatially misaligned training windows with no downstream error signal.

---

### C-05: Loss/scheduler selection uses unvalidated string codes

| Field | Value |
|-------|-------|
| ID | C-05 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Typo in config `loss_reg` or `loss_class` values |
| Location | `utils.py:42-66, 83-104` |

`choose_loss()` maps `"a"` → MSELoss, `"b"` → ShrinkageLoss with no enum or constant. These magic strings are validated only at runtime, not by Pydantic. A typo produces a clear `ValueError`, but the string codes are opaque.

See also C-38 for the structural OCP violation in these factories. Per Martin (Clean Architecture Ch 8, p.87): the `TRANSFORMS` registry in `config_initializer.py` demonstrates the correct pattern — the factories should follow suit.

---

### C-06: Config returns dict after Pydantic validation

| Field | Value |
|-------|-------|
| ID | C-06 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Downstream code accessing a key that Pydantic doesn't validate (via `extra="allow"`) |
| Location | `config_initializer.py:302-303` |

`ConfigInitializer.get_config()` validates via `HydraNetConfig` then returns `.model_dump()` as a plain dict. All downstream consumers use `config["key"]` or `config.get(key)` without type safety. The `extra = "allow"` setting means unvalidated keys pass through silently. Constrained by parent class (`ForecastingModelManager.configs`) requiring `isinstance(dict)`.

---

### C-07: Training loop lacks explicit per-window memory cleanup

| Field | Value |
|-------|-------|
| ID | C-07 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Large `windows_per_lesson` values or large `window_dim` on GPU |
| Location | `train_model.py:333-376` |

Window VolumeHandlers and tensors from `sampler.get_batch()` accumulate in the inner loop without explicit `del` or `gc.collect()`. For large window counts, this can stress GPU memory. The inference path correctly implements per-origin cleanup; the training path does not.

---

### C-08: North-Up flip symmetry is implicitly coupled

| Field | Value |
|-------|-------|
| ID | C-08 |
| Tier | 2 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Adding a new output path that bypasses `_valid_cell_indices()` |
| Location | `volume_handler.py:214` (from_df flip), `volume_handler.py:483` (_valid_cell_indices flip) |

The volume is flipped North-Up in `from_df()` and must be un-flipped in `_valid_cell_indices()` for output reconstruction. These two operations are implicitly coupled — no assertion or structural test verifies the flip count. A mismatch would produce silently inverted geographic coordinates. Currently guarded indirectly by `test_derivation_parity.py`.

---

### C-09: `torch.save(model)` full-object serialization

| Field | Value |
|-------|-------|
| ID | C-09 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Renaming or moving the `HydraBNUNet06_LSTM4` class |
| Location | `train_model.py:490`, `model_artifact_fetcher.py:94` |

Full model (not `state_dict`) is pickled via `torch.save()`. This couples saved `.pt` artifacts to the exact class definition and module path. Renaming the architecture class or moving it to a different module breaks deserialization of all existing artifacts. `weights_only=False` in load confirms full-object deserialization.

Per Martin (Clean Architecture Ch 32, p.275-278): "Don't marry the framework." The serialized artifact is married to PyTorch's pickle format and the concrete class path — the tightest possible coupling to a framework detail. `state_dict()` serialization would keep PyTorch at arm's length, making the architecture class freely renameable and movable.

---

### C-10: 13 test files require views_pipeline_core

| Field | Value |
|-------|-------|
| ID | C-10 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Developer runs tests in partial environment without pipeline_core |
| Location | `tests/conftest.py:9`, 13 test files |

Tests covering manager integration, PredictionFrame output, and subset symmetry cannot run without `views_pipeline_core`. The conftest gate (280 minimum) only triggers when running the full suite from `tests/`. In partial environments, 270 tests collect and the gate is bypassed, silently missing integration coverage.

Per Martin (Clean Architecture Ch 28, p.243-246): "Design for Testability" — tests should not depend on volatile things. The test suite's dependence on the framework layer (`views_pipeline_core`) at import time makes 13 test files fragile. Martin's "Testing API" (p.245) principle: decouple test structure from application structure. See also C-27 for the specific import chain that causes this.

---

### C-11: Direct _metadata access bypasses encapsulation

| Field | Value |
|-------|-------|
| ID | C-11 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Change to VolumeMetadata field names or structure |
| Location | `train_model.py:183`, `curriculum.py:45`, `feature_scaler.py:245-255`, `hydranet_inference.py:381` |

Multiple modules reach into `VolumeHandler._metadata.feature_cols`, `._metadata.identity_cols`, etc. instead of using properties. This couples them to the internal dataclass structure. Mitigated by `VolumeMetadata` being a frozen dataclass (structurally stable), but violates encapsulation convention.

---

### C-12: `wandb` imported unconditionally at module level

| Field | Value |
|-------|-------|
| ID | C-12 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Running in an environment without `wandb` installed |
| Location | `utils.py:9` |

`import wandb` runs unconditionally at module load even when W&B is not configured. Mitigated by `if wandb.run is not None` guard in `train_log()`, but the import itself would fail in environments without the package. Currently a soft dependency since `wandb` is not in `pyproject.toml` required deps.

---

### C-13: `_permute()` mutates VolumeHandler in-place

| Field | Value |
|-------|-------|
| ID | C-13 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Calling `_permute()` on a shared VolumeHandler reference |
| Location | `volume_handler.py:615-635` |

Unlike transformation methods that return new VolumeHandlers (`slice_time`, `collapse_to_point`), `_permute()` modifies `self._data` and `self._metadata` in-place. Inconsistent with the immutable-by-convention pattern. Currently used only in geometric tests, not in production paths.

---

### C-14: `flip()` mutates VolumeHandler in-place

| Field | Value |
|-------|-------|
| ID | C-14 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Calling `flip()` on a VolumeHandler that is referenced elsewhere |
| Location | `volume_handler.py:637-653` |

Like `_permute()`, `flip()` modifies `self._data` in-place rather than returning a new VolumeHandler. Used in the training augmentation path (`train_model.train()`). Safe in practice because the augmented handler is a per-window copy from `VolumeSampler`, but the mutation pattern is inconsistent with the immutable-by-convention design.

Per Martin (Clean Architecture Ch 6, p.70-76): "Segregation of Mutability" — separate the application into immutable (pure functional) and mutable (transactional) components. `VolumeMetadata` is correctly immutable (`frozen=True`). But `flip()` and `_permute()` break the segregation by mutating `_data` in-place. Martin would say: these are the "transactional memory" components that should be explicitly marked as mutable, or refactored to return new instances.

---

### C-15: `training_loop()` has 4+ responsibilities in 175 lines

| Field | Value |
|-------|-------|
| ID | C-15 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Modifying lesson orchestration, diagnostic biopsy, gradient clipping, or forensic finalization |
| Location | `train_model.py:279-453` |

`training_loop()` mixes lesson orchestration, gradient accumulation/clipping, diagnostic biopsy generation, forensic auditor finalization, and progress bar management in one function. Violates SRP — changes to diagnostic output require touching the same function that controls optimization.

Per Martin (Clean Architecture Ch 7, p.80-86): the function serves at least two actors — the data scientist (lesson/window/optimization strategy) and the platform engineer (diagnostic/forensic reporting). Martin's "Symptom 2: Merges" (p.83) applies: two people changing the same function for different reasons creates merge risk. The data scientist tuning gradient clipping and the platform engineer adding a new diagnostic biopsy should never collide.

---

### C-16: `visual_diagnostics.py` catch-all exception handlers hide bugs

| Field | Value |
|-------|-------|
| ID | C-16 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Bug in any `biopsy_*` method's plotting logic |
| Location | `visual_diagnostics.py:129-133` (and similar in other biopsy methods) |

All `biopsy_*` methods wrap their body in `try/except Exception` and log a warning on failure. If diagnostic code has a bug, it silently produces no plot with no test failure. The file is 985 lines with 12+ biopsy methods. Tests only verify the `active=True/False` toggle, not plot correctness or exception-free execution.

---

### C-17: `train()` function has 13 parameters

| Field | Value |
|-------|-------|
| ID | C-17 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Adding a new training feature that requires another parameter |
| Location | `train_model.py:150-163` |

`train()` takes model, optimizer, scheduler, criterion_reg, criterion_class, multitaskloss_instance, sample_handler, config, device, pbar, viz, stage_label, and forensics. Wide interface makes the function hard to call correctly and easy to wire incorrectly. Could be reduced by bundling training context into a dataclass.

---

### C-18: No end-to-end training smoke test

| Field | Value |
|-------|-------|
| ID | C-18 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Refactoring the training loop, changing config keys, or modifying lesson/window/sequence wiring |
| Location | `train_model.py:279-497` (entire training path) |

The training loop is the single most critical code path and has no automated end-to-end test. `test_train_loop.py` tests `_process_sequence()` in isolation. A wiring bug (wrong argument order, missing config key, changed return type) in the lesson→window→optimize chain won't be caught until someone runs a full training job (hours on GPU).

---

### C-19: `priogrid_gid > 0` validity assumption undocumented at ingestion

| Field | Value |
|-------|-------|
| ID | C-19 |
| Tier | 4 |
| Source | expert-review (2026-04-08) |
| Trigger | Upstream data source assigning `priogrid_gid == 0` to a valid cell |
| Location | `volume_handler.py:505` |

`_valid_cell_indices()` uses `mask = p_data[:, :, :, pg_idx] > 0` to identify valid cells. If any legitimate grid cell has `priogrid_gid == 0`, it is silently dropped from output. This assumption is not enforced or documented at ingestion (`DataSniffer`, `DataFetcher`). In practice, PRIO-GRID assigns GIDs starting from 1, but this is domain knowledge not codified in the system.

---

### C-20: Autoregressive inference has no soft magnitude guard

| Field | Value |
|-------|-------|
| ID | C-20 |
| Tier | 4 |
| Source | expert-review (2026-04-08) |
| Trigger | Model producing slightly incorrect predictions that compound over 36 autoregressive steps |
| Location | `hydranet_inference.py:287-288` |

`t0_autoreg = t1_pred.detach()` feeds model predictions back as input. `IntegrityGuardian` catches NaN/Inf and its hard ceiling (`> 10000`) catches extreme explosion, but gradual magnitude drift (e.g., predictions growing from 2 to 200 over 36 steps) goes undetected. No soft warning or clipping on autoregressive feedback inputs.

---

### C-21: Bare `except Exception` swallows errors in inference and diagnostics

| Field | Value |
|-------|-------|
| ID | C-21 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Corrupt handler data or unexpected type in diagnostic/time-index extraction |
| Location | `hydranet_inference.py:391`, `train_model.py:214`, `visual_diagnostics.py:129,181,214,301,359,484,593,663,771` |

11 locations use bare `except Exception` across 3 files. `hydranet_inference.py:391` catches all exceptions during time index extraction and continues silently. `train_model.py:214` swallows exceptions during diagnostic time index extraction. `visual_diagnostics.py` has 9 catch-all handlers across all `biopsy_*` methods. Should use specific exception types and log exception details.

---

### C-22: `cast(Any, model)` at 7+ call sites bypasses type safety

| Field | Value |
|-------|-------|
| ID | C-22 |
| Tier | 4 |
| Source | expert-review (2026-04-08) |
| Trigger | Changing the model's `forward()` return signature |
| Location | `train_model.py:105,199,242`, `hydranet_inference.py:122,137,147,151,157,225,257` |

The model's return type is untyped; 7+ call sites use `cast(Any, model)(input, h)` to suppress type checking. The actual contract (returns `(out_reg, out_class, h)` tuple) is invisible at every call site. A `Protocol` type would make the interface explicit and catch signature changes statically.

Per Martin (Clean Architecture Ch 11, p.104-108): DIP says "depend on abstractions, not on concretions." Every `cast(Any, model)` call is a concrete dependency on the model's undeclared interface. Martin's coding practice (p.105): "Don't refer to volatile concrete classes." A `Protocol` defining `__call__(x, h) -> (Tensor, Tensor, Tensor)` would invert this dependency, making the contract explicit and verifiable at type-check time.

---

## Disagreements

### C-23: `extrapolate_time()` has no direct unit test

| Field | Value |
|-------|-------|
| ID | C-23 |
| Tier | 3 |
| Source | test-review (2026-04-08) |
| Trigger | Change to time increment logic or future scaffold construction |
| Location | `volume_handler.py:573-613` |

`extrapolate_time()` creates future identity scaffolding by repeating the last time step and incrementing time indices. No test verifies temporal continuity, correct time channel incrementing, or shape preservation. Used in the forecast path when predictions extend beyond available history.

---

### C-24: InferenceOrchestrator temporal discontinuity failure mode untested

| Field | Value |
|-------|-------|
| ID | C-24 |
| Tier | 3 |
| Source | test-review (2026-04-08) |
| Trigger | Requesting an origin index beyond the handler's temporal bounds |
| Location | `inference_orchestrator.py:49-114` |

CIC documents "Temporal Discontinuity" as a failure mode — the orchestrator should fail if the requested origin does not exist within the provided history. No test verifies this behavior. Currently guarded implicitly by `VolumeHandler.slice_time()` bounds check, but the orchestrator-level contract is untested.

---

### C-25: Curriculum→Sampler zero-qualified-cells interaction untested

| Field | Value |
|-------|-------|
| ID | C-25 |
| Tier | 3 |
| Source | test-review (2026-04-08) |
| Trigger | Curriculum threshold too high for sparse targets, causing extended random-anchor fallback |
| Location | `curriculum.py:90-107`, `volume_sampler.py:76-81` |

When the curriculum's threshold yields zero qualified cells, `VolumeSampler._generate_window()` falls back to a random spatial anchor. No test verifies this fallback path or that it logs the transition. Extended random-anchor sampling degrades training quality without any error signal.

---

### C-26: VisualDiagnostics plot correctness untested

| Field | Value |
|-------|-------|
| ID | C-26 |
| Tier | 4 |
| Source | test-review (2026-04-08) |
| Trigger | Bug in any `biopsy_*` plotting logic |
| Location | `visual_diagnostics.py` (985 lines, 12+ biopsy methods) |

29 tests verify the `active=True/False` toggle and no-crash behavior, but zero tests verify that generated plots contain correct data, have non-zero file size, or reflect the volume state they claim to show. A plotting bug could produce plausible-looking but incorrect visualizations that mislead operators.

---

### C-27: `train_model.py` import structure blocks local testing

| Field | Value |
|-------|-------|
| ID | C-27 |
| Tier | 3 |
| Source | test-review (2026-04-08) |
| Trigger | Running tests in partial environment without views_pipeline_core |
| Location | `train_model.py:11` (`from views_pipeline_core.managers.model import ModelPathManager`) |

`ModelPathManager` is imported at module level but only used in `train_model_artifact()`. This cascades to block testing of `_process_sequence()`, `train()`, and `training_loop()` — the 3 most testable functions in the file. 12 tests currently fail due to this import chain.

---

### C-28: CIC test file references are stale

| Field | Value |
|-------|-------|
| ID | C-28 |
| Tier | 4 |
| Source | test-review (2026-04-08) |
| Trigger | CIC review or audit referencing non-existent test files |
| Location | `docs/CICs/HydranetManager.md` (Section 10), `docs/CICs/ConfigInitializer.md` (Section 10) |

Several CICs reference test files that no longer exist: `legacy_tests/test_manager_smoke.py`, `legacy_tests/test_manager_robustness.py`, `tests/test_red_team_the_abyss.py`, `tests/test_config_initializer.py`. These files were deleted during dead code cleanup but the CIC test alignment sections were not updated.

---

### C-29: Plausible misconfiguration scenarios untested

| Field | Value |
|-------|-------|
| ID | C-29 |
| Tier | 4 |
| Source | test-review (2026-04-08) |
| Trigger | Human operator providing technically valid but degenerate config values |
| Location | `config_initializer.py` (HydraNetConfig), `volume_sampler.py`, `train_model.py` |

No test verifies system behavior with edge-case configurations that Pydantic accepts but produce degenerate behavior: `window_dim=1` (single-pixel patches), `total_lessons=0` (no training), `windows_per_lesson=0` (empty lesson), `learning_rate=1e-20` (effectively zero). These are plausible human errors that pass validation but produce silent quality degradation.

---

### C-30: ModelArtifactFetcher has minimal test coverage

| Field | Value |
|-------|-------|
| ID | C-30 |
| Tier | 4 |
| Source | test-review (2026-04-08) |
| Trigger | Change to artifact loading, device placement, or timestamp extraction logic |
| Location | `model_artifact_fetcher.py`, `test_model_artifact_fetcher.py` (3 tests) |

Only 3 tests exist: happy path with latest artifact, happy path with specific artifact, and missing file error. No tests for timestamp extraction edge cases, device placement verification, or the `add_config` callback behavior.

---

### C-31: ADR-008 log-before-raise violations in 4 files

| Field | Value |
|-------|-------|
| ID | C-31 |
| Tier | 4 |
| Source | falsification-audit (2026-04-08) |
| Trigger | Operational debugging — exceptions raised without preceding log entry |
| Location | `training_forensics.py:79,143,169`, `config_initializer.py:265`, `volume_handler.py:285,308`, `mtloss.py:46` |

7 `raise` statements without preceding `logger.error()` violate ADR-008 (Observability and Explicit Failure). The exceptions have clear messages, so this is a logging hygiene issue rather than a silent failure risk. All are in error paths that produce descriptive error messages.

---

### C-32: VolumeSampler CIC failure modes untested and unregistered

| Field | Value |
|-------|-------|
| ID | C-32 |
| Tier | 4 |
| Source | falsification-audit (2026-04-08) |
| Trigger | Code change to `_generate_window()` extraction bounds or target resolution |
| Location | `volume_sampler.py:55-70` (ledger), `volume_sampler.py:84-88` (bounds) |

VolumeSampler CIC Section 6 declares "Geometric Overflow" (extraction outside bounds) and "Ledger Inconsistency" (target missing from volume) as failure modes. Both are handled in code (`np.clip` for bounds, `ValueError` for missing target) but have no test coverage and were not captured by the test-review.

---

### C-33: InferenceOrchestrator "Sequence Violation" failure mode untested

| Field | Value |
|-------|-------|
| ID | C-33 |
| Tier | 4 |
| Source | falsification-audit (2026-04-08) |
| Trigger | Bypassing `_run_inference_pipeline()` step order |
| Location | `inference_orchestrator.py:49-114` |

InferenceOrchestrator CIC Section 6 declares "Sequence Violation" as a failure mode — the system should raise if the ADR 039 step order (Predict → Align → Wrap → Invert → Collapse) is bypassed. In practice, the sequence is enforced by method composition (each step feeds the next), so bypass is unlikely. But the CIC promise is untested.

---

### C-34: `train_model.py:214` bare `except Exception` missed by C-21

| Field | Value |
|-------|-------|
| ID | C-34 |
| Tier | 4 |
| Source | falsification-audit (2026-04-08) |
| Trigger | Diagnostic time index extraction fails silently during training |
| Location | `train_model.py:214` |

The `train()` function has a bare `except Exception: pass` at line 214 for time index extraction during diagnostic biopsy. If `sample_handler.channel_map` is corrupt, the training continues without diagnostic data and no warning is logged. This location was missed by C-21's original enumeration (now corrected).

---

### C-35: `utils/` package violates Common Closure and Screaming Architecture

| Field | Value |
|-------|-------|
| ID | C-35 |
| Tier | 3 |
| Source | clean-architecture-review (2026-04-08) |
| Trigger | Adding a new module — unclear where it belongs; changing a training component forces retest of unrelated data pipeline tests |
| Location | `views_hydranet/utils/` (20 of 25 source files) |

The `utils/` package contains 20 files spanning 5 distinct domains: data pipeline (fetcher, sniffer, scaler, handler), training strategy (curriculum, sampler, forensics), inference (orchestrator, inference engine), observability (diagnostics, logging, guardian), and configuration. A single generic package name for 80% of the codebase.

Per Martin (Clean Architecture Ch 13, p.117-123): violates CCP (Common Closure Principle) — classes that change for different reasons are packaged together. A training strategy change and a data pipeline change both touch `utils/`. Per Martin (Ch 21, p.199-202): violates Screaming Architecture — the top-level structure should scream "conflict forecasting system," not "utilities." The directory should say `data_pipeline/`, `training/`, `inference/`, `observability/` — not `utils/`.

Per Martin (Ch 13, p.120-121): also violates CRP (Common Reuse Principle) — importing `IntegrityGuardian` (pure torch, no pandas) from `utils/` transitively exposes the consumer to `volume_handler`'s pandas/torch/pipeline_core dependency tree.

---

### C-36: VolumeHandler violates Interface Segregation Principle

| Field | Value |
|-------|-------|
| ID | C-36 |
| Tier | 3 |
| Source | clean-architecture-review (2026-04-08) |
| Trigger | Any consumer of VolumeHandler needing to understand methods irrelevant to its use case |
| Location | `volume_handler.py` (780 lines, 20+ methods, 9 dependents) |

VolumeHandler exposes a single monolithic interface to all consumers. The training loop uses `to_pytorch()`, `flip()`, `channel_map`. The inference path uses `wrap_predictions()`, `to_evaluation_pf()`, `slice_time()`. The data pipeline uses `from_df()`. Each consumer depends on the full 780-line interface but uses only a subset.

Per Martin (Clean Architecture Ch 10, p.100-103): ISP says "avoid depending on things you don't use." Each consumer is forced to depend on methods, imports, and complexity it never invokes. Martin (p.102): "depending on something that carries baggage that you don't need can cause you troubles that you didn't expect." The training path doesn't need `to_evaluation_pf()` but transitively depends on `PredictionFrame` because of it.

---

### C-37: VolumeHandler in SAP "Zone of Pain" — stable but not abstract

| Field | Value |
|-------|-------|
| ID | C-37 |
| Tier | 4 |
| Source | clean-architecture-review (2026-04-08) |
| Trigger | Need to provide an alternative VolumeHandler implementation (e.g., lazy-loading, GPU-resident) |
| Location | `volume_handler.py` — fan-in=9, fan-out=2 |

VolumeHandler has Instability I = 2/(9+2) ≈ 0.18 — highly stable. But it is entirely concrete — no abstract base class, no Protocol, no interface definition. Per Martin (Clean Architecture Ch 14, p.139-143): SAP says "a component should be as abstract as it is stable." A component with high stability and low abstractness sits in the "Zone of Pain" — it's painful to change because many depend on it, and there's no abstraction to insulate them from change.

Currently tolerable because VolumeHandler's interface is mature and rarely changes. Would become painful if a second volume carrier type were needed (e.g., lazy-loading for very large grids, or GPU-resident tensors for inference).

---

### C-38: Factory functions closed to extension (OCP violation)

| Field | Value |
|-------|-------|
| ID | C-38 |
| Tier | 4 |
| Source | clean-architecture-review (2026-04-08) |
| Trigger | Adding a new model architecture, loss function, or scheduler |
| Location | `utils.py:20-35` (`choose_model`), `utils.py:38-80` (`choose_loss`), `utils.py:83-104` (`choose_scheduler`) |

All three factory functions use `if/elif/else` chains on string config values. Adding a new model requires modifying `choose_model()`. Adding a new loss requires modifying `choose_loss()`. The factories are closed to extension — the opposite of OCP.

Per Martin (Clean Architecture Ch 8, p.87-93): "A software artifact should be open for extension but closed for modification." The `TRANSFORMS` registry in `config_initializer.py` is the correct pattern — a dict of callables that can be extended without modifying existing code. The factories should follow the same pattern: a `MODELS` registry, a `LOSSES` registry.

Note: C-05 already registers the string code opacity; this concern addresses the structural OCP violation.

---

### C-39: VolumeHandler Entity imports Framework type (Dependency Rule violation)

| Field | Value |
|-------|-------|
| ID | C-39 |
| Tier | 4 |
| Source | clean-architecture-review (2026-04-08) |
| Trigger | Change to `PredictionFrame` class in `views_pipeline_core` |
| Location | `volume_handler.py:430` (`from views_pipeline_core.data.prediction_frame import PredictionFrame`) |

VolumeHandler is an Entity-layer component (core data carrier, highest stability). `PredictionFrame` is from the Framework layer (`views_pipeline_core`). The import in `to_evaluation_pf()` violates the Dependency Rule — an inner-circle component depends on an outer-circle type.

Per Martin (Clean Architecture Ch 22, p.203-209): "Source code dependencies must point only inward, toward higher-level policies. Nothing in an inner circle can know anything at all about something in an outer circle." Currently mitigated by lazy import (inside the method body, not at module level), which limits the coupling to runtime rather than import-time. A full fix would extract `to_evaluation_pf()` into an Interface Adapter that imports both VolumeHandler and PredictionFrame.

---

### D-01: VolumeHandler scope — God Object vs Deep Module

| Field | Value |
|-------|-------|
| ID | D-01 |
| Source | expert-review (2026-04-08) |
| Perspectives | Martin (split — SRP Ch 7 p.80: serves 4 actors; ISP Ch 10 p.100: 20+ method interface; SAP Ch 14 p.139: Zone of Pain), Ousterhout (keep — successful deep module hiding complexity), Hickey (partial split — extract PF output path, keep volume ops together) |
| Resolution | Partial split recommended: extract PredictionFrame output path into dedicated assembler, keep volume operations in VolumeHandler. Clean Architecture analysis strengthens the split case via three independent SOLID violations (SRP, ISP, SAP) but Ousterhout's "deep module" counter-argument remains valid for the core volume operations. |

---

### D-02: Architecture extensibility — parameterize vs leave alone

| Field | Value |
|-------|-------|
| ID | D-02 |
| Source | expert-review (2026-04-08) |
| Perspectives | GoF (parameterize — 6 copy-pasted decoder blocks is anti-pattern), Beck/Feathers (leave alone — structural regex test guards against bugs, refactoring invalidates all .pt artifacts) |
| Resolution | Leave as-is. Cost of refactoring (breaking all artifacts) exceeds benefit. Structural test provides adequate safety. |

---

### D-03: Config monolith — complecting vs front-loading validation

| Field | Value |
|-------|-------|
| ID | D-03 |
| Source | expert-review (2026-04-08) |
| Perspectives | Hickey (split — 9 concerns conflated in one model), Ousterhout/Nygard (keep — single validation point, cross-field checksums require all fields visible) |
| Resolution | Keep single config. Cross-field checksum laws depend on simultaneous field access. |

---

## Resolved Concerns

### C-28: CIC test file references are stale — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-28 |
| Resolved | 2026-04-08 |
| Resolution | Updated test alignment sections in HydranetManager.md, HydraNetConfig.md, and ConfigInitializer.md to reference actual test files (test_config_typed.py, test_config_validation.py, test_manager_memory_hygiene.py, etc.) |

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements
- **Sources:** `repo-assimilation`, `expert-review`, `tech-debt-audit`, `falsification-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** `Total Concerns` and `Open Concerns` in the register header are manually maintained — update them whenever a concern is added or resolved
- **Governed by:** ADR-048
