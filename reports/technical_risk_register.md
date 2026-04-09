# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-04-09                           |
| Total Concerns    | 40                                   |
| Open Concerns     | 26                                   |
| Resolved Concerns | 14                                   |

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
| Trigger | When adding or removing a component from Manager's initialization sequence, verify file hasn't exceeded pure-wiring scope |
| Location | `manager/hydranet_manager.py` |

`hydranet_manager.py` imports 12 internal modules and wires all components manually. Any wiring change requires modifying this single 380-line file. Fan-out of 12 — highest in the codebase.

Per Martin (Clean Architecture Ch 26, p.228-232): the Manager correctly acts as the "Main Component" — the dirtiest component that creates everything and hands control to higher-level abstractions. High fan-out is expected for Main. The concern is not dirtiness but *size*: at 380 lines it exceeds a pure wiring role, mixing lifecycle orchestration with component construction. Martin: "Think of Main as a plugin to the application" — it should be replaceable without touching policy.

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
| Trigger | When importing any module from `utils/` in an environment without `wandb`, verify `utils.py` doesn't block the import chain |
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

Unlike transformation methods that return new VolumeHandlers (`slice_time`, `collapse_to_point`), `_permute()` modifies `self._data` and `self._metadata` in-place. Inconsistent with the immutable-by-convention pattern. Currently used only in geometric tests, not in production paths. See also C-14 (same mutation pattern on `flip()`).

---

### C-14: `flip()` mutates VolumeHandler in-place

| Field | Value |
|-------|-------|
| ID | C-14 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Calling `flip()` on a VolumeHandler that is referenced elsewhere |
| Location | `volume_handler.py:637-653` |

Like `_permute()` (C-13), `flip()` modifies `self._data` in-place rather than returning a new VolumeHandler. Used in the training augmentation path (`train_model.train()`). Safe in practice because the augmented handler is a per-window copy from `VolumeSampler`, but the mutation pattern is inconsistent with the immutable-by-convention design.

Per Martin (Clean Architecture Ch 6, p.70-76): "Segregation of Mutability" — separate the application into immutable (pure functional) and mutable (transactional) components. `VolumeMetadata` is correctly immutable (`frozen=True`). But `flip()` and `_permute()` break the segregation by mutating `_data` in-place. Martin would say: these are the "transactional memory" components that should be explicitly marked as mutable, or refactored to return new instances.

---

### C-16: `visual_diagnostics.py` catch-all exception handlers hide bugs

| Field | Value |
|-------|-------|
| ID | C-16 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Bug in any `biopsy_*` method's plotting logic |
| Location | `visual_diagnostics.py:129-133` (and similar in other biopsy methods) |

All `biopsy_*` methods wrap their body in `try/except Exception` and log on failure. If diagnostic code has a bug, it silently produces no plot with no test failure. The file is 985 lines with 12+ biopsy methods. Tests only verify the `active=True/False` toggle, not plot correctness or exception-free execution. Partial fix (2026-04-08): `biopsy_dataframe` catch block upgraded from `logger.warning` to `logger.error` per ADR-008 Section 4 (Fail-Safe constraint). Catch-all pattern itself is ADR-008 compliant (Observability Actors are permitted Fail-Safe). Remaining concern: plot correctness is untested (see also C-26).

---

### C-17: `train()` function has 13 parameters

| Field | Value |
|-------|-------|
| ID | C-17 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | When adding a parameter to `train()`, verify total count and consider bundling into a context dataclass |
| Location | `train_model.py:150-163` |

`train()` takes model, optimizer, scheduler, criterion_reg, criterion_class, multitaskloss_instance, sample_handler, config, device, pbar, viz, stage_label, and forensics. Wide interface makes the function hard to call correctly and easy to wire incorrectly. Could be reduced by bundling training context into a dataclass.

---

### C-19: `priogrid_gid > 0` validity assumption undocumented at ingestion

| Field | Value |
|-------|-------|
| ID | C-19 |
| Tier | 4 |
| Source | expert-review (2026-04-08) |
| Trigger | Upstream data source assigning `priogrid_gid == 0` to a valid cell |
| Location | `volume_handler.py:505` |

`_valid_cell_indices()` uses `mask = p_data[:, :, :, pg_idx] > 0` to identify valid cells. If any legitimate grid cell has `priogrid_gid == 0`, it is silently dropped from output. This assumption is not enforced or documented at ingestion (`DataSniffer`, `DataFetcher`). In practice, PRIO-GRID assigns GIDs starting from 1, but this is domain knowledge not codified in the system. Tier rationale: impact is catastrophic (silent data loss) but trigger probability is near-zero given PRIO-GRID's established numbering convention. Tier 4 reflects expected risk (impact × likelihood), not impact alone.

---

### C-20: Autoregressive inference has no soft magnitude guard

| Field | Value |
|-------|-------|
| ID | C-20 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | When modifying autoregressive feedback in `_run_autoregressive()`, verify that gradual magnitude drift (not just NaN/Inf) is detectable |
| Location | `hydranet_inference.py:287-288` |

`t0_autoreg = t1_pred.detach()` feeds model predictions back as input. `IntegrityGuardian` catches NaN/Inf and its hard ceiling (`> 10000`) catches extreme explosion, but gradual magnitude drift (e.g., predictions growing from 2 to 200 over 36 steps) goes undetected. No soft warning or clipping on autoregressive feedback inputs.

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
| Trigger | When a new module imports VolumeHandler, verify it doesn't transitively pull unused dependencies (e.g., PredictionFrame via the PF output path) |
| Location | `volume_handler.py` (780 lines, 20+ methods, 9 dependents) |

VolumeHandler exposes a single monolithic interface to all consumers. The training loop uses `to_pytorch()`, `flip()`, `channel_map`. The inference path uses `wrap_predictions()`, `to_evaluation_pf()`, `slice_time()`. The data pipeline uses `from_df()`. Each consumer depends on the full 780-line interface but uses only a subset.

Per Martin (Clean Architecture Ch 10, p.100-103): ISP says "avoid depending on things you don't use." Each consumer is forced to depend on methods, imports, and complexity it never invokes. Martin (p.102): "depending on something that carries baggage that you don't need can cause you troubles that you didn't expect." The training path doesn't need `to_evaluation_pf()` but transitively depends on `PredictionFrame` because of it. See also C-37 (SAP Zone of Pain) and D-01 (God Object vs Deep Module disagreement).

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

Currently tolerable because VolumeHandler's interface is mature and rarely changes. Would become painful if a second volume carrier type were needed (e.g., lazy-loading for very large grids, or GPU-resident tensors for inference). See also C-36 (ISP violation) and D-01 (God Object vs Deep Module disagreement).

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

### C-40: `validate_docs.sh` uses GNU-only `grep -oP` (not portable to macOS)

| Field | Value |
|-------|-------|
| ID | C-40 |
| Tier | 4 |
| Source | pr-review (2026-04-08) |
| Trigger | macOS contributor running `bash docs/validate_docs.sh` |
| Location | `docs/validate_docs.sh:60,75` |

Two `grep -oP` calls use Perl-compatible regex (`-P` flag), which requires GNU grep. macOS ships BSD grep, which does not support `-P`. The script will fail with an error on macOS unless the user has GNU grep installed or aliased. Currently low-impact: all known contributors use Linux, the script is a manual governance tool (not CI), and failure is loud and non-destructive. Fix is straightforward: rewrite with `sed -E` or `awk` for POSIX portability. Script was copied verbatim from base_docs template — fix should be upstreamed there as well.

---

### C-41: Falsification test stubs will fail in CI if not excluded

| Field | Value |
|-------|-------|
| ID | C-41 |
| Tier | 4 |
| Source | pr-review (2026-04-08) |
| Trigger | CI pipeline collecting `tests/test_falsification_all_risks_identified.py` without explicit exclusion |
| Location | `tests/test_falsification_all_risks_identified.py` (8 `assert False` stubs) |

TDD RED-state falsification stubs use `assert False` to mark unresolved risks. These are intentionally failing — they exist to remind developers that the underlying code issues (C-31 through C-33) are not yet fixed. Currently excluded via `--ignore` in manual test runs.

Risk: a future CI pipeline that runs `pytest tests/` without excluding this file will see 8 deterministic failures. The stubs should NOT be converted to `xfail` or `skip` (that would silence the RED signal, defeating their TDD purpose). The correct resolution is either: (a) fix the underlying code issues so the stubs can be replaced with real passing tests, or (b) ensure CI explicitly excludes `test_falsification_*.py` files until remediation.

---

## Disagreements

### D-01: VolumeHandler scope — God Object vs Deep Module

| Field | Value |
|-------|-------|
| ID | D-01 |
| Source | expert-review (2026-04-08) |
| Perspectives | Martin (split — SRP Ch 7 p.80: serves 4 actors; ISP Ch 10 p.100: 20+ method interface; SAP Ch 14 p.139: Zone of Pain), Ousterhout (keep — successful deep module hiding complexity), Hickey (partial split — extract PF output path, keep volume ops together) |
| Resolution | **Accepted** (2026-04-08): Partial split will be executed when VolumeHandler next needs a non-trivial change to the PF output path (`to_evaluation_pf`, `to_forecast_pf`, `_reconstruct_as_pf_dict`). Extract PredictionFrame output into dedicated assembler; keep core volume operations in VolumeHandler. Three independent SOLID violations (SRP, ISP, SAP = C-36, C-37, C-39) support the split. Ousterhout's "deep module" counter-argument remains valid for core volume operations but does not extend to the PF output path. This decision unblocks C-36, C-37, C-39 for execution when the trigger fires. |

---

### D-02: Architecture extensibility — parameterize vs leave alone

| Field | Value |
|-------|-------|
| ID | D-02 |
| Source | expert-review (2026-04-08) |
| Perspectives | GoF (parameterize — 6 copy-pasted decoder blocks is anti-pattern), Beck/Feathers (leave alone — structural regex test guards against bugs, refactoring invalidates all .pt artifacts) |
| Resolution | Leave as-is. Cost of refactoring (breaking all artifacts) exceeds benefit. Structural test in `tests/test_architecture.py` provides adequate safety — this test is load-bearing infrastructure; do not modify without understanding its role as the guard for this decision. |

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

### C-04: Spatial offset arithmetic in VolumeSampler is untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-04 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_window_offset_preserves_geographic_truth` in `tests/test_volume_sampler.py`. Test plants a sentinel value at known coordinates, extracts a window, and verifies geographic round-trip via `spatial_offset`. |

---

### C-24: InferenceOrchestrator temporal discontinuity failure mode untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-24 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_gate_slice_time_beyond_bounds_raises` and `test_gate_slice_time_origin_plus_duration_oob` in `tests/test_volume_handler_hard_gates.py`. Tests verify `slice_time()` raises `ValueError` on out-of-bounds origins. |

---

### C-25: Curriculum→Sampler zero-qualified-cells interaction untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-25 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_curriculum_high_threshold_triggers_fallback` in `tests/test_volume_sampler.py`. Test verifies CurriculumLearner + VolumeSampler interaction: extreme threshold yields `qualified=0` with valid batch via random fallback. |

---

### C-27: `train_model.py` import structure blocks local testing — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-27 |
| Resolved | 2026-04-08 |
| Resolution | Moved `from views_pipeline_core.managers.model import ModelPathManager` to `TYPE_CHECKING` guard with `from __future__ import annotations`. Import only runs during static analysis, not at runtime. |

---

### C-31: ADR-008 log-before-raise violations in 4 files — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-31 |
| Resolved | 2026-04-08 |
| Resolution | Applied ADR-008 "Narrative Failure" pattern (err_msg → logger.error → raise) to all 7 locations: `training_forensics.py` (3), `config_initializer.py` (1), `volume_handler.py` (2), `mtloss.py` (1). Also added `logger` to `mtloss.py` which previously had none. |

---

### C-02: Duplicated setup between eval and forecast — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-02 |
| Resolved | 2026-04-08 |
| Resolution | Added `forecast=True` parameter to `_setup_evaluation()`. `_forecast_model_artifact()` reduced from 25 lines of duplicated setup to a single `_setup_evaluation("forecasting", forecast=True)` call. Forecast path correctly skips partition lookup and uses `partition_bound=None`. |

---

### C-23: `extrapolate_time()` has no direct unit test — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-23 |
| Resolved | 2026-04-08 |
| Resolution | Added 4 tests in `tests/test_volume_handler_hard_gates.py`: shape preservation, temporal continuity (time channel increment verification), non-time channel cloning, and single-step edge case. |

---

### C-07: Training loop lacks explicit per-window memory cleanup — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-07 |
| Resolved | 2026-04-09 |
| Resolution | Added `del sample_handler, losses, w_loss` after `backward()` in the inner window loop of `training_loop()` in `training_engine.py`. Matches the per-origin cleanup pattern already used in the inference path. |

---

### C-08: North-Up flip symmetry is implicitly coupled — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-08 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_gate_flip_symmetry_from_df_to_output` in `tests/test_volume_handler_hard_gates.py`. Test plants geographic-row values, runs through `from_df()`, and verifies the North-Up flip maps correctly at every array index. Flip symmetry is now structurally tested. |

---

### C-15: `training_loop()` has 4+ responsibilities — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-15 |
| Resolved | 2026-04-09 |
| Resolution | Split `train_model.py` into `training_engine.py` (Entity layer, pure training logic) and `train_model.py` (Framework wiring, 38 lines). The file-level SRP violation is eliminated — `training_engine.py` serves the data scientist, `train_model.py` serves the platform. Function-level diagnostic mixing in `training_loop()` remains but is now contained in a single-responsibility module. |

---

### C-21: Bare `except Exception` swallows errors in inference and diagnostics — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-21 |
| Resolved | 2026-04-09 |
| Resolution | All 11 locations now comply with ADR-008. The 2 core-logic locations (`hydranet_inference.py:391`, `training_engine.py:224`) upgraded from silent `pass` to `logger.error(..., exc_info=True)` per Fail-Safe constraint. The 9 `visual_diagnostics.py` locations already logged as `logger.error`. All catch-all patterns are ADR-008 Section 4 compliant (Observability Actors permitted Fail-Safe). |

---

### C-18: No end-to-end training smoke test — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-18 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_training_smoke_end_to_end` in `tests/test_training_engine.py`. Runs full `training_loop` on 8x8 synthetic data (2 lessons, 1 window each). Verifies: completes without error, returns expected keys, records loss history, model parameters change from initialization. |

---

### C-32: VolumeSampler CIC failure modes untested and unregistered — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-32 |
| Resolved | 2026-04-08 |
| Resolution | Ledger Inconsistency already tested by existing `test_red_unknown_target`. Added `TestGeometricOverflow` class (2 tests) verifying bounds clamping with edge anchors and max-dim extraction. CIC Section 6 notes: code uses `np.clip` (silent correction) rather than raising — correct behavior, CIC language ("Fails if...") is aspirational rather than literal. |

---

### C-28: CIC test file references are stale — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-28 |
| Resolved | 2026-04-08 |
| Resolution | Updated test alignment sections in HydranetManager.md, HydraNetConfig.md, and ConfigInitializer.md to reference actual test files (test_config_typed.py, test_config_validation.py, test_manager_memory_hygiene.py, etc.) |

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps in numbering indicate merged or resolved entries
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `tech-debt-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** `Total Concerns` and `Open Concerns` in the register header are manually maintained — update them whenever a concern is added or resolved
- **Governed by:** ADR-048
