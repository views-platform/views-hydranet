# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-04-08                           |
| Total Concerns    | 14                                   |
| Open Concerns     | 14                                   |
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

---

## Resolved Concerns

(none)

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements
- **Sources:** `repo-assimilation`, `expert-review`, `tech-debt-audit`, `falsification-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** `Total Concerns` and `Open Concerns` in the register header are manually maintained — update them whenever a concern is added or resolved
- **Governed by:** ADR-048
