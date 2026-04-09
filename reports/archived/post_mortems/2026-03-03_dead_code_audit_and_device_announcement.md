# Post-Mortem: Dead Code Audit, Device Announcement, and Schema Hardening

**Date:** 03.03.2026
**Status:** Completed
**Subject:** Full dead code audit across 10 categories, GPU/CPU device announcement, and targeted schema hardening (weight_decay, h_init, init_h unification).

---

## 1. Executive Summary

This session performed a systematic dead code audit across the entire HydraNet codebase, eliminating unreachable branches, phantom config keys, duplicate utility functions, low-value tests, and orphan scripts. In parallel, a GPU/CPU device announcement system was added at the entry gate of all three manager operations (training, evaluation, forecasting). The session concluded with a targeted schema hardening pass that corrected a silent optimizer misconfiguration, removed a phantom Pydantic field, and unified two divergent hidden-state initialisation methods into a single canonical form. All 168 tests remained green throughout.

---

## 2. Background & Context

HydraNet had accumulated several layers of technical debt across multiple ADR cycles, refactors, and migration phases. The debt was not causing visible failures — the test suite was green, training ran correctly — but the codebase contained a category of hazard that is more dangerous than broken code: **silent divergence**. Specifically:

- The optimizer was not applying the `weight_decay` regularisation the user had configured, because `AdamW` was constructed without the parameter. The config field existed, was validated, was logged — but was never consumed. PyTorch's default (`0.01`) was silently in effect instead.
- Two hidden-state initialisation methods (`init_h` and `init_hTtime`) existed side by side. Training used the legacy square-only `init_h`; inference used the correct rectangular `init_hTtime`. Any future non-square grid would have caused a shape mismatch in training with no runtime error at model construction time.
- A `h_init` config field was declared in `HydraNetConfig` but read by no production code. It passed validation, it appeared in config dicts, it did nothing.

These three issues share a common failure mode: **the code looks correct at a glance, gives no error, and silently does something other than what the author intended**.

The dead code audit surfaced these and ten other categories of waste.

---

## 3. What Was Done

### 3.1 Device Announcement (Observability Gate)

**Problem:** The pipeline gave no clear signal at startup about whether it was running on a GPU or CPU. CPU runs are severely degraded and a researcher could waste hours before noticing.

**Change:** Added `log_device_report(device, run_type)` to `utils_logging.py`. The function emits a structured banner at the start of every manager operation:
- GPU path: a `👾 ===` banner displaying GPU name, VRAM (MiB), and device count.
- CPU path: a loud `🚨 ===` WARNING banner with a `logging.warning()` call to ensure capture by all log handlers. No hard stop — the pipeline proceeds.

Called from three entry points in `hydranet_manager.py`:
- `_train_model_artifact` → `log_device_report(self.device, "training")`
- `_evaluate_model_artifact` → `log_device_report(self.device, eval_type)`
- `_forecast_model_artifact` → `log_device_report(self.device, "forecasting")`

**Implementation note:** `torch` is a project-level dependency but must not be imported at module level in `utils_logging.py` (would create a circular import path). Resolved with a `TYPE_CHECKING` guard for the annotation and a local `import torch` inside the function body.

---

### 3.2 Dead Code Audit (10 Categories)

A structured audit identified 15 items across the following categories:

| Category | Finding |
|---|---|
| Unused functions | `execute_freeze_h_option()` in `utils.py` — a standalone copy of a live `HydraNetInference` method. Zero call sites. |
| Unreachable branches | `try/except ValueError: pass` around `channel_map.index(time_col)` in `VolumeHandler.from_df` — `time_col` is always at index 0 by construction. |
| Unreachable branches | `if not feature_indices: return tensor` fallback in `VolumeHandler.to_pytorch` — an empty `feature_cols` is a construction error, not a runtime condition. |
| Unreachable branches | `try/except ValueError: pass` around `channel_map.index(m_col)` in `VolumeHandler.extrapolate_time`. |
| Config schema gaps | `weight_decay` declared in `HydraNetConfig` but not passed to `AdamW`. |
| Config schema gaps | `h_init` declared in `HydraNetConfig` but never read. |
| Split-brain API | `init_h(hidden_channels, dim)` — legacy square-only hidden state init, used only by training. `init_hTtime(hidden_channels, H, W)` — correct rectangular form, used by inference. |
| Orphan scripts | `scripts/audit_freeze_h_vectorization.py` — unreferenced, no CI entry, no imports. |
| Orphan scripts | `scripts/profile_inference.py` — unreferenced, contained a `time.sleep()` mock model. |
| Low-value tests | `test_utils_train_log.py` — 100% mocked, only asserted W&B call signature. |
| Low-value tests | `test_manager_model_path_regression.py` — only tested Python attribute assignment, no pipeline logic. |
| Low-value tests | `legacy_tests/test_manager_smoke.py` — fully `@pytest.mark.skip`, referenced a non-existent method. |

---

### 3.3 Dead Code Removal Round 1

Executed the safe, unambiguous removals first:

**`views_hydranet/utils/utils.py`**
Deleted `execute_freeze_h_option()` (22 lines). The live implementation lives in `HydraNetInference.execute_freeze_h_option()`. Having both was a copy-paste artefact from an earlier extraction. The standalone version was already diverging (missing the `freeze_h` config key branch).

**`views_hydranet/utils/volume_handler.py`**
- Removed the `try/except ValueError: pass` in `from_df` around `time_col` indexing. Time column is always first — the exception was not just unreachable but would have masked a real construction error.
- Replaced the `if not feature_indices: return ...` fallback in `to_pytorch` with a hard `assert`. An empty `feature_cols` is a programming error at construction time, not a recoverable runtime condition. The fallback was hiding potential misconfigurations.
- Removed the `try/except ValueError: pass` in `extrapolate_time` around `m_col` indexing. Same reasoning.

**Deleted tests:**
`tests/test_utils_train_log.py`, `tests/test_manager_model_path_regression.py`, `legacy_tests/test_manager_smoke.py`.

---

### 3.4 Dead Code Removal Round 2 — Schema and API Hardening

**`weight_decay` wired into AdamW (`views_hydranet/utils/utils.py`)**

Before:
```python
optimizer = torch.optim.AdamW(
    unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
)
```
After:
```python
optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config["learning_rate"],
    betas=(0.9, 0.999),
    weight_decay=config["weight_decay"],
)
```

The field had been in `HydraNetConfig` since the ADR 046 lifecycle audit. It was validated, logged, and silently ignored. PyTorch's `AdamW` default of `0.01` was in effect for all prior training runs. This is a **behavioural change**: all future training runs will apply the configured `weight_decay` (typically `1e-4` to `1e-2` depending on config). Results from runs prior to this commit used `weight_decay=0.01` regardless of what was in the config.

**`h_init` removed from `HydraNetConfig` (`views_hydranet/utils/utils_config.py`)**

`h_init: str = Field(...)` was a required field that accepted values like `"zeros"` but was never consumed. Removing it required verifying `class Config: extra = "allow"` was set — it is, so test fixtures that include `h_init` tolerate its removal silently. No test fixture edits were required.

**`init_h` → `init_hTtime` unification**

Three-file change:

1. `views_hydranet/train/train_model.py`: replaced `init_h(hidden_channels=model.base, dim=window_dim)` with `init_hTtime(hidden_channels=model.base, H=window_H, W=window_W)`. Rather than passing the single `window_dim` scalar, H and W are now extracted independently from the tensor shape (`train_tensor.shape[-2]` and `train_tensor.shape[-1]`). This makes the non-square grid case correct by construction rather than by coincidence.

2. `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py`: deleted the `init_h` method (4 lines, docstring "Legacy initialization. Use init_hTtime.").

3. `tests/test_optimization_gate.py`: updated the mock lambda from `lambda hidden_channels, dim:` to `lambda hidden_channels, H, W:`.

The two methods produced identical tensors in all current training configurations (training always uses square windows), so this is a zero-numerical-change migration. The invariant is now enforced architecturally rather than by convention.

**Orphan scripts deleted:**
`scripts/audit_freeze_h_vectorization.py`, `scripts/profile_inference.py`. The `scripts/` directory was removed as it became empty.

---

## 4. Lessons Learned

### 4.1 Silent Config Consumption is the Hardest Bug

`weight_decay` is a textbook example of an insidious class of defect: the code is correct syntactically, the config is valid semantically, the test suite is green — and the system is nonetheless doing something different from what the researcher believes. Pydantic validates that the field exists and has the right type. It cannot validate that the field is actually consumed.

**Guardrail:** Config fields that control optimiser or loss behaviour should have an end-to-end integration test that verifies the field changes observable model behaviour (e.g. `test_gate_18_weight_mutation_numerical` already partially covers this for the optimiser).

### 4.2 "Legacy" Comments are a Technical Debt Timer

The `init_h` docstring said "Legacy initialization. Use init_hTtime." for an unknown number of commits, yet `train_model.py` continued to call `init_h`. The comment was correct; the migration never happened. Comments that declare something deprecated but leave the deprecated thing in place are a debt instrument — they accumulate interest until someone pays them.

**Guardrail:** If something is deprecated, either migrate all call sites in the same commit or delete the deprecated method and let the compiler enforce the migration.

### 4.3 Unreachable Branches Hide Errors

The `try/except ValueError: pass` blocks in `VolumeHandler` were not just dead — they were actively dangerous. If a construction error caused `channel_map.index(col)` to raise, the exception would be swallowed and the downstream tensor would silently be wrong-shaped or wrong-indexed. Converting these to explicit `assert` statements or allowing the exception to propagate is strictly safer.

### 4.4 Low-Value Tests Have a Negative ROI

`test_utils_train_log.py` tested that `wandb.log` was called with three specific dictionary keys. It gave false confidence (the test passed even if the loss values were NaN), added maintenance overhead (had to be updated every time log keys changed), and contributed nothing to verifying pipeline correctness. Deleting it reduced noise without reducing coverage of anything meaningful.

### 4.5 Ruff on Untracked Files

The ruff run at session end surfaced violations in three new untracked test files (`test_inference_logic.py`, `test_sweep_and_hardening_gates.py`, `test_temporal_causality_audit.py`) that exist in the working tree but were not part of the current branch's committed history. These are pre-existing violations, not introduced by this cleanup. They should be addressed in a dedicated lint hygiene pass before those files are committed.

---

## 5. Files Changed

| File | Change |
|---|---|
| `views_hydranet/utils/utils_logging.py` | Added `log_device_report()` with GPU/CPU banners |
| `views_hydranet/utils/utils_device.py` | Replaced `print` with `logger.debug` |
| `views_hydranet/manager/hydranet_manager.py` | Added `log_device_report` calls at training/eval/forecasting entry |
| `views_hydranet/utils/utils.py` | Deleted `execute_freeze_h_option()`; wired `weight_decay` into `AdamW` |
| `views_hydranet/utils/utils_config.py` | Removed `h_init` field from `HydraNetConfig` |
| `views_hydranet/utils/volume_handler.py` | Removed 3 dead `try/except` blocks; replaced fallback with `assert` |
| `views_hydranet/train/train_model.py` | Migrated `init_h` → `init_hTtime` with explicit H/W extraction |
| `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` | Deleted legacy `init_h()` method |
| `tests/test_optimization_gate.py` | Updated mock lambda signature for `init_hTtime` |
| `tests/test_utils_device.py` | Removed stdout assertions; added `log_device_report` tests |
| `tests/test_utils_train_log.py` | **Deleted** (low-value) |
| `tests/test_manager_model_path_regression.py` | **Deleted** (low-value) |
| `legacy_tests/test_manager_smoke.py` | **Deleted** (fully skipped, dead reference) |
| `scripts/audit_freeze_h_vectorization.py` | **Deleted** (orphan) |
| `scripts/profile_inference.py` | **Deleted** (orphan) |
| `scripts/` (directory) | **Deleted** (now empty) |

---

## 6. Verification

```
168 passed, 53 warnings in 58.07s
```

All 168 tests green. No new ruff violations introduced by any file touched in this session.

---

## 7. Next Steps

- **Manager duplication refactor (deferred):** `HydranetManager` contains structurally duplicated blocks across `_train_model_artifact`, `_evaluate_model_artifact`, and `_forecast_model_artifact`. This was identified in the audit but explicitly deferred pending architectural investigation. The risk is non-trivial: the duplication is the result of intentional divergence across run types, not an oversight.
- **Lint hygiene pass on untracked tests:** Address ruff violations in `test_inference_logic.py`, `test_sweep_and_hardening_gates.py`, `test_temporal_causality_audit.py` before those files are committed.
- **Behavioural audit of weight_decay change:** Runs prior to this commit used `weight_decay=0.01` unconditionally. Current configs should be audited to ensure the configured value is intentional and appropriate for the target regularisation regime.
