# Visual Diagnostics & Training Forensics — Full Code Review

**Date:** 2026-02-21
**Branch:** `feature/spatiotemporal-plotting-diagnostics`
**Scope:** `visual_diagnostics.py`, `training_forensics.py`, all callsites, and ADR compliance
**Triggered by:** Pre-merge quality gate

---

## TL;DR

The two modules are architecturally sound and the core design decisions are correct. The Null Object pattern, the TrainingForensics independence contract, and the 7-stage pipeline coverage are all well-executed. However there are **three ADR gaps** (037 unimplemented, 035 partially unvisualized, 034 wrong medium), **pervasive ADR 003 zero-magic violations**, **one orphaned public method** (`biopsy_tensor`), **one internal DRY violation** (channel-selection logic duplicated between `biopsy_volume` and `biopsy_sample`), and several small code quality issues. The ADR gaps are the only items that block merge.

---

## Files Reviewed

| File | Lines | Role |
|------|-------|------|
| `views_hydranet/utils/visual_diagnostics.py` | 684 | Visual Truth Engine — 8 public methods, PNG output |
| `views_hydranet/utils/training_forensics.py` | 164 | Independent Forensic Auditor — metric history, no I/O |
| `views_hydranet/train/train_model.py` | 387 | Primary training callsite — Stages 4, 5 |
| `views_hydranet/manager/hydranet_manager.py` | ~300 | Manager callsite — Stages 1, 2, 3, 7 |
| `views_hydranet/utils/inference_orchestrator.py` | ~150 | Inference callsite — Stage 6 |
| `views_hydranet/utils/hydranet_inference.py` | ~310 | Inference callsite — Stage 5 autoregressive |

**ADRs against which compliance was checked:** 003, 005, 006, 008, 034, 035, 037

---

## Section 1 — What Works Well

### Null Object Pattern — correct and consistent
Every public method opens with `if not self.active: return`. No exceptions, no side effects, no files. The CIC documents this. The 8 inactive Beige Gate tests in `test_visual_diagnostics.py` confirm it. This is exactly what boring architecture looks like.

### TrainingForensics separation of concerns — excellent
`TrainingForensics` receives raw tensors, does independent computation, hands dicts to VisualDiagnostics. Zero matplotlib imports, zero file I/O, zero filesystem awareness. The namespaced key design (`"REG:lr_feat_a"`, `"CLS:lr_feat_a"`) prevents silent collisions between regression and classification targets that share a name. The `finalize_lesson` / `record` lifecycle is clean; the accumulator-reset pattern is correct. This module genuinely earns the word "independent."

### `_calculate_stats` — simple, safe, single-responsibility
Handles the all-NaN edge case cleanly. Returns a deterministic string. No branching on configuration. Good.

### Callsite coverage is complete (minus one method)
7 of 8 public methods have live production callsites:

| Method | Callsite |
|--------|----------|
| `biopsy_dataframe` | `hydranet_manager.py` — Stages 1, 2, 7 |
| `biopsy_volume` | `hydranet_manager.py` — Stage 3; `inference_orchestrator.py` — Stage 6 |
| `biopsy_tensor` | **None — orphan** |
| `biopsy_sample` | `train_model.py` — Stage 4 |
| `biopsy_autoregressive` | `hydranet_inference.py` — Stage 5 inference |
| `biopsy_training_performance` | `train_model.py` — Stage 5 training |
| `biopsy_loss_curves` | `train_model.py` — per lesson |
| `biopsy_feature_dossier` | `train_model.py` — per lesson, per target |

### Timestamp-keyed save_dir — correct
`reports/plots/diagnostics/{ts}/` isolates runs. Propagating `run_timestamp` from the manager ensures all PNGs for a given run land in one folder regardless of which module generates them.

---

## Section 2 — ADR Violations and Gaps

### ADR 037 — `HealthConstellation` radar plot: NOT IMPLEMENTED

**Status:** Accepted (06.02.2026). Mandate is explicit: a `HealthConstellation` radar plot of L2 weight norms per functional block, saved alongside the `.pt` artifact at the end of `train_model_artifact`.

**Current state:**
- `train_model_artifact` calls `training_loop`, receives a `summary` dict containing `weight_norms`, then saves the `.pt` file. No visualization is triggered.
- `VisualDiagnostics` has no `biopsy_health_constellation` or radar plot method anywhere.
- `biopsy_loss_curves` shows loss evolution — this addresses ADR 035's loss-convergence requirement but is orthogonal to ADR 037.

**Verdict: ADR 037 is unimplemented.** The radar plot does not exist anywhere in the codebase. This blocks merge.

---

### ADR 035 — Training Health Audit: computationally done, not visually audited

**Mandate:** Post-training audit showing Final/Min/Max loss, L2 norms per layer, binary health verdict (CRITICAL FAILURE if non-finite), "saved as a PNG artifact."

**Current state:**
- `training_loop` correctly returns `weight_norms`, `final_loss`, `min_loss`, `max_loss`.
- `IntegrityGuardian.monitor()` guards against NaN/explosion during training.
- `biopsy_loss_curves` shows loss history. ✓
- The `weight_norms` dict is returned from `training_loop` and then discarded — no method in `VisualDiagnostics` consumes it, no health verdict PNG is produced.

**Verdict: The numerical infrastructure exists; the visual output (L2 norm chart, health verdict) does not.** Partial gap — the loss curves satisfy part of ADR 035, but the weight norm visualization is absent. Blocks merge.

---

### ADR 034 — Prediction Diagnostic Summary: wrong medium

**Mandate:** "Mandatory, permanent diagnostic summary" via `HydranetManager._log_prediction_summary`, providing a **terminal text table** of Min/Max/Mean/NaN/Inf counts per column for every output sequence.

**Current state:**
- `hydranet_manager.py` calls `viz.biopsy_dataframe(...)` at Stage 7. This produces a spatial heatmap PNG.
- A PNG heatmap and a terminal text table serve different purposes and are not substitutes. The PNG shows spatial distribution; the table answers "is any column all-NaN, is any value Inf."
- No `_log_prediction_summary` method exists in `hydranet_manager.py` based on callsite analysis.

**Verdict: ADR 034 mandates a terminal text table; the implementation provides a PNG instead.** The PNG is valuable but is not ADR 034 compliance. Blocks merge.

---

### ADR 008 — Error Propagation: deliberate tension, one inconsistency

ADR 008 mandates the three-step Narrative Failure pattern (`err_msg → logger.error → raise`) for all exceptions. `VisualDiagnostics` intentionally deviates — it catches, logs, and swallows. The CIC §6 documents this as the explicit contract ("NO exception propagation is permitted from any public method"). This is a defensible override.

**The inconsistency:** `biopsy_dataframe` uniquely re-raises in DEBUG mode:
```python
if logger.getEffectiveLevel() <= logging.DEBUG:
    raise e
```
No other method does this. If DEBUG re-raising is valuable for `biopsy_dataframe`, the policy should apply to all 8 methods or none. The current state means developers must know to look for this special case in one specific method.

**Verdict:** The no-propagation contract is defensible but the `biopsy_dataframe` DEBUG re-raise is an inconsistency. Either generalize it to a class-level DEBUG mode or remove it. Does not block merge, but should be resolved.

---

### ADR 003 — Zero Magic: pervasive violations

ADR 003 Law 2 ("Zero-Magic") requires all magic values to be named and explicit. The visualization code contains numerous unnamed constants scattered across multiple methods:

| Location | Magic value | What it means |
|----------|-------------|----------------|
| `biopsy_autoregressive` | `n_times = 6` | Number of temporal columns in the grid |
| `biopsy_autoregressive` | `figsize=(18, 10)` | Hardcoded figure dimensions |
| `biopsy_training_performance` | `figsize=(18, 12)` | Different hardcoded figure dimensions |
| `biopsy_loss_curves` nested fn | `figsize=(10, 12)` | Yet another hardcoded figure size |
| `biopsy_dataframe`, `biopsy_volume`, `biopsy_sample` | `5` in `np.linspace(..., 5, ...)` | "Show 5 timestamps" |
| `biopsy_autoregressive` | `labelpad=80` | Axis label padding |
| `_plot_grid` | `labelpad=60` | Same concept, different value |
| All `plt.savefig` calls | `dpi=100` | Save resolution |
| `train_model.py:134` | `max(0, (seq_len // 2) - 3)` | "Middle 6 steps" calculation |
| Colormaps | `'magma'`, `'viridis'`, `'Reds'` | Semantic color choices without names |

These should be named module-level constants:
```python
# Proposed constants
N_BIOPSY_TIMES = 5           # Timestamps shown per spatial biopsy grid
N_AUTOREGRESSIVE_STEPS = 6   # Columns in Truth/Pred/Delta grid
BIOPSY_DPI = 100
BIOPSY_FIGSIZE_WIDE = (18, 10)
BIOPSY_FIGSIZE_TALL = (10, 12)
BIOPSY_LABELPAD_WIDE = 80
BIOPSY_LABELPAD_NARROW = 60
CMAP_REGRESSION = 'magma'
CMAP_FEATURES = 'viridis'
CMAP_DELTA = 'Reds'
```

The inconsistent `labelpad` values (80 vs. 60) in different methods are especially symptomatic — either they are the same concept (in which case there should be one constant) or they are intentionally different (in which case the intent should be documented).

**Verdict:** Widespread ADR 003 violation. Does not block merge but should be addressed in this branch since the methods that contain the magic numbers are entirely new code introduced on this branch.

---

## Section 3 — Code Quality Issues

### Internal DRY violation: channel-selection duplicated in two methods

`biopsy_volume` and `biopsy_sample` both contain an identical block for selecting which channels to display:

```python
# Duplicated verbatim in both methods:
interesting = []
meta_order = [vh.time_col, vh.id_col, "c_id"] + list(vh.spatial_cols)
for c in meta_order:
    if c in vh.channel_map and c not in interesting:
        interesting.append(c)
for c in vh._metadata.feature_cols:
    if c in vh.channel_map and c not in interesting:
        interesting.append(c)
```

This is a DRY violation inside the view layer itself. Extract to `_select_display_channels(vh: VolumeHandler) -> List[str]`. See also the `biopsy_sample` design discussion below.

### `biopsy_tensor` — public method with no callsite (orphan)

7 of 8 public methods have live callsites. `biopsy_tensor` is documented in the CIC, tested in the test suite, but never triggered in any production path. Either wire it into the inference path or retire it. A public method with a contract and no caller is contract debt.

### Nested `_generate_plot` inside `biopsy_loss_curves`

A function defined inside a public method to handle one boolean (`is_log`) is non-standard and makes the outer method harder to read. Extract as a private method:
```python
def _plot_loss_evolution(self, history_reg, history_cls, history_total, stage_label, is_log: bool):
    ...
```
Called twice from `biopsy_loss_curves`. The closure is unnecessary.

### `im = ax.imshow(...)` dead variable in `_plot_grid`

The return value of `ax.imshow()` is captured as `im` but never used (no colorbar, no further manipulation). Remove the assignment.

### In-function imports at wrong scope

`import matplotlib.patches as patches` in `_plot_grid_with_context` and `from datetime import datetime` in `__init__` should be at the module header. In-function imports hide dependencies and slow down subsequent calls.

### `signal_feat = interesting[-1]` in `biopsy_sample` — unguarded index

If `interesting` is empty (VolumeHandler with no feature_cols), this raises `IndexError`. The surrounding `try/except` catches it, but the logged error message will not explain what happened. Add a length guard before this line.

### `biopsy_feature_dossier` — unguarded dossier key access

`dossier["y_bar"]`, `dossier["y_hat_bar"]`, `dossier["bias_instant"]`, `dossier["bias_running"]` are accessed by key without membership checks. If the dossier is incomplete, the exception is a bare `KeyError` which is hard to diagnose from the log. A check like `if "y_bar" not in dossier: logger.warning(...); return` would produce a meaningful error.

### TrainingForensics silently filters `y_hat_bar` without logging

```python
self.reg_metrics = [m for m in raw_reg_metrics if m.lower() != "y_hat_bar"]
```
If a developer adds `"y_hat_bar"` to `regression_metrics` in config, it is silently dropped. Should emit `logger.warning(f"TrainingForensics: 'y_hat_bar' is a computed field, not a configurable metric; removing from reg_metrics.")`.

### Loss curve fixed filenames are correct but undocumented

`loss_evolution.png` and `loss_evolution_log.png` are overwritten on each lesson, always showing the complete history to date. This is correct — the final file is the full training curve. But it is not obvious from reading the code. A one-line comment stating the intent would close the confusion.

---

## Section 4 — The `biopsy_sample` Location Debate

The argument presented was: the data extraction logic inside `biopsy_sample` (slicing `g_data`, computing offsets) is "business logic leaking into the view layer," and the counter-argument is that moving it to `VolumeHandler` would bloat the core data class with diagnostic-specific patterns. The verdict was "acceptable for now."

**I partially disagree with the framing, and I disagree with the verdict.**

The view-vs-model argument is a distraction here. The real problems with `biopsy_sample` are two distinct things that should not be conflated:

**Problem 1: VolumeHandler slicing in `biopsy_sample` is NOT diagnostic-specific.**
The operation in question — extracting a 2D spatial map for a given channel at given time indices — is a general-purpose data access pattern. A method like `get_channel_maps(channel_idx: int, time_indices: List[int]) -> np.ndarray` on VolumeHandler would be used by any consumer of the handler, not only diagnostics. Adding it would not "bloat VolumeHandler with diagnostic patterns"; it would add a legitimate general accessor. The counter-argument, as presented, is based on a false premise about the specificity of the operation.

However — and this is important — the proposed destination (VolumeHandler) is still probably wrong. `biopsy_sample` passes the data to `_plot_grid_with_context` which immediately iterates over it. The extraction and the plotting are adjacent. Pulling the extraction into VolumeHandler would create an awkward coupling where VolumeHandler must understand the concept of "context slices for a biopsy." That coupling is not better.

**Problem 2: The real smell is duplication, not layer violation.**
The channel-selection block (`interesting`, `meta_order`, the two loops) is copy-pasted verbatim from `biopsy_volume`. This is a DRY violation **within** VisualDiagnostics. That is the actual smell. Extracting `_select_display_channels(vh)` as a private helper resolves it cleanly, without touching VolumeHandler, and without a debate about "view layer" purity.

**The `signal_feat = interesting[-1]` line is the specific fragile moment.** It is not general-purpose data access — it is a heuristic ("use the last interesting channel as the global context signal"). That heuristic belongs in VisualDiagnostics, not in VolumeHandler. But it should not be a magic index access on a list that could be empty.

**Revised verdict:**
- Do NOT move extraction logic to VolumeHandler. The counter-argument is right on outcome but wrong on reasoning.
- DO extract `_select_display_channels(vh) -> List[str]` as a private helper in VisualDiagnostics to eliminate the duplication.
- DO add a guard for `interesting[-1]` before dereferencing it.
- The "if it gets more complex, move to a helper" verdict is a cop-out. The duplication already exists NOW. The helper should be extracted NOW.

---

## Section 5 — Priority Actions

### Must address before merge (ADR gaps)

1. **Implement `biopsy_health_constellation(weight_norms, stage_label)`** — radar plot per ADR 037. Called from `train_model_artifact` after `training_loop` returns. Data is already available in the `summary["weight_norms"]` dict.

2. **Add `_log_prediction_summary(df_list)` to `HydranetManager`** — terminal text table per ADR 034. Complement (not replace) the Stage 7 `biopsy_dataframe` PNG.

3. **Add a `biopsy_health_audit(summary, stage_label)` method to `VisualDiagnostics`** — to render the ADR 035 weight norm visualization. Input is the summary dict from `training_loop`. Produces: one PNG with L2 norms per layer and a colour-coded health verdict (green/yellow/red per threshold).

### Should address before merge (new code, no excuse for leaving ADR 003 violations)

4. **Promote all magic numbers to named module-level constants** — figsize, dpi, n_times, labelpad, fontsize, colourmap strings. These are all new code on this branch.

5. **Extract `_select_display_channels(vh)` private helper** — eliminate the DRY violation between `biopsy_volume` and `biopsy_sample`.

6. **Extract `_generate_plot` nested function → private method `_plot_loss_evolution(..., is_log)`.**

7. **Add `logger.warning` in TrainingForensics when `y_hat_bar` is silently filtered.**

8. **Remove the dead `im =` assignment in `_plot_grid`.**

9. **Move in-function imports (`patches`, `datetime`) to module header.**

10. **Resolve the `biopsy_dataframe` DEBUG re-raise inconsistency** — apply the policy uniformly or remove it.

### Can defer (post-merge)

11. Resolve `biopsy_tensor` orphan — wire it into the inference path or retire it.
12. Guard `signal_feat = interesting[-1]` with a length check.
13. Add a one-line comment explaining the fixed-name loss curve overwrite behaviour.
14. Check and guard dossier key access in `biopsy_feature_dossier`.

---

## Open Decision — `biopsy_sample` Refactor Shape

**Status: Unresolved. Must be decided before further work on `biopsy_sample` or `biopsy_volume`.**

### What is undecided

Items 5 and 12 in the priority list above both touch `biopsy_sample`. Item 5 (extract `_select_display_channels`) has a clear answer — it is a straightforward DRY fix inside VisualDiagnostics with no architectural risk. Item 12 (guard `interesting[-1]`) is a trivial safety fix. Neither of those is what is undecided.

What is undecided is **whether the global-context extraction logic in `biopsy_sample` should eventually be lifted out of VisualDiagnostics**, and if so, where it should go. The debate surfaced during review and was not resolved:

- **Position A (keep it in VisualDiagnostics):** The slicing of `g_data` at specific time/channel indices is tightly coupled to what `_plot_grid_with_context` needs. Moving it elsewhere creates indirection without payoff. The `try/except` already handles failures gracefully. After extracting `_select_display_channels` and guarding the fragile index, the remaining complexity is tolerable.

- **Position B (extract to a private helper in `utils/` or a VolumeHandler method):** The manual axis-permutation loop (`slc = [slice(None)] * g_data.ndim`) reimplements reasoning that VolumeHandler already owns via `get_axis_idx`. If VolumeHandler's internal axis convention ever changes, `biopsy_sample` will silently produce wrong global maps with no test catching it. A `get_channel_maps(channel_idx, time_indices)` method on VolumeHandler would be a legitimate general accessor, not a diagnostic-specific one.

### Why this is hard

Both positions have merit. The disagreement is genuinely about **where the spatial-indexing responsibility boundary lies** — a question that also affects `biopsy_volume` (which does its own axis permutation) and potentially any future consumer of VolumeHandler data that needs to extract 2D maps. This is an ADR 012 (Volume Ledger and Topology) question as much as a view-layer question.

Making the wrong call and then reversing it is low-cost (both directions are small refactors). The risk is in *not making the call* and having the two methods continue to diverge independently.

### Recommended next step

Before the next substantive change to either `biopsy_sample` or `biopsy_volume`: read ADR 012 and determine whether `get_channel_maps` belongs in VolumeHandler's responsibility surface. If yes, implement it there and update both methods. If no, extract a `_extract_spatial_maps(vh, channel_idx, time_indices)` private helper in VisualDiagnostics and use it in both methods. Either answer closes the question cleanly.

**Do not leave this unresolved past the next PR that touches either method.**

---

## Summary Scorecard

| ADR | Requirement | Status |
|-----|------------|--------|
| 003 | Zero Magic — named constants | ❌ Widespread violations in new code |
| 005 | No Test No Merge — both modules tested | ✅ 41 new tests, all passing |
| 006 | CIC for non-trivial classes | ✅ Both CICs present |
| 008 | Narrative Failure pattern | ⚠️ Deliberate override in CIC; `biopsy_dataframe` inconsistent |
| 034 | Prediction Diagnostic Summary (terminal table) | ❌ PNG produced instead of text table |
| 035 | Training Health Audit + weight norm visualization | ⚠️ Loss curves present; weight norm viz absent |
| 037 | Health Constellation radar plot | ❌ Not implemented |
