# Test Suite Forensic Audit
**Date:** 2026-02-19
**Scope:** All active tests (`tests/`), all legacy tests (`legacy_tests/`), ADR 005, CICs
**Test Result at Time of Audit:** 73/73 PASS — which is exactly the problem.

> ADR 005: "Happy-Path Only is Failure." A suite that always passes is not a safety net — it is a false alibi.

---

## 1. ADR 005 Compliance Summary

ADR 005 mandates a **Triple-Team Taxonomy**:

| Team | Mandate | Status |
|---|---|---|
| 🟩 Green (Resilience) | Bit-perfect round-trips, convergence, alignment | **GOOD** |
| 🟫 Beige (Realistic Misuse) | Missing configs, mismatched resolutions, ambiguous columns — must fail loudly | **WEAK** |
| 🟥 Red (Adversarial) | Shuffling, NaN/Inf injection, out-of-distribution configs — must be invincible | **WEAK** |

The suite has strong Green coverage. It has nearly no Beige coverage for data geometry. It has zero Red coverage for the specific failure modes that caused the current performance collapse.

---

## 2. Silent Passing Tests — False Confidence

These tests pass but do not verify what they appear to verify.

### 2.1 Structure Without Values

**`tests/test_volume_handler_hard_gates.py` — `test_gate_12_6_head_dressing()`**
```python
assert "pred_lr_a" in pred_handler.channel_map
assert "pred_by_a" in pred_handler.channel_map
```
Verifies column *existence*, not column *content*. Wrapped predictions could be all zeros, all NaN, or spatially scrambled — this test is blind to all of it.

**`tests/test_volume_handler_hard_gates.py` — `test_gate_13_14_topography_restoration()`**
```python
assert isinstance(df_res.index, pd.MultiIndex)
assert df_res.index.names == ['month_id', 'priogrid_gid']
assert not any("INTERNAL" in col for col in df_res.columns)
```
Verifies structural integrity of the round-trip DataFrame, not spatial correctness. The data values could be remapped to the wrong cells and this test passes without complaint.

**`tests/test_point_collapse_survival.py` — `run_survival_audit()`**
```python
assert handler_4d.data.shape == (T, H, W, C)
```
Verifies shape of the collapsed volume. A silent NaN propagation or all-zeros collapse produces the correct shape. Mathematics not checked.

### 2.2 Circular Validation (Testing Implementation with Itself)

**`tests/test_volume_handler_geometric.py` — `test_flip_spatial()` and siblings**
```python
vh.flip("W")
expected = np.array([[[[2], [1]], [[4], [3]]]], dtype=np.float32)
assert np.array_equal(vh.data, expected)
```
The expected value is the developer's manual re-implementation of the flip. There is no independent oracle. If the flip logic is wrong in both the implementation and the expected value, the test passes. This is axiomatic, not empirical.

**`tests/test_volume_handler_geometric.py` — `test_permute_axes()`**
```python
vh.permute((0, 3, 1, 2))
assert vh.axes == ("T", "C", "H", "W")
assert vh.data.shape == (1, 1, 2, 2)
```
Checks that the axes tuple and shape updated — not that the underlying data array was actually transposed correctly. A no-op permute that only updated the metadata would pass.

### 2.3 The Conftest Injected a Known Bug and No Test Catches It

**`tests/conftest.py`** provides `valid_config_dict` with:
```python
"weight_init": "xavier_norm"
```

**`views_hydranet/utils/utils.py:79-86`** — `init_weights()`:
```python
if config['weight_init'] == 'xavier_uni':
    nn.init.xavier_uniform_(m.weight)
elif config['weight_init'] == 'kaiming_uni':
    nn.init.kaiming_uniform_(m.weight)
# 'xavier_norm' falls through. No init. No error. No warning.
```

The canonical test fixture describes a configuration that the codebase silently ignores. **Every test using `valid_config_dict` that touches model initialization is testing a model with default PyTorch init while believing it is testing `xavier_norm`.** The model appears to function, so the test passes. The bug is invisible to the suite.

### 2.4 Hardcoded Magic Indices Without Justification

**`tests/test_vader_bridge.py` — `test_bridge_vader_alignment_shuffle()`**
```python
posterior[0, 3, 0, 0] = 10.0  # PGID 1
posterior[0, 3, 1, 0] = 20.0  # PGID 2
posterior[0, 2, 0, 0] = 30.0  # PGID 3
posterior[0, 2, 1, 0] = 40.0  # PGID 4
```
The index `[0, 3, 0, 0]` encodes the expected result of `VolumeHandler.from_df()` + North-Up flip. If the flip or offset logic is wrong, these hardcoded indices point to the wrong cells. The test would then be asserting correctness against an incorrect expectation — and passing. The comment `# PGID 1` explains what but not why the index is `[0, 3, 0, 0]`.

### 2.5 Mocks That Absorb the Error Being Tested

**`tests/test_optimization_gate.py`**
```python
def mock_forward(t0, h):
    pred = torch.ones((1, 1, 1, 1), requires_grad=True)
    return pred, pred, h
model.forward = mock_forward
```
The training loop is being tested but the model is replaced with a stub that accepts any input. If the training loop called the model with a wrong shape (e.g., feeding 4 channels to a 3-channel model), the mock would silently absorb it. The test verifies the optimization *logic*, not the shape *contract*.

---

## 3. Blind Spots — Missing Tests

These test scenarios do not exist anywhere in `tests/` or `legacy_tests/`.

### 3.1 Negative Index Wrapping in VolumeHandler [CRITICAL]

**The fault:** `volume_handler.py:147-148`:
```python
r_idx = (df[y_col] - row_offset).astype(int).values
```
If `df[y_col].min() < row_offset`, `r_idx` contains negative values. NumPy fancy indexing wraps them to the end of the array. The volume is silently scrambled. No exception is raised.

**What exists:** `test_gate_15_geographic_anchoring()` tests an exact offset match. It never tests a mismatch. The current test suite only validates the happy path — never the adversarial case that produces the failure mode under investigation.

**Missing test:**
- Config: `row_offset=50`, DataFrame `row` values starting at 20
- Expected: Hard error raised before writing to volume
- Currently: Silent spatial inversion

### 3.2 `xavier_norm` Weight Initialization [CRITICAL]

**The fault:** `init_weights()` has no handler for `'xavier_norm'`. The model keeps PyTorch default init silently.

**What exists:** Zero tests. The config fixture encodes the broken value but nothing asserts that it works or fails loudly.

**Missing test:**
- Apply `init_weights` to a fresh model with `weight_init='xavier_norm'`
- Assert either: (a) weights differ from default init, or (b) a `ValueError` is raised
- Currently: No test verifies either outcome

### 3.3 Unsorted DataFrame Input [HIGH]

**The fault:** `DataFetcher.fetch_df()` and `DataFetcher.standardize_raw_df()` apply no sort. The DataFrame arrives at `VolumeHandler.from_df()` in file-storage order.

**What exists:** No test for sort-order correctness or even sort-order verification. `DataSniffer.sniff_ingestion()` checks uniqueness and finiteness — not ordering.

**Missing test:**
- Create a DataFrame with rows in scrambled `(month_id, priogrid_gid)` order
- Feed to `VolumeHandler.from_df()`
- Assert result matches a correctly-sorted reference
- Currently: No test. VolumeHandler uses fancy indexing so it is actually position-agnostic — but the question of whether DataSniffer should enforce ordering is unanswered.

### 3.4 DataSniffer Offset Drift Detection [HIGH]

The `DataSniffer` CIC states: *"Anchor Drift: Raises a critical error if geographic offsets (`row_offset`, `col_offset`) drift between data partitions."*

**What exists:** `DataSniffer.sniff_forecast_alignment()` checks temporal continuity (`month_id` range). It does not check offset consistency.

**Missing test:**
- Call `sniff_forecast_alignment()` with a handler whose `spatial_offset` differs from the config
- Assert a hard error is raised
- Currently: No such check in the code, no test, no error. CIC promise is unimplemented.

### 3.5 Loss Mask Alignment [MEDIUM]

The training loop computes a 6-element loss tensor and feeds it to `MultiTaskLoss(is_regression=[T,T,T,F,F,F])`. If the model ever outputs a different head count, the tensor size and the mask size diverge. NumPy/PyTorch broadcasting can silently produce a wrong scalar loss.

**What exists:** `test_mtloss.py` tests internal weighting logic with hardcoded `is_regression = torch.Tensor([True, False])`. It never tests the alignment between the config-driven target count and the hardcoded loss mask.

**Missing test:**
- Build MultiTaskLoss with `is_regression` length ≠ the number of losses returned by the model
- Assert a hard error, not silent broadcasting

### 3.6 Spatial Content Verification [MEDIUM]

No test verifies that after `VolumeHandler.from_df()`, the *values* at specific spatial locations match the input DataFrame values at those coordinates. All spatial tests check shape, index structure, or channel names — not spatial content.

**Missing test (The Gradient Test from the Visual Diagnostics Plan):**
- Build a DataFrame where `value = row_index` (a perfect north-south gradient)
- Create a volume via `VolumeHandler.from_df()`
- Assert that the volume's spatial slice shows a monotonically increasing gradient
- Currently: No test. This is the exact test the Visual Diagnostics Plan was designed to perform visually — it should also exist as an automated assertion.

### 3.7 Autoregressive Shape Contract [LOW]

The inference loop feeds `t1_pred` (shape `[B, 3, H, W]`) back as `t0`. If `input_channels` changes, the loop silently feeds the wrong shape into the model's first conv layer.

**Missing test:**
- Verify `output_channels == input_channels` for the autoregressive model
- Assert a hard error if they diverge

---

## 4. Faulty Test Implementations

### 4.1 Bare Assertions (Silent CI Failures)

Bare `assert` statements produce uninformative `AssertionError` on CI with no context. Found in:

| File | Line | Example |
|---|---|---|
| `test_volume_handler_hard_gates.py` | 33, 47, 50–53, 73–77 | `assert tensor.shape == (1, 1, 2, 4, 4)` |
| `test_point_collapse_survival.py` | 94 | `assert handler_4d.data.shape == (T, H, W, C)` |

Every bare assert should include a diagnostic message: `assert condition, f"Expected X, got {actual}"`.

### 4.2 Non-Deterministic Tests (Missing Seed Fixing)

Several tests use `np.random.rand()` or equivalent without fixing the seed, making them non-reproducible on CI failure:

| File | Lines |
|---|---|
| `test_red_team_the_abyss.py` | 29, 45 |
| `test_naming_symmetry_hard_gates.py` | 39, 52 |
| `test_backtest_unbreakable_audit.py` | 41 |
| `test_memory_fingerprint.py` | 31 |

### 4.3 Tolerance Without Justification

**`tests/test_focal_loss.py:33`:**
```python
assert pytest.approx(fl_loss.item()) == expected.item()
```
Uses default `pytest.approx` tolerance of `1e-6`. For stochastic volumes with many samples, floating-point accumulation can exceed this. The tolerance is unexplained and may be wrong in either direction. Document it explicitly with `rel=` and `abs=` arguments.

### 4.4 Over-Mocked Integration Tests

**`tests/test_optimization_gate.py`** patches both `CurriculumLearner` and `VolumeSampler`. The test verifies that the optimizer steps at the right frequency — but under conditions that can never occur in real training (zero-value volume, always-same-shape output). This is a unit test masquerading as an integration test.

---

## 5. Legacy Test Assessment

| File | Assessment |
|---|---|
| `legacy_tests/test_utils_data.py` | **ORPHANED.** Tests `get_data()` which loaded `.npy` files. Current pipeline uses `DataFetcher` + `.parquet`. Tests a dead code path. |
| `legacy_tests/test_scaling_parity.py` | **MISLEADING.** Tests hardcoded `log1p` JIT scaling. Current pipeline uses configurable `FeatureScaler`. These two paths can diverge silently and the test won't catch it. |
| `legacy_tests/test_utils_df_to_vol_conversion.py` | **ORPHANED.** Tests `df_to_vol()`, `vol_to_df()`, `calculate_absolute_indices()`. Canonical path is now `VolumeHandler.from_df()`. These utilities may be dead code. |
| `legacy_tests/test_manager_smoke.py` | **FRAGILE.** Smoke tests depend on external model artifacts and data files. Will fail in clean environments with no signal about the actual failure. |
| `legacy_tests/test_train_smoke.py` | **FRAGILE.** Same issue as manager smoke. |
| `legacy_tests/test_native_parity.py` | **POTENTIALLY VALUABLE.** Tests that output is identical across runs. But may test an old interface. Worth migrating rather than deleting. |
| `legacy_tests/test_scaling_parity.py` | **DANGEROUS.** If scaling logic has diverged between the legacy path and `FeatureScaler`, this test gives false confidence that scaling is correct. |

**Verdict:** The legacy test directory is a liability. Three categories: (1) orphaned (testing dead code), (2) dangerous (testing old code that diverges silently), (3) smoke tests that are environment-dependent. None of them are currently run by the CI-equivalent pytest invocation. They should either be migrated into `tests/` with updated contracts or deleted.

---

## 6. ADR 005 Compliance by Component

| Component | 🟩 Green | 🟫 Beige | 🟥 Red | Verdict |
|---|---|---|---|---|
| `VolumeHandler` | ✅ Good | ⚠ Weak | ❌ None | Offset wrapping untested |
| `DataFetcher` | ✅ Good | ❌ None | ❌ None | No error path tests at all |
| `DataSniffer` | ✅ Good | ✅ Good | ⚠ Weak | Offset drift CIC promise unimplemented and untested |
| `FeatureScaler` | ✅ Good | ⚠ Weak | ❌ None | Missing-column path not adversarially tested |
| `init_weights()` | ❌ None | ❌ None | ❌ None | `xavier_norm` completely untested |
| `MultiTaskLoss` | ✅ Good | ⚠ Partial | ❌ None | Mask/tensor misalignment untested |
| Training Loop | ✅ Good | ⚠ Mocked | ❌ None | Real shape contracts not enforced in tests |
| `InferenceOrchestrator` | ✅ Good | ✅ Good | ✅ Good | Best-covered component |

---

## 7. Priority Fix List

### Tier 1 — These directly correspond to the active performance collapse investigation:

1. **Test negative offset wrapping** — `VolumeHandler` should raise on `r_idx < 0`, not wrap. No test for this. `(tests/test_volume_handler_hard_gates.py)`
2. **Test `xavier_norm` init** — Either `init_weights()` should handle it (and a test confirms it does), or it should raise a `ValueError` (and a test confirms it does). Currently: neither. `(tests/test_architecture.py)`
3. **Test the spatial gradient property** — A DataFrame with a known gradient should produce a volume with that exact gradient. This is the automated version of the Visual Diagnostics Gradient Test. `(tests/test_volume_handler_hard_gates.py)`

### Tier 2 — Missing Beige/Red coverage that ADR 005 mandates:

4. **Test DataSniffer offset drift** — CIC promises it, code does not implement it, no test. `(new test file)`
5. **Test FeatureScaler with a column not in the transform dict** — Should fail loudly. `(tests/test_root_fix.py)`
6. **Test loss mask misalignment** — `MultiTaskLoss` with wrong `is_regression` length. `(tests/test_mtloss.py)`

### Tier 3 — Code quality fixes:

7. **Add diagnostic messages to all bare asserts**
8. **Fix random seeds** in `test_red_team_the_abyss.py`, `test_naming_symmetry_hard_gates.py`, `test_backtest_unbreakable_audit.py`, `test_memory_fingerprint.py`
9. **Document or remove legacy tests** — migrate `test_native_parity.py`, delete the three orphaned files
10. **Document magic index numbers** in `test_vader_bridge.py` with derivation comments
