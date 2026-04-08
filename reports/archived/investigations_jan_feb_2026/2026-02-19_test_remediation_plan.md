# Test Suite Remediation Plan
**Date:** 2026-02-19
**Status:** Planned
**Author:** Derived from `2026-02-19_test_suite_audit.md`
**Mandate:** ADR 005 — Testing as Mandatory Critical Infrastructure

---

## Preamble: What Is Actually Wrong

Before describing what to build, it is worth being precise about what has failed. The problem is not that there are not enough tests. The problem is that **the test suite provides false confidence** — it passes on the exact conditions that are currently breaking the model, and it does so silently, without any indication that something is wrong.

There are three specific mechanisms by which this false confidence is generated:

**Mechanism 1 — The Conftest Encodes a Known Bug as a Feature.**
`conftest.py` provides `"weight_init": "xavier_norm"` as the canonical config fixture for the entire test suite. The `init_weights()` function in `utils.py` has no handler for `'xavier_norm'`. Every test that uses this fixture and touches model initialization is silently testing PyTorch default init while believing it is testing the intended init scheme. The bug is baked into the test infrastructure itself. This is not a missing test — it is an existing test that actively lies.

**Mechanism 2 — Structural Assertions Are Not Semantic Assertions.**
The suite's spatial and round-trip tests verify that shapes are correct, index names are correct, and column names exist. None of them verify that the *values* at specific spatial locations are correct. A VolumeHandler that perfectly maps every data point to the wrong cell on the grid would pass every test in the suite without raising a single alarm. The tests are checking the envelope, not the letter inside.

**Mechanism 3 — Adversarial Cases Are Absent.**
ADR 005 explicitly mandates Red Team tests that assume hostile or worst-case input. The most dangerous failure mode in the system — a wrong spatial offset causing numpy fancy indexing to wrap negative indices silently — has zero Red Team coverage. Not one test probes what happens when `row_offset > df.row.min()`. The suite cannot detect the highest-probability root cause of the current performance collapse.

This plan addresses all three mechanisms in order of severity. It does not treat these as isolated bugs to be patched. It treats them as **symptoms of a structural gap** that must be corrected systematically.

---

## Part 1: Immediate Critical Tests (The Investigation Unlockers)

These three tests must be written first. They are not general quality improvements — they are diagnostic instruments for the active performance collapse investigation. Until they exist, we are running blind.

---

### Test 1.1 — The Spatial Gradient Preservation Test

**The Fault Line It Closes:** Fault Line 2 — Spatial scrambling via wrong `row_offset`/`col_offset`.

**Why This Is the Most Important Test in the Repo Right Now:**
The Visual Diagnostics Plan (2026-02-19) describes a "Gradient Test" as the primary mechanism for detecting spatial scrambling. That plan proposes running a pipeline stage and eyeballing a PNG. This test is the automated, deterministic, always-running equivalent. It does not require a human to look at a plot. It either passes or it fails. If this test had existed before the performance collapse, the collapse would have been caught at the first training run.

**What It Must Prove:**
Given a DataFrame where each cell's value is set to its known local row index (a perfect north-south gradient), after `VolumeHandler.from_df()` the resulting volume must contain exactly that gradient at the correct spatial locations. The test must verify actual values at specific coordinates, not just shape or column names.

**Explicit Test Contract:**

```
GIVEN:
  - A config with row_offset=R, col_offset=C, height=H, width=W
  - A DataFrame where:
      df['row'] = R + local_row  (global coordinates, matching offset)
      df['col'] = C + local_col
      df['value'] = local_row    (the gradient: value equals the row index)
      one time step, all (H x W) cells present

WHEN:
  - VolumeHandler.from_df(df, config) is called

THEN:
  - For every cell at local row r, col c:
      volume.data at (row=r, col=c) must equal r (or H-1-r after North-Up flip)
  - The spatial gradient must be monotonically increasing (or decreasing after flip)
  - No cell must have the value of a different row (i.e., no scrambling)
  - np.all(np.diff(volume[:, :, 0, feature_idx], axis=0) >= 0) or <= 0 (monotonic)
```

**What a Failing Test Looks Like:**
If `row_offset` is wrong (e.g., set to 0 when data starts at 87), `r_idx = row - 0 = 87+` which indexes into the volume beyond height, numpy wraps, and the gradient is not monotonic. The assertion `np.all(np.diff(...) >= 0)` fails. The test reports the actual values at each row, making the scrambling immediately visible.

**Target file:** `tests/test_volume_handler_hard_gates.py` — add as `test_gate_16_spatial_gradient_preservation()`

**Acceptance Criteria:**
- The test must fail if `row_offset` is set to any value other than `df['row'].min()`
- The test must fail if `col_offset` is set to any value other than `df['col'].min()`
- The test must pass for any valid (H, W) grid size from (4x4) to (180x180)
- The test must explicitly print the actual vs expected gradient slice on failure

---

### Test 1.2 — The `xavier_norm` Initialization Test

**The Fault Line It Closes:** Confirmed bug in `utils.py:init_weights()` — `'xavier_norm'` is silently ignored.

**Why This Is Urgent:**
This bug has affected every training run since `'xavier_norm'` was added to the config. The model has never been trained with the intended initialization. We do not know whether this is contributing to the performance collapse, but we know it is a silent lie in the system. More critically, `conftest.py` encodes this broken value as the canonical config fixture. Every test using that fixture is running against an unverified initialization assumption.

**What It Must Prove:**
`init_weights()` must either (a) correctly apply `nn.init.xavier_normal_` when called with `'xavier_norm'`, OR (b) raise an explicit `ValueError` with a clear message identifying the unknown init scheme. What it must NOT do is fall through silently.

**This test has two phases:**

**Phase A — Immediate Red Gate (write this now):**
```
GIVEN:
  - A simple nn.Conv2d layer with known default weights
  - A config with weight_init='xavier_norm'

WHEN:
  - init_weights(layer, config) is called

THEN:
  - EITHER: The layer's weights differ from the default PyTorch init
            (proven by comparing to a fresh, un-initialized Conv2d of same shape)
  - OR:     A ValueError is raised with a message containing 'xavier_norm'

IF:
  - The function returns without error AND the weights are identical to default
  - FAIL with message: "init_weights silently ignored 'xavier_norm'. This is the confirmed bug."
```

**Phase B — After the bug is fixed (write alongside the fix):**
```
GIVEN:
  - A nn.Conv2d layer
  - A config with weight_init='xavier_norm'

WHEN:
  - init_weights(layer, config) is called

THEN:
  - The layer's weights follow the Xavier Normal distribution
  - Verify: weight variance ≈ 2 / (fan_in + fan_out) (the Xavier formula)
  - Verify: no weights are exactly at the PyTorch Kaiming default values
```

**Companion action:** The `conftest.py` fixture must be updated to reflect the actual canonical behavior. If `'xavier_norm'` is to be supported, update conftest after the fix. If it is to be replaced with `'xavier_uni'`, update conftest before writing Phase B. Do not leave conftest encoding a broken value.

**Target file:** `tests/test_architecture.py` — add as `test_weight_init_xavier_norm_is_not_silent()`

**Acceptance Criteria:**
- The test must fail on the current codebase (it is a Red Gate — it must detect the existing bug)
- After the fix is applied, Phase B must pass
- `conftest.py` must be updated in the same PR as the fix

---

### Test 1.3 — The Negative Offset Rejection Test

**The Fault Line It Closes:** Fault Line 2 (adversarial variant) — `row_offset > df.row.min()` causes silent numpy wrapping.

**Why This Must Be a Hard Error:**
NumPy fancy indexing with negative indices does not raise. It silently writes to the wrong locations. `volume[r_idx, c_idx, m_idx, i] = df[col_name].values` with `r_idx = [-87, -86, ..., -1]` writes to the last 87 rows of the volume, inverting the map. The VolumeHandler currently has no guard for this. The DataSniffer has no guard for this. There is no gate between a misconfigured offset and a silently corrupted volume.

**What It Must Prove:**
If `(df[y_col].min() - row_offset) < 0`, `VolumeHandler.from_df()` must raise a `ValueError` before writing a single value to the volume. The error message must state the computed negative index, the actual data minimum, and the configured offset, so the developer can immediately identify which offset is wrong.

**Explicit Test Contract:**

```
GIVEN:
  - A config with row_offset=50, col_offset=0, height=100, width=100
  - A DataFrame where df['row'].min() = 20  (i.e., 20 - 50 = -30, negative)

WHEN:
  - VolumeHandler.from_df(df, config) is called

THEN:
  - A ValueError is raised BEFORE any data is written to the volume
  - The error message contains the actual negative index value (-30)
  - The error message contains the configured row_offset (50)
  - The error message contains df['row'].min() (20)

ALSO TEST:
  - Exact boundary: row_offset = df['row'].min() → must NOT raise (valid case)
  - Col offset negative: same test for col_offset > df['col'].min()
  - Both offsets wrong simultaneously: one error covering both
```

**What a Passing Test Demonstrates:**
That the system fails **loudly and precisely** when given a misconfigured offset — rather than producing a spatially inverted volume that trains for hours before anyone notices the model outputs random noise.

**Target file:** `tests/test_volume_handler_hard_gates.py` — add as `test_gate_17_negative_offset_rejection()`

**Important note on implementation dependency:** This test will FAIL on the current codebase because `VolumeHandler.from_df()` does not currently contain this guard. The test must be written first (as a Red Gate), then the guard must be added to `VolumeHandler.from_df()` to make it pass. Do not write the guard without the test. Do not write the test and then not write the guard.

**Acceptance Criteria:**
- The test must fail on the current codebase (confirming the guard is absent)
- After the guard is added to `VolumeHandler.from_df()`, the test must pass
- The guard must not be added inside a try/except that swallows the error — it must propagate

---

## Part 2: ADR 005 Compliance Tests (The Structural Gap)

These tests close the gap between what ADR 005 mandates and what exists. They are not tied to the immediate investigation but are mandatory per the ADR's "No Test, No Merge" rule. They must be written before any new feature work proceeds.

---

### Test 2.1 — DataSniffer Offset Drift Detection

**Mandate source:** DataSniffer CIC, line 53-55: *"Anchor Drift: Raises a critical error if geographic offsets (`row_offset`, `col_offset`) drift between data partitions."*

**Current reality:** `DataSniffer.sniff_forecast_alignment()` checks temporal continuity. It does not check that the `spatial_offset` of the history VolumeHandler matches the `spatial_offset` of the forecast VolumeHandler. The CIC promise is unimplemented. There is no test because there is nothing to test — the feature does not exist.

**This test must be written as a Red Gate (failing test first):**
```
GIVEN:
  - A history VolumeHandler with spatial_offset=(87, 310)
  - A forecast VolumeHandler with spatial_offset=(88, 310)  ← drifted by 1 row

WHEN:
  - DataSniffer.sniff_forecast_alignment(df, history_handler, is_forecast=True) is called

THEN:
  - A ValueError or equivalent critical error is raised
  - Error message identifies the specific offset that drifted

CURRENT RESULT:
  - No error. The sniffer checks months only. The drift is invisible.
  - This test will FAIL on the current codebase, confirming the gap.
```

**Action required:** After writing the test, implement offset drift detection in `DataSniffer.sniff_forecast_alignment()`. This closes a CIC promise that is currently a dead letter.

**Target file:** New file `tests/test_data_sniffer_contracts.py`

---

### Test 2.2 — FeatureScaler Column Coverage Enforcement

**The contract:** `HydraNetConfig` validates that all features and targets are in the `transform` dict. `FeatureScaler` applies transforms based on that dict. If a column is present in the DataFrame but absent from the transform dict, it should fail loudly, not silently pass through untransformed.

**What this tests:**
```
GIVEN:
  - Config with transform={'log1p': ['feat_a']}
  - DataFrame with columns ['feat_a', 'feat_b']  ← feat_b unmapped

WHEN:
  - FeatureScaler(config).fit_transform(df) is called

THEN:
  - A ValueError is raised identifying feat_b as unmapped
  - The scaler does NOT silently pass feat_b through as raw values

WHY THIS MATTERS:
  - If the scaler silently passes raw counts to the model, gradients explode
  - The current validation in HydraNetConfig should catch this — but only if
    ConfigInitializer is called and only if the Pydantic model is strict
  - This test verifies the behavior at the FeatureScaler boundary itself,
    independent of the config validation layer
```

**Target file:** `tests/test_root_fix.py` — add as `test_scaler_rejects_unmapped_columns()`

---

### Test 2.3 — MultiTaskLoss Task Count Alignment

**The contract:** `MultiTaskLoss` is initialized with an `is_regression` tensor of length N. The training loop must provide exactly N individual loss values. If N_losses != N_tasks, behaviour is undefined. Currently this silently produces wrong task weights.

**What this tests:**
```
GIVEN:
  - MultiTaskLoss initialized with is_regression=[True, True, False] (3 tasks)
  - A losses tensor of length 2 (mismatch)

WHEN:
  - multitaskloss_instance(losses) is called

THEN:
  - A RuntimeError or ValueError is raised
  - The error identifies the size mismatch

ALSO TEST:
  - Length 4 losses with 3-task mask → error
  - Length 3 losses with 3-task mask → passes (the valid case)
```

**Target file:** `tests/test_mtloss.py` — add as `test_mtloss_rejects_mismatched_task_count()`

---

### Test 2.4 — Collapse Mathematical Correctness

**The contract:** `VolumeHandler.collapse_to_point()` reduces the stochastic `S` dimension by computing the arithmetic mean (or other aggregation). The current test in `test_point_collapse_survival.py` only verifies that the shape changes from `[T, H, W, C, S]` to `[T, H, W, C]`. It does not verify the math.

**What this tests:**
```
GIVEN:
  - A 5D VolumeHandler with known values:
      S=0: all cells = 2.0
      S=1: all cells = 4.0
      S=2: all cells = 6.0
  - Expected arithmetic mean = 4.0

WHEN:
  - vh.collapse_to_point(method='arithmetic_mean') is called

THEN:
  - The resulting 4D volume contains exactly 4.0 at every cell
  - np.allclose(collapsed.data, 4.0) must be True

ALSO TEST:
  - NaN propagation: if any S contains NaN, verify collapse uses nanmean (not mean)
    to avoid poisoning the entire output
  - Empty S dimension: assert error, not silent zeros
```

**Target file:** `tests/test_point_collapse_survival.py` — add as `test_collapse_mathematical_correctness()`

---

### Test 2.5 — Spatial Content Round-Trip (Value Preservation)

**The contract:** After `VolumeHandler.from_df()` → `VolumeHandler.to_evaluation_df()`, the values at each `(month_id, priogrid_gid)` must be identical to the values in the original DataFrame. This is the bit-perfect round-trip test that ADR 005 mandates but that the current suite does not actually perform at the value level.

**What this tests:**
```
GIVEN:
  - A DataFrame with known, distinct values at each (month_id, priogrid_gid)
  - Specifically: value = priogrid_gid (each cell has a unique, verifiable value)

WHEN:
  - df → VolumeHandler.from_df() → VolumeHandler.to_evaluation_df()

THEN:
  - For every (month_id, priogrid_gid) in the original:
      original_df[month_id, pgid, 'value'] == reconstructed_df[month_id, pgid, 'value']
  - Tolerance: exact equality (no loss of precision)

WHY THIS IS STRONGER THAN EXISTING TESTS:
  - Existing round-trip tests check that the MultiIndex exists and names are correct
  - This test checks that the VALUE at each cell survived the round-trip
  - A spatially inverted map would have the right structure but wrong values
```

**Target file:** `tests/test_volume_handler_hard_gates.py` — add as `test_gate_18_value_round_trip()`

---

## Part 3: Code Quality Corrections (The Silent Failure Factories)

These are not new tests. They are corrections to existing tests that currently provide misleading signals.

---

### Correction 3.1 — Add Diagnostic Messages to All Bare Assertions

Every bare `assert condition` must become `assert condition, f"..."` with a message that includes the actual value, the expected value, and enough context to diagnose the failure without reading the test code.

**Files to update:**
- `tests/test_volume_handler_hard_gates.py` — lines 33, 47, 50, 51, 52, 53, 73, 75, 77
- `tests/test_point_collapse_survival.py` — line 94
- Any other files found by `grep -n "^    assert " tests/*.py` where no message follows

**Standard:** Every assert must follow this pattern:
```python
assert actual == expected, (
    f"\nExpected: {expected}"
    f"\nActual:   {actual}"
    f"\nContext:  {describe what this value represents}"
)
```

---

### Correction 3.2 — Fix Non-Deterministic Tests

These tests use `np.random.rand()` or `torch.rand()` without seeding. They are non-reproducible on CI failure. Fix by adding explicit seeds at the top of each test function:

**Files and locations:**
- `tests/test_red_team_the_abyss.py:29, 45` — add `np.random.seed(42)` before data creation
- `tests/test_naming_symmetry_hard_gates.py:39, 52` — same
- `tests/test_backtest_unbreakable_audit.py:41` — same
- `tests/test_memory_fingerprint.py:31` — same

**Standard:** All random data in tests must use an explicit, documented seed. The seed should be a constant defined at the top of the test module, not inline:
```python
TEST_SEED = 42  # Fixed for reproducibility per ADR 005
```

---

### Correction 3.3 — Document Magic Indices in `test_vader_bridge.py`

The indices `posterior[0, 3, 0, 0]` etc. in `test_bridge_vader_alignment_shuffle()` are derived from the VolumeHandler's North-Up flip and offset logic applied to a 4x4 grid. They are currently undocumented. Any future change to the flip or offset logic that is correct will still break this test, and the developer will not know why.

**Required correction:** Add a derivation comment above each hardcoded index:
```python
# DERIVATION: PGID 1 maps to row=3, col=0 because:
#   global_row=0 → local r_idx = 0 - row_offset = 0
#   After North-Up flip on H=4: flipped_r = (H-1) - 0 = 3
#   global_col=0 → c_idx = 0 - col_offset = 0 (no flip)
posterior[0, 3, 0, 0] = 10.0  # PGID 1: (r=3, c=0) after flip
```

**This is not optional.** Undocumented magic constants in spatial tests are a maintenance trap. The next developer who changes the flip direction will not know whether the test is wrong or the code is wrong.

---

### Correction 3.4 — Update `test_optimization_gate.py` Mock to Enforce Shape Contract

The current mock absorbs any input shape silently:
```python
def mock_forward(t0, h):
    pred = torch.ones((1, 1, 1, 1), requires_grad=True)
    return pred, pred, h
```

This must be updated to at minimum assert the expected input shape:
```python
EXPECTED_INPUT_CHANNELS = 3  # Must match config['input_channels']

def mock_forward(t0, h):
    assert t0.shape[1] == EXPECTED_INPUT_CHANNELS, (
        f"Training loop fed wrong channel count to model: "
        f"expected {EXPECTED_INPUT_CHANNELS}, got {t0.shape[1]}"
    )
    pred = torch.ones_like(t0[:, :EXPECTED_INPUT_CHANNELS])
    return pred, pred, h
```

This converts a mock that silently swallows shape errors into one that actively enforces the autoregressive loop contract.

---

## Part 4: Legacy Test Disposition

The `legacy_tests/` directory must be decisively handled. It currently contains tests for dead or diverged code paths. It does not participate in the current CI run. It gives no value and creates confusion about what is canonical.

**Disposition decision for each file:**

| File | Decision | Rationale |
|---|---|---|
| `test_utils_data.py` | **DELETE** | Tests `get_data()` loading `.npy` files. Dead code path. DataFetcher is canonical. |
| `test_scaling_parity.py` | **DELETE** | Tests hardcoded JIT log1p. FeatureScaler is canonical. Keeping this risks false confidence that scaling is correct when it may have diverged. |
| `test_utils_df_to_vol_conversion.py` | **DELETE** | Tests `df_to_vol()`, `vol_to_df()`. VolumeHandler is canonical. |
| `test_native_parity.py` | **MIGRATE** | Tests output consistency across runs. This is a valid property that belongs in `tests/` under a new name, updated to use the current VolumeHandler API. |
| `test_manager_smoke.py` | **ARCHIVE** | Keep for manual local runs with full data, but mark as `pytest.mark.skip` with reason: "requires external data artifacts". |
| `test_train_smoke.py` | **ARCHIVE** | Same as above. |
| All others | **REVIEW** | Evaluate individually against current API. If the function being tested no longer exists, delete. If it tests a current function via old API, migrate or delete. |

**Process:** Do not delete silently. For each deleted file, write one paragraph in a `legacy_tests/DISPOSITION.md` explaining what was deleted and why. This preserves the historical record without keeping dead code in the repo.

---

## Part 5: Implementation Sequence

The tests must be written in this specific order. The order is not arbitrary — it reflects the dependency graph between tests and fixes, and the priority of closing the active investigation.

### Phase 1 — Red Gates First (Write Failing Tests Before Fixes)
These tests must be written BEFORE the corresponding fixes. A test written after a fix is a documentation exercise, not a safety net.

1. `test_gate_17_negative_offset_rejection()` — write it, watch it fail, then add the guard to `VolumeHandler.from_df()`
2. `test_weight_init_xavier_norm_is_not_silent()` — write Phase A, watch it fail, then add the `'xavier_norm'` handler to `init_weights()`
3. `test_datasniffer_detects_offset_drift()` — write it, watch it fail, then implement the check in `DataSniffer.sniff_forecast_alignment()`
4. `test_mtloss_rejects_mismatched_task_count()` — write it, determine whether MultiTaskLoss should raise or if the training loop should validate before calling

### Phase 2 — Value Tests (No Code Changes Required)
These tests verify existing behaviour more thoroughly. They should pass on the current codebase if the system is working correctly. If they fail, that is a discovery.

5. `test_gate_16_spatial_gradient_preservation()` — this is the pivotal test. If it passes, Fault Line 2 is closed. If it fails, we have confirmed spatial scrambling.
6. `test_gate_18_value_round_trip()` — verifies bit-perfect value preservation through the full DF→Volume→DF pipeline
7. `test_collapse_mathematical_correctness()` — verifies collapse math

### Phase 3 — Corrections (No New Behaviour)
These do not change what is tested, only how clearly failures are reported.

8. Add diagnostic messages to all bare assertions
9. Fix random seeds in all non-deterministic tests
10. Document magic indices in `test_vader_bridge.py`
11. Update `test_optimization_gate.py` mock to enforce shape contract

### Phase 4 — New ADR 005 Coverage
12. `test_scaler_rejects_unmapped_columns()`
13. `test_weight_init_xavier_norm_is_not_silent()` Phase B (after fix)
14. FeatureScaler NaN/Inf injection test (Red Team gap)

### Phase 5 — Legacy Cleanup
15. Execute the legacy test dispositions from Part 4
16. Write `legacy_tests/DISPOSITION.md`

---

## Part 6: Success Criteria

The test suite remediation is complete when all of the following are true:

1. **`test_gate_17` fails on the current codebase and passes after the VolumeHandler guard is added.** This is non-negotiable. The guard must exist.

2. **`test_weight_init_xavier_norm_is_not_silent()` Phase A fails on the current codebase and Phase B passes after the fix.** The `'xavier_norm'` handler must exist and be verified.

3. **`test_gate_16_spatial_gradient_preservation()` passes.** This is the automated equivalent of the Visual Diagnostics Gradient Test. If it passes, we have machine-verified that the VolumeHandler correctly maps spatial coordinates for any valid config.

4. **`test_datasniffer_detects_offset_drift()` fails on the current codebase and passes after the DataSniffer is updated.** The CIC promise must be implemented and tested.

5. **All tests in `tests/` pass with explicit assert messages.** No bare assertions.

6. **All tests in `tests/` use fixed seeds.** No non-deterministic tests.

7. **`legacy_tests/` is either empty or contains only `pytest.mark.skip`-annotated smoke tests with a `DISPOSITION.md` explaining the decisions.**

8. **ADR 005 compliance table (from the audit) shows no component with zero Red Team coverage.** Every component must have at least one adversarial test.

---

## Appendix: Why the Gradient Test Is the Load-Bearing Test

Among all the tests in this plan, `test_gate_16_spatial_gradient_preservation()` deserves special emphasis. It is not the most urgent (that is `test_gate_17`) but it is the most *load-bearing* in the long run.

The entire failure mode under investigation — model output indistinguishable from random noise — is consistent with spatial scrambling. The Visual Diagnostics Plan was written to detect spatial scrambling visually. The forensic analysis identified it as the highest-probability root cause. But visual inspection is not a test. It is a debugging tool. It requires a human, a working display, and a subjective judgment about whether a gradient "looks smooth."

The gradient preservation test converts this visual judgment into a machine-verifiable invariant. It says: *the VolumeHandler must, at all times, under all valid configs, produce a volume where a monotonic spatial gradient in the input corresponds to a monotonic spatial gradient in the output.* This is not a property that can be accidentally broken during a refactor and go unnoticed. Once this test exists, spatial scrambling — the failure mode that may have silently destroyed the model's performance — becomes impossible to introduce without a failing test catching it immediately.

That is what ADR 005 means by "critical infrastructure." Not infrastructure that catches bugs after they happen. Infrastructure that makes certain classes of bugs impossible to deploy.
