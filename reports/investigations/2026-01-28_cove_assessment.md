# CoVe Assessment Report: views-hydranet
**Date:** 2026-01-28
**Status:** Crash Recovery & Assessment

## 1. Repository Reality Check (Verified)

### A. Test Suite Status
- **Command:** `conda run -n views-hydranet-env pytest`
- **Result:** **PASS** (51/51 tests passed)
- **Duration:** ~8 seconds
- **Warnings:** 9 warnings (DeprecationWarnings for SQLAlchemy, Pydantic, NumPy).
- **Critical Observation:** The test suite is passing, meaning the *baseline* is stable. We are not in a broken state regarding existing tests.

### B. "Crash Site" Analysis (Experimental Module)
- **Context:** The user indicated a "crash" during "experimentation of an alternative solution".
- **Findings:**
    - `views_hydranet/experimental/evaluate_model.py` and `views_hydranet/experimental/generate_forecast.py` are **identical** (verified via `diff`), except for 3 lines of debug prints in `generate_forecast.py`.
    - Both files contain *forecasting* logic (`forecast_posterior`, `forecast_with_model_artifact`), despite `evaluate_model.py`'s name.
    - **Hypothesis:** `evaluate_model.py` was likely repurposed or copy-pasted to `generate_forecast.py` to debug path/import issues (evidenced by `print("Current Working Directory:", ...)`).
- **Verdict:** We have a "split brain" situation where two files claim to do the same thing. `generate_forecast.py` appears to be the active "debug" version.

### C. Logging & Output Posture
- **Current State:** Mixed `print()` and `logging`.
- **Evidence:** 
    - `views_hydranet/utils/utils_device.py`: `print(f"Using device: {device}")`
    - `views_hydranet/utils/utils_prediction.py`: `print(..., end='\r')` (Progress bar via print)
    - `views_hydranet/experimental/generate_forecast.py`: `print("Current Working Directory:", ...)`
- **Risk:** High noise in production/CI logs. Progress reporting via `print` is brittle.

### D. Type Hints Posture
- **Current State:** Partial / Inconsistent.
- **Evidence:**
    - `views_hydranet/utils/utils.py`: `def choose_model(config: dict, device: torch.device) -> nn.Module:` (Good)
    - `views_hydranet/experimental/generate_forecast.py`: `def forecast_posterior(model, views_vol, config, device):` (Missing hints)
- **Opportunity:** The "core" utils have some typing, but the experimental/manager layers lack it.

## 2. Forward Plan (Recovery & Consolidation) 

**Strategy:** Stabilize the "experimental" fork, eliminate the duplicate, and professionalize the output.

### Step 1: Resolve Identity Crisis (Low Risk)
- **Goal:** Remove ambiguity between `evaluate_model.py` and `generate_forecast.py`.
- **Action:** 
    1.  Verify if any code imports `evaluate_model`.
    2.  If safe, delete `evaluate_model.py` (since it's a stale copy).
    3.  Promote `generate_forecast.py` (renaming/moving it if necessary) to be the canonical entry point.

### Step 2: Formalize the "Experiment" (Medium Risk)
- **Goal:** Make `generate_forecast.py` a first-class citizen.
- **Action:**
    1.  Add Type Hints to `forecast_posterior` and `forecast_with_model_artifact`.
    2.  Replace `print` debugs with proper `logging` (or remove them).
    3.  Create a basic "smoke test" for this module to prevent future regressions.

### Step 3: Pipeline Integration (High Value)
- **Goal:** Ensure `PipelineConfig` and `ModelOutputs` are used strictly.
- **Action:** Refactor the config dictionary access in `generate_forecast.py` to use typed objects (as per the original Phase 2 plan).

## 3. Immediate Next Action
**Target:** `views_hydranet/experimental/generate_forecast.py`
**Task:** "Stabilize & Type"
1.  Run the file to see if it even runs (it has debug prints, so it was meant to be run).
2.  Add type hints to its signature.
3.  Replace `print()` with `logging`.
4.  Delete the duplicate `evaluate_model.py`.

This resolves the "crash site" ambiguity and moves us back to the robustness plan.
