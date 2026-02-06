# Plan: VolumeHandler Refactor (Hygienic/Explicit API)

**Goal:** Replace the brittle, polymorphic `to_df` method in `VolumeHandler` with three explicit, purpose-built methods. This enforces strict architectural separation between History, Evaluation (Backtesting), and Forecasting (Operations).

---

## 1. Architectural Changes

We will remove `to_df` and introduce the following API:

### A. `to_historical_df()`
*   **Context:** Used when the volume *is* the history (e.g., input validation).
*   **Behavior:** Uses its own `priogrid_gid` channel to mask Ocean cells.
*   **Dependencies:** None (Self-contained).

### B. `to_evaluation_df(history: VolumeHandler, start_idx: int)`
*   **Context:** Used during rolling-origin evaluation (Backtesting).
*   **Behavior:**
    1.  Calculates its own duration (T).
    2.  Slices the provided `history` handler from `[start_idx : start_idx + T]`.
    3.  Uses this *slice* as the identity provider (Truth).
*   **Dependencies:** Requires the full History volume and the start index of the prediction window.

### C. `to_forecast_df(history: VolumeHandler)`
*   **Context:** Used during operational forecasting (Future).
*   **Behavior:**
    1.  Calculates its own duration (T).
    2.  Extrapolates the provided `history` handler *into the future* by T steps.
        *   Copies static identities (`row`, `col`, `priogrid_gid`, `c_id`).
        *   Increments `month_id`.
    3.  Uses this *synthetic scaffold* as the identity provider.
*   **Dependencies:** Requires the History volume (to clone the static grid).

### D. `_reconstruct_from_provider(provider: VolumeHandler)` (Private)
*   **Context:** Shared implementation detail.
*   **Behavior:**
    1.  Extracts data from `self` (Predictions) and `provider` (Identities).
    2.  Aligns axes ([H, W, T, C]).
    3.  Resolves `priogrid_gid` index dynamically from the *provider's* channel map.
    4.  Creates the Boolean Mask (`provider[..., pg_idx] > 0`).
    5.  Flattens and combines columns into a DataFrame.

---

## 2. Implementation Steps

### Step 1: Create `_reconstruct_from_provider`
Refactor the existing logic from `to_df` into a private helper method. This method will take a *guaranteed aligned* provider and perform the mechanical flattening.

### Step 2: Implement `to_historical_df`
Simple wrapper that calls `_reconstruct_from_provider(self)`.

### Step 3: Implement `to_evaluation_df`
Implement the slicing logic (wrapping the existing `slice_time` helper) and delegate to `_reconstruct_from_provider`.

### Step 4: Implement `to_forecast_df`
Implement the extrapolation logic (wrapping the existing `extrapolate_time` helper) and delegate to `_reconstruct_from_provider`.

### Step 5: Update `HydranetManager`
*   Replace `to_df(identity_provider=slice)` with `to_evaluation_df(history, origin+1)`.
*   Replace `to_df(identity_provider=future)` with `to_forecast_df(history)`.

### Step 6: Verify
*   Run the verification scripts (or `test_forecast_contract.py`) to ensure no regressions.
