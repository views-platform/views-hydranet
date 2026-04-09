# Architectural Manifesto: Hardening the Evaluation Handshake

## 1. Executive Mission: The Handover Protocol
This document serves as the authoritative, foundation-level context for a structural evolution of the `views_pipeline_core`. 

**The Goal**: Evolve the pipeline from a **Static Data Repository** (Passive) to an **Instructional Data Factory** (Active).

**The Handover Agent must understand**: Touching the core is a high-risk operation. This change is designed to be the *only* change required to support HydraNet and all future "Manufacturing" models. We are resolving technical debt by introducing a standard protocol where the Core provides the material and the Model provides the manufacturing logic.

---

## 2. Technical Context & History
### 2.1 The HydraNet Evolution (ADR 046)
Until recently, the `views` ecosystem operated on **"Implicit Discovery"**. We assumed that if a user wanted to evaluate a signal like `by_sb_best`, that column already existed in the database or the parquet file on disk. 

HydraNet has broken this assumption via **ADR 046 (Symmetric Feature Lifecycle)**. We now use an **Instructional Blueprint**. The configuration contains `derivations`: instructions that tell the model manager how to manufacture targets (e.g., "if counts > 0, then 1, else 0"). 

### 2.2 The Ontological Clash (The KeyError)
The conflict arises during the evaluation phase:
1. **HydraNet Manager** knows how to manufacture `by_sb_best` using the Blueprint.
2. **Pipeline Core** doesn't know manufacturing exists. It only knows how to `read_dataframe()`.
3. **The Crash**: During evaluation, the Core loads raw data from disk and **immediately** tries to slice it using `config["targets"]`. Because the manufactured columns don't exist on disk, the system throws a `KeyError`.

**Philosophical Pivot**: We are moving from **Ontological Assumption** (columns exist) to **Ontological Ownership** (the Model Manager is the boss of the ground truth).

---

## 3. Phase 1: Foundation Hardening (views_pipeline_core)
The core library must be modified to allow a "Standardization Handshake" during evaluation.

### 3.1 Step 1: Defining the Generic Protocol
**Location**: `views_pipeline_core/managers/model/model.py`
**Class**: `ModelManager` (The base class for all managers)

We must introduce a "Preparation Hook." It must be a no-op by default to ensure **absolute backwards compatibility** with all existing models.

**Code Signature**:
```python
    def prepare_viewser_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interface hook for model-specific data preparation.

        Allows subclasses to augment or transform the ground-truth DataFrame
        (Actuals) before the evaluation logic slices it. By default, returns
        the DataFrame unchanged. 
        """
        return df
```

### 3.2 Step 2: Activating the Handshake
**Location**: `views_pipeline_core/managers/model/model.py`
**Method**: `_evaluate_prediction_dataframe` (Approx line 2670)

The core must transition from a "Naked Load" to a "Sanctified Handshake."

**Hardened Logic**:
```python
        # 1. Load the raw material
        df_viewser = read_dataframe(df_path)
        
        # 2. THE HANDSHAKE: Give the manager a chance to manifest its blueprint
        df_viewser = self.prepare_viewser_df(df_viewser) 
        
        # 3. Proceed with slicing
        logger.info(f"df_viewser read and prepared from {df_path}")
        df_actual = df_viewser[self.configs["targets"]]
```

---

## 4. Phase 2: Implementation at the Edge (views-hydranet)
Once the core is hardened, HydraNet will "Fulfill the Contract."

### 4.1 Step 1: The Blueprint Engine (DataFetcher)
We will implement `DataFetcher.apply_blueprint(df, config)`. This static utility serves as the "Manufacturing Floor," executing the math defined in the config.

**Logic**:
- Iterate through `config['derivations']`.
- Apply operations (e.g., `binary` thresholding).
- This ensures the ground truth used for Evaluation is bit-perfect with the ground truth used for Training.

### 4.2 Step 2: Fulfilling the Contract (HydranetManager)
We will override the hook in `HydranetManager`:
```python
    def prepare_viewser_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uses DataFetcher to apply the instructional blueprint."""
        return DataFetcher.apply_blueprint(df, self.configs)
```

---

## 5. Risk Matrix & Mitigation
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Breaking Existing Models** | Critical | The base hook MUST be a no-op (`return df`). Every model manager inherits from this; it cannot be abstract. |
| **Performance Degradation** | Low | The hook is only called once per evaluation partition (e.g., once for 'validation', once for 'test'). |
| **Type Mismatch** | Medium | Ensure the hook signature explicitly accepts and returns a `pd.DataFrame`. |
| **Core File Corruption** | High | Handover agent must use surgical `replace` calls. `model.py` is too large for `write_file`. |

---

## 6. Architectural Rationale: Why This Is Not a Hack
### 6.1 Decoupling via Protocol
By adding `prepare_viewser_df` to the base class, we are not adding "HydraNet-specific code" to the core. We are adding a **generic protocol**. The core remains data-agnostic; it simply agrees to let the manager handle the preparation.

### 6.2 Model Ownership
This follows the principle: **"What changes together, stays together."** The definition of `by_sb_best` lives in the HydraNet config. The logic to create it should live in the HydraNet manager. 

### 6.3 Scalability
This solution scales to any future model. If a model needs to calculate "Lags" or "Interaction Terms" on-the-fly for evaluation, it can now do so by overriding the same hook.

---

## 7. Implementation Protocol for Handover Agent
1. **Verification**: Confirm `import pandas as pd` exists in `model.py`.
2. **Surgical Hook**: Inject `prepare_viewser_df` into `ModelManager`.
3. **Surgical Call**: Insert the call into `_evaluate_prediction_dataframe` exactly after the `read_dataframe` call.
4. **Smoke Test**: Run an existing model (like `purple_alien`). It should proceed without noticing the change.

---

## 8. Success Definition
Success is achieved when `views_pipeline_core` no longer crashes during HydraNet evaluation, because the `prepare_viewser_df` handshake has manufactured the missing signals required by the `config["targets"]` slice.
