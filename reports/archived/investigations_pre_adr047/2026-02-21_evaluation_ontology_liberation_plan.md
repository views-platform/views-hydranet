# Architectural Manifesto: Evaluation Ontology Liberation

## 1. Executive Mission: From Gatekeeper to Passenger
This document serves as the authoritative context for a structural evolution of the `views-evaluation` library. 

**The Goal**: Transform the `EvaluationManager` from a rigid, prefix-dependent **Gatekeeper** into a data-agnostic **Passenger**.

**The Problem**: Currently, the evaluator uses "Magic Strings" (prefixes like `ln_`, `lx_`, `lr_`) to infer how to transform data back to raw counts. If it encounters a target it doesn't recognize (like HydraNet's new `by_sb_best`), it throws a `ValueError`. This architecture is brittle, prevents innovation, and violates the **Separation of Concerns**.

---

## 2. Technical Diagnosis
### 2.1 The Failure Site
Location: `views_evaluation/evaluation/evaluation_manager.py`
Method: `transform_data` (Approx line 65)

**The Legacy Logic**:
```python
        for t in target:
            if t.startswith("ln") or t.startswith("pred_ln"):
                # (Inverse Log logic)
            elif t.startswith("lx") or t.startswith("pred_lx"):
                # (Inverse Lx logic)
            elif t.startswith("lr") or t.startswith("pred_lr"):
                # (Identity logic)
            else:
                # THE CRASH SITE
                raise ValueError(f"Target {t} is not a valid target")
```

### 2.2 The Philosophical Flaw
The evaluator is currently responsible for **Inverse Transformation**. This is architecturally incorrect. The Model Manager (e.g., HydraNet) is the semantic authority; it should prepare the data for evaluation. The Evaluator should simply calculate metrics on the numbers it is given.

---

## 3. Surgical Instructions: "Ontology Liberation"
We will evolve the `transform_data` method to be "Loud on Knowns, Silent on Unknowns."

### 3.1 Step 1: Remove the Gatekeeper
We will replace the `ValueError` with a **Default Identity (No-op)**. This ensures that any model manufacturing its own targets (like HydraNet) can pass its numbers through the evaluator without being rejected.

### 3.2 The New Logic (Implementation Detail)
```python
        for t in target:
            if t.startswith("ln") or t.startswith("pred_ln"):
                # ... Keep legacy inverse log logic ...
            elif t.startswith("lx") or t.startswith("pred_lx"):
                # ... Keep legacy inverse lx logic ...
            else:
                # --- LIBERATION (Joyful) ---
                # Default to Identity logic. 
                # If we don't recognize the prefix, we assume the caller 
                # has provided pre-prepared data. No more ValueError.
                df[[t]] = df[[t]].applymap(
                    lambda x: x if isinstance(x, (list, np.ndarray)) else x
                )
```

---

## 4. Success Definition
1. **Backwards Compatibility**: Existing models using `ln_` and `lx_` continue to have their data inverted correctly.
2. **Extensibility**: New models using any prefix (like `by_`, `logit_`, `scaled_`) pass through the evaluator successfully.
3. **Decoupling**: The `views-evaluation` library no longer needs to be updated every time a model inventor creates a new type of target.

**Success is achieved when the evaluator calculates metrics on what it is given, rather than crashing on what it hasn't seen before.** 🖖🛡️⚖️
