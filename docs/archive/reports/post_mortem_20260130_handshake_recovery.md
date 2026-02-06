# Post-Mortem: The Handshake Recovery (30-01-2026)

## 1. Incident Summary
A fatal crash during a critical refactor resulted in the physical loss of core methods and introduced deep-seated regressions in the model orchestration layer. The system entered a "fragile state" where tests passed despite production code being broken, primarily due to inconsistent handling of inherited state and implicit dependencies.

## 2. The Regressions
1.  **Orchestration Signature Mismatch:** `HydranetManager` called a base class training method that invoked a local override (`_train_model_artifact`) without the required arguments, causing a `TypeError`.
2.  **Attribute Amnesia:** The manager relied on `self.model_path` from the base class, which was lost or uninitialized in certain execution paths, leading to `AttributeErrors`.
3.  **Config Propagation Failure:** Misspelled hyperparameters (e.g., `logp1`) bypassed validation and crashed the model hours into execution during the JIT scaling phase.
4.  **Logging Deadlocks:** `wandb` calls were not guarded, causing hard crashes if training started without an active remote session.

## 3. Root Causes
- **Implicit Dependency on Unowned Memory:** Relying on the base class's `configs` and `model_path` properties created a "Property Trap"—accessing them triggered logic in the parent class that expected state we hadn't set.
- **Silent Validation:** Validation errors were logged but didn't always halt execution, or were reported with empty error lists due to narrow filtering.
- **Global Test Pollution:** Tests were mutating the `HydranetManager` class globally (`type(m).config = ...`), creating a false sense of security where broken code worked only because the tests "fixed" it at runtime.

## 4. Corrective Actions Taken
- **"Safe-Mode" Architecture:** Implemented internal, guaranteed state holders (`_hydranet_config`, `_model_path`) that bypass base class properties.
- **The Strict Handshake:** Unified all config validation into a single Pydantic-powered boundary check that fails fast and provides detailed, field-specific reports.
- **Autonomous Training:** Overrode the training orchestration to explicitly load data volumes and manage signatures locally.
- **Resilient Utilities:** Wrapped `wandb` and JIT scaling in safety checks to ensure they only run when valid state is present.

## 5. Lessons Learned
- **Own Your State:** Never rely on base class properties for critical filesystem paths or configuration in an override.
- **Fail-Fast at the Boundary:** Configuration must be validated at the entry point of the manager, not deep in the execution loop.
- **Test Isolation:** Avoid `type(Class).property = ...` in tests as it contaminates the global environment. Prefer instance-level overrides or standard attribute setting.
