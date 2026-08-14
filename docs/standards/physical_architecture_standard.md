# Physical spatiotemporal Invariance (ADR-025)

This standard defines the mandatory structural rules for this repository to ensure **predictable discovery** and **absolute maintainability**.

---

## 1. The 1-Class-1-File Standard

**Every non-trivial class must live in its own file named after the class in `snake_case`.**

- **Correct:** `ClassName` lives in `class_name.py`.
- **Incorrect:** Bundling multiple classes in `utils.py` or `models.py`.
- **Exception:** Trivial data containers or exceptions directly related to a class may coexist in the same file.

---

## 2. Directory Ontology (Ontological Separation)

Files must be located in directories that match their **functional category**.

- `architectures/`: Model/architecture definitions (training loop lives in `train/`).
- `distributions/`: The ADR-067 output-distribution family subsystem.
- `utils/`: Mathematical operations, loss functions, and optimizers.
- `infrastructure/`: Callbacks, logging, and hardware management.
- `data/`: Data loading and transformation pipelines.

---

## 3. Symmetrical Hubs

Heterogeneous logic (patches, third-party callbacks, exceptions) must be consolidated into **Symmetrical Hubs** to prevent logic fragmentation.

These are **reserved conventions** — create the module when the need first arises; none exist today:

- `utils/patches.py`: monkey-patches or framework fixes.
- `utils/callbacks.py`: framework/third-party training callbacks (this repo currently uses none — no
  PyTorch Lightning or Darts).
- `utils/exceptions.py`: custom project-wide exceptions (errors are currently raised inline per ADR-008).

---

## 4. Import Conventions

- **Explicit Imports:** Avoid `from module import *`.
- **Circular Dependency Guard:** Follow ADR-002 to ensure a hierarchical dependency tree. Components in `utils/` must not depend on `architectures/`.

---

## 5. Enforcement

Compliance with this standard is verified during **ADR Compliance Audits**. PRs violating this standard will be rejected until the structure is rectified.

🖖 **"The structure of the files is as rigorous as the logic of the code."**
