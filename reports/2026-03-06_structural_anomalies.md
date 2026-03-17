# Structural Anomalies — Backlog Note

Date: 2026-03-06
Source: Structural audit (feature/sample_for_fao)

---

## 1. Filename typo: `shringkage_loss.py`

**Anomaly:** `utils/shringkage_loss.py` is missing the 'i' — should be `shrinkage_loss.py`.
The corresponding test file (`tests/test_shrinkage_loss.py`) already uses the correct spelling.

**Risk:** Any import using the misspelled name is fragile; a refactor or autocomplete will produce the wrong path.

**Suggestion:** Rename the file to `shrinkage_loss.py`, update the import in `utils/utils.py` (the sole internal importer), and verify no other references exist.

---

## 2. Duplicate README: `views_hydranet/README.md`

**Anomaly:** `views_hydranet/README.md` is a byte-for-byte duplicate of the root `README.md`. Two copies will inevitably diverge and become contradictory.

**Suggestion:** Delete `views_hydranet/README.md` and, if a package-level doc is desired, replace it with a short stub that references the root README.

---

## 3. `data_sniffer.py` — `TYPE_CHECKING` import guard

**Anomaly:** `utils/data_sniffer.py` imports some symbols only under `TYPE_CHECKING`. This means the runtime import set differs from the static analysis import set. mypy and IDEs will see dependencies that do not exist at runtime.

**Risk:** If the guarded imports are ever used outside of type annotations (e.g. in an isinstance check or at class body level), it will raise a NameError at runtime.

**Suggestion:** Audit the guarded imports. If they are only used in annotations, add `from __future__ import annotations` at the top of the file (PEP 563) to make all annotations strings by default, which is the canonical pattern. Document why the guard exists.

---

## 4. Three untracked test files

**Anomaly:** The following test files exist on disk but are not tracked by git:

- `tests/test_inference_logic.py`
- `tests/test_sweep_and_hardening_gates.py`
- `tests/test_temporal_causality_audit.py`

**Risk:** They are invisible to CI and could be lost or inconsistently maintained.

**Suggestion:** `git add` them in the next commit, or explicitly add them to `.gitignore` if they are intentionally excluded (with a comment explaining why).

---

## 5. No CI configuration found

**Anomaly:** No `.github/workflows/`, `.gitlab-ci.yml`, `Makefile`, or equivalent CI entrypoint exists in the repository.

**Risk:** Tests, linting (ruff), and type checking (mypy) are only run manually. Regressions can merge undetected.

**Suggestion:** Add a minimal CI pipeline (e.g. GitHub Actions) that runs `pytest`, `ruff check`, and `mypy` on push and pull request. The tooling is already configured in `pyproject.toml` — it just needs a runner.
