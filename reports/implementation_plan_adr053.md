# Implementation Plan: ADR-053 — Build and Package Management Migration

**ADR:** [053 — Build and Package Management Tooling](../docs/ADRs/proposed/053_build_and_package_management_tooling.md)
**Status:** Not started (ADR is Waiting)
**Estimated effort:** 1-2 hours of hands-on work + 1 week soak period
**Risk level:** Low (no source code changes, only tooling configuration)

---

## Pre-Conditions

Before starting this migration:

- [ ] Development branch is merged to main (no active feature work)
- [ ] Full test suite passes on the current setup
- [ ] Platform-wide tooling decision has been made, OR a concrete pain point justifies solo migration
- [ ] `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

---

## Current State

```
pyproject.toml
├── [project]                              ← PEP 621 (already standard)
│   ├── name, version, description, etc.
│   ├── requires-python = ">=3.11,<3.15"
│   └── dependencies = [
│       "views-pipeline-core (>=2.0.0,<3.0.0)",  ← poetry version syntax
│       "torch>=2.2.1,<3.0.0"                     ← PEP 508 syntax
│   ]
├── [build-system]                         ← poetry-core
│   ├── requires = ["poetry-core>=2.0.0"]
│   └── build-backend = "poetry.core.masonry.api"
├── [tool.poetry.group.dev.dependencies]   ← poetry-specific dev deps
│   ├── ruff, mypy, pytest, pre-commit, pytest-cov
└── [tool.ruff], [tool.mypy]              ← tool config (unchanged)

Environment: conda (views-hydranet-env)
Lock file: poetry.lock
```

---

## Step-by-Step Migration

### Step 1: Update `pyproject.toml`

**1a. Switch build backend** (2 lines)

```toml
# BEFORE
[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

# AFTER
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**1b. Add hatchling package declaration** (new section, 3 lines)

```toml
[tool.hatch.build.targets.wheel]
packages = ["views_hydranet"]
```

**1c. Fix poetry version syntax in dependencies** (1 line)

```toml
# BEFORE
"views-pipeline-core (>=2.0.0,<3.0.0)",

# AFTER
"views-pipeline-core>=2.0.0,<3.0.0",
```

**1d. Migrate dev dependencies to PEP 735** (replace section)

```toml
# BEFORE
[tool.poetry.group.dev.dependencies]
ruff = ">=0.14.14,<0.15.0"
mypy = ">=1.19.1,<2.0.0"
pytest = "<9.0.0"
pre-commit = ">=4.5.1,<5.0.0"
pytest-cov = ">=5.0.0,<6.0.0"

# AFTER
[dependency-groups]
dev = [
    "ruff>=0.14.14,<0.15.0",
    "mypy>=1.19.1,<2.0.0",
    "pytest<9.0.0",
    "pre-commit>=4.5.1,<5.0.0",
    "pytest-cov>=5.0.0,<6.0.0",
]
```

### Step 2: Generate uv lock file

```bash
uv lock
```

This creates `uv.lock` from the `pyproject.toml` dependencies. Inspect the output for resolution conflicts — particularly around torch wheels and views-pipeline-core.

### Step 3: Sync and verify environment

```bash
uv sync                          # Creates .venv/ and installs all deps
uv run python -c "import views_hydranet; print('OK')"
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Verify that torch CUDA support works. The pip wheels (`nvidia-cuda-*`) should resolve identically to the current conda setup since the current conda env also uses pip-installed torch.

### Step 4: Run full test suite

```bash
uv run pytest tests/ -v --ignore=tests/test_eval_integration_toy.py
uv run ruff check .
uv run ruff format --check .
```

All tests must pass. If any fail, compare against the conda baseline — failures must be migration-related, not pre-existing.

### Step 5: Verify downstream compatibility

```bash
# In a clean temporary venv, verify pip install works
python -m venv /tmp/test-hydranet-install
/tmp/test-hydranet-install/bin/pip install .
/tmp/test-hydranet-install/bin/python -c "from views_hydranet.manager.hydranet_manager import HydranetManager; print('OK')"
rm -rf /tmp/test-hydranet-install
```

This confirms PEP 517 compliance — downstream consumers using `pip install` see no change.

### Step 6: Update project documentation

**CLAUDE.md:** Replace any `conda run -n views-hydranet-env` references with `uv run`.

**Memory files:** Update `feedback_use_conda_env.md` to reference `uv run` instead of conda.

**README.md:** Update developer setup instructions.

### Step 7: Clean up old tooling artifacts

```bash
rm poetry.lock                   # Replaced by uv.lock
git add uv.lock pyproject.toml
git rm poetry.lock
```

Do NOT remove the conda environment yet — keep it for 2 weeks as a rollback path.

### Step 8: Decommission conda environment (after soak period)

After 2 weeks with no issues:

```bash
conda env remove -n views-hydranet-env
```

---

## What Changes for Developers

| Task | Before (conda) | After (uv) |
|------|----------------|------------|
| Setup environment | `conda create -n views-hydranet-env python=3.11` + manual pip installs | `uv sync` |
| Run tests | `conda run -n views-hydranet-env pytest` | `uv run pytest` |
| Run linting | `conda run -n views-hydranet-env ruff check .` | `uv run ruff check .` |
| Add a dependency | Edit `pyproject.toml` + `pip install` | `uv add <pkg>` |
| Update lock file | `poetry lock` | `uv lock` |
| Activate environment | `conda activate views-hydranet-env` | `source .venv/bin/activate` (optional — `uv run` doesn't need activation) |

---

## What Does NOT Change

- **Source code:** Zero modifications to any `.py` file
- **Test suite:** All tests run identically
- **Package structure:** `views_hydranet/` remains monolithic with 6 subpackages
- **Dependencies:** Same packages, same versions, same constraints
- **Downstream consumers:** `pip install views-hydranet` works identically (PEP 517)
- **CI logic:** Same commands, different prefix (`uv run` instead of `conda run`)
- **ruff/mypy config:** Unchanged (`[tool.ruff]`, `[tool.mypy]` sections are tool-agnostic)

---

## Rollback Procedure

If issues are discovered after migration:

1. `git checkout HEAD~1 -- pyproject.toml poetry.lock` — restore old config
2. `conda activate views-hydranet-env` — switch back to conda
3. `poetry install` — restore poetry-managed deps (if needed)
4. Revert CLAUDE.md and documentation changes

The rollback is mechanical and takes < 5 minutes. No source code is affected.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| torch CUDA wheels fail under uv | Low | High | Verify CUDA in Step 3. Current setup already uses pip wheels, not conda channel — same resolution path. |
| views-pipeline-core resolution conflict | Low | Medium | uv's resolver is stricter than pip's. If it fails, pin the exact version that works. |
| Test failures from environment differences | Low | Medium | Run full suite in Step 4. Compare against conda baseline. |
| Contributor confusion | Medium | Low | Document in CLAUDE.md. `uv run` is a direct analogue of `conda run`. |
| CI breakage | Low | Medium | Update CI workflow in same commit. Test in a branch first. |

**Overall risk: Low.** No source code changes. Pure tooling configuration. Full rollback possible in 5 minutes.

---

## Trigger Conditions

This plan should be executed when ANY of the following occur:

1. **Platform standardization:** VIEWS platform decides to standardize on hatchling + uv
2. **Poetry resolution failure:** A Python or torch version upgrade causes poetry to fail resolution (poetry's `<3.15` upper bound is a known risk)
3. **Conda environment drift:** The conda environment diverges between contributors, causing "works on my machine" issues
4. **CI migration:** The project moves to GitHub Actions CI that already uses uv
5. **Stabilization window:** A natural break between sprints where tooling changes are low-risk
