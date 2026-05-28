# ADR 053: Build and Package Management Tooling

| ADR Info            | Details                                      |
|---------------------|----------------------------------------------|
| Subject             | Build backend and package manager for views-hydranet |
| ADR Number          | 053                                          |
| Status              | Waiting                                      |
| Author              | Simon / Claude                               |
| Date                | 28.05.2026                                   |
| Implementation Plan | [`reports/implementation_plan_adr053.md`](../../reports/implementation_plan_adr053.md) |

---

## 1. Context

**Why is this being considered?**

The VIEWS platform is splitting into two tooling stacks. Eight older repositories (including views-hydranet) use **poetry-core** as the build backend and **poetry** as the package manager. Four newer repositories (views-bayesian, views-datafactory, views-lab00, views-metric-lab) use **hatchling** as the build backend and **uv** as the package manager.

views-hydranet currently uses poetry-core for building and conda for environment management — a hybrid that works but is not aligned with either stack. The project is a monolithic single package (`views_hydranet/`) and has no architectural need for multi-package layout, which was the primary driver for views-bayesian's adoption of hatchling (see views-bayesian ADR-013). The question is whether the secondary benefits — resolution speed, standards compliance, reproducibility, and platform convergence — justify the migration for a monolithic project.

**Current state of views-hydranet tooling:**

| Aspect | Current Setup |
|--------|--------------|
| Build backend | `poetry.core.masonry.api` (line 17 of `pyproject.toml`) |
| Package layout | Monolithic: `views_hydranet/` with 6 subpackages |
| Metadata format | **PEP 621** `[project]` — already standard (unusual for a poetry project) |
| Dev dependencies | `[tool.poetry.group.dev.dependencies]` — poetry-specific |
| Version specifiers | Mixed: PEP 508 for torch, poetry syntax `(>=x,<y)` for views-pipeline-core |
| Lock file | `poetry.lock` present and committed |
| Environment manager | conda (`views-hydranet-env`) — not poetry's built-in venv |
| Torch installation | pip wheels via pypi (not conda pytorch channel) |
| CUDA toolkit | pip-installed `nvidia-cuda-*` packages (not conda-managed) |

**Key observation:** The project already uses PEP 621 `[project]` metadata instead of `[tool.poetry]`. This is atypical for a poetry project and means the migration surface is smaller than usual. The only poetry-specific sections are the build backend declaration and the dev dependency group.

**Key observation:** Torch and CUDA are installed from PyPI via pip wheels, not from the conda `pytorch` channel. This eliminates the primary argument for conda over uv — conda's advantage is CUDA toolkit management, but this project doesn't use it.

---

## 2. Decision

Migrate views-hydranet from **poetry-core + conda** to **hatchling + uv**.

- Build backend: `hatchling.build`
- Package manager: `uv`
- Lock file: `uv.lock` (replacing `poetry.lock`)
- Developer commands: `uv run pytest`, `uv run ruff check .`, etc.
- Environment: uv-managed `.venv/` (replacing conda `views-hydranet-env`)

**This ADR is in Waiting status.** Implementation is deferred until either:
- The VIEWS platform makes a platform-wide standardization decision, or
- A concrete pain point forces the migration (e.g., poetry resolution failures with new Python/torch versions, conda environment drift)

See the [implementation plan](../../reports/implementation_plan_adr053.md) for the step-by-step migration procedure.

---

## 3. Rationale

### 3.1 Why migrate at all?

**Resolution speed.** views-hydranet depends on `torch>=2.2.1` (~2 GB) and `views-pipeline-core>=2.0.0` (which pulls in pandas, numpy, wandb, and dozens of transitive dependencies). A cold `poetry install` or `conda install` takes 2-5 minutes to resolve this tree. `uv sync` completes in seconds. For a project that runs sweeps (20 iterations per sweep, each requiring a fresh environment in CI), this difference compounds.

**Reproducibility.** The `poetry.lock` file is present but the conda environment is the actual runtime — and conda environments have no project-scoped lock file. The environment could drift between machines or over time. `uv.lock` locks the exact environment that `uv run` uses.

**Standards compliance.** The `pyproject.toml` already uses PEP 621 `[project]` metadata. Migrating the build backend and dev dependencies to PEP standards (hatchling, PEP 735 dependency groups) makes the project fully tool-agnostic — any PEP 517-compliant installer can build it.

**Platform convergence.** Four VIEWS repositories already use hatchling + uv. If the platform standardizes (which views-bayesian ADR-013 leaves as an open question), early migration reduces future disruption.

### 3.2 Why not migrate now?

**The current setup works.** 617 tests pass. Training, evaluation, forecasting, and sweeps all function correctly. The conda environment is stable. There is no acute pain point.

**Contributor disruption.** Developers familiar with `conda activate views-hydranet-env` and `pytest` would need to switch to `uv run pytest`. This is a small change but requires coordination.

**Risk timing.** The development branch has active feature work (sampling strategies, loss functions, sweep infrastructure). Changing the build/environment tooling mid-sprint introduces unnecessary risk. Better to migrate during a stabilization window.

### 3.3 Why hatchling specifically?

views-hydranet is monolithic — it doesn't need hatchling's multi-package support. The choice is driven by **platform alignment** (matching views-bayesian, views-datafactory, views-lab00) rather than a technical requirement unique to this project. If the platform standardized on a different PEP 621-native build backend (e.g., flit-core), that would work equally well for views-hydranet. Hatchling is chosen for consistency.

---

## 4. Considered Alternatives

### Alternative A: Stay on poetry-core + conda (status quo)

- **Pros:** Zero migration effort. Familiar to current contributors. Working and tested.
- **Cons:** Diverges from newer VIEWS repos. conda environment is not locked to the project. Resolution is slow. Dev dependencies use poetry-specific syntax.
- **Reason deferred, not rejected:** This is viable indefinitely. The migration is a quality-of-life improvement, not a correctness fix.

### Alternative B: Migrate to hatchling + uv now

- **Pros:** All benefits of the decision, realized immediately.
- **Cons:** Disrupts active development. Requires updating CLAUDE.md, CI instructions, contributor guides, and memory files. Risk of environment issues during feature work.
- **Reason deferred:** No acute pain point justifies the timing risk.

### Alternative C: Keep poetry-core, switch to uv as package manager only

- **Pros:** Gets the speed benefit (uv) without changing the build backend. Minimal `pyproject.toml` changes.
- **Cons:** uv can install poetry-core projects, but dev dependencies in `[tool.poetry.group.*]` are poetry-specific — uv doesn't read them. Would still need to migrate dev deps to PEP 735. Leaves the project on a non-standard build backend for no benefit.
- **Reason rejected:** If we're touching the dev deps anyway, switching the build backend is 2 additional lines of change. Half-migrating creates a third tooling stack.

### Alternative D: Keep conda, switch build backend only

- **Pros:** Modernizes the build backend without changing the developer workflow (still `conda activate` + `pytest`).
- **Cons:** Doesn't solve the resolution speed or lock file issues. Conda environment drift remains unaddressed. Gains only standards compliance.
- **Reason rejected:** The build backend alone is the least impactful part of the migration. The value is in uv's resolution and locking.

---

## 5. Consequences

### Positive (when implemented)

- `uv sync` resolves in seconds instead of minutes
- `uv.lock` provides deterministic, project-scoped environment locking
- `pyproject.toml` becomes fully PEP 621/735 compliant — portable across tools
- Aligns with views-bayesian, views-datafactory, views-lab00, views-metric-lab
- Build backend and package manager are independently replaceable

### Negative (when implemented)

- Contributors must learn `uv run` workflow (mitigation: direct analogue of `poetry run`)
- Conda environment `views-hydranet-env` must be decommissioned (mitigation: document in migration guide)
- CLAUDE.md memory entries referencing `conda run -n views-hydranet-env` become stale (mitigation: update as part of implementation plan)
- Briefly, two environment management approaches may coexist during transition

### Downstream impact

- **views-models (pink_pirate, purple_alien, etc.):** Zero impact. These install views-hydranet via `pip install`. hatchling is PEP 517-compliant — `pip install .` works identically regardless of build backend.
- **views-pipeline-core:** views-hydranet depends on it (not the reverse). No impact.
- **CI:** Must be updated to use `uv` instead of conda. The `astral-sh/setup-uv` GitHub Action handles installation.

---

## 6. Verification & Monitoring

- **Pre-migration gate:** Full test suite must pass with `uv run pytest` before removing conda environment
- **Cross-tool build test:** `pip install .` in a clean venv must succeed — validates PEP 517 compliance
- **Lock file freshness:** `uv.lock` committed to version control; stale locks cause `uv sync` warnings
- **Rollback path:** Keep `poetry.lock` and conda environment for 2 weeks after migration. If issues arise, revert `pyproject.toml` and restore conda env.
- **Reconsider if:** uv development stalls; platform decides to standardize on poetry; a conda-only dependency is added (e.g., CUDA toolkit from conda channel)

---

## 7. References

- views-bayesian [ADR-013: Build and Package Management Tooling](../../../views-bayesian/docs/ADRs/013_build_and_package_management_tooling.md) — the template decision for newer VIEWS repos
- [PEP 517 — Build system interface](https://peps.python.org/pep-0517/)
- [PEP 621 — Project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [PEP 735 — Dependency groups](https://peps.python.org/pep-0735/)
- [Implementation Plan](../../reports/implementation_plan_adr053.md) — step-by-step migration procedure
