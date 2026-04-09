# ADR 036: Structured Dependency Scaffolding (Typed Context)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Replacing Raw Dicts with Typed Dependency Injection |
| ADR Number          | 036               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
Currently, the HydraNet pipeline relies heavily on passing a raw `config: dict` across all components. While this centralizes configuration, it violates **Law 2 (Zero-Magic)** because component requirements are "hidden" behind string keys. Accessing `config["features"]` throughout the codebase is prone to typos and lacks IDE support/type-checking.

## 2. Decision: The "Context-Aware" Architecture
We transition from raw dictionary passing to a **Typed Dependency Injection** pattern.

### 2.1 The Dependency Container (`HydraContext`)
We introduce a structured dataclass or Pydantic model (`HydraContext`) that serves as the single source of truth for a run.
*   **Encapsulation:** Instead of passing the whole config, we pass a subset of typed dependencies (e.g., `DeviceContext`, `SpatialContext`).
*   **Validation:** Dependencies are validated once at the entry point (`HydranetManager`) and then passed as immutable objects.

### 2.2 Component Handshakes
Components (e.g., `BacktestOrchestrator`, `HydraNetInference`) will now accept a context object:
```python
# PROPOSED PATTERN
def __init__(self, ctx: HydraContext, model: nn.Module):
    self.device = ctx.device
    self.spatial = ctx.spatial  # Includes height, width, offset
```

## 3. Consequences

**Positive Effects:**
- **Type Safety:** IDEs can autocomple attributes, eliminating string-key errors.
- **Explicit Requirements:** A component's signature clearly states exactly what it needs from the environment.
- **Easier Testing:** Mocking a typed object is simpler and safer than constructing a complex nested dictionary.

**Negative Effects:**
- **Refactoring Overhead:** Requires updating signatures across the core utilities.

## 4. Rationale
In a Boring Architecture, we prefer explicit objects over generic containers. By moving to typed context, we turn runtime `KeyErrors` into compile-time (or lint-time) type errors, further hardening the pipeline's "Fail Loud and Proud" mandate.
