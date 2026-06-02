# ADR 049: Sampling Strategy Registry for Anchor Selection

| ADR Info            | Details                                      |
|---------------------|----------------------------------------------|
| Subject             | Configurable importance sampling in VolumeSampler |
| ADR Number          | 049                                          |
| Status              | Experimental                                 |
| Author              | Simon / Claude                               |
| Date                | 26.05.2026                                   |

## 1. Context

VolumeSampler uses importance sampling to concentrate training on conflict-rich regions. The original implementation (ADR-011 Section 2) uses a hard threshold: cells with `activity >= threshold` are sampled uniformly; all others are excluded. CurriculumLearner controls difficulty by lowering the threshold over training (a "cooling" schedule).

This works perfectly when data is identical across runs. However, investigation into purple_alien vs bright_starship prediction divergence (May 2026) revealed that the hard threshold is a structural sensitivity amplifier. When two data backends produce near-identical but not bit-identical activity counts, cells near the threshold boundary flip between included and excluded. A single cell's activity changing from 49 to 50 produces a discrete jump in the sampling distribution, which propagates through the training trajectory via the butterfly effect.

Three amplification mechanisms were identified:

1. **Threshold discreteness** — the hard cutoff creates a discontinuous probability surface over the activity grid.
2. **Shrinkage zero-gate** — zero-valued targets in sampled windows receive zero gradient, so which windows are sampled directly controls which parameters update.
3. **Weight-update feedback loop** — different early windows produce different weight updates, shifting the loss surface for subsequent lessons.

The hard threshold is the only mechanism we can address at the sampling level. The others are architectural properties of the model.

## 2. Decision: Strategy Registry Pattern

We introduce a **sampling strategy registry** — a config-driven lookup table mapping strategy names to pure anchor-selection functions. This mirrors the existing `LOSS_REG_REGISTRY` pattern in `utils.py`.

```python
config["sampling_strategy"] = "boltzmann"  # or "threshold", "power_law", "sigmoid"
```

The default is `"threshold"`, which preserves exact production behaviour. Non-default strategies are experimental and must be opted into explicitly.

### 2.1 Why a registry, not a single replacement

- **Empirical comparison.** The "best" strategy depends on the data regime and the downstream metric. A registry makes strategies interchangeable hyperparameters — testable via sweep.
- **Backward compatibility.** Default `"threshold"` produces bit-identical behaviour to the original code. No existing model config changes.
- **Extensibility.** New strategies can be added by implementing a function with the standard signature and registering it.

### 2.2 What we decided against

- **Replacing the hard threshold outright.** The hard threshold has 18 months of production history and known behaviour. Replacing it silently would invalidate all existing reproducibility baselines.
- **Modifying CurriculumLearner.** The curriculum's cooling schedule (threshold decay over lessons) is orthogonal to how anchors are selected given a threshold. The two concerns are kept separate.
- **Continuous relaxation of the threshold only.** The sigmoid strategy does this, but it's one option among four. Different use cases favour different distributions.

## 3. Strategy Definitions

All strategies share the same function signature:

```python
def select_anchor(
    activity: np.ndarray,      # [H, W] int — count_nonzero per cell
    threshold: int,             # from CurriculumLearner
    min_events: int,            # floor from config
    rng: np.random.Generator,   # seeded RNG for reproducibility
    config: dict,               # full config (strategy reads its own params)
) -> tuple[int, int]:           # (row, col) anchor coordinates
```

All strategies guarantee:
- Cells with zero activity are never selected.
- Cells below `min_events` are never selected.
- Output is deterministic given the same `rng` state.

### 3.1 Hard Threshold (`"threshold"`)

**Formula:** Uniform over `{cells : activity[r, c] >= threshold}`.

**Fallback:** If no cells qualify, uniform random over the entire grid.

**Properties:** Binary inclusion/exclusion. A cell at `threshold - 1` has probability 0; a cell at `threshold` has probability `1/N_qualified`. This is the original production behaviour.

**Config params:** None (uses the curriculum-provided threshold directly).

**When to use:** Exact reproduction of existing runs. Baseline comparisons.

**Reference:** The hard-threshold importance sampling approach is standard in curriculum learning. See Bengio et al. (2009), "Curriculum Learning," *ICML*, which establishes the principle of training on progressively harder examples. The specific busy-cell threshold mechanism is HydraNet-specific (ADR-011).

### 3.2 Power Law (`"power_law"`)

**Formula:** `p(r, c) proportional to activity[r, c]^alpha`, for cells with `activity >= min_events`.

**Implementation:** Log-space arithmetic for numerical stability: `log_w = alpha * log(activity)`, then log-sum-exp normalization. Avoids float64 overflow for `alpha >= 137` with realistic activity values.

**Config params:**
- `sampling_alpha` (default 1.0) — exponent controlling concentration. Higher alpha concentrates more probability on the highest-activity cells. `alpha = 0` gives uniform over eligible cells; `alpha -> inf` gives argmax.

**Properties:** Smooth — a cell's probability changes continuously with its activity. Does not use the curriculum threshold (only `min_events` as a floor). The curriculum's cooling effect must come from adjusting `alpha` over training (not currently implemented, but the parameter is exposed).

**When to use:** When you want sampling probability to scale with activity intensity without a hard cutoff. Natural for power-law-distributed data (which conflict event counts often are).

**Reference:** Power-law weighting for importance sampling is used in prioritized experience replay (Schaul et al., 2016, "Prioritized Experience Replay," *ICLR*), where transition priorities follow `p_i proportional to |delta_i|^alpha`. Our `activity^alpha` is the spatial analogue — prioritizing cells by observed conflict intensity.

### 3.3 Boltzmann (`"boltzmann"`)

**Formula:** `p(r, c) proportional to exp(activity[r, c] / tau)`, for cells with `activity >= min_events`.

**Implementation:** Log-space with shift: `log_w = activity / tau`, subtract max for numerical stability, then exponentiate.

**Config params:**
- `sampling_temperature` (default 10.0) — temperature parameter. Low tau concentrates on high-activity cells (approaching argmax as `tau -> 0`). High tau approaches uniform over eligible cells (as `tau -> inf`).

**Properties:** Smooth, with a natural "cooling" interpretation. The temperature parameter maps directly onto the curriculum's difficulty schedule: high temperature = easy (broad sampling), low temperature = hard (concentrated on hotspots). This makes Boltzmann the most natural fit for curriculum-based training.

**When to use:** Cross-backend parity testing (high tau smooths out small activity differences). Curriculum schedules where you want a single parameter (temperature) to control concentration.

**Reference:** Boltzmann exploration is foundational in reinforcement learning (Sutton & Barto, 2018, *Reinforcement Learning: An Introduction*, Section 2.3). The softmax action selection `p(a) = exp(Q(a)/tau) / sum(exp(Q/tau))` is directly analogous. In the spatial sampling context, Muller et al. (2019) use temperature-scaled softmax over spatial attention maps for object detection training prioritization.

### 3.4 Sigmoid Soft Threshold (`"sigmoid"`)

**Formula:** `p(r, c) proportional to sigmoid(k * (activity[r, c] - threshold))`, for cells with `activity >= min_events`.

**Implementation:** `x = k * (activity - threshold)`, clipped to [-500, 500] to avoid overflow, then standard sigmoid `1 / (1 + exp(-x))`.

**Config params:**
- `sampling_steepness` (default 1.0) — steepness of the sigmoid transition. `k -> inf` recovers the hard threshold exactly. `k -> 0` gives uniform over eligible cells. Moderate `k` (0.1-1.0) creates a soft boundary where cells near the threshold get intermediate probabilities.

**Properties:** Smooth relaxation of the hard threshold. Unlike power-law and Boltzmann, sigmoid directly uses the curriculum threshold — cells above threshold get high probability, cells below get low (but nonzero) probability. This preserves the curriculum's semantic meaning (threshold = difficulty) while eliminating the discrete jump.

**When to use:** When you want to keep the curriculum's threshold-based difficulty schedule but remove the hard cutoff sensitivity. The most conservative departure from the production strategy.

**Reference:** Sigmoid relaxation of hard thresholds is a standard technique in differentiable optimization. Jang, Gu, & Poole (2017), "Categorical Reparameterization with Gumbel-Softmax," *ICLR*, use temperature-scaled sigmoid/softmax to relax discrete choices. The sigmoid soft threshold is the 1D case of this principle applied to spatial sampling.

## 4. Implementation

| File | Change |
|------|--------|
| `views_hydranet/utils/sampling_strategies.py` | New file. Four strategy functions + `SAMPLING_STRATEGY_REGISTRY` dict. |
| `views_hydranet/utils/volume_sampler.py` | Resolves strategy from registry in `__init__`, calls via `self._select_anchor()` in `_generate_window()`. |
| `views_hydranet/utils/config_initializer.py` | Adds `sampling_strategy`, `sampling_alpha`, `sampling_temperature`, `sampling_steepness` fields with defaults. Adds `field_validator` checking against registry. |

## 5. Strategy Selection Guide

| Scenario | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| Reproduce existing runs exactly | `threshold` (default) | Bit-identical to original code |
| Cross-backend parity testing | `boltzmann` (tau=50-100) | Smooth distribution minimizes sensitivity to +-1 activity differences |
| Power-law distributed activity | `power_law` (alpha=1-2) | Matches the data generating process |
| Conservative threshold relaxation | `sigmoid` (k=0.5) | Keeps curriculum semantics, removes hard cutoff |
| Hyperparameter sweep / research | Any | Registry makes all strategies interchangeable |

## 6. Relation to Other ADRs

- **ADR-011** (Curriculum and Training Topology): This ADR extends ADR-011's "Busy-Search Mechanism" from a single hard threshold to a configurable strategy. The curriculum's cooling schedule (Section 2.1 of ADR-011) is unchanged; only the anchor selection mechanism within VolumeSampler is parameterized.
- **ADR-009** (VolumeSampler Specification, archived): The original sampler spec. Superseded by ADR-011 for curriculum integration; this ADR further extends the sampling mechanism.
- **ADR-048** (Technical Risk Register): C-77 (power-law overflow, resolved) and C-78 (test coverage gaps, resolved) were registered during implementation.

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Non-default strategy silently changes training behaviour | Default is `"threshold"` — must opt in explicitly via config |
| Numerical overflow with extreme parameters | All strategies use log-space arithmetic; C-77 resolved power-law overflow |
| Strategy-specific parameters are confusing | Each strategy reads only its own parameter from config; unused params are ignored |
| New strategy untested through full training | Integration tests verify all strategies through VolumeSampler; full training validation is the experimenter's responsibility |
