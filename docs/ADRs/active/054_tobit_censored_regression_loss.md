# ADR-054: Tobit Censored-Normal Likelihood as Regression Loss

**Status:** Accepted
**Date:** 2026-05-29
**Deciders:** Simon / Claude
**Informed:** MD&D Team

---

## 1. Context
**Why are we doing this now?**

- **Problem:** The sweep isolation experiment (S2) demonstrated that the hurdle mask (`hurdle_threshold=0.0` in `training_engine.py:172-178`) causes gradient starvation. With ~95% of PRIO-Grid cells at zero, the regression head receives gradient from only ~5% of cells. This produces training non-convergence (oscillating loss 2300-4300 vs baseline's monotonic 1200->221) and autoregressive divergence during 36-step inference.

- **Architectural observation:** The model's decoder heads apply `F.relu()` as the final activation (lines 432, 464, 496 in `HydraBNrecurrentUnet_06_LSTM4.py`). This is literally the Type-I Tobit observation equation: `y = max(0, z*)` where `z*` is the latent intensity. The architecture already implements left-censoring at zero. It is using the wrong loss function for the censoring model it already embodies.

- **Urgency:** This is Step 1 of 6 in the zero-inflation instability remediation sequence (A -> E -> D/B -> C/F). All subsequent paths depend on stable baseline regression. Issue #36.

---

## 2. Decision
**Replace the hurdle mask with Tobit censored-normal negative log-likelihood.**

- **Statement:** We add `TobitLoss` to the loss registry as `loss_reg: 'tobit'`. When selected, the training engine uses pre-ReLU latent activations (`output.reg_latent`) for loss computation, providing dense gradient from ALL cells. Post-ReLU values continue to be used for autoregressive inference feedback and forensic recording.

- **In-Scope:**
  - `TobitLoss` class with fixed sigma parameter (`views_hydranet/utils/tobit_loss.py`)
  - `ModelOutput.reg_latent` field exposing pre-ReLU activations
  - Training engine wiring: `needs_latent` protocol for loss-model coupling
  - Config validation: `loss_reg='tobit'` with `hurdle_threshold` raises `ValueError`

- **Out-of-Scope:**
  - Learned sigma (heteroscedastic Tobit) -- future refinement if fixed sigma proves sensitive
  - Heckman selection correction for active-cell bias -- Path B (issue #37)
  - Tail-aware extensions beyond normal latent -- Path D (issue #39)
  - Autoregressive exposure bias mitigation -- Path E (issue #40)

---

## 3. Rationale & Integrity Impact
**The logic behind the choice.**

- **Logic:** The hurdle mask treats y=0 as "ignore this cell" (zero gradient). The baseline shrinkage loss treats y=0 as "predict zero" (full gradient). Tobit treats y=0 as "the latent intensity was <= 0, and we observed the censored value" (principled gradient). The censored-cell loss `-log Phi(-mu/sigma)` pushes the latent mu negative, teaching the model "this cell should be quiet" without discarding 95% of the training signal.

- **Literature support:** Five independent groups validate deep Tobit:
  - Zhang et al. (2021): DTN-I ReLU = Type-I Tobit. Outperforms standard DNNs.
  - Danăilă & Buiu (2024): Reparametrized Tobit has globally concave log-likelihood.
  - Jacobson & Zou (2024): MSE advantage grows with censoring proportion q. At q~0.95 the advantage is massive.
  - Wu et al. (2026): Convergence rate and selection consistency guarantees.
  - O'Neill (2024): TOBART confirms Tobit works with nonlinear function approximators.

- **Fortress State:** The `needs_latent` protocol ensures the pre-ReLU/post-ReLU distinction is explicit and type-checked. Config validation prevents contradictory `tobit + hurdle_threshold` combinations. The `is True` check on `needs_latent` prevents MagicMock-induced bugs in test mocks.

- **Fail-Loud:** Config validator raises `ValueError` if `loss_reg='tobit'` and `hurdle_threshold` is set simultaneously. `TobitLoss.__init__` raises on `sigma <= 0`.

---

## 4. Consequences
**The honest trade-off.**

### Positive
- Dense gradient from ALL cells eliminates gradient starvation
- Principled statistical treatment of zero-inflation (censoring, not masking)
- ReLU + Tobit is a mathematically coherent model (Type-I Tobit)
- Pre-ReLU latent exposure enables future work (Heckman correction, Path B)

### Negative
- Assumes normal latent distribution (conflict fatalities have power-law tails)
- Fixed sigma is a hyperparameter; wrong value may require tuning
- 4th field in `ModelOutput` breaks legacy `r, c, h = model(x, h)` tuple unpacking
- Marginal memory increase from carrying `reg_latent` alongside `reg`

---

## 5. Validation
**How do we prove it works?**

- **Invariants:**
  - `output.reg == F.relu(output.reg_latent)` (enforced by `test_relu_relationship`)
  - `output.reg >= 0` always (enforced by `test_reg_latent_is_pre_relu`)
  - Gradient flows to ALL cells including y=0 (enforced by `test_gradient_flows_to_all_cells`)

- **Tests:** 28 tests in `tests/test_tobit_loss.py`:
  - Interface (5): importable, nn.Module, needs_latent flag, scalar output, sigma validation
  - Censored (4): all-censored, gradient direction, loss monotonicity, hand-computed
  - Observed (3): all-observed, perfect prediction, hand-computed
  - Mixed (3): batch, gradient density, spatial shapes
  - Numerical stability (4): large positive/negative mu, large residual, small sigma
  - Registry (4): registered, factory, config accepts, config rejects tobit+hurdle
  - ModelOutput (5): field exists, forward returns it, pre-ReLU property, relu relationship, eval-mode None

- **Failure Mode:** If isolation test S2a-Tobit shows training loss oscillation similar to S2-hurdle, the normal latent assumption is inappropriate. Proceed to Path D (tail-aware extension).

---

## 6. Implementation Notes
- **Location:** `views_hydranet/utils/tobit_loss.py`, `ModelOutput` in architecture file, `LOSS_REG_REGISTRY` in `utils.py`, training engine latent wiring, config validator in `config_initializer.py`.
- **References:** Issue #36, ADR-050 (superseded for the hurdle approach), `reports/paths_forward.md`.
- **sigma handling roadmap:** Fixed sigma=1.0 (this ADR) -> learned scalar sigma -> reparametrized gamma=1/sigma (Danăilă 2024) -> heteroscedastic sigma(x). Each step is a separate config option; no architectural change needed.
