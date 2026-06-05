# ADR 027: Autoregressive Inference and Hidden State Strategy

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | The Forecasting Feedback Loop |
| ADR Number          | 027               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## Context
HydraNet is a recurrent architecture designed for multi-step forecasting. While training is supervised by ground-truth sequences, inference requires an **Autoregressive (Recursive)** approach where model predictions become inputs for subsequent time steps. This process requires precise hidden state management to prevent numerical explosion and preserve long-term dependencies.

## Decision
We enforce a standardized execution pattern for multi-step inference, centered on **Bootstrapped Feedback** and **Selective Hidden State Freezing**.

### 1. The Autoregressive Loop (The Feedback)
*   **The Bootstrap:** For step $t=0$, the model is initialized with the final observed historical frame.
*   **The Chain:** For $t > 0$, the model input $X_t$ is the prediction $\hat{y}_{t-1}$ from the previous step.
*   **Dimensional Symmetry:** Predictions MUST be reshaped and normalized to match the input feature space (e.g., Log-Space) before being fed back as $X_t$.

### 2. Hidden State Management (`hs` vs `hl`)
To manage model "memory" during long horizons, we implement three explicit freezing strategies:
1.  **None:** Standard update of both Short-term (`hs`) and Long-term (`hl`) memory.
2.  **Freezing (`hs` or `hl`):** Selectively prevents the update of specific memory components. This is used to test the stability of spatial features (`hs`) vs. temporal momentum (`hl`).
3.  **Random Freezing:** A stochastic stability test where hidden state channels are partially updated.

### 3. The Persistence Gate
*   **Hidden State Initialization:** Hidden states must be initialized spatially based on the target grid resolution (ADR 025). 
*   **Detach Law:** Hidden states must be **detached** between samples to prevent gradient leakage if backpropagation is ever attempted during evaluation.

## Verification Protocol (Team Audit)

### Green Team (Accuracy)
- Prove that for Step 1, the model input matches the final frame of history.
- Prove that for Step 2, the model input matches the prediction from Step 1.

### Beige Team (Robustness)
- Verify that if the model produces non-finite values (`NaN`, `Inf`), the autoregressive loop fails immediately (Panic Check).
- ~~Verify that `freeze_h` options outside the defined list (`hs`, `hl`, `none`, `random`) raise a `ValueError`.~~ **Retired 2026-06-05** — `freeze_h` removed (see Rationale update); the rollout always evolves the full state. Guard: `tests/test_inference_logic.py::test_freeze_h_option_retired`.

### Red Team (Invincibility)
- Verify that the hidden state `h` is never "shared" between independent stochastic samples, ensuring each sample path is mathematically isolated.

## Rationale
This strategy ensures that HydraNet's forecasting behavior is consistent and auditable.

> **Update 2026-06-05 — `freeze_h` retired.** The `freeze_h` mechanism (modes
> `hs`/`hl`/`all`/`none`/`random`) was removed. The pre-registered `freeze_h` ablation
> (`reports/results_freezeh_ablation.md`) showed every mode — including `all` — explodes
> identically under the C-113 runaway, proving the divergence rides the prediction→input
> *feedback* path, not the recurrent state; freezing was therefore **inert** against the
> failure it was meant to study, while creating a train/inference mismatch (training
> evolves the full state, inference froze part of it). The rollout now always evolves the
> full ConvLSTM state (the former `"none"` mode). The durable fix for autoregressive
> drift is **Axis-B rollout training** (`reports/2026-06-05_rollout_training_dossier/`,
> ADR-058 candidate; register C-113/C-125/C-126).
