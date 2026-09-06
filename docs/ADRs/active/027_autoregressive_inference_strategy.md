# ADR 027: Autoregressive Inference and Hidden State Strategy

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | The Forecasting Feedback Loop |
| ADR Number          | 027               |
| Status              | Accepted (amended 2026-06-05: §2 `freeze_h` retired; amended 2026-09-05: §2.1 cell clamp admitted to production) |
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

### 2. Hidden State Management
The rollout performs a **standard update of both short-term (`hs`) and long-term (`hl`) memory at every step** — the hidden state always evolves freely.

> **Superseded 2026-06-05 — `freeze_h` retired.** This section originally defined three selective hidden-state *freezing* strategies (`none` / `hs`·`hl` / `random`). They were removed: the ablation showed freezing was inert against the C-113 runaway (the divergence rides the prediction→input feedback path, not the recurrent state) while creating a train/inference mismatch. Only the former `none` behaviour remains. See the Rationale update.

### 2.1 The cell clamp (`freeze_recurrent`) — admitted to production 2026-09-05

**This section does NOT reinstate `freeze_h`.** That mechanism stays retired, its guard test
(`tests/test_inference_logic.py::test_freeze_h_option_retired`) stays green, and
`execute_freeze_h_option` must never reappear. What is admitted here is a different mechanism,
answering a different question, on different evidence.

**Decision.** `freeze_recurrent` is promoted from a diagnostic-only constructor argument to a
**validated configuration field**, defaulting to `None` (the §2 behaviour, byte-identical). The
value `"cell"` holds the long-term half (`hl_*`) at its end-of-seed-step value for the free-running
rollout and is **permitted in production**.

**Why this does not contradict the 2026-06-05 retirement.** The two findings answer different
questions and both stand:

| | June 2026 (`freeze_h`) | September 2026 (`freeze_recurrent='cell'`) |
|---|---|---|
| question | does freezing stop the **C-113 runaway**? | does clamping the cell improve **rollout skill**? |
| metric | divergence / explosion | `AP` over a 36-step free-running rollout |
| answer | **No — inert.** Every mode, `all` included, exploded identically. | **Yes.** ΔAP +0.0367 (h18), **+0.0591 (h36)**, **4/4 seeds**, +0.0000 at h1. |
| status | **unchanged and still correct** | new evidence, unavailable in June |

June proved the divergence rides the prediction→input *feedback* path, not the recurrent state.
That remains true. It does not follow that the recurrent state is irrelevant to *skill*, and the
2026-09 measurements show it is not: the cell drains 41× during free-running (**M50**), it carries a
spatial map that drives placement (**M54/M60**), and clamping it is the only intervention in six
attempts that improves the rollout (**M48/M56**, re-verified from raw artifacts 2026-09-05, **M65**).

**The train/inference mismatch objection is acknowledged, not dissolved.** June cited it as a cost
and it is still a cost: training evolves the full state, a clamped rollout does not. It is accepted
here because the mismatch was **measured** rather than reasoned about, and the feared trade-off did
not appear — **M58**: direction skill under the clamp is *slightly better* at h18 (+0.0137, 4/4) and
indistinguishable at h36, and `crps_events` improves on 4/4 seeds in every arm. What the clamp does
change is the **dispersion of the predicted per-cell change**, which falls −0.4563 (h18) and −0.6593
(h36), 4/4 seeds. A delivered forecast under the clamp is therefore **equally good at saying which
places change and flatter about how much** — a product change that MUST be disclosed in the release
note, not discovered by a user.

**Scope of the permission.** `"cell"` only. `"hidden"` has no consistent effect (**M58**, contested
at both horizons) and `"all"` is not evidenced; both remain available for diagnostics and neither is
recommended for production. `freeze_recurrent_weight` stays a field so the clamp is a dial rather
than a hard prior — **M41's saturation at w≈0.1 was measured on the 40-lesson vehicle and has never
been re-tested at L=300**, so the production default is the measured `1.0` and the dial is open.

**What this section does not claim.** The clamp is a **mitigation, not a fix**. It does not close
#258, it does not touch the magnitude ceiling (`size_ratio` is exactly 0 in all 16 arm-seeds), and
the durable answer to autoregressive drift remains open — currently **#310, direct multi-horizon**,
which deletes the recursion rather than clamping its state.

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
- **Added 2026-09-05 (§2.1):** verify that `freeze_recurrent` rejects any value outside
  `{None, "hidden", "cell", "all"}` with a `ValueError`, that `freeze_recurrent_weight` outside
  `[0, 1]` is rejected, and that a config **omitting** `freeze_recurrent` produces a rollout
  byte-identical to the §2 behaviour. The last of these is the load-bearing one: it is what keeps
  this amendment from silently changing every existing model.

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
> ADR-058 candidate — **PARKED** (no ADR-058 file was written; the direction is parked, and
> `rollout_horizon > 1` is guarded off by C-264 until the pushforward path is wired); register
> C-113/C-125/C-126).
