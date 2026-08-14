# Class Intent Contract: HydraBNUNet06_LSTM4 (`views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-08-14
**Related ADRs:** ADR-067 (distribution-family subsystem), ADR-063/064 (regression/classification
head output activation), ADR-060/061 (static exogenous / coordinate input channels), ADR-057
(variational dropout for AR stability), ADR-056 (scheduled sampling), ADR-054/055 (Tobit censored
loss / learnable per-target sigma), ADR-030 (dynamic slicing handshake)

---

## 1. Purpose

> A recurrent Batch-Norm U-Net with a 4-cell ("Quad") ConvLSTM temporal core (line 45) that maps a
> SINGLE timestep `x [B,C,H,W]` + carried hidden state `h` to six task heads — 3 regression + 3
> classification for the State-Based / Non-State / One-Sided targets — and the next hidden state
> (lines 502-685).

`forward` is strictly one recurrent step. The T-loop that feeds each step's output back as the next
step's input (the autoregressive rollout) lives OUTSIDE this class, in `training_engine`
(`_process_sequence`, `training_engine.py:253`) and `hydranet_inference` (`hydranet_inference.py:454`).
This class owns spatial feature extraction, temporal memory, and head emission — nothing about how
steps are chained.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** own the recurrent rollout or AR feedback. `forward` returns `h_next` and per-head
  outputs for ONE step (line 685); chaining, teacher-forcing/scheduled-sampling (ADR-056), and
  feeding predictions back into the next input are the training/inference orchestrators' job
  (`training_engine._process_sequence`, `hydranet_inference.predict`).
- Does **not** own the output distribution's likelihood, scoring, sampling, or composition. It holds
  a `DistributionFamily` only to size the reg heads and activate their raw channels (lines 99,
  138-139, 106-107); `nll`/`sample`/`mean`/gate-composition belong to the family (see its CIC) and
  the loss/inference layers.
- Does **not** own the loss. The train-only `reg_latent` field exists so a censored loss (TobitLoss,
  ADR-054/055) can read the pre-ReLU mu (lines 42, 681, 366 in `training_engine.py`); the class emits
  it, it does not consume it.
- Does **not** validate config semantics. `choose_model` (`utils/utils.py:30`) reads the config dict
  and passes primitives in; this class validates only its own local invariant (quantile count, line
  126).
- Does **not** persist, log, or perform I/O. Pure `nn.Module` compute.

---

## 3. Responsibilities and Guarantees

**Public API.**

- `__init__(input_channels, total_hidden_channels, output_channels, dropout_rate,
  output_distribution="standard", n_static_channels=0, static_top_skip=True, reg_activation=None,
  n_quantiles=None)` (lines 64-75). `total_hidden_channels` MUST be divisible by 8 (docstring line 82;
  consumed at line 157 `num_lstm_state_layers = total/(4*2)` and line 520 `split_h = h.shape[1]/8`).
- `forward(x, h) -> ModelOutput(reg, cls, h_next, reg_latent)` (lines 502-685). Single timestep.
  `reg` `[B, n_reg, H, W]` post-activation, `cls` `[B, n_cls, H, W]` logits, `h_next`
  `[B, total_hidden_channels, H, W]`, `reg_latent` `[B, n_reg, H, W]` pre-ReLU or `None`.
- `set_locked_dropout(enabled: bool)` (lines 687-701) / `reset_locked_dropout()` (lines 703-709) —
  toggle / refresh MC-dropout (ADR-057). `set_locked_dropout` switches ONLY the `LockedDropout`
  submodules to train mode (line 700), leaving BatchNorm running stats and the train-only
  `reg_latent` path untouched (line 694).
- `init_hTtime(hidden_channels, H, W) -> [1, hidden_channels, H, W]` zero state, float32 (lines
  711-724).

**`ModelOutput`** (lines 26-42) is a `NamedTuple`, not a bare tuple. Adding the 4th field
`reg_latent` (default `None`) intentionally BROKE 3-var unpacking — consumers use NAMED access
(`output.reg`, `output.cls`, `output.h_next`; docstring lines 33-37). All real consumers already do
(`training_engine.py:347`, `hydranet_inference.py:460`).

**Guarantees.**

- The forward emits exactly 3 regression heads and 3 classification heads, concatenated on the
  channel axis in fixed head order H1/H2/H3 (lines 682-683).
- `reg_latent` is populated ONLY in training mode (`self.training`, line 681); at eval it is `None`,
  so the censored-loss latent path cannot silently fire outside training.
- Static top-skip re-injection (ADR-060/061) is byte-neutral when `n_static_channels=0` or
  `static_top_skip=False`: `coords` is `None` (line 517) and the head skip concat is skipped (line
  563), leaving pre-ADR-061 behavior identical.
- `output_distribution="standard"` reproduces pre-#100 behavior byte-for-byte: `resolve_family`
  returns `None` (line 99), reg activation falls to `F.relu` (line 114), reg heads keep width
  `output_channels` (line 141).

---

## 4. Inputs and Assumptions

- `x` is `[B, input_channels, H, W]`, `h` is `[B, total_hidden_channels, H, W]` (docstring lines
  507-508). `H` and `W` must survive two `MaxPool2d(2,2)` down/ up steps (lines 171, 175) — even
  dimensions.
- **AR-feedback width invariant.** For legacy heads the caller must satisfy
  `input_channels == 3*output_channels + n_static_channels` (comments lines 121-122, 135-137): the 3
  regression heads each emit `output_channels`, and their concatenation plus statics becomes the next
  step's input. Family and quantile heads size ONLY the reg heads (`reg_out_ch = family.n_params` or
  `n_quantiles`, lines 138-141) while `output_channels` stays the feedback/class width — so the
  invariant is preserved unchanged by those heads.
- The last `n_static_channels` of `x` are the static/coordinate channels, captured raw at full
  resolution before `x` is fused with the hidden state (line 517). `static_top_skip=False` keeps them
  in the ENCODER input but drops them from the head re-injection (C-228, comment lines 145-147).
- When `output_distribution` names a registered family, `resolve_family` (line 99, from
  `views_hydranet.distributions`, line 8) returns a `DistributionFamily`; the class assumes its
  `n_params` and `activate` are the emit contract (ADR-067).

---

## 5. Outputs and Side Effects

- Returns `ModelOutput` (line 685). `reg` is activated per the selected reg activation (applied at
  lines 589, 625, 661); `cls` is raw logits (no output activation — ADR-064; lines 607, 643, 679).
- `h_next` is the concatenation of the 4 short-term + 4 long-term ConvLSTM states (line 555) — the
  carried memory the orchestrator threads to the next step.
- Stochastic only via dropout. In eval with locked dropout OFF, `forward` is deterministic; with
  `set_locked_dropout(True)` each posterior sample draws one fixed mask per site until
  `reset_locked_dropout()` (lines 700-701, 703-709) — MC-dropout epistemic sampling.
- No I/O, no logging, no global mutation. `__init__` mutates only its own reg-head biases via the
  informed-init seeding (lines 280-288).

---

## 6. Failure Modes and Loudness

- `output_distribution="quantile"` with `n_quantiles` missing or `< 2` raises `ValueError`
  (lines 126-127) — a quantile head needs at least 2 monotone quantiles to be well-defined.
- A mis-sized `total_hidden_channels` (not divisible by 8) corrupts the state split
  (`torch.split(h, split_h, ...)`, line 521) — the class documents the divisibility precondition
  (line 82) but does not assert it; a caller violating it fails at the first forward via a shape
  error, not a silent wrong answer.
- Violating the AR-feedback width invariant surfaces loud as a channel-count mismatch in
  `enc_conv0` (line 162, expecting `input_channels + total_hidden_channels/2`) or when the next
  step's fed-back input is re-consumed — a torch shape `RuntimeError`, never silent.
- Reg-family / quantile head sizing and activation are chosen once at `__init__` and cannot drift at
  forward time; a head sized for the wrong family fails loud downstream at the family's `activate`
  (see DistributionFamily CIC), not here.
- Unpacking `ModelOutput` with 3 targets (`r, c, h = model(x, h)`) raises `ValueError` (too many
  values) — the intentional break that forces named access (docstring lines 33-37).

---

## 7. Boundaries and Interactions

- **Consumes** `resolve_family` (ADR-067, imports line 8; called line 99), `LockedDropout`
  (ADR-057, import line 7; instantiated as a 15-site `ModuleDict`, lines 316-335), and the quantile
  head helpers `init_quantile_conv_` / `monotone_quantiles` (import line 9; used lines 130-134,
  287-288). It treats the family as an opaque activation/param-count provider (DIP).
- **Produced by** `choose_model` (`utils/utils.py:30`), which maps config keys to the constructor
  args — the only instantiation site.
- **`forward` consumed by** `training_engine._process_sequence` (`training_engine.py:253`, reads
  `output.reg/cls/h_next` at 347 and `output.reg_latent` at 366) and
  `hydranet_inference` (`hydranet_inference.py:454, 460, 538`); the free-running diagnostic rollout
  duck-types `out.reg`/`out.h_next` (`utils/rollout_diagnostics.py:75`).
- Must NOT reach into the training loop, loss, config validation, or inference orchestration — those
  hold this class, not the reverse (ADR-002 layering).

---

## 8. Examples of Correct Usage

```python
# One recurrent step (the orchestrator owns the T-loop).
model = HydraBNUNet06_LSTM4(
    input_channels=8, total_hidden_channels=32, output_channels=1, dropout_rate=0.1
)
h = model.init_hTtime(hidden_channels=32, H=16, W=16)
out = model(x, h)                 # ModelOutput
reg, cls, h = out.reg, out.cls, out.h_next   # NAMED access, then feed h to the next step
```

```python
# MC-dropout posterior sampling at inference (ADR-057).
model.set_locked_dropout(True)    # locked masks, dropout active in eval
for _ in range(n_posterior):
    model.reset_locked_dropout()  # fresh, internally-consistent mask per sample trajectory
    trajectory = run_rollout(model, x0, h0)
```

---

## 9. Examples of Incorrect Usage

- `r, c, h = model(x, h)` — 3-var unpack of the 4-field `ModelOutput`; raises. Use named access.
- Building the AR rollout inside a caller by re-implementing feedback, then also calling this class
  as if it owned the loop — the rollout, teacher forcing, and clamp live in `training_engine` /
  `hydranet_inference`, not here.
- Reading `reg_latent` at eval time — it is `None` outside training (line 681); the censored-loss
  latent is a train-only signal.
- Passing `output_distribution="quantile"` without `n_quantiles>=2`, or a channel budget that breaks
  `input_channels == 3*output_channels + n_static` for a legacy head — both fail loud, but calling
  them "config the model tolerates" is a misuse.

---

## 10. Test Alignment

- **Green (core contract):** `tests/test_architecture.py` — instantiation, `init_hTtime` shape/dtype
  (zeros, float32), forward output shapes for `reg`/`cls`/`h_next`/`reg_latent`, and hidden-state
  evolution.
- **Green (dropout invariant):** `tests/test_locked_dropout.py` — asserts `model.dropout` is a
  per-site `ModuleDict` of exactly 15 DISTINCT `LockedDropout` instances (lines ~185-194), locked
  masks independent across sites, per-step default (unlocked) behavior, reset-between-samples
  changes output, and that the 15 instances add no state-dict keys (artifact-compat).
- **Green (head selection / activation):** `tests/test_reg_activation.py`,
  `tests/test_output_distribution_head.py`, `tests/test_quantile_head.py`,
  `tests/test_falsify_reg_head_dead_relu.py` (C-178 softplus-not-dead-ReLU),
  `tests/test_onset_bias_init.py` (informed init).
- **Green (static top-skip / coords):** `tests/test_static_top_skip.py`,
  `tests/test_coordinate_channels.py`, `tests/test_channel_role_census.py`.
- **Green (latent / censored path):** `tests/test_learnable_sigma.py`, `tests/test_per_target_sigma.py`,
  `tests/test_tobit_loss.py`, `tests/test_falsification_per_target_sigma.py`.
- **Beige (integration through the rollout):** `tests/test_training_engine.py`,
  `tests/test_pipeline_integration.py`, `tests/test_scheduled_sampling.py`,
  `tests/test_bn_recalibrate.py` (BN-recal for the C-184 recurrent-BN seed-bimodality),
  `tests/test_inference_orchestrator_pf.py`, `tests/distributions/test_head_wiring.py`.
- Must protect against regression: the 15-site `ModuleDict` structure (masks change the posterior),
  the named-access `ModelOutput` shape, and the `"standard"` byte-identical fallback.

---

## 11. Evolution Notes

- **Stable:** the single-timestep forward contract (`(x, h) -> ModelOutput`), the 3-reg/3-cls head
  topology, the AR-feedback width invariant, and named `ModelOutput` access.
- **Un-adopted / banked:** the per-site `LockedDropout` `ModuleDict` (ADR-057 / C-128) is wired but
  MC-dropout is not the default inference path; enabling it changes the epistemic posterior, so it is
  a deliberate opt-in, not free.
- **History, not to re-litigate:** recurrent BatchNorm caused seed-bimodal basins (C-184), mitigated
  by post-train BN recalibration (default-on, tested in `test_bn_recalibrate.py`) — a property of the
  BN layers here, addressed outside the class.
- **Expected to change:** new distribution families extend behavior WITHOUT new branches in this file
  (the ADR-067 strangler-fig: `resolve_family` returns `None` for legacy, a family for new — lines
  96-99, 138-141). A new head param-count arrives via `family.n_params`, not an edit here.

---

## End of Contract

This document defines the **intended meaning** of `HydraBNUNet06_LSTM4`.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
