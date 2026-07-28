# Class Intent Contract: HydraNetConfig

**Status:** Active
**Owner:** Schema
**Last reviewed:** 26.05.2026
**Related ADRs:** ADR-009, ADR-046, ADR-049, ADR-050, ADR-054

---

## 1. Purpose

The `HydraNetConfig` is the **Schema** of the HydraNet pipeline. Its primary purpose is to define the exhaustive, validated configuration state as a Pydantic `BaseModel`, enforcing field types, checksums, and cross-field invariants at construction time.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** fetch, store, or manage configuration files.
- This class does **not** interact with models, data, or the file system.
- This class does **not** implement pipeline logic or orchestration.

---

## 3. Responsibilities and Guarantees

- **Field Validation:** Guarantees that all 88 fields are type-checked and constraint-validated (e.g., `dropout_rate` in [0.0, 1.0], `input_channels >= 1`, `rollout_horizon >= 1`). *(`ss_feedback` added 2026-07-27 — EXP-4/GTF bloom dossier: what scheduled sampling feeds back for a FAMILY head, `mean` (default, log1p E[y]) vs `sample` (composition-aware draw = train-exposure==deploy-exposure); validated in {mean, sample}; legacy point heads ignore it.)* *(`rollout_feedback` added 2026-07-26 — H-SAMPLE / EXP-2 bloom dossier: the autoregressive feedback copy, `mean` vs `sample` (ancestral) vs `teacher_forced` (oracle); HydraNetInference fails loud on a bad value or `sample` without a registered family. **Default changed 2026-07-27 (ADR-070 / Epic #193 S4): `None` = AUTO — resolves to `sample` for a registered family head (the C-113 bloom mitigation, T=0-neutral) and `mean` for a legacy head (byte-identical).** Field type `str | None`; count unchanged. **Verified (Epic #193 S7): sample bounds the bloom 9/9 arms×seeds; mean blooms 9/9.** T=0-neutrality is byte-exact at BOTH the distribution level (emit-mean/gate/params) AND the scored D×K cube — the latter via the per-`(pass,step)` sampler seeding in `to_cube_samples` (`66a95ea`), which isolates h=1's draws from the feedback-changed h≥2 params.)* *(`freeze_h` retired 2026-06-05; `rollout_horizon` added 2026-06-06 — Axis B / #78; `loss_reg_theta_init`, `learnable_theta`, `loss_class_pos_weight` added 2026-06-10 — hurdle-NB / D2 / #99; `output_distribution` added 2026-06-10 — hurdle-NB head activation / #100; `min_free_disk_gb` added 2026-06-13 — disk-headroom guard / C-154 / #107; `reg_activation` added 2026-06-21 — emit-activation decouple / Exp-B; `n_quantiles` + `loss_reg_tau`/`loss_reg_cap` added — quantile head / pinball body dial; `body_mask` added and `hurdle_threshold` retired 2026-07-18 — ADR-065 / Epic #158 (later retired, see below); `n_head_samples` added 2026-07-20 — ADR-067 / Epic #167 A-S5 (#172); `max_posterior_cube_gb` added 2026-07-21 — ADR-067 / Epic #167 A-S8 (#175); `pi_penalty_weight` + `pi_penalty_prior_logit` added 2026-07-21 — ADR-067 / Epic #167 A-S9 (#176) family π-ridge (C-200); `forecast_composition` + `gate_threshold` added 2026-07-24 — ADR-069 / Epic #183 S3 (#186), the composition axis (self_zeroed / soft_gate / threshold_gate + τ) enforcing the emit-time gate+body composition rules fail-loud, and the `output_distribution` allowed set is now the LEGACY values `{standard, hurdle_nb, hurdle_lognormal, hurdle_shrinkage, dense_nb, quantile}` **unioned with the registered distribution-family names** `family_names()` (currently `{nb, zinb}`), the two required disjoint (C-197). The classification head is reused as the hurdle gate `P(y>0)`; θ is the loss-owned Parameter in `TruncatedNBLoss`, not a model head. `body_mask` retired for `body_supervision ∈ {all, active}` + `onset_lead` + `cessation_lag` 2026-07-28 — ADR-065 amend.: the graded, asymmetric supervision window (net +2 fields → 88); resolved by `body_supervision.resolve_body_supervision`; validated by `validate_body_supervision` + `validate_body_supervision_latent` (C-193). See CIC `BodySupervisionResolver.md`.)*
- **Feedback Clamp (C-113):** `feedback_clamp_log1p` is an optional per-target log1p ceiling (`list[float] | None`, default `None`) bounding only the autoregressive feedback input, never an emitted prediction. See `reports/preanalysis_feedback_clamp.md`.
- **Disk-Headroom Guard (C-154):** `min_free_disk_gb` is an optional pre-evaluation budget (`float | None`, default `None`, `> 0`) that aborts the run (fail loud) before the ~2.5 GB prediction writes if free disk space is below it. `None` disables (default-off ⇒ unchanged behaviour). See `views_hydranet/utils/disk_guard.py`.
- **Static Channels (ADR-060, C-153):** `static_channels` (`list[str]`, default `[]`) declares input-only channels (e.g. coordinates) — injected, never predicted, never in targets, re-injected every autoregressive step. Validated: **I1** (a static channel must NOT appear in `regression_targets`/`classification_targets` → raises), and the input-channel laws become `input_channels == len(features) + len(static_channels) == 3*output_channels + len(static_channels)`. Derived over the grid in `views_hydranet/utils/static_channels.py`; empty ⇒ unchanged.
- **Balancer Freeze (C-113 bisect):** `freeze_multitask_balancer` (`bool`, default `False`) excludes the MultiTaskLoss `log_vars` from the optimizer (pre-C-111 equal-weighting regime). See `reports/preanalysis_balancer_bisect.md`.
- **Checksum Laws (ADR-009):** Guarantees `input_channels == len(features)` and `time_steps == len(steps)`.
- **Feature Lifecycle Law (ADR-046):** Guarantees that all required columns (features + targets) are accounted for in `transformations` or `derivations`.
- **Full error collection:** `handle_typos_and_missing_guidance` (a `model_validator(mode="before")`) injects sentinels for registry/enum fields so Pydantic validates ALL fields and reports every error at once, rather than stopping at the first missing one. (There is no `evalution_mode` key-rename shim — value-level typos in `evaluation_mode` are rejected loud by its field validator.)
- **Enum Validation:** Validates `run_type`, `evaluation_mode`, and `aggregate_method` against strict allowlists, with alias support for `aggregate_method` (e.g., `"mean"` → `"arithmetic_mean"`).
- **Conditional Parameter Validation:** Guarantees that strategy-specific parameters are explicitly provided for the active choice in `sampling_strategy`, `loss_reg`, `loss_class`, `hurdle_threshold` (QS99 params), and `target_weights` (regression target coverage). No silent defaults — missing parameters raise immediately.
- **Per-Target Sigma Validation (issue #44):** `loss_reg_sigma` accepts `float` (shared across targets) or `Dict[str, float]` (per-target). Dict form is only valid for `loss_reg='tobit'`. Validates: all regression targets present, no extra keys, all values positive.
- **Dict Compatibility Layer:** Provides `__getitem__`, `__contains__`, `get()`, and `keys()` for gradual migration from `config["key"]` access patterns.

---

## 4. Inputs and Assumptions

- **Construction:** Assumes keyword arguments matching the 64 defined fields. Extra fields are tolerated (`extra = "allow"`).
- **Immutability:** Once constructed, the configuration should be treated as immutable. Pydantic does not enforce frozen mode, but downstream consumers must not mutate.

---

## 5. Outputs and Side Effects

- **Validated Instance:** Produces a fully validated `HydraNetConfig` object.
- **Warnings:** Logs a warning when `evaluation_mode='stochastic'` and `aggregate_method` is set (since it will be ignored).
- **Fatal Exceptions:** Raises `ValueError` on checksum, lifecycle, or enum violations.

---

## 6. Failure Modes and Loudness

- **Checksum Mismatch:** Raises `ValueError` with explicit counts (e.g., "input_channels (7) != features (6)").
- **Feature Lifecycle Violation:** Raises `ValueError` listing unaccounted columns.
- **Invalid Evaluation Mode:** Raises `ValueError` with the valid options listed — a bad `evaluation_mode` value is rejected loud, never silently corrected.
- **Invalid Loss Function (Regression):** Raises `ValueError` when `loss_reg` is not in `LOSS_REG_REGISTRY`, listing valid options.
- **Invalid Loss Function (Classification):** Raises `ValueError` when `loss_class` is not in `LOSS_CLASS_REGISTRY`, listing valid options.
- **Missing Loss Reg Parameter:** Raises `ValueError` when the active regression loss's required parameter is not provided (e.g., `shrinkage` requires `loss_reg_a` and `loss_reg_c`, `basu_dpd` requires `loss_reg_alpha` and `loss_reg_sigma`, `tobit` requires `loss_reg_sigma`).
- **Missing Loss Class Parameter:** Raises `ValueError` when the active classification loss's required parameter is not provided (e.g., `focal` requires `loss_class_alpha` and `loss_class_gamma`).
- **Invalid Aggregate Method:** Raises `ValueError` when `aggregate_method` is not in `[arithmetic_mean, geometric_mean, median]`.
- **Degenerate Slope Ratio:** Raises `ValueError` when `slope_ratio <= 0.0` (causes division-by-zero in curriculum).
- **Degenerate Roof Ratio:** Raises `ValueError` when `roof_ratio <= 0.0` (eliminates curriculum variation).
- **Degenerate Window Dim:** Raises `ValueError` when `window_dim < 2` (single-pixel patches have no spatial context).
- **Inverted Ratio Range:** Raises `ValueError` when `min_ratio >= max_ratio` (breaks curriculum sampling).
- **Invalid Sampling Strategy (ADR-049):** Raises `ValueError` when `sampling_strategy` is not in `SAMPLING_STRATEGY_REGISTRY`, listing valid options.
- **Missing Sampling Strategy:** Raises `ValidationError` — `sampling_strategy` is a required field with no default.
- **Missing Strategy Parameter (ADR-049):** Raises `ValueError` when the strategy's required parameter is not provided (e.g., `power_law` requires `sampling_alpha`, `boltzmann` requires `sampling_temperature`, `sigmoid` requires `sampling_steepness`).
- **Contradictory Tobit + Hurdle (ADR-054):** Raises `ValueError` when `loss_reg='tobit'` and `hurdle_threshold` is set. Tobit handles zero-inflation internally via censored likelihood; the hurdle mask is contradictory.
- **Missing Hurdle QS99 Parameter (ADR-050):** Raises `ValueError` when `hurdle_threshold` is set with `qs99_weight > 0` but `qs99_tau` is not provided.
- **Invalid QS99 Weight (ADR-050):** Raises `ValidationError` when `qs99_weight < 0` (negative weight inverts the penalty direction).
- **Invalid Pi-Penalty Weight (ADR-067 / C-200):** Raises `ValidationError` when `pi_penalty_weight < 0` (a negative weight would invert the family π/μ-ridge). `None`/`0` disables the ridge (no-op).
- **Invalid QS99 Tau (ADR-050):** Raises `ValidationError` when `qs99_tau` is not in `(0.0, 1.0)` (pinball loss quantile must be a valid probability).
- **Degenerate Basu DPD Alpha (ADR-050):** Raises `ValueError` when `loss_reg='basu_dpd'` and `loss_reg_alpha <= 0` (α=0 degenerates to MLE, α<0 is undefined).
- **Degenerate Basu DPD Sigma (ADR-050):** Raises `ValueError` when `loss_reg='basu_dpd'` and `loss_reg_sigma <= 0` (σ=0 causes division by zero).
- **Invalid Target Weights (ADR-050):** Raises `ValueError` when `target_weights` contains negative values or is missing a regression target.
- **Per-Target Sigma for Non-Tobit (issue #44):** Raises `ValueError` when `loss_reg_sigma` is a dict but `loss_reg` is not `'tobit'`. Dict sigma is only meaningful for Tobit censored-normal loss.
- **Per-Target Sigma Non-Positive (issue #44):** Raises `ValueError` when any value in the `loss_reg_sigma` dict is ≤ 0.
- **Per-Target Sigma Missing Target (issue #44):** Raises `ValueError` when the `loss_reg_sigma` dict is missing an entry for a regression target.
- **Per-Target Sigma Extra Key (issue #44):** Raises `ValueError` when the `loss_reg_sigma` dict contains a key not in `regression_targets` (catches typos).
- **Invalid Scheduled Sampling Schedule (ADR-056):** Raises `ValueError` when `ss_schedule` is not in `['linear', 'inverse_sigmoid', 'exponential']`.
- **Missing Scheduled Sampling Warmup (ADR-056):** Raises `ValueError` when `ss_schedule='linear'` and `ss_warmup_lessons` is not provided.
- **Missing Scheduled Sampling K (ADR-056):** Raises `ValueError` when `ss_schedule` is `'inverse_sigmoid'` or `'exponential'` and `ss_k` is not provided.
- **Invalid Scheduled Sampling K (ADR-056):** Raises `ValueError` when `ss_schedule='exponential'` and `ss_k >= 1.0` (divergent schedule).
- **Invalid Output Distribution (ADR-067):** Raises `ValueError` when `output_distribution` is not in the LEGACY values unioned with `family_names()` (the registered distribution families), listing the valid union.
- **Family/Legacy Name Collision (ADR-067 / C-197):** Raises `ValueError` (a `model_validator` that always runs) when a registered distribution-family name equals a legacy `output_distribution` value — the two name-sets must be disjoint, else a legacy config would silently route to a new family.
- **Head Samples Without a Family (ADR-067):** Raises `ValueError` when `n_head_samples > 1` but `output_distribution` is not a registered family (a legacy point/quantile head has no per-cell sampler, so `K>1` would be silently ignored); legacy heads keep `n_head_samples=1`.
- **Family Targets Not log1p-Transformed (ADR-067 / C-198):** Raises `ValueError` (a `model_validator`) when `output_distribution` is a registered count family but one or more `regression_targets` is not in `transformations["log1p"]`. A count family's `nll` de-transforms predictions to raw counts via `expm1` (the `log1p` inverse); any other transform silently yields non-integer or wrong-scale "counts", corrupting the likelihood without an error signal.
- **Oversize D×K Posterior Cube (ADR-067 / A-S8):** `generate_posterior_samples` raises `RuntimeError` **before allocating** when the `[T,H,W,C,S]` cube (`S = n_posterior_samples × n_head_samples`) would exceed 85% of the machine's currently-available RAM (auto-measured, adapts to any host), or — when the optional `max_posterior_cube_gb` budget is set — exceeds that hard cap. Prevents the 37GB-cache-scar class (silent OOM/thrash) via `disk_guard.assert_cube_fits`. `max_posterior_cube_gb=None` disables only the opt-in cap; the auto RAM guard is always on.

---

## 7. Boundaries and Interactions

- **Gatekeeper:** Instantiated exclusively by `ConfigInitializer.get_config()`.
- **Consumers:** The `model_dump()` output is consumed by all pipeline components via dict access.

---

## 8. Examples of Correct Usage

```python
# Via ConfigInitializer (canonical path)
config_obj = HydraNetConfig(**raw_config)
config_dict = config_obj.model_dump()

# Dict-compatibility access
value = config_obj["learning_rate"]
has_key = "steps" in config_obj
all_keys = config_obj.keys()
```

---

## 9. Examples of Incorrect Usage

- **Direct Mutation:** Setting `config_obj.learning_rate = 0.1` after construction.
- **Bypassing Validators:** Using `model_construct()` to skip validation.

---

## 10. Test Alignment

- **🟩 Green Team:** Valid configuration construction and dict access in `tests/test_config_typed.py`. Hurdle+Basu DPD integration paths in `tests/test_hurdle_basu_integration.py`. Tobit config acceptance and tobit+hurdle rejection in `tests/test_tobit_loss.py`.
- **🟫 Beige Team:** Checksum violations, lifecycle law, stochastic mode warning in `tests/test_config_validation.py`. Config path guards (hurdle disabled, qs99_weight=0) in `tests/test_hurdle_basu_integration.py`.
- **🟥 Red Team:** Invalid run_type, evaluation_mode, hidden channels divisibility, missing fields in `tests/test_config_validation.py`. QS99 range validation, Basu degenerate params in `tests/test_falsification_hurdle_params.py`. Hurdle parameter enforcement, target_weights validation in `tests/test_hurdle_basu_integration.py`. CIC field count drift in `tests/test_falsification_loss_param_validation.py`.

---

## End of Contract

This document defines the **intended meaning** of `HydraNetConfig`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
