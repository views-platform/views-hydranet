# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-06-02                           |
| Total Concerns    | 107                                  |
| Open Concerns     | 29                                   |
| Resolved Concerns | 78                                   |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Open Concerns

### C-01: Manager monolith orchestration

| Field | Value |
|-------|-------|
| ID | C-01 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | When adding or removing a component from Manager's initialization sequence, verify file hasn't exceeded pure-wiring scope |
| Location | `manager/hydranet_manager.py` |

`hydranet_manager.py` imports 12 internal modules and wires all components manually. Any wiring change requires modifying this single 380-line file. Fan-out of 12 — highest in the codebase.

**Graph-quantified coupling (2026-04-21, graphify):** Knowledge graph confirms HydranetManager as the second-highest-degree node (157 edges, up from 112 on 2026-04-19), bridging 8+ communities. Edge growth is primarily from expanded semantic extraction coverage (ADR/CIC documents now included), not from new code coupling.

Per Martin (Clean Architecture Ch 26, p.228-232): the Manager correctly acts as the "Main Component" — the dirtiest component that creates everything and hands control to higher-level abstractions. High fan-out is expected for Main. The concern is not dirtiness but *size*: at 380 lines it exceeds a pure wiring role, mixing lifecycle orchestration with component construction. Martin: "Think of Main as a plugin to the application" — it should be replaceable without touching policy.

**Tech-debt-cleanup (2026-04-27):** Two additional symptoms identified: (1) `_setup_evaluation()` is 92 lines — performs model loading, conditional data pipeline dispatch, origin calculation, and orchestrator construction in a single method. Should decompose into `_load_model_artifact()`, `_compute_origins()`, `_create_orchestrator()`. (2) `hasattr(self, "_config_manager")` guard (line ~269) falls back to a no-op lambda instead of failing loudly — defensive hack around uncertain parent class initialization.

---

### C-03: Architecture hardcodes 3+3 heads

| Field | Value |
|-------|-------|
| ID | C-03 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Adding or removing a regression/classification target |
| Location | `HydraBNrecurrentUnet_06_LSTM4.py:68-167`, `hydranet_manager.py:165-176` |

The `HydraBNUNet06_LSTM4` model has 6 decoder heads physically baked into the class definition. Adding a target requires duplicating ~50 lines of layer definitions + forward() code, plus updating the preflight check. Currently stable (no planned target changes).

Per Martin (Clean Architecture Ch 8, p.87-93): this violates OCP — the architecture is closed to extension. Adding a head requires modifying both `__init__()` and `forward()`. Martin's "hierarchy of protection" (p.91) says the model entity should be the most protected component — but here it's the component most exposed to change if target count evolves.

---

### C-06: Config returns dict after Pydantic validation

| Field | Value |
|-------|-------|
| ID | C-06 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Downstream code accessing a key that Pydantic doesn't validate (via `extra="allow"`) |
| Location | `config_initializer.py:302-303` |

`ConfigInitializer.get_config()` validates via `HydraNetConfig` then returns `.model_dump()` as a plain dict. All downstream consumers use `config["key"]` or `config.get(key)` without type safety. The `extra = "allow"` setting means unvalidated keys pass through silently. Constrained by parent class (`ForecastingModelManager.configs`) requiring `isinstance(dict)`.

**Tech-debt-cleanup (2026-04-27) — config mutation hazard:** `self.configs` is reassigned via `ConfigInitializer(self.configs).get_config()` in both `_train_model_artifact()` (line 198) and `_setup_evaluation()` (line 259). If `get_config()` is not idempotent, repeated calls may diverge. In sweep scenarios, shared mutable state on `self.configs` means one trial's config initialization could leak into another. Safe in current single-threaded execution but structurally fragile.

---

### C-35: `utils/` package violates Common Closure and Screaming Architecture

| Field | Value |
|-------|-------|
| ID | C-35 |
| Tier | 3 |
| Source | clean-architecture-review (2026-04-08) |
| Trigger | Adding a new module — unclear where it belongs; changing a training component forces retest of unrelated data pipeline tests |
| Location | `views_hydranet/utils/` (20 of 25 source files) |

The `utils/` package contains 20 files spanning 5 distinct domains: data pipeline (fetcher, sniffer, scaler, handler), training strategy (curriculum, sampler, forensics), inference (orchestrator, inference engine), observability (diagnostics, logging, guardian), and configuration. A single generic package name for 80% of the codebase.

Per Martin (Clean Architecture Ch 13, p.117-123): violates CCP (Common Closure Principle) — classes that change for different reasons are packaged together. A training strategy change and a data pipeline change both touch `utils/`. Per Martin (Ch 21, p.199-202): violates Screaming Architecture — the top-level structure should scream "conflict forecasting system," not "utilities." The directory should say `data_pipeline/`, `training/`, `inference/`, `observability/` — not `utils/`.

Per Martin (Ch 13, p.120-121): also violates CRP (Common Reuse Principle) — importing `IntegrityGuardian` (pure torch, no pandas) from `utils/` transitively exposes the consumer to `volume_handler`'s pandas/torch/pipeline_core dependency tree.

---

### C-36: VolumeHandler violates Interface Segregation Principle (partially addressed)

| Field | Value |
|-------|-------|
| ID | C-36 |
| Tier | 3 |
| Source | clean-architecture-review (2026-04-08), updated D-01 execution (2026-04-11) |
| Trigger | When a new module imports VolumeHandler, verify it doesn't transitively pull unused dependencies; when adding methods to VolumeHandler, consider whether they belong to a separate adapter |
| Location | `volume_handler.py` (658 lines, ~17 methods, 9 dependents) |

**Partially addressed (2026-04-11):** D-01 partial split executed. PredictionFrame output path (`to_evaluation_pf`, `_valid_cell_indices`, `_reconstruct_as_pf_dict`, ~127 lines) extracted into `PredictionFrameAssembler`. VolumeHandler shrunk from 787 → 658 lines, 20+ → ~17 methods. Inference orchestrator now imports the assembler explicitly; training loop never depended on the PF path.

**Residual concern:** VolumeHandler still exposes a single interface with ~17 methods spanning data ingestion (`from_df`), model entry (`to_pytorch`), prediction wrapping (`wrap_predictions`), spatial/temporal manipulation (`slice_time`, `extrapolate_time`, `flip`, `_permute`, `collapse_to_point`), and feature engineering (`_execute_derivations`). These are tighter cohesion than the PF path (all operate on the underlying volume), but the ISP gap is reduced rather than eliminated.

**Graph-quantified coupling (2026-04-21, graphify):** Knowledge graph confirms VolumeHandler as the dominant god node: 451 edges (up from 398 on 2026-04-19), bridging 16+ communities. Edge growth from expanded extraction coverage (ADR/CIC/spec documents). The EXTRACTED method edges define the blast radius boundary — changing any signature ripples across all communities.

Per Martin (Clean Architecture Ch 10, p.100-103): ISP says "avoid depending on things you don't use." See also C-37 (SAP Zone of Pain) and D-01 (resolved — partial split executed).

---

### C-37: VolumeHandler in SAP "Zone of Pain" — partial abstraction at PF boundary

| Field | Value |
|-------|-------|
| ID | C-37 |
| Tier | 4 |
| Source | clean-architecture-review (2026-04-08), updated D-01 execution (2026-04-11) |
| Trigger | Need to provide an alternative VolumeHandler implementation (e.g., lazy-loading, GPU-resident) |
| Location | `volume_handler.py` — fan-in=9, fan-out=1 (after PF extraction) |

**Partially addressed (2026-04-11):** D-01 partial split executed. The PredictionFrameAssembler is now an interface adapter at the entity/framework boundary — its `assemble_evaluation()` method abstracts the PF assembly contract from the underlying volume operations. VolumeHandler's fan-out dropped from 2 to 1 (no more `views_pipeline_core` import).

**Residual concern:** VolumeHandler itself still has no abstract base class, Protocol, or interface definition. The core volume operations (transpose/flip/slice/wrap) remain a concrete monolithic class. Resolving the residual would require extracting an `IVolumeHandler` Protocol — out of scope for the partial split. Currently tolerable because the interface is mature and rarely changes.

See also C-36 (ISP partially addressed) and D-01 (resolved — partial split executed).

---

### C-49: Flat config schema may not scale — no nested structure for regularizers, strategies, or per-target settings

| Field | Value |
|-------|-------|
| ID | C-49 |
| Tier | 4 |
| Source | manual (2026-04-10) |
| Trigger | When the number of flat config keys exceeds ~40-50, or when adding a feature requires 3+ related keys that would be cleaner as a nested group |
| Location | `config_initializer.py` (`HydraNetConfig`), `CORE_GENOME` in `reproducibility_gate.py` |

`HydraNetConfig` uses flat keys for all parameters: `loss_reg_alpha`, `loss_reg_sigma`, `loss_class_gamma`, `qs99_weight`, `qs99_tau`, `hurdle_threshold`, etc. This is consistent, simple, and works well at the current scale (~25 keys). But the pattern has a threshold of inconvenience: as regularizers, per-target settings, and training strategies accumulate, flat keys become ambiguous (does `alpha` belong to the loss, the regularizer, or the curriculum?), hard to group visually in config files, and awkward for the genome audit (which keys belong to which feature?).

The alternative is nested structure: `regularizers: { qs99: { weight: 0.01, tau: 0.99 } }`. This is cleaner for extensibility but breaks the flat-key pattern, complicates Pydantic validation, and requires changes to the genome audit.

Current decision: stay flat. Revisit when either (a) total config keys exceed 50, or (b) a feature requires 4+ related keys that create naming collision risk. The migration is mechanical (rename keys, update configs) but touches every model config in views-models.

---

### C-73: Legacy `evalution_mode` typo shim in HydraNetConfig

| Field | Value |
|-------|-------|
| ID | C-73 |
| Tier | 4 |
| Source | manual (2026-04-21) |
| Trigger | When all model configs in `views-models` have been confirmed to use `evaluation_mode` (not `evalution_mode`), remove the `handle_typos` model_validator shim |
| Location | `config_initializer.py:143-153` (`handle_typos` model_validator) |

`HydraNetConfig` has a `model_validator(mode="before")` shim that silently rewrites the legacy typo key `evalution_mode` → `evaluation_mode`. One known consumer (`views-models`) has been fixed (2026-04-21), but other model configs in the `views-models` repo may still use the old key. The shim should be removed once a grep across all configs in `views-models` confirms zero remaining instances of `evalution_mode`. Removing it prematurely would break any config still using the typo — Pydantic's `extra="allow"` would silently accept the misspelled key and leave `evaluation_mode` at its default.

---

### C-75: Duplicated derivation logic between DataFetcher and VolumeHandler

| Field | Value |
|-------|-------|
| ID | C-75 |
| Tier | 3 |
| Source | tech-debt-cleanup (2026-04-27) |
| Trigger | When adding a new derivation operation type (e.g., "logarithmic", "scaling") to the instructional blueprint, verify both paths are updated identically |
| Location | `utils/data_fetcher.py:144-194` (`apply_blueprint`), `utils/volume_handler.py:595-644` (`_execute_derivations`) |

`DataFetcher.apply_blueprint()` and `VolumeHandler._execute_derivations()` implement nearly identical derivation logic independently — same mandatory-key validation, same "binary" threshold check, same error messages, same `NotImplementedError` for unknown ops. A bug fix or new operation must be applied in both places. `test_derivation_parity.py` guards against divergence, but the duplication itself increases maintenance cost. Extracting a shared `DerivationEngine` would centralize the logic.

See also C-36 (VolumeHandler ISP), D-01 (VolumeHandler scope).

---

### C-76: `apply_blueprint` hard-codes operation types — violates OCP

| Field | Value |
|-------|-------|
| ID | C-76 |
| Tier | 4 |
| Source | tech-debt-cleanup (2026-04-27) |
| Trigger | When adding a second derivation operation type beyond "binary" |
| Location | `utils/data_fetcher.py:178-192`, `utils/volume_handler.py:635-644` |

`apply_blueprint()` uses an `if op == "binary" ... else raise NotImplementedError` conditional. Adding a new operation (e.g., log-transform, z-score) requires modifying the method body. A strategy pattern or operation registry would make the method open for extension without modification. Currently low urgency — only "binary" is used across all configs.

See also C-75 (duplicated derivation logic).

---

### C-79: No pipeline-level reproducibility comparison test

| Field | Value |
|-------|-------|
| ID | C-79 |
| Tier | 4 |
| Source | /falsify merge-readiness audit P2 (2026-05-26), originally noted in C-42 resolution |
| Trigger | When modifying inference orchestrator, posterior sampling, or aggregation logic — no test verifies that two identical runs produce identical outputs |
| Location | `tests/test_falsification_cradle_to_grave.py` (F3-06 stub), `views_hydranet/utils/hydranet_inference.py`, `views_hydranet/utils/inference_orchestrator.py` |

`ReproducibilityGate.lock_entropy()` sets all RNG seeds (C-42 resolved), but no test actually runs the inference pipeline twice with the same seeds and compares outputs. Pipeline-level determinism is assumed, not proven. The F3-06 falsification stub encodes this gap. Noted as a "residual test gap" in C-42 resolution text but never registered.

See also C-42 (resolved — entropy locking).

---

### C-87: Hurdle mechanism applies uniform loss parameters across targets with different rare-event ratios

| Field | Value |
|-------|-------|
| ID | C-87 |
| Tier | 3 (was 2, mitigated by Path A) |
| Source | manual review (2026-05-27), informed by repeated OS/NS near-zero prediction failures |
| Trigger | When training with hurdle + Basu DPD on multi-target configs where OS and NS are 3-5x rarer than SB — gradient starvation causes the model to effectively abandon rare-target regression heads |
| Location | `views_hydranet/train/training_engine.py:169-196` (hurdle loop), `views_hydranet/utils/config_initializer.py` (no per-target param support) |

The hurdle mechanism masks regression loss to positive observations per-target, but all targets share a single `loss_reg` (Basu DPD with global alpha/sigma), a single `qs99_weight`/`qs99_tau`, and no per-target loss weighting. When SB has ~5% non-zero cells and OS/NS have ~1%, the regression gradient signal for rare targets is ~5x weaker. Historical outcome: models predict near-zero for OS and NS.

**Path A (IMPLEMENTED):** Per-target loss weighting via `target_weights` config dict (`Dict[str, float] | None`). Applied inside the hurdle loop at per-target loss computation — multiplies regression loss, QS99 penalty, and non-hurdle loss by the configured weight. Validator enforces all regression targets present and non-negative weights. Tested by 4 integration tests in `test_hurdle_basu_integration.py`.

**Path B (future — per-target Basu parameters):** Separate `alpha`/`sigma` per target. OS gets lower alpha (more sensitivity), SB gets higher alpha (more robustness). More expressive but requires nested config structure (C-49 scaling concern). Deferred until Path A proves insufficient.

Tier 2 → Tier 3: Path A mitigates the immediate gradient starvation. Residual risk is that uniform alpha/sigma may still under-serve rare targets if weight scaling alone is insufficient.

**Test gap (test-review 2026-05-27):** `target_weights` is only tested with a single-target config. No test verifies correct per-target weight application with multiple regression targets — a bug in target-name lookup would pass single-target tests but silently misweight in production. See C-88.

See also ADR-050 (hurdle-decomposed loss), C-49 (flat config scaling).

---

### C-85: Flip probability 0.5 hardcoded in training_engine — not config-driven

| Field | Value |
|-------|-------|
| ID | C-85 |
| Tier | 4 |
| Source | /falsify magic-numbers audit P1 (2026-05-27) |
| Trigger | When running augmentation sensitivity experiments and needing flip probability other than 0.5 — requires source code change instead of config change |
| Location | `views_hydranet/train/training_engine.py:290-292` |

Data augmentation flip on/off is config-driven (`random_flips: bool`), but the flip probability is hardcoded at `0.5` (fair coin). This is the only behavior-affecting numeric literal in `training_engine.py` that isn't sourced from config. Symmetric by definition (H/W flips), so `0.5` is defensible — but a researcher doing augmentation experiments would need to modify source code to test other probabilities.

See also C-65 (resolved — `random_flips` added to schema).

---

### C-89: `_SumReducer` and `_make_tiny_model` duplicated across test files

| Field | Value |
|-------|-------|
| ID | C-89 |
| Tier | 4 |
| Source | /test-review (Beck W1) (2026-05-27) |
| Trigger | When modifying `ModelOutput` or the model forward signature — both copies must be updated independently, and forgetting one produces confusing test failures |
| Location | `tests/test_cluster_e.py:317-340`, `tests/test_hurdle_basu_integration.py:327-346` |

Identical `_SumReducer` and `_make_tiny_model` helpers are defined in two test files. A third copy is likely in the next PR that adds `_process_sequence` tests. Should be extracted to `conftest.py` as shared fixtures.

**Path E amplification (2026-05-29):** Scheduled sampling implementation (issue #37) will require another copy of the tiny model fixture for `tests/test_scheduled_sampling.py`. Extract to `conftest.py` before implementing Path E tests to avoid a fourth copy.

**Test review amplification (2026-06-02):** Additionally, `_tobit_config()` helper is duplicated across 3 test files (test_tobit_loss.py, test_per_target_sigma.py, test_learnable_sigma.py) with slightly different base configs. Same DRY concern, different fixture.

Tier 4 rationale: code quality / DRY violation. Single-developer scope. No correctness impact.

---

### C-93: `_evaluate_sweep` not implemented — sweep evaluation crashes with `NotImplementedError`

| Field | Value |
|-------|-------|
| ID | C-93 |
| Tier | 2 |
| Source | expert-code-review (2026-05-28), falsify merge-readiness (2026-05-28) |
| Trigger | Running `python main.py -r calibration --sweep` on any HydraNet model — training completes but evaluation crashes, aborting the sweep agent |
| Location | `views_hydranet/manager/hydranet_manager.py` (missing override), `views_pipeline_core/managers/model/model.py:780-820` (abstract contract) |
| Cross-refs | C-01, D-04 |

`HydranetManager` implements `_evaluate_model_artifact` (single runs) but not `_evaluate_sweep` (wandb sweeps). The base class `ForecastingModelManager` marks it `@abstractmethod`. Root cause: `_setup_evaluation()` (lines 224-314) couples model loading (lines 268-279) with data pipeline + orchestrator wiring (lines 281-314). Sweep needs the data pipeline but not the disk load — the model is in-memory. Fix requires decomposing `_setup_evaluation()`: extract model loading into `_load_model_artifact()`, make `model` a required parameter of `_setup_evaluation()`, then add a 5-line `_evaluate_sweep()` override. Sibling managers (views-baseline, views-stepshifter) implement this method; HydraNet is the only one missing it.

Tier 2 rationale: structural fragility — every sweep run crashes today. Clear trigger. Blocks hyperparameter optimization workflow.

---

### C-95: Tobit S2 MCR asymmetry — lr_sb=0.983, lr_os=0.005 — systematic calibration bias

| Field | Value |
|-------|-------|
| ID | C-95 |
| Tier | 3 |
| Source | S2 Tobit experiment (2026-05-29), wandb run summary |
| Trigger | When evaluating Path E (scheduled sampling) results against the S2 baseline — MCR asymmetry may persist or worsen, and should be diagnosed before declaring Gate 2 passed |
| Location | Evaluation metrics (wandb), not a code defect. Upstream: `views_hydranet/utils/tobit_loss.py`, `views_hydranet/train/training_engine.py` |
| Cross-refs | C-87 (per-target loss weights) |

S2 Tobit experiment (150 lessons, `loss_reg=tobit`, `loss_reg_sigma=1.0`) shows extreme MCR asymmetry across targets: lr_sb MCR_sample=0.983 (nearly all predictions above marginal median — systematic upward bias), lr_os MCR_sample=0.005 (nearly all below — systematic underprediction). The sample-vs-mean gap for lr_sb (0.983 sample vs 0.555 mean) indicates individual posterior samples are consistently biased high while the posterior mean is more centered — the stochastic spread does not straddle the median.

This is not a code defect but a model behavior concern. Possible causes: (1) Tobit censored likelihood with σ=1.0 may overestimate latent z* for zero-cells, pushing predictions upward for the most zero-inflated target (SB ~95% zeros). (2) Per-target loss weights may need recalibration for Tobit (current weights were tuned for hurdle+Basu). (3) The fixed σ may be too large or too small for different targets.

Tier 3 rationale: model quality concern that affects evaluation interpretation, not silent corruption. No code fix needed — requires experimental investigation (σ sensitivity, per-target σ, target_weights recalibration).

---

### C-96: Tobit loss converges in ~60 lessons — total_lessons=150 wastes ~60% training compute

| Field | Value |
|-------|-------|
| ID | C-96 |
| Tier | 4 |
| Source | S2 Tobit experiment (2026-05-29), training loss curves |
| Trigger | When configuring `total_lessons` for Tobit loss runs — using the MSE/Shrinkage-calibrated default of 20 lessons is too few, but 150 is excessive |
| Location | Config parameter `total_lessons` in model configs (`views-models`), `views_hydranet/utils/config_initializer.py:105` |

S2 training curves (linear and log-scale) show regression loss plateauing at ~25.8 by lesson 60, with lessons 60-150 oscillating in a ±0.3 noise band (log-scale) around the plateau. Classification loss shows similar convergence by lesson 60 (current: 3.15). Total multi-task loss converges to ~48 by lesson 60.

Tobit converges faster than hurdle+MSE because it provides dense gradient from ALL cells (including y=0 censored observations), eliminating the gradient starvation that slowed MSE convergence. The optimal `total_lessons` for Tobit is likely 60-80, saving ~50-60% training time compared to 150.

Tier 4 rationale: efficiency concern, not correctness. Training produces correct results at 150 lessons, just wastes compute. Single-developer scope.

---

### C-97: Step-wise CRPS degradation quantifies exposure bias — Path E baseline metric

| Field | Value |
|-------|-------|
| ID | C-97 |
| Tier | 3 |
| Source | S2 Tobit experiment (2026-05-29), wandb run summary |
| Trigger | When evaluating Path E (scheduled sampling) Gate 2 — compare step-wise CRPS against this S2 baseline to determine if scheduled sampling reduces long-horizon degradation |
| Location | Evaluation metrics (wandb). Code path: `views_hydranet/utils/hydranet_inference.py:292-295` (autoregressive feedback loop) |
| Cross-refs | Issue #37 (Path E), Issue #42 (roadmap Gate 2) |

S2 Tobit baseline step-wise CRPS: lr_sb=0.166, lr_ns=0.047, lr_os=0.052. Month-wise CRPS: lr_sb=0.147, lr_ns=0.046, lr_os=0.074. The step-wise > month-wise gap for lr_sb (13% degradation) and lr_ns (3%) is consistent with exposure bias: prediction errors compound over the 36-step autoregressive horizon because the model was trained only on ground-truth inputs (teacher forcing) but sees its own predictions during inference.

lr_os shows the opposite pattern (step-wise 0.052 < month-wise 0.074) — likely because one-sided violence is so rare that longer horizons average out noise rather than compounding errors.

This entry serves as the quantitative Gate 2 baseline. Path E (scheduled sampling) should reduce the lr_sb step-wise/month-wise gap. If it does not, escalate to GTF per the roadmap.

Tier 3 rationale: quantified performance gap affecting forecast quality at operational horizons (36 steps). Not a code defect but a structural limitation of pure teacher forcing.

---

### C-98: Implicit `input_channels == 3 × output_channels` constraint — unvalidated architectural invariant

| Field | Value |
|-------|-------|
| ID | C-98 |
| Tier | 3 |
| Source | Path E exploration (2026-05-29) |
| Trigger | When creating a new model config where `input_channels ≠ 3 × output_channels` — autoregressive inference will crash with a cryptic Conv2d shape mismatch, and scheduled sampling will crash similarly during training |
| Location | `views_hydranet/utils/hydranet_inference.py:294` (`t0_autoreg = t1_pred.detach()`), `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:520` (`torch.concat([out_reg1, out_reg2, out_reg3])`), `views_hydranet/utils/config_initializer.py` (missing validation) |
| Cross-refs | C-03 (hardcoded 3+3 heads), D-02 (architecture extensibility) |

The model architecture has 3 hardcoded regression decoder heads, each producing `output_channels` channels. The concatenated regression output has shape `[B, 3 × output_channels, H, W]`. During autoregressive inference, this output is fed directly as the next input, which expects `[B, input_channels, H, W]`. This requires `input_channels == 3 × output_channels` — an invariant that is never validated at config construction or model initialization.

In practice, all configs use `output_channels=1` and `input_channels=3` (3 features = 3 regression targets), so the constraint holds. But it is enforced only by convention. Scheduled sampling (Path E) will add a second code path that depends on this same invariant during training. A single-line validator in `HydraNetConfig.validate_laws()` would make this explicit: `if self.input_channels != 3 * self.output_channels: raise ValueError(...)`.

Tier 3 rationale: a misconfigured config produces a cryptic error deep in the forward pass. Multiple developers (anyone writing configs) could trigger it. Fix is trivial but the invariant should be documented.

---

### C-99: Tobit `reg_latent` vs `reg` dual-path creates refactoring hazard for scheduled sampling

| Field | Value |
|-------|-------|
| ID | C-99 |
| Tier | 4 |
| Source | Path E exploration (2026-05-29) |
| Trigger | When implementing scheduled sampling — ensure the mixer uses `output.reg` (post-ReLU, non-negative) as the feedback input, NOT `output.reg_latent` (pre-ReLU, can be negative). A future refactoring that consolidates the two paths could introduce negative inputs to the autoregressive loop. |
| Location | `views_hydranet/train/training_engine.py:150-151` (latent routing for loss), `views_hydranet/train/training_engine.py:145-147` (forward pass output), `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:519-523` (reg_latent vs reg) |
| Cross-refs | ADR-054 (Tobit loss) |

`_process_sequence()` uses `output.reg_latent` (pre-ReLU latent μ, can be negative) for Tobit loss computation and `output.reg` (post-ReLU, non-negative) for everything else including forensic recording. Scheduled sampling must use `output.reg` as the feedback input — the model's input features are non-negative (log1p-transformed fatality counts), and `reg_latent` values can be arbitrarily negative.

The two paths are currently distinct (line 150: `t1_pred_for_loss = output.reg_latent if use_latent else t1_pred`, line 147: `t1_pred = output.reg`). But they originate from the same forward pass, and a refactoring that merges variable names or simplifies the output handling could accidentally route `reg_latent` into the scheduled sampling mixer. A unit test should assert that the mixer input is always non-negative.

Tier 4 rationale: no current bug. Single-developer scope. The risk is future-facing and easily mitigated with a test assertion.

---

### C-100: `validate_basu_dpd_range` TypeError crash when `loss_reg_sigma` is dict — validator ordering hazard

| Field | Value |
|-------|-------|
| ID | C-100 |
| Tier | 2 |
| Source | /falsify per-target sigma audit P1 (2026-05-30) |
| Trigger | When constructing `HydraNetConfig` with `loss_reg='basu_dpd'` and `loss_reg_sigma={'lr_sb': 1.0, ...}` — the basu validator does `self.loss_reg_sigma <= 0` which raises `TypeError: '<=' not supported between instances of 'dict' and 'int'` instead of a clean `ValueError` |
| Location | `views_hydranet/utils/config_initializer.py:453` (`validate_basu_dpd_range`) |
| Cross-refs | C-49 (flat config scaling), issue #44 |

After `loss_reg_sigma` was widened from `float | None` to `float | Dict[str, float] | None` for per-target Tobit sigma (issue #44), the existing `validate_basu_dpd_range` validator crashes with a `TypeError` when given a dict. The validator runs before `validate_per_target_sigma` (which would cleanly reject dict sigma for non-tobit losses), so the user gets a cryptic crash instead of a helpful error message.

Fix: guard the comparison with `isinstance(self.loss_reg_sigma, (int, float))` before `<= 0`, or move `validate_per_target_sigma` above `validate_basu_dpd_range`.

Tier 2 rationale: structural fragility — a misconfigured config produces an unhandled TypeError (not a ValueError), which may not be caught by error handlers expecting ValueError. Clear trigger exists.

---

### C-101: Extra keys in per-target sigma dict silently accepted — typo masking

| Field | Value |
|-------|-------|
| ID | C-101 |
| Tier | 4 |
| Source | /falsify per-target sigma audit P5 (2026-05-30) |
| Trigger | When writing a per-target sigma config with a typo like `'lr_TYPO': 2.0` alongside valid targets — the validator checks for missing targets but not extra ones, so the typo is silently included |
| Location | `views_hydranet/utils/config_initializer.py:543-548` (`validate_per_target_sigma`) |
| Cross-refs | Issue #44 |

The `validate_per_target_sigma` validator checks that all `regression_targets` have entries in the dict, but does not check for extra keys. A config with `{'lr_sb': 1.0, 'lr_ns': 0.75, 'lr_os': 0.5, 'lr_TYPO': 2.0}` passes validation. The extra key is harmless at runtime (unused), but masks configuration errors.

Fix: add `extra = [k for k in self.loss_reg_sigma if k not in self.regression_targets]` check.

Tier 4 rationale: no correctness impact (extra keys are ignored). Single-developer scope.

---

### C-102: Stale type annotations for `criterion_reg` after per-target sigma change

| Field | Value |
|-------|-------|
| ID | C-102 |
| Tier | 4 |
| Source | /falsify per-target sigma audit P2 (2026-05-30) |
| Trigger | When a type checker (mypy, pyright) or IDE inspects `_process_sequence` or `TrainingContext` — the annotation `criterion_reg: nn.Module` will flag dict values as errors, and `choose_loss` return type is wrong |
| Location | `views_hydranet/train/training_engine.py:109`, `views_hydranet/train/training_engine.py:266`, `views_hydranet/utils/utils.py:108` |
| Cross-refs | Issue #44 |

Three type annotations still declare `criterion_reg` as `nn.Module` after the per-target sigma change made it `nn.Module | dict[str, nn.Module]`. The `choose_loss` return type annotation `tuple[nn.Module, nn.Module, MultiTaskLoss]` is also stale. No runtime impact (Python doesn't enforce annotations), but misleads static analysis and IDE users.

Tier 4 rationale: code quality. No correctness or reliability impact.

---

### C-103: CIC HydraNetConfig.md stale after per-target sigma — missing failure mode and type change

| Field | Value |
|-------|-------|
| ID | C-103 |
| Tier | 4 |
| Source | /falsify per-target sigma audit P4 (2026-05-30) |
| Trigger | When a contributor reads the CIC to understand `loss_reg_sigma` validation behavior — the CIC says `loss_reg_sigma` is float and lists no per-target failure modes |
| Location | `docs/CICs/HydraNetConfig.md` Section 3 and Section 6 |
| Cross-refs | C-55 (resolved — CIC drift pattern), issue #44 |

The CIC for HydraNetConfig does not document: (1) `loss_reg_sigma` now accepts `Dict[str, float]` for per-target Tobit sigma, (2) the `validate_per_target_sigma` validator and its three failure modes (non-tobit, non-positive, missing target), (3) the updated field count (64 → still 64, but type changed).

Tier 4 rationale: documentation drift. Same pattern as C-55 (resolved), but a new instance. No correctness impact.

---

### C-104: ParetoLoss registered in LOSS_REG_REGISTRY but has no test file

| Field | Value |
|-------|-------|
| ID | C-104 |
| Tier | 4 |
| Source | repo-assimilation (2026-06-02) |
| Trigger | When a researcher sets `loss_reg='pareto'` in a model config — the loss function is untested and could silently produce wrong gradients |
| Location | `views_hydranet/utils/pareto_loss.py`, `views_hydranet/utils/utils.py:76-79` (LOSS_REG_REGISTRY entry) |

`ParetoLoss` is registered in `LOSS_REG_REGISTRY` with `loss_reg='pareto'` and config parameter `loss_reg_pareto_alpha`. The config validator accepts it. But no `test_pareto_loss.py` exists — the loss function has zero test coverage. All other 6 loss functions have dedicated test files.

Tier 4 rationale: no production config uses pareto today. Single-developer scope. The risk is dormant until someone enables it.

---

### C-105: No validator enforces `features ⊆ regression_targets` — shape mismatch risk in autoregressive feedback

| Field | Value |
|-------|-------|
| ID | C-105 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-02) |
| Trigger | When creating a config where `features` includes columns not in `regression_targets` (e.g., confounders, lagged variables) — the model's regression output has fewer channels than its input, causing a shape mismatch during autoregressive inference |
| Location | `views_hydranet/utils/config_initializer.py` (missing validator), `views_hydranet/utils/hydranet_inference.py:294` (`t0_autoreg = t1_pred.detach()`) |
| Cross-refs | C-98 (input_channels == 3 × output_channels), C-03 (hardcoded 3+3 heads) |

During autoregressive inference, `output.reg` (shape `[B, 3*output_channels, H, W]`) feeds back as the next input (expected shape `[B, input_channels, H, W]`). This requires `features` and `regression_targets` to have the same count. The Feature Lifecycle Law (ADR-046) validates that all columns are accounted for in transformations, but it does NOT validate that `features == regression_targets`.

A config with `features=['lr_sb', 'lr_ns', 'lr_os', 'temperature']` and `regression_targets=['lr_sb', 'lr_ns', 'lr_os']` would pass all validators but crash during inference with a cryptic Conv2d shape mismatch.

Tier 3 rationale: the trigger is realistic (adding confounder features is a natural researcher action). The error is cryptic, not at the config boundary. Multiple developers could hit this.

---

### C-107: `hydranet_inference.py` (380 lines) has zero unit tests — only integration coverage

| Field | Value |
|-------|-------|
| ID | C-107 |
| Tier | 4 |
| Source | test-review (2026-06-02) |
| Trigger | When modifying drift detection thresholds, freeze_h option dispatch, or autoregressive loop logic — no isolated test catches regressions; only discovered through slow integration tests |
| Location | `views_hydranet/utils/hydranet_inference.py` (380 lines), `tests/` (no test_hydranet_inference.py exists) |

`HydraNetInference` is the 4th largest module (380 lines) implementing the entire autoregressive inference loop: history digestion, seed step, 36-step autoregression, freeze_h dispatch (5 options), and drift detection. All testing is via `InferenceOrchestrator` integration tests — no unit tests verify the individual behaviors. If drift thresholds change (C-51) or freeze_h logic changes, regressions are caught only by slow end-to-end tests.

Tier 4 rationale: integration tests do provide coverage. The gap is speed and isolation, not correctness. Single-developer scope.

---

### C-108: 46% of test classes (82/178) lack ADR-005 taxonomy markers (Green/Beige/Red)

| Field | Value |
|-------|-------|
| ID | C-108 |
| Tier | 4 |
| Source | test-review (2026-06-02) |
| Trigger | When auditing test coverage for a specific component — unable to tell from class names whether error paths (Red) are tested, only whether tests exist at all |
| Location | 30+ test files across `tests/` |
| Cross-refs | C-60 (resolved — initial taxonomy adoption, 16% → 36%) |

C-60 was resolved in April 2026, bringing taxonomy adoption from 16% to 36%. Since then, 33+ new test files have been added (per-target sigma, learnable sigma, scheduled sampling, falsification stubs) and the overall test count grew from ~350 to 704. Many new tests DO use the taxonomy (TestGreen, TestRed), but 82 classes across 30 files predate the convention or were added without markers.

Tier 4 rationale: test quality, not correctness. All tests run and pass. The gap is visibility — a developer can't quickly assess Red coverage by scanning class names.

---

### C-109: 13 skipped falsification tests are stale investigation artifacts

| Field | Value |
|-------|-------|
| ID | C-109 |
| Tier | 4 |
| Source | test-review (2026-06-02) |
| Trigger | When running the full test suite — 13 skipped tests create noise in the output and inflate the "investigated" impression without providing current value |
| Location | `tests/test_falsification_identical_window_selection.py`, `tests/test_falsification_sensitivity_attribution.py`, `tests/test_falsification_sweep_root_cause.py`, `tests/test_falsification_sweep_understanding.py`, `tests/test_falsification_two_phase_divergence.py` |

13 tests are permanently skipped (`pytest.skip()`) — they reference investigation experiments (purple_alien divergence, sweep root cause) that concluded in May 2026. The investigations produced findings registered in the risk register and resolved. The skipped tests no longer serve as active probes — they're preserved as historical artifacts but add noise to test output.

Triage options: (a) convert to xfail with documented reason, (b) delete and reference the investigation commit, (c) convert to passing verification tests if the underlying claim can now be tested.

Tier 4 rationale: no correctness impact. Test suite hygiene.

---

## Disagreements

### D-01: VolumeHandler scope — God Object vs Deep Module

| Field | Value |
|-------|-------|
| ID | D-01 |
| Source | expert-review (2026-04-08) |
| Perspectives | Martin (split — SRP Ch 7 p.80: serves 4 actors; ISP Ch 10 p.100: 20+ method interface; SAP Ch 14 p.139: Zone of Pain), Ousterhout (keep — successful deep module hiding complexity), Hickey (partial split — extract PF output path, keep volume ops together) |
| Resolution | **Executed (2026-04-11):** Partial split implemented. Extracted `to_evaluation_pf`, `_valid_cell_indices`, `_reconstruct_as_pf_dict` (~127 lines) from `volume_handler.py` into a new `views_hydranet/utils/prediction_frame_assembler.py` containing a stateless `PredictionFrameAssembler` class. VolumeHandler shrunk from 787 → 658 lines and no longer imports `views_pipeline_core` (fully resolves C-39). `InferenceOrchestrator` now constructs an assembler instance and calls `assembler.assemble_evaluation(signal=..., history=..., start_idx=..., all_targets=...)` instead of `pred_handler.to_evaluation_pf(...)`. C-36 and C-37 remain open as "partially addressed" — VolumeHandler still has ~17 methods and no abstract base class, but the Framework-layer dependency is gone and the worst ISP offender (PF path) is extracted. Ousterhout's "deep module" counter-argument remains valid for the surviving core volume operations. Hickey's "partial split" recommendation was followed. |

---

### D-02: Architecture extensibility — parameterize vs leave alone

| Field | Value |
|-------|-------|
| ID | D-02 |
| Source | expert-review (2026-04-08) |
| Perspectives | GoF (parameterize — 6 copy-pasted decoder blocks is anti-pattern), Beck/Feathers (leave alone — structural regex test guards against bugs, refactoring invalidates all .pt artifacts) |
| Resolution | Leave as-is. Cost of refactoring (breaking all artifacts) exceeds benefit. Structural test in `tests/test_architecture.py` provides adequate safety — this test is load-bearing infrastructure; do not modify without understanding its role as the guard for this decision. |

---

### D-03: Config monolith — complecting vs front-loading validation

| Field | Value |
|-------|-------|
| ID | D-03 |
| Source | expert-review (2026-04-08) |
| Perspectives | Hickey (split — 9 concerns conflated in one model), Ousterhout/Nygard (keep — single validation point, cross-field checksums require all fields visible) |
| Resolution | Keep single config. Cross-field checksum laws depend on simultaneous field access. |

---

### D-04: `_evaluate_sweep` fix — parameter injection vs method extraction

| Field | Value |
|-------|-------|
| ID | D-04 |
| Source | expert-code-review (2026-05-28) |
| Perspectives | Beck (add `model` param to `_setup_evaluation()` — simplest change), Martin/Hickey (decompose — extract `_load_model_artifact()`, make `model` required param, no boolean branching), Ousterhout (preserve method depth — don't fragment the setup into many shallow pieces) |
| Resolution | **Consensus: Option C (decompose).** Make `model` a *required* parameter of `_setup_evaluation()`. Extract model loading into `_load_model_artifact()`. Three existing callers load then pass; sweep passes in-memory model. No boolean params, no duplication, no complecting. See C-93. |

---

## Resolved Concerns

### C-94: `reg_latent` tensor allocated during inference — wasteful memory — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-94 |
| Resolved | 2026-05-29 |
| Resolution | Gated `out_reg_latent` concatenation on `self.training` in `forward()`. Eval mode now returns `reg_latent=None`, eliminating one `[B, C, H, W]` allocation per autoregressive step. Verified by `test_reg_latent_none_in_eval_mode`. |

---

### C-88: No integration test for `target_weights` multi-target application — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-88 |
| Resolved | 2026-05-27 |
| Resolution | Added `test_train_with_hurdle_basu_target_weights` in `tests/test_hurdle_basu_integration.py::TestGreenTrainEntryPoint` — exercises the full `train()` → `config.get()` → `_process_sequence` path with Basu DPD, hurdle, QS99, and target_weights. Also added `test_target_weights_multi_target_applies_per_target` with 2-channel model and asymmetric weights to verify per-target weight application. See C-87. |

---

### C-08: North-Up flip symmetry is implicitly coupled — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-08 |
| Resolved | 2026-04-08 (initial), 2026-05-28 (hardened) |
| Resolution | **Phase 1 (2026-04-08):** Added `test_gate_flip_symmetry_from_df_to_output` in `tests/test_volume_handler_hard_gates.py`. **Phase 2 (2026-05-28):** Full hardening — added `SpatialConvention` enum to `VolumeMetadata` (GEOGRAPHIC/NORTH_UP), `raise ValueError` guards in `PredictionFrameAssembler._valid_cell_indices()`, convention propagation through all 8 VolumeHandler creation sites, and 40 tests across `tests/test_flip_symmetry_hardening.py` (32 tests) and `tests/test_falsification_flip_hardening.py` (8 tests) covering round-trips, source inspection, domain-knowledge invariants (hemisphere land ratios), augmentation, visualization, and convention propagation paths. |

---

### C-91: SpatialConvention propagation through pipeline methods has incomplete test coverage — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-91 |
| Resolved | 2026-05-28 |
| Resolution | Added 6 dedicated convention-preservation tests in `tests/test_falsification_flip_hardening.py::TestF01ConventionPropagation` covering `slice_time`, `extrapolate_time`, `wrap_predictions`, `collapse_to_point`, `inverse_transform_volume`, and `_permute`. Added asymmetric mismatch test (provider=GEOGRAPHIC, signal=NORTH_UP) in `TestF02AsymmetricMismatch`. All propagation sites now have regression coverage. |

---

### C-92: Convention guards in PredictionFrameAssembler use `assert` (stripped by `-O`) — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-92 |
| Resolved | 2026-05-28 |
| Resolution | Upgraded convention guards in `PredictionFrameAssembler._valid_cell_indices()` from `assert` to `if not ...: raise ValueError(...)`. Guards are now unconditional regardless of Python optimization level. Verified by source-inspection test in `tests/test_falsification_flip_hardening.py::TestF07GuardsUseRaise`. |

---

### C-90: CIC HydraNetConfig §10 Test Alignment stale — missing new test files — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-90 |
| Resolved | 2026-05-27 |
| Resolution | Updated CIC §10 to list all test files covering HydraNetConfig: `test_config_typed.py` (green), `test_config_validation.py` (beige + red), `test_falsification_hurdle_params.py` (red), `test_falsification_loss_param_validation.py` (red), `test_hurdle_basu_integration.py` (green + beige + red). |

---

### C-84: `_process_sequence` guard does not check `qs99_tau is not None` — TypeError on direct call — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-84 |
| Resolved | 2026-05-27 |
| Resolution | Added `qs99_tau is not None` to the guard at `training_engine.py:183`. Now all four conditions (`qs99_weight is not None`, `qs99_weight > 0`, `qs99_tau is not None`, `mask.any()`) must hold before QS99 arithmetic executes. See also C-81. |

---

### C-86: Four `config.get` calls with shadow or contradictory fallback defaults — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-86 |
| Resolved | 2026-05-27 |
| Resolution | Removed all four fallback defaults: `config.get("random_flips")`, `config.get("clip_grad_norm")`, `config.get("regression_targets")`, `config.get("classification_targets")`. Schema guarantees all fields present after `HydraNetConfig` validation. 2 guard tests in `tests/test_falsification_magic_numbers.py::TestRedShadowDefaults`. |

---

### C-81: QS99 parameters accept out-of-domain values — no range validation — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-81 |
| Resolved | 2026-05-27 |
| Resolution | Added `ge=0.0` constraint on `qs99_weight` and `gt=0.0, lt=1.0` constraints on `qs99_tau` via Pydantic Field validators. 3 red tests in `tests/test_falsification_hurdle_params.py::TestRedQS99Range`. |

---

### C-82: ADR-050 §5 red-team claim unimplemented — Basu α=0, σ=0 accepted at config — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-82 |
| Resolved | 2026-05-27 |
| Resolution | Added `validate_basu_dpd_range` model validator: rejects `loss_reg_alpha <= 0` and `loss_reg_sigma <= 0` when `loss_reg='basu_dpd'`. 2 red tests in `tests/test_falsification_hurdle_params.py::TestRedBasuDegenerate`. ADR-050 §5 claim now matches implementation. See also C-05. |

---

### C-83: Risk register C-48 resolution text stale after parameter hardening — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-83 |
| Resolved | 2026-05-27 |
| Resolution | Updated C-48 resolution text to reflect `None` defaults and strict conditional validation. |

---

### C-77: Power-law sampling strategy overflows float64 with extreme alpha — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-77 |
| Resolved | 2026-05-26 |
| Resolution | Replaced direct exponentiation `flat ** alpha` with log-space arithmetic: `alpha * np.log(flat)` followed by log-sum-exp normalization (same pattern as Boltzmann strategy). All three soft strategies now use consistent numerical stabilization. Falsification test stub `test_falsify_p4_power_law_extreme_alpha_overflow` flipped GREEN. |

---

### C-78: Sampling strategy test suite has four coverage gaps — low discriminative power — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-78 |
| Resolved | 2026-05-26 |
| Resolution | All four gaps addressed: (P1) added `test_green_ratio_matches_power_law_formula` — verifies p(a)/p(b) ≈ (act_a/act_b)^α, catches Boltzmann substitution. (P2) added `test_green_high_steepness_recovers_threshold` to TestSigmoid. (P3) added parametrized `TestNonDefaultStrategyIntegration` testing power_law/boltzmann/sigmoid through VolumeSampler. (P4) added `test_red_invalid_sampling_strategy` to test_config_validation.py. |

---

### C-80: Sentinel injection pattern for multi-field error collection has no direct test — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-80 |
| Resolved | 2026-05-26 |
| Resolution | Added `test_red_multi_field_missing_reports_all_errors` in `tests/test_config_validation.py`. Test removes 3 sentinel-governed fields (`sampling_strategy`, `evaluation_mode`, `loss_reg`) simultaneously and asserts all 3 appear in the resulting `ValidationError`. Sentinel injection pattern now has direct coverage. |

---

### C-09: `torch.save(model)` full-object serialization — no integrity verification — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-09 |
| Tier | 2 |
| Source | repo-assimilation (2026-04-08), updated falsify-F2-07 (2026-04-19) |
| Trigger | Loading a `.pt` artifact from shared storage, CI pipeline, or network transfer — corrupted/tampered file executes arbitrary code or produces garbage predictions silently |
| Location | `train/train_model.py:70`, `utils/model_artifact_fetcher.py:94` |

Full model (not `state_dict`) is pickled via `torch.save()`. This couples saved `.pt` artifacts to the exact class definition and module path. `weights_only=False` in load confirms full-object deserialization — a known arbitrary code execution vector (PyTorch CVE class).

**Upgraded from Tier 3→2 (falsify-F2-07, 2026-04-19):** Three compounding gaps make this deployment-blocking: (1) `weights_only=False` enables pickle ACE, (2) no hash/checksum verification — corrupted files load silently, (3) no config snapshot saved alongside artifact — no way to verify that loaded model matches expected architecture beyond a hardcoded 3+3 head check in `_run_preflight_check()`. For high-stakes deployment where model artifacts transit shared infrastructure, this is a supply-chain attack surface.

See also C-03 (hardcoded 3+3 heads).

**Resolution (2026-04-20):** Three-part fix: (1) Save switched from `torch.save(model)` to `torch.save(model.state_dict())` in `train_model.py`, eliminating pickle ACE on new artifacts. (2) Architecture config sidecar (`.pt.config.json`) written alongside artifact, enabling model reconstruction without pickle. (3) Dual-mode loader in `model_artifact_fetcher.py`: new-format uses `weights_only=True`; legacy full-object uses `weights_only=False` with deprecation warning and re-save guidance. SHA-256 verification (C-30) already covers integrity. 3 new tests in `test_model_artifact_fetcher.py`.

---

### C-11: Direct _metadata access bypasses encapsulation — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-11 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Change to VolumeMetadata field names or structure |
| Location | `train_model.py:183`, `curriculum.py:45`, `feature_scaler.py:245-255`, `hydranet_inference.py:381` |

Multiple modules reach into `VolumeHandler._metadata.feature_cols`, `._metadata.identity_cols`, etc. instead of using properties. This couples them to the internal dataclass structure. Mitigated by `VolumeMetadata` being a frozen dataclass (structurally stable), but violates encapsulation convention.

**Resolution (2026-04-20):** Added `feature_cols`, `identity_cols`, and `history` properties to VolumeHandler. All external consumers migrated to use properties instead of `_metadata.*` access. Internal callers in `volume_handler.py` still use `_metadata` directly (correct — they are the implementation).

---

### C-13: `_permute()` mutates VolumeHandler in-place — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-13 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Calling `_permute()` on a shared VolumeHandler reference |
| Location | `volume_handler.py:615-635` |

Unlike transformation methods that return new VolumeHandlers (`slice_time`, `collapse_to_point`), `_permute()` modifies `self._data` and `self._metadata` in-place. Inconsistent with the immutable-by-convention pattern. Currently used only in geometric tests, not in production paths. See also C-14 (same mutation pattern on `flip()`).

**Resolution (2026-04-20):** Refactored `_permute()` to return a new VolumeHandler instance with transformed data and updated axes/history. Original instance is never mutated. Immutability verified by `test_permute_returns_new_instance`.

---

### C-14: `flip()` mutates VolumeHandler in-place — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-14 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | Calling `flip()` on a VolumeHandler that is referenced elsewhere |
| Location | `volume_handler.py:637-653` |

Like `_permute()` (C-13), `flip()` modifies `self._data` in-place rather than returning a new VolumeHandler. Used in the training augmentation path (`train_model.train()`). Safe in practice because the augmented handler is a per-window copy from `VolumeSampler`, but the mutation pattern is inconsistent with the immutable-by-convention design.

Per Martin (Clean Architecture Ch 6, p.70-76): "Segregation of Mutability" — separate the application into immutable (pure functional) and mutable (transactional) components. `VolumeMetadata` is correctly immutable (`frozen=True`). But `flip()` and `_permute()` break the segregation by mutating `_data` in-place. Martin would say: these are the "transactional memory" components that should be explicitly marked as mutable, or refactored to return new instances.

**Resolution (2026-04-20):** Refactored `flip()` to return a new VolumeHandler instance with flipped data and updated history. Training engine caller updated to `sample_handler = sample_handler.flip(...)`. Immutability verified by `test_flip_returns_new_instance`.

---

### C-16: `visual_diagnostics.py` catch-all exception handlers hide bugs — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-16 |
| Tier | 3 |
| Source | expert-review (2026-04-08) |
| Trigger | Bug in any `biopsy_*` method's plotting logic |
| Location | `visual_diagnostics.py:129-133` (and similar in other biopsy methods) |

All `biopsy_*` methods wrap their body in `try/except Exception` and log on failure. If diagnostic code has a bug, it silently produces no plot with no test failure. The file is 985 lines with 12+ biopsy methods. Tests only verify the `active=True/False` toggle, not plot correctness or exception-free execution. Partial fix (2026-04-08): `biopsy_dataframe` catch block upgraded from `logger.warning` to `logger.error` per ADR-008 Section 4 (Fail-Safe constraint). Catch-all pattern itself is ADR-008 compliant (Observability Actors are permitted Fail-Safe). Remaining concern: plot correctness is untested (see also C-26).

**Resolution (2026-04-20):** Standardized all 9 catch-all handlers to ADR-008 Section 4 Fail-Safe pattern: `logger.error("...", stage_label, exc_info=True)`. All handlers now include full traceback in logs. 8 new tests in `test_visual_diagnostics.py` verify each biopsy method logs ERROR with `exc_info` on failure.

---

### C-19: `priogrid_gid > 0` validity assumption undocumented at ingestion — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-19 |
| Tier | 4 |
| Source | expert-review (2026-04-08) |
| Trigger | Upstream data source assigning `priogrid_gid == 0` to a valid cell |
| Location | `volume_handler.py:505` |

`_valid_cell_indices()` uses `mask = p_data[:, :, :, pg_idx] > 0` to identify valid cells. If any legitimate grid cell has `priogrid_gid == 0`, it is silently dropped from output. This assumption is not enforced or documented at ingestion (`DataSniffer`, `DataFetcher`). In practice, PRIO-GRID assigns GIDs starting from 1, but this is domain knowledge not codified in the system. Tier rationale: impact is catastrophic (silent data loss) but trigger probability is near-zero given PRIO-GRID's established numbering convention. Tier 4 reflects expected risk (impact × likelihood), not impact alone.

**Resolution (2026-04-20):** Added explicit `id_col > 0` check in `DataSniffer._check_identity_values()`. Non-positive IDs now raise `ValueError` at ingestion (Fail Loud), preventing silent data loss downstream.

---

### C-26: VisualDiagnostics plot correctness untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-26 |
| Tier | 4 |
| Source | test-review (2026-04-08) |
| Trigger | Bug in any `biopsy_*` plotting logic |
| Location | `visual_diagnostics.py` (985 lines, 12+ biopsy methods) |

29 tests verify the `active=True/False` toggle and no-crash behavior, but zero tests verify that generated plots contain correct data, have non-zero file size, or reflect the volume state they claim to show. A plotting bug could produce plausible-looking but incorrect visualizations that mislead operators.

**Resolution (2026-04-20):** Test suite now has 37 tests (8 BEIGE null-object, 15 GREEN active-mode with PNG output and math verification, 6 RED adversarial inputs, 8 RED error-logging with `exc_info`). All 8 public biopsy methods have coverage including file output, stats correctness, and error observability. Combined with C-16 resolution, visual diagnostics is comprehensively tested.

---

### C-29: Plausible misconfiguration scenarios untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-29 |
| Tier | 4 |
| Source | test-review (2026-04-08) |
| Trigger | Human operator providing technically valid but degenerate config values |
| Location | `config_initializer.py` (HydraNetConfig), `volume_sampler.py`, `train_model.py` |

No test verifies system behavior with edge-case configurations that Pydantic accepts but produce degenerate behavior: `window_dim=1` (single-pixel patches), `total_lessons=0` (no training), `windows_per_lesson=0` (empty lesson), `learning_rate=1e-20` (effectively zero). These are plausible human errors that pass validation but produce silent quality degradation.

**Resolution (2026-04-20):** Added 4 field_validators to HydraNetConfig: `slope_ratio > 0`, `roof_ratio > 0`, `window_dim >= 2`, `min_ratio < max_ratio`. Added 6 red team tests in `test_degenerate_configs.py`. Degenerate values now raise with diagnostic error messages at config validation time.

---

### C-30: ModelArtifactFetcher has minimal test coverage — zero red team tests — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-30 |
| Tier | 3 |
| Source | test-review (2026-04-08), test-review (2026-04-19) |
| Trigger | Change to artifact loading, device placement, or timestamp extraction logic; malformed artifact or architecture mismatch in production |
| Location | `model_artifact_fetcher.py`, `test_model_artifact_fetcher.py` (3 tests) |

Only 3 tests exist: happy path with latest artifact, happy path with specific artifact, and missing file error. No tests for timestamp extraction edge cases, device placement verification, or the `add_config` callback behavior.

**Test-review update (2026-04-19):** CIC §6 declares three failure modes (Missing Artifact, Checksum Failure, Incompatible Weights). Only Missing Artifact is tested. Zero red team tests exist. Tier upgraded 4→3 because CIC §6 failure modes are untested — a weight shape mismatch or malformed timestamp would produce an uncaught error in production. See CIC §10 for required test alignment.

**Resolution (2026-04-20):** Added 3 red team tests to `test_model_artifact_fetcher.py`: SHA-256 mismatch raises RuntimeError, valid SHA-256 loads successfully, missing hash file warns but loads (legacy compat). Test count: 3 → 10 (2 green, 2 beige, 6 red). CIC §6 Checksum Failure mode now tested.

---

### C-33: InferenceOrchestrator "Sequence Violation" failure mode untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-33 |
| Tier | 4 |
| Source | falsification-audit (2026-04-08) |
| Trigger | Bypassing `_run_inference_pipeline()` step order |
| Location | `inference_orchestrator.py:49-114` |

InferenceOrchestrator CIC Section 6 declares "Sequence Violation" as a failure mode — the system should raise if the ADR 039 step order (Predict → Align → Wrap → Invert → Collapse) is bypassed. In practice, the sequence is enforced by method composition (each step feeds the next), so bypass is unlikely. But the CIC promise is untested.

**Resolution (2026-04-20):** Converted the falsification RED stub to a GREEN verification test that uses `inspect.getsource()` to confirm the pipeline method contains all 5 sequence steps in composition. Sequence is enforced architecturally (data flow), not by explicit state machine — the test verifies the composition is intact.

---

### C-41: Falsification test stubs will fail in CI if not excluded — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-41 |
| Tier | 4 |
| Source | pr-review (2026-04-08) |
| Trigger | CI pipeline collecting `tests/test_falsification_all_risks_identified.py` without explicit exclusion |
| Location | `tests/test_falsification_all_risks_identified.py` (8 `assert False` stubs) |

All 8 original falsification stubs have been converted to passing verification tests: C-31 stubs (4) verify logger.error precedes each raise, C-32 stubs (2) verify test classes exist, C-21 stub (1) verifies except handler compliance, C-33 stub (1) verifies pipeline composition contains all 5 sequence steps.

Residual risk: None. All stubs now pass as GREEN verification tests.

**Resolution (2026-04-20):** CI workflow (`.github/workflows/ci.yml`) updated to `--ignore` all three falsification stub files (`test_falsification_all_risks_identified.py`, `test_falsification_deployment_readiness.py`, `test_falsification_cradle_to_grave.py`). RED stubs remain as audit artifacts; CI no longer fails on them.

---

### C-46: Shrinkage loss threshold c=0.001 may be suboptimal for log1p-transformed targets — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-46 |
| Tier | 4 |
| Source | manual (2026-04-10), metric-lab autoresearch Finding F4/Round 3 |
| Trigger | When evaluating purple_alien's CRPS against blue_strange/violet_visitor — the difference may be attributable to suboptimal c, not the loss function itself |
| Location | `views-models/models/purple_alien/configs/config_hyperparameters.py` (`loss_reg_c: 0.001`) |

The autoresearch found that Shrinkage loss with `c=1.0` marginally outperforms Basu DPD and NLL on log-space magnitude errors (Finding 6.4). The hydranet production default is `c=0.001`, calibrated for the U-Net's normalized feature space where typical errors are in the 0-1 range. But purple_alien's targets are log1p-transformed, where the natural error scale is 0-7 (log1p of 0-1000 fatalities). A threshold of `c=0.001` in log-space means "suppress errors below 0.1% in magnitude" — virtually no suppression. The autoresearch suggests `c=1.0` ("suppress errors below 2.7x in magnitude") is more appropriate for log-space operation. Note: the `a` parameter (steepness) at 258 in purple_alien is also very different from the autoresearch optimal of 10 — but the LSTM hurdle and U-Net have different residual distributions, so direct transfer is not guaranteed. Empirical testing on HydraNet with `c=1.0, a=10` is recommended before changing the production default.

**Resolution (2026-04-20):** Deferred to views-models. The hydranet library default is `loss_reg_c=0.2` (`config_initializer.py`), which is appropriate for the normalized feature space. The concern about `c=0.001` applies exclusively to `purple_alien`'s model-specific config in the external `views-models` repo. No hydranet library code change needed.

---

### C-72: Misspelled placeholder `requirments.txt` in package root — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-72 |
| Tier | 4 |
| Source | graphify (2026-04-21) |
| Trigger | When setting up dependency management or CI/CD, a tool or contributor may look for `requirements.txt` and miss this file (or find it and get no content) |
| Location | `views_hydranet/requirments.txt` |

Graphify extraction discovered `views_hydranet/requirments.txt` — a misspelled filename containing only the placeholder text "To come...". The file serves no purpose: the project uses conda for dependency management (pinned in the conda environment, confirmed by 53 identical wandb snapshots). The misspelling means automated tools looking for `requirements.txt` will not find it, and any tool that does find it gets no useful content. Either delete the file or rename it to `requirements.txt` and populate it.

**Resolution (2026-04-21):** File deleted. Project uses conda for dependency management; no `requirements.txt` needed.

---

### C-51: Autoregressive drift warning-to-halt gap is 100x — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-51 |
| Tier | 4 |
| Source | repo-assimilation (2026-04-10) |
| Trigger | When model predictions diverge during autoregressive inference, check whether the 100→10,000 gap wastes significant GPU time before halting |
| Location | `hydranet_inference.py:293-298` (warn at 100), `integrity_guardian.py:38` (halt at 10,000) |

During autoregressive inference, `HydraNetInference.predict()` logs a WARNING when `max |pred| > 100.0` (C-20 resolution), but execution continues until `IntegrityGuardian.monitor()` raises `RuntimeError` at the hard ceiling of 10,000. The 100x gap between the soft warning and the hard halt means a diverging model can waste significant GPU time generating predictions that are already astronomically wrong. On PRIO-GRID data in log1p-space, `|pred| > 100` represents `expm1(100) ≈ 2.7×10^43` fatalities — beyond physical reality. In practice the gap is tolerable because: (a) the warning is logged per-step and visible in real-time, (b) the hard halt catches the explosion before NaN/Inf, and (c) reducing the halt threshold would require careful calibration to avoid false positives on legitimate high-magnitude predictions.

See also C-20 (resolved — added the soft warning).

**Resolution (2026-04-20):** Three-tier escalation: WARNING at |pred| > 100, ERROR at |pred| > 500, HALT at |pred| > 1000. `IntegrityGuardian.PREDICTION_MAGNITUDE_CEILING` class constant lowered from 10000 to 1000. Gap reduced from 100× to 10×. ERROR-level log at 500 references the ceiling value so operators know the halt threshold.

---

### C-55: CIC drift after code changes — recurring documentation hygiene gap — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-55 |
| Tier | 4 |
| Source | review-diff (2026-04-10), recurring across PR #23 and PR #24 |
| Trigger | When adding a `field_validator` or changing a documented failure mode in a CIC-governed class, update the CIC's Section 6 (Failure Modes) in the same PR |
| Location | `docs/CICs/HydraNetConfig.md` Section 6 (missing `loss_reg`/`loss_class` validators added in C-05), `docs/CICs/DataFetcher.md` Section 6 (still says `apply_blueprint` skips on missing source after C-50 changed it to raise) |

CICs documenting class contracts have been observed to drift from code in two consecutive PRs: PR #23 (C-05) added `field_validator` for `loss_reg`/`loss_class` to `HydraNetConfig`, but the CIC's Section 6 still lists only `run_type`, `evaluation_mode`, and `aggregate_method` as validated enums. PR #24 (C-50) changed `DataFetcher.apply_blueprint()` from "skip on missing source" to "raise on missing source", but the CIC still describes the old behavior. Both gaps were flagged at suggestion-severity in review-diff and not blocked, but the recurring pattern indicates a process gap.

This is a maintainability concern, not a correctness risk: the code is the source of truth, the CIC lags. The risk is that the CIC becomes a misleading document — readers (especially silicon-based contributors per ADR-007) may trust the CIC over the code, leading to wrong assumptions about failure modes. The fix is procedural: add a CIC update to the standard PR checklist when touching a CIC-governed class.

C-28 (resolved 2026-04-08) addressed a one-time stale CIC update. C-55 captures the recurring pattern that motivates a process change rather than a one-time fix.

See also C-28 (resolved — one-time CIC test references update).

**Resolution (2026-04-20):** Fixed both specific drifts: added `loss_reg`, `loss_class`, `aggregate_method` validators to HydraNetConfig.md Section 6; added blueprint-source-missing raise to DataFetcher.md Section 6. Added `tests/test_cic_drift_detection.py` (4 tests) to detect future Section 6 drift for these CICs.

---

### C-61: Eval/forecast paths never lock entropy — non-reproducible outputs — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-61 |
| Resolved | 2026-04-19 |
| Resolution | Added `ReproducibilityGate.lock_entropy()` call in `_setup_evaluation()` (shared by eval and forecast). Added `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, and `CUBLAS_WORKSPACE_CONFIG` to `lock_entropy()`. Falsification stubs F2-01a/b/c flipped GREEN. **Residual test gap:** No pipeline-level reproducibility comparison test exists (F3-06 — tracked but not registered separately as it's a test completeness concern, not a code defect). |

---

### C-62: IntegrityGuardian absent from inference path — unguarded predictions — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-62 |
| Resolved | 2026-04-19 |
| Resolution | Added `IntegrityGuardian.monitor_numpy()` static method. Wired into `InferenceOrchestrator._run_inference_pipeline()` after Step 1 PREDICT. `predict()` now raises `RuntimeError` instead of returning NaN arrays on model explosion. Falsification stubs F2-02a, F2-05a, F2-05b flipped GREEN. |

---

### C-63: No CI/CD pipeline — zero automated quality gates — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-63 |
| Resolved | 2026-04-19 |
| Resolution | Created `.github/workflows/ci.yml` with lint (ruff check + format) and test (pytest) jobs, triggered on push to main/development and pull requests. Falsification stub F2-04 flipped GREEN. |

---

### C-64: Silent NaN propagation — model explosion becomes invisible in output — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-64 |
| Resolved | 2026-04-19 |
| Resolution | `predict()` in `hydranet_inference.py` now raises `RuntimeError` instead of returning `np.full(..., np.nan)`. `IntegrityGuardian.monitor_numpy()` added as a second guard in the orchestrator. Both changes enforce ADR-003 Fail Loud. Falsification stubs F2-05a/b flipped GREEN. |

---

### C-65: Unvalidated config fields used at runtime — `sweep`, `random_flips` — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-65 |
| Resolved | 2026-04-19 |
| Resolution | Added `sweep: bool = Field(default=False)`, `random_flips: bool = Field(default=True)`, and `diagnostic_visualizations: bool = Field(default=False)` to `HydraNetConfig` schema in `config_initializer.py`. Pydantic now validates type — `sweep="true"` raises `ValidationError`. Falsification stub F2-06 flipped GREEN. |

---

### C-66: Validation partition has zero manager-level integration tests — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-66 |
| Tier | 2 |
| Source | falsify-F3-01 (2026-04-19) |
| Trigger | A partition boundary bug in validation-specific window calculation ships undetected — validation evaluation produces wrong rolling-origin windows |
| Location | `tests/test_pipeline_integration.py`, `tests/test_manager_integration_local.py`, `tests/test_audit_manager_eval_survival.py` (all use `run_type="calibration"` exclusively) |

All manager-level integration tests hardcode `run_type="calibration"`. The validation partition — with its own `_partition_dict` boundaries, different `test_start`/`test_end` values, and distinct rolling-origin window calculations — is exercised only in production. The single validation-adjacent test (`test_eval_integration_toy.py`) tests the external `views_evaluation` package contract, not the HydraNet manager pipeline.

Cross-refs: C-69 (`_partition_dict` mechanism untested), C-29 (misconfiguration scenarios).

**Resolution (2026-04-19):** `tests/test_lifecycle_integration.py` — `TestBeige::test_validation_partition_produces_predictions` exercises `run_type='validation'` with `_partition_dict` through the full manager pipeline. `TestBeige::test_calibration_and_validation_use_different_origins` verifies that different partition boundaries produce different rolling-origin counts. F3-01 stub flipped GREEN.

---

### C-67: Primary E2E test has zero numeric value assertions on predictions — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-67 |
| Tier | 3 |
| Source | falsify-F3-02 (2026-04-19) |
| Trigger | A bug in the inference pipeline produces all-zero or all-constant predictions — `test_pipeline_integration.py` passes because it only checks dict keys and shapes |
| Location | `tests/test_pipeline_integration.py:188-203` (eval assertions), `tests/test_pipeline_integration.py:276-286` (forecast assertions) |

`test_pipeline_integration.py` is the primary end-to-end test for the manager lifecycle. It contains 17 assertions: 2 type checks, 8 key-presence checks, 2 shape checks, 2 size checks, and 2 dict-length checks. Zero assertions verify that computed prediction values are numerically correct. Only `test_audit_manager_eval_survival.py` uses `np.testing.assert_allclose()` on predictions. A model producing garbage values would pass all other integration tests.

Cross-refs: F3-05 (identity value assertions also weak).

**Resolution (2026-04-20):** Added `np.isfinite(pf.y_pred).all()` assertions to both eval and forecast paths in `test_pipeline_integration.py`. F3-02 stub flipped GREEN.

---

### C-68: No cradle-to-grave lifecycle test — train→infer chain untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-68 |
| Tier | 2 |
| Source | falsify-F3-03 (2026-04-19) |
| Trigger | A training code change produces a model that serializes correctly but generates wrong predictions at inference — no test catches this because training and inference are tested in isolation |
| Location | `tests/` (no file connects training to inference) |

No single test file imports both training functions (`make`, `training_loop`, `train_model_artifact`) and inference functions (`InferenceOrchestrator`, `generate_prediction_frames`, `_evaluate_model_artifact`) as real implementations. The lifecycle is tested in disconnected fragments: `test_training_engine.py` trains but never infers; `test_manager_integration_local.py` infers but mocks the model. The handoff — "does a model trained on data X produce correct predictions when evaluated?" — has zero coverage.

Cross-refs: C-18 (resolved — training smoke test only covers the training fragment).

**Resolution (2026-04-19):** `tests/test_lifecycle_integration.py` — `TestGreen` class trains a TinyModel via `training_loop()` then evaluates and forecasts via `manager._evaluate_model_artifact()` and `manager._forecast_model_artifact()`. Three tests verify finite non-zero predictions and correct identity column values. F3-03 stub flipped GREEN.

---

### C-69: Partition boundary mechanism (`_partition_dict`) entirely untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-69 |
| Tier | 2 |
| Source | falsify-F3-04 (2026-04-19) |
| Trigger | `_partition_dict` returns wrong boundaries for a run_type — evaluation uses incorrect time window, producing evaluation against wrong data |
| Location | `manager/hydranet_manager.py:286-298` (`_partition_dict` lookup and origin calculation) |

`_setup_evaluation()` at line 286 reads `_partition_dict[run_type]` to determine `test_start`, `test_end`, and rolling-origin windows. No test ever sets `_partition_dict` on a manager instance. `test_data_pipeline_extraction.py` tests `partition_bound=5` directly, bypassing the `_partition_dict` lookup entirely. The mechanism that distinguishes calibration from validation data slicing is exercised only in production via `views_pipeline_core`.

Cross-refs: C-66 (validation partition untested).

**Resolution (2026-04-19):** `tests/test_lifecycle_integration.py` — `TestBeige` sets `_partition_dict` with asymmetric calibration/validation boundaries and verifies different origin counts. `TestRed` tests fallback paths: missing partition key → single origin, no `_partition_dict` attribute → single origin. F3-04 stub flipped GREEN.

---

### C-70: 33 lint errors and 61 format violations — CI lint job will fail on push — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-70 |
| Tier | 2 |
| Source | falsify-F4-02 (2026-04-20) |
| Trigger | Pushing to main or development — CI lint job runs `ruff check .` and `ruff format --check .`, both fail |
| Location | 9 files: `tests/test_falsification_cradle_to_grave.py`, `tests/test_falsification_deployment_readiness.py`, `tests/test_falsification_end_to_end_claim.py`, `tests/test_manager_integration_local.py`, `tests/test_temporal_causality_audit.py`, `tests/test_training_engine.py`, `tests/test_volume_handler_hard_gates.py`, `views_hydranet/utils/data_sniffer.py`, `views_hydranet/utils/hydranet_inference.py` |

Breakdown: 12 unsorted imports (I001), 8 unused imports (F401), 2 unused variables (F841), 2 ambiguous variable names (E741), 2 f-strings without placeholders (F541), 4 lines too long (E501). Additionally, 61 of 94 files fail `ruff format --check`. The CI pipeline created in C-63 resolution enforces these checks — any push in the current state will fail the lint job.

**Resolution (2026-04-20):** `ruff check --fix .` auto-fixed 23 errors; remaining 10 fixed manually (E741 `l`→`ln`, E501 line breaks, F841 unused variables). `ruff format .` reformatted 62 files. Both `ruff check .` and `ruff format --check .` now pass cleanly.

Cross-refs: C-63 (resolved — CI pipeline created).

---

### C-71: Risk register header counts do not match actual entries — C-34 missing — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-71 |
| Tier | 3 |
| Source | falsify-F4-04 (2026-04-20) |
| Trigger | Next register review or audit relying on header counts for governance reporting |
| Location | `reports/technical_risk_register.md` header (lines 8-10), entry sequence (C-34 gap) |

Register header claims 69 total / 19 open / 50 resolved. Actual entry count: 68 entries (C-34 is missing from the sequence entirely), 18 open, 50 resolved. Two double `---` separators (lines 91/93 and 369/371) indicate structural artifacts from deleted or moved entries. The register's own maintenance rules (line 914-917) state header counts are manually maintained — they have drifted.

**Resolution (2026-04-20):** Double separators removed. C-34 gap is expected per register rules (gaps indicate merged entries). Header updated to 70 total / 18 open / 52 resolved matching actual counts after C-70 and C-71 resolution.

---

### C-59: HydranetManager has zero failure-mode tests — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-59 |
| Resolved | 2026-04-19 |
| Resolution | Created `tests/test_manager_integration_local.py` with 6 tests (TestGreen: 2, TestBeige: 2, TestRed: 2) using real VolumeHandler, FeatureScaler, DataSniffer, and InferenceOrchestrator with TinyModel. Tests cover eval/forecast lifecycle, stochastic/point mode interaction, config checksum violation, and component failure propagation. |

---

### C-60: 84% of test files lack explicit ADR-005 taxonomy markers — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-60 |
| Resolved | 2026-04-19 |
| Resolution | Added TestGreen/TestBeige/TestRed taxonomy classes to 10 test files. Coverage increased from 9/55 (16%) to 20/55 (36%) files with 51 total taxonomy markers. Key files restructured: `test_volume_handler_hard_gates.py`, `test_temporal_causality_audit.py`, `test_training_engine.py`, `test_prediction_frame_assembler.py`, `test_model_artifact_fetcher.py`, `test_pipeline_integration.py`, and others. |

---

### C-56: `artifact_name` parameter silently ignored in `_setup_evaluation` — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-56 |
| Resolved | 2026-04-19 |
| Resolution | Passed `model_artifact_name=artifact_name` to `fetch_model_artifact()` in `_setup_evaluation`. Verified by `tests/test_falsification_end_to_end_claim.py::TestArtifactNameSilentlyIgnored`. |

---

### C-57: Partial projection branch contains latent `slice_time` overflow — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-57 |
| Resolved | 2026-04-19 |
| Resolution | Replaced buggy `slice_time` call with explicit `NotImplementedError` explaining partial projection is unsupported. Branch remains unreachable in normal flow. Verified by `tests/test_falsification_end_to_end_claim.py::TestPartialProjectionSliceOverflow`. |

---

### C-58: Forecast sniffer validation (`is_forecast=True`) is dead code — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-58 |
| Resolved | 2026-04-19 |
| Resolution | Added `forecast: bool` parameter to `_run_data_pipeline`, wired through to `sniff_forecast_alignment(is_forecast=forecast)`. Forecast path now passes `forecast=True`. Verified by `tests/test_falsification_end_to_end_claim.py::TestForecastSnifferNeverCalled`. |

---

### C-04: Spatial offset arithmetic in VolumeSampler is untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-04 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_window_offset_preserves_geographic_truth` in `tests/test_volume_sampler.py`. Test plants a sentinel value at known coordinates, extracts a window, and verifies geographic round-trip via `spatial_offset`. |

---

### C-24: InferenceOrchestrator temporal discontinuity failure mode untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-24 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_gate_slice_time_beyond_bounds_raises` and `test_gate_slice_time_origin_plus_duration_oob` in `tests/test_volume_handler_hard_gates.py`. Tests verify `slice_time()` raises `ValueError` on out-of-bounds origins. |

---

### C-25: Curriculum→Sampler zero-qualified-cells interaction untested — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-25 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_curriculum_high_threshold_triggers_fallback` in `tests/test_volume_sampler.py`. Test verifies CurriculumLearner + VolumeSampler interaction: extreme threshold yields `qualified=0` with valid batch via random fallback. |

---

### C-27: `train_model.py` import structure blocks local testing — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-27 |
| Resolved | 2026-04-08 |
| Resolution | Moved `from views_pipeline_core.managers.model import ModelPathManager` to `TYPE_CHECKING` guard with `from __future__ import annotations`. Import only runs during static analysis, not at runtime. |

---

### C-31: ADR-008 log-before-raise violations in 4 files — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-31 |
| Resolved | 2026-04-08 |
| Resolution | Applied ADR-008 "Narrative Failure" pattern (err_msg → logger.error → raise) to all 7 locations: `training_forensics.py` (3), `config_initializer.py` (1), `volume_handler.py` (2), `mtloss.py` (1). Also added `logger` to `mtloss.py` which previously had none. |

---

### C-02: Duplicated setup between eval and forecast — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-02 |
| Resolved | 2026-04-08 |
| Resolution | Added `forecast=True` parameter to `_setup_evaluation()`. `_forecast_model_artifact()` reduced from 25 lines of duplicated setup to a single `_setup_evaluation("forecasting", forecast=True)` call. Forecast path correctly skips partition lookup and uses `partition_bound=None`. |

---

### C-23: `extrapolate_time()` has no direct unit test — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-23 |
| Resolved | 2026-04-08 |
| Resolution | Added 4 tests in `tests/test_volume_handler_hard_gates.py`: shape preservation, temporal continuity (time channel increment verification), non-time channel cloning, and single-step edge case. |

---

### C-07: Training loop lacks explicit per-window memory cleanup — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-07 |
| Resolved | 2026-04-09 |
| Resolution | Added `del sample_handler, losses, w_loss` after `backward()` in the inner window loop of `training_loop()` in `training_engine.py`. Matches the per-origin cleanup pattern already used in the inference path. |

---

### C-15: `training_loop()` has 4+ responsibilities — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-15 |
| Resolved | 2026-04-09 |
| Resolution | Split `train_model.py` into `training_engine.py` (Entity layer, pure training logic) and `train_model.py` (Framework wiring, 38 lines). The file-level SRP violation is eliminated — `training_engine.py` serves the data scientist, `train_model.py` serves the platform. Function-level diagnostic mixing in `training_loop()` remains but is now contained in a single-responsibility module. |

---

### C-21: Bare `except Exception` swallows errors in inference and diagnostics — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-21 |
| Resolved | 2026-04-09 |
| Resolution | All 11 locations now comply with ADR-008. The 2 core-logic locations (`hydranet_inference.py:391`, `training_engine.py:224`) upgraded from silent `pass` to `logger.error(..., exc_info=True)` per Fail-Safe constraint. The 9 `visual_diagnostics.py` locations already logged as `logger.error`. All catch-all patterns are ADR-008 Section 4 compliant (Observability Actors permitted Fail-Safe). |

---

### C-18: No end-to-end training smoke test — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-18 |
| Resolved | 2026-04-08 |
| Resolution | Added `test_training_smoke_end_to_end` in `tests/test_training_engine.py`. Runs full `training_loop` on 8x8 synthetic data (2 lessons, 1 window each). Verifies: completes without error, returns expected keys, records loss history, model parameters change from initialization. |

---

### C-32: VolumeSampler CIC failure modes untested and unregistered — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-32 |
| Resolved | 2026-04-08 |
| Resolution | Ledger Inconsistency already tested by existing `test_red_unknown_target`. Added `TestGeometricOverflow` class (2 tests) verifying bounds clamping with edge anchors and max-dim extraction. CIC Section 6 notes: code uses `np.clip` (silent correction) rather than raising — correct behavior, CIC language ("Fails if...") is aspirational rather than literal. |

---

### C-42: No reproducibility gate — seeds partially set, no manifest audit — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-42 |
| Resolved | 2026-04-10 |
| Resolution | Added `ReproducibilityGate.lock_entropy(np_seed, torch_seed)` in `views_hydranet/infrastructure/reproducibility_gate.py`. Locks all 4 RNG sources: Python random, NumPy, PyTorch CPU, PyTorch CUDA. `training_engine.py` now calls `lock_entropy()` instead of manual `np.random.seed()` + `torch.manual_seed()`. 7 TDD tests verify determinism. |

---

### C-43: No reproducibility gate — no parameter genome audit — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-43 |
| Resolved | 2026-04-10 |
| Resolution | Added `ReproducibilityGate.audit_manifest(config)` that validates config completeness before training. Checks 16 core genome parameters (presence + non-None), validates loss_reg/loss_class against `LOSS_REG_REGISTRY`/`LOSS_CLASS_REGISTRY`, and validates loss-specific params from registry `"params"` lists. Raises `ValueError` with clear message on missing parameters. 7 TDD tests. |

---

### C-17: `train()` function has 13 parameters — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-17 |
| Resolved | 2026-04-10 |
| Resolution | Added `TrainingContext` dataclass that bundles the 10 "wired once" components (model, optimizer, scheduler, 3 loss components, config, device, viz, forensics). `train()` reduced from 13 params to 4: `ctx`, `sample_handler`, `pbar`, `stage_label`. Created once in `training_loop()`, passed to every `train()` call. |

---

### C-20: Autoregressive inference has no soft magnitude guard — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-20 |
| Resolved | 2026-04-10 |
| Resolution | Added soft magnitude WARNING in `hydranet_inference.py` autoregressive loop: logs warning when `max |pred| > 100.0` at any step. Does not clip — warns only, allowing operators to detect gradual drift before it reaches the hard NaN/Inf ceiling. |

---

### C-45: Regression heads receive gradients from zero-valued pixels — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-45 |
| Resolved | 2026-04-10 |
| Resolution | Added `hurdle_threshold` config key to `_process_sequence()`. When set (e.g., 0.0), regression loss is computed only on `target > threshold` pixels. Backward compatible: `None` = all pixels (v1 behavior). 3 TDD tests (all-zeros, mixed data, None bypass). |

---

### C-47: Pareto Loss not available in loss registry — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-47 |
| Resolved | 2026-04-10 |
| Resolution | Implemented `ParetoLoss` (6 lines, Kozerawski et al. 2022). Registered as `loss_reg='pareto'` with `loss_reg_pareto_alpha` config. 5 TDD tests. |

---

### C-48: No QS99 tail regularizer — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-48 |
| Resolved | 2026-04-10 |
| Resolution | Distribution-free asymmetric pinball loss on `mu` (ML expert Suggestion 3 — no sigma, no distributional assumption). Config keys: `qs99_weight` (None=disabled), `qs99_tau` (None=must be explicit when active). Only active when `hurdle_threshold` is not None and `qs99_weight > 0`. Strict conditional validation added by ADR-050 parameter hardening: `qs99_tau` is required when `hurdle_threshold` is set and `qs99_weight > 0` — no silent defaults. Added to `_process_sequence()` after MultiTaskLoss. 10 TDD tests (2 original + 8 hurdle-Basu integration). Addresses Lerch et al. (2017) Forecaster's Dilemma. |

---

### C-44: Classification head bias initializes to 50% event probability — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-44 |
| Resolved | 2026-04-10 |
| Resolution | Added `_init_classification_head_bias(model, bias_value)` in `training_engine.py`. Targets `dec_conv4_head{N}_class` layers via `named_modules()` after weight init. New config parameter `onset_bias_init` (None = PyTorch default, -5.0 = 0.67% prior). Called from `make()`. 5 TDD tests including training smoke test. All 3 views-models configs updated with `onset_bias_init: -5.0`. Based on metric-lab autoresearch Finding F1 (98.5% metric improvement). |

---

### C-38: Factory functions closed to extension (OCP violation) — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-38 |
| Resolved | 2026-04-10 |
| Resolution | Replaced `if/elif/else` chain in `choose_loss()` with `LOSS_REG_REGISTRY` and `LOSS_CLASS_REGISTRY` dicts. Adding a new loss requires only adding a registry entry. Opaque letter codes (`a`, `b`, `c`, `d`) replaced with readable names (`mse`, `shrinkage`, `basu_dpd`, `lognormal_nll`). Model and scheduler factories remain as if/elif (1-2 options each, low pressure). |

---

### C-28: CIC test file references are stale — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-28 |
| Resolved | 2026-04-08 |
| Resolution | Updated test alignment sections in HydranetManager.md, HydraNetConfig.md, and ConfigInitializer.md to reference actual test files (test_config_typed.py, test_config_validation.py, test_manager_memory_hygiene.py, etc.) |

---

### C-39: VolumeHandler Entity imports Framework type — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-39 |
| Resolved | 2026-04-11 |
| Resolution | D-01 partial split executed. Extracted PredictionFrame output path from VolumeHandler into a dedicated `PredictionFrameAssembler` class in `views_hydranet/utils/prediction_frame_assembler.py`. The lazy `from views_pipeline_core.data.prediction_frame import PredictionFrame` import that was inside `VolumeHandler.to_evaluation_pf()` now lives only inside `PredictionFrameAssembler.assemble_evaluation()`. VolumeHandler is fully decoupled from the Framework layer — `grep "views_pipeline_core" volume_handler.py` returns zero matches. The Dependency Rule violation is eliminated. C-36 and C-37 are partially addressed by the same refactor (narratives updated). |

---

### C-22: `cast(Any, model)` at 7+ call sites bypasses type safety — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-22 |
| Resolved | 2026-04-10 |
| Resolution | Defined `ModelOutput` NamedTuple in `architectures/HydraBNrecurrentUnet_06_LSTM4.py` with `reg`, `cls`, `h_next` fields. `forward()` now returns `ModelOutput(...)`; NamedTuple supports tuple unpacking so legacy `r, c, h = model(x, h)` continues to work. Removed `cast(Any, model)` wrappers in `training_engine.py` and `hydranet_inference.py` — consumers now use named access (`output.reg`, `output.cls`, `output.h_next`) or simple unpacking. Updated 5 test mocks (conftest.py TinyModel, test_temporal_causality_audit, test_inference_logic, test_inference_memory_hygiene, test_cluster_e, test_optimization_gate) to return ModelOutput. The remaining `cast(Any, multitaskloss_instance)` is unrelated to C-22. |

---

### C-40: `validate_docs.sh` uses GNU-only `grep -oP` — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-40 |
| Resolved | 2026-04-10 |
| Resolution | Replaced both `grep -oP` calls in `docs/validate_docs.sh` with portable `sed -nE` regex extraction. Lines 60 (ADR number extraction) and 75 (protocol path extraction) now work on macOS BSD grep. Script verified — `bash docs/validate_docs.sh` passes cleanly. Should be upstreamed to base_docs template. |

---

### C-50: Derivation asymmetry — DataFrame path skips, Volume path raises — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-50 |
| Resolved | 2026-04-10 |
| Resolution | Made `DataFetcher.apply_blueprint()` raise `ValueError` (matching `VolumeHandler._execute_derivations()`) when a derivation source column is missing from the DataFrame. Updated cross-reference comments in both files. Updated `test_data_fetcher.py`: removed `test_beige_blueprint_missing_source_skips`, added `test_red_blueprint_missing_source_raises`. The two paths now have symmetric behavior verified by `test_derivation_parity.py` and the new red test. |

---

### C-53: Weight head implementation guide has unresolved technical issues — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-53 |
| Resolved | 2026-04-10 |
| Resolution | Added Section 0 (Critical Review) to `views-metric-lab/reports/experiments/hydranet_weight_head_implementation_guide.md` documenting all 5 technical issues with corrections: (1) use `reduction='none'` with `LOSS_REG_REGISTRY`, don't bypass; (2) define `ModelOutput` NamedTuple with optional weights field, don't use conditional return type; (3) use per-pixel entropy or variance penalty, not `H(mean(w))`; (4) sequence proxy validation → ConvLSTM proxy → lightweight head → single-target pilot before full rollout; (5) defer heteroscedastic sigma until fuzzy CRPS evaluates it. Recommended sequencing block added at end. The guide is now safe to follow with the corrections in Section 0. |

---

### C-54: onset_bias_init default -5.0 suboptimal — dilution study supports -7.0 — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-54 |
| Resolved | 2026-04-10 |
| Resolution | Updated `onset_bias_init` from `-5.0` to `-7.0` in all three views-models configs (blue_strange, violet_visitor, purple_alien). Comment updated to reference dilution study. One-line change per file, no code changes required. |

---

### C-10: 12 test files require views_pipeline_core — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-10 |
| Resolved | 2026-04-10 |
| Resolution | Added `pytest.importorskip("views_pipeline_core")` to 10 test files, `pytest.importorskip("views_evaluation")` to 1 file, and `pytest.importorskip("polars")` to 1 file. Collection errors replaced with clean skip markers. In partial environments: 349 tests collect, 12 skip, 0 errors. In full environments: 414 tests collect, 0 skip. Conftest minimum-test gate (280) continues to function correctly. |

---

### C-05: Loss/scheduler selection uses unvalidated string codes — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-05 |
| Resolved | 2026-04-10 |
| Resolution | Added `field_validator` for `loss_reg` and `loss_class` in `HydraNetConfig`. Validators lazy-import `LOSS_REG_REGISTRY` / `LOSS_CLASS_REGISTRY` from `utils.py` and reject unregistered values at config construction time. Typos now fail at Pydantic validation, not at `choose_loss()` runtime. |

---

### C-12: `wandb` imported unconditionally at module level — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-12 |
| Resolved | 2026-04-10 |
| Resolution | Moved `import wandb` from module-level in `utils.py` to inside `train_log()` function body. `wandb` is now only imported when the function is actually called, and only used when `wandb.run is not None`. Environments without `wandb` installed can now import `utils.py` without error. |

---

### C-52: 15 tests broken by recent refactors — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-52 |
| Resolved | 2026-04-10 |
| Resolution | Updated `train()` call signatures to use `TrainingContext` (C-17 API), fixed mock paths from `train_model` → `training_engine` (C-15 split), updated loss codes `'a'`→`'mse'`, `'b'`→`'focal'` (C-38 rename), deferred `training_loop` import to avoid `sys.modules` interaction. All 7 training/optimization tests restored. Streaming tests passed once `views_pipeline_core` was available. 6 beige config tests added. `TinyModel` extracted to conftest.py. Result: 412 passed, 2 failed (1 intentional RED + 1 pre-existing). |

---

### C-74: DataFetcher.fetch_df() hardcodes viewser filename — breaks datafactory consumers — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-74 |
| Resolved | 2026-04-27 |
| Resolution | `fetch_df()` now accepts an optional `cached_path` parameter. When provided, it loads from that exact path instead of constructing the hardcoded `{run_type}_viewser_df` filename. The single call site in `HydranetManager._run_data_pipeline()` passes `cached_path=self._get_cached_data_path()`, a framework method that returns the source-aware path (viewser or datafactory). Fallback preserved for backward compatibility. Two new tests added (`test_green_fetch_df_cached_path`, `test_beige_cached_path_ignores_run_type`). CIC Section 8 updated with cached_path usage example. |

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps in numbering indicate merged or resolved entries
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `tech-debt-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** `Total Concerns` and `Open Concerns` in the register header are manually maintained — update them whenever a concern is added or resolved
- **Governed by:** ADR-048
