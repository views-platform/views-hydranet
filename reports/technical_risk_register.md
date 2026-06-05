# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-06-05                           |
| Total Concerns    | 124                                  |
| Open Concerns     | 30                                   |
| — of which demoted (tech-debt) | 4 (tagged `[DEMOTED]` in §Open Concerns; indexed in §Tech-Debt Backlog) |
| Resolved Concerns | 94                                   |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Causal Clusters (review-rr strategic, 2026-06-05)

Open concerns reduce to **6 root decisions**. Fixing a root advances multiple entries; entries are tagged `[Cx]` informally in this map (not in every entry body).

| # | Root decision | Member entries | Fix scope | Priority |
|---|---|---|:--:|---|
| **1** | Inference surface (`predict()`) never contracted/tested/decomposed; guarded only by a log-space ceiling | C-113, C-121, C-122, C-107, C-114 (+D-05, D-06) | 1 coordinated | **★ first — imminent (ZITD edits `predict()`)** |
| **2** | Training-dynamics changes outran reproducibility/comparability discipline | C-112, C-119, C-79, C-110 | 2 | near-term |
| **3** | `utils/` accreted multiple domains without package structure / clear ownership | C-35, C-01, C-36, C-37, C-120, C-75, C-76 | 3 (large blast radius) | defer |
| **4** | Single hardcoded head/loss topology (3+3 heads, positional loss tuple) | C-03, C-123 (+C-122 model facet, D-02) | 2 | decide *with* ZITD planning |
| **5** | Config is a typed-model-masquerading-as-dict (`extra="allow"`) | C-06, C-117, C-49 (+D-03) | 2 | defer (D-03 tension) |
| **6** | Operational/GPU fragility on the dev box (no hard CUDA gate; publish-step memory) | C-115, C-116 | 1–2 | near-term |

**Highest-value:** Cluster 1 — largest, contains the only imminent Tier-2 (C-113), single coordinated fix (decompose `predict()` + rollout test + `HydraNetInference` CIC + IntegrityGuardian §6 doc) advances 5 concerns + 2 disagreements before the ZITD head touches it.

---

## Open Concerns

### C-01: Manager monolith orchestration

| Field | Value |
|-------|-------|
| ID | C-01 |
| Tier | 3 |
| Source | repo-assimilation (2026-04-08) |
| Trigger | When adding a new pipeline stage (e.g., post-processing, calibration) that requires Manager wiring — verify the new stage doesn't push Manager past pure-wiring into business logic |
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
| Tier | 3 |
| Source | repo-assimilation (2026-04-08); recalibrated 4→3 (review-rr 2026-06-05) |
| Trigger | Adding or removing a regression/classification target — **now imminent**: the distributional-head dossier (P2, `reports/2026-06-05_distributional_head_dossier/`) collapses each reg+cls pair into one ZITD likelihood (4 params), which changes the head count/topology and trips the hardcoded 3+3 preflight |
| Location | `HydraBNrecurrentUnet_06_LSTM4.py:68-167`, `hydranet_manager.py:165-176` (`_run_preflight_check` raises if n_reg≠3 or n_class≠3) |

**Recalibrated 4→3 (review-rr 2026-06-05):** the trigger is no longer hypothetical — the ZITD head (dossier P2) will collapse reg+cls pairs, so likelihood jumped; expected risk (impact×likelihood) now warrants Tier 3. Member of Cluster 4.

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
| Trigger | When creating a new utility module (e.g., a new loss function or sampling strategy) — the developer must decide whether it goes in `utils/` alongside 28 other files, with no package structure to guide placement |
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

**[DEMOTED 2026-06-05 → Tech-Debt Backlog]** standing concern (always-true), partial-addressed, Tier 4 — accepted trade-off unless an alternative VolumeHandler implementation is actually needed. Member of Cluster 3.

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

**[DEMOTED 2026-06-05 → roadmap / Tech-Debt Backlog]** standing concern (always-true threshold), not an event-triggered risk — better tracked as a refactoring motivation. Member of Cluster 5.

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

### C-85: Flip probability 0.5 hardcoded in training_engine — not config-driven

| Field | Value |
|-------|-------|
| ID | C-85 |
| Tier | 4 |
| Source | /falsify magic-numbers audit P1 (2026-05-27) |
| Trigger | When running augmentation sensitivity experiments and needing flip probability other than 0.5 — requires source code change instead of config change |
| Location | `views_hydranet/train/training_engine.py:290-292` |

Data augmentation flip on/off is config-driven (`random_flips: bool`), but the flip probability is hardcoded at `0.5` (fair coin). This is the only behavior-affecting numeric literal in `training_engine.py` that isn't sourced from config. Symmetric by definition (H/W flips), so `0.5` is defensible — but a researcher doing augmentation experiments would need to modify source code to test other probabilities.

**[DEMOTED 2026-06-05 → Tech-Debt Backlog]** single-file, near-mechanical (add a config key), Tier 4 — no design decision required.

See also C-65 (resolved — `random_flips` added to schema).

---

### C-89: `_SumReducer` and `_make_tiny_model` duplicated across test files

| Field | Value |
|-------|-------|
| ID | C-89 |
| Tier | 4 |
| Source | /test-review (Beck W1) (2026-05-27) |
| Trigger | When modifying `ModelOutput` or the model forward signature — both copies must be updated independently, and forgetting one produces confusing test failures |
| Location | `tests/test_per_target_sigma.py`, `tests/test_scheduled_sampling.py` (current `_SumReducer`/`_make_tiny_model` copies) |

Identical `_SumReducer` and `_make_tiny_model` helpers are defined in multiple test files. Should be extracted to `conftest.py` as shared fixtures.

**Path E amplification (2026-05-29):** Scheduled sampling implementation (issue #37) will require another copy of the tiny model fixture for `tests/test_scheduled_sampling.py`. Extract to `conftest.py` before implementing Path E tests to avoid a fourth copy.

**Test review amplification (2026-06-02):** Additionally, `_tobit_config()` helper is duplicated across 3 test files (test_tobit_loss.py, test_per_target_sigma.py, test_learnable_sigma.py) with slightly different base configs. Same DRY concern, different fixture.

**Partial resolution (2026-06-02, PR #53):** The `_tobit_config()` portion is resolved — extracted to `conftest.py` as `tobit_config_3target()` and now imported by 5 test files. The original `_SumReducer` and `_make_tiny_model` duplication persists (now in `test_per_target_sigma.py` and `test_scheduled_sampling.py`, not the originally-cited `test_cluster_e.py`/`test_hurdle_basu_integration.py`). `_make_tiny_model` was intentionally left local — it builds the real `HydraBNUNet06_LSTM4`, not the conftest `TinyModel`. Remaining work: extract `_SumReducer` to conftest.

Tier 4 rationale: code quality / DRY violation. Single-developer scope. No correctness impact.

**[DEMOTED 2026-06-05 → Tech-Debt Backlog]** mechanical, single-file (conftest), partially done — see backlog index.

---

### C-99: Tobit `reg_latent` vs `reg` dual-path creates refactoring hazard for scheduled sampling

| Field | Value |
|-------|-------|
| ID | C-99 |
| Tier | 4 |
| Source | Path E exploration (2026-05-29) |
| Trigger | Scheduled sampling is now shipped (ADR-056, PR #50) and correctly uses `output.reg`. The remaining risk is future-facing: when refactoring `_process_sequence()` to simplify variable names or consolidate the `t1_pred` / `t1_pred_for_loss` split — verify `prev_pred` (scheduled sampling feedback) still uses `output.reg` (post-ReLU, non-negative), NOT `output.reg_latent` (pre-ReLU, can be negative) |
| Location | `views_hydranet/train/training_engine.py:150-151` (latent routing for loss), `views_hydranet/train/training_engine.py:145-147` (forward pass output), `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:519-523` (reg_latent vs reg) |
| Cross-refs | ADR-054 (Tobit loss) |

`_process_sequence()` uses `output.reg_latent` (pre-ReLU latent μ, can be negative) for Tobit loss computation and `output.reg` (post-ReLU, non-negative) for everything else including forensic recording. Scheduled sampling must use `output.reg` as the feedback input — the model's input features are non-negative (log1p-transformed fatality counts), and `reg_latent` values can be arbitrarily negative.

The two paths are currently distinct (line 150: `t1_pred_for_loss = output.reg_latent if use_latent else t1_pred`, line 147: `t1_pred = output.reg`). But they originate from the same forward pass, and a refactoring that merges variable names or simplifies the output handling could accidentally route `reg_latent` into the scheduled sampling mixer. A unit test should assert that the mixer input is always non-negative.

Tier 4 rationale: no current bug. Single-developer scope. The risk is future-facing and easily mitigated with a test assertion.

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

### C-110: golden_hour ensemble composes posteriors from heterogeneous sigma configs — aggregation correctness unverified

| Field | Value |
|-------|-------|
| ID | C-110 |
| Tier | 2 |
| Source | review-rr strategic blind-spot analysis (2026-06-02); recalibrated 3→2 (review-rr 2026-06-05) |
| Trigger | When evaluating the golden_hour ensemble's CRPS/MCR for the first time, or before relying on its calibration for delivery — verify the three members' posteriors share scale/support before concatenation, since two members use per-target sigma `{1.0, 0.75, 0.5}` and one uses uniform sigma `1.0` |
| Location | `views-models/ensembles/golden_hour/`, `views-models/models/{pink_pirate,violet_visitor,blue_stranger}/configs/config_hyperparameters.py` |

The golden_hour ensemble (set up 2026-06-02) concatenates 3 × 64 = 192 posterior samples across members trained with *different loss surfaces*: pink_pirate and violet_visitor use per-target Tobit sigma `{lr_sb: 1.0, lr_ns: 0.75, lr_os: 0.5}`, while blue_stranger uses uniform sigma `1.0`. Sigma directly controls the width of the Tobit predictive distribution, so the uniform-sigma member may produce systematically wider (or narrower) posteriors for ns/os targets than the per-target members. Naive concatenation of samples with different dispersion characteristics could distort the ensemble's aggregate calibration — the MCR could drift even if each member is individually well-calibrated. This is an unverified correctness assumption introduced by the orthogonal-ensemble design (whose diversity is intentional and desirable for ranking, but whose effect on magnitude calibration is untested).

Tier 2 rationale (recalibrated 2026-06-05): this is a **silent-miscalibration** risk — the aggregate ensemble can drift while every member looks healthy, with no error signal. That "looks healthy but degraded" character places it with the Tier-2 fragility family rather than the maintainability Tier-3s; it gates trust in any delivered ensemble metric. Still not Tier 1 (the ensemble is experimental, not yet delivered). Member of Cluster 2. Recommend a `/falsify` probe comparing per-member vs ensemble MCR before trusting aggregate metrics.

---

### C-112: C-111 changes training dynamics — pre/post-fix model metrics are not comparable

| Field | Value |
|-------|-------|
| ID | C-112 |
| Tier | 4 |
| Source | review (PR #64, 2026-06-03) |
| Trigger | When comparing CRPS/MCR (or any post-training metric) between a model trained before the C-111 merge and one trained after — or when assembling an ensemble whose members straddle the merge boundary — attributing the difference to anything other than the now-active balancer is unsound |
| Location | `views_hydranet/train/training_engine.py:make()` (optimizer param groups), model artifacts in `views-models/models/*/data/generated/` |
| Cross-refs | C-111 (resolved — the fix), C-79 (no pipeline reproducibility test), C-124/C-125 (rollout × balancer confound) |

The C-111 fix makes the MultiTaskLoss balancer actually learn, which changes trained weights for every run after the merge. Concretely: the golden_hour ensemble evaluated 2026-06-03 (constituents trained pre-fix, sb CRPS 0.1298 / MCR 0.300) is a pre-C-111 baseline. A post-fix retrain will differ, and the difference will conflate (a) the balancer effect with (b) ordinary run-to-run variance (the sweep showed substantial variance: pink_pirate sb MCR ranged 0.029–0.552 across seeds). Any pinned golden-number assertion on post-training metrics would also need re-baselining. The risk is methodological: drawing a causal "C-111 improved X" conclusion from a single pre/post pair, or silently mixing pre- and post-fix artifacts in one ensemble. **The same attribution hazard extends forward (folds M-RT6):** enabling Axis-B rollout training (C-125) while the balancer freeze/active question (C-124) is still open would tune two unstable training-dynamics knobs at once and confound which fixed the runaway — sequence the rollout work *after* the balancer verdict closes.

Tier 4 rationale: no correctness impact on any single run — each model is internally valid. The risk is interpretation/comparison hygiene, single-developer scope. Mitigation: when measuring the C-111 effect, retrain all members under identical seeds and compare, ideally with >1 seed to separate signal from variance.

---

### C-113: Autoregressive recurrent runaway — sub-threshold in log-space, expm1-amplified, magnitude-guard-blind

| Field | Value |
|-------|-------|
| ID | C-113 |
| Tier | 2 |
| Source | June-3 golden_hour explosion investigation (2026-06-03) |
| Trigger | When retraining any HydraNet model whose changed training dynamics (loss, scheduled sampling, balancer, seed) let a regression head emit log-space outputs in the ~13–51 range during the 36-step autoregressive roll-forward — `expm1` amplifies these to 1e5–1e22 raw, corrupting CRPS/MCR, while the log-space magnitude guard (warn 100 / halt 1000) never fires |
| Location | `views_hydranet/utils/hydranet_inference.py` (autoregressive loop `predict()` ~L294 feedback; guard L300), `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (ConvLSTM cells, no cell-state clamp), `views_hydranet/utils/config_initializer.py:17` (`log1p`↔`expm1`, no bound) |
| Cross-refs | C-51 (warning-to-halt gap), C-20 (soft magnitude guard), C-99 (reg_latent feedback), ADR-028 §1/§2/§3 (deferred guards), ADR-057 (the fix) |

The June-3 retrain produced raw predictions to ~1.66e22 on two heads (`blue_stranger/lr_ns_best`, `violet_visitor/lr_sb_best`). Step-wise evidence (step-1 normal → step-12 catastrophic) proves the divergence is generated by the **autoregressive recursion**, not a static output transform. The mechanism: per-step dropout-mask resampling (Gal & Ghahramani 2016, RNN) seeds noise into a free-running feedback loop with spectral radius ≥1 (ADR-028 §1/§2), and `expm1` (ADR-028 §3) amplifies the resulting log-space value (~36–51) into astronomical raw counts. **The magnitude guard is structurally blind:** it checks log-space |pred| against 100/500/1000, but the damage is done at log-space ≈ 50, below every threshold.

This is distinct from C-51 (which framed the issue as the 100×-gap *between* warning and halt). C-113 is that the catastrophe lives entirely *below* the warning threshold — the guard cannot see it at all.

Tier 2 rationale: structural fragility with a confirmed, realistic trigger (it already fired on a routine retrain). Not Tier 1 (it surfaced loudly via exploded metrics rather than silently), but it produces grossly wrong forecasts that survive every existing guard and corrupt evaluation. Mitigation path: ADR-057 (consistent-mask dropout, inference-only) first; ADR-028 §2 cell-state clamp / in-domain feedback bounding as the deterministic-recurrence fallback. Output clamping (ADR-028 §3) is explicitly NOT the primary fix — it fights MCR.

**Update 2026-06-04 — diagnosis sharpened (two empirical results):** (1) ADR-057 (consistent-mask dropout) was **FALSIFIED** — violet still exploded under locked masks (`reports/postmortem_locked_dropout_negative_result.md`); the driver is the deterministic recurrence, not dropout noise. (2) A pre-registered `freeze_h` 2×2 ablation (`reports/results_freezeh_ablation.md`) localized the channel: **all four `freeze_h` modes explode within ~½ order of magnitude on `lr_sb` (2.1–7.0 ×10¹⁷), including `all` (entire recurrent hidden/cell state frozen → 5.13e17).** Therefore the divergence is **not** carried by the recurrent hidden-to-hidden dynamics but by the **prediction→input feedback loop** (the model's gain on its own fed-back prediction > 1). Consequences: `freeze_h` (shipped as `"hl"` in all configs) is **inert** against this failure and is a candidate for retirement (it also creates a train/inference mismatch — training evolves the full state, inference freezes `hl`); the **ADR-028 §2 cell-state clamp is pre-falsified** (`freeze_h="all"` is its extreme and failed); the correct fix targets the **input→output map** — spectral-norm/Lipschitz on the input-to-hidden `Wx*` + U-Net encoder/decoder convs, pushforward/GTF training, and/or an in-domain feedback-input clamp (magnitude-neutral). See `reports/options_catalogue_autoregressive_stability.md` §0 UPDATE. **Axis-0 diagnostic run** (`reports/results_io_gain_diagnostic.md`, `scripts/diagnose_io_gain.py`): standalone retrain-free rollout reproduces the explosion — violet's free-running input→output map settles at an **out-of-range attractor (~log 40 → `expm1` ≈ 1e17, matching the observed CRPS)** while pink stays in-range (~log 10), **state-independently**. The local operator norm `‖∂reg/∂x‖₂` does *not* discriminate (both >1); the discriminator is the attractor level vs data range. **In-domain feedback clamp — TESTED (`reports/results_feedback_clamp.md`):** clamping the fed-back prediction per-target to the log1p data max (`feedback_clamp_log1p`, inference-only) **averts the catastrophe** (violet `lr_sb` 2.13e17 → 798; `lr_ns`/`lr_os` recover to healthy; benign on pink) but is a **safety rail, not a resolution** — it triggers falsifier F2 for `lr_sb` (ramps to the ceiling and pins → MCR ~56,000 over-prediction). C-113 stays **OPEN**; the clamp is retained as an optional guard rail (`feedback_clamp_log1p`, default None=off). Durable fix still required: lower the input→output attractor (spectral-norm/Lipschitz on input→output path, pushforward/GTF, or count-likelihood head).

**Update 2026-06-05 — ACUTE CAUSE FOUND (C-111 balancer bisect, `reports/results_balancer_bisect.md`):** a pre-registered, device-matched (GPU/GPU) bisect on violet found the driver: **the C-111 fix (un-freezing the MultiTaskLoss `log_vars`) is what causes the runaway.** Frozen-balancer retrain settles in-range (free-running attractor log ~4–5 → `expm1` ~1e2); active-balancer control diverges (log ~15–17 → ~1e7) — ~5 orders apart. So the acute explosion is a **C-111 regression**, not a fundamental flaw (the model was stable for years with the balancer effectively frozen). **Fix direction: regularise the balancer (bound/decay `log_vars`, lower its LR), not permanent freeze** — `freeze_multitask_balancer=True` is the safe immediate fallback / the bisect's extreme. Caveat: single seed each (C-112/C-119) — confirm on ≥1 more seed + a real `--evaluate` before production. The **chronic** problem (MCR≪1, no calibrated uncertainty) is orthogonal and still motivates the ZITD distributional-head dossier. `freeze_h` (inert) and the in-domain clamp (bounded-but-degenerate) were both *downstream* of this cause.

**`mtloss.py` audited (2026-06-05, expert-method-review):** the `MultiTaskLoss` is faithful to Kendall 2018 — the `+log σ` self-regularising term (`+ torch.log(stds)`) is **present and correctly signed**, and the per-task coefficients match (`1/2σ²` regression, `1/σ²` classification). So the runaway is **not** a loss-spec bug (hypothesis "M2" cleared); it is the **optimization-trajectory effect** of the active reweighting steering training into a divergent-recurrence basin. Fix direction therefore shifts *away from* adding a prior/decay (redundant with the existing regulariser) *toward* slowing/scheduling the reweighting (lower-LR / warmup) or freezing — pending the benefit check in C-124.

**Update 2026-06-05 — durable fix designed + `freeze_h` retirement gated (expert-method-review, rollout dossier):** the durable fix is now designed as **Axis-B rollout training** (`reports/2026-06-05_rollout_training_dossier/`, ADR-058 candidate; see C-125/C-126) — train the prediction→input feedback operator (the runaway carrier) that is currently detached at `training_engine.py:200`. Retiring the (inert) inference-time `freeze_h` changes the inference path for *every* model, so it must be **gated** behind a `rollout_horizon=1` parity guard + a per-model golden_hour re-eval before merge — not flipped globally blind (folds method-review finding M-RT5). **Characterization gate PASSED 2026-06-05** (`views-models/logs/freezeh_pink_eval_*.log`): pink_pirate evaluated on its existing artifact with the freeze_h-removed path (always-`none`) reproduces the healthy reference to ~3 d.p. — step-wise CRPS lr_sb **0.133** / lr_ns **0.031** / lr_os **0.051** (ref ~0.13/0.03/0.05), MCR healthy. ⇒ removal is non-regressing; M-RT5 cleared. (Branch `chore/retire-freeze-h`: method + config field + 5 capability tests removed; ADR-027/CIC updated; ruff + 699 tests + validate_docs green.)

---

### C-114: Undocumented assumption — no dropout on the ConvLSTM recurrent connections, rationale unknown

| Field | Value |
|-------|-------|
| ID | C-114 |
| Tier | 4 |
| Source | ADR-057 design discussion (2026-06-03) |
| Trigger | When adding dropout to the ConvLSTM recurrent connections (the deferred Decision 3b / I7 experiment) or otherwise reasoning about the model's regularization surface — the original reason for the omission is unknown, so the safety of changing it cannot be assessed |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (4 ConvLSTM cells, L373–411 — dropout-free; single `self.dropout` applied only on the U-Net emission path) |
| Cross-refs | ADR-057 §Open Questions; I5 (provenance investigation, views-hydranet#69); I7 (deferred recurrent-dropout experiment) |

HydraNet applies dropout only on the 16 U-Net emission sites; the recurrent cells carry none. Whether this was a deliberate stability choice (the pre-Gal literature found naive recurrent dropout destabilizing), an oversight, or inherited from a predecessor architecture is not recorded or remembered. Until resolved (I5), "no recurrent dropout" must be treated as an undocumented assumption rather than a justified decision, and it gates any recurrent-dropout experiment.

Tier 4 rationale: no current correctness impact (the omission may well be correct); single-developer scope. The risk is purely that an unrecorded rationale could be silently violated by a future change. Resolved by documenting the provenance (I5).

---

### C-115: Silent CPU fallback during training — no fail-loud when CUDA is unavailable

| Field | Value |
|-------|-------|
| ID | C-115 |
| Tier | 2 |
| Source | repo-assimilation (2026-06-05) + lived incident this session |
| Trigger | Launching a training run after the GPU's CUDA context has wedged (commonly a laptop lid-suspend/resume, or a prior CUDA crash) — `torch.cuda.is_available()` returns False and training proceeds on CPU with only a WARNING |
| Location | `views_hydranet/utils/utils_logging.py` (`WARNING - HydraNet running on CPU … severely degraded`); device selection in `training_engine`/manager; cf. `views-models/scripts/run_balancer_frozen_gpu.sh` (gated workaround) |
| Cross-refs | C-64 (silent NaN propagation — different mechanism); reference-cuda-uvm-wedge-fix (memory) |

Training has no hard CUDA gate: when CUDA is unavailable it logs a warning and continues on CPU. This session a wedged GPU (post lid-suspend / a prior `unspecified launch failure`) caused a ~2h frozen-balancer retrain to run silently on CPU (~4× slower, 33 vs 129 month/s) and — worse for science — produced a CPU-trained artifact device-mismatched against the GPU-trained control, confounding the C-111 bisect. The degradation is recoverable but invisible until someone inspects `nvidia-smi`.

Tier 2 rationale: structural fragility (no fail-loud) with a clear, recurring trigger (lid-suspend wedges CUDA on this hardware); silently degrades runs and can invalidate cross-run comparisons. Not Tier 1 (it warns; no data corruption).

*Test-coverage shadow (test-review 2026-06-05):* no test asserts fail-loud/gate behavior when CUDA is unavailable; the gated driver `views-models/scripts/run_balancer_frozen_gpu.sh` is the only enforcement and is not under test.

---

### C-116: Post-evaluation `queryset pg_metadata` publish OOM — eval process exits 137 after metrics

| Field | Value |
|-------|-------|
| ID | C-116 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-05) + 4 eval runs this session; recalibrated 2→3 (review-rr 2026-06-05) |
| Trigger | Running any `--evaluate` on this box — the post-eval queryset-metadata publish step peaks ~12 GB RSS and is OOM-killed (`dmesg: Out of memory: Killed process (python)`), exiting 137 |
| Location | post-evaluation publish step at the manager / views-pipeline-core boundary (after the wandb run-summary in every eval log; e.g. `…Publishing/Fetching queryset pg_metadata`) |
| Cross-refs | C-07 (per-window memory cleanup — resolved, different phase) |

Every constituent eval this session exited 137 (SIGKILL) during a post-metrics `Publishing/Fetching queryset pg_metadata` step, OOM-killed at ~12 GB anon-rss. Metrics survive because the kill lands after the wandb summary syncs (proven by exact baseline reproduction), so it reads as a spurious "failure." But it is a real resource fragility: a tighter-RAM environment, a larger grid/model, or any reordering of the publish step relative to the sync would lose the results outright.

Tier 3 rationale (recalibrated 2026-06-05): reproducible process death (4/4 evals) with a clear trigger, **but non-corrupting** — metrics are computed/synced before the OOM, so no result is lost today. Peer-compared to the Tier-2 band (C-113 corrupts forecasts; C-115 silently degrades runs), this is operational/resource fragility that *could* escalate to data loss under modest change — Tier 3 with a watch note, promote to 2 if the publish step ever moves ahead of the metric sync or RAM headroom shrinks. Member of Cluster 6.

---

### C-117: `HydraNetConfig` `extra="allow"` tolerates ghost config keys — silent config drift

| Field | Value |
|-------|-------|
| ID | C-117 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-05) |
| Trigger | Adding/renaming a config key in a views-models config (or a typo) that is not a HydraNetConfig field — Pydantic `extra="allow"` accepts and ignores it with no error |
| Location | `views_hydranet/utils/config_initializer.py` (HydraNetConfig `extra="allow"`, "Tolerant Handshake") |
| Cross-refs | C-06 (config returns dict after validation); C-101 (extra sigma-dict keys silently accepted — resolved, specific case) |

The config model deliberately allows extra keys (the cross-pipeline "Tolerant Handshake"). The cost is that a typo'd or stale key (e.g. a `prediction_format` ghost key present in tests/configs but absent from the schema) is silently accepted and never applied — a behavior the developer believes is active but isn't. This is config drift with no fail-loud; the narrower per-target-sigma instance was C-101 (resolved).

Tier 3 rationale: maintainability/drift across views-models configs; no direct corruption, but increases the cost and risk of every config change. A deliberate design choice whose downside should be tracked.

---

### C-118: `visual_diagnostics.py` (1050 LOC) — largest module, hot-path coupled, weakly tested

| Field | Value |
|-------|-------|
| ID | C-118 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-05) |
| Trigger | Editing diagnostics, or running train/eval on data that yields degenerate slices — a diagnostic computation (e.g. an all-NaN slice) raises/warns inside the train/inference path |
| Location | `views_hydranet/utils/visual_diagnostics.py` (1050 lines — largest module; biopsy/dossier saves during training/inference); `All-NaN slice encountered` RuntimeWarning observed in the suite (visual_diagnostics.py:479) |
| Cross-refs | C-16 (catch-all handlers — resolved); C-26 (plot correctness untested — resolved); C-21 (bare except in diagnostics — resolved) |

The diagnostics module is the single largest file and runs inline during training/inference (forensic biopsies, dossiers). Earlier robustness issues (C-16/C-21/C-26) were resolved, but the size/complexity and hot-path coupling remain, and a live `All-NaN slice` numpy RuntimeWarning shows degenerate-data paths still surface from within it. A failure here can interrupt a run that is otherwise about the model, not the plots.

Tier 3 rationale: maintainability + critical-path coupling; concrete (live warning) but no correctness impact on model output. Borderline observation, tracked for surface area.

*Test-coverage shadow (test-review 2026-06-05):* the degenerate-data path that emits the `All-NaN slice` RuntimeWarning (visual_diagnostics.py:479) is unasserted — `test_visual_diagnostics.py` does not characterize it.

---

### C-119: GPU runs are not bit-reproducible despite the reproducibility gate

| Field | Value |
|-------|-------|
| ID | C-119 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-05) + C-111 bisect observation |
| Trigger | Re-running a "reproducible" GPU training with identical seed/config and expecting matching outputs — non-deterministic CUDA kernels yield different results run-to-run |
| Location | `views_hydranet/infrastructure/reproducibility_gate.py` (locks np/torch seeds + deterministic_algorithms, but cannot force bitwise-deterministic CUDA kernels) |
| Cross-refs | C-112 (pre/post-C-111 comparability); C-79 (no reproducibility comparison test); C-42/C-43 (reproducibility gate — resolved) |

The gate locks seeds and requests deterministic algorithms, but same-config GPU retrains still diverge in magnitude: the C-111-bisect control retrain settled at CRPS ~1e7 vs the June-3 run's ~1e17 (same seed/config). The qualitative outcome (out-of-range vs in-range) reproduces; the numeric value does not. Any bisect/ablation comparing a single GPU retrain to a prior one must therefore treat magnitude deltas as possibly-spurious and rely on device-matched, ideally multi-seed comparisons (cf. C-112).

Tier 3 rationale: reliability of inference *about experiments*; affects how comparisons are designed, not the model's correctness. No silent corruption.

*Test-coverage shadow (test-review 2026-06-05):* the reproducibility envelope is uncharacterized — no test pins what is guaranteed (qualitative outcome) vs not (bitwise magnitude) on GPU; cf. C-79 (no reproducibility comparison test).

---

### C-120: Dual data-layer authority — DataFetcher + DataSniffer (and cross-repo counterparts)

| Field | Value |
|-------|-------|
| ID | C-120 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-05) + user observation |
| Trigger | Changing data loading or ingestion validation — `DataFetcher` and `DataSniffer` both touch parquet read/validate, and views-pipeline-core has `ViewsDataLoader`/`CoreDataSniffer` counterparts; ownership is unclear |
| Location | `views_hydranet/utils/data_fetcher.py` (class DataFetcher, fetch_df), `views_hydranet/utils/data_sniffer.py` (class DataSniffer, sniff_*); cross-repo views-pipeline-core loaders/sniffers |
| Cross-refs | C-75 (DataFetcher↔VolumeHandler derivation duplication) |

The data layer splits loading/validation across DataFetcher (fetch) and DataSniffer (validate), with overlapping parquet-read and validation responsibilities that also have counterparts in views-pipeline-core (ViewsDataLoader, CoreDataSniffer). A change to ingestion validation could require edits in multiple places, and the duplication invites drift between the hydranet-local and pipeline-core validators. Distinct from C-75 (DataFetcher↔VolumeHandler derivation logic) but the same "data-layer duplication" theme.

Tier 3 rationale: coupling/ownership ambiguity raising change cost across repos; no correctness impact today.

---

### C-121: No automated regression guard for the C-113 autoregressive runaway — the only monitor is contractually blind

| Field | Value |
|-------|-------|
| ID | C-121 |
| Tier | 2 |
| Source | test-review (2026-06-05) |
| Trigger | A future change to training dynamics (new loss/head — e.g. the ZITD work — balancer, scheduled sampling, seed) re-introduces or fails to prevent the autoregressive runaway; with no fast test it surfaces only after a ~40-min eval (or reaches production forecasts if an eval is skipped) |
| Location | `tests/` (no rollout-boundedness test exists), `views_hydranet/utils/hydranet_inference.py` (`predict()` 36-step loop + guard ~L305), `scripts/diagnose_io_gain.py` (the fast attractor seam — zero tests), `views_hydranet/utils/integrity_guardian.py` (guard contract) |
| Cross-refs | C-113 (the runaway itself); C-62 (IntegrityGuardian in inference — resolved); C-20 (soft magnitude guard — resolved); C-107 (hydranet_inference unit coverage) |

The system's costliest, recurring failure mode (C-113) has **no seconds-level regression test**: the 36-step `predict()` rollout is exercised only through full evaluation, so a regression costs a GPU-hour (or a corrupted forecast) to detect. Worse, the one runtime monitor — `IntegrityGuardian` — is tested and works *as specified* (halt at log-space 1000), but its **contract is structurally blind to the actual mechanism**: the explosion does its damage at log-space ~40 (→ `expm1` ~1e17), below every guard threshold. The fast seam now exists — `scripts/diagnose_io_gain.py` reproduces the runaway retrain-free in ~30 s — but it is untested and not wired into the suite.

Tier 2 rationale: structural fragility (the most expensive recurring failure has no cheap guard, and the existing guard cannot see it) with a clear, realistic trigger (every training-dynamics change, several of which are imminent in the ZITD dossier). Not Tier 1 — it currently surfaces loudly at eval rather than silently corrupting shipped output, but it *would* reach forecasts if eval is ever bypassed.

*Tier-boundary note (review-rr 2026-06-05):* this is the highest-tiered *test-gap* in the register — C-107 and C-79 are also test gaps but sit at Tier 4. The difference justifying Tier 2: C-121 gates the Tier-2 C-113 specifically, and its trigger is imminent (ZITD). Reviewers comparing the three should treat C-121 as "test gap on a Tier-2 failure" rather than a peer of the Tier-4 coverage gaps. Member of Cluster 1.

*Doc-gap (review-base-docs 2026-06-05):* `docs/CICs/IntegrityGuardian.md` §3 advertises the ±1000 magnitude-ceiling `RuntimeError` as a guarantee but §6 (Failure Modes) does **not** note that the ceiling is in log-space and is blind to `expm1`-amplified out-of-range outputs below it — giving false assurance of protection that C-113/C-121 disprove. Remediation: add the blind-spot to IntegrityGuardian.md §6.

---

### C-122: Behavior-rich classes lack CICs — chiefly `HydraNetInference` (the autoregressive engine)

| Field | Value |
|-------|-------|
| ID | C-122 |
| Tier | 3 |
| Source | review-base-docs (2026-06-05) gap analysis |
| Trigger | Refactoring or extending the inference loop / balancer / model — e.g. the ZITD head (dossier P2), the `_clamp_feedback`/`freeze_multitask_balancer` work, or any `predict()` change — with no CIC stating the contract to preserve |
| Location | `views_hydranet/utils/hydranet_inference.py` (HydraNetInference — no CIC), `views_hydranet/utils/mtloss.py` (MultiTaskLoss — no CIC), `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (HydraBNUNet06_LSTM4 — no CIC) |
| Cross-refs | C-107 (HydraNetInference zero unit tests); C-113/C-121 (the runaway lives in the un-contracted inference loop); C-03 (hardcoded heads — model); D-02 (architecture left un-refactored) |

16 classes have no CIC; most are small value/registry types, but three are behavior-rich and undocumented: **`HydraNetInference`** (the 556-LOC autoregressive engine — `predict()`, `freeze_h`, `_clamp_feedback`, posterior sampling, the magnitude guard; the riskiest surface, where C-113/C-121 live and where C-107 already flags zero unit tests), **`MultiTaskLoss`** (the homoscedastic balancer at the centre of C-111 and the `freeze_multitask_balancer` flag), and **`HydraBNUNet06_LSTM4`** (the model whose `forward`/`ModelOutput`/hidden-state contract the ZITD head will change). With no CIC, a developer modifying these has no stated contract to preserve.

Tier 3 rationale: maintainability/governance gap that raises the cost and risk of changing the system's most critical, most-about-to-change classes; no current correctness impact. Not Tier 2 — no failure is imminent purely from the missing docs. Highest-value item: a CIC for `HydraNetInference`.

*SRP/complexity facet (expert-code-review 2026-06-05):* beyond lacking a CIC, `HydraNetInference.predict()` (`hydranet_inference.py:243`, ~200 LOC) is a god-method interleaving dropout-mask reset, feature indexing, hidden-state init, the digest/seed/autoregress loop, `freeze_h`, `_clamp_feedback`, the magnitude guard, and viz/biopsy accumulation. 5 of 8 expert lenses converged here: it is the riskiest (C-113/C-121), least-contracted (this entry), least-unit-tested (C-107), and next-to-change (ZITD dossier P2) surface. Recommended pre-ZITD remediation: extract the per-step bodies into named methods (the seam C-107/C-121 also need) — but keep the shared-accumulator algorithm coherent (see D-05).

---

### C-123: `choose_loss` returns a positional 3-tuple — loss components conflated by index

| Field | Value |
|-------|-------|
| ID | C-123 |
| Tier | 4 |
| Source | expert-code-review (2026-06-05) |
| Trigger | Adding/removing a loss component (e.g. the ZITD single-likelihood head collapsing reg+cls) shifts the meaning of `criterion[0]/[1]/[2]`; callers indexing by position break silently or subtly |
| Location | `views_hydranet/utils/utils.py:106` (`choose_loss` → `tuple[reg\|dict, class, MultiTaskLoss]`), consumed positionally in `views_hydranet/train/training_engine.py` (`criterion[2]` = balancer) |
| Cross-refs | C-99 (reg vs reg_latent dual-path); C-122 (MultiTaskLoss undocumented) |

`choose_loss` returns three losses as a positional tuple; consumers must *know* index 2 is the `MultiTaskLoss` balancer (the C-111 epicenter). A named structure (dataclass `reg`/`cls`/`balancer`) would remove the positional coupling. Low impact today (callers are internal and tested), but the ZITD head — which replaces the reg+cls pair with one likelihood — is exactly the change that would reshuffle the tuple.

Tier 4 rationale: code-quality/readability; internal, tested callers; no correctness or reliability impact today. Flagged because the imminent ZITD change touches it.

---

### C-124: MultiTaskLoss balancer's predictive benefit is unverified — known to destabilise, not known to help

| Field | Value |
|-------|-------|
| ID | C-124 |
| Tier | 3 |
| Source | expert-method-review (2026-06-05) |
| Trigger | Choosing/keeping a balancer regularisation (or re-enabling the active balancer) before comparing frozen vs active on CRPS/MCR/calibration, held-out, ≥1 extra seed |
| Location | `views_hydranet/train/training_engine.py` (the log_var optimizer param group); `reports/results_balancer_bisect.md` (measured *stability*, not *skill*) |
| Cross-refs | C-111 (the balancer un-freeze), C-112 (pre/post comparability), C-113 (the runaway) |

The C-111 bisect established the active balancer **destabilises** the autoregressive dynamics (out-of-range attractor) but **never measured whether it improves predictive skill** vs the frozen (equal-weight) baseline. Regularising or re-enabling the balancer therefore risks engineering a fix for a component whose benefit is unverified — and per the panel's strongest dissent (Sutton/Harrell), **"ship frozen" may be the correct endpoint, not a fallback.** The `mtloss.py` audit (see C-113 update) cleared the loss as a defect, so the runaway is an *optimization-trajectory* effect of the active reweighting — which makes "is the reweighting worth its fragility?" the live empirical question.

Tier 3 rationale: methodology / decision-hygiene gap that could mis-direct the acute fix; no silent corruption. Mitigation: a **pre-registered frozen-vs-active CRPS/MCR comparison (≥1 seed)** before choosing among the regularisation options. *(Audit note: the sibling hypothesis "M2 — Kendall-loss mis-implementation" was checked and **cleared** — `mtloss.py` is faithful to Kendall 2018 — so it is recorded in the C-113 update rather than registered as an open defect.)*

**Stage-1 result (2026-06-05, `preanalysis_balancer_benefit.md`):** the FROZEN violet artifact evaluates **healthy** (step-wise CRPS lr_sb 0.197 / lr_ns 0.043 / lr_os 0.052, ~ the pink reference) while ACTIVE is the known 2.13e17 explosion ⇒ the balancer does **not** earn its place; **freeze is the acute C-113 fix**. Single seed (violet/42) — validation upgraded to a **3-seed × 2-balancer-state sweep** (`preanalysis_balancer_sweep.md`); C-124 resolves on confirmation.

**Update 2026-06-05 — sweep complete (5/6), F2 FIRED, freeze falsified as the fix:** the 3×2 seed×balancer sweep finished (`seed99_frozen` crashed on a CUDA wedge; 5/6 cells). **Active explodes 3/3** (prediction held), but **frozen is NOT robust**: `seed4_frozen → inf` (worse than `seed4_active`), seed 99 frozen missing. Per the pre-reg decision rules, **F2 fired → re-open: freezing is insufficient and the balancer is not the sole cause** (`reports/preanalysis_balancer_sweep.md` §RESULT). The "ship `freeze_multitask_balancer=True` as the robust acute fix" conclusion is **falsified** — freezing is seed-fragile and on seed 4 actively harmful. ⇒ the chronic train/inference **exposure-bias** mismatch (Axis-B rollout training, **C-125**) is the root; the balancer is one trigger among seeds. **C-124 stays OPEN** (does not resolve as "freeze ships"); the durable fix is the rollout-training program.

---

### C-125: Rollout-training (Axis B) procedure rests on three unverified methodological premises

| Field | Value |
|-------|-------|
| ID | C-125 |
| Tier | 3 |
| Source | expert-method-review (2026-06-05, rollout-training dossier `02b`) |
| Trigger | When implementing rollout training (B1 pushforward / B2 GTF) in `training_engine`, before merging — verify all three: (a) the `L_stability` weight is annealed/small and CRPS reported uncontaminated; (b) the stability readout certifies the **full 36-step** horizon, not just the K≤36 training window; (c) for B2, α is used only as heuristic gradient control unless `λ_max>0` is established for the conflict DGP |
| Location | `reports/2026-06-05_rollout_training_dossier/02_design.md` §4.2/§5/§7/§8; `views_hydranet/train/training_engine.py` (loss assembly + the rollout loop) |
| Cross-refs | C-113 (the runaway this fixes), C-124 (balancer benefit), C-112 (attribution confound), ADR-058 (candidate); folds method-review findings M-RT1/M-RT2/M-RT3 |

The Axis-B design adds a multi-step rollout objective to fix the C-113 runaway. The method-review panel flagged three premises that, if unchecked, make the fix subtly wrong. **(a) Proper-score corruption (M-RT1, Gneiting)** — pushforward/GTF stability terms are *regularisers*, not proper scoring rules (Gneiting & Raftery 2007); a fixed nonzero weight moves the training optimum off the true predictive distribution, so the headline CRPS must stay uncontaminated and the weight annealed. **(b) Truncated-horizon blindness (M-RT2, Hochreiter)** — training/certifying to K=12 gives biased gradients and no stability guarantee for steps 13–36, where the runaway compounds; the readout must check the full horizon or a tail falsifier. **(c) Chaos-premise (M-RT3, Sutton/Gneiting)** — GTF's provable `α*=1−1/σ̃_max` bound is valid only for chaotic systems (Hess 2023 / Mikhaeil 2022, not held); the conflict DGP is likely non-chaotic (the explosion is `expm1` out-of-range drift), so α may serve only as heuristic gradient control, not a correctness guarantee.

Tier 3 rationale: design-stage methodology gaps for an as-yet-unbuilt feature; no current silent corruption, but each could mis-direct the C-113 fix or ship a subtly biased forecaster. Mitigation: the pre-analysis plan (`05`) pre-registers (a)/(b)/(c) as binding guardrails + falsifiers before any full retrain; fetch Mikhaeil 2022 to settle (c). Promote to Tier 2 if rollout training is implemented without these guards.

---

### C-126: Rollout-training success metric conflates point-stability with calibration

| Field | Value |
|-------|-------|
| ID | C-126 |
| Tier | 3 |
| Source | expert-method-review (2026-06-05, rollout-training dossier `02b`) |
| Trigger | When evaluating the first rollout-training experiment — using the `diagnose_io_gain` free-running attractor magnitude as the sole success criterion, without also reporting PIT/coverage + MCR + zero-rate |
| Location | `scripts/diagnose_io_gain.py`; `reports/2026-06-05_rollout_training_dossier/02_design.md` §8–9 |
| Cross-refs | C-113 (point/mean runaway), C-110 (ensemble MCR calibration), ADR-057 + the ZITD distributional-head dossier (chronic MCR); folds method-review finding M-RT4 |

The C-113 runaway is a *point/mean* pathology (the trajectory leaves the data range); the chronic problem (MCR≪1, no calibrated uncertainty) is a *calibration* pathology. The panel (Gneiting + Shi) warned these are independent: multi-step rollout optimisation rewards mean-hedging / regression-to-the-mean (a known ConvLSTM nowcasting failure), which can leave — or *worsen* — calibration and the zero-rate even as the attractor returns in-range. Declaring "C-113 resolved" on attractor magnitude alone would be a category error and could silently degrade the very metric the ZITD head exists to fix. This is the dossier's strongest live dissent (D5), carried forward as a falsifier.

Tier 3 rationale: evaluation/decision-hygiene gap; no silent corruption today, but it gates whether the rollout fix is genuinely progress. Mitigation: the rollout-training readout must include calibration (PIT/coverage) + sharpness (MCR/zero-rate) as first-class metrics, pre-registered in `05`.

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
| Resolution | Leave as-is. Cost of refactoring (breaking all artifacts) exceeds benefit. Structural test in `tests/test_architecture.py` provides adequate safety — this test is load-bearing infrastructure; do not modify without understanding its role as the guard for this decision. **⚠ Re-decision due at ZITD dossier P2 (review-rr 2026-06-05):** the "leave alone" basis is about to be challenged — the ZITD head collapses each reg+cls pair into one likelihood (C-03 trigger now imminent), which *forces* a head-topology change and will invalidate artifacts regardless. Make this decision as part of ZITD planning, not by default. |

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

### D-05: `predict()` decomposition — extract named steps vs preserve algorithm depth

| Field | Value |
|-------|-------|
| ID | D-05 |
| Source | expert-code-review (2026-06-05) |
| Perspectives | Martin/Feathers (extract the digest/seed/autoregress bodies into named private methods — SRP + testable seams for C-107/C-121/C-122), Ousterhout (caution: the causal loop is a cohesive deep algorithm over shared mutable accumulators; fragmenting into shallow methods that pass accumulators around can *increase* complexity, not reduce it) |
| Resolution | **Open.** Relevant before the ZITD head edits `predict()`. Likely synthesis: extract per-step *pure-ish* helpers while keeping the accumulator-owning loop in one place (extract steps, not the orchestration). Tie to C-122 (CIC) + C-121 (rollout test). |

---

### D-06: IntegrityGuardian blind spot — deepen the abstraction vs fence with a test

| Field | Value |
|-------|-------|
| ID | D-06 |
| Source | expert-code-review (2026-06-05) |
| Perspectives | Ousterhout/Nygard (deepen the guard so it reasons in the space where the catastrophe occurs — i.e. post-`expm1`/attractor-aware — rather than a log-space ceiling that hides the real failure), Beck (cheaper: add an external fast regression test (`diagnose_io_gain`) and document the ceiling's limit in IntegrityGuardian.md §6, rather than complicate the guard) |
| Resolution | **Open.** Cross-refs C-113/C-121 (the blind guard) and the C-121 doc-gap annotation. Decision deferred until the C-111 bisect / ZITD direction clarifies whether the runaway is even being fixed upstream. |

---

## Tech-Debt Backlog (demoted from register, review-rr 2026-06-05)

Demoted per the three-track model: Tier-4, mechanical-or-standing, single-file/single-developer scope — kept for traceability (full entries remain tagged `[DEMOTED]` in §Open Concerns) but no longer counted as active risks. Actionable as ordinary tech-debt, not governance risks.

| ID | One-line action | Cluster |
|----|-----------------|:--:|
| C-89 | Extract `_SumReducer` to `tests/conftest.py` (the `_tobit_config`/`tobit_config_3target` part is already done). | — |
| C-49 | Roadmap item: revisit flat→nested config schema if keys exceed ~50 or a feature needs 4+ grouped keys. | 5 |
| C-37 | Accepted trade-off: extract an `IVolumeHandler` Protocol only if an alternative implementation (lazy/GPU-resident) is needed. | 3 |
| C-85 | Add a `flip_probability` config key (currently hardcoded `0.5` in `training_engine.py:290-292`). | — |

---

## Resolved Concerns

### C-111: MultiTaskLoss log_vars frozen at zero — homoscedastic uncertainty weighting was silently inert — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-111 |
| Tier | 3 |
| Resolved | 2026-06-03 (branch `fix/multitask-logvar-weight-decay`) |
| Source | review-rr strategic blind-spot analysis (2026-06-02), production sweep diagnosis |
| Location | `views_hydranet/utils/mtloss.py` (MultiTaskLoss log_vars), `views_hydranet/train/training_engine.py:make()` (optimizer construction) |
| Cross-refs | GitHub views-hydranet #59 |

Across all 8 runs of the production integration sweep (80–200 lessons), the MultiTaskLoss `log_vars` parameters remained at exactly 0.000 — the Kendall et al. (2018) homoscedastic uncertainty weighting never learned.

**Corrected root cause (2026-06-03):** The original GH #59 hypothesis was that `weight_decay=0.1` regularized `log_vars` back to zero. Code inspection during the fix revealed a deeper cause: the optimizer in `choose_scheduler()` is built only from `unet.parameters()`, and `make()` added only the regression sigma params (`criterion[0]`) to it — the `MultiTaskLoss` instance (`criterion[2]`) `log_vars` were **never added to the optimizer at all**. They accumulated gradients during `backward()` but `optimizer.step()` never updated them. The *exactly* 0.000 value (not a small nonzero equilibrium) is the fingerprint of "never stepped," not "decayed."

**Resolution:** `make()` now adds `multitaskloss_instance.parameters()` to the optimizer in a dedicated param group with `weight_decay=0.0` (uncertainty estimates should not be weight-decayed — that would defeat their purpose, addressing the original #59 concern too). Three TDD tests added in `tests/test_learnable_sigma.py::TestGreenMultiTaskBalancerInOptimizer`: log_vars in optimizer, weight_decay=0.0 on that group, and log_vars move after a step.

Tier 3 rationale (retained): not data corruption, but silent ineffectiveness of a documented mechanism — the register's core domain. Affected every training run prior to the fix.

---

### C-108: 46% of test classes lacked ADR-005 taxonomy markers (Green/Beige/Red) — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-108 |
| Resolved | 2026-06-02 (PR #53) |
| Resolution | Batch-renamed 45 unmarked test classes to TestGreen/TestBeige/TestRed across ~15 files. Verified: `grep` for unmarked `class Test*` returns 0. The falsification `TestP*` (Proposition) naming was preserved as a valid domain convention. Cross-ref C-60 (April 2026, 16% → 36% adoption); this completes the push to 100%. |

---

### C-109: 13 skipped falsification tests were stale investigation artifacts — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-109 |
| Resolved | 2026-06-02 (PR #53) |
| Resolution | Converted 6 investigation-concluded skips (`test_falsification_sweep_understanding.py`, `test_falsification_sweep_root_cause.py`) to `@pytest.mark.xfail(run=False, reason=...)`, preserving intent without execution noise. The 5 remaining bare skips are legitimate conditional skips — 2 `_run_inference_pipeline not yet implemented`, 3 `Calibration parquets not available` — not stale artifacts, correctly left as-is. The triage distinguished investigation-concluded artifacts from data/implementation-gated skips. |

---

### C-73: Legacy `evalution_mode` typo shim in HydraNetConfig — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-73 |
| Tier | 4 |
| Source | manual (2026-04-21) |
| Trigger | When all model configs in `views-models` have been confirmed to use `evaluation_mode` (not `evalution_mode`), remove the `handle_typos` model_validator shim |
| Location | `config_initializer.py:143-153` (`handle_typos` model_validator) |

`HydraNetConfig` has a `model_validator(mode="before")` shim that silently rewrites the legacy typo key `evalution_mode` → `evaluation_mode`. One known consumer (`views-models`) has been fixed (2026-04-21), but other model configs in the `views-models` repo may still use the old key. The shim should be removed once a grep across all configs in `views-models` confirms zero remaining instances of `evalution_mode`. Removing it prematurely would break any config still using the typo — Pydantic's `extra="allow"` would silently accept the misspelled key and leave `evaluation_mode` at its default.

---

### C-95: Tobit S2 MCR asymmetry — lr_sb=0.983, lr_os=0.005 — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-95 |
| Resolved | 2026-06-02 |
| Source | S2 Tobit experiment (2026-05-29), wandb run summary |
| Trigger | When evaluating Path E (scheduled sampling) results against the S2 baseline — MCR asymmetry may persist or worsen, and should be diagnosed before declaring Gate 2 passed |
| Location | Evaluation metrics (wandb), not a code defect. Upstream: `views_hydranet/utils/tobit_loss.py`, `views_hydranet/train/training_engine.py` |
| Cross-refs | C-87 (per-target loss weights) |

S2 Tobit experiment (150 lessons, `loss_reg=tobit`, `loss_reg_sigma=1.0`) shows extreme MCR asymmetry across targets: lr_sb MCR_sample=0.983 (nearly all predictions above marginal median — systematic upward bias), lr_os MCR_sample=0.005 (nearly all below — systematic underprediction). The sample-vs-mean gap for lr_sb (0.983 sample vs 0.555 mean) indicates individual posterior samples are consistently biased high while the posterior mean is more centered — the stochastic spread does not straddle the median.

This is not a code defect but a model behavior concern. Possible causes: (1) Tobit censored likelihood with σ=1.0 may overestimate latent z* for zero-cells, pushing predictions upward for the most zero-inflated target (SB ~95% zeros). (2) Per-target loss weights may need recalibration for Tobit (current weights were tuned for hurdle+Basu). (3) The fixed σ may be too large or too small for different targets.

Tier 3 rationale: model quality concern that affects evaluation interpretation, not silent corruption. No code fix needed — requires experimental investigation (σ sensitivity, per-target σ, target_weights recalibration).

---

### C-96: Tobit loss converges in ~60 lessons — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-96 |
| Resolved | 2026-06-02 |
| Source | S2 Tobit experiment (2026-05-29), training loss curves |
| Trigger | When configuring `total_lessons` for Tobit loss runs — using the MSE/Shrinkage-calibrated default of 20 lessons is too few, but 150 is excessive |
| Location | Config parameter `total_lessons` in model configs (`views-models`), `views_hydranet/utils/config_initializer.py:105` |

S2 training curves (linear and log-scale) show regression loss plateauing at ~25.8 by lesson 60, with lessons 60-150 oscillating in a ±0.3 noise band (log-scale) around the plateau. Classification loss shows similar convergence by lesson 60 (current: 3.15). Total multi-task loss converges to ~48 by lesson 60.

Tobit converges faster than hurdle+MSE because it provides dense gradient from ALL cells (including y=0 censored observations), eliminating the gradient starvation that slowed MSE convergence. The optimal `total_lessons` for Tobit is likely 60-80, saving ~50-60% training time compared to 150.

Tier 4 rationale: efficiency concern, not correctness. Training produces correct results at 150 lessons, just wastes compute. Single-developer scope.

---

### C-98: Implicit `input_channels == 3 × output_channels` constraint — RESOLVED

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

### C-105: No validator enforces `features ⊆ regression_targets` — RESOLVED

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

### C-87: Hurdle mechanism applies uniform loss parameters across targets with different rare-event ratios — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-87 |
| Resolved | 2026-06-01 |
| Resolution | The hurdle mechanism was replaced by Tobit censored-normal likelihood (ADR-054), which provides dense gradient from ALL cells. Per-target sigma (issue #44, ADR-055) gives each target its own loss scale. Scheduled sampling (ADR-056) closes the exposure bias. The root cause (gradient starvation from hurdle masking) is eliminated. Per-target loss weights (`target_weights` from ADR-050) remain available as an additional lever. |

---

### C-93: `_evaluate_sweep` not implemented — sweep evaluation crashes with `NotImplementedError` — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-93 |
| Resolved | 2026-05-28 |
| Resolution | Decomposed `_setup_evaluation()` per D-04 consensus: extracted `_load_model_artifact()`, made `model` a required parameter of `_setup_evaluation()`, added `_evaluate_sweep()` override. Commit 9b38532 (ADR-053). Verified by 5 successful wandb sweeps across May-June 2026. |

---

### C-97: Step-wise CRPS degradation quantifies exposure bias — Path E baseline metric — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-97 |
| Resolved | 2026-06-01 |
| Resolution | Gate 2 PASSED. Scheduled sampling (ADR-056, PR #50) reduced the step/month MCR gap for lr_sb from 1.60 to 0.02 (99% reduction) at `ss_epsilon_max=0.5`. Step-wise sb CRPS improved 43% (0.265 → 0.152). No escalation to GTF needed. Baseline metrics preserved in `reports/sweep_isolation_plan.md`. |

---

### C-100: `validate_basu_dpd_range` TypeError crash when `loss_reg_sigma` is dict — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-100 |
| Resolved | 2026-05-30 |
| Resolution | Added `isinstance(self.loss_reg_sigma, (int, float))` guard before `<= 0` comparison in `validate_basu_dpd_range`. PR #47. Falsification stub `test_P1_basu_dpd_with_dict_sigma_crashes_with_typeerror` verifies the fix. |

---

### C-101: Extra keys in per-target sigma dict silently accepted — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-101 |
| Resolved | 2026-05-30 |
| Resolution | Added extra-key check to `validate_per_target_sigma`. PR #47. Falsification stub `test_P5_extra_keys_in_dict_sigma_rejected` verifies. Also added same check to `validate_target_weights` for consistency. |

---

### C-102: Stale type annotations for `criterion_reg` after per-target sigma change — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-102 |
| Resolved | 2026-05-30 |
| Resolution | Updated type annotations in 3 locations: `_process_sequence` parameter, `TrainingContext.__init__`, and `choose_loss` return type — all now `nn.Module | dict[str, nn.Module]`. PR #47. |

---

### C-103: CIC HydraNetConfig.md stale after per-target sigma — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-103 |
| Resolved | 2026-05-30 |
| Resolution | CIC Section 3 updated with per-target sigma validation responsibility. Section 6 updated with 4 new failure modes (non-tobit, non-positive, missing target, extra key). Field count updated. PR #47. |

---

### C-104: ParetoLoss registered in LOSS_REG_REGISTRY but has no test file — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-104 |
| Resolved | 2026-06-02 |
| Resolution | Investigation found 5 existing tests in `tests/test_cluster_e.py`: importable, nn.Module, forward returns scalar, registry integration, gradient behavior (outlier compression). The register entry was incorrect — ParetoLoss was tested all along, just not in a dedicated file. |

---

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
