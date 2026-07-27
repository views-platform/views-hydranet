# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-07-27                           |
| Total Concerns    | 221                                  |
| Open Concerns     | 99                                  |
| — of which demoted (tech-debt) | 5 (tagged `[DEMOTED]` in §Open Concerns; indexed in §Tech-Debt Backlog) |
| Resolved Concerns | 122                                  |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Causal Clusters (review-rr strategic, 2026-06-05; refreshed 2026-06-24, 2026-07-27)

Open concerns reduce to **13 root decisions**. Fixing a root advances multiple entries; entries are tagged `[Cx]` informally in this map (not in every entry body). Clusters 1–6 are the original 2026-06-05 map (C-01…C-123); clusters 7–13 were added 2026-06-24 to cover the ~46 entries from the ZINB epic, the channel-role refactor, the over-smoothing investigation, the name-coupling review, and the views-frames migration that the original map predated.

| # | Root decision | Member entries | Fix scope | Priority |
|---|---|---|:--:|---|
| **1** | Inference surface (`predict()`) never contracted/tested/decomposed; guarded only by a log-space ceiling | ~~C-121~~ RESOLVED, C-113 (mitigated), C-122, C-107, C-114 (+D-05, D-06) | 1 coordinated | **largely addressed (2026-07-27):** ZITD head shipped; `predict()` gained the D×K sampler + T=0-neutral rollout_feedback (ADR-070) + IntegrityGuardian weight scan + behavioral non-finite/rollout tests; **C-113→evidenced-mitigation, C-121→resolved**. Residual: C-122/107/114 (unit coverage, model facets) — lower urgency |
| **2** | Training-dynamics changes outran reproducibility/comparability discipline | ~~C-119, C-79~~ RESOLVED, C-112 (seed-in-sidecar shipped, S5a — verify), C-110 | 2 | **mostly closed (2026-07-27)** — bit-repro + determinism guard shipped; residual C-110 (+ confirm C-112) |
| **3** | `utils/` accreted multiple domains without package structure / clear ownership | C-35, C-01, C-36, C-37, C-120, C-75, C-76 | 3 (large blast radius) | defer |
| **4** | Single hardcoded head/loss topology (3+3 heads, positional loss tuple) | C-03, C-123 (+C-122 model facet, D-02) | 2 | decide *with* ZITD planning |
| **5** | Config is a typed-model-masquerading-as-dict (`extra="allow"`) | C-06, C-117, C-49 (+D-03) | 2 | defer (D-03 tension) |
| **6** | Operational/GPU fragility on the dev box (no hard CUDA gate; publish-step memory) | C-115, C-116 | 1–2 | near-term |
| **7** | ZINB/hurdle likelihood never fully committed + train/inference objective mismatch | C-137, C-141, C-143, C-144, C-145, C-146, C-148, C-149, C-150, C-129 (+D-08, D-09; C-140 RESOLVED) | 2–3 | **head SHIPPED (ADR-067 nb/zinb + bloom verdict), but all 10 objective/likelihood members still open — NEEDS member-level status review** (several likely closeable post-bloom-epic; not verified) |
| **8** | `feature_cols` overloads model-inputs and training-targets (C-156 is the named root) | C-156 (root), C-160, C-166 (C-157/158/159 RESOLVED) | 1–2 | mostly closed — C-156/160/166 remain |
| **9** | Evaluation is resolution-blind ⇒ in-sample over-smoothing unmeasured/confounded | C-167, C-168, C-169 (+C-136) | 2 | near-term |
| **10** | Upstream name/format string is the unmediated contract + join-key + role | C-173, C-174, C-175, C-176, C-177 (+D-10, D-11) | 2 | defer (D-11 tension) |
| **11** | Overridable phase-template silently drops the wandb/bookkeeping lifecycle | C-132, C-133, C-134 (+D-07) | 1–2 | near-term |
| **12** | #110 baseline-run operational readiness (config drift / decision-rule / runtime harness) | ~~C-161, C-162~~ RESOLVED, C-163, C-164 | 1–2 | unblock before #110 — C-163/164 remain |
| **13** | Rollout-training & balancer methodology rest on unverified premises | C-124, C-125, C-126, C-128, C-170 | 2 | decide with rollout work |

**Highest-value (refreshed 2026-07-27):** Cluster 1 is no longer the priority — the bloom epic (#183/#193) + today's cleanup addressed most of it (C-113 mitigated via ADR-070, C-121 resolved, predict() sampler + tests + IntegrityGuardian shipped). The **live research front is Cluster 7** (ZINB objective/likelihood): the head is built but the train/inference-objective and likelihood-commitment questions are unverified — **the top follow-up is a member-level status review of Cluster 7** to separate genuinely-open from epic-closed. Clusters 2, 8, 12 are mostly closed (residuals: C-110/112; C-156/160/166; C-163/164). The **magnitude/amount-ceiling** (not a register risk) remains the standing research ceiling.

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

### C-37: VolumeHandler in SAP "Zone of Pain" — partial abstraction at PF boundary [DEMOTED]

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

### C-49: Flat config schema may not scale — no nested structure for regularizers, strategies, or per-target settings [DEMOTED]

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

### C-85: Flip probability 0.5 hardcoded in training_engine — not config-driven [DEMOTED]

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

### C-89: `_SumReducer` and `_make_tiny_model` duplicated across test files [DEMOTED]

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

**Update 2026-06-09 — step-1 read isolates the two axes (the hurdle works; the explosion is purely the rollout):** A teacher-forced **step-1** read of the Arm-1 hurdle eval (`predictions_calibration_20260609_051916` vs the Tobit baseline `…165326`) shows the magnitude head **un-collapsed at step 1** (rollout not yet engaged): MCR_pos `lr_sb` 0.11→**0.19**, `lr_ns` 0.02→**1.29**, `lr_os` 0.03→**0.73**. The explosion appears only from step-2 onward, when the now-nonzero magnitudes feed the **untrained** free-running loop — re-confirming the runaway carrier is the prediction→input feedback (Axis B), **not** the output/magnitude representation. **Corollary (folds M-Z1/M-Z3 from the 2026-06-09 distributional-head method reviews):** the claim that a sub-exponential (softplus / count-head) output link "dissolves the runaway by construction" is **unproven** — the only proof on this data (DynAttn / Iacus 2025) is a *direct* (non-autoregressive) ZINB that never faces the feedback loop; no held paper studies autoregressive stability of a distributional count head. ⇒ any count/hurdle-NB head must be **gated by a `diagnose_io_gain` 36-step explosion check before its eval is trusted** (pre-registered in the distributional-head dossier `05 §0`). See C-136 (the confound this conflation created) and C-129 (sequencing).

---

**Update 2026-07-27 (S8, Epic #193, ADR-070) — EVIDENCED MITIGATION (bloom bounded 9/9 by the sample-on default; io-gain>1 NOT eliminated).** ADR-070 makes `rollout_feedback=sample` the default for family heads: the AR loop feeds back a sparse, in-distribution family draw instead of the diffuse emit-mean, so the deployed rollout no longer *excites* the >1 input→output gain. Counted verdict (`reports/2026-07-25_t0_rollout_skill_dossier/06_bloom_verification_verdict.md`): on 6 retrained known-seed models (matched 40 lessons), mean-feedback blooms **9/9** arms, sample-feedback bounded **9/9** (field `crps_none` mean 36–95 → sample 0.002–0.35; `M_mean` mean 285–751 → sample 0.02–2.49). **This is a mitigation, not a resolution:** it removes the *trigger* (OOD dense feedback) but the underlying input→output io-gain>1 is unchanged, so a dense-feedback path (`rollout_feedback=mean`, or a future non-sampling consumer) can still bloom. The durable fix (spectral-norm/Lipschitz on the input→output map, or rollout/GTF training) stays **OPEN**; C-113 remains open at Tier 2 as a now-*contained* risk. The seconds-level regression guard that catches a re-introduction is **C-121 (resolved)**. T=0-neutrality of the default is byte-exact (ADR-070 §3; sampler fix `66a95ea`).

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

**UPDATE 2026-06-18 — the cited gate no longer exists.** `views-models/scripts/run_balancer_frozen_gpu.sh` (the "only enforcement") was **verified absent** this session (`find views-models -name '*.sh'` — gone). So there is currently **no CUDA gate at all**: a lid-suspend/resume wedged `nvidia_uvm` (CUDA `is_available()→False`) and a run ground silently on CPU for hours before detection. The systemic gap is now tracked as **C-163** (rebuild the guarded-run harness). Member of Cluster 6.

---

### C-116: Post-evaluation `queryset pg_metadata` publish OOM — eval process exits 137 after metrics

| Field | Value |
|-------|-------|
| ID | C-116 |
| Tier | 2 |
| Source | repo-assimilation (2026-06-05) + 4 eval runs this session; recalibrated 2→3 (review-rr 2026-06-05); **re-escalated 3→2 (2026-06-15, determinism-validation runs)** |
| Trigger | Running any `--evaluate` on this box — the post-eval queryset-metadata publish step peaks RSS and is OOM-killed (`dmesg: Out of memory: Killed process (python)`), exiting 137 |
| Location | post-evaluation publish step at the manager / views-pipeline-core boundary (after the wandb run-summary in every eval log; e.g. `…Publishing/Fetching queryset pg_metadata`) |
| Cross-refs | C-07 (per-window memory cleanup — resolved, different phase); C-135 (eval-OOM of an exploding model — likely related memory axis) |

Every constituent eval this session exited 137 (SIGKILL) during a post-metrics `Publishing/Fetching queryset pg_metadata` step, OOM-killed at ~12 GB anon-rss. Metrics survive because the kill lands after the wandb summary syncs (proven by exact baseline reproduction), so it reads as a spurious "failure." But it is a real resource fragility: a tighter-RAM environment, a larger grid/model, or any reordering of the publish step relative to the sync would lose the results outright.

Tier 3 rationale (recalibrated 2026-06-05): reproducible process death (4/4 evals) with a clear trigger, **but non-corrupting** — metrics are computed/synced before the OOM, so no result is lost today. Peer-compared to the Tier-2 band (C-113 corrupts forecasts; C-115 silently degrades runs), this is operational/resource fragility that *could* escalate to data loss under modest change — Tier 3 with a watch note, promote to 2 if the publish step ever moves ahead of the metric sync or RAM headroom shrinks. Member of Cluster 6.

**UPDATE 2026-06-15 — escalated to Tier 2; the documented escalation trigger ("RAM headroom shrinks") has fired.** Both determinism-validation runs (violet no-coords, seed 42; `journalctl -k`) were OOM-killed at **anon-rss 16.6 GB** (pids 1421921 @ 02:38:55, 1433425 @ 03:50:36), on a **32 GB box with swap exhausted** (`global_oom, constraint=NONE`) — up from the ~12 GB documented 2026-06-05. The kill stage is confirmed unchanged (`viewser …queryset.py: Publishing/Fetching queryset pg_metadata` → `Processing features [05:26]` → kill, ~24 min after metrics+artifact+predictions were written, so still non-corrupting *today* — the determinism verdict is unaffected). **What is NOT yet established:** the cause of the 12→16.6 GB growth. Regression window is the ZINB epic (commits 2026-06-10..06-13: #99 HurdleNBLoss, #100/#101 hurdle-NB head+inference-mean, #106/#108 coord seam) — but no single recent change is a verified ~4.6 GB allocator (`_emit_magnitude` float64 is per-step/transient; `feature_scaler.inverse_transform_volume` 2× full-volume copy is pre-existing). Could also be partly environmental (swap state). **Do not assert a mechanism without per-stage RSS measurement** (this lineage already mis-diagnosed C-135's OOM and retracted). Coordinate channels will push the peak *higher* (more input channels) → this gates the coord epic. Probe in flight: `n_posterior_samples` dropped 16→3 in the violet config to test whether our posterior/inverse-transform volume (scales with samples) or the viewser publish step (does not) dominates the peak.

**UPDATE 2026-06-18 — ROOT CAUSE MEASURED (per-stage RSS sampler).** The 12→18+ GB growth is a **sample-scaled double-buffer at the publish tail**, not the model. Measured trajectory (8-sample run, quiet box): flat **~2.7 GB through training AND eval/posterior-sampling** → climbs to ~9 GB during prediction-frame assembly → **doubles 9→18 GB in ~60 s at the publish tail** → OOM at **anon-rss ~18–20 GB** (`min avail 0–1 GB`). Mechanism: `evaluation_mode='stochastic'` (violet config) **skips `collapse_to_point`** (`inference_orchestrator.py:116-117`), so PredictionFrames carry the full **`(N, S)`** matrix (`prediction_frame_assembler.py:178`); the tail then holds **two full copies** of the S-scaled `[T,H,W,C,S]` structure simultaneously — `feature_scaler.inverse_transform_volume`'s `work_data = vh.data…copy()` (`feature_scaler.py:219`, runs *before* any collapse) **and** the viewser pandas materialization (`Processing features`). **Sample-scaled and confirmed by bisection:** 3 samples completes (`Done. Runtime`), 8 OOMs — both **online and offline wandb (~18 vs ~20 GB)**, so wandb is NOT the cause (that hypothesis tested and falsified). The historical 30–300-sample runs fit because S was collapsed before the heavy step; the current path carries S to publish. **Minimal fix:** collapse S→point (or a streaming CRPS accumulator) *before* `inverse_transform_volume`, and transform in place (drop the `.copy()`). **Test-coverage shadow:** no test asserts peak-RSS sub-linearity in `n_posterior_samples` — exactly why this regressed invisibly for two weeks (see new **C-164** seam-test gap, **C-163** runtime-harness gap). GitHub **#124**.

**CORRECTION 2026-06-19 — the stochastic-(N,S) attribution above is FALSIFIED.** A point-mode run (`evaluation_mode='point'`, which collapses S→`(N,1)`) **also OOM'd at ~16 GB** (exit 137, `min avail 0`). So the dominant ~16 GB hog is **mode-independent**, NOT the carried `(N,S)` PredictionFrames (those account for only the ~2 GB delta between point=16 GB and stochastic=18 GB). *Why point mode can't help:* `inverse_transform_volume` must invert **all S samples in raw space before** the arithmetic-mean collapse (ADR-021 — averaging then inverting ≠ inverting then averaging for log1p/expm1), so the full S-scaled volume is materialized + `.copy()`'d regardless of mode (collapse is `inference_orchestrator.py:117`, *after* invert at `:113`). **Leading (NOT yet line-isolated) candidate:** the `inverse_transform_volume` full-volume `.copy()` (`feature_scaler.py:219`) and/or the viewser `Processing features` publish — the spike is at the forecast/publish tail (after eval, which streams safely). **Established:** mode-independent, sample-scaled (3 fits / 8 OOMs), at the publish tail. **Not established:** which exact allocation. **Decisive next step = NO more full guessing runs** — git-archaeology of the output path since the 30–300-sample era + per-stage in-process RSS instrumentation (the C-164 fast seam). *Process note: I've now asserted a mechanism twice and been wrong twice — this entry's claims must be measurement-backed, not inferred.*

**PROBE RESULT 2026-06-19 (`scripts/probe_inverse_transform_memory.py`, CPU, seconds) — our inverse-transform is NOT the hog.** Measured peak RSS of `inverse_transform_volume` on the real forecast shape `[36,180,180,11,S]` float32: **S=3 → 0.16 GB, S=6 → 0.31 GB, S=8 → 0.42 GB** (in-place variant ≈ same). The volume is **0.41 GB at S=8, not 4.1 GB** — my earlier shape math was a **10× arithmetic error**; the `.copy()` is trivial. **Both code-side hypotheses (stochastic `(N,S)` publish, inverse-transform `.copy()`) are now falsified by measurement.** ⇒ the ~16 GB lives **downstream in the `views-pipeline-core` publish/report tail** (the `Publishing/Fetching queryset pg_metadata` + `Processing features [05:26]` step), which materializes pandas dataframes (`reporting/stage.py::generate_forecast_report` → `_load_historical_data` → `read_dataframe`; `prediction_frame_converter.py:73`); `skip_predictions_delivery` only gates parquet *delivery*, not this. **NOT yet pinned to the exact function. Decisive next test (no guess):** run eval **without `-re`** — if it completes clean, the report/publish tail is the hog (and eval alone yields the baseline); if it still OOMs, the hog is inside `-e`. The probe is the seed of the **C-164** seam test.

**ROOT CAUSE CONFIRMED 2026-06-19 (proof by removal) — it is the `-re` report stage, NOT our code.** Running eval **without `-re`** (`python main.py -r calibration -e --saved`) **completed clean: exit 0, `Done. Runtime: 14 min`, peak RSS 2.41 GB, min avail 13 GB, no OOM** — vs 16–20 GB OOM on every `-t -e -re` run. So the entire ~16 GB is the **`-re` reporting/publish tail in `views-pipeline-core`** (`reporting/stage.py` materializing the full historical pandas dataframe + the viewser `Processing features`/`pg_metadata` step). **Our model/eval/inverse-transform path is memory-safe (~2.4 GB).** Eval-without-`-re` **persists the predictions + metric parquets** (all `mcr_readout` / the #110 decision rule need). **Disposition:** C-116 **demoted from coord-epic blocker** — run experiments with `-e` (no `-re`); the proper fix (pipeline-core report memory) is a **separate, deferrable** views-pipeline-core concern, not views-hydranet. The coordinate epic is **unblocked**.

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

*Update (falsify, 2026-06-06) — sequencing/decision gap (RT-P2):* concluding B1 also requires a **decided MultiTaskLoss balancer config** to retrain under (C-124 open: active destabilises, frozen is seed-fragile) **and** resolution of the **circular R6 gate** — `02_design` §7.0 says "sequence Axis B after the C-111 balancer verdict closes," but C-124 stays open and *defers to Axis B as the fix*, so R6 cannot close on its own terms. Record the balancer-config decision + an explicit "R6 satisfied / proceed" call in the pre-analysis plan (`05`). A RED stub marks the gap: `tests/test_falsification_rollout_plan.py`. Cross-ref C-124.

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

**Update 2026-06-09 (folds M-Z5 — count-head facet):** the same guard applies to the count / hurdle-NB head — on ~95%-zero data a degenerate near-zero forecast scores well (the "F5 zero-rate trap"). Judge the count head on **positive-subset proper scores (twCRPS/CRPS on `y>0`) + PIT/coverage + a posterior-predictive zero-rate/tail check**, never aggregate CRPS alone. Pre-registered in the distributional-head dossier `05 §0`; see C-137.

**Update 2026-06-22 — F5 timid-prophet CONFIRMED empirically at T=0 (multi-seed); recommendation corrected to the locked framework.** Overnight count-vs-sharp ×3 seeds (`reports/2026-06-22_body_multiseed_dossier/results.md`): the sharp/shrinkage body **wins CRPS on all 3 targets, every seed**, while emitting **~1–4% of the true magnitude (MCR≈0.03)** and **losing the QS (QS99) guardrail to the count body** — i.e. CRPS-based selection picks the timid prophet exactly as F5 warns, and the QS99/MCR guardrails correctly veto it. ⇒ **never select a body/config on CRPS (or visual sharpness) alone.** NOTE: the "twCRPS / positive-subset" recommendation above is **superseded by the locked FAO-02 framework** ([[reference_fao02_locked_eval_framework]], `brain/2_projects/fao02/.../pre_release_note_05`): score on **CRPS + QS99 + Brier + MCR over the FULL dataset** — twCRPS and positive-subset/extremes-conditioning were considered and REJECTED there (the forecaster's dilemma, Lerch2017).

---

### C-127: Duplicate dict keys in model configs (F601) — later definition silently shadows the earlier

| Field | Value |
|-------|-------|
| ID | C-127 |
| Tier | 4 |
| Source | tech-debt-cleanup (ruff F601, 2026-06-06) |
| Trigger | Editing the *first* occurrence of one of these keys expecting it to take effect — Python keeps the *last* definition, so the earlier edit is silently dead |
| Location | `views-models/models/{heavy_strider,light_strider,white_ranger}/configs/config_hyperparameters.py` (`skip_predictions_delivery` ×2); `views-models/models/new_rules/configs/config_sweep.py:124,130` (`expansion_coefficient_dim` ×2) |
| Cross-refs | C-06 (config `extra="allow"` masks unknown keys) — distinct: F601 is linter-visible, not silent passthrough |

Four model configs define a dict key twice; Python silently retains the last assignment. **Severity is Low because the values mostly agree:** all three `skip_predictions_delivery` duplicates are `True` (pure redundancy, no behavioural effect — the earlier "could change delivery" worry is *not* realised). The one real divergence is `new_rules` sweep `expansion_coefficient_dim`: line 124 `[64,128]` is dead, line 130 `[32,64,128]` wins — so the sweep search space is `[32,64,128]` regardless of the misleading earlier line. Sweep-scoped (hyperparameter search), not production delivery → no silent *output* corruption.

Tier 4 rationale: linter-visible (ruff F601, not silent), values mostly identical, the one divergence affects only a sweep search space. Mitigation: delete the dead (earlier) duplicate in each file, preserving current behaviour; confirm `new_rules` intended `[32,64,128]`.

**Fix applied 2026-06-06** (views-models `e9ced12`, pushed to `development`): the earlier (dead) duplicate removed in all four files, current behaviour preserved; ruff F601 clean repo-wide. Open pending merge to main; `new_rules`'s `[32,64,128]` kept as the effective value — author to confirm `[64,128]` was not the intent.

---

### C-128: Locked-dropout posterior is the inference default, but its calibration is unvalidated across models

| Field | Value |
|-------|-------|
| ID | C-128 |
| Tier | 3 |
| Source | review-diff (PR #75, 2026-06-06) |
| Trigger | Trusting or delivering a model's MC-dropout uncertainty (MCR/coverage) from the locked-mask posterior before the per-model calibration analysis (dossier I3) has been run on a model other than `pink_pirate` |
| Location | `views_hydranet/utils/hydranet_inference.py:83-84` (`set_locked_dropout(True)`, unconditional); `views_hydranet/architectures/locked_dropout.py` |
| Cross-refs | ADR-057 (the mask change); C-113 (its falsified C-113-fix justification); C-126 (point-stability ≠ calibration); C-110 (the Tier-2 ensemble-calibration cousin) |

ADR-057 makes inference use a **locked** dropout mask (fixed across the 36-step roll-forward, fresh per posterior sample) instead of per-step fresh masks, and PR #75 turns this **on by default** (unconditional at `HydraNetInference.__init__`). The locked mask **narrows the posterior spread** vs per-step dropout, so MCR/coverage can shift. It was **spot-checked benign on `pink_pirate`** (the freeze_h characterization eval reproduced pink's reference CRPS/MCR), but the planned cross-model calibration analysis (dossier I3 — "is the fixed-mask posterior calibrated or too tight?") has **not** been run on the other members. Risk: a delivered posterior could be silently mis-calibrated (too tight) on an unchecked model with no error signal — the C-110-style silent-miscalibration failure mode.

Tier 3 rationale: a **validation gap**, not a known defect — spot-checked benign on pink, MC-dropout was already approximate, and the change is intended (ADR-057, consciously accepted at the #75 merge). Escalate to Tier 2 (à la C-110) if a locked-mask posterior is relied on for delivery before I3. Mitigation: run the I3 calibration analysis (PIT/coverage + MCR) across models; or gate locked dropout behind an opt-in flag if I3 is deferred.

---

### C-129: Rollout-training (Axis B) × ZITD distributional-head coordination is untracked

| Field | Value |
|-------|-------|
| ID | C-129 |
| Tier | 3 |
| Source | falsify (2026-06-06, "nothing blocks concluding Axis B" audit, P4) |
| Trigger | Progressing the Axis-B rollout-training (B1) and the ZITD distributional head independently — each editing `training_engine` and the `log1p` autoregressive feedback — without a coordination plan for how they compose or which lands first |
| Location | `reports/2026-06-05_rollout_training_dossier/02_design.md` + `reports/2026-06-05_distributional_head_dossier/02_design.md`; `views_hydranet/train/training_engine.py` (rollout loss + feedback); the autoregressive feedback re-encoding |
| Cross-refs | C-125/C-126 (rollout premises / calibration), C-113 (the shared target), the two dossiers |

The two active research programs both modify the **same** training loop and the **same** autoregressive feedback path, but their **interaction is unanalysed**. The rollout dossier declares itself "distinct from the ZITD dossier"; the ZITD dossier never mentions Axis B. Yet B1's rollout loss would have to train *through* ZITD's feedback re-encoding (`log1p(mean or sample)` from a softplus-link distributional head), and ZITD's softplus link is itself the *other* proposed cure for the same `expm1` runaway. Open questions neither plan owns: do B1 (rollout gradients) and ZITD (output representation) compose or conflict? Which lands first? Does B1's stability term interact with ZITD's NLL/CRPS objective? Left untracked, the two could collide in `training_engine` (merge/sequencing) and in feedback semantics.

Tier 3 rationale: a cross-program coordination/dependency gap (no silent corruption today), but it raises cost-of-change and could mis-sequence the two largest research efforts. Mitigation: a one-paragraph coordination note in each dossier's `02_design` (or a shared sequencing decision) — likely "ship one cure for the runaway first (Axis B *or* the ZITD softplus link), measure, then layer the other," recorded before either retrain.

**Update 2026-06-09 — evidence of coupling + escalation re-scoped (two method reviews + a step-1 read):** The distributional-head escalation is now **hurdle-NB-first**, not Tweedie/ZINB-mixture (a review caught the π mis-specification — see C-137). More importantly, the step-1 read (C-113 update) shows the magnitude fix (hurdle) **works one-step** and the explosion is purely the rollout — so the two programs are not merely "uncoordinated", they are **empirically coupled**: a magnitude fix un-collapses the head, which then *needs* rollout training to stay bounded. **Revised sequencing:** rather than "one cure or the other", the leading plan is **hurdle + scheduled-sampling (rollout training) together** — the hurdle un-collapses the head *first*, which dissolves the old D5/C-126 worry about training the rollout on a collapsed head. Recorded in the distributional-head dossier `00/02/05 §0`. See C-136.

---

### C-130: `aggregate_method` silently inert under `evaluation_mode='stochastic'`

| Field | Value |
|-------|-------|
| ID | C-130 |
| Tier | 4 |
| Source | review (config_hyperparameters.py sanity check, 2026-06-07) |
| Trigger | Setting or tuning `aggregate_method` expecting it to affect output while `evaluation_mode='stochastic'`, or switching to `'point'` and assuming the prior `aggregate_method` had been active |
| Location | `views-models/models/violet_visitor/configs/config_hyperparameters.py:118-119`; behavior in `views_hydranet` `HydraNetConfig` |

The violet config sets both `evaluation_mode='stochastic'` and `aggregate_method='arithmetic_mean'`. In stochastic mode the full posterior is preserved and `aggregate_method` has **no effect** — `HydraNetConfig` emits a warning banner saying so. Harmless redundancy, not a defect, but a developer could tune `aggregate_method` expecting it to matter and be misled. The warning is the safety net.

Tier 4 rationale: a config redundancy with an explicit runtime warning; no correctness or reliability impact. Single-config observation.

---

### C-131: `weight_decay=0.1` is large in absolute terms (intentional, but unflagged)

| Field | Value |
|-------|-------|
| ID | C-131 |
| Tier | 4 |
| Source | review (config_hyperparameters.py sanity check, 2026-06-07) |
| Trigger | Investigating posterior calibration/sharpness or regularization strength without accounting for the large `weight_decay` |
| Location | `views-models/models/violet_visitor/configs/config_hyperparameters.py:51` (and `config_sweep.py`) |

`weight_decay=0.1` is high relative to typical values (1e-4 to 1e-2). It is the **established** value (matches `config_sweep.py` and the test baseline), so not a defect — but it is large enough to materially affect regularization and posterior width, and anyone diagnosing those should know it's intentional and large rather than assume a conventional small value.

Tier 4 rationale: a code/config observation, not a correctness issue; the value is deliberate. No silent corruption. Cross-ref C-126 (calibration metric).

---

### C-132: HydranetManager `_execute_model_training` override silently drops the wandb train-run lifecycle

| Field | Value |
|-------|-------|
| ID | C-132 |
| Tier | 2 |
| Source | falsify + 3-agent investigation (wandb training-logging bug, 2026-06-07) |
| Trigger | Running a **single training run** (`main.py -r calibration -t`) and expecting wandb to contain training-phase metrics (loss, `mtl_log_var/*`, sigma, `ss_epsilon`) — or relying on those curves to diagnose a training run |
| Location | `views-hydranet/views_hydranet/manager/hydranet_manager.py:185-187` (override) vs base `views-pipeline-core/views_pipeline_core/managers/model/model.py:~1186` (`_execute_model_training`) |
| Cross-refs | C-112 (pre/post-C-111 training-dynamics comparison — affected if training curves are needed) |

`HydranetManager._execute_model_training` overrides the base `ModelManager._execute_model_training` with a bare `self._train_model_artifact()`, dropping the base method's `with self._wandb_module.initialize_run(job_type="train"): ...` wrapper (and also its `TrainingStage.finalize_training` + `ModelTrainingException` handling). Consequently, on the **single-run `-t` path only**, `wandb.run is None` throughout training and every guarded `wandb.log` in `training_engine.py` (L640/651/664) silently no-ops — no error, no warning, no train run on the dashboard. **Scope is path-specific:** the sweep path (`_execute_model_sweeping`) and eval path (`_execute_model_evaluation`) are NOT overridden, so they keep their wandb runs open and log correctly (verified: pink_pirate sweep TRAIN runs + all EVAL runs logged fine through 2026-06-05). No impact on training correctness, artifacts, or eval metrics — observability loss only.

Tier 2 rationale: a silent divergence from a base-class contract (a subclass phase-override drops a lifecycle the base guarantees), with a clear trigger and zero error signal; latent since ~March 2026 and only surfaced when per-lesson training logging was added 2026-06-01 (it had nowhere to land on the `-t` path). Not Tier 1 (no model-output corruption). Fix direction (pending /expert-code-review): wrap the override body in `initialize_run("train")` — NOT a plain delete, since the override deliberately bypasses `finalize_training`. Why-not-caught: workflow is sweep-centric (`-s`) + eval-metric-centric (`-e`), both of which log fine; no test asserts an active train run.

---

### C-133: Overridable phase-template pattern lets subclasses silently drop the wandb/bookkeeping lifecycle

| Field | Value |
|-------|-------|
| ID | C-133 |
| Tier | 2 |
| Source | expert-code-review (wandb lifecycle, 2026-06-07) |
| Trigger | A `ModelManager` subclass overriding any base `_execute_*` phase method (`_execute_model_training`/`_evaluation`/`_sweeping`/`_forecasting`), or a new phase added to the base, dropping the wandb-run + post-phase bookkeeping the base guarantees |
| Location | `views-pipeline-core/views_pipeline_core/managers/model/model.py` phase methods (~L1139/1186/1253/1537); exemplar override `views-hydranet/views_hydranet/manager/hydranet_manager.py:185` |
| Cross-refs | C-132 (the instance), C-134 (silent no-op), D-07 |

Template-Method shape where the *template* phase methods are freely overridable and `_`-named identically to the *hook* methods (`_train_model_artifact` etc.) subclasses are meant to extend. The LSP postconditions the template guarantees (run created, `finalize_training`, `ModelTrainingException` wrapping — documented at model.py:1178-1181) are doc-only, not enforced. C-132 is one realized instance; this pattern is the reusable trap — the next subclass or new phase reintroduces it silently.

Tier 2 rationale: structural fragility with a clear, recurring trigger and zero error signal; root cause of the C-132 class. Prevention: make templates non-overridable (or `@final`-style), separate hook vs template names, and/or centrally assert the run-lifecycle invariant. Not Tier 1 (no output corruption).

---

### C-134: Silent no-op telemetry when `wandb.run` is None (no fail-loud)

| Field | Value |
|-------|-------|
| ID | C-134 |
| Tier | 2 |
| Source | expert-code-review (wandb lifecycle, 2026-06-07) |
| Trigger | A phase (training especially) runs while `wandb.run is None` — the guarded `wandb.log` calls drop all metrics with no warning or error |
| Location | `views-hydranet/views_hydranet/train/training_engine.py:640/651/664`; `views_hydranet/utils/utils.py:~204` (`train_log`) |
| Cross-refs | C-132, C-133 |

`if wandb.run is not None:` makes "no observability" indistinguishable from "healthy." A ~90-minute training run can lose all telemetry silently — exactly how C-132 stayed hidden, and it bites during the C-112/C-113 investigations that most need training dynamics. Prevention: emit a one-time WARNING (or assert, in non-sweep/non-test runs) when a training loop proceeds with `wandb.run is None`.

Tier 2 rationale: silent failure mode that masks other defects (defense-in-depth gap); clear trigger, no error signal. Observability-only (no correctness impact) → not Tier 1.

---

### C-135: Eval of an exploding (C-113) model OOM-killed the process during a sweep — cause unconfirmed

| Field | Value |
|-------|-------|
| ID | C-135 |
| Tier | 3 |
| Source | overnight sweep OOM (violet posterior-expansion sweep, 2026-06-08); diagnosis CORRECTED after user pushback |
| Trigger | Evaluating a model whose predictions explode (C-113, ~1e11 / `expm1`→inf) — the eval/posterior-sampling phase can balloon RAM enough to hit the global OOM-killer |
| Location | eval/posterior-sampling path during `_execute_model_sweeping`/eval; `views_hydranet` inference; root = C-113 explosion |
| Cross-refs | **C-113 (root cause)**, C-126 |

**RETRACTED first diagnosis:** the original framing ("sweeps accumulate ~2.6 GB/trial across trials → OOM on multi-trial runs") was WRONG. Counter-evidence: healthy-model sweeps (e.g. `pink_pirate`) run dozens of trials over hours without OOM, so sweep trials DO free memory between them — a sweep does not use more RAM than its constituent single run. The accumulation-across-trials mechanism is refuted.

What is known: the process was OOM-killed (`Killed process 2097902 (python) anon-rss ~13 GB, global_oom`, 2026-06-08 ~01:09) **mid-eval** (drawing posterior samples) of an **exploding** model; ~17 GB baseline on a 31 GB box. R1 (a single exploding run) survived its explosion, so OOM is not deterministic per trial. **Leading (UNVERIFIED) hypothesis:** the C-113 explosion inflates eval-phase memory (huge/inf posterior-sample tensors) enough to OOM on some trials — i.e. this is a **symptom/amplification of C-113, not a sweep-infrastructure defect.**

Tier 3 (downgraded from 2): cause unconfirmed and likely a facet of C-113 rather than an independent structural bug. To confirm: measure peak RSS during eval of an exploding vs healthy config (single runs), and within a trial vs across trials. Mitigation is really "fix C-113"; interim, watch memory when evaluating exploding configs.

---

### C-136: Magnitude/output fixes judged on a rollout-confounded test — Arm-1 was mischaracterized as a clean failure

| Field | Value |
|-------|-------|
| ID | C-136 |
| Tier | 3 |
| Source | user pushback + step-1 readout (2026-06-09) |
| Trigger | Judging any magnitude/loss/output-head change (hurdle, count-likelihood, sigma) by its **full 36-step free-running** metrics (MCR/CRPS/explosion) without a **teacher-forced / step-1** read to isolate the magnitude axis from the rollout axis |
| Location | `reports/2026-06-08_magnitude_calibration_dossier/07` (EXP-A1 verdict); `reports/RESULTS_LEDGER.md` (Arm-1 row); the magnitude-vs-rollout axis split |
| Cross-refs | C-113 (the rollout runaway), C-129 (the coupling), C-112 (attribution hygiene) |

Arm-1 (the lognormal hurdle) was recorded as "FAILED → explosion → go structural." A teacher-forced **step-1** read (C-113 update) shows that is a **conflation of two axes**: at step 1 the hurdle *succeeded* — it un-collapsed magnitude (`lr_ns` MCR_pos 0.02→1.29, `lr_os` 0.03→0.73); it only "failed" because those magnitudes then fed the untrained rollout and exploded (Axis B). On ~95%-zero, 36-step autoregressive data, **any** magnitude fix that un-collapses the head will tend to explode on the free-running rollout — so a full-rollout "FAIL" verdict does not test the magnitude fix, it tests the (separate, untrained) rollout. Realized cost: the hurdle was nearly discarded and several turns were spent designing a from-scratch count-likelihood rebuild on the strength of a confounded verdict.

Tier 3 rationale: decision/attribution-hygiene gap (peer of C-112/C-119/C-126) — no silent output corruption, but it already mis-directed the research and could discard good magnitude fixes again. Mitigation: for any magnitude/output change, report **step-1 (teacher-forced) MCR_pos/CRPS** to isolate the magnitude axis, and judge full-rollout stability only as the *separate* Axis-B question; re-evaluate previously-discarded magnitude fixes (Tobit/lognormal) under rollout training (or a step-1 read) before treating them as dead.

**Update 2026-06-09 — two reviews bound the supporting evidence (folds M-R1, M-R2 + the code-review of `/tmp/step1_mcr.py`).** The reframe is sound in *direction* (the hurdle un-collapsed magnitude; teacher-forcing is symmetric across baseline and Arm-1, so the contrast is controlled) but weak in *strength* and *durability*:
- **M-R1 (method):** `MCR_pos` is a first-moment **ratio, not a proper score** — "un-collapsed" is supported, "calibrated/succeeded" is not; no positive-subset twCRPS/PIT was computed. *Trigger:* treating step-1 MCR as a skill verdict.
- **M-R2 (method):** **1 origin, 1 seed, n_pos 50–130, ratio-of-means, no CI** — against the project's own shrinkage-volatility + C-112/C-119 discipline. *Trigger:* a quantitative go/no-go on the step-1 point values without a 2nd seed + bootstrap CI.
- **Code-review:** the readout lives in an **unversioned/untested** `/tmp` script (provenance gap, cf. C-79) and its **prediction↔actual join is unguarded** (no raw-index uniqueness check, no match-rate assertion, silent NaN-drop) — a subtly wrong join would corrupt the numbers with no signal.
- **Remediation (folded into R4 / #93):** promote to a version-controlled `scripts/mcr_readout.py` with a guarded join, reporting a positive-subset proper score + bootstrap CI + per-cell distribution, step-1 **and** full-36, multi-seed. Record wording softened to "un-collapsed (directional)" in `07` / `RESULTS_LEDGER` / memory on 2026-06-09.

---

### C-137: Count-likelihood escalation (hurdle-NB / ZINB / Tweedie) carries likelihood-specification + parameterization design risks

| Field | Value |
|-------|-------|
| ID | C-137 |
| Tier | 3 |
| Source | expert-method-review ×2 (distributional-head design, 2026-06-09) |
| Trigger | When building the count-likelihood escalation head — specifically (a) reusing the focal classifier as a zero-inflation π, (b) choosing a single global dispersion θ, (c) pursuing Tweedie over NB, (d) adding a GPD/EVT tail head, or (e) a hybrid distributional-NLL + per-cell shrinkage penalty |
| Location | `reports/2026-06-05_distributional_head_dossier/02_design.md §0.0`; `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (head); a future `*NBLoss`; `views_hydranet/utils/utils.py` (`LOSS_REG_REGISTRY`) |
| Cross-refs | C-03 (hardcoded 3+3 head topology), C-113/C-129 (the rollout coupling), C-126 (the F5 calibration guard) |

Three design risks for the count-likelihood head, surfaced by two method reviews and folded here:
- **(M-Z6) π-specification conflation.** Reusing the focal classifier (`sigmoid(cls)`, trained on `by_*`=`1[y>0]`) as a **ZINB structural zero-inflation π** mis-specifies the likelihood: the classifier learns the *marginal* `P(y>0)`, not the structural gate (in a ZINB-mixture, zeros come from both π *and* the NB). The proven head on this data (DynAttn) uses a *dedicated* π. **Resolved by design** by adopting the **hurdle-NB** framing (classifier *is* the gate `P(y>0)`; positives = a **zero-truncated** NB), which makes the reuse principled and matches "zero = no event" (Mullahy 1986). The risk re-arises if anyone builds a ZINB-mixture reusing the classifier as π.
- **(M-Z7) Global θ under-parameterization.** A single per-target dispersion θ likely cannot capture the spatial heterogeneity of conflict counts; pre-registered as an MVP simplification with a region/feature-varying-θ ablation queued.
- **(M-Z2) Tweedie-density blocker.** Tweedie NLL (1<ρ<2) needs the Dunn&Smyth series/saddlepoint evaluation — a real implementation+validation cost. NB / hurdle-NB is closed-form and avoids it; Tweedie is the tail-escalation only.
- **(M-Z5) F5 zero-rate trap** (see C-126): on ~95%-zero data a near-zero forecast scores well — judge on positive-subset proper scores + zero-rate, not aggregate CRPS. **Refinement 2026-06-21 (panel Gneiting/Lerch + the proper-score gate, folds M-6):** but conditioning the score *only* on `y>0` (or extremes) is itself improper — the **forecaster's dilemma** (Lerch 2017): it rewards always-predict-the-event. Use **integrand-weighted threshold-weighted CRPS on ALL cells** (proper; `calculate_twcrps_native`) as the selector, with a positive-cell PIT as a *conditional diagnostic* only. The 2026-06-21 gate (`scripts/proper_score_audit.py`, all-cells) showed the hurdle head's defect — predicted mass leaked onto truly-zero cells — is visible *only* to an all-cells proper score, not to positive-subset scoring.
- **(M-4) GPD/EVT tail ξ instability.** An end-to-end GPD/GEV tail head collapses the shape ξ to a "safe" (high-scale/low-shape) mode that **underfits the ~200k tail without erroring**, unless built with a threshold-invariant reparameterization (Wang 2023), an ξ-constraint + bias-init (Galib 2022 DeepExtrema), and two-phase freeze-bulk training. Do not add a tail head without these three.
- **(M-5) Incoherent hybrid.** A "distributional NLL + per-cell shrinkage penalty on the mean" is **not a coherent likelihood** — no posterior corresponds to it, and it double-counts the mean–variance trade-off the dispersion already encodes (Jørgensen 1987 / Bishop). Reading its output as calibrated uncertainty is a silent mis-specification. The coherent alternative is a jointly-trained zero-inflated mixture (clean π₀ + body).

Tier 3 rationale: design-stage methodology gaps for an as-yet-unbuilt head (peer of C-125); no current corruption, but π-conflation would silently mis-specify uncertainty and the others could ship a subtly biased forecaster. Mitigation: the distributional-head dossier `02 §0.0`/`05 §0` pre-registers the hurdle-NB spec, the θ ablation, and the F5/positive-subset eval; gate the build on review. **Update 2026-06-21:** M-4/M-5/M-6 added from the `/expert-method-review` slate (dossier `2026-06-21_proper_score_gate_dossier/02_panel_review.md`); the proper-score gate corroborated that the likelihood (not the backbone) is the lever, so this escalation is now live — gate each candidate on all-cells proper scores + positive-cell calibration.

---

### C-138: Stale test import breaks suite collection — `test_eval_integration_toy` imports a removed `views_evaluation` module

| Field | Value |
|-------|-------|
| ID | C-138 |
| Tier | 3 |
| Source | full-suite run during R2 verification (2026-06-10) |
| Trigger | Running the full `pytest` suite (CI or local) **without** `--continue-on-collection-errors` — the import error in `test_eval_integration_toy.py` aborts collection, so the other ~743 tests **do not run** and a real regression elsewhere is masked behind a single loud error |
| Location | `tests/test_eval_integration_toy.py:6` (`from views_evaluation.evaluation.evaluation_manager import EvaluationManager`) + `:18` (`EvaluationManager()`); installed `views_evaluation/evaluation/` (no `evaluation_manager.py`; exposes `native_evaluator`, `evaluation_frame`, `metric_catalog`, `metrics`, `native_metric_calculators`, `config_schema`, `evaluation_report`) |
| Cross-refs | C-52 (stale tests — resolved precedent), C-10 (importorskip guards), C-79/C-107 (test-coverage gaps) |

`test_eval_integration_toy.py` imports `views_evaluation.evaluation.evaluation_manager.EvaluationManager`, which **no longer exists** in the installed `views_evaluation` — the `EvaluationManager` class/module was removed or renamed upstream (the current package routes evaluation through `native_evaluator` / `EvaluationFrame`). This is **stale-test vs upstream-API drift**, unrelated to the magnitude/rollout program (surfaced incidentally during R2's pre-commit suite run). It is *loud* (ImportError, nonzero exit) — not silent corruption — but because a collection error **interrupts the whole run** by default, a developer or CI seeing "1 error, interrupted" may not realize the other 743 tests never executed, masking unrelated regressions.

Tier 3 rationale: test-integrity / dependency-drift; no model-output impact, but it degrades the suite's value as a regression gate (the masking-by-interrupt hazard). Mitigation: update `test_eval_integration_toy.py` to the current `views_evaluation` entrypoint (likely `native_evaluator` / `EvaluationFrame`), or `pytest.importorskip` it (C-10 pattern) / remove if the toy integration is obsolete; optionally set `--continue-on-collection-errors` in the CI config as defense-in-depth. Tracked in **#95**.

---

### C-139: Program pivot — committed to the ZINB distributional head; all other directions parked/superseded

| Field | Value |
|-------|-------|
| ID | C-139 |
| Tier | 3 |
| Source | chair decision (2026-06-10) + the since-February catalog + the open-issue inventory |
| Trigger | A future session or contributor **re-opening** a parked direction (rollout training, the inference gate, Tweedie, direct multi-horizon, ADR-057 locked-dropout, or the loss-swap zoo) without realizing it was deliberately parked in favour of the ZINB head — i.e. re-entering the 3-week circle the pivot was meant to end |
| Location | Closed issues #91 #94 #81–#89 #40 #63 #59 (superseded); #77 #78 #93 #65–#73 #38 #39 #41 #42 #45 #49 #57 #58 #60 #61 #62 (parked); archived dossiers under `reports/archived/`; live: ZINB dossier `reports/2026-06-10_zinb_distributional_head_dossier/`, epic #97, checklist #104 |
| Cross-refs | C-111 (balancer — **mooted** by ZINB), C-113 (the explosion), C-136, C-137 (count-head design risks — now the *live* direction), `reports/2026-06-10_since_february_catalog.md` |

On 2026-06-10, after a since-February stock-take, the program committed to **ONE direction — the ZINB distributional head** (epic #97) — and parked or superseded every other direction. The 3-week mess was caused by too many open directions at once; this entry exists so the parked ones are **not silently re-opened**.

- **SUPERSEDED (replaced):** the magnitude hurdle/gate program (#81–#89); the reversed "count-likelihood = escalation-only / hurdle+rollout = live" framing (#91, #94); the C-111 balancer question (#59 — **mooted**: a single ZINB likelihood has no regression-vs-classification balancer to freeze); the old ZINB investigation stubs (#40, #63 → recreated as the fresh sub-issues #98–#103).
- **PARKED (documented fallbacks; revisit only if the ZINB head fails):** rollout training (#77, #78, #93); the ADR-057 locked-dropout / MC-dropout-stability program (#65–#73); the loss/calibration "investigate-X" set (#38, #39, #41, #42, #45, #49, #57, #58, #60, #61, #62).
- **Three dossiers archived** to `reports/archived/` (distributional-head + magnitude-calibration = superseded; rollout = parked fallback), each with a supersession header + DISPOSITION pointer to the ZINB dossier.

Discipline (the harness): a frozen linear roadmap (checklist #104), one box at a time, findings logged but non-steering, **two exits only** (ship the ZINB head, or revert to commit `e029e63` and ship that), brake word **CIRCLE**.

Tier 3 rationale: governance / traceability. The risk is process (re-circling), not silent data corruption — but without the recorded trail, the failure mode (re-opening parked work) is exactly what cost three weeks.

---

### Gate Resolutions (2026-06-10) — #97 gate decisions D1–D5

The five blocking decisions are resolved (see `2026-06-10_zinb_distributional_head_dossier/02_design.md`):

| Concern | Decision | Where |
|---------|----------|-------|
| **C-146** | Likelihood = **hurdle-NB** (not ZINB) — reused `cls` learns marginal `P(y>0)` = the hurdle gate; ZINB needs a structural π → mis-spec | 02_design D1/§0 |
| **C-141, D-08** | **One joint hurdle-NB NLL per target** — a class-weighted Bernoulli gate-term *replaces* focal + a truncated-NB body; both are NLLs → additive → **reg-vs-cls balancer dissolved**; cross-target sum equal-weight/frozen | 02_design D2/§2 |
| **C-140, D-09** | Emit **`log1p(E[y])`** (the existing `expm1` inverse recovers `E[y]`); count-space emit would double-`expm1` | 02_design D3/§4 |
| **C-145** | θ = a loss-owned **`Parameter`**, not a head channel (preserves `input_channels==3×output_channels`) | 02_design D4/§1 |
| **C-148** | Deleted "dissolves by construction"; the 36-step `diagnose_io_gain` **explosion-check is the load-bearing test** | 02_design D5/§6 |

These concerns **remain OPEN until the implementing sub-issue's tests pass** — the decision mitigates the *design* risk; the *code* must honour it. Downstream acceptance (C-142 probe-validation, C-147 π-reliability, C-149 QS99-binding, C-150 PIT/PPC) is tracked on #101/#102 + `05_analysis_plan`.

---

### C-141: ZINB does NOT dissolve the balancer if the focal classification loss is kept separate

| Field | Value |
|-------|-------|
| ID | C-141 |
| Tier | 2 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Trigger | Training the ZINB head (#99/#100/#102) with the focal `by_*` loss retained as a separate term routed through `MultiTaskLoss` alongside the ZINB regression NLL |
| Location | dossier `02_design.md §2`; `views_hydranet/utils/mtloss.py:39-73`; epic #97 premise; closed issue #59 |
| Cross-refs | C-111 (the balancer regression), C-139 (the pivot premise), D-08 |

The epic's load-bearing claim is that ZINB "**dissolves the balancer / C-111 by construction**." But the design keeps **focal on `by_*` as a separate loss**, and `mtloss.py` still stacks regression + classification losses → the Kendall balancer **still runs over two families** → C-111 instability (and its seed-fragility) is **not** removed. Closing #59 as "mooted" is premature **unless π is trained inside the ZINB NLL** (one unified likelihood, no separate focal). Decide unified-NLL vs two-losses **before #99/#100**. **Tier 2:** an unaddressed wrong premise reintroduces the exact instability the pivot was meant to escape.

---

### C-143: train/inference objective mismatch — the composed `(1−π)·μ` is never scored in training

| Field | Value |
|-------|-------|
| ID | C-143 |
| Tier | 2 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Trigger | Emitting/feeding back `(1−π)·μ` at #101 while π is trained only by focal and μ only by the NB body — no training loss scores the composed `E[y]` |
| Location | dossier `02_design.md §2/§4`; `views_hydranet/utils/hydranet_inference.py:267` |
| Cross-refs | C-137 (count-head calibration risks), C-141 |

π and μ are optimized independently; the emitted and fed-back forecast is their **product**, which no loss sees during training. The composed `E[y]` can be miscalibrated (and can feed instability into the rollout) even when each head trains cleanly. Either score the composed `E[y]` in training, or document the product as an inference-time construct with **no calibration guarantee** and judge it on positive-subset proper scores. **Tier 2:** structural — calibration of the shipped quantity is not guaranteed by construction.

*Method side (expert-method-review Pass-2, M5):* monitor a **proper score on the issued `E[y]`** — a count-space CRPS and/or PIT — during training, not just the per-head losses (Gneiting: score the predictand you actually ship).

---

### C-144: no validator forbids `hurdle_threshold` × `output_distribution="hurdle_nb"`

| Field | Value |
|-------|-------|
| ID | C-144 |
| Tier | 3 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Trigger | A config that sets both `hurdle_threshold` (the C-45 mask) and `output_distribution="hurdle_nb"` |
| Location | `views_hydranet/utils/config_initializer.py` (loss/hurdle validators ~532-555); `views_hydranet/train/training_engine.py:234` (`use_latent` bypasses the C-45 branch) |
| Cross-refs | C-141 |

ZINB self-handles the zero/positive split; combining it with the C-45 `hurdle_threshold` mask is contradictory, and if ZINB sets `needs_latent=True` the C-45 branch is silently bypassed anyway. No validator currently forbids the combination → silent double/dropped masking. Add a validator (mirroring the existing tobit+hurdle guard) before #99/#100. **Tier 3:** config-coherence; currently unguarded.

---

### C-145: θ (NB dispersion) as a model head channel would break the architectural invariant

| Field | Value |
|-------|-------|
| ID | C-145 |
| Tier | 3 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Trigger | Implementing θ as a model head channel/output (rather than a loss-owned `Parameter`) at #100 |
| Location | `ModelOutput` (`views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:10-27`); `views_hydranet/utils/config_initializer.py:233` (`input_channels==3×output_channels` invariant); dossier `02_design.md §1` |
| Cross-refs | C-137 (M-Z7 spatial-θ ablation) |

A per-target θ head would break the invariant `input_channels==3×output_channels` and the feedback shape. The MVP θ is a learnable scalar → implement it as a **loss-owned `Parameter`** (like `LogNormalFixedSigmaLoss`'s sigma), not a head; document this so a later spatial-θ ablation (C-137/M-Z7) knows it must add a head deliberately. **Tier 3:** architecture coupling / maintainability.

---

### C-146: likelihood conflation — "ZINB" vs "hurdle-NB" are different models

| Field | Value |
|-------|-------|
| ID | C-146 |
| Tier | 2 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Trigger | Implementing `ZINBLoss` (#99) without first committing to ONE likelihood and writing its exact NLL |
| Location | dossier `2026-06-10_zinb_distributional_head_dossier/02_design.md §0/§2`; issue #99 |
| Cross-refs | C-137 (count-head likelihood-spec), D-08 (unified-NLL decision) |

The design names the head both "**ZINB**" (zeros from a Bernoulli gate **and** the NB's own zero mass — Lambert 1992) and "**zero-truncated NB on positives / hurdle_nb**" (zeros **only** from the gate, truncated positive body — Cragg 1971 / Mullahy 1986). **These are distinct likelihoods** with distinct NLLs and identifiability: in ZINB a zero has two explanations → π and the NB zero-prob are partially confounded; the hurdle factorizes cleanly but needs the truncated-NB normaliser. Implementing the wrong NLL for the intended model is a silent spec error (wrong gradients, wrong calibration). **Commit to one and write its exact NLL before #99.** **Tier 2:** structural mis-specification feeding everything downstream.

---

### C-148: "softplus dissolves the autoregressive explosion by construction" is unproven

| Field | Value |
|-------|-------|
| ID | C-148 |
| Tier | 2 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Trigger | Treating C-113 as solved / soft-pedaling the #102 explosion-check on the strength of the "by construction" prose |
| Location | dossier `00_README.md`, `02_design.md §0` |
| Cross-refs | C-142 (the gate), C-113 (the explosion), C-151 (the empirical post-run clamp-confound sibling), C-152 (the load-bearing-analogy sibling) |

The claim that softplus `E[y]` feedback dissolves the C-113 runaway "**by construction**" is unproven and contested by dynamical-systems theory (Mikhaeil 2022 / Hess 2023 / Durstewitz: the blow-up is a property of the recurrent **operator's gain** / Jacobian spectral radius, not the output nonlinearity alone). `02_design §6` correctly gates on `diagnose_io_gain`, but the "by construction" prose elsewhere invites skipping the gate. Delete the over-claim; treat the explosion-check as the **load-bearing** test. **Tier 2:** an over-claim that, if believed, wastes the build and re-explodes.

---

### C-149: NB upper tail likely under-fits the heavy conflict tail (QS99 the binding guardrail)

| Field | Value |
|-------|-------|
| ID | C-149 |
| Tier | 3 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Trigger | Reading a QS99 (tail) guardrail miss at eval (#102/#103) as a whole-model failure rather than an expected NB-tail limitation |
| Location | dossier `05_analysis_plan.md` |
| Cross-refs | C-137 (Tweedie/tail escalation) |

Conflict fatality counts are heavy/long-tailed (near power-law); the NB tail decays roughly geometrically and likely under-predicts extremes → **QS99 (tail sanity) is the most probable binding guardrail failure.** Pre-register that expectation + the Tweedie / GPD-tail escalation path, so a tail-miss triggers the right escalation rather than abandoning the model. **Tier 3:** anticipated limitation with a defined escalation.

---

### C-150: analysis plan lacks PIT + positive-tail posterior-predictive check

| Field | Value |
|-------|-------|
| ID | C-150 |
| Tier | 4 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Trigger | Finalizing the #102 eval readout without a PIT histogram + a positive-count PPC in the analysis plan |
| Location | dossier `05_analysis_plan.md` |
| Cross-refs | C-136 (MCR-not-proper) |

The plan has Coverage + F-zero-rate + multi-seed (good) but no **PIT calibration histogram** and no **posterior-predictive check on the positive-count distribution**. A ZINB/hurdle can match the 95% zero-rate while mis-fitting the positive body; PIT + a positive-tail PPC catch that. Add both to the readout. **Tier 4:** analysis-completeness; improves diagnostic value, no correctness impact.

---

### C-152: ADR-061's "why now" rests on a literature analogy (El Jurdi prior-loss CoordConv) that may not transfer

| Field | Value |
|-------|-------|
| ID | C-152 |
| Tier | 3 |
| Source | external ADR review + verify-first reflection (coordinate-grounding session, 2026-06-13) |
| Trigger | Reading a coordinate-channel **null / ambiguous** experiment result as "CoordConv doesn't work" rather than "the El Jurdi analogy didn't hold"; OR over-weighting the literature in the coords go/no-go instead of the §5 pre-registered falsifier |
| Location | `docs/ADRs/active/061_coordinate_channels.md` §3 + C2; `reports/2026-06-11_coordinate_grounding_dossier/01_literature.md` |
| Cross-refs | C-148 (the "by construction" prose over-claim), C-151 (the empirical clamp confound), ADR-061 §3/§5 |

ADR-061's "why now" leans on El Jurdi et al. (2021): CoordConv-Unet stabilizes training and evades local minima **under prior-based losses** — and we train a prior-based likelihood (hurdle-NB). The original ADR text said this was *"exactly the regime"* CoordConv was found to help. **That equivocates on "prior":** El Jurdi's "prior" is an **added spatial/shape regularizer** (size/clDice-type) bolted onto a pixel-wise base loss, and CoordConv's role was stabilizing that **two-term interchange**. A hurdle-NB is a **distributional likelihood family** — there is no added spatial term and no equivalent interchange — so the mechanism may simply not exist in our setting. The ADR/dossier text has been **corrected** to "plausibly analogous, not identical," with the disanalogy explicit. The **residual risk** is decision-hygiene: a load-bearing justification that may not transfer could (a) inflate confidence going into the coords experiment, or (b) cause a null/ambiguous result to be mis-attributed to "CoordConv fails here" when the real lesson is "the analogy didn't hold" (→ premature escalation, or wrong placement/ablation conclusions).

**Mitigation:** the §5 pre-registered experiment + its falsifier are the arbiters, **not** the analogy; on a null, ablate placement (the unbacked input+top-skip choice — see dossier `04`) and re-check the gate-forensic before concluding CoordConv is the wrong lever. **Tier 3:** methodology / decision-hygiene; no silent corruption, clear trigger. Already de-risked by the §3 text correction — registered so the analogy isn't silently re-promoted to "received wisdom" as the design hardens.

---

### C-156: `feature_cols` overloads model-inputs and training-targets (root of C-157/158/159)

| Field | Value |
|-------|-------|
| ID | C-156 |
| Tier | 2 |
| Source | channel-role side-quest — multi-expert review + census-by-test (2026-06-13) |
| Trigger | Adding any channel that is a model **input** but not a training **target** (coordinates, and the future covariates), OR running with `static_channels` non-empty |
| Location | `volume_handler.py:159` (`kept_feature_cols = features + static_channels`); consumed at `curriculum.py:45`, `training_engine.py:435`, `train_model.py:75-85` |
| Cross-refs | C-157 / C-158 / C-159 (its three faces), C-36 (the Custodian it lives in), C-153 (the seam), ADR-060, ADR-062 |

`feature_cols` carries **two roles at once** — "channels fed to the model" *and* "channels the model predicts/trains on." For the bounded baseline those sets were identical, so the overload was invisible; the static-channel seam (#108) broke the identity but only some consumers were taught the difference. The result is one defect with three faces (C-157/158/159) plus a re-break for every future input-only channel. Pinned empirically in `tests/test_channel_role_census.py`. **Mitigation:** ADR-062 §2.1 gives roles a first-class home (`model_input_cols` / `target_cols` / `static_cols`); resolves when that lands and the census `xfail`s flip to XPASS. **Tier 2:** structural fragility, clear trigger, recurs on realistic change.

**Status (2026-06-20) — FUNCTIONALLY RESOLVED; final alias flip DEFERRED.** #115 4a/4b shipped the role accessors (`model_input_cols`/`target_cols`/`static_cols`) + `tensor_cols` (the de-overloaded kept-channels reader), rewired the kept-channel consumers (`to_pytorch`, training/inference `feature_names`) to `tensor_cols`, and fixed all three faces — **census C-157/158/159 are green**. The remaining step (flip the `feature_cols` accessor to a pure `model_input_cols` alias) is **cosmetic hygiene, not a functional unblock**: pre-flip, `feature_cols` consistently means kept-channels and every live reader treats it as such, so there is **no active mismatch** — coords are unblocked now. The flip is the single widest edit (audit found ~9 consumer reads needing a coordinated switch to `tensor_cols` — `volume_handler` ×5 reconstructions, `volume_sampler:106`, `feature_scaler:255`, `hydranet_inference:422`, `volume_handler:718`, plus `curriculum:51` re-touch, `visual_diagnostics:437`/C-166, `wrap_predictions` output-side roles, the 2 `test_channel_role_accessors.py` alias `xfail`s, and the VolumeHandler/Curriculum CICs). Deferred deliberately as the riskier action whose byte-identity net is no-coords-only; the 2 `#114` alias tests stay `xfail` until it lands. Remaining open facet of C-156 is **this flip only**.

---

### C-160: the channel-role refactor activates the VolumeHandler god-node blast radius

| Field | Value |
|-------|-------|
| ID | C-160 |
| Tier | 2 |
| Source | channel-role side-quest planning (2026-06-13) |
| Trigger | Executing the Phase-4 channel-role refactor / VolumeHandler decomposition — any signature change on the Custodian |
| Location | `volume_handler.py` (whole class); the 8 creation sites; the role consumers |
| Cross-refs | C-36 (451-edge god node), C-37 (no Protocol — being reconsidered), C-75 (derivation duplication), ADR-062 |

ADR-062 deliberately refactors the Custodian, which C-36 quantifies as a **451-edge god node bridging 16+ communities** — "any signature change ripples across all communities." The risk: an unintended change to baseline (no-coords) output slips past the unit suite (which gave false confidence before). **Mitigation (the side-quest's defining discipline):** an end-to-end **parity gate** — a no-coords run on current code must be **bit-identical** to a no-coords run on the refactored code (Phases 2/5) — plus the characterization net (Phase 3) and byte-identical-when-off (I5) at every step. Registered so the refactor is executed under that gate, never on the unit suite alone. **Tier 2:** structural fragility under a planned, broad change.

---

### C-163: No runtime resource/environment harness — runs OOM/wedge/grind silently (the missing guarded-run)

| Field | Value |
|-------|-------|
| ID | C-163 |
| Tier | 2 |
| Source | expert-code-review (meta, 2026-06-18) — Nygard/GoF; corroborated by this session's lost week |
| Trigger | Launching any real GPU run (`main.py -t -e -re`) on the dev box — no pre-flight or runtime gate exists, so a wedge/OOM/hang is detected only by a human hours later |
| Location | run launch path (`views-models/models/violet_visitor/main.py` invocation); the deleted `views-models/scripts/run_balancer_frozen_gpu.sh` (gone — see C-115) |
| Cross-refs | C-115 (CUDA gate gone), C-116 (publish OOM), C-135 (eval OOM), Cluster 6 |

The repo is *operated like production* (90-min GPU jobs, external services, hard 32 GB limit) with **zero runtime reliability engineering**. This session, three distinct failures each ground/OOM'd **silently** because nothing gated them: a lid-suspend wedged CUDA → multi-hour CPU grind (no health gate); `wandb run.finish()` DNS-hung 38 min (no timeout/circuit-breaker); the publish step OOM'd (no RAM budget/backpressure). Each guard was **re-improvised ad hoc** (systemd-inhibit, RSS sampler, CUDA pre-flight) — and the canonical guarded-run that once held them (`run_balancer_frozen_gpu.sh`) has been deleted. **Tier 2:** structural/operational fragility with a clear trigger (every run), causing repeated multi-hour losses. **Fix:** one reusable guarded-run script — CUDA `is_available()` pre-flight (abort), RAM-headroom budget (abort pre-write), disk check, `systemd-inhibit` (block suspend), **unbuffered** live logs, peak-RSS logging, fast-fail-on-no-GPU-within-N-min. Promote resource "watch" entries (C-116) into this gate.

---

### C-164: No end-to-end output-path seam test (peak-RSS vs samples) — memory regressions ship invisibly

| Field | Value |
|-------|-------|
| ID | C-164 |
| Tier | 3 |
| Source | expert-code-review (meta, 2026-06-18) — Feathers/Beck/Kleppmann |
| Trigger | Any change to the eval→invert→assemble→publish output path or `n_posterior_samples` — no test exercises the seam or asserts memory scaling |
| Location | `views_hydranet/utils/inference_orchestrator.py`, `feature_scaler.py:inverse_transform_volume`, `prediction_frame_assembler.py`; CI `tests/` |
| Cross-refs | C-116 (the regression this would have caught), C-113 (analogous "no seconds-level seam test" gap, register §C-113) |

The CIC/unit suite tests components in isolation; **every regression this session lived in an untested seam** (eval→publish memory, model→artifact-reload, process→GPU). The C-116 memory regression scaled with `n_posterior_samples` for ~2 weeks invisibly because **no test asserts peak-RSS as a function of S** on the output path. The human had to act as the integration test. **Tier 3** (test-coverage/maintainability gap — but high-value: it directly enabled a Tier-2 failure). **Fix:** a fast end-to-end fixture (1 origin, 2 lessons, saved artifact or tiny CPU grid) through the full output path that asserts peak RSS is **sub-linear in `n_posterior_samples`** (i.e. S is collapsed before the heavy step). This is the single test that catches the C-116 class in seconds. Pair with a fast feedback loop (eval-only on a saved artifact; unbuffered logs) so output-path iteration is minutes, not 90.

---

### C-165: CI "green" is misleading — `--ignore` flags mask real breakage (#95 collection error)

| Field | Value |
|-------|-------|
| ID | C-165 |
| Tier | 3 |
| Source | expert-code-review (meta, 2026-06-18) + falsify P4 (2026-06-16) |
| Trigger | Reading "CI green" / "full suite green" as a health signal, or relying on it as the refactor safety net (#114/#115 DoD) |
| Location | `.github/workflows/ci.yml` (6 `--ignore` flags); `tests/test_eval_integration_toy.py` (#95 stale import → collection error) |
| Cross-refs | #95 (stale import), C-116/C-164 (false confidence theme) |

CI passes only because `ci.yml` `--ignore`s six files; one of them, `test_eval_integration_toy.py`, is a **real collection error** (#95, stale `views_evaluation.evaluation_manager` import) — `pytest tests/` errors without the flags. So "suite green" means "green modulo silently-excluded breakage," and the channel-role refactor's "full suite green" gate (#114/#115) rests on it. **Tier 3:** false-confidence / maintainability. **Fix:** resolve #95 so a plain `pytest tests/` collects clean, or make the ignore-set explicit and justified in the DoD; distinguish "aspirational falsification stubs" (legitimately ignored) from "broken tests" (must fix).

---

### C-166: diagnostic plots show input-only statics as predicted signal (benign display drift)

| Field | Value |
|-------|-------|
| ID | C-166 |
| Tier | 4 |
| Source | review-diff (#115 4b·biopsy, 2026-06-20) + census suspectA (`test_channel_role_census.py`) |
| Trigger | Enabling `diagnostic_visualizations` with `static_channels` non-empty, then reading the Stage-5 biopsy / `_select_display_channels` plots to interpret per-channel signal — the static (geometry) channels appear as if they were predicted targets |
| Location | `views_hydranet/utils/visual_diagnostics.py` (`_select_display_channels`, ~:437) |
| Cross-refs | C-156 (root — `feature_cols` overload), C-157 (the crash face, now fixed), C-118 (visual_diagnostics module), ADR-062 §2.1 |

A fourth, **benign** face of the C-156 overload: `_select_display_channels` derives "interesting channels" from `feature_cols`, which now includes input-only statics (CoordConv row/col). The diagnostics therefore plot geometry as if it were model signal. No crash, no model-output or training impact (the C-157 crash face is fixed; this is display-only), but it can mislead a researcher reading the biopsy plots. **Tier 4:** cosmetic/interpretation, no correctness or reliability impact. **Fix:** select display channels from `target_cols` (or exclude `static_cols`) once the role accessors are the single source — natural tidy-up alongside the flip commit or Phase-6 harden. Pinned (CLASSIFY, non-xfail) by `test_census_suspectA_visualdiagnostics_static_classification`.

---

### C-167: no spatial-sharpness / resolution metric — evaluation is calibration-only (resolution-blind)

| Field | Value |
|-------|-------|
| ID | C-167 |
| Tier | 2 |
| Source | expert-method-review (2026-06-20, Gneiting/Gelman seats) |
| Trigger | Declaring any head/loss/architecture change "works" on the in-sample over-smoothing using calibration metrics (Brier/ECE/MCR) without a sharpness/resolution score |
| Location | eval stack (`views_evaluation` native calculators); `scripts/gate_reliability.py`; dossiers `05_analysis_plan` |
| Cross-refs | C-147 (calibration check that was resolution-blind), C-150 (PIT/PPC — distributional, not spatial), C-126 (rollout readout sharpness=MCR/zero-rate conflation) |

The current evaluation measures **calibration** (Brier/ECE/reliability, C-147) and **magnitude** (MCR/zero-rate, called "sharpness" but it is not spatial sharpness). None of these can see the visually-obvious defect: the in-sample, teacher-forced prediction is **spatially over-smooth** (diffuse blobs vs sharp sparse truth) — exactly why C-147's low ECE wrongly read as "gate fine." A proper **CRPS** (sharpness-subject-to-calibration, Gneiting & Raftery 2007), the **Brier resolution term** (Murphy 1973 decomposition), and a **spatial posterior-predictive statistic** (hot-cell count / blob-size / spatial autocorrelation; cf. Fractions Skill Score, Roberts & Lean 2008 — to fetch) are all absent. **Tier 2:** a resolution-blind audit will certify a spatially-useless model as fine (already caused one wrong conclusion); fix by building the sharpness instrument (EXP-1) before iterating on the head/loss.

**Update 2026-06-21 — instrument BUILT; resolution-blindness closed, with a caveat (folds panel M-1).** The sharpness/proper-score instruments now exist and are unit-tested: `scripts/sharpness_scorecard.py` (FSS/area-ratio) and `scripts/proper_score_audit.py` (CRPS + SCRPS + threshold-weighted CRPS + randomized-PIT, scored on **all** cells; 6 tests; validated — it reproduces the FSS scorecard before its new scores are trusted). Dossier `2026-06-21_proper_score_gate_dossier/`. **Caveat that keeps a thin residual open:** the cheap FSS/area-ratio metric **overstates** the defect (~10× area-ratio vs ~1.4–3.4× on CRPS) and is blind to the zero-vs-positive split — so decide head/loss changes on the **proper scores**, with FSS only corroborating. Largely addressed; do not select on FSS magnitude alone.

---

### C-168: hurdle truncation removes dense per-cell supervision — plausible root of the in-sample over-smoothing; untested vs full ZINB

| Field | Value |
|-------|-------|
| ID | C-168 |
| Tier | 2 |
| Source | expert-method-review (2026-06-20, Harrell/Bishop seats) + diagnostic plots |
| Trigger | Committing further to the truncated-NB body without comparing against a full ZINB/NB likelihood scored on **all** cells (the dense-supervision EXP-2) |
| Location | `views_hydranet/utils/truncated_nb_loss.py` (`positive = raw_y > 0`); dossier `2026-06-10_zinb_distributional_head_dossier/02_design §2` |
| Cross-refs | C-146 / C-145 (hurdle-vs-ZINB decided on calibration/identifiability, NOT sharpness), M-Z6, C-147, [[project_bloom_is_feedback]] |

The truncated-NB body computes loss **only on `y>0` cells**, so the ~99.7% zero cells give the magnitude head **no gradient** → it smoothly interpolates between events (a conv net's natural low-frequency fill). The old shrinkage loss penalised **every** cell densely → sharp suppression. The hurdle-vs-ZINB choice (C-146) was made on identifiability/calibration grounds and **never examined through the sharpness lens**; a full ZINB/NB scored on all cells supervises zeros (the `P(y=0)` term) and is **still a proper likelihood** — and the in-domain prior (Iacus 2025 DynAttn, ZINB-on-VIEWS) used exactly that and won. **Tier 2:** a structural likelihood choice plausibly causing the over-smooth model output, untested against the proven principled alternative; resolve via EXP-2 (dense-supervision, one-variable, teacher-forced).

**Update 2026-06-21 — over-smoothing CONFIRMED real and likelihood-side; mechanism sharpened (proper-score gate, one seed).** Honest scoring of the count/hurdle run vs a shrinkage run at the out-of-sample first step (`proper_score_audit.md`): the shrinkage model **wins CRPS and twCRPS@1 on all three targets** and full-distribution calibration (PITnc-all 0.001 vs 0.179) — so the smear is *not* a metric artifact, a proper score agrees. **But the truncation framing is only part of it:** the earlier dense-NB experiment (RUN-3) showed dense supervision *tamed magnitude without sharpening*, and here the precise defect is that the hurdle/count model **leaks ~0.14 predicted mass onto truly-zero cells** (no clean zero; MCR-all sb 1.028 achieved by smearing). So the lever is the **zero-mass / clean point-mass-at-0 handling** (a sharper gate or a coherent zero-inflated mixture with a learned π₀), not truncation-vs-dense per se. The gap is ~1.4–3.4× (CRPS), not the ~10× FSS implied; the count head is actually *better* on positive-cell magnitude calibration. One seed/origin set — confirm before closing.

**Update 2026-06-22 — multi-seed confirmation; the binding axis is MAGNITUDE CALIBRATION (MCR), not sharpness.** Count (hurdle-NB) vs sharp (shrinkage) ×3 seeds {42,4,7}, pipeline metrics @ teacher-forced T=0 (`reports/2026-06-22_body_multiseed_dossier/results.md`; matches `proper_score_audit` exactly). The body is confirmed the lever, but the binding *unsolved* axis is **MCR**, not sharpness — **neither body lands near MCR=1**: count **over**-fires (MCR 1.5–6.5×, worst on os) while sharp **under**-fires (MCR≈0.03, a near-zero collapse). **Neither body is FAO-eligible** under the locked criteria ([[reference_fao02_locked_eval_framework]]): sharp wins CRPS but fails the QS99 (tail) + MCR (under) guardrails — the timid prophet (see C-126); count fails CRPS superiority + MCR (over). New reliability observation: here the **count** body is the seed-**volatile** one (CRPS CV 0.17; os MCR ±2.0) — the historical "shrinkage is volatile" did NOT reproduce (shrinkage CV 0.004, but that is likely *stably-collapsed*, not robust). **Next lever:** a body whose magnitude lands near MCR=1 without smearing (count) or collapsing (sharp).

---

### C-169: loss-vs-architecture confound — in-sample smoothness may be the ConvLSTM/U-Net low-pass, not the loss

| Field | Value |
|-------|-------|
| ID | C-169 |
| Tier | 3 |
| Source | expert-method-review (2026-06-20, Shi dissent) |
| Trigger | Redesigning the likelihood/loss to fix sharpness before isolating whether the convolutional low-pass (blur-tolerant objective) is the cause |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (conv encoder/decoder, up/down-sampling); the loss stack |
| Cross-refs | C-168 (the competing loss hypothesis), C-113 (distinct: rollout *magnitude*, not spatial smoothness) |

Conv + up/down-sampling is a spatial low-pass; MSE/likelihood objectives tolerate blur — the well-known ConvLSTM "blurry nowcast" problem (the precipitation-nowcasting → DGMR/Ravuri 2021 lineage, to fetch). The old model may have been sharp because shrinkage's per-cell penalty *fought* the blur, not because the backbone is sharp. If EXP-2's dense supervision does **not** sharpen, the cause is architectural/objective (adversarial or sharpness-aware score), and likelihood redesign is wasted effort. **Tier 3:** attribution/methodology gap — misdirects effort if unbroken; no corruption. EXP-2's decision rule (sharpens→loss; stays smooth→architecture) is the discriminator.

**RESOLUTION CANDIDATE 2026-06-21 (proper-score gate; direction = the LOSS; one seed — confirm multi-seed before closing).** This is the panel's M-2, now broken in favour of the loss/objective, not the architecture: (a) a single-tile overfit of the **real** backbone with a plain per-cell MSE reproduced a sharp sparse field **exactly** (MSE→1e-6, area-ratio 1.00×, peak-recovery 1.00, 7/7 cells — `overfit_capacity.png`) ⇒ the conv/U-Net + ConvLSTM backbone CAN represent sharpness; and (b) a single MC-dropout draw is **no sharper** than the 8-draw mean (count sb area 2.97× vs 3.26×) ⇒ not an averaging low-pass either. So no backbone/adversarial redesign is needed; the lever is the likelihood (see C-168 update). `scripts/single_window_overfit.py`, dossier `07_results.md`. Kept Open pending a multi-seed/real-window confirmation per the no-hard-verdict discipline.

---

### C-170: Un-freezing the multi-task balancer to fix ns/os under-firing can STARVE the rare targets (Gaussian homoscedastic assumption on heavy-tailed counts)

| Field | Value |
|-------|-------|
| ID | C-170 |
| Tier | 2 |
| Source | expert-method-review (2026-06-21, Kendall seat) |
| Trigger | Setting `freeze_multitask_balancer=False` (or otherwise re-enabling Kendall homoscedastic weighting) to fix the `lr_ns`/`lr_os` under-firing, without logging per-task `log_var` trajectories and a σ-divergence stop rule |
| Location | `views_hydranet/utils/mtloss.py` (`MultiTaskLoss.forward`, the `1/((is_regression+1)·σ²)` coefficients); `views_hydranet/train/training_engine.py:116` (`freeze_multitask_balancer` gate) |
| Cross-refs | C-124 (balancer regularisation choice), C-113/C-111 (the explosion the freeze was for — distinct axis: rollout, not step-0 weighting), C-112 (pre/post comparability), [[project_proper_score_gate_finding]] |

The frozen balancer (`log_vars=0`) sums the 6 task losses with fixed weights (regression ×0.5, classification ×1.0), equal-weighting `lr_sb`/`lr_ns`/`lr_os` despite their very different scales — the plausible cause of the rare targets (ns 99.92% / os 99.88% zero) under-firing. The obvious fix is to un-freeze the Kendall (2018) homoscedastic balancer. **But its derivation assumes a Gaussian (homoscedastic) task likelihood**, and the targets are extreme heavy-tailed counts (kurtosis 962–15,543). On such a task the rare-event residuals are huge, so the balancer reads high task *noise* and learns a **large σ_ns/σ_os → coefficient `1/2σ²`→0 → it DOWN-weights the already-starved rare tasks** — the opposite of the intended fix, and silent (visible only as degraded ns/os metrics). Mitigation (pre-register before any unfreeze): log per-task `log_var` trajectories; a stop rule (freeze back if σ_ns or σ_os exceeds σ_sb by >~4× mid-training); try the cheaper static `target_weights` probe first; judge at step-0/T=0 decoupled from the C-113 rollout question. **Tier 2:** a realistic next change (the ns/os fix) that can structurally worsen model output on the rarest, most policy-relevant targets.

---

### C-171: FocalLoss docstring falsely claims it "reduces to BCE when gamma=0 and alpha=0.5" (it is 0.5·BCE)

| Field | Value |
|-------|-------|
| ID | C-171 |
| Tier | 4 |
| Source | falsify (2026-06-23, gate-loss audit, probe P2) |
| Trigger | A developer reads the docstring and swaps `focal`↔`bce` (or sets α=0.5 expecting a BCE-equivalent gate) assuming equal loss *scale*, without checking the α constant factor |
| Location | `views_hydranet/utils/focal_loss.py:13-15` (class docstring) |
| Cross-refs | C-172 (same file, same audit), [[project_gate_loss_finding]] |

The docstring states FocalLoss "reduces to Binary Cross Entropy (BCE) when gamma=0 and alpha=0.5." Verified false: at α=0.5 the `alpha_t` factor is a constant 0.5, so `focal(γ=0, α=0.5) == 0.5·BCE` (probe P2: ratio exactly 0.5000). True BCE-equivalence needs α disabled (α<0) **and** γ=0. The **computed value is correct** — focal matches `torchvision.ops.sigmoid_focal_loss` exactly (probe P1) — so this is a documentation defect, not a math bug. Practical edge: because α scales the classification loss magnitude, a swap made on the false premise of equal scale would silently shift the multi-task reg-vs-cls balance. **Tier 4:** code-quality/doc inaccuracy, no production correctness impact (gate sweep used `reduction='mean'`, α∈{0.25,0.75}, where the formula is verified). Fix: correct the docstring. Failing stub: `tests/test_falsify_gate_losses.py::test_focal_docstring_bce_equivalence_is_accurate`.

---

### C-172: FocalLoss internal `unsqueeze(0)` leaks a leading dim under `reduction='none'`

| Field | Value |
|-------|-------|
| ID | C-172 |
| Tier | 4 |
| Source | falsify (2026-06-23, gate-loss audit, probe P5) |
| Trigger | Any future use of `FocalLoss(reduction='none')` for per-cell/masked/spatially-weighted classification loss (e.g., a hurdle_threshold-style mask on the gate, or a weight-head) — the `[1,*input]` output silently broadcasts against an `[*input]` mask |
| Location | `views_hydranet/utils/focal_loss.py:44` (`logits, targets = logits.unsqueeze(0), targets.unsqueeze(0)`) |
| Cross-refs | C-171 (same file, same audit), [[project_gate_loss_finding]] |

`FocalLoss.forward` unsqueezes a leading dim ("matches expected pipeline volume format"), so with `reduction='none'` it returns shape `[1, *input]` instead of `[*input]` (probe P5: input `(4,8,8)` → focal `(1,4,8,8)`; `WeightedBCEWithLogitsLoss` correctly preserves `(4,8,8)`). Harmless under `mean`/`sum` (the scalar is unaffected — production + the gate sweep use `mean`), but it is a latent contract inconsistency: focal is the only loss whose `none`-mode output rank differs from its input, so any per-cell weighting code that works for the other losses would silently mis-broadcast with focal. **Tier 4:** no current correctness impact; latent shape-contract trap. Fix: drop the internal `unsqueeze`. Failing stub: `tests/test_falsify_gate_losses.py::test_focal_reduction_none_preserves_input_shape`.

---

### C-173: Channel ROLE is inferred from the column-name prefix — silent mis-classification of incoming columns

| Field | Value |
|-------|-------|
| ID | C-173 |
| Tier | 2 |
| Source | expert-code-review (2026-06-24, naming-coupling review) |
| Trigger | Ingesting a new feature/covariate whose column name begins with a reserved prefix (`by_` / `lr_` / `pred_`) but is NOT that role — e.g. a covariate named `by_region` — with no explicit external→role mapping to override the prefix heuristic |
| Location | `views_hydranet/utils/data_sniffer.py:362` (`startswith("by_")` → "generated binary"); `views_hydranet/utils/feature_scaler.py:228` (`startswith(BINARY_PREFIX)` → skip inverse-transform), `:235` (`removeprefix(PRED_PREFIX)`); `views_hydranet/utils/volume_handler.py:22` (`BINARY_PREFIX`/`PRED_PREFIX`) |
| Cross-refs | C-160 (channel-role refactor — the fix vehicle: roles as data, not prefixes), C-174, C-120 |

The model derives a channel's **semantic role** (onset gate vs regression magnitude vs model-prediction) by **parsing the column-name prefix**, not from an explicit role declaration. A future incoming column whose name happens to start with a reserved prefix would be **silently** treated as that role — skipped from inverse-transform (`feature_scaler`) or treated as a generated binary target (`data_sniffer`) — with **no error**, producing wrong outputs. The user explicitly flagged "other features coming in at some point," which is exactly the triggering scenario. **Tier 2:** silent mis-classification (no error signal) under a realistic, anticipated change (new features); becomes Tier-1-like if it ever fires on a real covariate. Fix: resolve role from an explicit boundary schema (external name → {role, target_id, transform}); consume roles, not prefixes (aligns with the ADR-062 channel-role refactor, C-160).

**views-frames migration note (source-checked 2026-06-24, `views-frames/src/views_frames/feature_frame.py`):** the planned `FeatureFrame` does NOT fix this. It carries `feature_names: list[str]` (flat, no per-feature role), `FrameMetadata` (model/run_type/timestamp/seed — frame-level provenance only), and `SpatialLevel` (cm/pgm). It is a deliberately thin typed container (ADR-011/013), so **role stays implicit in the name** — the prefix parsing survives, just reading `frame.feature_names`. ⇒ the fix is hydranet's own ingestion boundary (C-160 + a `feature_names→role` adapter), NOT the migration, and is NOT blocked on upstream.

---

### C-174: Column-name string is the unmediated contract + join-key + role — no boundary adapter ⇒ rename/new-feature ripple

| Field | Value |
|-------|-------|
| ID | C-174 |
| Tier | 3 |
| Source | expert-code-review (2026-06-24, naming-coupling review) + user observation |
| Trigger | Any upstream column rename (e.g. views-models#151 → #136: `lr_*_best`→`lr_ged_*`) or a new target/feature added to the data — forcing coordinated edits across config target lists, `channel_map`, `feature_scaler`, `data_sniffer`, and ~465 test-fixture literals |
| Location | `config_initializer.py:39–42` (`regression_targets`/`classification_targets`) + `:252`/`:629`/`:666` (features/target_weights/loss_reg_sigma all keyed by the same name strings); `volume_handler.py:137–178` (`channel_map` from names); `tests/` (~465 hardcoded `lr_*`/`by_*` literals across 40 files) |
| Cross-refs | C-120 (dual data-layer authority, cross-repo), C-49 (flat config schema), C-160 (role refactor), io_format_landscape #138 (no unified I/O contract) / #140 (dual sniffers) |

The raw upstream column name is overloaded as (a) the cross-repo **contract**, (b) the internal **join key** (the same strings index `regression_targets`, `features`, `target_weights`, `loss_reg_sigma`, and `channel_map`), and (c) the **role** (prefix, see C-173). There is **no anti-corruption layer / single source of truth** at the data boundary, so the name propagates raw into every layer and into 465 test fixtures. Consequence: a single upstream rename is a high-friction, order-dependent, cross-repo edit (the real data rename #151 must precede hydranet's data-coupled edits) with breakage risk, and adding a feature is an edit (not an extension). **Tier 3:** maintainability/coupling raising change cost across repos and the test surface; no standalone correctness impact (the correctness facet is C-173). Fix: a thin boundary adapter mapping external names → internal `{target_id, role, transform}` (single source of truth), + a fixture factory so the vocabulary appears once.

**views-frames migration note (source-checked 2026-06-24):** the dataframe→views-frames switch is a CONTAINER change, not a naming-contract fix. `FeatureFrame` carries names + `SpatialLevel` (typed level) + `FrameMetadata` (typed provenance) — so the migration *does* decomplect **level** and **provenance** out of the name string (real wins, orthogonal to this entry), but **identity + role stay in the name** → this coupling survives. The adapter/single-source-of-truth fix lives at hydranet's ingestion boundary regardless of the migration; the migration just hands a cleaner validated `feature_names` to adapt from.

---

### C-175: lr_↔by_ pairing convention is unenforced ⇒ partial-rename / inconsistency (demonstrated)

| Field | Value |
|-------|-------|
| ID | C-175 |
| Tier | 3 |
| Source | expert-code-review (2026-06-24) + user observation |
| Trigger | Renaming or adding a regression target without its onset companion — as #136 did (`lr_*_best`→`lr_ged_*` while leaving `by_*_best`) — with no validator asserting magnitude↔onset pairing completeness |
| Location | `config_initializer.py` (no cross-validator pairing `regression_targets` ↔ `classification_targets` by base id); `tests/` (post-#136 state: `lr_ged_sb` alongside `by_sb_best`) |
| Cross-refs | C-174 (same root: no schema), C-173 |

The naming scheme encodes a paired convention (every magnitude `lr_<x>` has an onset `by_<x>`), but **nothing enforces it**. The #136 rename renamed only the `lr_` regression targets, leaving `by_*_best` — a half-renamed, inconsistent scheme that passed green because the suite has no completeness invariant. **Tier 3:** maintainability + latent config/data-mismatch risk (a true mismatch would likely be caught loudly by the existing `features==regression_targets` cross-checks, hence not Tier 2). Fix: a cheap schema-completeness test (every regression target has a matching onset target and vice-versa) — would have caught the `by_sb_best` inconsistency.

---

### C-176: views-frames PredictionFrame dropped the empty-frame (N=0 / S=0) input validation the old pipeline-core class enforced

| Field | Value |
|-------|-------|
| ID | C-176 |
| Tier | 3 |
| Source | review-diff + falsify-style API probe during S3 of the views-frames migration (#140, 2026-06-24); assembler-path coverage gap added from external code review (2026-06-24, finding #5) |
| Trigger | An evaluation/forecast rolling-origin where a target yields **zero valid cells** (e.g. an all-masked window, or a target with no `pred_` channel after a config change) — the assembler builds a `(0, S)` frame and it now flows downstream silently instead of raising at construction |
| Location | `views_hydranet/utils/prediction_frame_assembler.py:_reconstruct_as_pf_dict` (construction); leaf-level contract characterized in `tests/test_prediction_frame_suite.py::test_views_frames_accepts_empty_n_validation_relaxed` / `::..._empty_s_...` — but the **assembler's own** empty-mask path (`tests/test_prediction_frame_assembler.py`) is UNTESTED |
| Cross-refs | Migration epic #138 (S2 #137, S3 #140) |

The retired `views_pipeline_core` PredictionFrame **rejected** an empty frame (`N=0` rows or `S=0` sample columns) with a `ValueError`; hydranet relied on that loud failure as a backstop against a degenerate origin/target producing nothing. The `views_frames` leaf (1.3.0) **accepts** both — verified by direct probe. So an empty eval frame would now pass construction and reach downstream scoring/merge as a zero-row frame rather than failing fast at the boundary. **Tier 3, not 2:** in practice the PGM mask always has valid cells and S≥1, so the degenerate case is not expected on the live path — but the safety net that used to catch a misconfiguration is gone, and there is no hydranet-side guard replacing it. Two old "Red" rejection tests were converted to characterization tests (asserting the relaxed acceptance) so that a future views-frames tightening re-surfaces this for revisit. *Note: the integer-dtype requirement on `SpatioTemporalIndex` is strictly stronger than before (NaN-in-time now raises `TypeError` at index construction) — a net validation gain, orthogonal to this gap.* Candidate fix (deferred, not in migration scope): a cheap assertion at the assembler boundary that each produced frame has `N>0` and `S>0`, or an explicit decision to allow empty frames with a logged warning. **Coverage facet (external review #5):** the leaf's N=0 acceptance is now characterized, but `_reconstruct_as_pf_dict` driving an all-masked provider to a `(0, S)`/`(0, 1)` frame is not exercised by the assembler's own test suite — if a guard is added per the candidate fix, add the assembler-level empty-mask test alongside it.

---

### C-177: PredictionFrames are emitted with empty FrameMetadata — no model/run_type/seed/run_id provenance stamped

| Field | Value |
|-------|-------|
| ID | C-177 |
| Tier | 4 |
| Source | external code review of the #137 assembler migration (2026-06-24, finding #3) |
| Trigger | A downstream consumer (ensemble assembly, audit, or FAO-facing delivery) needs to attribute a persisted PredictionFrame back to the run that produced it (model, run_type, seed, run_id) — e.g. when merging the 8 ensemble members or tracing a forecast artifact |
| Location | `views_hydranet/utils/prediction_frame_assembler.py:_reconstruct_as_pf_dict` (`PredictionFrame(y_pred=, index=)` passes no `metadata=`) |
| Cross-refs | C-176 (same construction site); migration epic #138 |

`PredictionFrame(y_pred=, index=)` omits the optional `metadata=` argument, so every assembled frame carries an empty `FrameMetadata`. The `views_frames` leaf supports run identity (ADR-013; `run_id`/`data_version` added in v1.4.0 — repo is on 1.4.0 though pipeline-core 3.0.0 pins `^1.3`), and the assembler is the natural place to stamp `model`/`run_type`/`seed`/`run_id`. **Tier 4:** no correctness or reliability impact today (downstream does not yet rely on frame-level provenance), purely a missing-capability/quality observation. Explicitly **out of scope for #137** (the reviewer flagged it as a future enhancement). Candidate fix: thread run identity into `assemble_evaluation` and pass `metadata=FrameMetadata(...)` at construction once a downstream consumer needs it (likely with the ensemble-merge work).

---

### C-179: `reg_activation` is arch-affecting but NOT persisted in the artifact sidecar — silent activation mismatch on reload

| Field | Value |
|-------|-------|
| ID | C-179 |
| Tier | 2 |
| Source | /falsify "regression head + mask now 100% correct" round 2 (2026-06-26) — Finding A, SOFT |
| Trigger | Reload (eval/forecast/replay) a model trained with an **explicit `reg_activation` override**, or a pre-#178 relu-trained `hurdle_shrinkage`/`hurdle_lognormal` artifact, while the live config's activation default differs from training |
| Location | `views_hydranet/train/train_model.py:75-92` (`arch_keys` / `config_snapshot` — persists `output_distribution`, `static_channels`, but NOT `reg_activation`); `views_hydranet/utils/utils.py` `choose_model` (`reg_activation=config.get("reg_activation")`) |
| Cross-refs | C-159 (same sidecar-drift class — but that one crashed loud; this is silent), C-178 (the softplus fix this completes), ADR-063 |

The regression-head output activation `reg_activation` changes the forward function but is **absent from the persisted sidecar `arch_keys`**. On reload, `choose_model` therefore derives the activation from the *current* default (keyed off `output_distribution`, which IS persisted), **not** from what the model was trained with. Because softplus and ReLU share weight shapes, `load_state_dict` succeeds silently — so a model trained with one activation runs the forward with another, producing **wrong predictions with no error signal**. Demonstrated: a relu-trained `hurdle_shrinkage` artifact reloads as softplus (the round-2 probe hit this). **Tier 2:** silent-but-gated — it bites only when the trained activation differs from the reload-time default (explicit override, or a pre-#178 artifact); the production `hurdle_nb` path defaulted to softplus before and after, so it is unaffected. The fix mirrors the adjacent `output_distribution` line (`train_model.py:92`, whose comment already says "persist the head flag (else hurdle_nb reloads as ReLU)"): add `reg_activation` to the snapshot. Failing test: `tests/test_falsify_head_mask_round2.py::test_reg_activation_round_trips_through_sidecar`.

---

### C-180: `active_window` hurdle mask is silently ignored under a latent loss — config no-op with no warning

| Field | Value |
|-------|-------|
| ID | C-180 |
| Tier | 3 |
| Source | /falsify "regression head + mask now 100% correct" round 2 (2026-06-26) — Finding B, SOFT |
| Trigger | Set `hurdle_mask_mode='active_window'` together with a `needs_latent=True` loss (`tobit`, `hurdle_nb`, `dense_nb`) |
| Location | `views_hydranet/train/training_engine.py:223` (`if hurdle_threshold is not None and not use_latent and hurdle_mask_mode == "active_window"`) |
| Cross-refs | C-178, ADR-063; the active_window mask (dossier `2026-06-23_body_sweep_dossier/16`) |

`active_cell` is computed only when `not use_latent`, and the masked hurdle-loss branch is likewise gated on `not use_latent`. So a config that asks for `active_window` decay supervision **while using a latent loss** gets **no active-window supervision at all — silently, with no warning or error**. The behaviour is *semantically* defensible (latent losses model the zeros/censoring themselves, so a hurdle mask does not apply), but the silent no-op of an explicitly-set flag means the user believes decay supervision is on when it is not — exactly the kind of invisible config drift that produced a multi-week mis-attribution before. **Tier 3:** no correctness corruption (the latent loss is doing the right thing), but a maintainability/honesty gap that misleads experiment design. Fix: log a warning (or fail-loud reject) when `active_window` is combined with a latent loss. Failing test: `tests/test_falsify_head_mask_round2.py::test_active_window_with_latent_loss_warns_or_raises`.

---

### C-181: classification (gate) loss has no valid-cell mask — the gate trains on ~60% structural-zero ocean cells (train/eval distribution mismatch)

| Field | Value |
|-------|-------|
| ID | C-181 |
| Tier | 4 |
| Source | /falsify "classification head + mask 100% correct" (2026-06-26) — P5, SOFT; downgraded after the A/B (benign) |
| Trigger | Training any onset gate; acute when diagnosing gate under-prediction / the ns-os under-firing (the gate hedges low) |
| Location | `views_hydranet/train/training_engine.py:330` (`criterion_class(t1_pred_class[:, j], y_cls[:, j])` — full grid, unmasked); `views_hydranet/utils/volume_handler.py:236` (`np.zeros` grid fill) |
| Cross-refs | C-168 / C-170 (gate localization / over-mass on rare targets), C-178 (the reg-side mask findings), ADR-064 |

The classification loss is computed on the **entire** `[H, W]` grid with **no valid-cell mask**, but the grid is **zero-filled** (`np.zeros`), so the ~60% of window cells that are ocean (no priogrid_gid) are supervised as `by_=0` negatives. This is **asymmetric**: the regression body is hurdle-masked (land positives / active cells only) and the evaluation is priogrid-masked (land only), but the gate is **neither**. Because there is no land-mask feature, the gate cannot distinguish quiet land (0 input) from ocean (0 input), so the structural-zero ocean negatives **dilute the learned base rate**, biasing the gate toward under-prediction on land — a plausible contributor to the gate-hedges-low effect behind ns/os under-firing. **Tier 3:** not a silent wrong-output on scored cells (eval masks to land; gate-reliability C-147 found land calibration OK at STEP-1, so the dilution is not catastrophic), but a real, undocumented, **untested** train/eval distribution mismatch on the gate. **Fix is a DECISION** (mask the cls loss to valid/land cells to match reg + eval — changes gate training dynamics, so it needs an A/B to confirm it helps, not a blind change). Failing test: `tests/test_falsify_classification_mask.py::test_classification_loss_restricted_to_valid_cells`.

**UPDATE — A/B RUN (2026-06-26, dossier `2026-06-23_body_sweep_dossier/18`): masking is BENIGN, no benefit.** The opt-in `cls_valid_mask` was implemented and A/B'd (land-masked gate vs full-grid, softplus active_window base, seeds 11/12). Result: MCR_pos / CRPS_pos are **within seed noise on all three targets** (sb 0.448→0.437 / 16.96→17.03, ns/os unchanged at ~0.005/~0.007) — sb marginally *worse* masked. ⇒ the ocean dilution does **not** materially affect outputs (the gate already separates ocean from land via the zero input); the train/eval mismatch is real but immaterial. ns/os under-firing is therefore **NOT** a maskable training artifact — it's the irreducible rare-onset hedge. **Downgraded to Tier 4** (confirmed no correctness/reliability impact). **Disposition: do NOT adopt land-masking as default.** The `cls_valid_mask` opt-in is **KEPT** (off by default, byte-unchanged, tested) as a ready lever for a future smarter gate design that might exploit it. Kept open (not resolved) as that future-work hook.

---

### C-182: `total_hidden_channels` divisibility-by-8 contract is documented but unenforced — fails late with a cryptic unpack error

| Field | Value |
|-------|-------|
| ID | C-182 |
| Tier | 4 |
| Source | /falsify "the convolutional lstm is 100% correctly implemented" (2026-06-26) — P4 (contract), SURVIVED as observation |
| Trigger | When a new config or capacity sweep sets `total_hidden_channels` to a value not divisible by 8 (e.g. tuning recurrent memory width) — the run dies deep in `forward()` instead of at construction |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:65-66` (docstring "Must be divisible by 8"), `:101` (`num_lstm_state_layers = int(total_hidden_channels / 8)`, silent floor), `:426-427` (`split_h = int(h.shape[1] / 8)` + 8-way unpack that raises) |
| Cross-refs | C-114, C-184 (same recurrent-cell architecture, undocumented/unguarded properties) |

The constructor documents "`total_hidden_channels` Must be divisible by 8" but does **not** validate it. A non-divisible value (e.g. 12) is silently floored by `int(.../8)` when sizing the gate convs, then later **raises `ValueError: too many values to unpack (expected 8)`** from the `torch.split` unpack in `forward()` — a cryptic error far from the cause. **No silent corruption** (P4 confirmed it crashes loud, not wrong-output), so this is **Tier 4 / ergonomic only**: correctness is intact, but a developer tuning capacity gets an opaque failure at the first forward instead of a clear constructor message. Fix (optional): add `if total_hidden_channels % 8: raise ValueError(...)` in `__init__`. No failing test stub — the /falsify verdict was SURVIVED (no hard/soft falsification); registered as a maintainability observation per user request.

---

### C-183: ConvLSTM forget-gate bias is not initialized to 1 (PyTorch default ≈0) — memory starts "off"

| Field | Value |
|-------|-------|
| ID | C-183 |
| Tier | 4 |
| Source | /falsify "the convolutional lstm is 100% correctly implemented" (2026-06-26) — P5 (adequacy), observation |
| Trigger | When diagnosing slow/failed long-horizon temporal retention or tuning the recurrence — rule out the near-0 forget-gate warm-start before attributing it to data or architecture |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (`self.Wxf_1..4` / `self.Whf_1..4` forget-gate convs — `bias=True`, PyTorch default init; no `forget_bias=1` set) |
| Cross-refs | C-114, C-184 (recurrent-cell architecture choices that are undocumented/unexamined) |

The forget-gate convs (`Wxf_*`, `Whf_*`) leave their biases at PyTorch's default (≈0, uniform), rather than the common `forget_bias=1` warm-start (Jozefowicz 2015) that starts the cell biased to **retain** memory. This is a **training-speed/optimization convention, not a correctness requirement** — the LSTM is mathematically correct either way (P1/P2 confirmed live memory + BPTT). **Tier 4:** no correctness or reliability impact; flagged so that any future investigation of weak temporal memory considers the warm-start as a cheap lever before larger changes. No failing test stub (SURVIVED verdict).

---

### C-184: BatchNorm runs inside the recurrent loop — running stats accumulate T× per window over temporally-correlated steps

| Field | Value |
|-------|-------|
| ID | C-184 |
| Tier | 2 ⬆ (was 4 — UPGRADED 2026-06-27: confirmed ROOT CAUSE of the seed-bimodal eval collapse) |
| Source | /falsify ConvLSTM (2026-06-26) P5; **CONFIRMED root cause via BN-recal experiment (2026-06-27)** |
| Trigger | FIRES NOW on every training run: ~40% of seeds land BN running-stats that over-amplify at eval → gate saturates → composed E[y] explodes (the seed-bimodality + much of C-113). Acute on any retrain. |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (`bn_enc_conv0/1`, `bn_bottleneck_conv`, the `bn_dec_conv*` head BNs — all invoked inside the per-timestep `forward`, which the engine calls T times per window) |
| Cross-refs | C-114 (undocumented recurrent-regularization surface), C-183, C-113 (rollout dynamics) |

The encoder/bottleneck/decoder `BatchNorm2d` layers are inside the single-timestep `forward`, called T× per window over **temporally-correlated** activations (cf. Cooijmans 2016 recurrent BN). Originally logged Tier-4 ("stable design choice"). **⬆ UPGRADED Tier-2 — CONFIRMED ROOT CAUSE (2026-06-27).** The 2026-06-26 perf program found the production floor is **seed-bimodal (~40% of seeds collapse: saturated gate π̄≈0.1–0.36, rollout MCR_pos 30–260×)**. Triangulated the cause: NOT the loss (pos_weight sweep flat), NOT the weights (per-layer spectral norms + gate-head bias identical good-vs-bad), NOT the training trajectory (good/bad train-time gate-logit identical, because **training uses batch-stats BN**). The decisive test (`bn_mode_probe.py`): every seed is calibrated under **train-mode BN** (π̄≈0.002–0.005) but saturates under **eval-mode BN** (π̄ 0.4–0.998), worst for the bad seeds (which have lower BN `running_var` → eval BN over-amplifies). **FIX CONFIRMED + UNIVERSAL:** recompute BN running stats post-training (forward-only over real windows, reset BN + `momentum=None`) flips **6/6 bad seeds BAD→GOOD and preserves 2/2 good** — bad-basin rate ~40%→0%, rollout MCR_pos collapses to 2.5–8.3× (e.g. seed 201: step-1 CRPS 33.8→0.24, MCR 259→5.8). So this is **silent eval-time model-output corruption on ~40% of trained models** (Tier-2: not Tier-1 only because it surfaces as loud explosions, not a quiet wrong answer, and is now fixable). **Resolution paths:** (a) post-training BN-recal pass before artifact save [cheapest, validated], (b) fix the recurrent-BN momentum/update at the root, (c) GroupNorm/LayerNorm (no train/eval gap; needs retrain). Opt-in `bn_recal_from` flag in `training_engine.py` (uncommitted) implements the test. Tools: `/tmp/run_bn_recal_all.sh`, `/tmp/bn_mode_probe.py`, `/tmp/recal_all_score.py`. Cross-ref C-113 (this is a large part of the eval-explosion), C-147 (gate-calibration), the perf program.

---

### C-185: U-Net is only 2 encoder levels deep (÷4) — limited receptive field on large grids

| Field | Value |
|-------|-------|
| ID | C-185 |
| Tier | 4 |
| Source | /falsify "the U-Net is 100% correctly implemented" (2026-06-26) — P6 (adequacy), observation |
| Trigger | When scaling to a larger spatial grid, or diagnosing weak long-range spatial context / under-use of distant cells — consider adding encoder depth before attributing it to the loss or data |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:106-123` (enc_conv0 → pool0 → enc_conv1 → pool1 → bottleneck; 2 pooling levels) |
| Cross-refs | C-169 (the conv low-pass / in-sample smoothness confound — related but distinct: depth/receptive-field vs blur) |

The "U-Net" has only **2 encoder levels** (two `MaxPool2d(2,2)` → total ÷4 downsampling) before the bottleneck. On a large conflict grid the effective receptive field is small relative to the domain, limiting how much distant spatial context any cell's prediction can integrate. This is **documented and intended** ("2 Encoder levels" in the class docstring) and the model has performed acceptably, so it is **not a defect** — **Tier 4:** a design trade-off (depth vs cost/stability) flagged so a future investigation of weak spatial context considers adding encoder depth as a lever. No failing test stub (SURVIVED verdict).

---

### C-186: single bottleneck shared across all 6 heads — per-head representational capacity is coupled

| Field | Value |
|-------|-------|
| ID | C-186 |
| Tier | 4 |
| Source | /falsify "the U-Net is 100% correctly implemented" (2026-06-26) — P6 (adequacy), observation |
| Trigger | When one head (e.g. a rare target) needs to diverge representationally from the others, or when adding/removing heads — all 6 decoders branch from the single shared bottleneck `b`, so their capacity is coupled at that point |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:476-477` (`b` computed once), `:481-586` (all 6 heads consume the same `b`, `e1s`, `e0s_topskip`) |
| Cross-refs | C-03, C-123 (hardcoded 3+3 head topology, Cluster 4 — related but distinct: code-duplication-to-add-a-head vs shared-bottleneck capacity coupling) |

All 6 decoder heads (3 reg + 3 class) branch from the **single shared bottleneck `b`** (and the shared encoder skips `e1s`, `e0s_topskip`). The heads have independent decoder weights (verified head-isolated in P1), but their input representation is a common bottleneck — so per-head representational capacity is **coupled** there: a head that needs features the shared bottleneck does not preserve cannot recover them. This is an intentional parameter-economy choice and **not a correctness defect** (P1/P4/P5 all clean) — **Tier 4:** registered so that if a head (e.g. a rare ns/os target) underperforms in a way that looks representational, the shared-bottleneck coupling is a known candidate. No failing test stub (SURVIVED verdict).

---

### C-187: stride-2 ConvTranspose upsampling — checkerboard-artifact prone

| Field | Value |
|-------|-------|
| ID | C-187 |
| Tier | 4 |
| Source | /falsify "the U-Net is 100% correctly implemented" (2026-06-26) — P6 (adequacy), observation |
| Trigger | When investigating grid-periodic / checkerboard artifacts in the spatial predictions, or refreshing the upsampling path — `ConvTranspose2d(stride=2)` can produce them; upsample-then-conv is the standard mitigation |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (`upsample0_head*`, `upsample1_head*` — all `nn.ConvTranspose2d(..., stride=2, kernel=2)`) |
| Cross-refs | C-190 (skip-path high-freq throttle — the sibling efficiency item), C-185/186 |

All 12 decoder upsampling layers (2 per head × 6 heads) use `ConvTranspose2d` with `stride=2`, which is known to produce **checkerboard artifacts** (Odena 2016, "Deconvolution and Checkerboard Artifacts") when kernel size is not divisible by stride and weights are unlucky. Here kernel=2, stride=2 (divisible), which mitigates but does not eliminate the risk. **Measured present (2026-06-27, /falsify skip-effectiveness, P5):** a structured-input forward on a good-basin artifact shows a **~13% even-vs-odd pixel-grid mean asymmetry** in the regression output — the checkerboard is real but small (injects a periodic *artifact* high-freq, distinct from skip-delivered detail). Still **Tier 4:** a periodic artifact, not a correctness defect; the fix (`Upsample`+`Conv` instead of `ConvTranspose`) is a known one-line swap and is the cheapest sharpness-side A/B alongside C-190. No failing test stub.

---

### C-188: U-Net skip geometry has no grid-divisibility guard — non-÷4 grid fails late with a cryptic skip-`cat` error

| Field | Value |
|-------|-------|
| ID | C-188 |
| Tier | 4 |
| Source | /falsify "the U-Net is 100% correctly implemented" (2026-06-26) — P2 (skip alignment), SURVIVED as observation |
| Trigger | When a new config or data window uses a spatial grid whose `H` or `W` is not divisible by 4 — the two-pool/two-upsample U-Net dies deep in `forward()` at the first skip `cat`, not at construction |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:471-473` (pool0/pool1 floor-divide), `:483/:490` etc. (`torch.cat([upsampleN(...), skip], 1)` — mismatches when the floored pool size ≠ the transpose-upsampled size) |
| Cross-refs | C-182 (the LSTM `total_hidden_channels % 8` sibling — same "documented shape contract, no upfront guard, cryptic late crash" pattern, different axis: spatial grid vs hidden channels) |

The U-Net's two `MaxPool2d(2,2)` floor odd sizes while the two `ConvTranspose2d(stride=2)` exactly double, so a grid not divisible by 4 yields skip/upsample size mismatches: P2 confirmed grid 14 raises `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 6 but got size 7` at the first skip `cat`. **Loud, no silent corruption** (the output is never wrongly cropped/padded), so **Tier 4 / ergonomic only** — but the failure surfaces deep in `forward()` with an opaque message rather than a clear up-front "grid must be divisible by 4" check. Same class as C-182. Optional fix: validate `H % 4 == 0 and W % 4 == 0` at the start of `forward` (or document the constraint on the input contract). No failing test stub (SURVIVED verdict).

---

### C-189: `inverse_sigmoid` scheduled-sampling schedule has no `k≥1` validation — silently wrong curriculum or mid-training crash

| Field | Value |
|-------|-------|
| ID | C-189 |
| Tier | 3 |
| Source | /falsify "the curriculum learning is 100% correctly implemented" (2026-06-26) — P5, SOFT falsification |
| Trigger | When a config sets `ss_schedule="inverse_sigmoid"` with `ss_k < 1` (or `0`) — the schedule silently starts at high epsilon (no teacher-forcing warmup, curriculum defeated) for `k∈(0,1)`, or raises `ZeroDivisionError` mid-training for `k=0` |
| Location | `views_hydranet/utils/scheduled_sampling.py:37-38` (constructor guards `k<1` for `exponential` only), `:56-59` (`k/(k+exp(shifted/k))` — `k=0` divides by zero), `views_hydranet/utils/config_initializer.py:701-704` (validator guards `exponential` k only) |
| Cross-refs | C-182, C-188 (same missing-validation pattern); C-156 (the adjacent curriculum-subject coupling, P4) |

The scheduled-sampling curriculum (ADR-056, Bengio et al. 2015) validates `k<1` as **invalid for `exponential`** in *both* `ScheduledSamplingMixer.__init__` and `HydraNetConfig.validate_scheduled_sampling_params`, but is **silent on the symmetric Bengio requirement `k≥1` for `inverse_sigmoid`** in *neither*. Confirmed by P5: `inverse_sigmoid` with `k=0.3` is accepted and runs a **silently wrong schedule shape** — epsilon starts at ~0.77 instead of ~0, so there is no teacher-forcing warmup and the curriculum is defeated; `k=0` raises a bare `ZeroDivisionError` deep in `get_epsilon` mid-training. **Tier 3:** not silent corruption on a *correct* config (the production schedules are unaffected; all three schedules are monotone-increasing as designed), but a real validation/robustness gap that mistrains or crashes under a plausible misconfig, asymmetric with the existing `exponential` guard (a copy-paste-asymmetry smell). The `CurriculumLearner` core (cooling, oscillation, relative thresholding) survived all probes — this is the scheduled-sampling sibling only. **Failing tests:** `tests/test_falsify_curriculum_ss.py` (3 stubs, red): mixer rejects `inverse_sigmoid k<1`; `k=0` → clean `ValueError` not `ZeroDivisionError`; config validator adds the `== "inverse_sigmoid"` k-bound guard. Fix: add the symmetric `k<1` reject for `inverse_sigmoid` in both the mixer constructor and the config validator.

---

### C-190: skip-connection high-frequency path is throttled by BatchNorm + dropout (sharpness efficiency headroom)

| Field | Value |
|-------|-------|
| ID | C-190 |
| Tier | 4 |
| Source | /falsify "the skip connections are correctly and effectively wired up" (2026-06-27) — P1/P2/P4, SOFT (efficiency) |
| Trigger | When attacking the regression over-smoothing from the **architecture** side (only after the loss-side levers C-168/169), or refreshing the encoder→decoder skip path — the high-freq detail the skips carry is attenuated before the merge |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:467` (`e0s = self.dropout(e0s_)` — finest skip dropped), `:472` (`e1s = self.dropout(F.relu(self.bn_enc_conv1(...)))` — coarse skip BN'd + dropped), decoder cats `:483/:490` etc. |
| Cross-refs | C-187 (checkerboard — sibling skip-path efficiency item), C-168 / C-169 (the over-smoothing this could marginally help; the LOSS is the primary lever), C-185 / C-186 (other U-Net architecture observations) |

The /falsify **effectiveness** audit confirmed the skips are **correctly wired and heavily load-bearing** — ablation gave **e0s 81% / e1s 27%** L2 output dependence (not dead, not ignored). BUT the high-frequency detail they exist to deliver is **throttled on the path to the decoder**: *both* skips pass through `LockedDropout` (15% zeroed; re-randomised per MC sample at inference), and `e1s` additionally through `BatchNorm` (which normalises away the high-variance detail). Measured (P2): the trained model is a **strong low-pass** — a broadband sparse-impulse input's high-freq power fraction **0.32 → 0.07** at the output (~78% of high-freq energy discarded); and the finest skip currently carries mostly **low**-frequency content (zeroing it *raises* the output high-freq fraction). **Tier 4 — NOT a correctness or wiring defect:** the over-smoothing is **loss-driven, not a hard skip ceiling** — the *same* architecture produced sharp regression heads under shrinkage/MSE (user testimony; consistent with C-169 "backbone overfits a sharp tile trivially" and the intrinsic spectral bias to low frequencies, Rahaman 2019 / Fourier features). Registered as **independent efficiency headroom**, explicitly downstream of the loss work. Cheapest skip-side A/Bs IF pursued: (a) remove dropout from the skip path (keep it on the bottleneck), (b) `Upsample`+`Conv` over stride-2 `ConvTranspose` (C-187). Both one-liners; **need an A/B to confirm a benefit — do not change blind.** No failing test stub (CONTESTED — efficiency hypothesis, not a wiring bug).

---

### C-191: `output_distribution='hurdle_shrinkage'` is a misnomer — the compose is loss-agnostic (gate·expm1(point)), not shrinkage

| Field | Value |
|-------|-------|
| ID | C-191 |
| Tier | 3 |
| Source | user observation (2026-07-02) — surfaced during the T=0 calibration reconciliation |
| Trigger | When configuring or reasoning about a point-body hurdle arm (mae/huber/pareto/shrinkage), or reading ADR-063 — the config value names the compose after ONE loss (shrinkage) although the compose (`hurdle_point_expected_log1p`) is loss-agnostic |
| Location | `views_hydranet/utils/config_initializer.py:380` (validator whitelist), `views_hydranet/utils/hydranet_inference.py:215` (dispatch), `views_hydranet/utils/hurdle_nb.py:74` (`hurdle_point_expected_log1p`), `docs/ADRs/active/063_regression_head_output_activation.md:29` |
| Cross-refs | C-149 / C-168 (the T=0 calibration work this naming obscured) |

The `output_distribution` value `hurdle_shrinkage` selects the compose `hurdle_point_expected_log1p(reg, prob) = log1p(P(y>0)·expm1(reg))` — a **hurdle with a log1p-space POINT body**, which is **loss-agnostic**: mae, huber, pareto, and shrinkage all decode through it identically. The name derives from the first loss that happened to use it (`ShrinkageLoss`), not from what it computes — there is no shrinkage in the compose. This actively misled reasoning about the loss↔compose coupling (2026-07-02: an explanation treated the name as if it were meaningful). The underlying function is correctly named (`hurdle_point`); the config value is the misnomer. **Fix:** rename `output_distribution='hurdle_shrinkage'` → `'hurdle_point'` with a back-compat alias in the validator + inference dispatch; update ADR-063. No correctness defect (the compose is right) — clarity/maintainability only. No failing test stub.

---

### C-192: #144 grid-name flip (priogrid_gid→priogrid_id) not wired into DataSniffer — floor config blocks at ingestion

| Field | Value |
|-------|-------|
| ID | C-192 |
| Tier | 2 |
| Source | bulk-calibration dossier P2 smoke (2026-07-16) — realized failure |
| Trigger | Launching any hydranet run (train/eval/forecast) from a config whose grid name (`identity_cols`/`index_names`/`id_col`) does not match the current parquet's grid column — e.g. running the on-disk floor config (`priogrid_gid`) against today's `priogrid_id` viewser/parquet data |
| Location | `views_hydranet/utils/data_sniffer.py:299-320` (`_check_obligatory_columns`, reads hardcoded `config['identity_cols']`/`['features']`/`['spatial_cols']`); the incomplete fix: `views_hydranet/utils/grid_naming.py::grid_id_col` (added by #144 / `1f707d3`) wired only into `data_fetcher.py` + `scripts/mcr_readout.py`; stale config `models/violet_visitor/configs/config_hyperparameters.py` (still `priogrid_gid`) |
| Cross-refs | C-174 (name-as-contract, no boundary adapter — this is a realized instance), C-120 (dual data-layer authority), C-173 (prefix-role parsing), C-19 (priogrid_gid>0 ingestion assumption) |

The platform grid-entity rename **priogrid_gid → priogrid_id** (GH #144) is now **live** in the data: both the cached `calibration_viewser_df.parquet` and a fresh viewser fetch emit `priogrid_id`. The #144 fix (`1f707d3`) introduced `grid_naming.grid_id_col` (name-set membership, fail-loud) and wired it into `data_fetcher` (load path) and `mcr_readout` (truth-join key) — but **did not wire it into `DataSniffer._check_obligatory_columns`**, which still validates the df against the hardcoded `config['identity_cols']` grid name. Consequence: **any run launched from the on-disk floor config (which still declares `priogrid_gid`) fails at ingestion** with `ValueError: Missing Obligatory Columns: ['priogrid_gid']`, before training starts. **Verified:** the first P2 smoke launch died exactly here; worked around by setting `id_col`/`identity_cols`/`index_names` → `priogrid_id` in the run config. The same hardcode existed in `reports/2026-07-16_bulk_calibration_dossier/tools/bulk_score.py` (`set_index('priogrid_gid')` for the truth join) and was fixed there to derive the grid column from data. **Tier 2:** a realized structural fragility that hard-blocks the primary train/eval workflow from the canonical config; the #144 fix is demonstrably incomplete (2 of 3 grid-name consumers). **Not Tier 1** — fail-loud (ValueError), no silent corruption. **Fix:** wire `grid_naming.grid_id_col` into `DataSniffer._check_obligatory_columns` (derive the grid entity from the df, same pattern as `data_fetcher`) so the sniffer is grid-name-agnostic; and/or update the floor config's grid name to `priogrid_id`. A grep for remaining hardcoded `priogrid_gid` across `views_hydranet/` + configs + fixtures would scope the full #144 residue.

---

### C-193: `body_mask` masking silently ignored under a latent loss — trains dense while config says masked

| Field | Value |
|-------|-------|
| ID | C-193 |
| Tier | 2 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Nygard/ADR-008 |
| Trigger | Sweeping `body_mask` (or setting `hurdle_threshold`+mode) to a masking value while `loss_reg` is a latent likelihood (`hurdle_nb`/`lognormal_nll`/`tobit`) |
| Location | `views_hydranet/train/training_engine.py:255-263, 343` (`if hurdle_threshold is not None and not use_latent`; warn-once C-180) |
| Cross-refs | C-194 (same interface), C-180 (the warn-once), ADR-008, ADR-003 Law 1 |

The point-body mask is silently a **no-op under a latent loss** — only a warn-once fires (C-180). A run can be configured "masked" and train **dense**, invisibly, with no error and no metric signal. Violates ADR-003 Law 1 (Fail Loud — it explicitly names "silent truncation") and ADR-008. **Tier 2:** silent wrong-training under a realistic sweep, no error signal. Fix: a hard `ValueError` at config validation when `body_mask ∈ {pos_cells,pos_timelines}` and the loss is latent (mirror the tobit/`hurdle_threshold` contradiction at `config_initializer.py:627`).

### C-194: `hurdle_mask_mode` read raw + un-validated — a typo silently degrades the mask to per_step

| Field | Value |
|-------|-------|
| ID | C-194 |
| Tier | 2 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Nygard/ADR-009 |
| Trigger | Setting `hurdle_mask_mode` in a config with a typo (e.g. `active-window` vs `active_window`) |
| Location | `views_hydranet/train/training_engine.py:549` (`config.get("hurdle_mask_mode","per_step")`); NO field in `config_initializer.py` |
| Cross-refs | C-193, ADR-009 (config as validated boundary) |

`hurdle_mask_mode` is not a config field — it's read straight from the dict with a `"per_step"` default, so any typo silently trains the wrong mask (e.g. `active_window` intended, per_step trained). No validation, no error. Violates ADR-009 (all boundaries validated). **Tier 2:** silent mis-training with no signal. Fix: the validated `body_mask` enum becomes the sole front door; the raw `config.get` read is deleted.

### C-195: dual authority over "what is an event" — mask threshold vs binary-derivation threshold can drift

| Field | Value |
|-------|-------|
| ID | C-195 |
| Tier | 3 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Martin/Kleppmann/ADR-046 |
| Trigger | Changing the binary-target derivation threshold (`config['derivations']['binary'][...]['threshold']`) without changing the mask's hardcoded `> threshold` |
| Location | mask literal in `training_engine.py:263/349` vs `config_initializer.py:53` (`derivations`) |
| Cross-refs | C-193/C-194, ADR-046 (Transformations vs Derivations), ADR-003 Law 6 |

"A cell is an event where `y > 0`" is defined in **two** places — the binary-target derivation (config `derivations`) and the mask threshold in the training loop. They can silently diverge, so `by_*` labels and the body mask would disagree on which cells are events. **Tier 3:** maintainability/consistency hazard, no current corruption (both are 0 today). Fix: the mask sources its event threshold from the derivation config (single authority).

### C-196: `body_mask='none'` refactor must be byte-identical to the current foundation — else silent drift

| Field | Value |
|-------|-------|
| ID | C-196 |
| Tier | 3 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Feathers/Beck |
| Trigger | Refactoring the two-knob mask into `body_mask` without a characterization net |
| Location | `training_engine.py` masking path; `tests/` (no end-to-end characterization test today) |
| Cross-refs | C-193/194/195, ADR-005 |

The foundation (all-cell MSE gated) is the lodestar baseline. If the `body_mask` refactor changes the masked cell-set at `none` even slightly, the foundation shifts silently and every comparison to it is invalidated. There is currently **no** config→behaviour characterization test. **Tier 3:** regression risk on a load-bearing baseline. Fix: a characterization test snapshotting the current masked-cell-set for all three legacy knob-combos BEFORE the refactor, asserted identical after.

---

### C-197: distribution registry / legacy `output_distribution` name collision → silent legacy hijack

| Field | Value |
|-------|-------|
| ID | C-197 |
| Tier | 2 |
| Source | /falsify adequacy audit of ADR-067 §3 (2026-07-20) |
| Trigger | Registering a `DistributionFamily` in `DISTRIBUTION_REGISTRY` whose name equals a legacy `output_distribution` value (`standard`/`hurdle_shrinkage`/`hurdle_nb`/`hurdle_lognormal`/`dense_nb`/`quantile`) |
| Location | `views_hydranet/distributions/registry.py` (planned); `views_hydranet/utils/config_initializer.py` valid-list `~388-403` (`FAMILY_NAMES ∪ legacy`) |
| Cross-refs | ADR-067 §3; Epic A #167 (A-S2 #169 registry, A-S5 #172 config); C-196 (byte-identical foundation) |

The strangler-fig integration (ADR-067) unions the registry family names with the legacy `output_distribution` values into one valid-list and dispatches via `resolve_family(name)`. If the two name-sets **intersect**, a legacy config value routes to the new family instead of its untouched legacy branch — silently changing a proven, byte-identical model with **no error**, and invalidating every comparison to the lodestar baseline. **Tier 2:** structural fragility with a specific, realistic trigger (a future family author picking a colliding name). Fix: a fail-loud validator + test asserting `FAMILY_NAMES ∩ legacy = ∅` (registry names must be disjoint from legacy values); an acceptance criterion of A-S5.

---

### C-201: self-zeroed ZINB decouples the classification (gate) head from the forecast — frozen-ruler AP/Brier then score a head the forecast ignores

| Field | Value |
|-------|-------|
| ID | C-201 |
| Tier | 2 |
| Source | /falsify (2026-07-20), P5 |
| Trigger | Scoring a self-zeroed `nb`/`zinb` family on the frozen lodestar ruler's gate metrics (AP/Brier) |
| Location | the lodestar scorer `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`; planned `distributions/` `prob_positive`; A-S11 (#178) eval |
| Cross-refs | C-199/C-200; ADR-067 (self-zeroed); F1 pre-registration; **C-211 (empirical confirmation — 300-lesson M1: count-only occurrence AP ~0.27 vs cls-gate ~0.44)** |

A self-zeroed ZINB produces its zeros from the distribution (`P(Y>0)=(1−π)·(1−NB(0))`), **not** from the classification head. But the frozen ruler computes gate quality (AP/Brier) on the cls head. So the reported gate metric describes an occurrence estimate the ZINB forecast does not use — the two can diverge silently, mis-informing the M1/M2 go/no-go. **Tier 2:** silent mis-attribution in the evaluation that gates production decisions. Fix: for self-zeroed families the ruler must score the **distribution-implied** `P(Y>0)` (family exposes `prob_positive`), or the eval must explicitly document that the cls head is decoupled and not the forecast's gate.

---

### C-204: `[DEMOTED]` inverse-softplus link now exists in 3 places under 2 names — consolidate when Epic B migrates the legacy losses

| Field | Value |
|-------|-------|
| ID | C-204 |
| Tier | 4 |
| Source | /code-review max (2026-07-20), A-S3 (#170) reuse finder |
| Trigger | A future change to the inverse-softplus identity, or Epic B (#181) migrating `DenseNBLoss`/`TruncatedNBLoss` onto the distribution abstraction |
| Location | `views_hydranet/distributions/nb_core.py` `inverse_softplus` (new subsystem copy) + legacy `views_hydranet/utils/dense_nb_loss.py:34` and `views_hydranet/utils/truncated_nb_loss.py:34` (`_inverse_softplus`) |
| Cross-refs | Epic B #181 (legacy-loss migration); C-199 (informed init uses it) |

A-S3 added a stable `inverse_softplus` to `nb_core` (the subsystem's shared NB math), fixing the overflow-prone `log(expm1(y))` form — but the identity now also lives, identically, in two legacy loss modules (`dense_nb_loss`, `truncated_nb_loss`). Any fix to the link must be applied in all copies. **Tier 4 / [DEMOTED 2026-07-20 → Tech-Debt Backlog]:** near-mechanical, no design decision; the natural time to consolidate to one shared util is when Epic B (#181) migrates those legacy losses onto the abstraction. Not an active governance risk.

---

### C-209: the D×K sampler folds epistemic + aleatoric uncertainty onto one flat `S` axis with no `(D,K)` factorization recorded

| Field | Value |
|-------|-------|
| ID | C-209 |
| Tier | 3 |
| Source | /expert-code-review (A-S10 #177), Kleppmann/Hickey |
| Trigger | A future analysis needing to separate epistemic (MC-dropout `D`) from aleatoric (head-draw `K`) uncertainty from a **stored** PredictionFrame (rather than a fresh run) |
| Location | `views_hydranet/utils/hydranet_inference.py` `generate_posterior_samples` (`posterior_S = D*K`); the `(N,S)` frame |
| Cross-refs | C-206 (`n_head_samples` manifest capture); D-12 (per-origin CRN) |

The D×K sampler fills the `[T,H,W,C,S]` cube with `S = D×K` where `D` = MC-dropout passes (model/epistemic) and `K` = per-cell head draws (outcome/aleatoric), but the two are folded onto one **flat** `S` axis with no `(D,K)` factorization stored. `(D=8,K=1)` and `(D=1,K=8)` are indistinguishable once scored, and the epistemic/aleatoric decomposition is unrecoverable without a re-run. **Tier 3:** benign for the committed CRPS M-program (it needs only the marginal `S`), but a lossy, irreversible data-contract choice. Fix: record `(D,K)`/`n_head_samples` in the frame metadata so the axis factorization is recoverable.

---

### C-210: `[DEMOTED]` `_standard_gamma` silently falls back to the region mean on 64-iteration non-acceptance instead of warning

| Field | Value |
|-------|-------|
| ID | C-210 |
| Tier | 4 |
| Source | /expert-code-review (A-S10 #177), Nygard |
| Trigger | A degenerate-parameter run where a cell's Gamma concentration never accepts within the 64-iteration Marsaglia-Tsang loop |
| Location | `views_hydranet/distributions/nb_core.py:91-104` |
| Cross-refs | C-208 (the same sampler's uncharacterised spread) |

`_standard_gamma`'s rejection loop keeps `out = d.clone()` (the accepted-region mean) for any cell still un-accepted after 64 iterations, degrading **silently** rather than failing/warning. Probability ~`0.05⁶⁴` (astronomically negligible; ~0.95 accept/iteration). **Tier 4 / loudness only:** no realistic correctness impact, but a silent-degradation gap on the CRPS-bearing sampler. Fix: emit a one-time warning if `remaining.any()` after the loop.

---

### C-211: count-only self-zeroed scoring UNDERSELLS the family's occurrence (the sharp cls gate carries spatial precision); the residual crps gap is a BODY-magnitude issue, gate-independent

| Field | Value |
|-------|-------|
| ID | C-211 |
| Tier | 3 |
| Source | A-S11 M1 300-lesson nb run + training-biopsy inspection + gate-vs-count-only re-score (2026-07-22) |
| Trigger | Reading a self-zeroed family's count-only AP/Brier (gate = fraction of its own samples > 0) as the family's occurrence ceiling, OR concluding an M1/M2 pass-kill from the metrics without separating the gate (occurrence) contribution from the body (magnitude) contribution |
| Location | `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` (count-only vs the `by_{t}_best` cls-gate template); the family self-zeroed occurrence (`prob_positive`/`samples>0`) vs the classification-head gate; `reports/2026-07-20_distributional_head_dossier/05_analysis_plan.md` M1 decision rule |
| Cross-refs | C-201 (self-zeroed gate/forecast decoupling — **this is its empirical confirmation + refinement**); C-146 (ZINB vs hurdle); C-209 (D×K axis); the foundation's `hurdle_shrinkage` (sharp gate + timid body won crps) |

Empirical, from the **300-lesson M1 nb** run (3 seeds, frozen ruler, T=0, N=170430): **(1)** the per-cell NB body **fixed the timid magnitude** the epic targeted — size-ratio jumped from ~0.02 (foundation) / 0.0 (40-lesson) to **~0.29** on sb (approaching white_ranger 0.39). **(2)** But the NB's **self-zeroed occurrence** (count-only, samples>0) is spatially **diffuse** — AP ~0.24–0.30; re-scoring the SAME predictions with the sharp classification gate (`by_{t}_best`, hurdle-style) lifts AP to **0.44 / 0.40 / 0.26** (sb/ns/os), **beating white_ranger** (0.33/0.22/0.16) on all 3. So **count-only undersells** the family's achievable occurrence; the sharp cls gate carries the spatial precision (plainly visible in the training-forensic biopsy: the diffuse NB-body row vs the sharp gate×body row). **(3)** Crucially, **crps-all is gate-INDEPENDENT** — computed on the body ensemble, it is **identical** (0.159/0.091/0.046) under count-only and gated scoring; the sharp gate does **not** close the crps gap to the foundation (nb 0.159 vs 0.137 on sb). So the residual crps gap is a **body-magnitude/calibration** issue, not an occurrence/gate one. (Brier trade: the cls gate ranks better — AP↑ — but is less calibrated, Brier 0.006→0.013.) **Tier 3 (evaluation/methodology, not silent-corruption):** the risk is mis-reading the M1/M2 verdict — treating count-only occurrence as the family's ceiling, or attributing the crps gap to the gate. Implication: the strong shape is **hurdle = sharp cls gate (occurrence) × per-cell body (magnitude)**; the self-zeroed nb undersells occurrence, and **ZINB (M2, structural π)** is the candidate fix for the body/crps gap. Reshapes M2 toward the `hurdle_nb` + `zinb` arms. A `hurdle_nb` 3×300 confirmatory run was launched 2026-07-22 to test the sharp-gate×NB-body hypothesis directly.

---


### C-215: per-lesson TRAINING reg/cls loss (and grad-norm) is not persisted numerically — the loss-balance / gradient-budget of a finished run is unrecoverable post-hoc

| Field | Value |
|-------|-------|
| ID | C-215 |
| Tier | 4 |
| Source | User request ("nice to know one day") + this-session loss-balance investigation (2026-07-24) |
| Trigger | Wanting to analyze the reg-vs-cls (or per-target) training loss balance / effective gradient budget of a **past/finished** run — the numeric data isn't there to reconstruct it |
| Location | `views_hydranet/train/training_engine.py` — `lesson_reg`/`lesson_cls` + `raw_grad_norm` (pre-clip) are computed per lesson but only feed the `biopsy_loss_curves` **PNG** (`02_training_dynamics`); the opt-in `_traj_writer` CSV (lesson_idx/raw_grad_norm/lesson_reg/lesson_cls/gate_mean) exists but is **OFF**; wandb logs only EVAL metrics |
| Cross-refs | C-111 (MultiTaskLoss balancer / log_vars); the gate `pos_weight` findings; the "timid body" thread |

We wanted to confirm whether reg/cls and per-target losses get a similar effective "step size." The frozen `MultiTaskLoss` balancer applies fixed coefficients (reg ×0.5, cls ×1.0) and `pos_weight=10` amplifies the cls BCE, so a start-of-training probe estimated **cls dominates reg ~5× in effective gradient** (targets balanced within ~2×) — consistent with the long-standing **timid body**. But this could **not** be confirmed on the real trained run: the per-lesson training reg/cls losses (and grad norms) are not logged anywhere readable (wandb = eval only; text log = tqdm-dominated; only the loss-curve PNG exists). So the reg/cls gradient-budget balance of any finished run is unrecoverable without a bespoke re-run. **Tier 4 (observability/convenience — no correctness or reliability impact).** Low-priority fix: default-on a lightweight per-lesson numeric log of reg/cls loss (+ raw grad norm, ideally per-target) — as wandb scalars or by enabling the `_traj_writer` CSV by default — so the loss-balance question is answerable post-hoc.

---

### C-216: `feedback_clamp_log1p` (the C-113 autoregressive-feedback safety rail) had ZERO effect on the family eval rollout — a guard that silently no-ops

| Field | Value |
|-------|-------|
| ID | C-216 |
| Tier | 3 |
| Source | Bloom investigation (2026-07-25), rung-2 τ/clamp sweep — `reports/2026-07-20_distributional_head_dossier/bloom_investigation.md` |
| Trigger | Relying on `feedback_clamp_log1p` to bound the autoregressive-feedback magnitude — e.g. a future rung-2 bloom mitigation, or any run that sets it expecting the runaway to be capped |
| Location | `views_hydranet/utils/hydranet_inference.py` — `_parse_feedback_clamp` / `_clamp_feedback` (the rail) vs the family AR-feedback path (~442/86) |
| Cross-refs | C-113 (the bloom / autoregressive feedback); `plan_bloom_fix_sparse_feedback.md` §NEXT; `bloom_investigation.md` |

Setting `feedback_clamp_log1p=[7,7,7]` on a `threshold_gate` nb eval produced a **byte-identical** rollout trajectory to no clamp (τ=0.5+clamp7 == τ=0.5, count/cell @T=35 = 6486.5 in both) — the clamp had **zero effect** even though the fed-back magnitude exceeded the ceiling (log1p 8.78 > 7). The rail either isn't wired into the family eval rollout, isn't consumed for `output_distribution` families, or the injected config value never reached `_parse_feedback_clamp`. **Cause NOT diagnosed** (est. ~15-min check: confirm the field is parsed on the eval path AND `_clamp_feedback` is actually called in the family rollout loop). **Tier 3:** no wrong *scored* output today (we score T=0; the clamp only touches the T>0 feedback we don't score), but **a safety rail that silently does nothing is a latent hazard** — a future bloom-mitigation or an operational forecast that trusts the clamp would be unprotected with no error signal, and it already **misled a rung-2 result** (we recorded "clamp doesn't help" when the truer statement may be "clamp wasn't applied"). Fix: verify the wiring; add a test that the clamp measurably bounds the feedback; if it is intentionally train-only, document that and fail-loud when it is set on an eval-only run.

---

### C-217: rollout-skill origins may leak across the train/validation boundary — every T>0 skill number would be optimistic

| Field | Value |
|-------|-------|
| ID | C-217 |
| Tier | 2 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — operational seat |
| Trigger | Pre-registering or scoring the T>0 rollout-skill curve (free-running or ancestral) before confirming the 36-future origin set is on the validation side of the train boundary |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/02_design.md` §7; `03_harness_and_invariants.md` §C-G4 |
| Cross-refs | FAO-02 locked eval (validation partition, `reference_fao02_locked_eval_framework.md`); the frozen lodestar ruler; C-112 (attribution hygiene) |

The T>0 ruler needs origins with a full 36-month realized future. These sit at the **early edge of the calibration partition** (origin + 36 ≤ max truth month), which is precisely where a train/validation boundary can be crossed. If any of those origins/months were seen in training, **every** rollout-skill number is optimistic and the ruler — which gates the whole bloom epic — is untrustworthy. **BLOCKER before pre-registration:** verify the origin set against the FAO-02 train/validation boundary (Hegre2019 partition discipline); if in-sample, re-pick origins from the validation side even at the cost of fewer/shorter-horizon origins. Tier 2: a structural read-validity fragility with a clear, imminent trigger (the first scored read).

---

### C-218: scoring the on-disk MEAN-feedback rollout as "deployed rollout skill" measures a broken-by-construction object

| Field | Value |
|-------|-------|
| ID | C-218 |
| Tier | 2 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — Salinas seat |
| Trigger | Reporting the GPU-free re-score of the existing `origin_*` dirs as the model's *deployed* rollout skill, rather than as a diagnostic of current behavior |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/04_roadmap.md` Phase 2; `02_design.md` §2 |
| Cross-refs | C-113 (the bloom); C-136 (rollout-confounded verdicts); C-126 (point-stability ≠ calibration); `plan_bloom_fix_sparse_feedback.md` §NEXT (H-SAMPLE) |
| Related work | Salinas2020 (DeepAR ancestral sampling) |

The rollout persisted on disk feeds back the **emit-mean**, which is not how a probabilistic recursive model is rolled out — DeepAR (Salinas2020) feeds back **ancestral samples**. The mean-feedback rollout is therefore **broken by construction**, and its bloom is partly a method artifact rather than a property of the model. Scoring it and labeling the result "deployed rollout skill" measures a strawman. The **deployed object is the ancestral (sample-feedback) rollout**; the skill verdict must be gated on that arm (dossier Phase 2b), with the GPU-free mean-feedback read explicitly labeled a *diagnostic of current behavior*. Tier 2: mislabeling here would send a corrupted "the rollout has/lacks skill" conclusion into every downstream fix decision (the exact corrupted-knowledge failure mode this epic exists to avoid).

---

### C-219: crps_all as a headline skill scalar is Goodhart-prone on the 99.7%-zero DGP — rewards timid-but-stable over honest-uncertainty

| Field | Value |
|-------|-------|
| ID | C-219 |
| Tier | 2 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — Gneiting/LeCun seats |
| Trigger | Reporting a single `crps_all` number per horizon as "skill", or ranking rollout variants on `crps_all` alone |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/02_design.md` §2 (metrics) |
| Cross-refs | C-211 (count-only self-zeroed underselling — crps-none/gate independence); the frozen lodestar crps-split; FAO-02 (twCRPS/PIT rejected); C-126 |
| Related work | Lerch2017 (Forecaster's Dilemma) |

On a ~99.7%-zero DGP, `crps_all` is dominated by the true-zero cells, so a **timid conservative-zero rollout** (e.g. τ≥0.8) can outscore an honestly-diffuse ensemble purely by being confidently zero — penalizing honest uncertainty (Lerch2017). **Guard (chair-ruled §6b):** the headline is the **`crps_all` / `crps_events` / `crps_none` split** + the locked FAO-02 **Brier / MCR / QS99** guardrails, read per horizon; `crps_all` is never reported alone. **NOT twCRPS and NOT PIT** — both are FAO-02-rejected and lab-tested-negative; they may return only after a fresh test re-earns them. CRPSS is computed only for the crossover visualization, never a decision metric. Tier 2: a metric-validity fragility that would silently certify the timid-but-stable rollout (τ) as "skillful" — the precise trap that motivates the ruler.

---

### C-220: the per-horizon scorer must consume the D×K sample cube, not the emit-mean

| Field | Value |
|-------|-------|
| ID | C-220 |
| Tier | 3 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — Gneiting seat |
| Trigger | The new `gather_all_horizons` loader / `rollout_skill_score.py` consuming `E[y]` instead of the per-cell D×K sample cube |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/` G1 loader (`rollout_skill_score.py`, to be built) |
| Cross-refs | C-219 (the metric-validity cluster); the frozen lodestar `crps_ensemble` |
| Related work | Gneiting2007 (strictly proper scoring) |

CRPS is strictly proper only when applied to the **predictive distribution** (the emitted D×K sample cube), not a point mean. If the per-horizon loader accidentally scores `E[y]`, the "CRPS" is a disguised absolute error that mis-credits sharpness/calibration. **Guard:** a unit test asserting the scored object is the sample cube; per-horizon calibration read via **MCR** (a locked guardrail), not PIT. Tier 3: a build-time correctness guard for a not-yet-written tool; caught cheaply by the test, but silent if omitted.

---

### C-221: |O|≈12 temporally-autocorrelated origins → iid-over-cells bootstrap CIs are wildly overconfident

| Field | Value |
|-------|-------|
| ID | C-221 |
| Tier | 3 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — Hyndman seat |
| Trigger | Computing any significance / KEEP claim on the rollout-skill curve with an iid-over-cells bootstrap |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/02_design.md` §6 (DQ2) |
| Cross-refs | C-112 (attribution hygiene / multi-seed); C-217 (same origin set) |

The 36-month-future window shrinks the origin set to ≈12 origins whose futures **overlap**, so the effective sample size is far below `12 × N_cells`. An iid-over-cells bootstrap ignores the temporal (and spatial) autocorrelation and yields absurdly tight CIs — a false-precision hazard for the crossover-horizon claim. **Fix:** compute CIs with a **block bootstrap over origins**; report widening CIs with horizon honestly. Tier 3: methodological, no silent output corruption, but would manufacture spurious significance.

---

### C-222: the free−oracle gap is not pure exposure bias — it is confounded by induced hidden-state drift

| Field | Value |
|-------|-------|
| ID | C-222 |
| Tier | 3 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — Hochreiter seat |
| Trigger | Attributing the entire `crps_free(h) − crps_oracle(h)` gap to the fed-back value, or calling it "the bloom's cost" without hedging |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/02_design.md` §3 |
| Cross-refs | C-113 (the runaway; the `freeze_h` ablation showed the state path is inert vs the runaway) |

Free-running and teacher-forced-oracle differ in the fed-back **input**, but the ConvLSTM hidden state `h_t` evolves from that input every step — so the gap = **input-exposure-bias ⊕ the induced hidden-state trajectory**, not cleanly "the fed-back value." **Fix:** relabel the oracle a *one-step-conditioned ceiling* (not "predictability ceiling"), and interpret the gap with the hedge; cite the retired-but-inert `freeze_h` result (C-113) as evidence the input path dominates, so the gap remains interpretable but not pure. Tier 3: an interpretation/labeling risk that would over-claim a clean exposure-bias decomposition.

---

### C-223: `[DEFERRED]` recursive rollout may not be the optimal product — direct-multi-horizon is a parked architectural alternative

| Field | Value |
|-------|-------|
| ID | C-223 |
| Tier | 4 |
| Source | expert-method-review (T>0 rollout skill ruler, 2026-07-25) — Hyndman seat (strongest live dissent, deferred) |
| Trigger | A large, growing free−oracle exposure-bias gap that PERSISTS after the sample-feedback fix (accumulation intrinsic to recursion) |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/02_design.md` §6; `02b_method_review.md` §6b |
| Cross-refs | C-125/C-126 (rollout-training premises); C-222 (the gap that would trigger this) |
| Related work | Makridakis2020 (M4 — recursive vs direct) |

Recursive rollout accumulates error by construction (the bloom is that pathology). Recursive was the **right pragmatic start** (1 model, ~36× cheaper *training* than a 36-separate-model direct scheme, horizon-flexible). The direct alternative that avoids accumulation is a **single-shot multi-horizon head** (run the expensive ConvLSTM-UNet body ONCE, read all 36 horizons off the final representation) — NOT a "decoder loop," which still runs 36 sequential passes and saves nothing. Even the single-shot head saves only the **35 extra decode passes**, which is **~10% of inference here** because history digestion **H≈335** months dominates (recursive = 335+36 vs direct = 335+1 body passes). So direct is not categorically cheaper — a real HydraNet architecture change with a modest, uncertain inference win, and possibly *less* accurate (one representation must serve all horizons). The oracle gap (C-222) already diagnoses whether accumulation is the problem, so no direct baseline is built now; this option is **parked** until that gap motivates it. Tier 4 (deferred architectural option — no current correctness impact); promote if the post-fix oracle gap says recursion is intrinsically capped.

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

### D-07: C-132/C-133 fix scope — minimal wrap vs structural enforcement

| Field | Value |
|-------|-------|
| ID | D-07 |
| Source | expert-code-review (wandb lifecycle, 2026-06-07) |
| Perspectives | Side A (Beck/Feathers/Martin-minimal): smallest change — wrap the hydranet `_execute_model_training` override body in `initialize_run("train")` (or delete the override so the base template runs) + a pinning test; ship now and unblock. Side B (GoF/Ousterhout/Hickey): that fixes only the instance; the overridable-template (C-133) + ambient-global-`wandb.run` (C-134) design reproduces the bug on the next subclass/phase — enforce the lifecycle in the base (non-overridable template / central invariant) and/or inject the logger instead of reading global state. |
| Resolution | **Open.** Cross-refs C-132/C-133/C-134. Decision gated on the "why does the override skip `finalize_training`?" investigation — if hydranet doesn't need to skip it, deleting the override (Side A, but structural) both unblocks and removes the divergence; otherwise a wrap + a cheap fail-loud/test (slice of Side B) is the low-regret middle. |

### D-08: Does the ZINB head dissolve the balancer (C-111)?

| Field | Value |
|-------|-------|
| ID | D-08 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Perspectives | Side A (the design / epic #97): a single ZINB likelihood means "nothing to weight" → the multi-task balancer and C-111 are dissolved **by construction**; this is the stated reason ZINB beats the freeze. Side B (Ousterhout/Kleppmann): the design **keeps focal on `by_*` as a separate loss** (`02_design §2`), so `mtloss.py` still stacks reg + cls losses and the Kendall balancer **still runs** — C-111 is only dissolved if π is trained *inside* the ZINB NLL. |
| Resolution | **Resolved 2026-06-10** → hurdle-NB with a class-weighted Bernoulli gate-term trained *inside* one joint NLL (focal replaced); both terms are NLLs → additive → the reg-vs-cls balancer is genuinely dissolved; #59 stays mooted correctly. See Gate Resolutions + 02_design D2/§2. |

---

### D-09: ZINB emit space — count-space `E[y]` vs `log1p(E[y])`

| Field | Value |
|-------|-------|
| ID | D-09 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Perspectives | Side A (the design, `02_design §4`): emit `E[y]` in **count space** and leave `inverse_transform_volume` unchanged. Side B (Nygard/Hickey): the inverse `expm1`s output channels (`feature_scaler.py:239-245`), so a count-space `E[y]` is **double-`expm1`'d** → re-explosion; emit `log1p(E[y])` so the existing inverse recovers `E[y]` (or tag the channel `identity`). |
| Resolution | **Resolved 2026-06-10** → Side B: emit `log1p(E[y])` so the existing `expm1` inverse recovers `E[y]`; a round-trip test is mandatory (#101). See Gate Resolutions + 02_design D3/§4. |

---

### D-10: Cure for the name-coupling — boundary adapter/schema vs needless indirection

| Field | Value |
|-------|-------|
| ID | D-10 |
| Source | expert-code-review (2026-06-24, naming-coupling review) |
| Perspectives | Hickey: the name complects identity + role + provenance; *decomplect* — pass role/identity as explicit data and an explicit external→internal schema, which is **simpler** and removes the ripple (C-174/C-173). Ousterhout: an adapter pays off **only if it is a deep module** (narrow interface — `role_of`/`targets`/`transform_of` — hiding the prefix mess); a wide or general "feature-registry framework" is needless indirection that adds its own complexity and should be rejected. |
| Resolution | **Open.** Likely synthesis: a **thin** boundary schema (≤3-method interface) — decomplect (Hickey) while keeping it shallow-to-use/deep-in-hiding (Ousterhout). Tie to C-160 (channel-role refactor is the natural vehicle) and C-174. |

---

### D-11: Is the disease the coupling, or just the 465× duplication?

| Field | Value |
|-------|-------|
| ID | D-11 |
| Source | expert-code-review (2026-06-24, naming-coupling review) |
| Perspectives | Martin/Kleppmann: the disease is the **coupling** — name-as-contract with no boundary schema; the prefix-role parsing is a latent correctness bug (C-173) a fixture fix does not address; add the contract. Cheapest-win/pragmatist: the disease is mostly the **465× fixture duplication** — a single-source-of-truth + fixture factory removes ~90% of the felt pain at ~10% of the cost; the adapter is a larger, deferrable bet. |
| Resolution | **Open.** Sequencing answer (not either/or): do the cheap single-source-of-truth + fixture factory **first** (low risk, kills most felt friction); treat the boundary adapter/schema (closes C-173) as the deliberate follow-up. See C-174 fix. |

---

### D-12: per-origin `torch.Generator` re-seed — common-random-numbers across origins

| Field | Value |
|-------|-------|
| ID | D-12 |
| Source | /expert-code-review (A-S10 #177), Kleppmann/Nygard vs the CRPS-program view |
| Perspectives | Kleppmann/Nygard: the D×K sampler re-seeds its `torch.Generator` from `torch_seed` at the start of **every** `generate_posterior_samples` call (per origin), so all origins draw the **same** random stream — undocumented as deliberate, so a future reader may "fix" it and break the S2 #121 determinism gate. Counter (CRPS-program): deterministic common-random-numbers is a **feature** (variance reduction across origins) and per-cell/per-origin CRPS is unbiased regardless — the reuse changes nothing the M-metrics measure. |
| Resolution | **Keep the behavior; document intent.** Add a one-line comment at the generator seed site (`hydranet_inference.py` `generate_posterior_samples`) stating the per-origin CRN re-seed is intentional (deterministic + unbiased for per-cell CRPS). Cross-refs C-209 (the S-axis fold) and the A-S8 /review-diff S-2 note. |

---

## Tech-Debt Backlog (demoted from register, review-rr 2026-06-05)

Demoted per the three-track model: Tier-4, mechanical-or-standing, single-file/single-developer scope — kept for traceability (full entries remain tagged `[DEMOTED]` in §Open Concerns) but no longer counted as active risks. Actionable as ordinary tech-debt, not governance risks.

| ID | One-line action | Cluster |
|----|-----------------|:--:|
| C-89 | Extract `_SumReducer` to `tests/conftest.py` (the `_tobit_config`/`tobit_config_3target` part is already done). | — |
| C-49 | Roadmap item: revisit flat→nested config schema if keys exceed ~50 or a feature needs 4+ grouped keys. | 5 |
| C-37 | Accepted trade-off: extract an `IVolumeHandler` Protocol only if an alternative implementation (lazy/GPU-resident) is needed. | 3 |
| C-85 | Add a `flip_probability` config key (currently hardcoded `0.5` in `training_engine.py:290-292`). | — |
| C-204 | Consolidate the inverse-softplus link (`nb_core.inverse_softplus` + `dense_nb_loss`/`truncated_nb_loss` `_inverse_softplus`) into one shared util when Epic B (#181) migrates the legacy losses. | Epic B |

---

## Resolved Concerns

<!-- 2026-07-27 register tidy (review-rr strategic): the entries below were resolved-in-place in §Open and physically relocated here. -->

### C-79: No pipeline-level reproducibility comparison test — RESOLVED

> **RESOLVED 2026-06-15 (`daab1c1`).** Determinism regression test added in `tests/test_training_engine.py` (`test_init_deterministic_regardless_of_prior_rng_state` + `test_training_run_is_reproducible`) — pins two-run weight-TENSOR identity. *(Awaiting physical relocation to §Resolved Concerns in the next register tidy.)*

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

### C-119: GPU runs are not bit-reproducible despite the reproducibility gate — RESOLVED

> **RESOLVED 2026-06-15 (`daab1c1`).** Root cause was **init-time RNG drift** (not CUDA kernels); fix = re-seed before init in `make()` + a determinism regression test (closes C-79). Bit-identical weights confirmed on the real config (GPU, multi-thread, dropout). Confounded prior results flagged in RESULTS_LOG / dossier 07 / #110. *(Awaiting physical relocation to §Resolved Concerns in the next register tidy.)*

| Field | Value |
|-------|-------|
| ID | C-119 |
| Tier | 1 |
| Source | repo-assimilation (2026-06-05) + C-111 bisect observation; **root-caused by the determinism investigation (2026-06-14)** |
| Trigger | Comparing any two single training runs (baseline-vs-experiment, an FAO eligibility row, a RESULTS_LOG entry, or the channel-role parity gate) **before the init re-seed fix lands** — run-to-run init variance silently confounds the delta |
| Location | **`views_hydranet/train/training_engine.py` `make()` (~L70-74: `choose_model` + `model.apply(init_fn)`)** draws weights from a torch-RNG state advanced a non-deterministic amount by work between the manager seed-lock (`hydranet_manager.py:279`) and `make()`; the re-seed at `training_engine.py:494` is post-init (too late). `reproducibility_gate.py::lock_entropy` (necessary but mis-placed relative to init). |
| Cross-refs | C-42/C-43 (reproducibility gate — "resolved" but **necessary-not-sufficient**: placement vs init matters); C-79 (no pipeline reproducibility test — the missing guard); C-160 (the parity gate this breaks); C-158 (volatility/multi-seed theme); C-112 (pre/post comparability); coord experiment #110 (verdict confounded) |

The gate locks seeds and requests deterministic algorithms, but same-config GPU retrains still diverge in magnitude: the C-111-bisect control retrain settled at CRPS ~1e7 vs the June-3 run's ~1e17 (same seed/config). The qualitative outcome (out-of-range vs in-range) reproduces; the numeric value does not. Any bisect/ablation comparing a single GPU retrain to a prior one must therefore treat magnitude deltas as possibly-spurious and rely on device-matched, ideally multi-seed comparisons (cf. C-112).

~~Tier 3 rationale: reliability of inference *about experiments*; affects how comparisons are designed, not the model's correctness. No silent corruption.~~ **Superseded — see Update 2026-06-14.**

**Update 2026-06-14 — ROOT-CAUSED, ESCALATED to Tier 1, FIX VALIDATED.** The 2026-06-05 hypothesis ("non-deterministic CUDA kernels — cannot force bitwise determinism") is **wrong**. A controlled bisection on the bounded hurdle-NB no-coords baseline (frozen data via `--saved`, identical code, seed 42) found the real cause is **init-time RNG drift**, not CUDA:
- `use_deterministic_algorithms(True, warn_only=False)` did **not raise** → no op lacks a deterministic impl (ConvTranspose2d/pooling are fine).
- **CPU** runs diverge → not CUDA-specific. **Verified single-thread** (`torch.get_num_threads()==1`) diverges → not threading. **`PYTHONHASHSEED=0`** diverges → not hash-ordering. **`dropout=0`** diverges → not the forward RNG. **Sampled-window data hashes are identical** → not data sampling.
- `make()` *in isolation* (lock→make immediately) is deterministic, but the **init-weights hash at training start differs across real-pipeline runs** → the model is initialised from a torch-RNG state advanced a non-deterministic amount by work between the manager's `lock_entropy` (`:279`) and `make()`. Different initial weights every run → ~20% downstream variance (no-coords FULL MCR sb **3.69 vs 2.99**, os **6.78 vs 8.47**; CRPS ±~20%).
- **Fix validated:** re-seeding (`lock_entropy`) immediately **before** init in `make()` yields **bit-identical** weights across two runs on the **real production config** (GPU, multi-threaded, dropout=0.15) — weight-tensor hash `5c8413bd…` identical, training loss identical to 5 decimals. One-line ordering fix (**Path A**); not yet applied.

**Why Tier 1 now:** this is **silent model-output-comparison corruption with no error signal**. It did not merely "affect comparison design" — it silently produced *wrong conclusions*: the coordinate experiment #110 read "coords made it worse" (coord 5.09 vs a baseline 2.55) when the **no-coords baseline alone swings 2.99–3.69 run-to-run**, and the FAO eligibility table / RESULTS_LOG single-run comparisons are likewise confounded. Resolves when Path A lands **and** a pipeline determinism regression test (C-79) pins two-run weight-tensor identity.

**Method caveat (record):** the saved **`.pt` file sha256 is NOT a reliable weight-identity check** — torch's `.pt` is a zip embedding file mtimes, so it differs even for identical weights. Use the **weight-tensor hash** (numpy `tobytes`), the **training loss**, or **MCR** as the determinism signal.

*Test-coverage shadow (test-review 2026-06-05):* the reproducibility envelope is uncharacterized — no test pins what is guaranteed vs not on GPU; cf. C-79. **The Path-A regression test closes this.**

---

### C-121: No automated regression guard for the C-113 autoregressive runaway — the only monitor is contractually blind — RESOLVED

> **RESOLVED 2026-07-27 (S8, Epic #193, ADR-070).** A fast regression guard now exists AND covers
> the actual fix, not just detection: `tests/distributions/test_rollout_feedback_bounds_bloom.py`
> (the sample-feedback invariant — the fed-back field is sparser than the mean, parametrized over
> the 3 deployable arms) + `tests/test_rollout_stability_guard.py` (the free-running attractor guard,
> `is_out_of_range` vs `DATA_LOG_MAX`), both running in seconds in the suite; `scripts/diagnose_io_gain.py`
> is the retrain-free attractor seam. A training-dynamics change that re-introduced the runaway now
> fails the suite in seconds, not after a ~40-min eval. *(Awaiting physical relocation to §Resolved
> Concerns in the next register tidy.)*

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

*Update (falsify, 2026-06-06) — now the ACUTE pre-retrain blocker:* a `/falsify` audit of the "clean base, ready to proceed" claim made this concrete. The repo is clean, but the **next planned step (rollout-training, ADR-058 candidate / `reports/2026-06-05_rollout_training_dossier/`) is a RETRAIN** — the exact trigger in this entry. With no guard, a rollout-training retrain could silently re-introduce the runaway. Sharpened trigger: **before any rollout-training (or other training-dynamics) retrain, this guard must exist.** A RED stub now marks the gap — `tests/test_falsification_clean_base.py::test_c113_runaway_regression_guard_exists` — and should turn green when the guard lands (the dossier's boundedness readout is its natural home). This is **step one** of the rollout-training build, not a parallel nicety.

*Update (2026-06-06) — first guard layer LANDED (#76):* the seconds-level mechanism guard now exists — `views_hydranet/utils/rollout_diagnostics.py::free_running_attractor` (SRP extraction, shared with `scripts/diagnose_io_gain.py`) + `tests/test_rollout_stability_guard.py` (contractive→bounded, expansive→flagged out-of-range); the falsify stub is now GREEN as a meta-guard. **Partial resolution:** this guards the runaway *detection mechanism* on controllable tiny models. The **remaining gap** is a real-artifact boundedness check wired into CI (load a `.pt`, assert the 36-step rollout stays in-range) — which folds naturally into the rollout-training *boundedness readout* (#77/#78). C-121 stays **OPEN** (downgraded in practice) until that real-model check runs in CI.

### C-162: #110 decision rule's run-to-run Δ_noise is identically zero under the C-119 determinism fix — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-162 |
| Tier | 2 |
| Source | /falsify (2026-06-16) — P2/P4 |
| Trigger | Applying the #110 pre-registered decision rule (run the 8-sample baseline **twice**, set `Δ_noise` = run-to-run difference) to call the coordinate ship/drop verdict |
| Location | GitHub #110 "Pre-registered DECISION RULE"; dossier `05_analysis_plan.md` operating-point banner |
| Cross-refs | C-119 (determinism fix — the root of the degeneracy), M-R2 (prior no-CI method risk), `scripts/mcr_readout.py:81` `_bootstrap_mcr_ci` (the correct within-run noise band) |

The rule derives the noise band from **two same-seed `--saved` runs**. But **C-119 made same-seed runs bit-identical** (proven 2026-06-15: 78/78 `y_pred` `np.array_equal`, PREDICTIONS IDENTICAL: True) → `Δ_noise ≡ 0` → the rule has **no noise floor** → coordinates would be accepted on **any** nonzero MCR improvement = a **false-positive verdict with no error signal**. The noise band must instead come from the **within-run bootstrap CI** that `mcr_readout` already computes. **Tier 2:** silent decision-incorrectness (unsound go/no-go) under the realistic action of applying the rule; the two-run framing is a determinism-era leftover.

**RESOLVED 2026-06-16 (#129):** #110 body + dossier `05` rewritten — coords win iff the coords-on FULL-MCR 95% bootstrap CI (`mcr_readout._bootstrap_mcr_ci`, one run) is non-overlapping-and-lower than the baseline CI on ≥2/3 targets + CRPS non-inferior; overlapping → escalate. All run-to-run/Δ_noise language removed; falsify guard green.

---

### C-178: dead-ReLU regression body silently emits identically-zero predictions for rare targets under the hurdle mask — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-178 |
| Tier | 1 |
| Source | /falsify "regression head + mask 100% correct" (2026-06-25) — P2, FALSIFIED |
| Trigger | Training a non-`hurdle_nb` hurdle point body (`output_distribution='hurdle_shrinkage'` or `'hurdle_lognormal'`, with `reg_activation` unset) under the `active_window` mask (heavy zero-supervision) on a sparse target — i.e. exactly the #66/#73 runs |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:85` (`self._reg_activation = F.softplus if output_distribution=="hurdle_nb" else F.relu`); interacts with the active_window zero-supervision in `training_engine.py` |
| Cross-refs | [[project_body_loss_not_the_lever]] (#66 flatline + #73 shrinkage puzzle — both explained by this); test `tests/test_falsify_reg_head_dead_relu.py` |

The regression head uses **ReLU** for every output_distribution except `hurdle_nb`. Under the active_window mask the rare targets' pre-activation `H_reg` drifts **negative on 100% of cells (including event cells)**, so `ReLU` clamps the body to **identically 0** and — because `ReLU'(<0)=0` — **no gradient flows back**, making it unrecoverably DEAD. Verified by real forward on the aw seed-11 artifact: lr_ns_best / lr_os_best emit `out_reg==0` everywhere (pre-activation max < 0) while the **gate fires normally** (σ up to 1.0) — so the composed `E[y]` is ~0 not because of the gate but because the body is dead. This is the silent mechanism behind the #66 ns/os flatline (MCR_pos=0.000, CRPS_pos=mean_truth exactly) and the #73 shrinkage "puzzle" (no body loss can resurrect a zero-gradient ReLU). **Tier 1:** silent model-output incorrectness with no error signal (the forecast for 2/3 targets is identically 0). **Mitigation already in place for production:** the shipped floor uses `output_distribution='hurdle_nb'` → **softplus** (always positive, non-zero gradient) → NOT affected; the defect is gated to the experimental hurdle point/shrinkage/lognormal bodies. **FIXED (shipped `ee5a593`, ADR-063):** softplus is now the architecture default for ALL hurdle bodies (`HydraBNUNet06_LSTM4:85`, `startswith("hurdle")`); failing test `test_falsify_reg_head_dead_relu.py` is green. The reload-completeness follow-on is **C-179**.

---


### C-214: the eval-side autoregressive forensic was silently DEAD for every nb/zinb run (the `return_params` early-return short-circuited its finalize) — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-214 |
| Tier | 3 |
| Source | Diagnostic-plotting-suite deep review (multi-agent audit, 2026-07-24) |
| Trigger | Running an `nb`/`zinb` family and expecting the eval-side autoregressive/rollout forensic PNG (`biopsy_autoregressive`) — it never rendered |
| Location | `views_hydranet/utils/hydranet_inference.py::predict` (the `return_params=True` early-return at the family/D×K path vs the Stage-5 finalize that follows it) |
| Cross-refs | C-213 (the training-side family-forensic counterpart); ADR-067 / A-S8 (introduced the `return_params` branch that caused the regression) |

`generate_posterior_samples`'s family branch calls `predict(..., return_params=True)`; inside `predict`, `return_params=True` hits an **early `return`** (the pre-emit params for the D×K sampler) that sits **before** the Stage-5 "Finalize Biopsy" block which calls `self.viz.biopsy_autoregressive(...)`. So for every `nb`/`zinb` run the eval autoregressive rollout forensic was **never produced** — the truth/pred accumulators were even filled during the rollout, then discarded. This is a **regression introduced by ADR-067/A-S8** (the `return_params` family branch): the AR forensic renders only for the quantile (Path A) and legacy/standard/hurdle paths (which pass `return_params=False`). **Tier 3 (diagnostic coverage gap):** a *missing* artifact for the families under active development, not a wrong plot — training/loss/sampler/scoring unaffected. The method itself is family-clean (its `pred_accumulator` holds per-target `log1p(E[y])`).

> **RESOLVED 2026-07-24, same session as discovery.** Extracted the finalize into `_finalize_ar_forensic(...)` and call it **once, right after the rollout loop — before the `return_params` early-return** — guarded by `sample_idx == 0 and self.viz.active`, so it fires for BOTH the family (`return_params=True`) and legacy paths. Regression test `tests/distributions/test_sampler_dxk.py::test_predict_return_params_still_renders_ar_forensic` (a `zinb` `predict(return_params=True)` with a spied viz asserts `biopsy_autoregressive` is called once with per-target `channel_names`). Full inference/sampler suites green; ruff clean. The eval rollout forensic now renders for nb/zinb runs.

### C-213: the `REGRESSION FORENSIC` dossier is point-head-era — for a family head it plots target-0's `(μ, θ, π)` mislabeled as the 3 targets → silently misled analysis — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-213 |
| Tier | 3 |
| Source | Forensic-plot inspection during the ZINB 3×300 run (2026-07-24) |
| Trigger | Reading the per-lesson `REGRESSION FORENSIC` dossier (`biopsy_feature_dossier`) for an `n_params>1` distribution family (nb/zinb) before this fix — the panels are per-**param**, not per-target |
| Location | `views_hydranet/train/training_engine.py` (the `forensics.record` site) + `views_hydranet/utils/training_forensics.py` (`TrainingForensics`) + `views_hydranet/utils/visual_diagnostics.py` (`biopsy_feature_dossier`) |
| Cross-refs | C-211 (self-zeroed vs gated scoring); C-212 (the ZINB NaN — why the ZINB body is reliable to reuse); ADR-067 (the family subsystem these plots visualize) |

`TrainingForensics.record` was fed `t1_pred[:, j]` — one raw reg channel per target index. That is correct for a point head (1 channel/target) but a distribution family emits `n_params` channels **per target, target-major** (the loss slices `t1_pred[:, j*npar:(j+1)*npar]`). So for ZINB (`n_params=3`) the three "target" panels actually plot **target-0's `μ` (idx0), `θ` (idx1) and `π` (idx2)** under the sb/ns/os labels — and in the wrong space (raw natural-space params vs the log1p-space truth). The dead giveaway is the "os" panel pinned flat ~0.95 (a `sigmoid` π, not a magnitude). This **silently misled analysis**: a per-target "calibration difference" read off the plot was really one target's three parameters. The sibling `biopsy_training_performance` had already been made family-aware (`log1p(family.mean)` per target); this dossier never was. **Tier 3 (diagnostic/methodology):** training, loss, sampler, and the frozen-ruler scoring are all family-aware and correct — zero impact on model or results; the only harm is misreading the diagnostic.

> **RESOLVED 2026-07-24, same session as discovery.** (1) **Calibration fix:** the record site now collapses each target's `n_params` slice via a shared `_family_target_log1p_mean(reg, family)` helper (which also DRY'd the biopsy's inline copy) and records per-target `log1p(family.mean)` — the honest self-zeroed forecast E[y], comparable to the log1p truth. Point/legacy heads keep the single-channel path (byte-identical). (2) **Parameter-health frame (upgrade):** a new `TrainingForensics.record_params` path stores the activated params; `finalize_lesson` computes per-target `μ̄`, θ **cross-cell CoV** (`std/mean` over ACTIVE cells) and π mean/min/max; `biopsy_feature_dossier` renders three new rows (μ̄, θ-CoV with the F1 guide lines **0.10 health / 0.02 collapse**, π mean+[min,max]) for family heads only (auto via key-presence; legacy dossiers unchanged) — turning the pre-registered F1 falsifier into a live per-lesson trace for **all** targets (ns/os params were previously never plotted). 8 regression tests in `tests/test_training_forensics.py` (calibration collapse C-213 + param-health math + point-head parity); render-verified on synthetic zinb/nb/legacy dossiers; full suite green. **Fix lands for future runs** (the current ZINB process runs the old code; its panels are mentally re-mapped to sb's μ/θ/π).

### C-212: `NBCore.log_prob_zero`'s `mu/(theta+mu)` float32 saturation → `log1p(-1)` singularity → NaN BACKWARD (finite forward) → killed 2/3 ZINB seeds mid-training — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-212 |
| Tier | 2 |
| Source | ZINB 3×300 M1 run failure + numerical falsification/debug (2026-07-23/24) |
| Trigger | Training a self-zeroed family (`zinb`) — or any future family/loss calling `NBCore.log_prob_zero` — where `theta` drifts to the clamp floor (1e-6) while `mu` grows past ~17 (so float32 `theta+mu` rounds to `mu`); OR any refactor that reintroduces a `mu/(theta+mu)` / `theta/(theta+mu)` form into a differentiated log, reviving the saturation |
| Location | `views_hydranet/distributions/nb_core.py::NBCore.log_prob_zero` (fix site); `zero_inflated_negative_binomial.py::nll` (the mixture caller whose `logaddexp`/`torch.where` masked the `-inf` forward while the backward went NaN); `negative_binomial.py::prob_positive` (also calls it, but forward-only in scoring → unaffected); `utils/integrity_guardian.py` (the fail-loud catch that made it a crash, not silent corruption) |
| Cross-refs | C-199 (the NBCore clamp / numerical-stability guard this lives beside); C-202 (θ head-channel gradient bound — related-but-distinct: that one is genuinely fine); C-211 (self-zeroed scoring); C-208 (sampler GoF) |

`NBCore.log_prob_zero` computed `theta * log1p(-mu/(theta+mu))`. In **float32**, once `theta < ½·ULP(mu)` the sum `theta+mu` rounds to `mu`, so `mu/(theta+mu)` becomes **EXACTLY 1.0** and `log1p(-1) = -inf`. The **forward** value survived because ZINB's mixture wraps it in `logaddexp(log_pi, log1m_pi + log_prob_zero)` / `torch.where`, masking the `-inf` to a finite number — but the **backward** hit `d/dz log1p(z)|₋₁ = 1/0 = inf`, and the mask multiplied it by 0 → `0·inf = NaN`, which the mean reduction sprayed to all 85 model params (`enc_conv0` through the LSTM gates). This killed **2 of 3 ZINB seeds** in the first 3×300 M1 run (seed 42 @ lesson 18, seed 44 @ lesson 29; seed 43 survived). **ZINB-specific:** only ZINB's mixture NLL calls `log_prob_zero` — the all-cell NB training NLL uses `NBCore.log_prob` only (delegates to `torch.distributions.NegativeBinomial`, saturation-robust for `y>0`), so all 3 NB seeds trained fine. `clip_grad_norm` was ON but useless: the `IntegrityGuardian` checks raw grads **before** the clip, and norm-clip can't rescue a NaN anyway. **Tier 2:** structural fragility that reliably crashed training under a realistic param regime — loud (fail-loud guardian), not silent, so not Tier 1.

> **RESOLVED 2026-07-24, same session as discovery.** Rewrote `log_prob_zero` to the mathematically-identical, singularity-free `-theta * log1p(mu/theta)` (verified equal to the old form to float precision in all well-conditioned regimes; C-201 small-`mu` accuracy preserved; no new overflow — `mu/theta` overflows only at `mu > 3e32`). Diagnosis chain: exhaustive param×count sweep proved the family NLL gradient is otherwise finite → an instrumented full-run guardian showed **all-NaN** (`inf=0`) from a finite loss (675) originating at the reg head → a 2-cell repro pinned `mu/(theta+mu) == 1.0` exactly. Added 3 regression tests (`test_nb_core.py::test_log_prob_zero_gradient_finite_at_probs_saturation` with a self-validating `ratio==1.0` assert + `test_log_prob_gradient_finite_at_probs_saturation` positive-branch guard; `test_zero_inflated_negative_binomial.py::test_nll_gradient_finite_when_theta_floors_and_mu_large`). Confirmed in **real training**: seed 42 (previously dead @ lesson 18) now completes 40 lessons, final loss 476, no explosion. Full suite green (0 new failures). **Residual (separate concern):** the fix makes training NaN-robust, but the *divergence* that drove `theta→floor`/`mu→large` (a degenerate basin) is a conditioning issue — the A-S9 π-ridge (`pi_penalty_weight`, currently off) is the candidate lever for the ZINB 3×300 re-run if divergence recurs.

---

### C-208: the hand-rolled generator-aware Gamma-Poisson sampler is guarded only for its MEAN — a future edit could skew the dispersion and silently corrupt CRPS — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-208 |
| Tier | 2 |
| Source | /expert-code-review (A-S10 #177), Feathers/Beck/Nygard |
| Trigger | Refactoring `nb_core._standard_gamma` or `NBCore.sample` (the Marsaglia-Tsang Gamma-Poisson), or changing `_EPS`/the `a<1` boost path, while only the mean-recovery test guards it |
| Location | `views_hydranet/distributions/nb_core.py:82-106` (`_standard_gamma`), `:137-156` (`NBCore.sample`); guarded only by `tests/distributions/test_nb_core.py:31-40` (mean only) |
| Cross-refs | C-202 (θ gradient bound); C-3 (generator determinism); A-S12 (#179) CRPS M2 comparison |

The generator-aware Gamma sampler is the subsystem's most intricate numerics (a vectorised Marsaglia-Tsang rejection loop on `torch.randn`/`torch.rand`, chosen because `torch.distributions` ignores a `Generator` — the S2 #121 determinism contract). It is distributionally **correct today** (empirical `Var ≈ mu + mu²/theta` within ~1% across `(mu,theta)`, verified in this review), but **no test pins its variance/dispersion** — only `mean ≈ mu` (`test_nb_core.py:40`). **Tier 2:** CRPS — the spread-driven metric that gates the `nb`-vs-`zinb` M2 decision (#179) — would be silently corrupted by any future edit that preserves the mean while skewing the dispersion; the mean-only test stays green and no error fires. Not an A-S11 blocker (the sampler is correct now). Fix: a `Var[sample] ≈ mu + mu²/theta` regression + a χ²/K-S goodness-of-fit at 2–3 `(mu,theta)`.

> **RESOLVED 2026-07-21 (A-S10 #177), same session as discovery.** Added two regression guards to `tests/distributions/test_nb_core.py`: `test_sample_variance_recovers_nb_dispersion` (asserts `Var[sample] ≈ mu + mu²/theta` within rtol 0.05 across 4 `(mu,theta)`) and `test_sample_zero_fraction_matches_prob_zero` (an **independent** goodness-of-fit — empirical `P(Y=0)` vs the analytic `NBCore.prob_zero`, which the sampler does not use, within atol 0.01). A future edit that skews the dispersion or the low-count spread now fails CI loudly instead of silently corrupting CRPS. The sampler was empirically correct at discovery; this closes the test-coverage gap that let that correctness go unguarded.

---

### C-199: per-cell ZINB `θ`/`π` ride on ~1% of cells with dead gradients at the conflict operating point → collapse / seed-instability without informed init — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-199 |
| Tier | 2 |
| Source | /falsify "amortized per-cell ZINB estimation is the right way" (2026-07-20; sim `scratchpad/zinb_falsify.py`) |
| Trigger | Training a `zinb` (or per-cell-`θ` `nb`) head with default channel init and an **unweighted** mean ZINB NLL |
| Location | planned `views_hydranet/distributions/` NB/ZINB head init + `nll`; A-S3 (#170) / A-S4 (#171) / A-S7 (#174) |
| Cross-refs | C-198 (transform), C-146 (ZINB vs hurdle), C-200/C-201; the quantile-head dead-fan init gotcha; F1 falsifier (05_analysis_plan) |

Simulation (N=1e6, zero-rate 0.989, positives 1.13%): **98.8% of the Fisher information about `π` lives in the ~1% positive cells**, and at the conflict operating point the gradients are near-dead — `|dNLL/d logit π|` collapses from 0.30 at π=0.5 to **0.0036 at π=0.99**, and `|dNLL/d η_θ|` **vanishes as θ→∞** (8e-7 at θ=500). So the identifying signal is a tiny, weakly-gradiented sliver; a default/zero-init head starts stuck with almost no signal to escape → `θ`/`π` collapse to constants (the F1 falsifier — reduces to the global-θ baseline this whole ADR exists to beat). **Tier 2:** structural fragility that silently defeats the feature's purpose under the realistic default (no init / unweighted loss). Fix: **informed init** (`π`≈empirical zero-rate, `θ`≈global-`θ`) as a *required* part of the family, active-cell weighting as a first-class `nll` option, and a seed-variance monitor on the `θ`/`π` fields.

> **Merged 2026-07-21 (/review-diff F-1, A-S7 #174):** the family body-masking path shares this entry's "graceful masking" scope. A family run with `body_mask != "none"` **and** `qs99_weight > 0` reaches `training_engine.py:355` (`error = target_j[mask] - pred_j[mask]`) where `pred_j[mask]` is `[N, n_params]` vs `target_j[mask]` `[N]` → **fail-loud** broadcast crash (not silent). The QS99 μ-pinball is also semantically undefined over a family's `(μ,θ[,π])` param-vector. Low severity (fail-loud, needs a nonsensical config combo; the default family config has `body_mask="none"` + no qs99). **A-S9 hardening action:** when switching family masking to the `nll` `weight=` path, also add a config guard rejecting `family + qs99_weight>0`. Cross-ref C-207 (the sibling multi-channel-slice concern, now resolved).

> **RESOLVED 2026-07-21 (A-S9 #176).** The structural fixes shipped: **informed init** landed already via C-203 (`initial_raw_bias` on the ABC — `π`≈empirical zero-rate, `θ`≈global-`θ`), and **active-cell weighting is now a first-class `nll` option** — the family body-mask path passes `weight=mask` into `family.nll` (`weighted_nll_mean`) instead of boolean-indexing (`training_engine.py`, C-199), and the merged F-1 crash is fixed (qs99 skipped for `FamilyLoss`). Verified by `test_penalty_mask_wiring.py` (`weight=` equals the boolean-index; empty mask = graph-connected 0; family+body_mask step finite with grads). The seed-variance **monitor** (a detection aid, not the fix) rides with the A-S11 GPU smoke where seed-stability is measured on real data.

---

### C-200: ZINB `π` is non-identified in deep-zero cells (the `π/μ` ridge) — free per-cell `π` needs a prior/penalty — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-200 |
| Tier | 3 |
| Source | /falsify (2026-07-20) |
| Trigger | Letting per-cell `π` and `μ` be free functions of the **same** features/backbone with no regularization |
| Location | planned `views_hydranet/distributions/zero_inflated_negative_binomial.py` |
| Cross-refs | C-199, C-146, C-201 |

For cells that are only ever zero, the likelihood depends solely on `P(Y=0)=π+(1−π)·NB(0)`, so `(π,μ)` is a confounded ridge — `π` is identified only by **excess** zeros beyond what `NB(μ,θ)` explains, which is absent where `μ→0`. With `π` and `μ` both free on the same features, gradient descent can park `π` anywhere on the ridge in deep-zero regions. **Tier 3:** not silent-corruption of a shipped forecast, but a modelling-stability/interpretability risk that inflates variance and can mask whether ZINB is genuinely doing zero-inflation. Fix: a mild prior/penalty on `π` (or a documented deep-zero handling); if `π` will not identify, that is positive evidence for the **hurdle** form (gate owns zeros, no `π`) — the `nb` vs `zinb` vs `hurdle_nb` M2 comparison (#179).

> **RESOLVED 2026-07-21 (A-S9 #176).** The mild π-ridge prior is wired: `ZINBFamily.pi_penalty` (via the C-205 `parameter_penalty` ABC hook) pulls `logit(pi)` toward `pi_penalty_prior_logit`, added to the reg loss as `pi_penalty_weight * ridge` (qs99/decay additive-penalty precedent, `training_engine.py`), gated by the two config fields (`None`/0 ⇒ byte-identical no-op). A run that wants to break the deep-zero `π/μ` ridge now sets `pi_penalty_weight`; the M2 comparison (#179) will judge whether it earns its place vs the hurdle form. Verified by `test_penalty_mask_wiring.py` (finite, shifts the loss, grads reach the head; nb no-op).

---

### C-202: NB `θ` value-clamp bounds the likelihood but not its gradient — `θ` driven to the `1e-6` floor backprops ~1e6 into the `θ` head channel — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-202 |
| Tier | 2 |
| Source | /code-review max (2026-07-20), A-S3 (#170) finder + numerical verify |
| Trigger | Training an `nb`/`zinb` head where a per-cell learned `θ` is pushed toward the `NBCore` `_EPS=1e-6` floor (heavy-tailed / near-Poisson-overdispersed cells) |
| Location | `views_hydranet/distributions/nb_core.py` `_clamp`/`log_prob` (`θ` floor); consumed by `NegativeBinomialFamily.nll` (A-S3 #170) and ZINB (A-S4 #171); loss wiring A-S7 (#174) |
| Cross-refs | C-199 (the *dead*-gradient extreme — this is the *exploding*-gradient counterpart at the opposite end of the `θ` range); C-200; C-8-style per-cell θ instability |

`NBCore._clamp` floors `θ` at `1e-6` so `log_prob` stays finite, but the value-clamp does **not** bound the gradient: `d log_prob / d θ ≈ digamma(θ) ~ 1/θ`, so as a cell's `θ → 1e-6` the score wrt `θ` explodes — numerically **~1e6 at θ=1e-6, ~1e4 at θ=1e-4** (verified). A cell whose learned `θ` is driven to the floor then backpropagates an enormous gradient through the `θ` head channel. **Tier 2:** structural fragility that can silently destabilize training of the very per-cell `θ` head this epic introduces (esp. under SGD / large LR), with no NaN or error to flag it — the loss stays finite while the update blows up. Fix (decide at A-S7 loss wiring / A-S9 hardening): a **soft** lower bound on `θ` (e.g. `θ_min + softplus(...)`) instead of a hard value-clamp, and/or `θ`-channel gradient clipping, and/or the C-199 prior pulling `θ` toward the global baseline.

> **RESOLVED 2026-07-21 (A-S9 #176) — by proof, no forward change.** The feared `~1e6` is `dNLL/dθ` (θ-space), **not** the raw-channel gradient the optimizer sees. The head emits `θ` via `softplus`, and the chain rule `dθ/d(raw) = sigmoid(raw) → 0` as `θ → 0` **cancels** the `1/θ` term at the head channel. Numerically verified: `|dNLL/d(raw_θ)|` stays **≤ 1** in the floor regime (raw_θ ≤ -10 ⇒ θ ≲ 5e-5) for any `y`, and →0 below the `_EPS` floor (`test_theta_gradient_bound.py`, `nb`+`zinb`, y up to 5000). No forward-distorting θ-floor added — it would have harmed the heavy tail (ξ≈0.8) this epic needs. (Large gradients at *moderate* θ with an *extreme* count are legitimate heavy-tail signal, not the floor pathology; the existing opt-in global `clip_grad_norm` at `training_engine.py:873` bounds them as belt-and-suspenders.)

---

### C-205: ZINB `pi_penalty` (the C-200 π-ridge regularizer) is a concrete-`ZINBFamily`-only method, off the `DistributionFamily` ABC — the loss wiring must `isinstance`-branch to reach it — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-205 |
| Tier | 3 |
| Source | /code-review max (2026-07-20), A-S4 (#171) altitude finder |
| Trigger | A-S7 (#174) loss wiring adds the C-200 π/μ-ridge penalty to the training loss while holding a `DistributionFamily` reference (not a concrete `ZINBFamily`) |
| Location | `views_hydranet/distributions/zero_inflated_negative_binomial.py` `pi_penalty`; the `DistributionFamily` ABC `base.py` |
| Cross-refs | C-203 (the symmetric seam just resolved for `initial_raw_bias` — promoted NB-only → ABC); C-200 (the ridge the penalty regularizes); C-146 |

`pi_penalty(params, *, prior_logit, scale, weight=None)` implements the C-200 mild π-ridge prior, but it lives only on the concrete `ZINBFamily`, not the `DistributionFamily` ABC. When A-S7 wires it into the loss, the loss holds the abstraction (ADR-067 DIP; the CIC states consumers "hold the ABC, never a concrete class"), so reaching `pi_penalty` forces an `isinstance(fam, ZINBFamily)` branch — exactly the per-family dispatch the subsystem exists to remove. **Tier 3:** coupling/maintainability — a concrete-only method a family-agnostic consumer must special-case. Fix: promote to a **default-0** ABC hook (e.g. `parameter_penalty(params, *, weight=None, **cfg) -> 0`) that `ZINBFamily` overrides — resolve at A-S7 when the loss-consumption shape (how `prior_logit`/`scale` arrive from config) is known. Deferred now (like C-203 was before its resolution point) rather than force a premature ABC signature. Currently harmless: no loss consumes it yet.

> **RESOLVED 2026-07-21 (A-S9 #176).** Promoted to a **default-0 concrete ABC hook** `parameter_penalty(params, *, prior_logit=0.0, scale=0.0, weight=None)` on `DistributionFamily` (`base.py`, torch-free via `params.new_zeros(())`); `ZINBFamily` overrides it → its `pi_penalty`, `NegativeBinomialFamily` inherits the 0. The A-S9 loss wiring calls `loss_fn_j.family.parameter_penalty(...)` **family-agnostically** — no `isinstance(ZINBFamily)`. CIC `DistributionFamily.md` §3 documents the concrete hook (distinct from the abstract seven). Verified by `test_parameter_penalty.py` (nb 0, zinb = its `pi_penalty`, default scale=0 ⇒ 0). Mirrors C-203's `initial_raw_bias` promotion.

---

### C-206: `n_head_samples` (K) is a config field but is NOT captured in the reproducibility snapshot — a D×K run will not be reproducible from its manifest until A-S8 adds it — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-206 |
| Tier | 3 |
| Source | /code-review (2026-07-21), A-S5 (#172) cross-file finder |
| Trigger | A-S8 (#175) wires `n_head_samples` into the inference D×K sampler (so K actually changes forecasts) without adding it to the training `config_snapshot` |
| Location | `views_hydranet/train/train_model.py:85` (`config_snapshot = {k: config[k] for k in arch_keys}`); the new field `views_hydranet/utils/config_initializer.py` `n_head_samples` |
| Cross-refs | C-43 (manifest audit); A-S8 (#175); the `n_quantiles`/`reg_activation` snapshot precedent (`train_model.py:101-105`) |

The persisted training `config_snapshot` is a **selective** `arch_keys` dict, not a full-config dump — so A-S5 adding `n_head_samples` correctly does NOT perturb legacy manifests (the #172 byte-identical AC holds). **But** once A-S8 makes K change the sampled `[T,H,W,C,S]` cube, a run's forecast depends on `n_head_samples`, and the manifest would omit it (like `n_quantiles`/`reg_activation` are conditionally captured) — two runs with different K would share a manifest, silently defeating reproducibility. **Tier 3:** a reproducibility gap that only bites when A-S8 wires the sampler; harmless at A-S5 (K is unused). Fix: in A-S8, add `n_head_samples` to `config_snapshot` (and, if it gates behaviour, to `reproducibility_gate.audit_manifest`). A-S5 correctly does not wire it.

> **RESOLVED 2026-07-21 (A-S8 #175).** K now (a) changes the sampled cube — `generate_posterior_samples` reads `posterior_K = config["n_head_samples"]` and fills `S = D×K` (`hydranet_inference.py`) — and (b) rides in the manifest: `train_model.py:110-115` adds `config_snapshot["n_head_samples"] = config.get("n_head_samples", 1)` inside the `elif resolve_family(_od) is not None:` branch, so it is captured **only for families** (legacy K=1 is unused ⇒ the selective `arch_keys` snapshot stays byte-identical, preserving the A-S5 #172 AC). Two family runs with different K no longer share a snapshot. Verified: full suite green (1053 passed, only the known 7 pre-existing) + determinism gate (`test_inference_orchestrator_pf`, F3_06).

---

### C-198: per-cell NB/ZINB loss inherits `to_raw_counts` hardcoded `expm1` → silently wrong loss under any non-`log1p` target transform — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-198 |
| Tier | 2 |
| Source | NB/ZINB parameterization design discussion (2026-07-20) |
| Trigger | Pairing an `nb`/`zinb` body with a target `transformations` other than `log1p` (e.g. `asinh`, identity/`none`), OR reusing `count_target_bridge.to_raw_counts` (hardcoded `torch.expm1`) unchanged in the new per-cell NB/ZINB loss |
| Location | `views_hydranet/utils/count_target_bridge.py` (`to_raw_counts` hardcodes `expm1`); planned `views_hydranet/distributions/` NB/ZINB `nll` + wiring in `training_engine`/`choose_loss` |
| Cross-refs | C-140/D-09 (emit-side sibling — count-space `E[y]` double-`expm1`, RESOLVED via `log1p(E[y])`); C-113 (`expm1`-amplified runaway); ADR-067; ADR-021 |

An NB/ZINB likelihood is defined on **raw counts**, so the loss must recover raw counts from the transformed target. `to_raw_counts` applies `expm1` **unconditionally** — i.e. it assumes the configured target transform is `log1p`. Under any other configured transform the recovered "counts" are wrong and the likelihood/gradients are **silently** wrong (no error). This is the loss/target-recovery analogue of the emit-side C-140 (resolved by keeping emit in `log1p(E[y])` so the config inverse recovers it). **Tier 2:** silent training corruption under a realistic config change; the fix must ship with the family. Fix (agreed, ADR-067): de-transform the target at the loss boundary using the **config's declared inverse**. The pipeline already has a config-driven registry — `config_initializer.TRANSFORMS[method] = (forward, inverse)` (`config_initializer.py:16-20`; only 3 methods: `log1p→expm1`, `asinh→sinh`, `identity`), consumed by `FeatureScaler.inverse_transform_volume`. But those are **numpy** funcs and the loss runs on **torch/GPU** tensors, which is why `to_raw_counts` hardcoded `torch.expm1` (log1p only). Fix = add a small **torch mirror** of `TRANSFORMS`' inverse and recover counts via the *configured* method — never a hardcoded `expm1`; and fail-loud at config load if an `nb`/`zinb` body is paired with a target transform whose inverse is not count-compatible. Pinned in ADR-067 §3 + the `NBDistLoss`/`DistributionFamily` CIC; acceptance of A-S3 (#170) + A-S7 (#174).

> **RESOLVED 2026-07-21 (A-S7 #174).** Closed via the **fail-loud** half (the simpler, sufficient fix): `validate_family_requires_log1p_targets` (`config_initializer.py:691-711`, a `model_validator`) raises at config load if `output_distribution ∈ family_names()` and any regression target is not in `transformations["log1p"]`. `log1p` is the **only** count-compatible transform (its inverse `expm1` yields integer-scale counts; `asinh`/`identity` do not), so requiring it makes `to_raw_counts`' hardcoded `expm1` provably correct — no torch transform-mirror needed. Documented in `docs/CICs/HydraNetConfig.md` §6 (Family Targets Not log1p-Transformed); verified by `tests/distributions/test_loss_wiring.py::test_c198_family_requires_log1p_targets` (nb+log1p constructs; nb+identity raises). The silent-wrong-detransform path no longer exists.

---

### C-207: after A-S6 a family reg head is widened to `n_params` channels/target, but the training-engine loss loop still slices 1 channel/target for any non-`QuantileLoss` → a family run trains on silently mis-mapped channels until A-S7 — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-207 |
| Tier | 2 |
| Source | /code-review (2026-07-21), A-S6 (#173) cross-file finder |
| Trigger | Running training with `output_distribution ∈ {nb, zinb}` in the window **after A-S6 (#173) and before A-S7 (#174)** wires the family-aware loss |
| Location | `views_hydranet/train/training_engine.py:319-323` (the per-target reg slice: `if isinstance(loss_fn_j, QuantileLoss): … else: pred_j = t1_pred_for_loss[:, j]`); consumes the widened `out.reg` from `HydraBNrecurrentUnet_06_LSTM4` (A-S6) |
| Cross-refs | A-S7 (#174) loss/engine wiring (the fix); C-201 (self-zeroed gate scoring); the v1-review "C-6 scattered-dispatch mis-emit" this concretely realizes at the loss site |

A-S6 correctly widens the reg head to `n_params` channels per target (nb 3×2, zinb 3×3). The training engine, however, only special-cases the **quantile** multi-channel layout (`pred_j = reg[:, j*k:(j+1)*k]`); every other loss falls to `else: pred_j = t1_pred_for_loss[:, j]` — **one channel per target**. So for an `nb` head `reg = [μ_sb, θ_sb, μ_ns, θ_ns, μ_os, θ_os]`, the loop reads `reg[:,0]=μ_sb` (target sb, ok), `reg[:,1]=θ_sb` **as target ns**, `reg[:,2]=μ_ns` **as target os** — shapes match `[B,H,W]`, so it computes a loss with **no error** on mis-mapped channels. Config accepts `nb`/`zinb` (A-S5) and the head builds it (A-S6), so the intermediate state is silently-wrong if run. **Tier 2:** silent model-output incorrectness with no signal, gated only by a clear trigger. Mitigation: the epic runs no family end-to-end until A-S8 (+ the A-S11 GPU smoke); exposure is one story. Fix (A-S7): make the loss/engine reg-slice family-aware (per-target `n_params` stride like the quantile branch) **and** add a fail-loud guard that `out.reg.shape[1]` matches the loss's expected per-target width. A throwaway guard was deliberately NOT added in A-S6 (it would be replaced by the A-S7 wiring one story later).

> **RESOLVED 2026-07-21 (A-S7 #174).** The training-engine per-target reg loop now branches on `isinstance(loss_fn_j, FamilyLoss)` **ahead of** the quantile branch and slices `t1_pred_for_loss[:, j*n_params:(j+1)*n_params].permute(0,2,3,1)` → `[B,H,W,n_params]` for `family.nll` (`training_engine.py:321-323`); every legacy loss keeps `else: reg[:, j]` (byte-identical). Verified by `tests/distributions/test_loss_wiring.py::test_family_slice_formula_and_shape_guard` (correct-slice `nll` finite + the pre-C-207 single-channel slice raises via the family shape-guard) and the `nb`/`zinb` real-model `_process_sequence` grad-flow tests.

---

### C-203: `initial_raw_bias` was an NB-only head-init recipe off the `DistributionFamily` ABC — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-203 |
| Tier | 3 |
| Source | /code-review max (2026-07-20), A-S3 (#170) altitude finder |
| Resolution | **RESOLVED IN A-S4 (#171), 2026-07-20 (user decision to resolve now rather than defer to A-S6).** Promoted `initial_raw_bias` from an NB-only method to an `@abstractmethod` on `DistributionFamily` (`base.py`) with a family-agnostic signature `initial_raw_bias(*, priors: dict \| None = None) -> [n_params]`. `NegativeBinomialFamily` migrated from `theta_prior=…` to reading `priors["theta"]`; `ZINBFamily` reads `priors["theta"]`+`priors["pi"]` → length-3 bias. The A-S6 head can now call it without knowing the concrete family. CIC `DistributionFamily.md` + ADR-067 §2 updated to the 7-member interface; contract test `tests/distributions/test_zero_inflated_negative_binomial.py::test_abc_initial_raw_bias_contract_is_family_agnostic` asserts both families return an `n_params`-length bias via the uniform signature. |
| Cross-refs | C-199 (informed init is the requirement this method serves); C-205 (the symmetric `pi_penalty` seam, still open) |

The coupling never bit: `initial_raw_bias` is now on the abstraction, so A-S6 head-init stays family-agnostic and adding a family requires no consumer-side branching. Resolved before any head was wired.

---

### C-140: ZINB count-space `E[y]` would be double-`expm1`'d by the unchanged inverse transform — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-140 |
| Tier | 1 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Resolution | **RESOLVED BY CONSTRUCTION + TEST (verified 2026-06-25, read-only).** The emit path `HydraNetInference._emit_magnitude` (`hydranet_inference.py:202-217`) returns **`log1p(E[y])`**, not count-space `E[y]`: `hurdle_nb_expected_log1p` (`hurdle_nb.py`) computes the exact zero-truncated mean `e_y = p·μ/(1−NB0(μ,θ))` then returns `torch.log1p(e_y)`, so the downstream `inverse_transform_volume` (`expm1`) recovers `E[y]` exactly — it never `expm1`s a free prediction. This is C-140 fix-option (a). Round-trip test `tests/test_hurdle_nb_inference.py::test_emit_roundtrip_recovers_e_y_no_double_expm1` (+ the `tests/test_hurdle_compose.py` known-value/round-trip suite for nb/lognormal/point) asserts `expm1(out)==E[y]` — C-140 recommendation (3). 13/13 green (CPU-only run). |
| Cross-refs | C-113 (the explosion this would have re-created), C-142, D-09 |

The #100/#101 implementation followed the recommended fix: emit `log1p(E[y])` so the existing inverse transform recovers count-space `E[y]`, guarded by a dedicated no-double-`expm1` round-trip test. The Tier-1 silent double-transform never materialized.

---

### C-161: violet config operating-point drift — `n_posterior_samples=3` while #110/dossier specify 8 — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-161 |
| Tier | 2 |
| Source | /falsify (2026-06-16, "two 8-sample runs" readiness) — P1 |
| Resolution | **RESOLVED (config fixed #128 2026-06-16; gating OOM check passed, confirmed 2026-06-25).** The config was set to `n_posterior_samples=8` (#128) and the resolution gate — "an 8-sample run completes clean without the C-116 eval OOM" — is now satisfied empirically: the per_step baseline (8 seeds), the views-frames migration smoke, and the active_window ensemble (seeds 11–12 exit 0, more running) all ran train+eval at 8 samples with no OOM (the C-116 OOM is specific to the `--report/-re` publish stage, which `main.py` guards off; eval-only peaks ~2.4 GB). Floor config confirmed at 8. |
| Cross-refs | C-116/#124 (the `-re` publish OOM — distinct, still open), #110, #127 |

The executable artifact was raised from the temporary C-116 probe value `3` to `8`, and multiple full 8-sample train+eval runs have since completed clean — so launching the documented "8-sample" #110 runs now runs at the correct operating point. The remaining `-re` publish OOM is tracked separately as C-116.

---

### C-147: π = 1−sigmoid(cls) borrowed from the focal head — calibration unverified — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-147 |
| Tier | 2 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Resolution | **CHECKED/FALSIFIED (2026-06-20)** — `scripts/gate_reliability.py`, dossier `2026-06-20_gate_calibration_dossier/`, commit `eb48924`. The gate (`weighted_bce`, not focal) is **calibrated teacher-forced** (STEP-1 ECE 0.005–0.007, mean π ≈ onset prevalence) and miscalibrates **only under the autoregressive rollout** (mean π≈0.70, ECE≈0.69). The "gate loss miscalibrates π" hypothesis is falsified as the cause; the live miscalibration risk migrates to the **rollout-feedback dynamics** (the C-113 amplifier — see [[project_bloom_is_feedback]]), tracked separately as a new pre-registered experiment. Do NOT drop the class weight as a remedy. |
| Cross-refs | C-143 (composed objective), C-137; live risk migrated to rollout dynamics (Cluster 13) |

π was hypothesised to be miscalibrated as a zero/onset gate because it is borrowed from a focal-trained head. The pre-registered EXP-01 check on R4/R5 across 3 targets falsified that: the gate is calibrated when teacher-forced and only blooms through the rollout. Resolved as a gate-loss concern.

---

### C-157: training diagnostic biopsy is load-bearing — crashes training with a static-widened model — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-157 |
| Tier | 2 |
| Source | channel-role side-quest census (2026-06-13); empirically confirmed (first coord smoke crashed here) |
| Resolution | **FIXED (2026-06-20, #115 4b, commit `b5a3242`)** — the Stage-5 biopsy re-attaches static channels (reads roles via ADR-062 `model_input_cols`), so the diagnostic forward matches the `[dynamic ⧺ static]` model and no longer crashes training. Census `tests/test_channel_role_census.py` green (per C-156 status note). |
| Cross-refs | C-156 (root), C-118, ADR-062 §2.1 |

The plotting-only Stage-5 biopsy re-ran the forward with `idx.feat` (dynamic only) into a model built for `[dynamic ⧺ static]` → `RuntimeError` in lesson 1. Fixed by the #115 4b role rewire.

---

### C-158: curriculum trains on input-only channels — silent window-sampling corruption — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-158 |
| Tier | 1 |
| Source | channel-role side-quest census (2026-06-13); empirically observed (coord run listed coords in subject maxima) |
| Resolution | **FIXED (2026-06-20, #115 4b, commit `11dc121`)** — the curriculum builds `subjects` from `target_cols` (ADR-062), excluding input-only statics, so coordinates are no longer rotated in as prediction subjects. The silent window-sampling corruption is removed; census green (per C-156 status note). |
| Cross-refs | C-156 (root), ADR-062 §2.1 |

`subjects = feature_cols` had included input-only statics, silently distorting which windows were sampled (no crash, no tripwire) — the most dangerous of the three faces. Fixed by reading `target_cols`.

---

### C-159: artifact sidecar schema drifts from `choose_model` — deferred reload crash — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-159 |
| Tier | 2 |
| Source | channel-role side-quest census (2026-06-13); empirically confirmed (seed42 trained, then eval crashed on reload) |
| Resolution | **FIXED (2026-06-20, #115 4b, commit `13239b0`)** — `static_channels` is now persisted in the artifact sidecar so `choose_model` rebuilds the model with the correct `n_static` and `load_state_dict` matches the trained checkpoint; coord models reload for eval. Census green (per C-156 status note). |
| Cross-refs | C-156 (root), C-09, ADR-062 §2.3 |

The sidecar `arch_keys` whitelist had omitted `static_channels`, so reload rebuilt a narrower model and size-mismatched the checkpoint — a deferred crash after full training cost. Fixed by persisting the key.

---

### C-153: ADR-060 static-channel seam — built coordinated across the pipeline — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-153 · Tier 3 · Resolved #108, 2026-06-13 |
| Cross-refs | ADR-060/061, C-142, C-154, C-155 |

The seam was flagged as cross-cutting (an incoherent half-seam being the risk). **Resolved (#108):**
implemented coordinated across every touch-point — a `static_channels` config block (declared separately,
**not** in `features`) + validators (I1 not-a-target; `input_channels == len(features) + len(static)` ==
`3*output_channels + len(static)`); `VolumeHandler.from_df` derives static channels over the full grid and
appends them **before the North-Up flip** (flip-synced, I6; window-sliced, I4); the model input is
`[dynamic ⧺ static]` with the static slice **re-attached every step** in BOTH the inference loop and the
scheduled-sampling training feedback (I3); `FeatureScaler` skips them (unknown ⇒ no inverse, I2); the
architecture needed no edit (input side already parametrized by `input_channels`). Invariants **I1–I6**
green in `tests/test_static_channel_seam.py`; `static_channels=[]` is byte-identical (I5); `test_p1`
flipped to a green guard.

---

### C-151: bounded 6-run hurdle-NB sweep — intrinsic, not clamp-masked — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-151 · Tier 2 · Resolved #107, 2026-06-13 |
| Cross-refs | C-142, C-148, C-152, C-113 |

The concern: was the bounded 6-run sweep intrinsically bounded, or masked by the `feedback_clamp`?
**Resolved (#107):** wandb logged `feedback_clamp_log1p: None` in **all 11** baseline runs ⇒ no clamping occurred ⇒ the observed bound (FULL MCR 2.4–13) is **intrinsic, not clamp-masked**. The "ablate the clamp" check is moot — it was already off. The "is C-113 really solved?" foundation question is answered: **yes, intrinsically.** Provenance pinned in coords dossier `05`.

---

### C-154: coords experiment disk budget — headroom guard added — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-154 · Tier 3 · Resolved #107, 2026-06-13 |
| Cross-refs | C-115, C-116 (Cluster 6), C-153 |

**Resolved (#107):** added `views_hydranet/utils/disk_guard.py::assert_disk_headroom` + an opt-in config field `min_free_disk_gb` (default `None`=off ⇒ byte-identical), wired into `hydranet_manager._setup_evaluation` to **abort (fail loud) before the ~2.5 GB prediction writes** if free < budget. Tested (`tests/test_disk_headroom_guard.py`). The coords run sets the budget so it cannot silently truncate as S3_seed4 did.

---

### C-155: baseline comparator config ambiguity — stale sweep aligned — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-155 · Tier 3 · Resolved #107, 2026-06-13 |
| Cross-refs | C-151, C-42, C-153 |

**Resolved (#107):** the stale `config_sweep.py` (tobit/focal) was **aligned** to the canonical `config_hyperparameters.py` (hurdle_nb/weighted_bce), so a sweep can no longer silently benchmark against Tobit. Baseline provenance **pinned** in coords dossier `05` (config + per-arm env `HN_*` + seeds {42,4} + the C-42 reproducibility lock + clamp off). `test_p5` is now a green guard.

---

### C-142: `diagnose_io_gain` explosion-check unvalidated for the hurdle-NB count-space output — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-142 |
| Tier | 2 |
| Source | expert-code-review (ZINB Pass-1, 2026-06-10) |
| Resolved | #106 (coordinate-grounding epic box prereq), 2026-06-13 |
| Cross-refs | C-113, C-140, C-153, C-121 |

The explosion-check (`free_running_attractor` / `diagnose_io_gain`) was built for the log1p point head — it fed back `output.reg` raw. For the hurdle-NB head inference instead feeds back `log1p(E[y])` (`E[y]=P(y>0)·μ/(1−NB0)`, C-140), so feeding `out.reg` raw measured **count-space μ against the log-space `DATA_LOG_MAX` bound** — a category mismatch that could report a false verdict. **Resolved (#106):** the exact mean compose is now a single source of truth — `views_hydranet/utils/hurdle_nb.hurdle_nb_expected_log1p` — called by **both** `HydraNetInference._emit_magnitude` (byte-identical refactor) **and** the probe via `free_running_attractor(emit_fn=…)` (threaded through `diagnose_io_gain` from the artifact's `hurdle_nb_theta`). So the probe now measures exactly what inference feeds back. Validated by `tests/test_rollout_stability_guard.py` (a healthy in-range count is no longer mis-flagged; a composed-E[y] runaway is flagged) and guarded by `tests/test_falsification_epic_planning_readiness.py::test_p4…` (now green). `emit_fn=None` keeps the standard path bit-identical.

---

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
