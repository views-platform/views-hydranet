# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-hydranet                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-09-03                           |
| ID accounting     | C-188 merged into C-182 on 2026-08-15; C-275/C-276 added the same day; C-284..C-287 added 2026-08-15 from PR #274's `/code-review max`; C-260 relocated → §Resolved 2026-08-15 (fix verified in source + test); C-288 added 2026-08-15 from PR #276's CI failure; C-292/C-293 added 2026-08-16 from PR #277's code review; C-294/C-295 the same day from the architecture read it prompted; **C-289/C-290/C-291 added 2026-08-16 from PR #278 (feedback-realism), filling a gap C-292 already cross-referenced — they had been assigned in conversation and never written; C-296/C-297 added the same day from PR #278's code review and from authoring its fixes; C-298 added 2026-08-17 from the Claims Ledger verification pass; C-299/C-300 added 2026-08-17 from `postmortem_floor_limited_vehicle.md`; C-301/C-302 added 2026-08-18 from the code read behind the lesson-curve pre-registration; **C-303/C-304 added 2026-08-22 from PR #283's `/code-review medium` + `/review-diff`; **C-308 added 2026-08-23 (a probe measured the wrong rollout phase; every downstream guard still passed); C-307 added 2026-08-23 from the user's observation that cheap screens keep being recorded as closures — a pattern predating this session; C-305/C-306 added 2026-08-22 from PR #292's reviews, and C-303 escalated from three to FIVE occurrences — the fourth inside the provenance document written to prevent it;** the same pass MERGED two findings into existing entries rather than adding new ones — GH #282 (persistence baseline silently zeroed for the first origin) is a second, already-shipping symptom of C-248's unloaded pre-origin months, and C-293's "AP is a ranking statistic so the comparison is valid" was CORRECTED: at S=1 with no gate, persistence is ranked on a two-level score while gated arms get a continuous probability.** **C-312..C-315 added 2026-08-26 from the training-loop gradient audit (forward/backward/gradient flow, ahead of the pushforward arm); the same pass recorded C-303's TENTH occurrence — the first inside production source rather than a report. C-316 added the same day (test-suite pollution found because the new tests failed only in full-suite runs). **C-312 FIXED and C-314 partly fixed on the same branch; C-313 and C-315 remain open by decision (C-112: changing the clamp or clipping the balancer would move training dynamics). C-312 and C-303 carry FIXED banners but stay in §Open pending the next curation relocation, as C-184 and the PR-#216 entries did.** `C-34`/`C-188` are intentional numbering gaps (merged entries). **C-319..C-322 added 2026-09-03 from the silence-vs-fade dossier: C-319 (field statistics blind to placement — occurrence, magnitude and alignment ALL survive a roll that destroys the forecast, so an internal statistic cannot close a causal claim about a truth-referenced score); C-320 (a falsifier band tighter than its own reference's sampling noise, which fired on the known-good control); C-321 (`--keep-cubes` silently disables the multi-arm contamination guard); C-322 (the model field's H axis runs opposite to priogrid row order — naive placement correlates 0.026, the flip gives 1.0000).** |
| Total Concerns    | 320                                  |
| Open Concerns     | 168                                  |
| — of which demoted (tech-debt) | 13 (tagged `[DEMOTED]` in §Open Concerns; indexed in §Tech-Debt Backlog) |
| — net active risks | 144                                 |
| Resolved Concerns | 152                                  |
| Last curation pass | **2026-08-15 (review-rr strategic).** 24 entries relocated §Open → §Resolved: the 12 PR-#216 bannered entries (C-138/234/235/236/237/238/239/240/241/242/243/247) whose relocation this header had flagged as pending, plus 12 whose fixes were verified in source but never recorded (C-132/146/179/180/193/194/195/196/197/201/251 + C-184, the last with residual C-273). C-188 merged into C-182; C-134 re-tiered 2→3; 7 Tier-4 entries demoted; 2 causal clusters added (14 positional coupling, 15 register↔code sync). Open 145 → 120, then → 122 with 2 blind-spot entries registered the same day (C-275 data vintage, C-276 forecast monitoring). |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Causal Clusters (review-rr strategic, 2026-06-05; refreshed 2026-06-24, 2026-07-27, 2026-07-31)

**2026-07-31 refresh (review-rr strategic) — 3 new clusters (mostly RESOLVED on PR #216):** (A) **emit_family_core rollout coherence** — C-234/239/240/242; (B) **data-backed static-channel hardening** — C-235/236/237/238/244 (C-244 deferred → #229); (C) **test-suite portability & collection integrity** — C-138/165/247 (C-165 partially: the toy collection error is fixed; the ci.yml `--ignore` set remains). Structural-coupling cluster extended: C-245/246 join C-01/03/35.

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

**2026-08-15 refresh (review-rr strategic) — 2 new clusters + a register↔code reconciliation:**

| # | Root decision | Member entries | Fix scope | Priority |
|---|---|---|:--:|---|
| **14** | **Positional coupling — ordinal position substitutes for a named binding** | C-270 (gate↔body `prob[:, :n_reg]`), C-260 (SS channel substitution by position), C-123 (`choose_loss` positional 3-tuple), C-03 (hardcoded 3+3 topology), C-255 (hardcoded ZINB param layout in the forensic) | 1–2 | **near-term — C-270 is the acute member** (the only Tier 2 with a silent model-output consequence and a one-validator fix). Doing it as the head of this cluster reframes C-03/C-123 from "deferred god-refactor" to "incremental de-positioning" |
| **15** | **Register↔code sync debt — the code cites concern IDs faithfully; the register did not read them back** | Closed on this pass: C-132/146/179/180/193/194/195/196/197/201/251 + the 12 PR-#216 bannered entries + C-184 (residual → C-273). Still live: C-274 (self-contradicting test verdicts) | done (this pass) | **largely CLOSED 2026-08-15** — 24 entries relocated to §Resolved, open count 145→120. Recurrence guard: a trigger that provably cannot fire is a resolution candidate; re-run the `grep -rhoE 'C-[0-9]{2,3}'`-vs-§Open cross-check each curation pass |

**2026-08-15b (review-rr prioritize) — 2 LATENT clusters promoted.** The prioritize pass found **60 of 109 active entries unclustered, including all 3 Tier 1s** — the map had not kept pace with the expert-method-review intake (28 entries) or the operational-readiness intake. The two largest latent groups outrank every mapped cluster:

| # | Root decision | Member entries | Fix scope | Priority |
|---|---|---|:--:|---|
| **16** | **Rollout-skill measurement integrity** — ✅ **LARGELY CLOSED 2026-08-15 by Epic #263** (S0–S7). Corrections to this row's original text: **|O| = 13, not ≈12** (the ≈12 came from `02b_method_review.md:72`, written *before* the partition was verified); the guard tests numbered **10, not 12** (that conflated `test_gw_stratified.py` with `test_score_v2_horizons.py`); and C-217 was already **cleared-with-residual**, not open. | C-217 ✅ asserted · C-218 ✅ asserted (runtime) · C-219 ✅ enforced in code · C-220 ✅ first-ever test · C-221 ✅ MDE stated · C-224 ✅ diagnostic exists (**Tier-1 governance ask UNCHANGED, still open**) · C-231 ✅ metric exists · C-248 ✅ inherited by the climatology · C-252 ✅ explicit · C-253 reused untouched · C-254 ✅ power stated. New: **C-277**. **Guard accounting: 10 pytest + 1 runtime** (C-218 cannot be a portable pytest without a cross-repo path — the C-247 sin). | done | **VERDICT DELIVERED:** the h36 "win" is an **ARTIFACT** — unanimous across 12/12 arm×target rows. See `reports/2026-08-15_rollout_ruler_trust_dossier/07_experiment_log.md`. |
| **17** | **Global-server deploy readiness — every guard in this register fires before deployment, none after** | C-115 (silent CPU fallback), C-116 (publish-tail OOM — mechanism asserted and retracted twice; MEASURE first), C-163 + C-164 (no runtime harness / no peak-RSS seam test), C-192 (grid-name flip not wired into DataSniffer — gates the S2 commit), C-245 (out-of-repo pipeline-core hooks pinned by range only), C-272 (rolling-origin silently truncated), C-275 (no data-vintage record), C-276 (no forecast monitoring), C-110 (heterogeneous-member ensemble aggregation unverified) | 2 (parallelisable; disjoint files from cluster 16) | **IMMINENT.** `heavy_freighter` (global grid) is explicitly untested in the ensemble smoke, and the flip is 8× the cells of africa. Sequence by cost: C-115 (≈10-line hard CUDA gate) → C-192 (unblocks S2) → C-275 → C-116/163/164 → C-276 before serving. |

**Cluster 7 status note (2026-08-15):** the register calls this "the live research front", but the V2 scoreboard has since delivered a verdict (gated_NB ≡ th_gated_NB ships; ZINB falsified as the crps_all front-runner and blooms in the free-running rollout). C-146 is now closed by the committed-likelihood docstring; several remaining members are likely mooted by that empirical result rather than by code. The member-level status review this map has requested since 2026-07-27 is still outstanding for the other 9.

**Highest-value (refreshed 2026-07-27):** Cluster 1 is no longer the priority — the bloom epic (#183/#193) + today's cleanup addressed most of it (C-113 mitigated via ADR-070, C-121 resolved, predict() sampler + tests + IntegrityGuardian shipped). The **live research front is Cluster 7** (ZINB objective/likelihood): the head is built but the train/inference-objective and likelihood-commitment questions are unverified — **the top follow-up is a member-level status review of Cluster 7** to separate genuinely-open from epic-closed. Clusters 2, 8, 12 are mostly closed (residuals: C-110/112; C-156/160/166; C-163/164). The **magnitude/amount-ceiling** (not a register risk) remains the standing research ceiling.

---

## Open Concerns

### C-228: ADR-061 top-skip re-injects raw static channels into the GATE heads → occurrence-AP collapse (negative transfer)

| Field | Value |
|-------|-------|
| ID | C-228 |
| Tier | 2 |
| Source | expert-method-review (covariate-ingestion panel, 2026-07-29) + S8b placebo control (Epic #203) |
| Trigger | Adding ANY static / data-backed `static_channels` covariate (re-running the population arm, or wiring a new covariate) and trusting occurrence output — the gate silently loses ranking skill with no error signal |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (`dec_conv1_head{1,2,3}_class`, `base*2 + n_static`); `views_hydranet/utils/volume_handler.py` (S7 data-backed static fill, ~243–257); `views_hydranet/utils/static_channels.py`; ADR-060/061 |
| Cross-refs | C-152 (the CoordConv analogy that didn't transfer — now evidenced), C-153/C-156 (the static seam / channel roles), C-149 (NB ξ=0), ADR-060, ADR-061 |

The ADR-061 top-skip concatenates the RAW static channel at every decoder head's full-resolution `dec_conv1` — **including the classification (gate) heads**. Seed-42 controls (S8b) prove this is a **structural defect, not a population fact**: a spatially-shuffled PLACEBO static channel (identical marginal, zero signal) collapses gate AP **sb 0.31→0.13**; real `ln_pop` 0.31→0.20; crps_all inert (0.142 across all arms); no-static+`--saved` is byte-identical to baseline. A raw high-resolution channel injected at the gate's near-final conv is **negative transfer the gate cannot suppress** in 40 lessons; the body (crps) is unharmed, so the damage is gate-specific and **silent** (crps looks fine). Any covariate added through this seam degrades occurrence ranking without an error. **Mitigation direction (panel):** statics do not belong in the U-Net skip (Ronneberger2015 — skips carry spatial detail, not semantics) — remove the top-skip re-injection of `static_channels` (keep encoder-input entry), and/or replace raw concat with learned modulation (FiLM). Verify with the encoder-only diagnosis BEFORE any covariate re-run.

**UPDATE 2026-07-31 — CONFIRMED on structured coords at 300 lessons (v2 `09`/`07` E2) → RETIRE the top-skip seam.** The coord A/B ran BOTH placements × 3 seeds: `top` (static_top_skip=True, top-skip-into-gate) is worse than `enc` (encoder-only) on AP AND markedly more seed-unstable (sb AP h18 top 0.149±.052 vs enc 0.194±.028). So the defect is **not** placebo-specific — the top-skip degrades occurrence even for a maximally-structured channel (coords), and adds instability. This is the direct, non-placebo confirmation the mitigation needed: **remove the ADR-061 top-skip re-injection of `static_channels`** (keep the encoder-input entry). Note `enc` (encoder-only) ALSO underperformed no-coords here — but that is the CoordConv-is-not-a-lever finding (C-152), separate from this placement defect. Recommendation firmed to: retire the top-skip seam; any future covariate uses learned modulation (FiLM), not raw concat (C-230).

---

### C-229: no covariate taxonomy — one `static_channels` concat seam won't scale to the ~78 dynamic covariates

| Field | Value |
|-------|-------|
| ID | C-229 |
| Tier | 3 |
| Source | expert-method-review (covariate-ingestion panel, 2026-07-29) |
| Trigger | Adding a TIME-VARYING covariate (vdem / shdi / travel-time) — through `static_channels` (which re-injects it as a constant, wrong) or as a target (blocked by the `features==regression_targets` invariant) |
| Location | `views_hydranet/utils/static_channels.py`; `views_hydranet/utils/config_initializer.py` (`static_channels`, `features==regression_targets` at :334); ADR-060 |
| Cross-refs | C-228, C-230, D-14 |

The datafactory exposes ~78 covariates spanning static (population), slowly-varying (shdi) and dynamic governance (vdem) — which per Lim2021 (Temporal Fusion Transformer) need **distinct pathways** (a static-covariate encoder + variable-selection vs the temporal/ConvLSTM path). HydraNet has exactly one input-only mechanism (`static_channels`, geometry-derived until S7's data-backed path) plus a hard `features==regression_targets` invariant that blocks input-only DYNAMIC covariates entirely. Pushing time-varying covariates through the static seam freezes them at a reference vintage (loses their signal) or forces per-covariate hacks. The seam should be designed once for static + dynamic, not per covariate — the nb/zinb-done-right analogue.

---

### C-230: raw-channel concatenation is the wrong conditioning primitive for semantic covariates

| Field | Value |
|-------|-------|
| ID | C-230 |
| Tier | 3 |
| Source | expert-method-review (covariate-ingestion panel, 2026-07-29) |
| Trigger | Building a covariate-ingestion seam on raw concatenation (encoder-input or skip) rather than learned modulation (FiLM / conditional-BN), an embedding, or a variable-selection network |
| Location | `views_hydranet/utils/static_channels.py` (data-backed path); `HydraBNrecurrentUnet_06_LSTM4.py` (input + `dec_conv1`) |
| Cross-refs | C-228, C-229, C-152 |

Concatenating a raw covariate channel is the weakest conditioning primitive (FiLM — Perez/Dumoulin, **to-fetch**): the network must learn to route a constant channel through spatial convs, and cannot cleanly down-weight a useless one (a variable-selection network would). The CoordConv precedent (Liu2018) that seeded ADR-061 validates concat ONLY for coordinate priors convs provably cannot compute — not arbitrary semantics (see C-152, C-228). The principled primitive is feature-wise modulation (γ,β from a covariate context, applied per-head) or a learned static-covariate encoder + variable selection (Lim2021). Registered so a covariate seam is not built on concat by default.

---

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

### C-37: `[DEMOTED]` VolumeHandler in SAP "Zone of Pain" — partial abstraction at PF boundary
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

### C-49: `[DEMOTED]` Flat config schema may not scale — no nested structure for regularizers, strategies, or per-target settings
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

### C-85: `[DEMOTED]` Flip probability 0.5 hardcoded in training_engine — not config-driven
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

### C-89: `[DEMOTED]` `_SumReducer` and `_make_tiny_model` duplicated across test files
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

**Update 2026-07-30 (v2-scoreboard method-review; Browning/Hawkes + Tong seats) — NEW RE-ARM TRIGGER (folds C-MR4).** A proposed self-exciting / Hawkes-intensity input FEATURE, or a mixture surge-expert, re-arms exactly this io-gain>1 loop IF its input is recomputed from FED-BACK samples during the rollout (Browning's α>1 explosive regime is the same pathology; cf. C-226 for the persistence-anchor re-import). Any such component must use a **bounded, observed-history-anchored kernel held stationary in rollout**, with `crps_none` as the bloom guardrail — do NOT recompute an excitation/intensity term from the model's own fed-back predictions. (Corroborating: the v2 scoreboard showed sample-feedback stabilises the *gated* composition but the *self-zeroed* ZINB still blooms — self-zeroing under AR is the finding-#6 instance of this entry.)

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

### C-127: `[DEMOTED]` Duplicate dict keys in model configs (F601) — later definition silently shadows the earlier

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

> **`[DEMOTED]` to the Tech-Debt Backlog 2026-08-15 (review-rr strategic signal-to-noise pass).** Tier 4, mechanical, single-file, no correctness or reliability impact — actionable as ordinary tech debt rather than a governance risk. Full entry retained here for traceability; indexed in §Tech-Debt Backlog; no longer counted as an active risk.

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

**Update 2026-08-03 (dev→main release review, PR #252 — the concrete mechanism identified).** The suspected mis-calibration has a named cause: `HydraBNrecurrentUnet_06_LSTM4.py:310` uses **ONE shared `LockedDropout` instance** across all 15 dropout sites, and `LockedDropout.forward` caches the locked mask keyed only by `(shape, device, dtype)` (`locked_dropout.py:66`). So every same-shaped site — encoder `e0s` + the 6 decoder heads (`H1/H2/H3_reg`+`_class`) + intermediates, all `[B,base,H,W]` — reuses **one** mask per forward, giving perfectly **correlated** epistemic dropout across layers/targets instead of the per-layer independent masks ADR-057 / Gal & Ghahramani intend. Per-target marginal expectation is preserved (inverted `1/(1−p)`, reset per posterior sample) so the central forecast + per-target mean are **unbiased**; but the effective epistemic diversity (posterior spread) is reduced/distorted — the exact "too tight" mode this entry anticipated. **Confirmed by code read.** It is the behavior ALL validated results (v2 scoreboard, Epic #230) used ⇒ **NOT a regression**; fixing it (a `LockedDropout` per call site → independent per-layer masks) would **change the scored posterior** and require re-scoring. **Dispositioned at the dev→main merge: track as follow-up, do NOT fix at release time** (preserve scored-result comparability). Folds the release-review finding into this entry (dedup).

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

### C-131: `[DEMOTED]` `weight_decay=0.1` is large in absolute terms (intentional, but unflagged)

| Field | Value |
|-------|-------|
| ID | C-131 |
| Tier | 4 |
| Source | review (config_hyperparameters.py sanity check, 2026-06-07) |
| Trigger | Investigating posterior calibration/sharpness or regularization strength without accounting for the large `weight_decay` |
| Location | `views-models/models/violet_visitor/configs/config_hyperparameters.py:51` (and `config_sweep.py`) |

`weight_decay=0.1` is high relative to typical values (1e-4 to 1e-2). It is the **established** value (matches `config_sweep.py` and the test baseline), so not a defect — but it is large enough to materially affect regularization and posterior width, and anyone diagnosing those should know it's intentional and large rather than assume a conventional small value.

Tier 4 rationale: a code/config observation, not a correctness issue; the value is deliberate. No silent corruption. Cross-ref C-126 (calibration metric).

> **`[DEMOTED]` to the Tech-Debt Backlog 2026-08-15 (review-rr strategic signal-to-noise pass).** Tier 4, mechanical, single-file, no correctness or reliability impact — actionable as ordinary tech debt rather than a governance risk. Full entry retained here for traceability; indexed in §Tech-Debt Backlog; no longer counted as an active risk.

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
| Tier | 3 (downgraded from 2 on 2026-08-15 — partial mitigation landed, see banner) |
| Source | expert-code-review (wandb lifecycle, 2026-06-07) |
| Trigger | A phase (training especially) runs while `wandb.run is None` — the guarded `wandb.log` calls drop all metrics with no warning or error |
| Location | `views-hydranet/views_hydranet/train/training_engine.py:640/651/664`; `views_hydranet/utils/utils.py:~204` (`train_log`) |
| Cross-refs | C-132, C-133 |

`if wandb.run is not None:` makes "no observability" indistinguishable from "healthy." A ~90-minute training run can lose all telemetry silently — exactly how C-132 stayed hidden, and it bites during the C-112/C-113 investigations that most need training dynamics. Prevention: emit a one-time WARNING (or assert, in non-sweep/non-test runs) when a training loop proceeds with `wandb.run is None`.

Tier 2 rationale (original): silent failure mode that masks other defects (defense-in-depth gap); clear trigger, no error signal. Observability-only (no correctness impact) → not Tier 1.

> **Partial mitigation landed; ⬇ Tier 2 → 3 on 2026-08-15 (review-rr strategic).** `training_engine.py:827-834` now lazily imports wandb at training start and emits `logger.warning("⚠️ wandb.run is None at training start — per-lesson training metrics will NOT be logged…")`, tagged "Fail-loud (C-134)" in the source. **This is the WARNING half of the prevention this entry asked for, not the assert half** — a non-sweep, non-test training run still proceeds to completion with no telemetry, so "no observability" is now *distinguishable* in the logs but not *refused*. The remaining gap is narrow (promote the warning to a hard fail outside sweeps/tests) and the masking risk that motivated Tier 2 is substantially reduced, so this is re-tiered to 3 rather than resolved. Note the guarded `wandb.log` sites cited in Location have since moved (now `training_engine.py:1041-1068` and `utils.py:306`).

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

**Update 2026-07-28 (expert-method-review, magnitude-fix panel — strengthened AND partially corrected).** Two refinements from the panel (Koenker, Davison seats): (1) it is not merely that the NB tail is "too light" — it is a **family-level veto**: the EVT tail index of ANY negative binomial is ξ=0 (finite variance for all μ,θ), whereas the truth is ξ≈0.8 (α≈1.25 ⇒ **infinite variance**). No loss reweighting or anchor can move ξ off 0; only an explicit ξ>0 component (pooled peaks-over-threshold GPD splice, DXtreMM/DeepExtrema `Abilasha2022`/`Galib2022`) can represent it. (2) **The premise "QS99 is the binding tail guardrail" is WRONG** — with 99.7% zeros the unconditional 99th percentile is 0 (99<99.7), so QS99 sits *inside the zero mass* and measures the onset boundary, not magnitude. The escalation-detection role this entry assigns to QS99 does not exist. The tail-detectability gap is now tracked as **[[C-224]]** (Tier 1). Cross-ref C-137 (Tweedie/GPD escalation), C-224 (eval tail-blindness).

**Update 2026-07-30 (v2-scoreboard — family-veto EMPIRICALLY CONFIRMED + a caveat on "accept the ceiling"; folds C-MR2).** The 3-seed×300-lesson v2 scoreboard shows `crps_events` **identical across nb / zinb / th_gated** (sb h1 ~15.4–15.7; size_ratio 0.10→0 by h18) — the predicted family-level signature: every NB-family composition is timid in the same way because all share the ξ=0, mean-tied light tail. **But the panel's B-camp (Koenker/Davison/Tong) flag this only demonstrates the ceiling on the conditional MEAN, WITHIN one exponential family — it is NOT demonstrated on a tail-DECOUPLED head** (GPD splice / positives-only upper-quantile / 2-component mixture-density), where the estimable jump-RISK/spread (0.79) could still move a proper, tail-sensitive score. ⇒ **"accept the magnitude ceiling" is currently under-evidenced**: before permanently retiring magnitude work, run one minimal tail-decoupled probe scored on a covariate-stratified proper metric (see C-224 update). Cross-ref C-224, D-13, C-232.

**Update 2026-08-02 (Epic #230 — the tail-decoupled MEAN probe is DONE; caveat narrowed to tail SHAPE).** The 2-component mixture-density NB head (mean-DECOUPLED positives, `μ2=μ1+softplus(Δ)`) was run 3-seed×300-lesson and scored on the pre-registered covariate-stratified proper metric + Giacomini–White (`reports/2026-08-01_tail_decoupled_head_dossier`, EXP-02). **Result: NULL** — significant 3/3 on the ex-ante high-risk stratum but sub-5% (1.5–3.4%), h=1-only; F2/F3/F4 all clean. Critically, F4 was **directly measured**: component-2 is genuinely **alive** (median `w|active` 0.71–0.92, median `μ2:μ1` 14×–690×) — the mean-decoupled tail is *used*, not collapsed. ⇒ the C-MR2 caveat is now **half-discharged**: a mean-decoupled head *within the NB family* was tested and is **insufficient**, so the residual open lever is specifically the **tail SHAPE (ξ>0)**, not mean-decoupling — a 2-NB is asymptotically light-tailed (ξ=0) and cannot reach ξ≈0.8 however large μ2 grows. "Accept the ceiling" remains under-evidenced ONLY for an explicit heavy-tail (GPD-splice / PIG) head; the mean-decoupling escape route is now closed. Cross-ref C-224, C-255 (the forensic that nearly misread F4).

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

**Update 2026-07-27 — FAO-02 reconciliation (C-167 close-out).** PIT is **FAO-02-REJECTED for *selection*** (`[[reference_fao02_locked_eval_framework]]`: CRPS primary + QS99/Brier/MCR guardrails; twCRPS/LogScore/PIT rejected). This entry's PIT recommendation is retained **as a diagnostic only, never as a gate** — `05_analysis_plan` §Method now drops active-cell PIT from the guardrail set and uses QS99/Brier/MCR. The **positive-tail posterior-predictive check** stands (it is not a rejected metric — it is a distributional sanity read). No selection role for PIT anywhere. Substantively addressed; the diagnostic value remains available in `proper_score_audit.py` (marked diagnostic-only there too).

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

**UPDATE 2026-07-29 — the analogy did NOT transfer; the "unbacked input+top-skip placement" is empirically harmful (expert-method-review + S8b).** The S7 data-backed static channel extended the coord-concat seam from geometry (coords) to a **semantic** covariate (population, `ln_pop`). Seed-42 controls prove the placement C-152 flagged is a real defect, not just a decision-hygiene risk: a spatially-shuffled **placebo** static channel collapses gate AP sb 0.31→0.13 (real `ln_pop` 0.31→0.20), crps_all inert. So concatenating a raw semantic channel via the input+top-skip is not merely "analogy may not transfer" — it **actively degrades occurrence**. The concrete architectural defect is tracked as **C-228**; the generalization-overreach and the wrong-primitive framing as **C-230**. C-152 stays open as the decision-hygiene root (don't re-promote the CoordConv analogy to justify covariate concat).

**UPDATE 2026-07-31 — CoordConv now tested DIRECTLY (v2 scoreboard `09`/`07` E2) → clean 3-seed negative; question CLOSED.** The prior "didn't transfer" was inferred from the *population* placebo; the coord A/B tested real geometry coords with the nb head for the first time (gated_NB + row/col × {enc, top} × 3 seeds × 300 lessons). Result: coords **HURT** occurrence — AP < no-coords at every horizon×target (sb AP h1 0.450→enc 0.426→top 0.406; F1 fired, P1+P2 falsified); crps_all inert. So CoordConv is not a lever on the distributional heads: absolute position is already implicit in each fixed-grid cell's own history, and the extra channels are a mild spatial-overfit shortcut. C-152's decision-hygiene concern is now MOOT (no analogy left to re-promote — the lever is empirically dead). Cross-ref C-228 (the placement half of the same result).

**UPDATE 2026-08-16 — the MECHANISM, supplied by the feedback-realism probe (`reports/2026-08-16_feedback_realism_dossier/`).** The July closure was a clean empirical negative with **no explanation**, which left the null looking like bad luck and the lever re-openable by anyone with a new placement idea. It is not bad luck. Coordinate channels change *which cells are likely* — a **marginal** property of the field. What actually fails in the rollout is the **joint** structure: `spatial_scramble` reproduces the collapse (gate AP 0.3008 → 0.0097 vs free-running 0.0070) with the active count and the magnitudes held identical, so 89% of the damage is cells being in the **wrong places**, not the wrong cells being individually more or less likely. Compounding it, the emitted field is drawn by an **independence-assuming** sampler, and the gate's own probability field diffuses (Moran's I 0.409 → 0.192 by step 6 on target sb, while the ORACLE holds 0.507 → 0.494). **A marginal fix cannot repair a joint failure** — which is why coords were inert on `crps_all` *and* on AP regardless of placement. Recorded because it converts C-152 from "we tried it and it didn't work" into a **class statement**: any future proposal that improves per-cell marginals (more statics, more covariate channels, richer position encodings) inherits this null unless it also changes the joint or the sampler. Scope: 40 lessons, seed 42, one vehicle — **INDICATIVE**. Cross-ref C-290 (the sampler independence assumption). ⛔ **ANNOTATED 2026-08-17 (C-299):** every figure in this update (0.3008 / 0.0097 / 0.0070, 0.409 → 0.192) was measured on `truncated_smoke`, whose control at h18 scores **below random ranking** (0.77× prevalence). The *direction* survives — replication on `violet_visitor` made the placement effect **stronger** (−93.7% vs +0.9%) — but these numbers are floor-limited and the class statement rests on I-C, which has **not** been re-derived on a vehicle with range. See `postmortem_floor_limited_vehicle.md`.

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

**Update 2026-07-31 (discovered at PR #216 merge-time — RE-TIER → 2; premise was wrong):** the earlier claim "CI *passes* / is green modulo `--ignore`" is itself **falsified — CI is FULLY RED and has been.** The GitHub Actions `test` job never runs: `ci.yml` sets `environment-file: environment.yml`, but **`environment.yml` is absent from the repo** → `setup-miniconda@v3` dies with `ENOENT ... environment.yml` in ~6 s, before pytest is invoked (so the `--ignore` set is moot — it has never been reached). The `lint` job also fails: it does `uv tool install ruff` (**unpinned latest**) then `ruff format --check .`, which flags `docs/ADRs/active/028_…md` — a newer ruff reformats what the repo-pinned `ruff 0.14.14` leaves clean (version drift). **Both are pre-existing CI-infrastructure breakage, unrelated to any branch's code** (the code is locally green: 1258 passing, ruff-clean under the pinned ruff). **This BLOCKS every merge to `development`/`main` until CI infra is repaired** — a real gate, hence **Tier 2** (structural, clear trigger = any merge). **Fix:** (a) add a tracked cross-platform `environment.yml` matching `pyproject`/the conda env; (b) pin ruff in `ci.yml` to the dev-dep version (`ruff==0.14.14`) and decide whether `ruff format --check .` should scan docs at all. Location: `.github/workflows/ci.yml` (`environment-file`, `uv tool install ruff`), missing `environment.yml`.

**Update 2026-07-31b (partial fix landed + the DEEPER blocker named):** the `lint` job is now fixable and fixed — `ci.yml` pins `uv tool install ruff==0.14.14` (verified: `ruff 0.14.14` → `ruff check .` clean + `ruff format --check .` = "234 files already formatted") → **lint goes GREEN**. A tracked `environment.yml` (python 3.11 + pip → `pip install .`) was added, which converts the opaque `ENOENT` into a precise, self-documenting failure. But the `test` job remains RED on a **cross-repo release-ordering blocker that nothing in this repo can fix**: pyproject requires `views-pipeline-core >=3.0.0` (the code uses the 3.x `ForecastingModelManager` API), yet **3.0.0 is unpublished** — PyPI tops out at **2.3.0**, GitHub tags at **2.3.1**, and no 3.x branch is pushed; the 3.0.0 the branch depends on exists ONLY as unmerged local work (`views-pipeline-core` branch `docs/322-adr046-realign`). So `pip install .` cannot resolve the core dep, and **the `test` job (and a CI-green merge of PR #216) is gated on the upstream `views-pipeline-core 3.0.0` release** — a maintainer/coordination action, not a views-hydranet code fix. The `environment.yml` auto-greens the job the moment 3.0.0 publishes. **Merge disposition:** either wait for the upstream release, or admin-merge past the release-blocked `test` job with the code verified locally (1258 passing). Location adds: `pyproject.toml` (`views-pipeline-core (>=3.0.0,<4.0.0)`); upstream `views-platform/views-pipeline-core` (no 3.x on PyPI/tags/branches).

### C-166: `[DEMOTED]` diagnostic plots show input-only statics as predicted signal (benign display drift)

| Field | Value |
|-------|-------|
| ID | C-166 |
| Tier | 4 |
| Source | review-diff (#115 4b·biopsy, 2026-06-20) + census suspectA (`test_channel_role_census.py`) |
| Trigger | Enabling `diagnostic_visualizations` with `static_channels` non-empty, then reading the Stage-5 biopsy / `_select_display_channels` plots to interpret per-channel signal — the static (geometry) channels appear as if they were predicted targets |
| Location | `views_hydranet/utils/visual_diagnostics.py` (`_select_display_channels`, ~:437) |
| Cross-refs | C-156 (root — `feature_cols` overload), C-157 (the crash face, now fixed), C-118 (visual_diagnostics module), ADR-062 §2.1 |

A fourth, **benign** face of the C-156 overload: `_select_display_channels` derives "interesting channels" from `feature_cols`, which now includes input-only statics (CoordConv row/col). The diagnostics therefore plot geometry as if it were model signal. No crash, no model-output or training impact (the C-157 crash face is fixed; this is display-only), but it can mislead a researcher reading the biopsy plots. **Tier 4:** cosmetic/interpretation, no correctness or reliability impact. **Fix:** select display channels from `target_cols` (or exclude `static_cols`) once the role accessors are the single source — natural tidy-up alongside the flip commit or Phase-6 harden. Pinned (CLASSIFY, non-xfail) by `test_census_suspectA_visualdiagnostics_static_classification`.

> **`[DEMOTED]` to the Tech-Debt Backlog 2026-08-15 (review-rr strategic signal-to-noise pass).** Tier 4, mechanical, single-file, no correctness or reliability impact — actionable as ordinary tech debt rather than a governance risk. Full entry retained here for traceability; indexed in §Tech-Debt Backlog; no longer counted as an active risk.

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

### C-171: `[DEMOTED]` FocalLoss docstring falsely claims it "reduces to BCE when gamma=0 and alpha=0.5" (it is 0.5·BCE)

| Field | Value |
|-------|-------|
| ID | C-171 |
| Tier | 4 |
| Source | falsify (2026-06-23, gate-loss audit, probe P2) |
| Trigger | A developer reads the docstring and swaps `focal`↔`bce` (or sets α=0.5 expecting a BCE-equivalent gate) assuming equal loss *scale*, without checking the α constant factor |
| Location | `views_hydranet/utils/focal_loss.py:13-15` (class docstring) |
| Cross-refs | C-172 (same file, same audit), [[project_gate_loss_finding]] |

The docstring states FocalLoss "reduces to Binary Cross Entropy (BCE) when gamma=0 and alpha=0.5." Verified false: at α=0.5 the `alpha_t` factor is a constant 0.5, so `focal(γ=0, α=0.5) == 0.5·BCE` (probe P2: ratio exactly 0.5000). True BCE-equivalence needs α disabled (α<0) **and** γ=0. The **computed value is correct** — focal matches `torchvision.ops.sigmoid_focal_loss` exactly (probe P1) — so this is a documentation defect, not a math bug. Practical edge: because α scales the classification loss magnitude, a swap made on the false premise of equal scale would silently shift the multi-task reg-vs-cls balance. **Tier 4:** code-quality/doc inaccuracy, no production correctness impact (gate sweep used `reduction='mean'`, α∈{0.25,0.75}, where the formula is verified). Fix: correct the docstring. Failing stub: `tests/test_falsify_gate_losses.py::test_focal_docstring_bce_equivalence_is_accurate`.

> **`[DEMOTED]` to the Tech-Debt Backlog 2026-08-15 (review-rr strategic signal-to-noise pass).** Tier 4, mechanical, single-file, no correctness or reliability impact — actionable as ordinary tech debt rather than a governance risk. Full entry retained here for traceability; indexed in §Tech-Debt Backlog; no longer counted as an active risk.

---

### C-172: `[DEMOTED]` FocalLoss internal `unsqueeze(0)` leaks a leading dim under `reduction='none'`

| Field | Value |
|-------|-------|
| ID | C-172 |
| Tier | 4 |
| Source | falsify (2026-06-23, gate-loss audit, probe P5) |
| Trigger | Any future use of `FocalLoss(reduction='none')` for per-cell/masked/spatially-weighted classification loss (e.g., a hurdle_threshold-style mask on the gate, or a weight-head) — the `[1,*input]` output silently broadcasts against an `[*input]` mask |
| Location | `views_hydranet/utils/focal_loss.py:44` (`logits, targets = logits.unsqueeze(0), targets.unsqueeze(0)`) |
| Cross-refs | C-171 (same file, same audit), [[project_gate_loss_finding]] |

`FocalLoss.forward` unsqueezes a leading dim ("matches expected pipeline volume format"), so with `reduction='none'` it returns shape `[1, *input]` instead of `[*input]` (probe P5: input `(4,8,8)` → focal `(1,4,8,8)`; `WeightedBCEWithLogitsLoss` correctly preserves `(4,8,8)`). Harmless under `mean`/`sum` (the scalar is unaffected — production + the gate sweep use `mean`), but it is a latent contract inconsistency: focal is the only loss whose `none`-mode output rank differs from its input, so any per-cell weighting code that works for the other losses would silently mis-broadcast with focal. **Tier 4:** no current correctness impact; latent shape-contract trap. Fix: drop the internal `unsqueeze`. Failing stub: `tests/test_falsify_gate_losses.py::test_focal_reduction_none_preserves_input_shape`.

> **`[DEMOTED]` to the Tech-Debt Backlog 2026-08-15 (review-rr strategic signal-to-noise pass).** Tier 4, mechanical, single-file, no correctness or reliability impact — actionable as ordinary tech debt rather than a governance risk. Full entry retained here for traceability; indexed in §Tech-Debt Backlog; no longer counted as an active risk.

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

**Second facet added 2026-08-15 (repo-assimilation, Phase 3 invert-step trace) — the `feature_scaler.py:228` guard is MIS-ORDERED and therefore DEAD for prediction channels.** The prefix heuristic is not only over-broad (the original narrative above), it is also evaluated against the **pre-strip** name: `if channel_name.startswith(BINARY_PREFIX)` runs at `:228`, but `removeprefix(PRED_PREFIX)` only happens at `:235`. The channels the guard exists to protect are named `pred_by_sb` / `pred_by_ns` / `pred_by_os` (built at `volume_handler.py:404` as `f"{PRED_PREFIX}{n}"`), which do **not** start with `by_` — so the "binary/probability heads are never inverse-transformed" guard **never fires for a prediction channel**. Those gate-probability channels are currently safe only incidentally: `by_*` targets are produced by `derivations` rather than listed in `transformations`, so the subsequent `method_lookup.get(base_name)` returns `None` and the channel is skipped anyway. Correct behaviour therefore depends on a config *convention*, not on the guard — and `validate_laws` (`config_initializer.py:412`) explicitly permits accounting for a target via **either** `transformations` **or** `derivations`, so a config listing `by_sb` under a non-identity transform would `expm1` a probability on the output path with no error. **No live trigger in current configs (defence-in-depth only), so this facet does not change C-173's Tier 2** — it strengthens the case that role must come from an explicit schema rather than prefix parsing. Narrow fix if taken separately: move the prefix check after `removeprefix`, or key it off `classification_targets` instead of the string prefix.

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

### C-182: `[DEMOTED]` architecture dimensional contracts documented but unenforced — `total_hidden_channels` divisibility-by-8 contract is documented but unenforced — fails late with a cryptic unpack error

| Field | Value |
|-------|-------|
| ID | C-182 |
| Tier | 4 |
| Source | /falsify "the convolutional lstm is 100% correctly implemented" (2026-06-26) — P4 (contract), SURVIVED as observation |
| Trigger | When a new config or capacity sweep sets `total_hidden_channels` to a value not divisible by 8 (e.g. tuning recurrent memory width) — the run dies deep in `forward()` instead of at construction |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:65-66` (docstring "Must be divisible by 8"), `:101` (`num_lstm_state_layers = int(total_hidden_channels / 8)`, silent floor), `:426-427` (`split_h = int(h.shape[1] / 8)` + 8-way unpack that raises) |
| Cross-refs | C-114, C-184 (same recurrent-cell architecture, undocumented/unguarded properties) |

The constructor documents "`total_hidden_channels` Must be divisible by 8" but does **not** validate it. A non-divisible value (e.g. 12) is silently floored by `int(.../8)` when sizing the gate convs, then later **raises `ValueError: too many values to unpack (expected 8)`** from the `torch.split` unpack in `forward()` — a cryptic error far from the cause. **No silent corruption** (P4 confirmed it crashes loud, not wrong-output), so this is **Tier 4 / ergonomic only**: correctness is intact, but a developer tuning capacity gets an opaque failure at the first forward instead of a clear constructor message. Fix (optional): add `if total_hidden_channels % 8: raise ValueError(...)` in `__init__`. No failing test stub — the /falsify verdict was SURVIVED (no hard/soft falsification); registered as a maintainability observation per user request.

> **Merged with C-188 on 2026-08-15 (review-rr strategic) — one entry, two axes.** C-188 ("U-Net skip geometry has no grid-divisibility guard") was an independently-found instance of the identical pattern: a dimensional contract that is documented but unenforced, floored silently at construction, and surfacing as a cryptic tensor error deep in `forward()`. Its evidence is folded in here:
>
> * **Hidden-channel axis (original C-182):** `total_hidden_channels` must be divisible by 8; `HydraBNrecurrentUnet_06_LSTM4.py:101` silently floors via `int(total_hidden_channels / 8)`, then `torch.split` raises `ValueError: too many values to unpack (expected 8)` in `forward()`.
> * **Spatial-grid axis (was C-188):** `H`/`W` must be divisible by 4; `:471-473` two `MaxPool2d(2,2)` floor odd sizes while two `ConvTranspose2d(stride=2)` exactly double, so `:483/:490` `torch.cat([upsampleN(...), skip], 1)` mismatches. P2 confirmed grid 14 raises `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 6 but got size 7` at the first skip `cat`. Trigger: a new config or data window whose grid is not ÷4 — **live for the global flip (360×720 is ÷4; a cropped region may not be).**
>
> Both are **loud, never silent corruption** — the output is never wrongly cropped or padded — so the merged entry stays **Tier 4 / ergonomic**. One fix closes both: validate both contracts in `__init__` with a message naming the offending value. `[DEMOTED]` to the Tech-Debt Backlog on the same pass (Tier 4, single-file, mechanical).

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


**CONFIRMED, INDICATIVE 2026-08-16 (`reports/2026-08-15_state_freeze_dossier`).** This concern was
recorded as OVERTURNED by #262 on the strength of the oracle probe ("NOT hidden-state / recurrent drift").
**That inference does not hold**, for exactly the reason recorded here: the oracle varies the *input* while
the state evolves normally, so it shows the state is healthy when never polluted and says nothing about the
polluted case.

Measured directly by holding the state during free-running: **~23% of the oracle gap is recovered**
(`all` +0.084 ΔAP at h18, +0.061 at h36; `hidden` alone +0.027). So the state path is a real mediator, not
inert, and the confound this entry names was doing exactly what it said.

**Two limits on that number, both material.** (1) **Which memory half carries it is NOT established** —
`hs` is a readout of `hl` in this ConvLSTM, so freezing the cell also constrains the hidden half and
`cell ≈ all` is architecturally predetermined (C-292). (2) The arms are compared only against the collapsed
control, with **no naive baseline**, so "the state carried information" is not yet separated from "a
frozen static risk map beats a collapsed gate" (C-293).

Scope: 40 lessons, one seed, one origin set, one target, one vehicle — and the pre-registration requires a
second vehicle before this is reported as anything but **INDICATIVE**. `violet_visitor` has not been run.

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

### C-224: eval governance is TAIL-BLIND — the sanctioned metric set cannot detect a magnitude/tail win

| Field | Value |
|-------|-------|
| ID | C-224 |
| Tier | 1 |
| Source | expert-method-review (magnitude-fix panel, 2026-07-28; Davison/EVT + Gneiting seats) |
| Trigger | Declaring ANY magnitude/tail fix (quantile-Δ head, GPD splice, heavier body) a "win" or "null" on the FAO-02 sanctioned set (CRPS + QS99), OR spending GPU on a tail fix before a tail-detecting diagnostic is agreed with the FAO-02 owner |
| Location | `reference_fao02_locked_eval_framework.md`; frozen `lodestar_score.py`; dossiers `05_analysis_plan.md` / `08_magnitude_bodymask_prereg.md` |
| Cross-refs | C-149 (NB tail veto), C-150 (PIT/PPC), C-167 (resolution metric), C-136 (rollout-confounded scoring) |

The magnitude effort's target — the ξ≈0.8 surge tail — is **invisible to the metrics we are allowed to select on**, for two independent, proven reasons. (1) **QS99 sits inside the zero mass:** 99th percentile of a 99.7%-zero field is 0, so QS99 measures the zero/positive onset boundary, not deep-tail magnitude (the real tail is above the 99.7th pct). (2) **No proper-score EXPECTATION discriminates tail behaviour:** Taillardat et al. 2023 + Brehmer&Strokorb 2019 prove the expected CRPS — *and even the FAO-02-banned twCRPS* — cannot distinguish forecasts with different ξ; non-tail-equivalent forecasts score almost identically to the ideal. So a real tail improvement (or a real family mis-specification) produces FLAT CRPS, and we cannot tell "fix worked, metric blind" from "family still wrong." **Tier 1 (silent incorrectness of the evaluation):** we would certify or kill magnitude designs on a metric structurally incapable of seeing the thing under test — the resolution-blind trap C-167 named, one level deeper (C-167 fixed spatial sharpness; this is distributional-tail sharpness). Mitigation is a **governance amendment, not a run**: agree an exceedance-conditional tail diagnostic (QS at 99.9 on POSITIVE cells only; exceedance reliability/PIT as a *diagnostic* — note PIT+twCRPS are both FAO-02-selection-banned, so this needs the FAO-02 owner's sign-off) BEFORE any magnitude/tail GPU spend.

**Update 2026-07-30 (v2-scoreboard method-review — the improper-SELECTION corollary + a concrete proper alternative; folds C-MR3).** Two additions. (1) **crps_events (the repo's own truth>0 CRPS split) must never be a SELECTION metric:** subsetting the score on the observed outcome is the Forecaster's Dilemma (Lerch2017, held) — it rewards an exaggerating forecaster — so it is display-only, exactly like the FAO-02-banned twCRPS. (2) **The concrete proper, tail-sensitive alternative (Davison seat):** condition the score on a COVARIATE, not the outcome — evaluate crps_all (or MCR) on the **high-PREDICTED-risk stratum**. This is proper AND tail-sensitive, and is the pre-registerable success criterion for any magnitude/tail probe (with `size_ratio` explicitly NOT a target — Goodhart). This gives C-224's "need a tail-detecting diagnostic" a specific proper construction that does not require the FAO-02-banned metrics.


> **Update 2026-08-15 (Epic #263 S5, #269) — a DIAGNOSTIC now exists; the Tier-1 governance ask is UNCHANGED and still open.**
> `scripts/rollout_ruler_core.py` implements the Taillardat2023 §3.3 index `T_u(F,G) = 1 − Ω_G/Ω_F`: CRPS treated as a *random variable*, its **distribution** compared via a PWM-fitted GPD on the exceedances and a Cramér–von Mises statistic. This detects tail behaviour **without a threshold weight**, so it does not violate FAO-02's twCRPS rejection. Nine numbers computed for `violet_visitor` vs the FAO-02 climatology (`sb`, h∈{1,18,36}, q∈{0.99,0.995,0.999}) in `reports/2026-08-15_rollout_ruler_trust_dossier/results/tail_index.md`.
>
> **Three structural railguards, because this is a Tier 1 whose failure mode is misuse, not absence:** (a) `taillardat_index` *requires* the reference vector, so no standalone sortable per-model number can exist; (b) every output is `diag_`-prefixed and carries `role="DIAGNOSTIC"`; (c) the pre-registered decision rule (`verdict_token`) reads **no** `diag_*` key, asserted by `test_no_diag_column_reaches_the_decision_rule` inspecting its source. Additionally `test_extremist_forecast_gets_a_HIGH_index` pins Taillardat's own caveat — an inflated, mis-calibrated forecaster scores **higher** — so its *passing condition is that the metric is gameable*, and promoting `diag_Tu` to a selection metric would require deleting a green test.
>
> **What has NOT changed:** the entry's Tier-1 governance ask — that a tail-detecting diagnostic be **agreed with the FAO-02 owner before magnitude/tail GPU spend** — is untouched. This dossier produced evidence, not an amendment (Epic #263 `SCOPE.md` #7). C-224 stays OPEN.
>
> **Known limitations, recorded rather than smoothed:** `T_u` is **not monotone in q** on real data (h=1: −0.296 at q=0.99, −1.013 at 0.995, +0.548 at 0.999), which is exactly why q was pre-registered as a fixed set with **no optimisation over q**. And the index is **undefined** (not "bad") when the pooled threshold leaves one arm with <50 exceedances — pinned by `test_index_is_undefined_when_the_two_tails_do_not_overlap`.

---

### C-225: outcome-weighted likelihood (`1+γ|Δ_true|`·NLL) is an improper objective — de-calibrates the emitted distribution

| Field | Value |
|-------|-------|
| ID | C-225 |
| Tier | 2 |
| Source | expert-method-review (magnitude-fix panel, 2026-07-28; Gneiting/Koenker/Salinas/Harrell seats — unanimous) |
| Trigger | Multiplying a parametric likelihood (NB/ZINB NLL) by any weight that is a function of the realized outcome y_t (e.g. `1+γ|Δ_true|`, `|y_t|`, an exceedance flag) and then selecting on CRPS |
| Location | `views_hydranet/train/training_engine.py` (the `weight=` path into `family.nll`); `family_loss.py`; any move/information-weighted body-loss design |
| Cross-refs | C-136 (scoring-hygiene / proper-score discipline), C-224 (why the resulting mis-calibration is also undetectable), C-149 |
| Related work | Gneiting&Raftery 2007; Lerch et al. 2017 (forecaster's dilemma); Ehm et al. 2016 (only outcome-INDEPENDENT weights preserve consistency); Salinas 2020 (DeepAR weights the SAMPLING, not the loss) |

An outcome-derived weight makes the objective's population minimizer `f* ∝ w·g` — the true predictive density **tilted toward the up-weighted region** (Euler–Lagrange), i.e. provably de-calibrated μ/θ. You then score on the frozen (un-tilted) CRPS whose optimum is g, so any CRPS movement confounds skill with the tilt, and γ is a free knob tuned straight into that confound (garden-of-forking-paths). It is distinct from the quantile case: a nonnegative weight on a **pinball** loss still elicits a quantile of the *tilted measure* (consistent, Koenker/Ehm), but a weighted **NB NLL** distorts a coupled KL projection and moves μ,θ non-decomposably. DeepAR's own imbalance remedy is `∝`-magnitude **importance sampling of training windows** (which changes what you see, not what a correct density is), never loss reweighting. **Tier 2 (structural fragility of any likelihood-head magnitude design):** the whole "information-weighted body loss" family is unsound on a parametric likelihood; if move-emphasis is wanted, use an outcome-INDEPENDENT weight `w(y_{t-1})` / a forecast-time surge-risk prior, OR do it inside a quantile head, OR importance-sample windows.

**Disposition (chair, 2026-07-28): DEPRIORITIZED, not killed.** We do not reject an idea we have not run — this entry records *why the outcome-weighted-NLL variant is currently unattractive to try* (strong prior it de-calibrates: theory `f*∝w·g` + our own scars — the winsorized τ-dial "rescales≠calibrates" [[project_body_knob_quest]], count_mean "collapses OOS" [[project_count_mean_fails_oos]] — AND it targets the μ channel while the measured leak is in occurrence), not a permanent ban. It becomes attractive to revisit if (a) recast with an **outcome-independent** weight `w(y_{t-1})`/surge-risk prior, or (b) ported into a **quantile** head (where nonneg weighting is consistent), or (c) the diagnosis shifts and shows the magnitude channel is the real lever. Filed as a reason-not-to-spend-GPU-now, to be re-opened deliberately.

---

### C-226: additive persistence anchor `log1p(y_{t-1})+Δ` on a COUNT/NB mean is incoherent + re-imports the C-113 bloom

| Field | Value |
|-------|-------|
| ID | C-226 |
| Tier | 2 |
| Source | expert-method-review (magnitude-fix panel, 2026-07-28; Salinas/Koenker/Harrell/Hamilton-Tong seats) |
| Trigger | Reparameterizing the NB/ZINB head to emit `ŷ = log1p(y_{t-1}) + Δ_θ` (additive persistence anchor on the count mean), especially with the anchor updated per rollout step |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (head); `hydranet_inference.py` (emit + AR feedback); any persistence-anchored Δ design |
| Cross-refs | C-113 (autoregressive bloom), C-201 (self-zeroed gate decouple), C-149/C-224 (still doesn't touch the tail), D-13 |
| Related work | Koenker 2005 (quantile monotone-transform equivariance — the anchor is EXACT for quantiles, Jensen-biased for a mean); Salinas 2020 (DeepAR anchors the SCALE ν, frozen, not the mean additively) |

The persistence anchor is the right *idea* (a true-zero cell anchors at 0 → structurally attacks the crps_none leak) but the wrong *vehicle* on a count likelihood: (1) **Jensen bias** — `E[g(Y)]≠g(E[Y])`, so a log1p-space additive anchor on an NB log-link mean is a link mismatch (a modelling bug, not a nuance); (2) it fixes the mean *location* but leaves θ (dispersion) and the ZINB structural π unanchored → the emitted mean/variance relationship no longer matches the level, and on a true-zero cell (anchor=log1p(0)=0) the anchor supplies the *least* information exactly where the occurrence failure lives; (3) an **additive** anchor updated each rollout step is an unbounded feedback term = the C-113 bloom by another name (DeepAR avoids this precisely by using a *frozen* per-series scale, not a per-step additive anchor). The coherent realizations: additive anchor inside a **quantile head** (exact via equivariance), or DeepAR **frozen mean-scaling** on the NB. **Tier 2 (structural fragility + silent bloom re-introduction):** a persistence-anchored NB as specified is internally inconsistent and rollout-unstable; if built, it must be in a quantile head or as frozen scaling, and F-B2-tested on the free-running curve, not just T=0.

---

### C-227: a single move-weighted / anchored head recreates the up-stable↔down-transition stratum-trade; crps_all hides it

| Field | Value |
|-------|-------|
| ID | C-227 |
| Tier | 3 |
| Source | expert-method-review (magnitude-fix panel, 2026-07-28; Hamilton-Tong seat; sibling views-lstm-lab doc 19 evidence) |
| Trigger | Shipping a SINGLE global-γ or single-anchored magnitude head as a champion candidate, and judging it on aggregated crps_all rather than per-regime CRPS |
| Location | dossier `05_analysis_plan.md` / `08_magnitude_bodymask_prereg.md` (decision rule); any single-head magnitude design |
| Cross-refs | C-170 (un-freezing the balancer starves rare targets — same crosstalk family), C-136 (crps_all aggregation hid the masking over-cook), C-224, D-13 |
| Related work | Jacobs et al. 1991 (mixtures-of-experts: shared weights receiving opposite-signed gradients = crosstalk); views-lstm-lab ADR-006 doc 19 (γ=2 fixed declines 200× but blew active_stable 9.99→23.57) |

The sibling lab established empirically that one global |Δ|-weight cannot serve both regimes — "don't move" (active_stable) and "drop hard" (de_escalation) need opposite biases, so a single shared parameter vector receiving opposite-signed gradients trades one stratum for another (Jacobs crosstalk). A single hydranet magnitude head will likely reproduce this. Compounding it, **crps_all aggregation masks the trade** — it is exactly the aggregate that hid the original `pos_cells` true-zero over-cook. **Tier 3 (decision/attribution-hygiene, peer of C-136/C-170):** no silent corruption, but a single-head win/kill judged on crps_all can be a stratum-trade in disguise. Mitigation: report **per-regime CRPS (stable-zero / active-stable / escalation / de-escalation) before any crps_all aggregation**, with the regime threshold declared in the pre-registration (not snooped). The single-head-vs-routed-mixture choice itself is tracked as D-13.

---

### C-231: crps_all is zero-dominated — a long-horizon "win" can be a zero-driven pseudo-improvement that hides an occurrence-skill LOSS

| Field | Value |
|-------|-------|
| ID | C-231 |
| Tier | 3 |
| Source | expert-method-review (v2-scoreboard panel, 2026-07-30; Gneiting/Chevillon seats) |
| Trigger | Pre-registering or headlining a HORIZON experiment (direct / climatology-blend / residual head) on raw crps_all — declaring a mid/long-horizon crps_all gain a "win" without AP (or a covariate-stratified crps_all) as the primary metric |
| Location | frozen `lodestar_score.py`; `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py`; dossier `05_prereg.md` |
| Cross-refs | C-227 / C-136 (crps_all aggregation hides a specific failure — same family), C-224 (eval tail-blindness), C-167 (resolution-blind eval, resolved) |

The v2 scoreboard demonstrated (07 finding #5) that gated_NB "beats" climatology on crps_all at h36 (0.877 vs 0.960) while being WORSE on AP (0.162 vs 0.195): the proper-score win is carried almost entirely by confident zeros on the 99% empty field (crps_none 0.004 vs 0.068, 17×), NOT by forecasting events. Because ~99.3% of cells are true zeros, crps_all is dominated by zero-cell calibration and can reward a model that has quietly converged to a *degraded* climatology while LOSING the occurrence ranking (AP) that is the actual product. **Tier 3 (decision/measurement-hygiene, peer of C-227):** the AP data already exists so this is not silent corruption — but headlining raw crps_all on a horizon experiment will certify a non-improvement (or an occurrence regression) as a win. Mitigation: for any horizon-decay experiment pre-register **AP@h18/h36 (or covariate-stratified crps_all) as PRIMARY**, with a hard rule that a crps_all gain without an AP improvement does NOT count.

---

### C-232: a 2-component mixture-density head on a 99.3%-zero field has label-switching / identifiability risk

| Field | Value |
|-------|-------|
| ID | C-232 |
| Tier | 3 |
| Source | expert-method-review (v2-scoreboard panel, 2026-07-30; Hamilton-Tong seat) |
| Trigger | Building a 2-component (calm/surge) mixture-density or routed-expert magnitude head and trusting the gate's regime assignments without an identifiability constraint or seed-stability check |
| Location | any mixture / route_star-style head in `views_hydranet/distributions/` |
| Cross-refs | D-13 (single-head vs routed-mixture choice), C-227 (mixture crosstalk), C-149 (the tail the surge-expert must carry) |

The panel's minimal magnitude proposal is a 2-component mixture-density NB head (a gate routing to a calm-expert and a surge-expert). On a 99.3%-zero field this is a 3-way structure whose two positive components are weakly identified: with few exceedances per cell the likelihood is near-symmetric under a swap of the two experts (label switching), so the learned regime assignments can be arbitrary/unstable across seeds and the "surge expert" may not correspond to surges. **Tier 3:** not silent corruption of a shipped forecast, but it can make a mixture experiment's per-regime interpretation and scoring untrustworthy. Mitigation: an identifiability constraint (ordered component means, or a gate anchored on the *predictable* jump-risk covariate rather than free), plus a seed-stability check on the regime assignment before trusting the surge-expert. Fetch Frühwirth-Schnatter (finite-mixture identifiability).

---

### C-233: teacher-forcing / scheduled-sampling curriculum is a known no-op here — spending a pre-ship experiment on it would be mis-directed

| Field | Value |
|-------|-------|
| ID | C-233 |
| Tier | 3 |
| Source | expert-method-review (v2-scoreboard panel, 2026-07-30; Salinas/Chevillon seats) |
| Trigger | Proposing a teacher-forcing / scheduled-sampling CURRICULUM as the fix for the horizon-decay (07 finding #2), consuming one of the scarce ~2 pre-ship experiments |
| Location | `views_hydranet/train/training_engine.py` (scheduled-sampling path); dossier `05_prereg.md` |
| Cross-refs | C-125 (rollout training — the pushforward/GTF path, distinct), C-113 (the AR pathology TF does NOT fix) |

Two independent panel seats flagged that a teacher-forcing/scheduled-sampling curriculum will not recover the lost long-horizon skill: (a) Salinas — the DeepAR programme reports scheduled sampling gave "no noteworthy accuracy improvement (and slowed convergence)"; (b) Chevillon — teacher-forcing addresses the train/test input-distribution gap (exposure bias) but NOT the compounding of a *mis-specified* one-step law, which is the operative failure when the one-step head is light-tailed on a ξ≈0.8 DGP. **Tier 3 (decision-hygiene, peer of C-126):** no corruption, but under a 2-experiment budget this burns a slot on a lever literature and theory both predict is inert; the horizon lever with headroom is a direct/climatology-residual head (see C-231's experiment), not TF. Fetch Bengio2015 (scheduled sampling) to formally close it if challenged.

---

### C-244: data-backed statics are NOT scaled by the model FeatureScaler — S5 ships a sanity rail only; proper model-side scaling deferred

| Field | Value |
|-------|-------|
| ID | C-244 |
| Tier | 3 |
| Source | Epic #218 S5 (#223) implementation decision — the residual of C-236 |
| Trigger | Wiring a data-backed `static_channels` covariate whose natural scale is genuinely large (>1e4 abs) yet legitimate (e.g. a raw count/GDP covariate we DO want), OR relying on the model to standardize a static the way it standardizes its dynamic features |
| Location | `views_hydranet/utils/volume_handler.py` (data-backed static validation, `_STATIC_SANITY_ABS_CEIL`); `views_hydranet/utils/feature_scaler.py` (does not touch `static_channels`) |
| Cross-refs | C-236 (parent — NaN/finiteness half now fixed in S4), C-229 (covariate taxonomy — the proper home for a static-covariate scaling pathway), C-235 |

Epic #218 S5 closed the acute half of C-236 (a data-backed static now fails loud on NaN/inf via S4, and on raw/unscaled magnitude via a hard **sanity rail** `_STATIC_SANITY_ABS_CEIL=1e4`). But the rail is a **stopgap, not real scaling**: it assumes a data-backed static arrives PRE-scaled (log/standardized, e.g. `ln_pop`) and merely rejects an obviously-raw channel. It does **not** (a) fit/apply a proper transform to a static the way FeatureScaler does for dynamic features, nor (b) admit a legitimately large covariate — that would trip the rail. **Tier 3 (deferred design, not a live corruption):** the sanity rail prevents the silent-domination failure today, so this is a maintainability/extensibility gap, not a Tier-1/2 risk. The proper fix — a **static-covariate scaling pathway** (fit on the train window, applied to the volume's static channels, with the guards covering them) — belongs with the covariate-taxonomy redesign (**C-229**), not bolted onto the fill site. GH issue: **#229** (S5 follow-up, Epic #218). Do NOT silently scale statics in `volume_handler` in the meantime — surface the decision.

---

### C-245: the entire run lifecycle depends on out-of-repo views-pipeline-core base-class hooks pinned only by a version range, with no in-repo contract test

| Field | Value |
|-------|-------|
| ID | C-245 |
| Tier | 3 |
| Source | repo-assimilation (2026-07-31, R-A1) |
| Trigger | Bumping `views-pipeline-core` within the `>=3.0.0,<4.0.0` range (a routine minor/patch update), OR the base `ForecastingModelManager` changing a hook name/signature/dispatch order |
| Location | `views_hydranet/manager/hydranet_manager.py` (hook overrides: `_train_model_artifact`, `_evaluate_model_artifact`, `_forecast_model_artifact`); base `views-pipeline-core/.../managers/model/model.py` (`_execute_model_tasks` dispatch `:1093–1102`); pin in `pyproject.toml:12` |
| Cross-refs | C-132, C-133 (the outbound side — our overrides silently drop base lifecycle), C-01 (manager monolith) |

`HydranetManager` inherits its whole train/eval/forecast dispatch from the base `ForecastingModelManager` and plugs into it via *hooks* (`_train_model_artifact`, `_evaluate_model_artifact`, …). That base contract lives out-of-repo and is pinned only by a **version range** (`>=3.0.0,<4.0.0`), and **no in-repo test exercises the base dispatch/hook contract**. A minor pipeline-core bump within the range that renames a hook, changes dispatch order, or alters the flag→method mapping (`--train/--evaluate/--saved/--artifact_name` are consumed *in the base class*, not here) would **silently change or break the run path** with no local signal until a full run. C-132/C-133 are the outbound direction of this same seam (our overrides dropping base lifecycle); this is the **inbound** direction (the base changing under us). **Tier 3 (structural coupling, no silent model-output corruption):** fragility surfaces at run time, not as corrupted forecasts. Mitigation direction (not proposed here): a base-contract smoke/pin.

---

### C-246: the per-timestep recurrent T-loop is implemented twice (training vs inference) with no train/inference parity test

| Field | Value |
|-------|-------|
| ID | C-246 |
| Tier | 3 |
| Source | repo-assimilation (2026-07-31, R-A2) |
| Trigger | Editing the recurrent step / feedback handling in ONE of the two loops (changing hidden-state threading, feedback composition, or static re-injection in `training_engine._process_sequence` without the mirror in `hydranet_inference.predict`); **OR enabling scheduled sampling (`ss_epsilon_max>0`) — which exercises `_family_feedback_log1p` against inference's `_sample_feedback` for the first time in a real run** |
| Location | `views_hydranet/train/training_engine.py` (`_process_sequence` step loop; `_family_feedback_log1p:217-245`) and `views_hydranet/utils/hydranet_inference.py` (`predict` causal loop; `_sample_feedback:293-334`) |
| Cross-refs | C-99 (reg_latent vs reg dual-path), C-113 (freeze_h train/inference state mismatch), C-259 (the config-decoupling root a parity test would catch), C-239 (the ZINBcore twin closed by arm-drop) |

The model's `forward()` is strictly **per-timestep** (`[B,C,H,W]`); the recurrent T-loop that threads hidden state and feeds back predictions lives **outside** the model, implemented **independently** in training (`_process_sequence`) and in inference (`predict`). They legitimately differ (training has teacher-forcing/scheduled-sampling; inference has the free rollout), but the **shared recurrent-state-threading + feedback semantics must stay behaviorally identical** — and nothing asserts that parity. A change to one (feedback composition, static re-attach, hidden-state carry) that isn't mirrored in the other silently produces a train/inference exposure mismatch (the class of bug C-113's freeze_h note and this session's C-234 both instantiate). **Tier 3 (maintainability/drift, not a guaranteed live corruption):** existing parity anchors cover *emit* but not the *recurrent-loop* contract across train/inference. Mitigation direction (not proposed here): a shared step primitive or a train/inference recurrent-parity characterization test.

---

### C-248: ex-ante risk-stratum LEAKAGE in the planned GW ruler extension — the easy path stratifies on the outcome

| Field | Value |
|-------|-------|
| ID | C-248 |
| Tier | 1 |
| Source | expert-code-review (2026-08-01, upfront design review of Epic #230 S3 #233) |
| Trigger | Implementing S3 (#233) by building the risk-stratum mask from the in-scope `truth` array in `_metric_row` (the only per-cell outcome available there), rather than from a pre-origin covariate that must be separately loaded |
| Location | `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py:78` (`truth = tmap[(m0+h-1,u)]` — the outcome), `score_horizons_v2:126` (`months = {m0+h-1 …}` — pre-origin months are NOT loaded into `tmap`) |
| Cross-refs | C-224 / C-MR3 (the improper-selection methodology + covariate-stratified proper alternative — this is the concrete code-level instance), C-227 (crps_all zero-hygiene), Lerch2017 (Forecaster's Dilemma) |

The pre-registered success metric for the mixture experiment (#230) is a proper CRPS **stratified on an EX-ANTE covariate** (recent conflict intensity), never on the observed outcome. But inside `_metric_row` the only per-cell array that resembles a stratifier is `truth` — *the outcome itself* — and the recommended recent-intensity covariate (`tmap[(m0-1,u)]`) is **not even loaded**, because `tmap` is populated only for the scored-horizon months `{m0+h-1}`. So the path of least resistance — reach for `truth`, subset on it — silently commits the Forecaster's Dilemma: the "proper" score becomes improper, and the Giacomini–White verdict is invalid **with no error signal**. This is the exact silent-nullifier class that has voided experiments here before. **Tier 1 (silent model-output/verdict corruption).** Fix: extend `months`/`_truth_map` to load the pre-origin month(s) for the stratifier; construct the stratum ONLY from `support`/pre-origin truth; add a **leakage regression test** — permuting the current-horizon `truth` must leave the stratum mask byte-identical.

**SECOND SYMPTOM FOUND AND MEASURED, 2026-08-21 (GH #282, PR #283).** The same unloaded pre-origin
month already breaks a shipped code path, not just a planned one: `_persistence_gathered` reads
`truth_map.get((m0 - 1, u), 0.0)` for the **persistence baseline**. Month `m0-1` is present only *by
accident* — origins are consecutive, so origin *k*'s history is origin *k−1*'s h=1 forecast month —
so the **first origin's entire persistence forecast is silently all zeros**. Reproduced to four
decimals on 13 origins: persistence AP h1 **0.1461 vs 0.1632** correct, h18 0.1077 vs 0.1152, h36
0.0834 vs 0.0870 — it **understates the baseline by 4–10%**, and the error scales with
`1/n_origins`, so a **single-origin study scores persistence as all zeros with no error signal**.
Direction is always *flattering to the arms*, which is why it survived: every comparison looked
better than it was. **This raises the entry from "planned S3 risk" to "already wrong in every
persistence comparison ever run here", including the C-293 measurement and ledger M1.** The fix is
the same one this entry already prescribes — load the pre-origin months — plus making the absence
*loud*: a missing **cell** inside a loaded month is a legitimate 0.0, a missing **month** is not, and
`.get(..., 0.0)` cannot tell them apart. Repaired in the dossier's own tool
(`fair_persistence.persistence_scores(..., months_loaded=)` raises) and regression-tested
(`tests/test_fair_persistence.py::test_unloaded_month_raises_instead_of_scoring_zeros`); **the shared
scorer itself is NOT yet fixed** — that is #282 and it should land before any new claim leans on
`score_v2_horizons --persistence`.

---

### C-249: mixture `log(w)` gradient explosion at the w→{0,1} collapse — NaN fires in the decisive-negative regime

| Field | Value |
|-------|-------|
| ID | C-249 |
| Tier | 1 |
| Source | expert-code-review (2026-08-01, upfront design review of Epic #230 S2 #232) |
| Trigger | Implementing the mixture `nll` with `torch.log(torch.sigmoid(raw_w))` (or `torch.log(w)` after activation) for the mixing-weight log terms, then training until the optimizer drives `w→1` (the pre-registered F4 decisive-negative signal) |
| Location | planned `views_hydranet/distributions/mixture_negative_binomial.py` (`nll`); contrast the stable link forms in `views_hydranet/distributions/nb_core.py` (`inverse_softplus`) |
| Cross-refs | C-212 (the same NaN-sprayed-by-the-mean-reduction class in `log_prob_zero`, RESOLVED), C-N/D-15 (the clamp-vs-log-sigmoid disagreement) |

The mixture NLL needs `log w` and `log(1-w)`. Computing them as `log(sigmoid(raw_w))` makes `d/d(raw_w)` blow up as `w→0/1` (`1/w` factor), and the mean reduction then sprays the resulting NaN to every upstream parameter — structurally identical to C-212. The aggravating factor unique to this experiment: **`w→1` is not a rare pathology, it is the central OBSERVABLE** — falsifier F4 (collapse to a single NB) is a *decisive-negative* result. So the run would crash **precisely when it is telling us the answer**, converting a clean negative into an uninterpretable failure. **Tier 1 (silent/half — NaN via the reduction, in the signal regime).**

**CORRECTED FIX (2026-08-01, empirically verified — scratch `gradcheck.py`, during S2 impl):** the original fix ("use log-sigmoid `-softplus(∓raw_w)`, no clamp") was based on a form that is **not cleanly available**: `nll` receives the **activated** `w` (a probability in (0,1)), NOT `raw_w`, so it cannot compute log-sigmoid-from-raw without breaking the `activate→nll` contract. Empirically, at `raw_w=20` (`w==1.0` exactly in fp32): **unclamped** `log1p(-w)` → value `-inf`, grad **NaN** (the real trap); **`log1p(-w.clamp(_EPS,1-_EPS))`** → finite value, grad **0** (safe); `-softplus(raw_w)` → finite, grad -1. ⇒ **Fix = clamp `w` to `[_EPS, 1-_EPS]` before the log, EXACTLY as `ZINBFamily` does for `pi`** (`zero_inflated_negative_binomial.py:74-75`) — NaN-safe and codebase-consistent. The only cost vs log-sigmoid is a *live* gradient at exact saturation (a pinned `w→1` cannot recover) — **accepted**: a collapse to NB is a valid F4 outcome, and ZINB accepts the identical trade for `pi`. Red test at `raw_w = ±20` (exact fp32 saturation — NOT `w=±(1e-6)` / `raw_w≈±14`, where fp32 has not yet saturated so even the unclamped form survives and the test would false-pass). See D-15 (reversed).

---

### C-250: non-deterministic mixture sampler if implemented as select-then-sample instead of draw-both-and-select

| Field | Value |
|-------|-------|
| ID | C-250 |
| Tier | 2 |
| Source | expert-code-review (2026-08-01, upfront design review of Epic #230 S2 #232) |
| Trigger | Implementing the mixture `sample()` by choosing a component per cell and then sampling only that component (data-dependent RNG consumption), instead of drawing both components and selecting |
| Location | planned `views_hydranet/distributions/mixture_negative_binomial.py` (`sample`); must compose with `views_hydranet/distributions/sampling.py:94-100` (per-`(pass,step)` sub-generator seeding, ADR-070) |
| Cross-refs | *v1-review finding "C-3" (generator-aware determinism) — NOT register entry C-03; see §Register Conventions*; S2 #121 (the determinism gate), `test_sampler_dxk.py` |

A select-then-sample mixture draws a data-dependent number/order of randoms per cell, which cannot be vectorized deterministically under a single shared `torch.Generator` — breaking the D×K cube's byte-reproducibility, the h=1 golden anchor, and cross-arm comparability (all of which the experiment's verdict relies on). **Tier 2 (structural fragility with a clear trigger; would surface as non-reproducible cubes).** Fix: mirror ZINB's draw-then-mask ordering — `s1=NBCore.sample(mu1,theta1,k,gen); s2=NBCore.sample(mu2,theta2,k,gen); sel=torch.bernoulli(w_k,gen); out=torch.where(sel.bool(), s1, s2)` — a fixed RNG-consumption order (`s1,s2,sel`); add determinism + reduces-to-single-NB-at-`w∈{0,1}` tests.

---

### C-252: GW pairing must respect the `del g` OOM guard — retaining both arms' cubes will OOM the real run

| Field | Value |
|-------|-------|
| ID | C-252 |
| Tier | 3 |
| Source | expert-code-review (2026-08-01, upfront design review of Epic #230 S3 #233) |
| Trigger | Implementing S3's paired Giacomini–White test by holding both arms' full `(N,S)` sample cubes in memory simultaneously to form the per-cell CRPS differential |
| Location | `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py:136` (`del g` per-arm OOM guard in `score_horizons_v2`) |
| Cross-refs | C-247 (same file — non-portability), the D×K cube disk/RAM guards (`disk_guard.assert_cube_fits`) |

`score_horizons_v2` deliberately frees each arm's `(N,S)` cube (`del g`) before gathering the next, because two full cubes do not fit in RAM. A naive GW implementation that keeps both arms' cubes to compute the per-cell CRPS differential defeats this guard and OOMs on the real 3-seed run. **Tier 3 (fails loud — OOM, not silent — but blocks the run).** Fix: retain only the per-cell CRPS `c` vectors (length `N`, cheap) plus the ex-ante instrument for each arm across the registry loop, and continue to `del` the cubes; compute the GW statistic on the paired `c` vectors.

---

### C-253: GW variance estimator understates SE under the origin×cell panel's serial + spatial dependence → false significance

| Field | Value |
|-------|-------|
| ID | C-253 |
| Tier | 2 |
| Source | expert-method-review (2026-08-01, Epic #230 S1 #231 — decisiveness; Driscoll–Kraay seat) |
| Trigger | Implementing S3's Giacomini–White test with a plain Newey–West HAC variance on the pooled origin×cell loss differential |
| Location | planned GW function in `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py` |
| Cross-refs | C-248 (the stratum), C-252 (the pairing), Giacomini2006, Driscoll-Kraay1998 (gap-to-fetch) |

The Giacomini–White loss differential over an origin×cell panel is dependent **both** serially (the 36-horizon autoregressive rollout couples horizons within an origin) **and** cross-sectionally/spatially (neighbouring cells co-activate). A plain Newey–West HAC on the pooled series treats these as far more independent than they are, **understating the standard error → manufacturing significance** → a wrong "mixture beats NB" verdict on a real null. **Tier 2 (silent verdict corruption via an anti-conservative test).** Fix: an **origin-block bootstrap** (resample whole origin-months, preserving within-origin dependence) for the GW statistic, or a two-way spatial+time cluster-robust variance; pre-register the estimator in S5.

---

### C-254: GW power binds on the evaluation ORIGIN count P (time dimension), not the cell cross-section — a thin origin set makes a null uninformative

| Field | Value |
|-------|-------|
| ID | C-254 |
| Tier | 3 |
| Source | expert-method-review (2026-08-01, Epic #230 S1 #231 — decisiveness; Giacomini seat) |
| Trigger | Running S6/S7 on a small eval origin set and reading a non-significant GW result as a decisive "within-family null" |
| Location | `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py` (`score_horizons_v2` origin/`_support_keys` construction) |
| Cross-refs | C-253 (the variance estimator), C-227 (crps_all hygiene), Giacomini2006 (asymptotics in P) |

Giacomini–White's asymptotics are in the number of out-of-sample forecast periods `P` (the origin/time dimension), **not** the cell cross-section — so despite ~100k stratum cell-months, the effective inferential sample for the GW test is governed by the number of eval origin-months. The empirical sketch confirms the cross-section is abundant (event-rich stratum, 155–169× density lift), so this is **not** a starvation problem — but a thin origin set would still leave the *time-dimension* underpowered, making a null merely suggestive rather than decisive. **Tier 3 (decision/measurement hygiene, peer of C-227):** not silent corruption, but headlining a small-`P` null as "the wall is real within-family" over-claims. Fix: use every origin with ≥36 futures; **report `P`** in the verdict; pre-register that a null at small `P` is suggestive-not-decisive.

---

### C-255: param-health forensic hardcodes the ZINB (μ,θ,π) param layout → the mixture head is silently MISLABELED (and its tail component invisible)

| Field | Value |
|-------|-------|
| ID | C-255 |
| Tier | 2 |
| Source | user-surfaced forensic review (2026-08-02, Epic #230 S7 — mixture nb-vs-NB verdict; two REGRESSION FORENSIC lr_ns_best L300 plots) |
| Trigger | Reading a ≥3-param family's REGRESSION/CLASSIFICATION FORENSIC plot (or its biopsy dossier) and interpreting the μ̄ / θ-CoV / π rows as those quantities for any family whose `activate()` idx0/1/2 are not μ/θ/π |
| Location | `views_hydranet/utils/training_forensics.py:133-138` (`_param_health_stat_names`) + `:177-191` (`_reduce_param_health`, idx0→mu_bar, idx1→theta_cov, idx2→pi_bar); `views_hydranet/utils/visual_diagnostics.py:943-1022` (row labels μ̄/θ-CoV/π hardcoded) |
| Cross-refs | C-213 (family-aware forensics — this is a gap in it), the mixture family `views_hydranet/distributions/mixture_negative_binomial.py` activate order `[w,μ1,θ1,μ2,θ2]`, Epic #230 F4 falsifier |

`_reduce_param_health` hardcodes the ZINB semantics — `mu_bar=mean(idx0)`, `theta_cov=CoV(idx1)`, `pi_bar=mean(idx2)` — and the plotter hardcodes the labels "μ̄ conditional magnitude / θ CoV / π structural zero". Neither is family-aware. For **mixture_nb** (`activate` = `[w, μ1, θ1, μ2, θ2]`) the forensic therefore plots **mean(w) under the "μ̄" label, CoV(μ1) under "θ CoV", mean(θ1) under "π"**, and **drops μ2, θ2, and (1−w) entirely** — the entire tail/second component is invisible. **Tier 2 (a misleading diagnostic can drive a wrong scientific verdict — the exact failure mode this repo has been burned by):** in Epic #230 the mixture forensic's "μ̄→1.0" is really **field-mean w→1.0**, which reads as "body magnitude saturates" but actually concerns the mixing weight, and — being a field-mean dominated by ~99.3% zeros — cannot distinguish component-2 dead-everywhere from alive-only-on-the-tail (the F4 question). It nearly anchored an over-hasty "F4 clean". Fix: make `_param_health_stat_names`/`_reduce_param_health` + the plotter **family-aware** (label channels by the family's own `param_names`/`activate` order; for the mixture add `w̄`, `w|active`, `min(w|active)`, `μ2:μ1`), or at minimum fail-loud/annotate when `n_params` doesn't match the μ/θ/π template. Interim mitigation: the direct `w|active` probe (Epic #230 S7) reads component-2 activity correctly.

---

### C-258: Release-review low-severity config/diagnostic footguns (dev→main PR #252)

| Field | Value |
|-------|-------|
| ID | C-258 |
| Tier | 4 |
| Source | code-review max (2026-08-03, dev→main release review PR #252) |
| Trigger | Relying on `loss_class_pos_weight` with a non-`weighted_bce` class loss; pairing `rollout_feedback='sample'` with a legacy head; or trusting forensic series that include BN-recal windows |
| Location | `views_hydranet/utils/config_initializer.py` (~:687, ~:993); `views_hydranet/utils/utils.py`; `views_hydranet/train/training_engine.py:589` |
| Cross-refs | C-128 (the release review's substantive finding), C-197 (family/legacy disjointness) |

Three low-severity items from the prioritized dev→main release review; **none affect a correctly-specified config's outputs** (all fail-loud or diagnostic-only):
(a) `rollout_feedback='sample'` on a legacy (non-family) `output_distribution` is not rejected at config **load** — it fails loud only at inference-object construction, i.e. after a full training run (wasted GPU run). Fix: a `model_validator` reading `output_distribution` vs `family_names()`.
(b) `loss_class_pos_weight` passes length-only validation for **any** `loss_class`, but only `weighted_bce` consumes it; with `loss_class='focal'`/`'bce'` it is **silently ignored** (a silently-different objective). Fix: tie `loss_class_pos_weight` to `loss_class=='weighted_bce'`.
(c) `_recalibrate_bn` runs `model.train()` with forensics attached; `stage_label=''` suppresses the biopsy plot but `forensics.record`/`record_params` are not gated by it, so BN-recal windows **pollute the forensic accumulators** (diagnostic-only; no weight/output impact). **Tier 4.**

---

### C-259: scheduled-sampling train/inference exposure DECOUPLED by two config keys (ss_feedback vs rollout_feedback) + ungated mean path

| Field | Value |
|-------|-------|
| ID | C-259 |
| Tier | 2 |
| Source | expert-code-review (2026-08-14, ADR-056 scheduled-sampling pre-run correctness review) |
| Trigger | Setting `ss_epsilon_max > 0` to run scheduled sampling on a family head while relying on the `ss_feedback` default (`"mean"`) — with `rollout_feedback` auto-resolving to `"sample"` for family heads, so training exposure ≠ deployment exposure |
| Location | `views_hydranet/utils/config_initializer.py:992-1028` (`validate_scheduled_sampling_params` — no coupling check); `views_hydranet/train/training_engine.py:684` (`ss_feedback` default `"mean"`), `:204-214` (`_family_target_log1p_mean` — UNGATED); `views_hydranet/utils/hydranet_inference.py:264-267` (inference mean IS gated via `compose_mean`), `:100-101` (`rollout_feedback` auto-resolve `None`→`"sample"`) |
| Cross-refs | C-234 (eval-side emit_family_core half-wire), C-239 (training-side twin, closed by ZINBcore arm-drop), C-240/C-242 (gating asymmetry), C-246 (the missing parity test that would catch this) |

`ss_feedback` (training) and `rollout_feedback` (inference) are **independent config keys with independent defaults** (`ss_feedback="mean"`; `rollout_feedback=None`→auto `"sample"` for family heads) and **no validator couples them**. The `"mean"` training feedback path (`_family_target_log1p_mean`) is **ungated** whereas inference's mean path composes the gate (`_emit_magnitude:264-267`). So a scheduled-sampling run left on the `ss_feedback="mean"` default while `rollout_feedback` auto-resolves to `"sample"` **trains on an ungated mean the model never emits** — a silent train/deploy exposure mismatch that makes any scheduled-sampling verdict measure a different object than it deploys. This **generalizes the C-234/C-239 `emit_family_core` mismatch** — which was closed by *dropping the ZINBcore arm + keeping `ss_epsilon_max=0`*, never by fixing the mechanism — to the **default config the moment `ss_epsilon_max>0`**. **Tier 2 (structural fragility, silent exposure mismatch, clear trigger; invalidates the experiment, not a correctly-specified forecast).** Fix direction: a raise in `validate_scheduled_sampling_params` (`ss_epsilon_max>0` ⇒ `ss_feedback == resolved(rollout_feedback)`); gate `_family_target_log1p_mean` + honor `emit_family_core`.

> **Update 2026-08-15 (/review-diff finding, ADDRESSED).** The `validate_scheduled_sampling_params` coupling landed (ss_feedback==resolved rollout_feedback; gated-`mean` reject; order-strict features==regression_targets — C-259/C-260). The `/review-diff` pre-merge pass found the ONE remaining un-guarded axis: `_family_feedback_log1p` is NOT core-aware, so `emit_family_core=True` + a self_zeroed family (zinb) under active SS would still mismatch (train on the self-zeroed sample, roll out on the π-stripped core — the C-234/C-239 axis). Now **guarded**: the validator raises for `ss_epsilon_max>0 + emit_family_core + self_zeroed family` (test `test_emit_family_core_selfzeroed_under_ss_raises`). Residual deferred: the ungated `_family_target_log1p_mean` (can't fire — gated `ss_feedback='mean'` is rejected) and the production `generator=None` non-reproducibility (C-261, docstring corrected).

---

### C-261: scheduled-sampling training feedback draw uses generator=None (global RNG) → non-reproducible; blocks byte-exact parity

| Field | Value |
|-------|-------|
| ID | C-261 |
| Tier | 3 |
| Source | expert-code-review (2026-08-14, ADR-056 scheduled-sampling pre-run correctness review) |
| Trigger | Running scheduled sampling (`ss_epsilon_max>0`, `ss_feedback='sample'`) and expecting byte-reproducibility under the S2 #121 determinism gate, OR asserting train↔inference feedback byte-equality in a test |
| Location | `views_hydranet/train/training_engine.py:231/243` (`family.sample(...,1,None)`; `compose_samples(...,None)`), call site `:348-357`; vs `views_hydranet/utils/hydranet_inference.py:319` + `:446-450` (seeded `fb_gen = torch_seed + sample_idx`) |
| Cross-refs | C-250 (RNG-consumption-order determinism), C-112 (seed-in-sidecar reproducibility), D-12 (per-origin generator re-seed), S2 #121 (determinism gate) |

The training-time scheduled-sampling feedback draw (`_family_feedback_log1p`, `sample` mode) passes `generator=None` → it consumes the **global** RNG, while the inference feedback draw uses a **seeded** `fb_gen` (`torch_seed + sample_idx`, the S2 #121 gate). So SS-trained runs are **not byte-reproducible** on the feedback path, and the pre-run parity test (C-246/C-259) **cannot assert byte-equality** against inference without a seed match. **Tier 3 (reproducibility/comparability, not a live forecast corruption).** Fix direction: thread a seeded `generator` into `_family_feedback_log1p` and its call site (mirroring inference's `fb_gen`).

---

### C-262: scheduled-sampling ε=0 byte-identical no-op is unpinned (only "finite loss" is tested)

| Field | Value |
|-------|-------|
| ID | C-262 |
| Tier | 4 |
| Source | expert-code-review (2026-08-14, ADR-056 scheduled-sampling pre-run correctness review) |
| Trigger | Refactoring the scheduled-sampling substitution / feedback branches (`training_engine.py:332/348`) and relying on `ss_epsilon_max=0` (or `ss_schedule=None`) remaining byte-identical to scheduled-sampling-absent |
| Location | `views_hydranet/train/training_engine.py:332` (`if ss_epsilon > 0.0 and prev_pred is not None`), `:348` (`if ss_epsilon > 0.0` fed-back-copy); `tests/test_scheduled_sampling.py:362` (`test_epsilon_zero_produces_finite_loss` — asserts finite, not byte-identical) |
| Cross-refs | C-246 (train/inference parity discipline), C-259 (same SS piping) |

The ε=0 parity anchor — scheduled sampling being a true no-op ⇒ **byte-identical** training to the scheduled-sampling-absent path — holds only **by construction** (the `if ss_epsilon>0` branch skip). No test pins the byte-identity; the existing `test_epsilon_zero_produces_finite_loss` asserts only *finite* loss. A future refactor of the substitution/feedback branches could silently break the no-op, corrupting the ε=0 baseline of any scheduled-sampling A/B (the 1-variable anchor). **Tier 4 (test-coverage gap; no current corruption).** Fix direction: a byte-identical ε=0 characterization test.

---

### C-263: informed head-init `priors=` is a dead extension point in production — data-derived init unwired; docstrings overclaim (C-199 closure partially overstated)

| Field | Value |
|-------|-------|
| ID | C-263 |
| Tier | 3 |
| Source | repo-assimilation (2026-08-14, delegated tech-debt pass — distributions subsystem) |
| Trigger | Tuning head init to cure a θ/π gradient-death, or expecting `π` to be seeded from the empirical zero-rate — the production head silently uses the hardcoded family defaults instead |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:282` (`initial_raw_bias()` — no priors); `initial_raw_bias` in all four families (`negative_binomial.py:92`, `zero_inflated_negative_binomial.py:132`, `mixture_negative_binomial.py:130`, `truncated_negative_binomial.py:152`); overclaiming docstrings `negative_binomial.py:96-98`, `zero_inflated_negative_binomial.py:135-138` |
| Cross-refs | C-199 (informed init — its RESOLVED note is partially overstated; see below), C-203 (`initial_raw_bias` promotion to ABC) |

Every family implements a `priors` dict lookup (`theta`/`pi`/`w`/`tail_gap`), but the sole production caller passes **no argument**, so `priors` is always `None` and the hardcoded defaults (θ=1.0, π=0.9, w=0.9, tail_gap=5.0) always win; every `priors={…}` call-site is under `tests/`. C-199's "RESOLVED — informed init landed (π≈empirical zero-rate, θ≈global-θ)" is therefore **partially overstated**: the *dead-gradient* fix is delivered by the non-saturated **defaults**, but the advertised **data-derived** init (π from the empirical zero-rate) is **not wired** — the head's own comment admits "the head has no data; data-derived priors are a later refinement," while the family docstrings still say "the A-S6 head inits its theta channel from this" / "the empirical zero-rate the A-S6 head supplies from data." **Tier 3 (maintainability + a latent init-sensitivity trap; no live corruption — the defaults are reasonable, which is exactly why the gap is invisible).** Fix direction: either wire the empirical-zero-rate / global-θ priors through at construction, or soften the docstrings to say the family *defaults* are used and `priors=` is a reserved extension point (and footnote C-199).

---

### C-264: `rollout_horizon` config knob has no runtime consumer (silent no-op)

| Field | Value |
|-------|-------|
| ID | C-264 |
| Tier | 3 |
| Source | repo-assimilation (2026-08-14, delegated tech-debt pass — train/inference/config seam) |
| Trigger | Running rollout-training with `rollout_horizon=K>1` (the value its own docstring names — "Candidate K=12 … the B1 pushforward stability term") expecting multi-step pushforward training, and silently getting ordinary one-step training |
| Location | `views_hydranet/utils/config_initializer.py:265-274` (field definition — the ONLY occurrence in `views_hydranet/`); no reader in `training_engine`/`hydranet_inference` (`training_loop`/`_process_sequence` always take the one-step path) |
| Cross-refs | C-125/C-126 (Axis-B rollout training / ADR-058 — the path that would consume it); the fail-loud-on-ignored-knob pattern (`reject_retired_hurdle_knobs`, `validate_head_samples_family`, `validate_scheduled_sampling_params`) |

`rollout_horizon` (`ge=1`, default 1) advertises real multi-step behavior at K>1, but grep confirms nothing reads it. This is exactly the silent-degradation class the config layer otherwise **fails loud** on — three sibling validators raise when a knob would be silently ignored; `rollout_horizon` is the one behavior-promising knob left with no consumer and no guard. `tests/test_rollout_horizon_config.py` asserts only that the field exists / defaults to 1, not any behavior, so it won't catch the trap. **Tier 3 (silent no-op of an experiment premise; clear trigger).** Fix direction: either a fail-loud validator rejecting `rollout_horizon != 1` until the ADR-058 B1 path lands, or a docstring/register note marking it a parked scaffold.

> **Guard added 2026-08-14 (tech-debt Commit C; entry stays in §Open pending a §Resolved relocation pass).** Added a fail-loud validator `reject_unwired_rollout_horizon` in `config_initializer.py` that raises if `rollout_horizon != 1` until the ADR-058 B1 consumer is wired (default 1 → every current config passes); the field description now says K>1 is rejected. Tests in `test_rollout_horizon_config.py` (K=2 raises, K=1 constructs). Fix-in-code done; the silent-no-op trigger is closed.

---

### C-265: VolumeHandler derivation-chaining parity gap vs DataFetcher (stale `channel_map` inside the derivation loop)

| Field | Value |
|-------|-------|
| ID | C-265 |
| Tier | 3 |
| Source | repo-assimilation (2026-08-14, delegated tech-debt pass — utils subsystem) |
| Trigger | A `derivations` config whose `from` references another derivation's `to` (chained derivations) — the DataFrame training path succeeds, the VolumeHandler path raises "Source feature not found in volume" |
| Location | `views_hydranet/utils/volume_handler.py:813/821` (`_execute_derivations` reads the stale `self.channel_map`; `self._metadata` committed once at `:871`) vs `views_hydranet/utils/data_fetcher.py:178` (`apply_blueprint` reads `df_out.columns` live) |
| Cross-refs | **C-75 (ROOT — the duplicated derivation logic this is a divergence instance of; linked 2026-08-15 review-rr)**; `test_derivation_parity` covers both-raise-if-missing but not chaining |

`_execute_derivations` appends derived channels to a **local** `new_channels`/`new_features` and only commits `self._metadata` at the end, so the in-loop source check reads the pre-derivation `channel_map` and cannot see a channel produced earlier in the same loop. Its declared parity sibling `DataFetcher.apply_blueprint` reads `df_out.columns` **live**, so it CAN chain. A chained-derivation config therefore trains on the DataFrame path but raises on the volume path — a silent train/eval divergence `test_derivation_parity` (both-raise-if-missing) does not cover. **Tier 3 (config-gated latent divergence; not a current live corruption because no shipped config chains derivations).** Fix direction: consult the running `new_channels` (or a merged view) for the source check inside the loop, and add a chained-derivation parity test.

---

### C-266: `to_cube_samples` guards `core=True`+self_zeroed but not the mirror (self_zeroed family under a gating composition) → zeros applied twice

| Field | Value |
|-------|-------|
| ID | C-266 |
| Tier | 4 |
| Source | repo-assimilation (2026-08-14, delegated tech-debt pass — distributions subsystem) |
| Trigger | A direct/ad-hoc `to_cube_samples(...)` call (NOT mediated by `HydraNetConfig`) drawing a `self_zeroed` family (zinb) with `core=False` under `soft_gate`/`threshold_gate` |
| Location | `views_hydranet/distributions/sampling.py:67-71` (guards core+self_zeroed, C-240) vs `:79-102` (no `family.self_zeroed` × `composition` check); `zero_inflated_negative_binomial.py:96` (structural π applied inside `sample()`) |
| Cross-refs | C-234/C-239/C-240/C-242 (emit_family_core cluster; C-240 is the symmetric guard already added) |

`sampling.py` fails loud on the `core=True + self_zeroed` double-count (C-240) but nothing checks a `self_zeroed` family drawn with `core=False` under a gating composition: the family applies its structural π inside `sample()` **and** the cube is then re-masked by the gate at `:102` → zeros applied twice. Today only upstream `HydraNetConfig` prevents this pairing; a non-config-mediated `to_cube_samples` call (the exact scenario the C-240 guard was added to defend) is unprotected for the symmetric direction. **Tier 4 (latent; config currently prevents it — flagged low-confidence-as-new, adjacent to C-240).** Fix direction: mirror the C-240 guard for `family.self_zeroed and composition in {soft_gate, threshold_gate}`.

> **Guard attempted + REVERTED 2026-08-14 (tech-debt Commit C) — stays OPEN.** A sampler-level fail-loud guard (`not core and family.self_zeroed and composition != 'self_zeroed'`) was added then reverted: it collides with `test_to_cube_samples_core_uses_bulk_body`, which legitimately draws a self_zeroed family under `soft_gate` with `core=False` as a mass-comparison probe — the sampler supports that tensor op by design. The invalid-*forecast* prevention already lives upstream in `HydraNetConfig`; a blanket sampler guard is over-reaching. Left as a documented low-tier risk (guard only a direct/ad-hoc `to_cube_samples` misuse, if ever needed).

---

### C-267: per-`(pass,step)` sub-generator seed mixing can collide for large D×T

| Field | Value |
|-------|-------|
| ID | C-267 |
| Tier | 4 |
| Source | repo-assimilation (2026-08-14, delegated tech-debt pass — distributions subsystem) |
| Trigger | Scaling MC-dropout passes or rollout horizon substantially, so two distinct `(pass, step)` pairs land on the same mixed seed |
| Location | `views_hydranet/distributions/sampling.py:94-97` (`base + pass_index*1_000_003 + tt*10_007`) |
| Cross-refs | C-250 (sampler determinism); the ADR-070 per-`(pass,step)` seeding fix (`66a95ea`) |

The per-stream seed is a **linear** mix, not a hash, so distinct `(pass, step)` pairs are not guaranteed collision-free; two streams could share a seed and correlate their aleatoric draws. Harmless at current ranges (~tens of passes × ~36 steps) and only mildly correlates draws if it ever hits, but a latent fragility if pass/step counts grow. **Tier 4 (minor; no current impact).** Fix direction: a proper hash of `(base, pass, step)` (or a per-stream `Generator` fork) if pass/step counts scale.

---

### C-268: `[DEMOTED]` diagnostic plotters under-guarded — unguarded index swallowed by a broad try/except, log-scale without a positivity guard, mutable default arg

| Field | Value |
|-------|-------|
| ID | C-268 |
| Tier | 4 |
| Source | repo-assimilation (2026-08-14, delegated tech-debt pass — utils subsystem) |
| Trigger | Calling the biopsy/forensics plotters with a non-empty-but-short `time_indices` (shorter than the plotted timestep count), or with all-zero/non-positive loss data, during a diagnostics run |
| Location | `views_hydranet/utils/visual_diagnostics.py:662` (`int(time_indices[t_idx]) if time_indices else t_idx` — truthiness not length; sibling guards `t_idx < len(time_indices)` at `:500`; `IndexError` swallowed by the `:612→:712` try/except → silent "skipping plot"); `:763` (`ax.set_yscale("log")` on unfiltered loss; the `:750` comment claims a filter that never happens); `:86` (`features: List[str] = []` mutable default) |
| Cross-refs | — (diagnostics-only; never touches the scored forecast) |

Three small robustness gaps in the diagnostic plotters, all diagnostic-only: a length-unchecked index that raises `IndexError` silently swallowed into a "skipped plot" (asymmetric with the guarded siblings), a log-scale plot with no positivity guard despite a comment asserting one, and a mutable default argument (currently read-only, so harmless, but the standard latent-bug pattern). **Tier 4 (diagnostic-only; no correctness impact).** Fix direction: length-check `time_indices`, filter non-positive before `set_yscale("log")` (or delete the misleading comment), normalize the default to `None`.

> **`[DEMOTED]` to the Tech-Debt Backlog 2026-08-15 (review-rr strategic signal-to-noise pass).** Tier 4, mechanical, single-file, no correctness or reliability impact — actionable as ordinary tech debt rather than a governance risk. Full entry retained here for traceability; indexed in §Tech-Debt Backlog; no longer counted as an active risk.

---

### C-269: ADR-072 amends ADR-019's forward FeatureScaler contract — ADR-019 + the FeatureScaler CIC must be updated in lockstep when frame-native input lands

| Field | Value |
|-------|-------|
| ID | C-269 (renumbered from C-259 on the docs/adr-072 branch merge 2026-08-15 — that branch's C-259 collided with this register's SS-parity C-259) |
| Tier | 3 |
| Source | falsification-audit (ADR-072 section-consistency `/falsify`, 2026-08-13, hard falsification P1) |
| Trigger | Implementing proposed ADR-072 — specifically, changing `FeatureScaler.fit_transform` / `inverse_transform` to consume a views-frames `FeatureFrame` instead of a `pd.DataFrame` |
| Location | `docs/ADRs/proposed/072_frame_native_input_ingestion.md` (Amends-scope + §4.2 Transform role); `docs/ADRs/active/019_feature_scaler_specification.md` (forward contract — Invariant 4 "fail loud on a missing column from the DataFrame"; test-alignment "does not mutate the input DataFrame in-place"); FeatureScaler CIC (`docs/CICs/FeatureScaler.md`) |
| Cross-refs | ADR-019, ADR-072, ADR-000/007 (no semantic change without a same-PR contract update); C-160/C-156 (ADR-062 channel-role — a related but distinct input-boundary coupling) |

Proposed ADR-072 (frame-native input ingestion) amends ADR-019's **forward** FeatureScaler contract from DataFrame-keyed to `FeatureFrame`(`feature_name`)-keyed. ADR-072 now **declares** this amendment (added during the `/falsify` pass that found it), but **ADR-019 itself and the FeatureScaler CIC still describe the DataFrame contract** (Invariant 4 references a missing *column* in the DataFrame; the test-alignment asserts the scaler "does not mutate the input DataFrame in-place"). Per ADR-000 §1 / ADR-007 (no semantic change without a same-PR contract update), the PR that implements ADR-072 **MUST** update ADR-019 (forward Role → `FeatureFrame`; Invariant 4 → "missing `feature_name`") **and** the FeatureScaler CIC in lockstep, or the authoritative FeatureScaler spec will silently contradict the code. This is **not** silent data corruption — the mismatch is loud the moment anyone reads either doc — so **Tier 3** (contract-drift / maintainability), and it is caught now at review rather than at runtime. ADR-019 Role-4 `inverse_transform_volume` is already numpy and is **not** affected. Fix: amend ADR-019 + the CIC in the implementing PR, and add a contract-vs-code test asserting the scaler's forward input type.

---

### C-270: gate↔body target pairing is positional and unenforced — a reordered `classification_targets` silently mis-pairs every composed forecast

| Field | Value |
|-------|-------|
| ID | C-270 |
| Tier | 2 |
| Source | repo-assimilation (2026-08-15, Phase 5 invariant sweep) |
| Trigger | Declaring `classification_targets` in a different ORDER from `regression_targets` (e.g. `[by_ns, by_sb, by_os]` against `[lr_sb, lr_ns, lr_os]`) — both lists complete, correctly named, every validator green — and running any gated `forecast_composition` |
| Location | `views_hydranet/utils/hydranet_inference.py:266` (`compose_mean(means, prob[:, :n_reg], …)`), `:329` (`gate = prob…[:, :n_reg]`); `views_hydranet/distributions/sampling.py:88` (`gate_t = …[..., :n_reg]`); `views_hydranet/train/training_engine.py:248` (`g = gate[:, :n_reg]`), `:501` (`decay_active[:, j]`); no pairing validator in `views_hydranet/utils/config_initializer.py` |
| Cross-refs | C-175 (lr_↔by_ *naming* completeness — necessary but not sufficient), C-260 (order-strict `features == regression_targets`, but only under active SS), C-03/C-123 (hardcoded 3+3 topology, cluster 4), C-174 (name-as-contract) |

Four independent code paths slice the classification gate positionally by `[:n_reg]` and assume gate channel *j* corresponds to regression target *j*. **Nothing validates this.** C-175 covers the *naming* convention (every `lr_<x>` has a `by_<x>`) and C-260 added an order-strict check for `features` vs `regression_targets` — but neither closes the reg↔cls **ordering** axis: a config can be complete, correctly paired by name, and pass every validator while multiplying `lr_sb`'s body by `by_ns`'s gate. The failure is fully silent — shapes match, values are finite, `IntegrityGuardian` is satisfied, and the emitted forecast is superficially plausible. Note C-175's own reasoning ("a true mismatch would likely be caught loudly by the existing `features==regression_targets` cross-checks, hence not Tier 2") does **not** hold on this axis: those cross-checks constrain reg-vs-features, never reg-vs-cls. **Tier 2:** structural fragility with a model-output-correctness consequence, reachable through an ordinary config edit; not Tier 1 only because current production configs happen to be consistently ordered, so no shipped forecast is known to be affected. Fix direction: an order-strict cross-validator pairing `regression_targets[i]` ↔ `classification_targets[i]` by base id, mirroring the C-260 pattern (cheap, one validator, closes C-175's completeness facet at the same time).

---

### C-271: `ReproducibilityGate.audit_manifest` has no production caller — the genome audit C-43 shipped never runs

| Field | Value |
|-------|-------|
| ID | C-271 |
| Tier | 3 |
| Source | repo-assimilation (2026-08-15, Phase 2 reachability sweep) |
| Trigger | Running training with a config missing a core-genome key or with one set to `None` (e.g. `clip_grad_norm`, `dropout_rate`, a loss-specific `params` entry) and expecting the documented fail-fast — the run instead proceeds on implicit defaults with an incomplete provenance record |
| Location | `views_hydranet/infrastructure/reproducibility_gate.py:87-160` (the method; its own docstring at `:98` states "Must be called before `lock_entropy()` and `training_loop()`"). `lock_entropy` **is** called at `hydranet_manager.py:300`, `training_engine.py:86`, `:820`; `audit_manifest` is called **only** from `tests/test_genome_audit.py` — grep finds no call site in `views_hydranet/` |
| Cross-refs | C-43 (RESOLVED — the entry this finding partially invalidates; annotated there), cluster 5 (C-06/C-117/C-49 — config-as-dict), C-112 (seed provenance in the sidecar) |

The 16-key `CORE_GENOME` audit — plus its `LOSS_REG_REGISTRY`/`LOSS_CLASS_REGISTRY` completeness checks — is fully implemented and fully tested (94% coverage, 7 tests) but **dead in production**. The reproducibility contract it advertises therefore rests entirely on `HydraNetConfig` Pydantic validation, which is *not* equivalent: Pydantic permits `extra="allow"` (`config_initializer.py:1141`) and does not check the loss-specific `params` lists the registry declares. C-43 was closed on the mechanism's *existence*; its *reachability* was never verified, and the test suite is green precisely because the tests call it directly. **Tier 3:** no correctness impact on a well-formed config, but a documented safety mechanism that does not run, plus a green suite that implies otherwise (the same false-confidence class as C-165). **Caveat:** an external caller in `views-models` cannot be ruled out from this repository — verify before treating as a defect. Fix direction: call it from `_train_model_artifact()` immediately before `lock_entropy`, or demote the docstring and re-scope C-43's resolution to "library helper provided, not wired".

---

### C-272: rolling-origin window count is silently truncated by a bare `min()` when the partition outruns available history

| Field | Value |
|-------|-------|
| ID | C-272 |
| Tier | 3 |
| Source | repo-assimilation (2026-08-15, Phase 3 evaluation-flow trace) |
| Trigger | Evaluating a partition whose declared test span exceeds what the volume supplies — e.g. a data pull returning fewer months than the partition dict assumes, an off-by-one in `_partition_dict`, or a `partition_bound` filter that drops months — the run scores FEWER origins than the partition declares and reports success |
| Location | `views_hydranet/manager/hydranet_manager.py:322-327` (`num_windows = test_end - (test_start - 1) - time_steps + 1`, passed to `get_rolling_origin_indices`); `views_hydranet/utils/utils_orchestration.py:31` (`last_origin = total_months - time_steps - 1`), `:38` (`actual_windows = min(num_windows, last_origin + 1)`); completion log at `hydranet_manager.py:358` reports only the realised count, with no declared-vs-realised comparison |
| Cross-refs | C-154 (disk-headroom truncation — the same silent-partial-result class, which WAS given a fail-loud guard), C-149-adjacent evaluation-completeness concerns, cluster 12 |

The number of evaluation windows is derived **twice** from independent sources — from the partition dict's month arithmetic in the manager, and from the volume's own length inside `get_rolling_origin_indices` — then reconciled with a bare `min()`. Nothing asserts the two agree, and nothing logs when they disagree: the function warns only in the total-failure case (`last_origin < 0`, `:34`). A short pull or a partition-boundary drift therefore yields a quietly smaller backtest, which downstream scoring averages over without noticing — and, because scores are reported per horizon rather than per origin, the shortfall is invisible in the results. This is precisely the class C-154 was raised and guarded for on the disk axis; the evaluation axis has no equivalent guard. **Tier 3:** silently degraded evaluation with a clear trigger; no data corruption and no wrong per-cell values, so not Tier 2. Fix direction: `logger.warning` (or raise, matching C-154's fail-loud choice) when `actual_windows < num_windows`, and surface declared-vs-realised origin counts in the manager's completion log.

---

### C-273: BN recalibration (C-184's fix, default ON) re-accumulates statistics on the curriculum's hottest windows only

| Field | Value |
|-------|-------|
| ID | C-273 |
| Tier | 3 |
| Source | repo-assimilation (2026-08-15, Phase 3 training-tail trace) — **flagged low-confidence-as-new; may be deliberate** |
| Trigger | Investigating residual eval-time miscalibration on a BN-recalibrated artifact, or tuning `bn_recal_windows`, and assuming the recalibration set is representative of the eval-time activation distribution — it is drawn exclusively from the high-intensity head of the curriculum |
| Location | `views_hydranet/train/training_engine.py:796-800` (`for w in range(n_windows): target, threshold = planner.get_lesson(w)` — indices start at 0); `:790` (`n_windows = config.get("bn_recal_windows", config["windows_per_lesson"] * 10)`); intensity schedule at `views_hydranet/utils/curriculum.py:85-94` (`ratio` decreases monotonically from `max_ratio*roof_ratio`, so low `w` ⇒ high threshold ⇒ busiest cells) |
| Cross-refs | C-184 (the root cause and this fix — Tier 2, "CONFIRMED root cause + universal fix"), C-113 (rollout stability), C-190 (BatchNorm on the skip path) |

`_recalibrate_bn` is on by default (`bn_recalibrate=True`) and mutates the saved artifact's BN buffers, so it determines the eval-mode activation statistics of every shipped model. Its window sampler is driven by `planner.get_lesson(w)` starting at `w=0`, and `get_intensity_ratio` decreases monotonically — so the recalibration set is drawn entirely from the highest-threshold (busiest-cell) end of the curriculum, while inference runs over the full global volume (~99.7% zero). The BN running statistics eval-mode then uses are consequently estimated on an activation distribution denser than the one inference encounters — the *direction* of the very train/eval BN gap C-184 exists to close. **Countervailing evidence:** the fix was empirically validated (6/6 bad seeds flipped BAD→GOOD, 2/2 good preserved, `training_engine.py:786`), so the effect may be immaterial or the head-of-curriculum choice may be deliberate; it is documented nowhere as a choice, and `bn_recal_windows` carries no guidance. **Tier 3** pending verification. Verification direction: A/B the recal window set (curriculum head, as today, vs. `get_lesson(w)` sampled across the full step range vs. uniform full-grid windows) against the frozen T=0 lodestar ruler before treating this as a defect. **Do not change blind** — C-184's fix is validated as-is.

---

### C-274: falsification tests assert on other tests' SOURCE TEXT, not behaviour — one file passes while self-describing as RED

| Field | Value |
|-------|-------|
| ID | C-274 |
| Tier | 4 |
| Source | repo-assimilation (2026-08-15, Phase 6 test-quality audit) |
| Trigger | Reading the suite as evidence that a previously-falsified claim is still falsified (or now closed), or relying on these tests to catch a regression in the gap they document — e.g. deleting the validation-partition coverage in `test_lifecycle_integration.py` and expecting F3-01 to go red |
| Location | `tests/test_falsification_cradle_to_grave.py` — module docstring `:8` ("Verdict: FALSIFIED … Tests are RED stubs — they FAIL to document the gap") vs. actual result (6 passed in 0.03 s); dead condition at `:44-49` (`"run_type.*validation" in source` is a regex written as a literal substring — it can never match — inside an unparenthesised `A or B and C`). Same `read_text()`-and-grep pattern in `tests/test_falsification_repo_clean.py`, `tests/test_falsification_magic_numbers.py`, `tests/test_falsify_dead_list.py` |
| Cross-refs | C-165 (CI `--ignore` set / "distinguish aspirational falsification stubs from broken tests" — the adjacent false-confidence entry), C-247 (collection integrity), ADR-005 (testing as critical infrastructure) |

Several falsification files verify claims by grepping other test files' source rather than by exercising behaviour. In `test_falsification_cradle_to_grave.py` the stated verdict and the observed result now contradict each other: the file announces itself as failing RED stubs, and every test passes. The underlying F3-01 claim **is** genuinely satisfied — `tests/test_lifecycle_integration.py:315,368` exercises `run_type="validation"` through `_evaluate_model_artifact`, and the register already records this at C-43-era resolution notes — but the test would not have detected its absence either: its first condition is unreachable and its precedence is accidental. These tests are structurally brittle (they break on any rename of the files they grep) and give assurance they cannot back. **Tier 4:** documentation/assurance quality; no runtime correctness impact, and the claims they cover are independently true today. Fix direction: convert to behavioural assertions, or migrate the resolved verdicts into the register/ADRs and delete the stubs, so the suite stops carrying self-contradicting status text. Note this file is one of the four CI `--ignore`s (C-165), so it is currently verified only in local runs.

---

### C-275: no record of the upstream data VINTAGE — a re-pull silently moves both the training data and the frozen lodestar ruler's truth

| Field | Value |
|-------|-------|
| ID | C-275 |
| Tier | 2 |
| Source | review-rr strategic blind-spot analysis (2026-08-15) — B1 |
| Trigger | Re-pulling `<run_type>_viewser_df.parquet` (or re-running the datafactory queryset) between a baseline run and a comparison run, then comparing their scores on the frozen T=0 lodestar ruler as if the truth were held constant |
| Location | `views_hydranet/utils/data_fetcher.py:36-64` (`fetch_df` — reads the parquet, logs a load report, records no vintage/hash); `views_hydranet/train/train_model.py:75-137` (`config_snapshot` sidecar — persists architecture, seeds and head config, but no data fingerprint); `reports/2026-07-28_datafactory_migration_dossier/` (the measurement) |
| Cross-refs | C-112 (seed provenance in the sidecar — the same "artifact must be self-identifying" principle, applied to weights but not to data), C-177 (empty FrameMetadata — no provenance on emitted frames), C-30 (artifact SHA-256 — integrity for the model, absent for its input), C-110 (heterogeneous-config ensemble composition) |

Nothing in this repository records **which vintage of the upstream panel a run was trained or scored on**. The Tier-A parity work measured that this is not hypothetical: a fresh pull against the same nominal queryset moved totals by **sb −0.35% / ns +1.25% / os +0.06%**, attributed to L1 nokgi + L2 vintage + L3 geocoding revisions — a real, quantified, non-bug drift in UCDP-derived truth. Two consequences follow. **(1) Comparability:** the frozen T=0 lodestar ruler is frozen in *code* (`lodestar_score.py`) but its *truth* is whatever parquet is on disk, so a "gated_NB beats climatology by 28% on sb h1" claim is anchored to an unrecorded data state and is not reproducible across a re-pull. **(2) Silent regression:** a revision that removes or reclassifies events changes the target field with no error, no warning, and no diff — the `DataSniffer` checks structure and finiteness, never content identity. **Tier 2:** structural fragility that will cause failures (specifically, false or unfalsifiable comparisons) under a realistic and *scheduled* change — the global-server flip will re-pull at a new scale. Not Tier 1 because it corrupts *comparisons* rather than any single forecast, and the direction is a known small drift rather than an arbitrary one. Fix direction: hash the loaded frame (rows × the target columns) at `fetch_df`, log it in the load report, and persist it in the `.pt.config.json` sidecar alongside the seeds — making an artifact self-identifying in its data as it already is in its weights (C-112); optionally fail loud when a scoring run's data hash differs from the artifact's.

---

### C-276: no production forecast monitoring — every guard in this register fires before deployment, none after

| Field | Value |
|-------|-------|
| ID | C-276 |
| Tier | 3 |
| Source | review-rr strategic blind-spot analysis (2026-08-15) — B2 |
| Trigger | Shipping the global (`region="land"`, 360×720) ensemble to the server and running operational `_forecast_model_artifact()` on a monthly cadence — the first month whose 36-step rollout degrades has no signal that would surface it |
| Location | `views_hydranet/manager/hydranet_manager.py:393-405` (`_forecast_model_artifact` — returns PredictionFrames and logs a count; no distributional check on what it emitted); `views_hydranet/utils/hydranet_inference.py:550-564` (the only rollout-health signal is a per-step `logger.warning`/`logger.error` on `max |pred|` > 100/500, and `IntegrityGuardian`'s hard ceiling at 1000); no artifact records a reference forecast distribution to compare against |
| Cross-refs | C-113 (the bloom — a failure that manifests **only** in the free-running rollout, i.e. exactly the regime operational forecasting runs in), C-163 (no runtime resource/environment harness — the operational-readiness sibling), C-177 (no provenance on emitted frames), C-219 (crps_all is Goodhart-prone — the metric a naive monitor would reach for first) |

All 120 open concerns are **pre-deployment**: training dynamics, evaluation methodology, artifact integrity, config validation. None asks *"how would we know that a deployed 36-month forecast has gone wrong?"* The gap is sharp because the system's best-documented failure mode is rollout-specific: the C-113 bloom is invisible at T=0 (where every scored gate lives) and appears only in the free-running trajectory that operational forecasting actually emits — and the V2 scoreboard confirmed ZINB still blooms there (crps_none 34× by h36, seed-42 catastrophic 2.61) even after the ADR-070 mitigation stabilised the gated arms. Today the sole runtime signal is a magnitude threshold on `|pred|`, which the bloom can stay under while the *field* degrades (the bloom's real signature is field-wide `crps_none` + `M_mean`, not `M_max` — established in Epic #193). **Tier 3:** no current correctness impact, since nothing is deployed operationally yet and the trigger is a planned future action; escalate to Tier 2 the moment the global-server ensemble serves. Fix direction: persist a reference forecast summary (per-horizon nonzero rate, mean, and a high quantile) with each artifact, and have `_forecast_model_artifact` compare the emitted frames against it and fail loud on a threshold breach — reusing the free-running-attractor probe in `utils/rollout_diagnostics.py`, which already computes the right quantity but is currently only reachable from `scripts/diagnose_io_gain.py`.

---


### C-277: `block_bootstrap_crps` computes single-arm support where the point estimate uses the cross-arm intersection → a CI that annotates a different cell set

| Field | Value |
|-------|-------|
| ID | C-277 |
| Tier | 3 |
| Source | repo-assimilation → Epic #263 S4 (#268), found while building the rollout-ruler MDE |
| Trigger | Calling `block_bootstrap_crps` to put a confidence interval on a `score_horizons` / `score_horizons_v2` point estimate when the compared arms do **not** have identical coverage — the CI is then computed over a larger cell set than the number it annotates |
| Location | `reports/2026-07-25_t0_rollout_skill_dossier/tools/rollout_skill_score.py:216` (`support = _fixed_support({label: g}, horizons)` — one arm) vs `:126` `score_horizons` and `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py:132` (`set.intersection(*[_support_keys(e[1], …) for e in registry])` — all arms) |
| Cross-refs | C-221 (the origin-block bootstrap this function implements), C-253 (variance estimation), C-217/G4 (the identical-support rule this violates), catalog C7 "different-months bug" |

`score_horizons` and `score_horizons_v2` deliberately intersect `_support_keys` across **every** arm in the registry before scoring, so all arms are compared on one cell set (the G4 identical-support rule; comparing across substrates is the "different-months bug"). `block_bootstrap_crps` does not: it gathers a single arm and calls `_fixed_support({label: g}, horizons)`, so its resampling universe is that arm's own coverage. Where arms differ in coverage — which is exactly when the intersection matters — the interval describes a different population than the point estimate it is attached to.

**It has never fired**, because the current arms have identical coverage (all 13 origins × 13,110 cells; verified 6/6 in `partition_audit.json`, Epic #263 S2), so the two supports coincide. **Tier 3:** latent, not silent-corrupting today, but it is a correctness trap sitting on the inference path.

**Deliberately pinned, NOT fixed** (Epic #263 `SCOPE.md`, "the one sanctioned edit" rule). Epic #263 S6 uses `gw_stratified.score_gw_v2`, which computes support correctly, so `block_bootstrap_crps` is unused there — fixing an unused function mid-epic is scope creep, and leaving it undocumented is a trap. The honest middle is a failing test that the suite knows about: `tests/test_rollout_ruler_trust.py::test_block_bootstrap_uses_cross_arm_support`, marked `xfail(strict=True)` with this ID in its reason. Fix direction (2 lines): pass the already-intersected `support` into `block_bootstrap_crps` instead of recomputing it, and flip the test to green.


> **Second instance found and FIXED 2026-08-15 (PR #273 review).** `/review-diff` + a deep read found the same defect re-implemented in `reports/2026-08-15_rollout_ruler_trust_dossier/tools/rescore_v2.py::_add_origin_block_ci`, which derived the bootstrap's cell universe from **one arm's** coverage while annotating a point estimate computed on the cross-arm intersection. Unlike the `block_bootstrap_crps` instance, this one was **on a live path** — it produced the CIs in `results/rescore.csv`. Fixed by passing the caller's intersected `support` in, which also allowed the climatology to be hoisted out of the arm loop (it is horizon- and arm-invariant under a fixed anchor), making a 7-horizon pass cheaper than the previous 3-horizon one. Guarded by `tests/test_rollout_ruler_trust.py::test_ci_pass_uses_the_caller_support_not_per_arm_coverage`. **The `block_bootstrap_crps` instance remains open and `xfail`-pinned** — still unused. That the same defect was written twice, once while explicitly registering the other, is the argument for fixing rather than only pinning it.

---


### C-278: FAO-02 — the LOCKED evaluation framework — is not in the repository, and every in-repo citation is a dangling path

| Field | Value |
|-------|-------|
| ID | C-278 |
| Tier | 3 |
| Source | Epic #263 S0/S7 (#264/#271) — surfaced while sourcing the climatology baseline FAO-02 mandates |
| Trigger | Anyone (a new contributor, a reviewer, a future session, CI) trying to check a selection decision against the framework that governs it — following any of the 6 in-repo citations lands on a path that does not exist in the tree |
| Location | Cited from `scripts/proper_score_audit.py:29-33`, `scripts/tail_scorecard.py:4,20`, `docs/ADRs/proposed/071_violet_visitor_datafactory_provider.md:83`, `views_hydranet/utils/count_mean_loss.py:11`, `views_hydranet/utils/lognormal_nll_loss.py:5` — all naming `reference_fao02_locked_eval_framework.md`, which resolves to **nothing** inside the repo. The only copies are a Claude memory file and a PDF under `~/brain/2_projects/fao02/`. |
| Cross-refs | C-224 (Tier 1 — its governance ask is *to the FAO-02 owner*), C-219/C-231 (the metrics FAO-02 blesses), C-275 (data vintage — the same "the substrate is not versioned in git" class) |

FAO-02 is the **LOCKED** framework governing every selection decision: CRPS primary, QS99/Brier/MCR guardrails, twCRPS/LogScore/PIT rejected, evaluation on the full dataset, an empirical conflictology baseline, 5%/1% margins, decisions on the validation partition. Five source files enforce it by name. **None of them can be followed**: the referenced document is outside the git tree, unversioned, and its in-repo pointer is dead.

Two consequences, both live. (1) **Unverifiable governance** — a reviewer cannot confirm that a claimed FAO-02 compliance is real. (2) **Silent drift** — the framework can change (or be misremembered) with no diff, no review, and no way for a test to notice; Epic #263 found the mandated climatology baseline had **never been implemented at all**, which is exactly the failure mode an unversioned contract produces.

**Tier 3:** no silent output corruption, and the framework's content is currently applied correctly where it is applied — but it is a governance contract that cannot be audited from the repository that depends on it. **Deliberately not fixed by Epic #263** (`SCOPE.md` #8: vendoring FAO-02 is a separate scope). Fix direction: vendor it into `docs/specs/` (or an ADR) with a version/date header, repoint the 5 citations, and add a link-integrity test — the repo already has the pattern in `test_risk_register_integrity.py`.

---


### C-279: `climatology_resample` duplicates views-baseline's `ConflictologyModel` with no parity test

| Field | Value |
|-------|-------|
| ID | C-279 |
| Tier | 3 |
| Source | maintainer challenge during Epic #263 review (2026-08-15) — the claim "the FAO-02 climatology was never implemented in code" was **false** and was corrected |
| Trigger | Any future comparison that scores one arm against `scripts/rollout_ruler_core.climatology_resample` and another against a deployed `ConflictologyModel` run (`white_ranger` / `light_strider`) — two objects with the same name and no guarantee they agree |
| Location | `scripts/rollout_ruler_core.py::climatology_resample` vs `views-baseline/views_baseline/model/models/distributional/conflictology.py::ConflictologyModel` (+ `model/frames/pooling.py::window_pool`); deployed as `views-models/models/{white_ranger,light_strider}` (`window_months=36`, `n_samples=64`, `seed=42`) |
| Cross-refs | C-75 / C-265 (the same duplicated-logic-with-a-parity-gap shape, for derivations), C-278 (FAO-02 outside the repo — the reason the duplication was not noticed), views-baseline **#82** (which window convention is correct) |

The FAO-02 empirical conflictology baseline **is** implemented — as `ConflictologyModel` in views-baseline, deployed as `white_ranger` and `light_strider`. Epic #263 nevertheless built a second implementation inside the hydranet scorer, because scoring against the deployed model requires its prediction cubes and those are deleted after scoring. The *need* is real; the **duplication** is the risk.

Three divergences existed on first write. Two were **mis-diagnosed**: `n_samples` and `seed` were aligned *to the canonical model* (64 / 42) when the correct target is the **arms' cube width** — `05_analysis_plan.md` pins "S = 16 (matches the arms' cube width)", and the parenthetical is the invariant. Drawing the reference at 64 against arms at 16 left an uncancelled `O(1/S)` CRPS estimator bias and a coarser AP rank resolution, both biasing toward the epic's own conclusion (**corrected in PR #274**: `rescore` derives S from the cubes via `reference_sample_width` and has no `n_samples` parameter; seed returns to the pre-registered 0). The third was real: an **off-by-one** in the pool bound (`range(end-window, end)` vs the canonical `time <= train_end`, i.e. 420–455 instead of 421–456 — **fixed**, the bound is now inclusive); and the pool **anchor** — canonical fixes it at `train_end`, the hydranet version originally slid it per origin (now `window_anchor`, defaulting to the canonical fixed behaviour, with the sliding variant retained and the question raised upstream as views-baseline #82).

**Fidelity is now a test, and the manual number had already drifted.** The recorded "0.9591 vs `light_strider`'s 0.9601 — 0.1% apart" was a stale intermediate: the run it described wrote **0.96216** (0.21%), and the figure had been copied into five files including a tracked module docstring, with nothing able to detect the divergence — the predicted consequence of "a one-off manual comparison, not a test", realised inside the same PR that registered it.

PR #274 replaces the prose figure with `test_stand_in_matches_the_archived_baseline`, which reads the shipped `rescore.csv` and `csv_decomposition.csv` and fails above 1%. At the corrected matched width the stand-in scores **0.96157** vs **0.96014** — **0.149%** apart, i.e. matching the arms made it *more* faithful, not less.

**Still open:** this pins agreement on one cell of one shipped artifact, not agreement between the two implementations. The preference order is unchanged — consume views-baseline as a dependency and delete the local copy; else a real parity test against a live `ConflictologyModel`.

**Tier 3:** no wrong output today (the Epic #263 verdict is ARTIFACT under all four parameterisations tested, so it does not turn on this), but two implementations of a *governance baseline* with no parity gate is precisely the C-75/C-265 shape, and this one sits on the selection path. Fix direction, in preference order: (a) declare `views-baseline` a dependency and consume `ConflictologyModel` directly, deleting the local copy; (b) if the cube-availability constraint makes that impractical, add a parity test pinning the reimplementation against a stored canonical output; (c) at minimum, keep the 0.9591-vs-0.9601 check as a documented, re-runnable comparison rather than a one-off.

---


### C-280: `partition_audit` renders an unrun provenance check as `"n/a"`, which reads as *not applicable*

| Field | Value |
|-------|-------|
| ID | C-280 |
| Tier | 4 |
| Source | `/expert-code-review` + deep read, PR #273 |
| Trigger | Renaming or removing `v2_ruler.V2_TRUTH_SHA256`, or running the audit where the truth parquet is absent — the sha comparison silently does not run and the report prints `n/a` |
| Location | `reports/2026-08-15_rollout_ruler_trust_dossier/tools/partition_audit.py:157` (`if truth_parquet is not None and Path(truth_parquet).exists()`), `:160` (`pin = getattr(v2, "V2_TRUTH_SHA256", None)`, then `if pin is not None and sha != pin`), `:229` (`'ok' if r['truth_matches_pin'] else 'n/a'`) |
| Cross-refs | C-278 (the same "a guard nobody can follow" family), the C-219 pattern of a guard that does not cover its own inputs |

The truth-substrate check degrades silently in two ways: a missing parquet skips it, and a renamed constant turns it off via `getattr(..., None)`. Either leaves `truth_matches_pin=None`, which the markdown renders as **`n/a`** — indistinguishable from "not applicable to this arm". A reader sees a completed audit table. **Tier 4:** inert today (all 6 arms report `True` against a present pin), and the failure is toward *not asserting* rather than asserting something false. Fix: render three states, and raise when a pin exists but the parquet does not.

---

### C-281: `csv_decompose` drops unmatched rows silently, so the denominator of "N of M arms beat climatology" is unreported

| Field | Value |
|-------|-------|
| ID | C-281 |
| Tier | 4 |
| Source | `/expert-code-review` + deep read, PR #273 |
| Trigger | Decomposing a board where the reference arm was not scored at every (target, h) — those arm-rows vanish from `csv_decomposition.csv` with no count and no warning |
| Location | `reports/2026-08-15_rollout_ruler_trust_dossier/tools/csv_decompose.py:83` (`if b is None: continue`) |
| Cross-refs | C-219 (headline completeness), C-280 |

189 rows were written from 198 archived rows; nothing reports how many were dropped or why. The dossier makes claims of the form *"12 of 13 arms beat climatology"*, whose denominator comes from this file. A silent drop moves that denominator invisibly. **Tier 4:** no wrong value, and the current drop count is explicable (the reference is not scored against itself), but a count belongs in the output. Fix: count and log the skips; fail if the drop rate exceeds a threshold.

---

### C-282: `float(x or "nan")` is falsy-triggered, and `size_ratio == 0.0` is load-bearing evidence

| Field | Value |
|-------|-------|
| ID | C-282 |
| Tier | 3 |
| Source | `/expert-code-review` + deep read, PR #273 |
| Trigger | Any refactor that feeds `decompose_all` float-valued rows instead of `csv.DictReader` strings — every genuine `size_ratio == 0.0` then silently becomes `nan` |
| Location | `reports/2026-08-15_rollout_ruler_trust_dossier/tools/csv_decompose.py:102-103` (`float(r.get("size_ratio") or "nan")`, `float(r.get("mcr_all") or "nan")`) |
| Cross-refs | C-231 (`size_ratio 0.0` is part of the ARTIFACT evidence), C-280/C-281 |

The `or` fires on **falsiness**, not absence. It is correct today only because `csv.DictReader` yields strings and `"0.0"` is truthy. **124 of 189 rows have `size_ratio_model == 0.0`**, and that zero is quoted as evidence in the ARTIFACT verdict — the model predicts no magnitude at all. One refactor to float rows would erase exactly the datum the conclusion rests on, silently, in the direction of "no data" rather than "zero". **Tier 3** rather than 4 because the failure mode destroys load-bearing evidence rather than merely under-reporting. Fix: `float(v) if (v := r.get("size_ratio")) not in (None, "") else float("nan")`.

---


### C-283: three repo-hygiene tests shell out to a `ruff` binary the CI test environment never installed — guaranteed-red, independently of the code

| Field | Value |
|-------|-------|
| ID | C-283 |
| Tier | 3 |
| Source | PR #273 CI investigation (2026-08-15) — the branch's `test` job failed; the same three tests fail identically on `development`'s own tip (60ed69f) |
| Trigger | Reading the CI `test` job's result to decide whether a change is safe to merge — it has been red since at least 2026-08-14 for an environment reason, so a genuine regression would be indistinguishable from the standing failure |
| Location | `environment.yml` (the `pip:` list installs `.`, `pytest`, `pytest-cov` — not `ruff`); `pyproject.toml` declares `ruff` only under `[tool.poetry.group.dev.dependencies]`, which `pip install .` does not install; consumed by `tests/test_falsification_repo_clean.py::TestF4_02_LintViolations::{test_ruff_check_passes,test_ruff_format_passes}` and `::TestF4_03_DeadCode::test_no_unused_imports_or_variables`, all of which `subprocess.run(["ruff", ...])` |
| Cross-refs | C-165 (the same file's other CI blocker, and the same class: "suite green" meaning something other than what it appears to), C-247 (test portability) |

Three tests invoke the `ruff` **binary** through `subprocess`. The GitHub Actions `test` job runs inside the conda env built from `environment.yml`, which has no `ruff`, so all three raise `FileNotFoundError: [Errno 2] No such file or directory: 'ruff'` on **every** run. Verified on two independent commits: PR #273's head and `development`'s tip.

Locally they pass, because a developer machine usually has a `ruff` on PATH — which is how the divergence persisted: the failure is invisible where the code is written and unavoidable where it is checked. Worse, a local binary can be a *different version* from the `ruff==0.14.14` the `lint` job pins, so the two jobs can disagree about what "clean" means.

**Tier 3:** no wrong output, but it is a false-confidence and signal-loss problem on the merge gate — exactly C-165's shape. **Fixed in this PR** by adding `ruff==0.14.14` to `environment.yml`'s pip list, matching the lint job's pin. Registered rather than merely fixed because the *class* — a test asserting on tooling the test environment does not provide — is worth having on record, and because it means every CI `test` result before 2026-08-15 should be read as "red for this reason unless shown otherwise".

---
### C-284: `gpd_pwm_fit` is run entirely outside its validity range, and the saturation is not cross-checked against the repo's own MLE fit

| Field | Value |
|-------|-------|
| ID | C-284 |
| Tier | 3 |
| Source | `/code-review max`, PR #274 (2026-08-15); saturation curve re-measured independently before registering |
| Trigger | Reading a `diag_gamma_*` at or above ~0.9 as an estimate — e.g. concluding from `tail_index.md`'s h=36 rows that the tail is heavy-but-finite-mean, or comparing two arms' fitted shapes in that regime |
| Location | `scripts/rollout_ruler_core.py::gpd_pwm_fit`; consumed by `taillardat_index`; published in `reports/2026-08-15_rollout_ruler_trust_dossier/results/tail_index.md`. The alternative fit is `reports/2026-07-15_volatility_ceiling_dossier/tools/s5_tail.py::gpd_xi` (scipy MLE) |
| Cross-refs | C-224 (the tail diagnostic this serves), C-279 (duplicated implementations with no parity test — the same shape) |

Probability-weighted moments are consistent only for `gamma < 0.5`, and the `a1 = E[Y(1-F)]` moment ceases to exist at `gamma >= 1`. Measured on exact GPD quantiles (n=200k, no sampling noise) the estimator saturates one-sidedly: true `gamma` 0.30/0.50/0.70/0.90/1.00/1.10/1.30 fits to 0.30/0.50/0.69/0.86/0.92/0.96/0.99.

All nine published shapes sit at 0.51–0.965, and the **three h=36 rows (0.951–0.965) are at the ceiling** — they cannot distinguish a heavy tail from the infinite-mean regime (`gamma >= 1`), which is precisely the question C-224 exists to ask. The only test pinned `gamma=0.5`, the last point where the estimator is still accurate, so no test covered the regime every published number occupies.

**Mitigated, not resolved, in PR #274:** `taillardat_index` now emits `diag_gamma_at_pwm_ceiling` and `tail_index.md` marks the affected rows `†` with the bias curve stated, so the caveat travels with the number. The estimator itself is unchanged, and **nothing cross-checks it against `s5_tail.gpd_xi`**, which is valid across this range. Swapping or cross-checking the fit is the open work. `diag_Tu` reaches no decision rule, which is why this is Tier 3 and not higher.

---

### C-285: the test environment and the lint environment are specified in three unsynchronised places, and `matplotlib` is in none of them

| Field | Value |
|-------|-------|
| ID | C-285 |
| Tier | 3 |
| Source | `/code-review max`, PR #274 (2026-08-15) |
| Trigger | Adding a test that imports `matplotlib`, or bumping the ruff pin in one file and not the others — the CI `test` job then fails for an environment reason that looks like a code regression |
| Location | `environment.yml` (pip list: `.`, `pytest`, `pytest-cov`, `ruff==0.14.14`); `pyproject.toml:33-36` (test extra, declares `matplotlib`); `.github/workflows/` (the `lint` job's own ruff pin) |
| Cross-refs | C-283 (the same file, the same class — fixed one instance of it), C-165 |

C-283 fixed the missing `ruff`. The underlying condition is unchanged: `pip install .` installs neither the test extra nor the dev group, so `pyproject.toml`'s declaration that `matplotlib` belongs to the test environment is not honoured by the environment CI actually builds — the next test to import it fails identically to C-283. And the ruff version is now pinned in `environment.yml`, `pyproject.toml` and the workflow independently, so the `lint` and `test` jobs can silently disagree about what "clean" means.

**Not fixed here** — the right fix is one source of truth for the test environment, which is a repo-wide change and outside this PR's scope (`SCOPE.md`).

---

### C-286: `partition_audit` records two fields that cannot express the state they name

| Field | Value |
|-------|-------|
| ID | C-286 |
| Tier | 4 |
| Source | `/code-review max`, PR #274 (2026-08-15) |
| Trigger | Reading `partition_audit.json` to establish that the leak check ran and passed, or filtering arms on `truth_matches_pin == False` expecting to find mismatches |
| Location | `reports/2026-08-15_rollout_ruler_trust_dossier/tools/partition_audit.py` — `leak` (raises on True immediately above, so the recorded value is always `False`), `truth_matches_pin` (`False` both when the sha mismatches and when the pin is simply absent) |
| Cross-refs | C-280 (the `"n/a"` rendering of the same field — the same *not applicable vs not checked* conflation) |

Neither field can carry bad news: a genuine leak raises before the record is built, so `leak: false` means "the audit completed", not "the leak check ran and found nothing" — indistinguishable from an audit where the check was somehow skipped. `truth_matches_pin` conflates *checked and mismatched* with *no pin to check against*; a mismatch also raises, so the only reachable `False` is the second meaning.

Tier 4: no wrong number today, and the raises do the real work. It is on record because the audit's stated purpose is making claims **auditable after the fact**, and a field that can only hold one value audits nothing.

---

### C-287: `--diagnostic-only` is run-global while the C-218 gate it disables is per-arm

| Field | Value |
|-------|-------|
| ID | C-287 |
| Tier | 3 |
| Source | `/code-review max`, PR #274 (2026-08-15) |
| Trigger | Auditing a mixed batch — one known `mean`-feedback arm alongside `sample`-feedback arms — and passing `--diagnostic-only` to get the first one through |
| Location | `reports/2026-08-15_rollout_ruler_trust_dossier/tools/partition_audit.py` — `main()` passes `args.diagnostic_only` to every `audit_arm` call; `audit_arm` applies it per arm |
| Cross-refs | C-218 (the invariant), C-286 (the same file's audit-record weaknesses) |

The flag exists so a broken-by-construction rollout can be scored as a **labelled diagnostic** rather than as deployed skill. Applied run-globally it suspends that gate for *every* arm in the batch, so a second arm that unexpectedly carries `rollout_feedback='mean'` passes silently and is recorded `diagnostic_only: true` — a true record of a decision nobody made about it. The gate should be opted out of per arm, e.g. `arm=dir:diagnostic`.

---
### C-289: a diagnostic's tie-breaking rule can manufacture the answer it is built to measure

| Field | Value |
|-------|-------|
| ID | C-289 |
| Tier | 2 |
| Source | authoring review of the gate-structure probe (feedback-realism dossier, 2026-08-16) |
| Trigger | Adding a new field-structure statistic, or reusing `topk_mask` on a field with many equal values (a saturated gate, a thresholded mask, an integer count field) |
| Location | `views_hydranet/utils/gate_field_structure.py` (`topk_mask`); `tests/test_gate_field_structure.py` |
| Cross-refs | C-292 (the same class: a diagnostic whose result is predetermined), C-290 |

`topk_mask` selects the k most probable cells to measure how clustered the gate's belief is. The first
implementation used `argsort`'s default ordering, which breaks ties **by flat index** — i.e. in raster order,
which is *spatially contiguous by construction*. On a gate field where thousands of cells share a value
(common: the gate saturates toward 0 over most of the map), the tie-break itself would have supplied the
clustering the probe was built to detect. Measured on real data before the fix: **1.00 against a true 1.06** —
a null manufactured out of the sort order.

Fixed by breaking ties **randomly** from the probe's own seeded generator. Registered rather than closed
silently because the defect class is not specific to `topk`: **any diagnostic that resolves ambiguity by a
rule correlated with the quantity under test will confirm itself.** The same trap sits in nearest-neighbour
selection, quantile binning on discrete counts, and any `argmax` over a plateau.

**Fix direction:** for every new structure statistic, ask what happens when the input is *constant*, and add
that as a test. A structure metric on a constant field must return the no-structure value, not a value
inherited from the traversal order.

---

### C-290: the emitted field is sampled per-cell independently — the joint structure is discarded by construction

| Field | Value |
|-------|-------|
| ID | C-290 |
| Tier | 2 |
| Source | feedback-realism dossier EXP-03 + the correlated-sampler follow-up (2026-08-16) |
| Trigger | Proposing any fix that improves per-cell **marginal** probabilities (new statics, covariate channels, position encodings, a better-calibrated gate) as a remedy for rollout collapse |
| Location | `views_hydranet/utils/hydranet_inference.py` (the rollout draw); `views_hydranet/utils/correlated_bernoulli.py` (the tested alternative) |
| Cross-refs | C-152 (why coords were inert — the mechanism), C-289, C-222 |

The fed-back occurrence field is drawn cell-by-cell from independent Bernoullis. Real conflict is spatially
clustered, so the drawn field is **correct in its marginals and wrong in its joint** — right number of active
cells, wrong arrangement. `spatial_scramble` shows this is where the damage is: holding active count and
magnitudes fixed and moving only the *locations* reproduces the collapse (gate AP 0.3008 → 0.0097 against a
free-running 0.0070).

Two measured facts keep this from being a straightforward fix:

1. The gate's **own probability field** diffuses during the rollout (Moran's I 0.409 → 0.192 by step 6 on target sb, while the ORACLE holds 0.507 → 0.494), so
   there is less joint structure to preserve than at h=1. Independent sampling is ~10× more destructive on a
   diffuse gate than a sharp one (25× vs 2.6×), so the two compound in a loop.
2. **A coherent sampler alone is not sufficient.** A Gaussian-copula sampler with exactly-preserved marginals
   was built and swept over length scale: fed-field clustering spans **0.011 → 1.064, a 100× range that
   brackets the real value of 0.449**, and gate AP **stays at ~0.007** against an oracle of 0.30. The null is
   credible because the sweep *overshot* the target — "it did not clump enough" is not available as an
   explanation. Clustering is a *proxy* for correct placement, not a substitute: a field can be perfectly
   clustered in the wrong places. **Read at one significant figure:** a generator desynchronisation (C-296)
   left treatment and control unpaired, so the direction and magnitude hold but "0.0069 vs 0.0070" is not a
   paired difference.

**Consequence, and the reason for Tier 2:** this converts a family of proposals into a known null. Any
intervention acting on marginals inherits C-152's result unless it also changes the joint *or* the sampler —
and the sampler alone is now measured as insufficient. Scope: 40 lessons, seed 42, one vehicle;
**INDICATIVE**.

---

### C-291: `spatial_scramble` cannot separate spatial structure from geographic grounding

| Field | Value |
|-------|-------|
| ID | C-291 |
| Tier | 3 |
| Source | feedback-realism dossier `SCOPE.md`, stated before the run (2026-08-16) |
| Trigger | Quoting the 89% occurrence share as the damage attributable to *clustering*, or designing a fix that targets clustering alone on the strength of that arm |
| Location | `views_hydranet/utils/feedback_field_transforms.py` (`spatial_scramble`); `reports/2026-08-16_feedback_realism_dossier/SCOPE.md` |
| Cross-refs | C-290, C-152 |

Permuting the positions of active cells necessarily breaks the field's alignment with the static channels,
because in this data the plausible locations **are** the clustering — they are not two properties that can be
varied independently. The arm therefore measures "spatial structure **and** its geographic grounding" as one
quantity, and no experiment in the current set splits them.

This is an **irreducible confound of the design, not a defect of the implementation** — registered so the
89% figure is not later quoted as a clustering-specific number. It was stated in `SCOPE.md` before the arm
ran rather than discovered afterwards, which is the only reason the arm is readable at all.

**Fix direction:** a scramble constrained to permute only among cells with matching static covariates would
hold grounding fixed while destroying structure. Not built; it may not be feasible at this grid resolution
if the matching classes are too small to permute within.

---

### C-298: headline figures are assembled from different cells of the results grid and quoted as one comparison

| Field | Value |
|-------|-------|
| ID | C-298 |
| Tier | 2 |
| Source | verification pass while building the Claims Ledger (2026-08-17) |
| Trigger | Writing a dossier README, an issue comment or a PR body that quotes a "X → Y" pair or an "A vs B" ratio drawn from a results grid with more than one axis (horizon x arm x target x step) |
| Location | `reports/RESULTS_LEDGER.md` §Verification pass; `reports/2026-08-16_feedback_realism_dossier/00_README.md` |
| Cross-refs | C-289 (a diagnostic whose rule manufactures its answer), C-291, C-296 |

Recomputing all nine headline figures of the feedback-realism dossier from the committed CSVs found **three
wrong or misleading, all failing the same way** — values taken from different cells of the results grid and
presented as a single comparison:

- **"Moran's I 0.50 → 0.16 by step 6"** paired the **oracle's** value (0.507) with the **free-running** one
  (0.16–0.19) as though one run produced both. Real free-running trajectory: 0.409 → 0.192.
- **"`thin:0.75` matches the collapse's activation rate (0.33 vs 0.27)"** paired `thin` at **h18** with the
  collapse at **h36**. At matched horizons the pairs are 0.332/0.291 (h18) and 0.317/0.266 (h36).
- **"25× vs 2.6×"** quoted the free-running **peak** (26.8× at step 12) against a genuinely stable oracle
  value, without noting that the free-running ratio runs 4.4× → 27× → 14×.

None changed a conclusion — the underlying effects are 20-30x and survive — but each was quotable, and two
had already propagated into this register. The results grid here is four-dimensional (arm x horizon x target
x step) and the prose is one-dimensional, so **every sentence silently collapses three axes**. That is the
mechanism, and it is not a lapse of care: the same author wrote the correct number in the log and the
conflated one in the summary.

**Fix direction:** a figure quoted in prose must name its cell — arm, horizon, target — or be a range across
the axis it collapses. The Claims Ledger's verification pass is the check; run it before a dossier's
conclusions are cited anywhere outside the dossier, not after they have propagated.

---

### C-299: a vehicle with no dynamic range makes nulls uninformative and positives understated

| Field | Value |
|-------|-------|
| ID | C-299 |
| Tier | 2 |
| Source | `reports/postmortem_floor_limited_vehicle.md` (2026-08-17) |
| Trigger | Choosing a vehicle for an intervention experiment, or reading a null off one — especially inheriting a vehicle from a previous dossier because it is fast |
| Location | `reports/2026-08-14_scheduled_sampling_dossier/` (VOID as run); `RESULTS_LEDGER.md` M1–M9, I-B/I-C/I-E |
| Cross-refs | C-293 (the sibling: a collapsed *reference*), C-298, C-300 |

Three days of experiments ran on `truncated_smoke`, whose control at h18 scores **0.0070 against a
prevalence of 0.009077 — 0.77×, i.e. BELOW random ranking**. A control with no room to move breaks
measurements **in both directions, asymmetrically**:

* a **null** is uninformative — nothing can move a number already at zero, and the readout cannot separate
  "no effect" from "no resolution";
* a **positive** is understated — a degradation arm cannot fall below a control that has already fallen.

Decisive case: `spatial_scramble` (active count and magnitudes held byte-identical, only locations
permuted) read **+0.9% of the gap** on `truncated_smoke` and **−93.7%** on `violet_visitor`. Same code,
same arms, different vehicle.

**Cost:** it voided a *correctly-designed* experiment — the 2026-08-14 ε sweep, one variable, seed fixed,
four doses, six GPU-hours, with three of its four arms at or below random ranking.

**Why nothing caught it:** every falsifier in that plan (F1, F2, F-DEGEN) assumes the readout is valid.
**No existing falsifier fires on an uninformative measurement.** A pre-registration can be perfect about
what would refute the hypothesis and silent about whether the instrument can see anything at all.

**Fix direction — a pre-registered floor gate on the control arm, checked before any treatment arm runs:**

* **FG-A** `AP_ctrl(h*) ≥ 5 × prevalence(h*)` — the ranker must beat chance at the readout horizon;
* **FG-C** `(1 − θ)·AP_ctrl(h*) ≥ 3 × MDE_AP(h*)` — the pre-registered effect must exceed the resolution.

Validated on committed data: `truncated_smoke` 0.77× (FAIL), `violet_visitor` 28.30× (PASS) — a 36×
separation. Computable from the control's own score CSV, zero extra GPU. **Not yet wired into any driver.**

---

### C-301: `total_lessons` silently reshapes the curriculum, so "train longer" is never one variable

| Field | Value |
|-------|-------|
| ID | C-301 |
| Tier | 3 |
| Source | code read during `reports/2026-08-18_lesson_curve_dossier` pre-registration (2026-08-18) |
| Trigger | Comparing two runs that differ in `total_lessons` — a lesson curve, a convergence check, or quoting a 300-lesson result against a 160-lesson one |
| Location | `views_hydranet/utils/curriculum.py:34,85`; every `config_hyperparameters.py` carrying `total_lessons` |
| Cross-refs | C-299 (the 40-lesson vehicle this interacts with), C-184 (BN recalibration) |

`CurriculumLearner.get_intensity_ratio` computes its cooling slope as

```python
self.total_steps = config["total_lessons"] * config["windows_per_lesson"]   # :34
b = (-self.max_ratio + self.min_ratio) / (self.total_steps * self.slope_ratio)   # :85
```

so **the difficulty schedule is normalised by the training length**. A 600-lesson run is therefore *not*
a 160-lesson run continued — it is the same curriculum shape stretched over 600 lessons, and every
lesson index sees a different event threshold than it would at another length. Changing `total_lessons`
changes two things at once, and only one of them is in the config key's name.

This is not a defect to fix — normalising the curriculum to the budget is a defensible design, and it is
what "set `total_lessons` higher" means in production. It is a **confound to declare**: any lesson-count
comparison answers "does a longer budget help", never "do more gradient steps help", and the two
questions have different answers if the curriculum is doing work.

**What is NOT coupled, checked at the same time so the scope is honest:** the LR schedule
(`warmup_decay_lr_scheduler.py:13`) is a fixed inverse-sqrt law stepped once per lesson
(`training_engine.py:1070`), so a longer run is a strict prefix-extension in learning rate. And BN
recalibration (`training_engine.py:778-801`) draws its 30 windows through the same `get_intensity_ratio`,
but those steps are roof-clipped at `max_ratio × roof_ratio` for every `total_lessons ≥ ~114` — so the
recal windows coincide across 160/300/600/900 and differ **only** against the 40-lesson rung.

Declared in `reports/2026-08-18_lesson_curve_dossier/05_analysis_plan.md` §2 and `SCOPE.md` §3.

---

### C-302: an opt-in training diagnostic is unreachable — its config key is dropped by schema validation

| Field | Value |
|-------|-------|
| ID | C-302 |
| Tier | 3 |
| Source | code read during `reports/2026-08-18_lesson_curve_dossier` pre-registration (2026-08-18) |
| Trigger | Setting `trajectory_log_path` in a config to diagnose a training trajectory, and concluding from the absent CSV that the run was fine |
| Location | `views_hydranet/train/training_engine.py:896-916`; `views_hydranet/utils/config_initializer.py:47,1166-1167` |
| Cross-refs | C-299 (the class: an instrument that cannot see is worse than no instrument) |

`training_engine.py:898` reads `config.get("trajectory_log_path")` and, when set, registers forward hooks
and writes a per-lesson CSV of grad-norm, losses and gate-logit mean — a genuinely useful convergence
instrument, and the only per-lesson trajectory record the codebase has (the training report keeps just
`final_loss` / `min_loss` / `max_loss`).

But `trajectory_log_path` is **not a field on `HydraNetConfig`**, and the resolved config is produced by

```python
config_obj = HydraNetConfig(**self._raw)   # :1166
return config_obj.model_dump()             # :1167
```

Pydantic's default `extra="ignore"` drops unknown keys, so the flag never reaches `training_engine`.
Setting it does nothing, silently — the failure mode is an absent file, which reads as "the diagnostic
ran and found nothing to report".

**Not fixed here.** The fix is one optional field on the schema plus a test, but it touches a shared
config surface for a diagnostic no current experiment needs; the lesson-curve programme was re-scoped to
read convergence from the rollout metrics and the three summary losses instead. Registered so the next
person to reach for the flag learns it is dead before they trust its silence.

---

### C-300: four shipped roster models were trained under a configuration the codebase now rejects

| Field | Value |
|-------|-------|
| ID | C-300 |
| Tier | 2 |
| Source | config/date audit while planning the SS successor experiment (2026-08-17) |
| Trigger | Citing `rescore.csv` rows for `blazing_meteor`, `bright_starship`, `pink_pirate` or `blue_stranger`; or reading the roster's scheduled-sampling/retention pattern as evidence about scheduled sampling |
| Location | `views-models/models/{blazing_meteor,bright_starship,pink_pirate,blue_stranger}/configs/`; `reports/2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv` |
| Cross-refs | C-259 (the rule they violate), C-299, views-models#404 |

All four scheduled-sampling-on roster models were trained **2026-08-12/13**. The C-259 validator coupling
`ss_feedback` to `rollout_feedback` landed **2026-08-14 04:19** (`c07a352`). `ss_feedback` defaults to
`"mean"` (`config_initializer.py:178`) and `training_engine.py:231` returns an **ungated** mean for any
mode other than `sample`.

So all four trained on an **ungated mean field while rolling out on a gated sample** — exactly the mismatch
C-259 was written to forbid. Their published rows are compromised twice: **un-rerunnable** (their configs
no longer load — views-models#404) *and* **produced under a rejected configuration**.

**Consequence for the research programme:** the roster's retention pattern (SS-off {0.54, 0.45} vs SS-on
{0.33, 0.21, 0.05, 0.02}) is **not** evidence that scheduled sampling hurts. It is evidence about the
mismatch. A sweep with `ss_feedback='sample'` — the only value the validator permits — tests a *different*
intervention and cannot settle it. The only SS data with a correct `ss_feedback` is `truncated_smoke`'s
sweep, which is floor-limited (C-299). **There is no valid, non-floored SS measurement anywhere.**

**Fix direction:** views-models#404 asks whether `ss_feedback: 'sample'` or `ss_epsilon_max: 0.0` is the
intended repair. This entry argues for the latter on the grounds that the *artifacts* were produced under
ε>0 with mean feedback, so setting ε=0 preserves the semantics the weights were trained under while
`ss_feedback='sample'` describes a model that was never trained.

---

### C-296: a diagnostic that consumes RNG differently from its control silently unpairs the comparison

| Field | Value |
|-------|-------|
| ID | C-296 |
| Tier | 2 |
| Source | `/code-review medium` on PR #278 (2026-08-16) |
| Trigger | Adding any inference-time diagnostic that replaces a sampling step, rather than only observing one — a different sampler, an extra draw, a skipped draw |
| Location | `views_hydranet/utils/hydranet_inference.py` (`_sample_feedback`, the correlated branch) |
| Cross-refs | C-290 (the arm this affected), C-113 (the original shared-generator coupling), C-289 |

`correlated_bernoulli` consumes a different number of variates from the shared generator than
`compose_samples`' Bernoulli. Every **later** step's body draw therefore came from a different stream in the
treatment arm than in the control, so the comparison mixed "coherent placement" with "different body noise" —
in an experiment whose entire content is a small difference between two arms.

This is the **third appearance of one defect class** in this codebase: C-113's shared `torch.Generator` across
the rollout, the fb_gen/transform separation this file's own comments document, and now this. The pattern is
that a *replacement* diagnostic is written as if it were an *observational* one.

**Fixed** by advancing the shared generator exactly as the control does (discarding the draw) and giving the
copula a third namespaced stream. **Registered rather than closed** because the rule generalises and is not
enforceable by a test on any single call site: *a diagnostic that replaces a sampling step must leave the
shared stream in the state the control would have left it.*

**Effect on the shipped result:** the EXP-05 null holds in direction and magnitude (clustering spans 100× while
AP stays ~0.007 against an oracle of 0.30 — RNG noise cannot cancel a 40x gap) but its two-significant-figure
comparison does not. Dossier and C-290 both amended.

---

### C-297: a sabotage check that selects no tests is indistinguishable from one that passes

| Field | Value |
|-------|-------|
| ID | C-297 |
| Tier | 3 |
| Source | authoring the PR #278 fixes (2026-08-16) |
| Trigger | Verifying a new guard by disabling it and running a filtered test selection (`pytest -k`, a single node id, a marker) |
| Location | process — the falsifier -> guard -> **sabotage** discipline used across `reports/*_dossier/03_harness_and_invariants.md` |
| Cross-refs | C-289 (a diagnostic that cannot fail), C-292 |

The sabotage check for the new splice guard reported **zero failures** and read as "the guard is redundant".
The `-k` filter had matched no tests: the name contained *splicing*, the filter said *splice*. Disabling a
guard and observing no failures is the exact signal the check exists to produce, so the tooling error and the
finding are the same observation.

The discipline this repo relies on — *a guard never seen to fail is not a guard* — has a blind spot: it
verifies the **outcome** of the sabotage run and not its **coverage**. Registered because every dossier's
harness section uses this method.

**Fix direction:** a sabotage check must assert on the number of tests **selected**, not only on the number
that failed. `pytest --collect-only -q -k <filter>` before the run, or drop `-k` and compare full-suite
failure counts.

---

### C-292: the recurrent-state arms cannot attribute damage to a memory half — `hs` is a readout of `hl`

| Field | Value |
|-------|-------|
| ID | C-292 |
| Tier | 2 |
| Source | `/code-review medium` on PR #277 (2026-08-16) |
| Trigger | Reading the `hidden` / `cell` / `all` arm ordering as evidence that one memory type carries the damage, or designing a soft prior on the cell state on that basis |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:527-529`; `reports/2026-08-15_state_freeze_dossier/07_experiment_log.md` EXP-02 |
| Cross-refs | C-222 (the mediator this bears on), C-289 (the same class: a diagnostic whose result is predetermined) |

The ConvLSTM computes `hl = f·hl + i·hl_tilde` and then **`hs = o ⊙ tanh(hl)`**. The hidden half is a
*readout* of the cell half. Pinning `hl` to the anchor therefore re-derives `hs` from the anchor every step,
so the `cell` arm structurally approximates the `all` arm **whatever the truth about where damage
accumulates**. The reverse does not hold — under `hidden`, `hl` integrates freely.

The published conclusion "it is the long-term memory, `cell` carries 89%" was withdrawn on this basis
(EXP-02 corrected in place). What survives is that the state path recovers ~23% of the oracle gap and that
freezing the short-term half alone recovers least.

**Fix direction:** an arm that holds `hl` *and* recomputes `hs` from the anchored `hl` (or the mirror)
separates them. No arm in the current set does. Until then, no claim about *which* memory carries the
damage is supported.

---

### C-293: rollout arms are scored only against the collapsed control — no naive baseline exists

| Field | Value |
|-------|-------|
| ID | C-293 |
| Tier | 2 |
| Source | `/code-review medium` on PR #277 (2026-08-16); independently raised by the Hyndman seat in the 2026-08-16 `expert-method-review` |
| Trigger | Reading any rollout arm's AP as *skill* rather than as *better than the collapsed arm* |
| Location | `reports/2026-08-15_state_freeze_dossier/tools/run_freeze_arms.py`; `reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py` — neither passes `--persistence` |
| Cross-refs | C-222, C-292, FAO-02 (which mandates an empirical baseline) |

Every rollout number in both dossiers is reported against the free-running control or the oracle, never
against persistence or a held-constant last-observed map. A totally frozen state makes the gate's
contribution constant — approximately a static risk map — and `all`'s AP is **flat at 0.069–0.091 from h6 to
h36**, exactly the signature of a static forecast scored against moving truth. Nothing in the design
separates "the state carried real information" from "a degenerate static map beats a collapsed gate".

`score_v2_horizons.py` has supported `--persistence` throughout and no driver has ever passed it. FAO-02
mandates an empirical baseline; the rollout programme has never had one. **Flagged twice — by an expert
method review and by a code review — before being acted on.**

**MEASURED 2026-08-16, and it fires.** Persistence (repeat the last observed map) scores gate AP
**0.112 / 0.108 / 0.083** at h6 / h18 / h36. Every state-freeze arm is **below** it from h6 onward — best
arm `all` reaches 0.091 at h18 against persistence's 0.108, and the free-running control (0.007) is **15×
worse** than the trivial baseline. So the ~23% oracle-gap recovery is real *relative to the collapsed
control* and still **does not reach a naive baseline**. Any rollout claim that does not clear persistence is
not a skill claim. (AP is a ranking statistic so the comparison is valid; note persistence is a 1-sample
forecast, so its CRPS is MAE and must not be used as a CRPS denominator — C-220.)

**CORRECTION 2026-08-21 (PR #283): "AP is a ranking statistic so the comparison is valid" is
INCOMPLETE, and the error runs against persistence.** `score_v2_horizons` forms the AP score as
`p = gate if has_gate else (cs > 0).mean(1)`. `_persistence_gathered` supplies **no gate**, so at S=1
persistence is ranked on a **two-level** score (`p ∈ {0.0, 1.0}`) while gated arms are ranked on a
continuous probability. AP cannot order within a tied set, so persistence is handicapped to the
maximum — the Epic #263 matched-reference rule (*the reference's S must equal the arms' cube width*)
applies to **AP's score resolution**, not only to CRPS. Ranking persistence by the persisted **value**
`truth[m0-1]` instead lifts h18 **0.1152 → 0.1416 (+23%)**; with the C-248/#282 month fix as well,
**0.1077 → 0.1416, +31%**. **Consequence for this entry: the measured numbers above understate
persistence, so C-293's conclusion was even stronger than recorded** — the arms were further below the
baseline than stated. **Consequence for M1:** same. Direction asserted over 200 random draws
(`test_binary_never_beats_value_on_a_random_sweep`). At L=300 with both fixes the model does finally
clear persistence at every horizon (ledger M34, n=4) — which is what this entry demanded all along.

---

### C-303: prose asserts a guard the code does not implement — TEN occurrences, one of which rendered a WRONG VERDICT

| Field | Value |
|-------|-------|
| ID | C-303 |
| Tier | 2 |
| Source | `/code-review medium` on PR #283 (2026-08-22) |
| Trigger | Writing a docstring, verdict string, or ledger row that describes what a rule/guard checks, without a test that fails when the described check is removed |
| Location | `reports/2026-08-21_persistence_reference_dossier/tools/aggregate_seeds.py` (docstring vs `main`); `reports/RESULTS_LEDGER.md` M37; `reports/2026-08-23_itf_pilot_dossier/07_experiment_log.md` (the "12 of 12 orderings" count, 8th occurrence); previously `scripts/lesson_curve_gate.py` and `scripts/ss_sweep_gate.py` verdict text |
| Cross-refs | C-148 (the "by construction" prose over-claim), C-291, C-298 |

`aggregate_seeds.py`'s docstring stated *"Refuses rather than averages when the supports differ.
Persistence identical across seeds is the evidence that the support is shared"* — and ledger row
**M37** repeated it as a method claim. The implemented check compared **only the per-horizon row
count `N`**, and `read_arm` explicitly *skipped* the persistence rows, so the stated evidence was
never read. Two seeds scored on different origin windows with equal `N` would have aggregated
silently, and the worst-seed verdict would have been computed across incomparable supports.

**This is the third instance of the same class in this programme, all within a week**, and all in
text a reader uses to decide: (1) the lesson-curve gate asserted "bound not narrower than theta" when
the real blocker was an unmeasured MDE; (2) the SS-sweep gate printed NULL-branch language for a
significant-but-undersized result; (3) this. **Tier 2 (structural fragility with no error signal):**
prose is the only interface most readers have to a rule, so a false description is a false result
that no test catches — none of the three were found by the suite, all three by a human or agent
re-reading the text against the code.

**Fixed on PR #283** — the persistence-equality check is now implemented (and sabotage-verified:
perturbing one seed's persistence by 0.01 makes it refuse by name), the docstring describes what the
code does, and M37 was rewritten. **Standing mitigation:** when a docstring or ledger row describes a
check, the check needs a test that fails when it is deleted. `tests/test_aggregate_seeds.py::test_refuses_when_persistence_differs_between_seeds`
and `::test_equal_N_does_not_excuse_different_persistence` are that test for this instance.

**FOURTH AND FIFTH INSTANCE, 2026-08-22 (PR #292) — and the fourth landed inside the document written
to prevent exactly this.** `05_analysis_plan.md` was added to the state-freeze-l300 dossier
specifically to stamp honest provenance on a partly-unregistered experiment. It asserted the falsifiers
were **"PRE-COMMITTED for all experiments, because they were enforced in code before any result was
read."** They were not: `tools/freeze_table.py` first enters git at `b18f177` (06:44) **in the same
commit as the score CSVs**, four hours after the overnight run finished (02:23). For EXP-01/02 the
falsifier enforcement is **retrospective**; only the M34 baseline *values* were written down beforehand,
as a comment, and the h1 invariant appears nowhere pre-run.

The fifth: the same §6 quoted h1 as **"0.47737082595880015 across all six arms"** as a falsifier for
*all* experiments. That is seed 43's value; seed 42 holds **0.4778833881292755**. The check itself is
correctly **per-seed** in code — so the *prose* would have made a reader auditing seed 42 conclude the
falsifier had failed when it had passed.

**What this escalates:** the class is no longer "a docstring drifted from its function". It now includes
**a provenance document overstating its own provenance**, which is the failure mode with the least
natural defence — the reader's only check on a provenance claim is the prose itself. Both were caught by
`/code-review medium` reading git timestamps against the text, not by any test.

**SIXTH AND SEVENTH INSTANCE, 2026-08-23 — and this pair is a *mechanical* variant worth separating.**
On `exp/gtf-sigma-max`, two fixes were **reported to the user as applied when they had not been**. Both
were `str.replace()` edits whose `old` text did not match (ruff had reflowed it), and **`str.replace`
returns the string unchanged rather than raising**. The report said "fixed"; the file was untouched.

* the `--stride`/provenance patch to `capture_states.py` — caught when the run failed with
  `AttributeError: 'Namespace' object has no attribute 'stride'`;
* the self-describing-JSON patch to `jacobian_sigma.py` — reported fixed **twice**, and both times the
  committed artifact still lacked `iters`/`n_states`/`artifact`. Caught only by grepping the source
  after claiming success.

**This is the same class as the rest of C-303 (a claim about code that the code does not support) but
with a mechanical cause and a mechanical fix:** every scripted edit must `assert t.count(old) == 1`
before replacing, and the result must be **verified in the file** — not inferred from the edit script
exiting cleanly. Where the edit changes an output artifact, regenerate the artifact and check it.

**Mitigation that would have caught it:** a provenance claim about *when* something existed is checkable
mechanically (`git log --diff-filter=A`). Any "pre-committed"/"pre-registered" assertion should cite the
commit that proves it, as §5 of that same document does correctly for the dial's decision table.

---

**Eighth occurrence (2026-08-23, itf-pilot `07`).** The experiment log was written asserting the
ordering `control > ITF > SS` held at *"h6, h18 and h36 on both seeds — 12 of 12 orderings"* **before
that count had been computed**. It was verified immediately afterwards and was correct, and the same
pass found a real exception the prose had not mentioned (`act_ratio` at h1 orders
`ITF < control < SS` on seed 43). **Correct-by-luck is the failure mode, not the exception** — this
is the identical shape as C-305, where a rule was overridden on grounds it did not contain and the
override turned out right by luck. The count is now stated with the h1 exception beside it.
**Rule reaffirmed: a specific number does not go into a write-up until the command that produces it
has been run.**

**Ninth occurrence (2026-08-24, truncated-nb EXP-01) — and the first to produce a WRONG VERDICT
rather than merely misleading prose.**

`05_analysis_plan.md` §5 registered: *"EFFECT: p ≤ 0.05 **and** mean ΔAP ≥ 3·MDE **and all four seeds
agree in sign**"*. The implementation computed the exact permutation p **one-sided for
`treatment > control`**. A strongly NEGATIVE effect therefore scores **p = 1.0** and can never reach
the EFFECT branch — and `verify_trunc.py` contained `state = "EFFECT" if mean_d > 0 else "EFFECT
(NEGATIVE)"`, a branch **I wrote and which was unreachable**, because the `p ≤ 0.05` guard above it
excludes `mean_d < 0`.

**The verdict it rendered was `NULL / UNDERPOWERED, p=1.0000` for a mean ΔAP of −0.2376 — an effect
7.5× the 3×MDE bar with all four seeds unanimous.** Had that been read and written up as published,
the programme's record would say *"removing the double-counted zero has no measurable effect"* when
the truth is that it collapses rollout AP by 70%. **The previous eight occurrences produced prose that
overstated a guard; this one produced a decision.**

**What generalises beyond the earlier instances:** a registered rule's *prose* can be two-directional
while its *implementation* is one-directional, and nothing in the harness compares them. The
tell was available before the data: **an unreachable branch in the verdict code**. Rule: **every
branch of a decision rule must be reachable by some input, and that should be demonstrated with a
unit test over synthetic verdict inputs — not left to whichever direction the experiment happens to
take.** The fix is recorded in that dossier's AMENDMENT 1, disclosed as post-hoc.

### C-308: a probe measured the wrong phase of the rollout, and every downstream guard still passed — TWO occurrences

| Field | Value |
|-------|-------|
| ID | C-308 |
| Tier | 2 |
| Source | `/code-review medium` on `exp/gtf-sigma-max` (2026-08-23), found independently while preparing the ritual |
| Trigger | Attaching a hook or probe to a model that is called in **more than one regime** — history digestion vs autoregression, seed vs rollout, warm-up vs steady state — without asserting which regime the captured samples came from |
| Location | `reports/2026-08-23_falsifier_checks/tools/capture_states.py`; `hydranet_inference.py:913` (`for t in range(origin + time_steps)`) |
| Cross-refs | C-303, C-305, C-306, C-307 (the "claim outran the measurement" family) |

`capture_states.py` hooked the model's forward and kept the **first six** calls. The rollout loop is
`for t in range(origin + time_steps)` with `origin = seq_len - 1` — **335 steps of history digestion on
real data, then 36 autoregressive steps.** The probe therefore sampled the **teacher-forced warm-up**
while the question (#294) was about the **free-running** regime.

**What makes this Tier 2 is that nothing looked wrong.** σ_max = 1.60 is a plausible value. The power
iteration converged. The registered convergence falsifier was satisfied. The write-up was internally
consistent. And the rising `max|h|` (0.000 → 2.867) read as a *satisfying confirmation* of cell-state
drift — it was published as *"the first direct observation of it"*. **On the real free-running phase the
state does the opposite: it collapses ~40×** (65.6 → 1.6). Direction, phase and interpretation were all
wrong, and **every downstream guard passed**.

**A plausible number from the wrong measurement is harder to catch than a wrong number.** No guard in
the chain can detect it, because each one is checking a property *of the measurement* rather than *of
what was measured*.

**Corrected values:** σ_max = **7.7628** on 8 states spanning the true autoregressive phase (calls
335–370), all converged to 0.00% drift — and σ is strongly state-dependent (3.4–7.8 early, ~1.47 late),
so a single scalar flattens a 5× swing.

**THE FIRST FIX DID NOT CLOSE THE FAILURE MODE — it moved the window without bounding it.** Adding
`--skip`/`--stride` put the captures in the autoregressive phase, but nothing stopped them running past
its end. The phase is calls 335–370 and the sample period is 371, so `--n-states 8 --stride 5` from 335
ends at exactly 370 — **it fits by luck**. `--n-states 10` would have captured 375 and 380, which are the
*next sample's* history digestion: the identical defect, silently. Caught by `/review-diff` on the
corrected branch. The window is now bounded by the **same `max|h| == 0` reset signal** that locates the
period, and stops **loudly and short** rather than mixing regimes.

**Standing rule adopted 2026-08-23.** Any probe attached to a model called in more than one regime must
**record the regime with each sample and assert it**, not infer it from call order. The phase boundary
here is discoverable without hard-coding a data property: the recurrent state is re-zeroed per posterior
sample, so `max|h| == 0` marks a boundary and the inter-boundary distance is `origin + time_steps`
(measured: 371 ⇒ origin = 335). `capture_states.py` now takes `--skip`/`--stride` and stamps the call
index and source artifact into every capture.

---

**Second occurrence (2026-08-23, state-range EXP-01) — the same defect, a different axis.**
`capture_regimes.py` replicated `predict()`'s origin resolution including its fallback,
`origin = seq_len - 1 = 383`. But that fallback applies only when **no** origin is passed; production
scoring rolls over `ctx.origins`, and the free-running phase actually begins at **335** (measured: sample
period 371, `time_steps` 36). The probe therefore built the state from **48 extra months of history** and
labelled it *"the state free-running inherits"*.

**It reported seed 43's `|R2|max` as 21.59. At the true origin it is 66.08** — against an independently
published **65.6**. **A 3× error in the headline quantity, and every falsifier still passed**: F1, F3, F4
and F5 were all green on the wrong-origin run, and its §4 verdict (IN-RANGE) was the same one the correct
run produced. **Nothing in the experiment could have caught it.** It surfaced only because the F2 vehicle
check went looking for the sample period and the number disagreed with a value published from other work.

**What generalises:** C-308's first instance was *"which phase"*; this is *"which origin"*. Both are
**a probe inheriting a default that is only correct when the caller supplies nothing**. The mitigation is
the same and now has two data points behind it: **a probe must assert the regime/index it sampled against
an independently sourced value, not against its own internal consistency** — every internal check agreed
with the wrong answer both times.

### C-307: a cheap screen's NO is recorded as a closure, with no false-negative mode and no reopen trigger

| Field | Value |
|-------|-------|
| ID | C-307 |
| Tier | 2 |
| Source | user observation, 2026-08-23, on the #290/#291/#294 falsifier checks |
| Trigger | Closing an investigative issue, or writing "the correspondence is superficial" / "the method is aimed at the wrong problem", on the strength of a **proxy** measurement rather than a trial of the thing itself |
| Location | `reports/2026-08-23_falsifier_checks/07_experiment_log.md` (all four checks); GitHub #290, #291, #294 |
| Cross-refs | C-303, C-305, C-306 (the same family: a claim stronger than what was measured) |

**The user's report of the pattern, which predates this session:** *"we have dropped multiple things on
the floor — and I keep telling you so — by doing quick smart tests to see if real implementation makes
sense, then dropping real implementation because the test told us to. Then me later insisting that we
try for real, and then it turns out that it in fact works. This happens so much."*

A cheap screen answers a **proxy** question. Its NO is evidence against the real thing only in
proportion to how tightly the proxy is coupled to it — and **that coupling has never been recorded**.
The write-ups state the verdict and the falsifiers on the *check*, but not the **false-negative mode of
the check itself**: the specific way the proxy could say no while the real method says yes.

**The instance that makes this Tier 2 rather than a style note.** #294 measured σ_max = 1.60, derived
GTF's α = 0.375, compared it to our measured w ≈ 0.10, and concluded *"correspondence superficial"*.
**The issue's own body had already listed why that comparison is invalid** — *"GTF re-anchors every
step; we anchor once … **these are not the same operator**"* — and the numeric comparison was made
anyway. Three independent reasons the screen can be a false negative there:

1. **σ_max was measured on the wrong model.** α is a *training* parameter and training under GTF
   *reshapes the Jacobian* — that is the method's entire mechanism. σ_max on a teacher-forced model is
   the σ_max of the model GTF would replace.
2. **The paper does not recommend a fixed α.** aGTF derives it per batch and anneals from α=1 downward,
   so "the paper predicts 0.375" is a static simplification the authors themselves supersede.
3. **The two weights parameterise different operators** (frozen anchor vs moving target), so there is no
   reason their optima should coincide even if the mechanism transfers.

**Tier 2 (structural, with a demonstrated history):** the failure is silent — a closed issue looks
settled, and the cost lands weeks later as work redone or a real effect never found. It compounds with
C-303/C-305/C-306, all of which are "the claim outran the measurement".

**Standing rule adopted 2026-08-23.** An investigative issue closed on a proxy must carry both:

* **the false-negative mode** — the specific way this screen could say no while the real method says
  yes; and
* **a reopen trigger** — a concrete condition that would make revisiting correct.

"CLOSED" then means *"screened out, here is what would bring it back"*, not *"settled"*. Applied
retroactively to #290, #291 and #294.

---

### C-305: a pre-registered decision rule was overridden post-hoc, then documented as "no branch matched"

| Field | Value |
|-------|-------|
| ID | C-305 |
| Tier | 2 |
| Source | `/code-review medium` on PR #292 (2026-08-22) |
| Trigger | Reading a pre-registered decision table's output and finding the fired branch inconvenient — the moment a post-hoc criterion (an MDE, a noise argument, a "within tolerance") is introduced to reach a different verdict |
| Location | `reports/2026-08-22_state_freeze_l300_dossier/05_analysis_plan.md` §5; ledger **M41** |
| Cross-refs | C-303 (the prose half of the same failure), C-298, C-291 |

The dial's decision table was **genuinely pre-registered** (`DIAL_PAUSED.md`, `da3156d` 10:58:07,
8.5 h before the first arm). Its branch 1 read: *`cell@0.5` > 0.3709 ⇒ a dial with an interior optimum*.
The measured value was **0.3715866** against a boundary of **0.3709158** — **branch 1 fired**, and its
prescribed follow-up ("sweep 0.25 and 0.75 next") is exactly what was run.

The verdict recorded was **"switch, not a dial"**, reached by introducing a **minimum-detectable-effect
argument that was not part of the registered rule**, and the write-up then stated *"a shape none of the
three branches predicted"* — building a process lesson on a premise contradicted by the data.

**Tier 2 (structural fragility with no error signal).** A registered rule that can be overridden by an
unregistered criterion provides none of the protection it was written for, and the override is invisible
because the write-up asserted no branch matched. Note the aggravating context: the *same document*
criticises EXP-01/02 for choosing a decision rule after seeing data.

**The override may well be correct** — 0.0007 is small — but that is a separate question from whether it
was licensed. Corrected on PR #292: §5 now states that branch 1 fired and was overridden, and the
override's premise is being *measured* (§5a) rather than asserted.

**Standing rule this earns:** when a registered branch fires and the outcome looks wrong, record *"the
rule returned X and I am overriding it because Y"*. Never re-describe the data so that no branch matched.

---

### C-306: an MDE from one contrast was used to declare a different, far more correlated contrast indistinguishable

| Field | Value |
|-------|-------|
| ID | C-306 |
| Tier | 2 |
| Source | `/code-review medium` on PR #292 (2026-08-22) |
| Trigger | Reusing a published MDE or CI half-width to judge a comparison **other than the one it was computed for** — especially between two arms that share more structure than the original pair did |
| Location | `reports/2026-08-22_state_freeze_l300_dossier/00_README.md`, `05_analysis_plan.md` §5a; ledger **M41**; `results/paired_ci.json` |
| Cross-refs | C-221 (block bootstrap over origins, never iid over cells), C-263/#263 (matched-reference rule), M40 |

**M41** declared the decay dial's interior points "indistinguishable" by comparing a 0.0022 spread
against a paired MDE of **0.0086**. That MDE is from `paired_ci.json` and is the interval for
**`cell` vs `none`** — two arms whose recurrent states diverge completely.

The contrast actually judged is **`cell@0.5` vs `cell`**, which differ only in a blend weight and are
therefore far more correlated. **M40's own argument** — that pairing cut the MDE 6.3× *because* the arms
share weights and origins — implies the correct interval for the interior contrast is **tighter still**,
possibly far below 0.0022.

**Tier 2:** the conclusion may be true but was **not established by the number cited**, and the error is
invisible because both quantities are called "the paired MDE at h18". It is the same family as the
**#263 matched-reference** rule: a yardstick is only valid for the comparison it was built for.

`scripts/ap_block_bootstrap.ap_diff_origin_block_ci` already computes the right interval; on PR #292 it
is being run on `cell@0.5` vs `cell` with the verdict **registered in advance** so the override in C-305
is not repeated.

---

**TENTH INSTANCE, 2026-08-26 (training-loop gradient audit) — the first one inside production
source rather than a report.** `HydraBNrecurrentUnet_06_LSTM4.py:104-106` explains the C-178 fix and
concludes: *"softplus is always positive with non-zero gradient, **so it cannot die**."* The first
clause is true; the conclusion is false, because `nb_core._clamp` applies `clamp_min(1e-6)` to the
activated `mu` and `theta` immediately downstream, and `clamp_min` passes **exactly zero** gradient
below its floor. Measured: `dNLL/d(raw_mu)` is `-6.389` at `raw_mu = -13.81` and `0.0` at `-13.82`.
The comment describes the activation in isolation and is read as a property of the path. Registered
in full as **C-313**, which also carries the pinning tests.

### C-304: dossier result directories accumulate state across runs — keyed by index, never cleared, silently mixable

| Field | Value |
|-------|-------|
| ID | C-304 |
| Tier | 3 |
| Source | `/code-review medium` + `/review-diff` on PR #283 (2026-08-22) |
| Trigger | Running a dossier driver a second time over the same `results/` dir with a different arm, seed, or origin count — especially after a partial emit that still exited 0 |
| Location | `reports/2026-08-21_persistence_reference_dossier/results/` (`identifiers/`, `run.log`, `repo.head`); the pattern generalises to every dossier `results/` tree |
| Cross-refs | C-159, C-291, C-298 |

Three instances in one dossier, each a different flavour of the same shape — **run-scoped state written
to a run-agnostic path**:

1. **`identifiers/` keyed by origin INDEX and shared across seeds.** A later seed emitting fewer
   origins leaves the earlier seed's extras in place, and the support becomes part seed A, part seed
   B. `support_from_identifiers`'s `n_origins == len(files)` guard **still passes** on such a mixed
   directory, so it cannot detect it.
2. **`run.log` is append-only across runs.** Grepping it for `ABORT` reports a failure fixed hours
   earlier — this misled the author mid-run during EXP-03. The scripts signal correctly through exit
   codes and per-seed `PERSIST_DONE_<model>` sentinels; the log is not a status source and reads like
   one.
3. **`repo.head` overwritten per seed**, so a multi-seed run records only the last seed's HEAD.

**Tier 3 (no silent corruption in the runs actually performed — support identity was independently
verified — but a realistic re-run mixes results with no error signal).** Fixed for (1) and (3) on PR
#283 (`rm -f identifiers/*.npz` before preserving; `repo.head` appends `HEAD  <model>`); (2) is
documented in `07_experiment_log.md` rather than changed, because append-only history is worth more
than a clean grep. **Standing rule for new dossier drivers:** clear or namespace any directory a
second run would write into, and never make a log the status source.

---

### C-294: the four ConvLSTM cells are identically configured and parallel — capacity, not structure

| Field | Value |
|-------|-------|
| ID | C-294 |
| Tier | 3 |
| Source | architecture read prompted by the PR #277 review (2026-08-16) |
| Trigger | Reasoning about "the four LSTMs" as if they were stacked layers or multi-scale cells — e.g. attributing a per-cell role, freezing one, or adding a fifth expecting a new scale |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:338-470` (the four `Wx*_n`/`Wh*_n` blocks), `:523-553` (the forward), `:556` (`torch.cat([x, hs_1..hs_4], 1)`) |
| Cross-refs | C-292 (the memory-half confound found the same way), C-295 |

All four cells take the **same** `input_channels`, `num_lstm_state_layers`, `kernel_size` and padding, all
consume the **same** `x`, and none consumes another's output. They are not stacked, not multi-scale, and not
differently parameterised — they differ **only by random initialisation**, and their hidden states are
concatenated onto the input before the U-Net.

That is close to one cell with 4× the channels minus the cross-mixing: extra capacity arranged as a
block-diagonal constraint, not extra structure. Nothing forces the four to specialise.

**Not a defect** — the LSTM math is textbook-correct (gates read the old hidden state, `o_t` is computed
before `h` is overwritten, the old cell is on the RHS) and these are genuine ConvLSTMs (`nn.Conv2d` gates).
Registered because the name `LSTM4` invites reading them as four *layers* or four *scales*, and a design
decision inherited by default should be made deliberately or simplified.

---

### C-295: memory is single-scale and local (3x3) while perception is multi-scale and memoryless

| Field | Value |
|-------|-------|
| ID | C-295 |
| Tier | 3 |
| Source | architecture read prompted by the PR #277 review (2026-08-16) |
| Trigger | Expecting the recurrent state to carry long-range spatial structure across rollout steps, or designing a fix that assumes it can |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:152` (`kernel_size = 3`, hardcoded), `:556` (LSTM output concatenated into the U-Net input), `:560-600` (encoder → bottleneck → decoders with skips) |
| Cross-refs | C-294, and the spatial-structure findings in `reports/2026-08-16_feedback_realism_dossier` |

The recurrence sits **in front of** the U-Net, not inside it: `x → 4 ConvLSTMs → concat → U-Net → heads`.
So **all** recurrence happens at full resolution through a **3×3** convolution — memory can propagate
information one cell outward per timestep — while all long-range spatial reasoning happens in the U-Net,
which is **memoryless** and re-derived from the current input every step.

The model therefore *perceives* at several scales but *remembers* at one, locally.

**Hypothesis, not a finding:** this is at least consistent with the measured failure — the gate's spatial
structure smears out under free-running (Moran's I 0.409 → 0.192 by h6 on target sb, against an oracle that holds 0.507 → 0.494), and the component that could carry
that structure across steps has a 3-cell horizon while the component that builds it has no memory at all.
**Untested.** Recorded so the idea is available and clearly labelled, not so it can be cited as evidence;
one architectural conclusion has already been withdrawn today for reasoning ahead of the evidence (C-292).

**UPDATE 2026-08-17 — the hypothesis is FALSIFIED; the description stands.** Tested against the gate-probe
data already in the dossier, before any GPU time was spent. Under the oracle the **same** architecture, the
**same** 3×3 recurrent kernel and the **same** 35 steps hold the gate's structure flat on target `sb`:
Moran's I **0.507 → 0.494 at step 6, 0.516 at step 35**. Free-running over identical steps falls **0.409 →
0.192**. A 3-cell memory horizon that can preserve spatial structure for 35 steps when fed a realistic field
cannot be what destroys it when fed the model's own.

The reasoning was also backwards on its own terms: repeated convolution is a **smoothing** operator, and
smoothing *raises* spatial autocorrelation — it is how `correlated_bernoulli` manufactures clustering from
white noise. A diffusion mechanism predicts Moran's I going **up**; the measurement shows it going **down**.
The hypothesis was inconsistent with the observation it was invented to explain.

**What survives:** the structural description — recurrence is single-scale, local, and in front of a
memoryless U-Net — is accurate and worth knowing. What is dead is the inference that this causes the gate to
smear. The smearing is driven by fed-back *content*, which is C-290's territory, not the architecture's.

**Consequence for planning:** "move recurrence inside the U-Net" is no longer motivated by this evidence. It
may still be a better architecture, but nothing measured here argues for it, and it must not be justified by
the smearing.

---

### C-288: seven tracked `reports/` tools still hardcode `/home/simon/...`; C-247's sweep fixed the tests, not the tools they load

| Field | Value |
|-------|-------|
| ID | C-288 |
| Tier | 3 |
| Source | PR #276 CI failure (2026-08-15) — tracking `score_v2_horizons.py` made it import in CI, which failed instantly on its absolute path |
| Trigger | Force-tracking any further `reports/**/tools/*.py`, or running one of the seven below on any machine but this one (CI, a colleague's clone, the server) — the import or the artifact read fails, or silently reads nothing |
| Location | `2026-07-17_densemse_mechanics_grid/tools/insample_probe.py:17,23,29`; `2026-07-25_t0_rollout_skill_dossier/tools/rollout_skill_score.py:32`, `s6_score_one.py:24,27`, `s7_verdict.py:20`; `2026-08-08_hydranet_ensemble_dossier/tools/s2_df_freshness_check.py:14` |
| Cross-refs | C-247 (same defect, resolved for the *test* files only), F4-07 / `test_P5a_no_tracked_test_hardcodes_absolute_machine_path` (the two guards that each cover half of this) |

C-247 is marked RESOLVED and its fix was real — but it covered `tests/test_score_v2_horizons.py` and its sibling, **not the dossier tool they load**. `score_v2_horizons.py` kept `_HN = "/home/simon/..."` and nothing caught it, because the file was untracked: CI never imported it, and `test_P5a_no_tracked_test_hardcodes_absolute_machine_path` only scans `tests/*.py`. Tracking it in PR #276 surfaced the defect in one CI run (`ModuleNotFoundError: No module named 'lodestar_score'`) and it was fixed there with the repo's own `Path(__file__).resolve().parents[3]` pattern.

**Seven tracked tools still carry the same hardcoded prefix.** None breaks CI today: no test imports them, and `rollout_skill_score.py:32`'s insert is a harmless no-op because whatever imports it has already put the lodestar directory on `sys.path`. That is luck, not design — `rollout_skill_score.py` is on the live import chain of both the v2 scoreboard and the Epic #263 ruler, so a change in import order turns it into a failure.

**Not fixed in PR #276, deliberately.** Four of the seven are frozen scorers that Epic #263's `SCOPE.md` explicitly ring-fences; sweeping them is a separate, reviewable change rather than collateral work on a PR about tracking. **Fix direction:** replace each with `Path(__file__).resolve().parents[3]`, then widen `test_P5a` (or F4-07) from `tests/*.py` to every tracked `.py`, which makes the guard match the concern.

---


### C-309: a confident NEGATIVE claim, made without a search — or from a search whose non-empty case was never seen

| Field | Value |
|-------|-------|
| ID | C-309 |
| Tier | 2 |
| Source | User escalation, 2026-08-23 — *"How did you not know this existed? If you don't know that this exists what else are you missing?"* |
| Trigger | About to write **"we have never tested X"**, **"X does not exist"**, **"there is no Y in this repo"**, **"nothing measures Z"**, or **"I can't find a rationale for W"** — in a message, an issue, a dossier, a ledger row or a commit — without having run a search **in that same message** |
| Location | Behavioural; instances: the `CurriculumLearner` exchange (2026-08-23); see narrative |
| Cross-refs | C-303 (prose outruns the code), C-305 (rule overridden on grounds it did not contain), C-306, C-307, C-308 — the "claim outran the measurement" family |

Asked whether a *data-sampling* curriculum had been tried, the answer given was **"We have not tested
that. It is a real, untested axis"** — with no search run first. `views_hydranet/utils/curriculum.py`
implements exactly that: `CurriculumLearner` decays an intensity **threshold** from `max_ratio=0.95` to
`min_ratio=0.05`, so training **starts on only the most active windows and opens up to sparse ones**.
It has its own CIC (`docs/CICs/CurriculumLearner.md`), its own active ADR
(`011_curriculum_and_training_topology.md`), and config fields in `HydraNetConfig` — **a file edited
earlier in the same session**, whose contract doc states the inverted-range rule on line 69. It runs in
every arm of every experiment in this repo.

A second negative claim in the same breath compounded it: *"somebody locked the direction in and I can't
find a measurement behind that"* — stated before opening ADR-011, which **does** give a rationale
("Signal Anchorage"). The accurate, narrower claim (found the stated goal, found no measurement
**comparing directions**) only became sayable after reading the file.

**Why Tier 2 rather than a style note.** A false negative about the codebase does not fail loudly. It
redirects work: it proposes rebuilding what exists, spends GPU-hours re-deriving a settled thing, or —
worse — frames an inherited, never-measured choice as *absent* rather than as *untested*, which are
different research findings with different next steps. **Failure to retrieve is indistinguishable from
absence from the inside**, which is precisely why the claim cannot be self-certified.

**The rule.** *No negative claim about this codebase without a search in the same message.* If the search
is not worth running, the claim is not worth making — downgrade it to a question ("does this exist?")
rather than an assertion. This is mechanical and checkable after the fact by rereading the message: a
negative assertion with no accompanying tool call is a violation regardless of whether it was true.

**Standing corollary (the asymmetry that makes this tractable):** claims backed by a shown command and
its output are a different reliability class from claims made from memory. **Positive, evidenced claims
are the reliable ones; unevidenced negative ones are the unreliable ones.** Treat that asymmetry as the
default calibration when reporting to the user, and say which class a statement is in when it matters.

**Second occurrence (2026-08-24) — and it defeats the first fix.** Surveying the paper library for
architecture work, I reported *"none of the 26 relevant papers has a single extracted claim"* and
built a recommendation on it (extract 4-5 by hand before designing anything). **All of them had
claims** — 70 across the 20 I later checked. The glob was `claims/<id>*.json` from the library root;
there is no `claims/` directory there. The real path is `papers/_claims/<id>.json`. Every lookup
returned empty, and I read empty as absent.

**This is the important part: I DID run a search, in the same message, as C-309 requires — and the
rule still failed.** A query whose **non-empty case has never been observed** carries no evidence at
all; an empty result from it is indistinguishable from a broken path. The mitigation as first written
is therefore insufficient.

**The tell was in the data and I walked past it:** a result of *exactly zero for all 26 papers*,
including four the library's own `status` had reported as added minutes earlier, is not a finding
about the library — it is a finding about the query. **A uniform-zero result across a heterogeneous
population should be treated as instrument failure until proven otherwise.**

**Amended rule:** a negative claim requires a search **plus a positive control** — the same query
shape must be shown returning a non-empty result for a case known to exist. If nothing in the
population returns anything, report the instrument as untrusted, not the population as empty.

### C-310: wall-clock estimates are made from too little evidence, and guards get sized from them

| Field | Value |
|-------|-------|
| ID | C-310 |
| Tier | 3 |
| Source | truncated-nb EXP-01 run notes (2026-08-24), plus the 2026-08-18 timeout losses |
| Trigger | Choosing a `timeout`, a finisher's wait ceiling, a disk-wait maximum, or telling the user how long a queue will take — from fewer than ~3 observations of the *same* configuration on the *same* box |
| Location | `reports/2026-08-24_truncated_nb_dossier/tools/finish_trunc.sh` (12 h ceiling); `reports/2026-08-18_lesson_curve_dossier/tools/run_lesson_arm.sh` (`TRAIN_TIMEOUT`, raised 36 → 72 → 150 s/lesson) |
| Cross-refs | C-163 (no runtime resource harness) |

Three instances, same shape. **(1)** `TRAIN_TIMEOUT` was sized at 36 s/lesson, then 72, then 150 —
**three arms were SIGKILLed by their own timeout, ~12 GPU-hours lost**, because there is no
mid-training checkpoint (`train_model.py` saves once, at the end) so a killed arm is lost entirely.
**(2)** During this session the user was told a seed would cost "~5 GPU-h"; the measured figure was
~2 h — a **2.5× overestimate** that would have discouraged a 4-seed design had it been believed.
**(3)** `finish_trunc.sh` was given a 12 h ceiling from a ~2 h/arm estimate; arms took ~2.4 h, the
ceiling expired mid-run, and the finisher **refused to assemble**.

**Instance (3) is the encouraging one and shows the correct shape:** the guard *failed safe*, writing
*"QUEUE_DONE never appeared — refusing to assemble a possibly stale verdict"* rather than emitting a
verdict built from three of four arms. **The estimate was wrong and nothing was corrupted.** That is
the property to preserve — the defect is the estimate, not the guard.

**Tier 3, not higher:** the failures are expensive and annoying rather than silently wrong, precisely
because the guards refuse instead of guessing. It is registered because the *estimate* keeps being
made from one or two observations under uncontrolled conditions (the same emit measured **6 min cool
and 24 min throttled**), and because a ceiling sized that way is the thing standing between a long run
and a stale result.

**Mitigation:** size a wait ceiling from the *measured* worst case × 2, not the expected case; state
the sample size behind any duration quoted to the user; and prefer a guard that refuses over one that
proceeds on a lapsed assumption.

### C-311: a reuse guard that validates configuration but not structure

| Field | Value |
|-------|-------|
| ID | C-311 |
| Tier | 3 |
| Source | arch-bakeoff EXP-01 run notes (2026-08-26) |
| Trigger | Writing or extending a guard that decides whether an existing artifact may be REUSED — checking its declared contents without checking that the artifact is structurally usable |
| Location | `reports/2026-08-18_lesson_curve_dossier/tools/run_queue.sh` (`ensure_arm` reuse branch); fixed in `scripts/arm_identity_check.py::missing_dirs` |
| Cross-refs | C-303, C-309 (guards that check less than a reader assumes) |

`run_queue.sh` reuses an arm directory when the label matches, gated on an identity check over config
keys. `dualfullzero_fortythree` had an **entirely correct** config — `DualStream`, seed 43, 300
lessons — and was accepted. It then failed **four seconds** later: `artifacts/`, `logs/`,
`notebooks/` and `reports/` were absent, so `ModelPathManager` raised at import.

**The config was never the problem, so a config check could never have caught it.** The failure was
cheap in compute and expensive in wall-clock: it consumed the last overnight slot of a 24-hour queue,
and the result was only available the following morning.

**What generalises.** A reuse guard answers *"is this the artifact I asked for?"* — but the question
that matters is *"is this artifact **usable**?"*. Identity and integrity are different properties and
checking the first reads as having checked both. Same family as C-303/C-309: the guard checks less
than its name implies, and nothing in the harness compares the two.

**Fixed** by `missing_dirs()`, which refuses an arm lacking any directory `ModelPathManager` requires,
with both directions tested. **Not fixed:** what emptied that directory is unknown — the config
timestamps say 02:08 and no rebuild appears in the queue log. Recorded as unexplained rather than
given a plausible cause.

> **RECURRED 2026-08-31, on four more arms — and the cause is still unknown.** All four
> `fullzero_{fortytwo,fortythree,fortyfour,fortyfive}` arms had lost `data/processed`, `notebooks`
> and `reports`. Those directories hold nothing but `.gitkeep`; `ModelPathManager` validates their
> existence and raises, so EXP-04's first launch died in seconds on all four seeds at once. The arms
> trained successfully in August, so the directories existed then.
>
> **This is now a pattern, not an incident**: it has struck an arm built by the bake-off builder and
> four arms built months apart by a different path. Nothing in this repo deletes them — the smoke
> harness removes only `data/generated/predictions_*`, and `make_*_arm.py` creates rather than
> removes. The likely culprit is something outside the repo that prunes empty directories (a backup,
> sync or cleanup tool on this machine), which would explain why only the `.gitkeep`-only ones go.
> **That is a hypothesis and has not been verified.**
>
> **Cheap mitigation not yet built:** `missing_dirs()` exists and is tested, but nothing calls it
> before an *emit*-only run — only the training queue's reuse path uses it. A one-line preflight in
> the emit drivers would have converted four silent failures into one clear refusal.

### C-312: the multi-task balancer can drive the loss negative, and two guards then stop training in silence — FIXED

| Field | Value |
|-------|-------|
| ID | C-312 |
| Tier | 2 |
| Source | training-loop gradient audit (2026-08-26) |
| Trigger | Writing a new arm config without `freeze_multitask_balancer: True`, or deliberately re-enabling the C-111 active balancer to revisit the C-113 bisect |
| Location | `views_hydranet/utils/mtloss.py:66`; `views_hydranet/train/training_engine.py:1015` (`if w_loss > 0`) and `:1027` (`if lesson_loss > 0`) |
| Cross-refs | C-111 (added `log_vars` to the optimizer), C-113 (the bisect the freeze flag exists for), **C-170 and C-124 — the SAME trigger (un-freezing the balancer), different mechanism**: C-170 is about the Kendall weighting *starving* the rare targets, this is about the guards below it *stopping training altogether*. Anyone acting on C-170's mitigation must read this too.**, C-314 |

`MultiTaskLoss.forward` returns `coeffs * losses + torch.log(stds + eps)`. The second term is
**negative whenever `log_var < 0`**. Per task, `term(L, v) = L/((r+1)e^v) + v/2` is minimised at
`v* = log(L/(r+1))`, giving `term = 1/2 + log(L/(r+1))/2` — **negative once `L < (r+1)/e`**. That is
not an exotic regime; it is what a well-fitted task looks like, and `v*` is exactly where gradient
descent on `log_vars` is heading.

ADR-014's guards read as "skip empty windows", but they key on the **sign of a quantity the balancer
can make negative**. When it does, `backward()` is skipped at `:1015` and `optimizer.step()` at
`:1027` — with no warning, no log record above DEBUG and no counter. The progress bar advances, the
loss curves are written, wandb receives rows, and the run is externally indistinguishable from a
healthy one.

**Measured** on the `loop_config` vehicle with `log_vars` placed at their own fixed point: **2
optimizer updates across 6 lessons**, versus 6 of 6 for the frozen control, same seed and data.

**Not currently exposed.** Every live arm config — the incumbents and all six architecture bake-off
arms — sets `freeze_multitask_balancer: True`, which pins `log_vars` at 0 so `log(stds) = 0` and
every term is non-negative. No past result is in question. But the **config default is `False`**
(`config_initializer.py:274`), so the exposure is one omitted line away, and 114 of the 156 model
configs in views-models do not mention the flag at all.

> **FIXED 2026-08-26.** The guards now ask ADR-014's actual question. A window backpropagates
> unless its loss is **exactly zero** (the empty-window case ADR-014 means); a **non-finite** loss
> now **raises** instead of being skipped like an empty window (a NaN used to fail `> 0` silently —
> C-212 was exactly such a bug); and the optimization gate keys on `windows_trained > 0`, i.e.
> whether gradient was actually produced. Measured after the fix: **6 of 6** lessons update, where
> the old guards gave 2.
>
> **The C-112 comparability question is answered by measurement, not assertion.** Under
> `freeze_multitask_balancer: True` the balancer contributes `log(stds) = 0` and every task term is
> non-negative, so the old `> 0` and new `!= 0` select the same windows.
> `test_the_new_guard_is_byte_identical_when_frozen` proves it by observing every window's total
> during a real frozen-balancer run rather than re-deriving the algebra, so it stays true if the
> loss composition changes. **No existing result moves.**
>
> Pinned by `tests/train/test_backward_gate.py` (10 tests). Mutation-verified: restoring the sign
> guard, removing the empty-window skip, removing the optimization gate, removing the non-finite
> check, and removing the `log(stds)` term are all caught.

---

### C-313: the NB clamp reinstates the C-178 dead-gradient trap that the architecture comment says is gone

| Field | Value |
|-------|-------|
| ID | C-313 |
| Tier | 3 |
| Source | training-loop gradient audit (2026-08-26) |
| Trigger | Raising `_EPS`, changing `reg_activation` or the head's bias init, or training long/hard enough that the reg head's raw output approaches -13.8 |
| Location | `views_hydranet/distributions/nb_core.py:20-24`; false claim at `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:104-106` |
| Cross-refs | **C-202 — same clamp, examined for `theta` ONLY and resolved on a theta-specific argument; `mu` was never in scope**; C-178 (the original dead-ReLU root cause), C-199 / C-203 (theta dies if it saturates), C-314 (the clip is the belt-and-braces C-202's resolution leans on), C-303 (tenth occurrence) |

`_clamp` applies `clamp_min(_EPS)` with `_EPS = 1e-6` to both activated NB parameters. `clamp_min`
passes zero gradient below its floor, so with a `softplus` head the dead-zone edge sits at exactly
`softplus(raw) == 1e-6`, i.e. `raw == log(expm1(1e-6)) ≈ -13.8155`.

The transition is **abrupt, not gradual**, and therefore a trap rather than a soft floor: the
gradient is finite immediately above the edge and **exactly `0.0`** immediately below, in both
directions and for any target. A unit that steps past the edge cannot climb back out. This is
mechanically the failure C-178 was opened for and fixed by ReLU→softplus; the clamp reinstates it
13.8 nats lower down.

**Relationship to C-202 — the real gap.** C-202 asked whether this same `_EPS` floor causes a
head-channel gradient explosion, and answered *no*, correctly: `dtheta/d(raw) = sigmoid(raw) -> 0`
cancels the `1/theta` term. But that argument, its test and its Location field are all about
**`theta`**. The identical clamp applies to **`mu`**, where the cancellation does not occur —
measured, with targets in log1p space as the loss expects, `|dNLL/d(raw_mu)|` just above the floor
is approximately **the observed count** (5.0 at count 5, 5000.0 at count 5000), against C-202's
`1.5` theta bound. C-202 is not wrong; it is **incomplete**, and its resolution note reads as
though the clamp question is settled for the head as a whole.

**How remote, measured rather than asserted.** Severity depends strongly on `theta`. The -6.389
figure quoted in early notes on this entry is at `theta = 0.693`; at the **incumbent's** measured
operating point (`raw_mu = -3.81`, `raw_theta = -9.99`) the mu-channel gradient is `-0.207` at
count 100 and `-2.07` at count 1000 — and its **sign pushes `mu` away from the cliff**, since any
positive count raises `raw_mu`. The gap to the edge is 10.0 nats, i.e. ~48,000 unclipped
`lr=1e-3` steps *in the wrong direction*. Tier 3 is if anything generous.

**Latent, not active — measured, not assumed.** Running the trained L=300 incumbent
(`fullzero_fortytwo`) forward on a sparse field, the reg heads' minimum raw outputs are `mu` **-3.81**
and `theta` **-9.99**, against an edge of -13.82. No past result is affected.

**Second finding, same probe:** `tests/distributions/test_theta_gradient_bound.py` asserts
`|d/d raw_theta| <= 1.5` at `raw_theta = -16.0`. That point is **inside the dead zone**, so the bound
is satisfied by a gradient of exactly `0.0` — by the channel being dead, not by softplus cancelling
an explosion as the docstring claims. Its `-10` and `-13` rows are above the edge and do demonstrate
the cancellation, so the test is sound apart from that one vacuous row; the row is documented rather
than deleted, because deleting it would erase the evidence that the trap exists.

**Not fixed** (C-112). Pinned by `tests/distributions/test_gradient_dead_zones.py`, 6 tests deriving
the edge from `_EPS` rather than hardcoding it. Mutation-verified: raising `_EPS` to 1e-3, or
removing the clamp, fails.

---

### C-314: the gradient clip is unconfigurable, was untested, and does not cover the balancer

| Field | Value |
|-------|-------|
| ID | C-314 |
| Tier | 3 |
| Source | training-loop gradient audit (2026-08-26) |
| Trigger | Tuning the clip threshold from config, or diagnosing an explosion by reading `max_raw_grad_norm` |
| Location | `views_hydranet/train/training_engine.py:1080-1081` (clip), `:1053-1060` (raw-norm audit); `config_initializer.py` (`clip_grad_norm: bool`) |
| Cross-refs | C-312 (the same parameters, unbounded in the loss), **C-202 — its resolution explicitly leans on this clip as the belt-and-braces for legitimate large-count gradients, so an untested clip weakened a closed entry**, C-215 (per-lesson grad-norm not persisted), C-184 |

Three distinct gaps in one guard:

1. **`max_norm=1.0` is a literal.** The `clip_grad_norm` config field is a **bool**, so the only
   thing a config can say is on/off. Changing the threshold requires editing the engine.
2. **It had zero behavioural coverage.** Deleting the call passed all 1630 tests. Measured on the
   audit vehicle: raw norm **5.678 / 5.614**, clipped **exactly 1.0000** — a large, binding
   intervention that nothing verified was happening.
3. **It covers `model.parameters()` only.** `MultiTaskLoss.log_vars` is a separate optimizer param
   group (C-111) and is therefore invisible to both the clip *and* the `max_raw_grad_norm` explosion
   audit above it. Since `coeffs = 1 / ((is_regression + 1) * stds**2 + eps)` has no upper bound, an
   unclipped `log_vars` is an unbounded amplifier on every task loss, unwatched by the guard whose
   job is to notice exactly that.

> **(1) and (2) FIXED 2026-08-26; (3) REMAINS OPEN.**
>
> (1) `clip_grad_max_norm: float = Field(default=1.0, gt=0.0)` replaces the literal. The default
> keeps every pre-existing config byte-identical (`test_the_clip_default_is_unchanged_at_one`).
> (2) `tests/train/test_gradient_health.py` observes the gradient norm the optimizer is actually
> handed — a test that patched `clip_grad_norm_` would have verified we call a function, not that
> the gradient is bounded. Mutation-verified: deleting the call, raising `max_norm` to 100.0, and
> ignoring the new config field are all caught.
>
> **(3) is deliberately NOT fixed:** clipping `log_vars` changes training dynamics whenever the
> balancer is active, which is a C-112 decision, not an audit side effect. The engine now carries
> a comment saying so at the clip site. **This entry stays open on (3) alone.**

---

### C-315: the training graph is untruncated over the full time axis, and only explosion is monitored

| Field | Value |
|-------|-------|
| ID | C-315 |
| Tier | 3 |
| Source | training-loop gradient audit (2026-08-26) |
| Trigger | Adding any per-step auxiliary term to `_process_sequence` (pushforward #289, GTF #294, BPTT-SA #288), or raising `window_dim` / the training time axis |
| Location | `views_hydranet/train/training_engine.py:350` (state rebound un-detached), `:1016` (the single `backward`), `:1053-1060` (explosion-only audit); `views_hydranet/utils/volume_sampler.py:88` |
| Cross-refs | C-07 (per-window memory release), C-246 (the T-loop implemented twice) |

There is **no BPTT truncation of any kind**. `h` is rebound to `output.h_next` with no `detach`,
`total_loss` accumulates every step, and there is exactly one `.backward()` in the package, called
once per *window* over all `seq_len - 1` steps. `VolumeSampler._generate_window` slices space only,
so `seq_len` is the full training time axis — ~383 steps in a production run. Every window therefore
holds a ~383-step autograd graph through 4 ConvLSTMs, a U-Net and six decoder branches until
backward. That is consistent with the 4.1–4.7 GiB the bake-off smoke measured.

This is a deliberate design and, on the evidence below, a *justified* one — but it was undocumented,
untested, and it bounds what can be added to the step loop.

**Measured cost of adding one** (`#289` pushforward, one production-shaped window — `window_dim=32`,
`T=336`, RTX 4070):

| | peak allocated | fwd | fwd+bwd |
|---|---|---|---|
| `pushforward_weight=0` | 2524 MiB | 3.22 s | 5.83 s |
| `=1`, state attached | 3990 MiB | 6.01 s | 10.24 s |
| `=1`, state detached | 3990 MiB | 5.97 s | 10.02 s |

So **+58% memory and ×1.76 wall-clock**, not the doubling a first reading of the graph suggests —
the extra step is one level deep from a detached field, not a second full sequence. Extrapolating
from the incumbent's measured 1.82 h/arm gives **~3.2 h/arm**. `pushforward_detach_state` costs
nothing either way, so that fork is free to test. (Re-measured after the BatchNorm-freeze fix,
which adds a per-step module walk; the earlier figures were ×1.65 / ~3.0 h.)

**The reach is real, and this corrects a prior.** Measuring `d||h_final||²/dx_i` at T=120 — which
isolates the recurrent path, since a frame can only reach `h_final` through memory:

| steps back | random init | trained L=300 |
|---|---|---|
| 0 | 1.14e+01 | 5.56e+01 |
| 20 | 1.40e-03 | 3.94e+00 |
| 60 | 2.98e-09 | 2.65e-01 |
| 118 | 2.81e-17 | 1.56e-02 |

The untrained recurrence is sharply contractive and geometric — `MillerHardt2019`'s stable regime,
where a truncated feed-forward model approximates it and the memory is decorative. **Training escapes
that regime**: the trained decay is sub-exponential and retains ~1e14× more gradient at 118 steps
back. So truncating BPTT would discard real signal, and **M46's `WideMemory` null is not a
vanishing-gradient story** — whatever caps that arm, it is not gradient failing to reach the
recurrence.

**Residual gap:** `max_raw_grad_norm` watches only for explosion. Nothing in the run measures
vanishing, so a future change that collapses the recurrence back toward the random-init regime would
be silent. Pinned by `tests/train/test_bptt_reach.py`; mutation-verified (inserting `.detach()` in
the state carry fails).

### C-316: a test deleted a live module from `sys.modules`, silently disarming every later monkeypatch of it

| Field | Value |
|-------|-------|
| ID | C-316 |
| Tier | 3 |
| Source | training-loop gradient audit (2026-08-26), found while three new tests failed only in full-suite runs |
| Trigger | Writing a test that reloads or removes a module from `sys.modules` in order to re-read it, when other modules already hold bound references into it |
| Location | `tests/test_reproducibility_gate.py::test_training_engine_uses_lock_entropy`; guard added as `::test_the_live_training_engine_module_is_the_one_training_loop_uses` |
| Cross-refs | C-303 / C-309 / C-311 (guards that check less than a reader assumes) |

`test_training_engine_uses_lock_entropy` did `del sys.modules["views_hydranet.train.training_engine"]`
and re-imported, purely so it could read the module's `__file__` and grep the source text. The
delete was never needed for that, and it was never undone: for the rest of the session
`views_hydranet.train.training_engine` was a **new** module object, while
`train_model.training_loop` — already bound — kept executing against the **old** module's globals.

**The consequence is exactly the failure mode this register keeps recording:** any later test that
monkeypatched a `training_engine` attribute was patching an object nothing called. The patch
appeared to succeed, the test ran, and it asserted against unpatched behaviour. Three tests in
`tests/train/test_backward_gate.py` passed in isolation and failed **only** in a full-suite run,
which is how it surfaced; a test written to be *less* strict would simply have passed, wrongly, in
both.

**Fixed** by reading the source through `Path(mod.__file__).read_text()` without touching
`sys.modules`. **Guarded** by an assertion that `training_loop.__globals__` *is* the `__dict__` of
the module currently in `sys.modules` — added because re-introducing the delete, once the three
tests had been hardened to patch `training_loop.__globals__` directly, **survived the entire
suite**. Hardening the victims removed the only signal that the defect existed; the guard restores
it, and the mutation is now caught.

### C-317: retraining does not reproduce — run-to-run scatter is the size of seed-to-seed scatter

| Field | Value |
|-------|-------|
| ID | C-317 |
| Tier | 2 |
| Source | pushforward dossier, F1 control-reuse gate (2026-08-30) |
| Trigger | Claiming a result is confirmed because a model was retrained and "got the same numbers", or setting any tolerance for comparing two trained artifacts |
| Location | `views_hydranet/infrastructure/reproducibility_gate.py` (`lock_entropy`); measured in `reports/2026-08-26_pushforward_dossier/05_analysis_plan.md` Amendment 1 |
| Cross-refs | C-42 (the repro gate, marked done), M34 (whose reproduction claim was about RE-EMITS, not retrains), C-112 (pre/post comparability) |

`lock_entropy` sets `torch.use_deterministic_algorithms(True, **warn_only=True**)`,
`cudnn.deterministic = True`, `cudnn.benchmark = False` and seeds every RNG. The programme has
treated that as making training reproducible. **It does not, and nobody had measured it.**

Retraining `fullzero_fortytwo` — same seed, same data, same GPU (RTX 4070), same Python 3.11.14,
same torch 2.6.0 — reproduces its archived twin to **148% / 114% / 94% / 78% of the seed-to-seed
sd** at h1 / h12 / h30 / h36. **Retraining the same seed moves AP about as much as changing the
seed.** `warn_only=True` is the likely mechanism: an op without a deterministic kernel falls back
silently.

**What this invalidates.** Any claim of the form *"we retrained and got the same result, therefore
X"*. M34's reproduction evidence is unaffected — it re-**emitted** from fixed `.pt` artifacts,
which is deterministic — but the distinction was never drawn, and this dossier's F1 inherited M34's
`5e-4` tolerance and applied it to a retrain, where no scalar tolerance can work.

**What it does NOT invalidate.** Multi-seed designs whose noise model is the observed spread across
runs: whether that spread comes from seeds or from nondeterminism, the permutation test uses it
either way. The pushforward 4v4 contrast stands for this reason.

**Not fixed.** Making training bit-reproducible means `warn_only=False`, which would raise on the
first non-deterministic op and may be unachievable with this architecture; and it would change
training dynamics (C-112). Registered so the *claim* stops being made, which is the actual defect.
**Open question worth its own probe:** which op is falling back — the run log shows no determinism
warning at all, which is itself unexplained.

### C-318: a sentinel value averaged as a measurement, into a published ledger row

| Field | Value |
|-------|-------|
| ID | C-318 |
| Tier | 2 |
| Source | `/expert-code-review` commissioned to design the next experiment (2026-09-02) |
| Trigger | Aggregating any column that encodes "undefined" as an in-band numeric value — in this repo `mean_magnitude_on_active`, `persistence` and `neighbour_pairs_per_active` all use `-1.0` |
| Location | `views_hydranet/utils/hydranet_inference.py:521-535`; consumed wrongly in `reports/2026-09-01_fade_mechanism_dossier/07_experiment_log.md` and `reports/RESULTS_LEDGER.md` (M50) |
| Cross-refs | C-303 (prose asserting what the code does not do), C-308 (a plausible number from the wrong measurement) |

`_record_feedback_stats` writes `-1.0` for `mean_magnitude_on_active` when `n_active == 0`. M50
averaged the column unfiltered and published *"magnitude 18.4 → −0.8"* as evidence that the fed
field collapses on magnitude as well as occurrence. By step 30, **97% of the control's records are
the sentinel**. Filtered, magnitude is **18.4 → 13.5 unclamped and 18.4 → 13.8 clamped** — the two
arms are the same, and **magnitude does not collapse at all**.

**Three separate signals were available and all were missed:**

1. **The column's own code comment warns against exactly this** — *"averaging the column would then
   mix empty fields with scattered ones — biasing the statistic downward exactly in the collapse
   regime this study is about."*
2. **The value is arithmetically impossible.** `counts = expm1(field).clamp(min=0.0)`, so a mean of
   clamped non-negatives cannot be negative.
3. **It was noticed and filed as a mystery.** M50's write-up records it under *"an anomaly,
   recorded not explained"* — the anomaly was the diagnosis, and writing it down was mistaken for
   handling it.

Worse than a wrong number, the published table compared the clamped arm's **15.71** (a real mean
over 153/156 records) against the control's **−0.74** (97% sentinel) as if they were one quantity.

**The finding survives.** `active_fraction` is `n_active / n_cells` and cannot be a sentinel, so the
1,547×-versus-2× occurrence result is untouched, as are the state numbers from a separate capture
path. **Corrected**: the collapse is purely occurrence, and the clamp preserves *where* the model
fires, not *how loudly* — which sits better with M32/M45 (placement is everything; magnitude is not).

**Standing rule:** an in-band sentinel must be filtered at the point of aggregation, and any
analysis touching these three columns asserts `n_active > 0` before reducing. **Noticing an
impossible value and recording it as unexplained is not handling it** — that is the specific
behaviour this entry exists to stop.

### C-319: a field statistic that cannot see placement, used to explain a placement effect

| Field | Value |
|-------|-------|
| ID | C-319 |
| Tier | 2 |
| Source | EXP-3/EXP-3b, silence-vs-fade dossier (2026-09-03) |
| Trigger | Explaining *why* an intervention changes AP/CRPS using any statistic computed from the emitted field alone — occurrence, body magnitude, gate–body alignment, `act_ratio`, `size_ratio` |
| Location | `reports/2026-09-02_silence_vs_fade_dossier/tools/decompose.py`; `reports/GLOSSARY.md` §4; consumed wrongly in `RESULTS_LEDGER.md` (M52) |
| Cross-refs | C-318 (a statistic read past its support), C-303 (prose asserting what the code does not do), C-126 (rollout sharpness conflation) |

Rolling the clamp anchor 90 cells left **every** field statistic essentially unchanged — occurrence
0.718 vs the clamp's 0.715, plain magnitude 0.628 vs 0.651, alignment 61.5 vs 69.3 — while `AP@h18`
fell **0.362 → 0.0075**, a 48× collapse. EXP-3b then measured *why*: the forecast is the clamp's
forecast **displaced**, matching at r ≈ 0.90 with the cross-correlation peak at exactly (90, 90).

These statistics are **marginal and internal**. They describe the shape of the emitted field and
whether the gate and body agree *with each other*; none references truth, so none can distinguish a
good forecast from the same forecast in the wrong place. M52 used alignment to explain the clamp's
AP benefit, which is invalid on its own terms — the rolled arms preserve alignment just as well and
score 1/48th. **Any causal story about a truth-referenced score must be closed by a truth-referenced
statistic.**

### C-320: pre-registered falsifiers written without checking what the statistic can deliver (THREE instances)

| Field | Value |
|-------|-------|
| ID | C-320 |
| Tier | 3 |
| Source | EXP-1 G1 check, silence-vs-fade dossier (2026-09-02) |
| Trigger | Writing an agreement threshold (`within X%`) between two instruments without first bounding the *reference's* sampling error in the regime where the comparison will be made |
| Location | `reports/2026-09-02_silence_vs_fade_dossier/05_analysis_plan.md` (F3); `tools/exp1_g1.py` |
| Cross-refs | C-319 (the sibling: a falsifier built on an untested assumption about a statistic), C-307 (cheap screens recorded as closures) |

F3 required two instruments to agree within 10% at every horizon. It fired — and then fired on the
**known-good control** too (2 of 36 horizons, 13.1% and 10.7%). The reference is a 16-sample
estimator of a heavy-tailed quantity whose nonzero-draw count collapses 599 → 13 → 1 → **0** across
h1/h18/h30/h36, so at late horizons no instrument can agree with it to 10% regardless of
correctness. The gate was **unsatisfiable by a correct instrument**. The statistically correct form —
agreement within the *reference's own* uncertainty — was available at drafting and was not written;
the dump passes it 4/4. Recorded fired rather than amended, and the chair authorised proceeding on
the record.

**GENERALISED 2026-09-03 — this is a class, not an incident. Three instances in eight days:**

| # | gate | defect | found by |
|---|---|---|---|
| 1 | **F3** (silence-vs-fade) | a 10% agreement band tighter than the reference's own sampling noise — it fired on the known-good control | running the control |
| 2 | **FR-4** (EXP-3) | required alignment to fall when placement was wrong, assuming alignment measures placement correctness — which the same experiment disproved | the experiment itself |
| 3 | **C.4** (drivers) | "the gain on new cells is at least as large" without naming the measure; absolute gain and skill-above-base favour continuation, relative gain favours onset | computing all three |

The common cause is not carelessness about thresholds — all three were written deliberately. It is
that **a falsifier encodes an assumption about what its statistic can deliver, and that assumption
is never itself tested**. Each was caught only by measurement after the fact, and each was recorded
fired/ambiguous rather than amended, which is the only reason the record is still trustworthy.

**Mitigation to apply at the next pre-registration, not retroactively:** for every numeric gate,
state (a) the measure, explicitly, when more than one is available, and (b) what the reference
instrument's own noise floor is in the regime where the gate will be evaluated. A gate that cannot
be passed by a correct instrument, or whose verdict flips with an unstated measure choice, is not a
falsifier — it is a coin toss with a citation.

### C-321: `--keep-cubes` silently disables the multi-arm contamination guard

| Field | Value |
|-------|-------|
| ID | C-321 |
| Tier | 2 |
| Source | EXP-1 batch run (2026-09-02) |
| Trigger | Passing `--keep-cubes` to `run_realism_arms.py` with more than one arm in `--arms` |
| Location | `reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py` (the `if before and not keep_cubes` guard) |
| Cross-refs | C-154 (disk), C-318 (a complete-looking artifact that is not what it claims) |

Every arm writes the **same** prediction path, named after the artifact rather than the run. The
start-of-arm guard refuses when a prediction directory already exists — but it is skipped entirely
when `--keep-cubes` is set. So a two-arm batch with `--keep-cubes` runs arm 1, keeps its cubes, and
lets arm 2 **overwrite them**; verified by file mtimes (arm 1 finished 22:13, its `origin_0` files
were stamped 22:15 by arm 2). The corruption surfaced only at the *scoring* step, as "expected
exactly one new prediction dir, found 0" — late, after ~35 minutes of GPU, and only because a second
guard happened to catch it. The flag that disables the guard is documented as "debug only; skips the
disk guard", which understates what it skips.

### C-322: grid orientation is flipped between priogrid placement and the model field

| Field | Value |
|-------|-------|
| ID | C-322 |
| Tier | 2 |
| Source | EXP-1 G1 check (2026-09-02) |
| Trigger | Comparing any grid built by `sharpness_scorecard.to_grid` / `build_unit_grid` against a model-native `[H, W]` field — a dumped state, a body-mean field, an attention map |
| Location | `scripts/sharpness_scorecard.py::to_grid` / `build_unit_grid`; `reports/2026-09-02_silence_vs_fade_dossier/tools/exp1_g1.py` |
| Cross-refs | C-136 (grid reconstruction uniqueness), C-319 (spatial statistics read off the wrong cells) |

The model field's `H` axis runs **opposite** to priogrid row order. Placing study cells at the
naive `(row − 87, col − 310)` correlates **0.026** against the model's own gate cube; with
`(179 − row, col)` the correlation is **1.0000** and the max difference is exactly **0**. A global
flip **cancels** inside FSS and Moran's I because both grids are built the same way, which is why
`to_grid` has never needed to care and why this was invisible until now. It does **not** cancel the
moment a grid is compared against a model-native field, and the failure is silent: every downstream
number is well-formed, plausible, and computed on the wrong cells.
### C-323: an ablation that perturbs a component the architecture REGENERATES measures nothing

| Field | Value |
|-------|-------|
| ID | C-323 |
| Tier | 2 |
| Source | architecture read prompted by the chair, 2026-09-03 |
| Trigger | Designing any ablation, freeze, roll, dropout or lesion arm on a recurrent component — before running it, ask whether the component is *carried forward* or *recomputed* each step |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:568-603`; consumed wrongly in `RESULTS_LEDGER.md` (M56 hidden arm, M60 hidden roll) |
| Cross-refs | C-292 (`hs` is a readout of `hl` — this makes it concrete), C-319 (a statistic that cannot see what the claim is about), C-320 (a gate whose statistic cannot deliver the verdict) |

`hs = o_t * tanh(hl)`: the hidden half is **recomputed from the cell every step**. So any perturbation
of `hs` — a roll, a freeze, a lesion — is regenerated from the *undisturbed* cell on the next step and
**self-heals**. Wave 2's hidden-roll arm returned **0/26 for every horizon**, and that result was
architecturally guaranteed: it would read 0/26 however much work the hidden state does. It was
reported (M60) as though it measured hidden's contribution.

The mirror failure is M56's `freeze hidden` arm. Holding `hs` at its anchor while the cell keeps
evolving does not *isolate* hidden — it feeds a stale readout to four gates and to the encoder input,
**breaking the LSTM's own recurrence**. Its +0.019 AP is real and replicated 4/4, but it measures
"what happens when you jam the readout", not "what hidden contributes".

**The general rule, which is what makes this Tier 2 rather than a note:** an intervention on a
component only measures that component if the architecture *carries the component forward*. Where a
component is a deterministic function of another, perturbing it tests the one-step map, not its role
in the dynamics — and the null it returns is indistinguishable from "this part does not matter".
Both readings were live in this programme for a day.

**What survives from the two arms:** the narrower and still-useful claim that **hidden carries no
spatial information across steps**, so it cannot be what drains during free-running. The cell-side
results are the trustworthy half throughout.


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

### D-13: single state-conditioned quantile head vs two-expert regime-routed mixture

| Field | Value |
|-------|-------|
| ID | D-13 |
| Source | expert-method-review (magnitude-fix panel, 2026-07-28) |
| Perspectives | **Side A (Koenker/quantile):** ONE monotone multi-quantile Δ head suffices — the τ-index absorbs the regimes (low-τ slices pin at 0 on true-zero cells, high-τ slices carry the surge), so the "two experts" are just different τ-slices of one monotone curve; regime flexibility can be added via a *state-conditioned* weight `γ(persistence-state)` in the single head. A softmax-routed two-expert mixture (Jacobs 1991) has **no monotonicity guarantee** across τ once you route between two full predictive objects — it reintroduces exactly the quantile-crossing the monotone head was built to kill. State-conditioned γ in one monotone head dominates. **Side B (Hamilton/Tong/regime):** the momentum-routed two-expert mixture **IS** the specified model — a SETAR (self-exciting threshold AR) with NN conditionals, and the Jacobs remedy to weight-crosstalk. The sibling lab proved one global γ can't serve both regimes (doc 19: γ=2 fixed declines 200× but blew active_stable 9.99→23.57), and DS3M (`Xu2021`) shows the ConvLSTM latent does NOT cleanly separate regimes on its own → an explicit hard router supplies discrete identifiability the recurrent latent lacks, at zero GPU cost. A single head is a false economy that will trade one stratum for another. |
| Resolution | **OPEN.** Both sides agree the eventual object may be regime-aware; they disagree on whether the regime work is done *implicitly by the τ-index in one monotone head* (Side A) or *explicitly by a discrete router over two heads* (Side B), and whether to skip the single head entirely. Cross-refs C-227 (the single-head stratum-trade risk), C-224 (neither is measurable until the tail-detectability gap is closed). Governing choice deferred to the magnitude pre-registration; recommended first step = single state-conditioned head with per-regime CRPS reporting (C-227), escalate to the routed mixture if the stratum-trade fires. |

---

### D-14: is the exogenous-covariate program worth a conditioning subsystem?

| Field | Value |
|-------|-------|
| ID | D-14 |
| Source | expert-method-review (covariate-ingestion panel, ceiling/parsimony seat vs forecasting seats, 2026-07-29) |
| Perspectives | **Parsimony/ceiling** (Box + conflict-diffusion, Schutte2011/Buhaug2008): crps_all is INERT to every static channel tried (0.142 across baseline / `ln_pop` / placebo); conflict's spatial persistence makes a static per-cell prior largely redundant with the cell's own history; the magnitude wall is the family/tail (C-149) — a FiLM/TFT covariate subsystem can at best sharpen OCCURRENCE, never the CRPS headline → possibly polishing brass. **Forecasting** (Lim/Salinas): occurrence is *half* the problem (the gate); a sharper gate has decision value for early warning even if CRPS is tail-bound. |
| Resolution | **Open — keep live.** Gate any covariate-conditioning-seam build (C-228/C-229/C-230 fixes) on a **demonstrated, decision-relevant occurrence gain** from the Step-1 encoder-only diagnosis; if population buys no real occurrence lift once the seam defect is removed, park the covariate program (the parsimony seat wins). |
| Cross-refs | C-149 (NB ξ=0 magnitude veto), C-224 (eval tail-blindness), C-228 (the seam defect), C-229/C-230, [[amount-ceiling]] |

---

### D-15: mixing-weight log — clamp-and-log (ZINB precedent) vs log-sigmoid

| Field | Value |
|-------|-------|
| ID | D-15 |
| Source | expert-code-review (2026-08-01, Epic #230 S2 #232 — Ousterhout/Hickey vs Nygard/Martin/Beck) |
| Perspectives | **Clamp-and-log (consistency/simplicity):** follow the shipped ZINB template `torch.log(pi.clamp(_PI_EPS, 1-_PI_EPS))` (`zero_inflated_negative_binomial.py:74-75`) for the mixture's `log w` too — one idiom across families, minimal surface. **Log-sigmoid (collapse-regime safety):** ZINB's `pi` rarely pins, so the clamp is benign there; the mixture's `w→1` collapse is the *central pre-registered observable* (F4), so the mixing-weight log MUST use `-softplus(∓raw_w)` to keep the gradient finite in the signal regime. |
| Resolution | **REVERSED 2026-08-01 (empirically, during S2 impl) → clamp-and-log (ZINB precedent) IS correct.** The initial resolution toward log-sigmoid was wrong on two counts: (1) `nll` receives the *activated* `w`, not `raw_w`, so log-sigmoid-from-raw isn't available without breaking the family contract; (2) `gradcheck.py` shows **clamp-and-log is NaN-safe** at exact saturation (finite value, grad 0) — the trap is *unclamped* `log(w)`, not "not-log-sigmoid." So the mixture clamps `w` before the log exactly as ZINB clamps `pi`. The only concession (dead gradient at exact `w=1`) is acceptable: a pinned collapse is a valid F4 decisive-negative. See the corrected C-249. |
| Cross-refs | C-249 (the Tier-1 concern this resolves), C-212 (the analogous log-domain NaN class) |

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
| C-210 | Warn (don't silently fall back to the region mean) when `_standard_gamma`'s 64-iteration Marsaglia-Tsang loop fails to accept — `nb_core.py:91-104`. *(Was tagged `[DEMOTED]` in §Open but missing from this index — added on the 2026-08-15 review-rr pass.)* | — |
| C-127 | Remove the duplicate dict keys in model configs (ruff F601) — the later definition silently shadows the earlier. | — |
| C-131 | Document (or config-surface) that `weight_decay=0.1` is deliberate and large in absolute terms. | — |
| C-166 | Stop rendering input-only static channels as predicted signal in the diagnostic plots (display drift only). | 8 |
| C-171 | Correct the `FocalLoss` docstring — it is `0.5·BCE` at γ=0/α=0.5, not BCE. | — |
| C-172 | Drop `FocalLoss`'s internal `unsqueeze(0)` so `reduction='none'` preserves input rank. | — |
| C-182 | Validate both architecture dimensional contracts in `__init__` — `total_hidden_channels % 8` and grid `H`/`W` `% 4` (absorbs the former C-188). | 4 |
| C-268 | Harden the diagnostic plotters — narrow the broad `try/except`, add a positivity guard before log-scale, replace the mutable default arg. | — |

**Backlog total: 13** (5 from the 2026-06-05 demotion pass, 1 index repair, 7 added 2026-08-15).

---

## Resolved Concerns

<!-- 2026-07-27 register tidy (review-rr strategic): the entries below were resolved-in-place in §Open and physically relocated here. -->

### C-260: scheduled-sampling channel substitution assumes features == regression_targets in ORDER (set-based warning misses it) — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-260 |
| Tier | 2 |
| Source | expert-code-review (2026-08-14, ADR-056 scheduled-sampling pre-run correctness review) |
| Trigger | Setting `ss_epsilon_max > 0` with a config whose `features` and `regression_targets` are the same length but a DIFFERENT order (e.g. reordering targets, or adding a dynamic covariate so the lists diverge) |
| Location | `views_hydranet/train/training_engine.py:329-334` (`t0_gt = t0[:, idx.feat]`; `torch.where(mask, prev_pred[:n_reg], t0_gt)`); `views_hydranet/utils/config_initializer.py:322-346` (`validate_laws` — set-based `logger.warning` only) |
| Cross-refs | C-98, C-105 (the count constraint, both marked RESOLVED via the set-based warning), C-259 (same SS-enablement trigger) |

The scheduled-sampling substitution replaces `t0_gt` (the `idx.feat` input channels) with `prev_pred` (the `n_reg` target forecasts) via `torch.where`. This assumes `features == regression_targets` in **count AND order**. `validate_laws` only `logger.warning`s on `set(features) != set(regression_targets)` (the "resolution" for C-98/C-105) — a **set** check that (a) never raises and (b) **passes same-length-different-order**, which would silently feed the `sb`-forecast into the `ns`-input channel (cross-target corruption). For the current conflict-only configs (`features == regression_targets`, same order) it is benign; enabling `ss_epsilon_max>0` on any reordered/extended config makes it a live silent corruption. **Tier 2 (silent model-input corruption under a realistic non-default config; clear trigger).** Fix direction: order-strict `list(features) == list(regression_targets)` **raise** when `ss_epsilon_max>0`.

**RESOLVED (verified 2026-08-15).** The fix direction named above landed: `config_initializer.py:1090-1097` raises when scheduled sampling is active and `list(features) != list(regression_targets)`, with the `C-260` marker in the message, and `tests/test_scheduled_sampling.py:264` asserts `pytest.raises(ValueError, match="C-260")`. The set-based `validate_laws` warning is unchanged and still only a warning — it is no longer load-bearing, because the order-strict raise fires first whenever the substitution can run.

**Its three siblings stay OPEN and were re-verified in the same pass:** C-259 is mitigated by rejection, not fixed (the gated-mean training feedback is marked a deferred fix in source); C-261 is open on its own terms — `training_engine.py:228-229` states that the production call site passes `generator=None`, so SS training feedback is not byte-reproducible and only the parity test exercises the seeded path; C-262 is open — only `test_epsilon_zero_produces_finite_loss` exists, with no byte-identical pin.

---

### C-197: distribution registry / legacy `output_distribution` name collision → silent legacy hijack — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-197 |
| Tier | 2 |
| Source | /falsify adequacy audit of ADR-067 §3 (2026-07-20) |
| Trigger | Registering a `DistributionFamily` in `DISTRIBUTION_REGISTRY` whose name equals a legacy `output_distribution` value (`standard`/`hurdle_shrinkage`/`hurdle_nb`/`hurdle_lognormal`/`dense_nb`/`quantile`) |
| Location | `views_hydranet/distributions/registry.py` (planned); `views_hydranet/utils/config_initializer.py` valid-list `~388-403` (`FAMILY_NAMES ∪ legacy`) |
| Cross-refs | ADR-067 §3; Epic A #167 (A-S2 #169 registry, A-S5 #172 config); C-196 (byte-identical foundation) |

The strangler-fig integration (ADR-067) unions the registry family names with the legacy `output_distribution` values into one valid-list and dispatches via `resolve_family(name)`. If the two name-sets **intersect**, a legacy config value routes to the new family instead of its untouched legacy branch — silently changing a proven, byte-identical model with **no error**, and invalidating every comparison to the lodestar baseline. **Tier 2:** structural fragility with a specific, realistic trigger (a future family author picking a colliding name). Fix: a fail-loud validator + test asserting `FAMILY_NAMES ∩ legacy = ∅` (registry names must be disjoint from legacy values); an acceptance criterion of A-S5.

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `config_initializer.py:777-793` `validate_family_legacy_disjoint` is a `model_validator(mode="after")` that raises when `family_names() & set(LEGACY_OUTPUT_DISTRIBUTIONS)` is non-empty; the code comment cites C-197 verbatim. It fires on every config load regardless of `output_distribution`, so the silent legacy-hijack path is closed. Relocated on the register↔code reconciliation pass.

---
### C-193: `body_mask` masking silently ignored under a latent loss — trains dense while config says masked — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-193 |
| Tier | 2 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Nygard/ADR-008 |
| Trigger | Sweeping `body_mask` (or setting `hurdle_threshold`+mode) to a masking value while `loss_reg` is a latent likelihood (`hurdle_nb`/`lognormal_nll`/`tobit`) |
| Location | `views_hydranet/train/training_engine.py:255-263, 343` (`if hurdle_threshold is not None and not use_latent`; warn-once C-180) |
| Cross-refs | C-194 (same interface), C-180 (the warn-once), ADR-008, ADR-003 Law 1 |

The point-body mask is silently a **no-op under a latent loss** — only a warn-once fires (C-180). A run can be configured "masked" and train **dense**, invisibly, with no error and no metric signal. Violates ADR-003 Law 1 (Fail Loud — it explicitly names "silent truncation") and ADR-008. **Tier 2:** silent wrong-training under a realistic sweep, no error signal. Fix: a hard `ValueError` at config validation when `body_mask ∈ {pos_cells,pos_timelines}` and the loss is latent (mirror the tobit/`hurdle_threshold` contradiction at `config_initializer.py:627`).

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `config_initializer.py:922-946` `validate_body_supervision_latent` raises exactly the `ValueError` this entry prescribed, keyed off the loss class's `needs_latent` flag in `LOSS_REG_REGISTRY` (not a hardcoded name list); the code comment cites C-193. The `body_mask` keyword itself is additionally rejected by `reject_retired_hurdle_knobs` (`:735-754`). Merged with C-180 (the same defect in its earlier `hurdle_mask_mode`/`active_window` form) and relocated.

---
### C-180: `active_window` hurdle mask is silently ignored under a latent loss — config no-op with no warning — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-180 |
| Tier | 3 |
| Source | /falsify "regression head + mask now 100% correct" round 2 (2026-06-26) — Finding B, SOFT |
| Trigger | Set `hurdle_mask_mode='active_window'` together with a `needs_latent=True` loss (`tobit`, `hurdle_nb`, `dense_nb`) |
| Location | `views_hydranet/train/training_engine.py:223` (`if hurdle_threshold is not None and not use_latent and hurdle_mask_mode == "active_window"`) |
| Cross-refs | C-178, ADR-063; the active_window mask (dossier `2026-06-23_body_sweep_dossier/16`) |

`active_cell` is computed only when `not use_latent`, and the masked hurdle-loss branch is likewise gated on `not use_latent`. So a config that asks for `active_window` decay supervision **while using a latent loss** gets **no active-window supervision at all — silently, with no warning or error**. The behaviour is *semantically* defensible (latent losses model the zeros/censoring themselves, so a hurdle mask does not apply), but the silent no-op of an explicitly-set flag means the user believes decay supervision is on when it is not — exactly the kind of invisible config drift that produced a multi-week mis-attribution before. **Tier 3:** no correctness corruption (the latent loss is doing the right thing), but a maintainability/honesty gap that misleads experiment design. Fix: log a warning (or fail-loud reject) when `active_window` is combined with a latent loss. Failing test: `tests/test_falsify_head_mask_round2.py::test_active_window_with_latent_loss_warns_or_raises`.

> **✅ RESOLVED — merged into C-193 and verified in source 2026-08-15 (review-rr strategic).** Same defect, earlier knob generation: `hurdle_mask_mode`/`active_window` are now rejected outright by `reject_retired_hurdle_knobs` (`config_initializer.py:735-754`), and the latent-loss silent no-op is raised on by `validate_body_supervision_latent` (`:922-946`). Both the knobs and the silent path are gone. See C-193 for the surviving record.

---
### C-194: `hurdle_mask_mode` read raw + un-validated — a typo silently degrades the mask to per_step — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-194 |
| Tier | 2 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Nygard/ADR-009 |
| Trigger | Setting `hurdle_mask_mode` in a config with a typo (e.g. `active-window` vs `active_window`) |
| Location | `views_hydranet/train/training_engine.py:549` (`config.get("hurdle_mask_mode","per_step")`); NO field in `config_initializer.py` |
| Cross-refs | C-193, ADR-009 (config as validated boundary) |

`hurdle_mask_mode` is not a config field — it's read straight from the dict with a `"per_step"` default, so any typo silently trains the wrong mask (e.g. `active_window` intended, per_step trained). No validation, no error. Violates ADR-009 (all boundaries validated). **Tier 2:** silent mis-training with no signal. Fix: the validated `body_mask` enum becomes the sole front door; the raw `config.get` read is deleted.

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `config_initializer.py:735-754` `reject_retired_hurdle_knobs` is a `model_validator(mode="before")` that raises if `hurdle_threshold`, `hurdle_mask_mode` or `body_mask` appear in the config at all, naming the `body_supervision` migration. A typo in the key can no longer silently degrade the mask — the key itself is refused. This entry's trigger is now unreachable.

---
### C-179: `reg_activation` is arch-affecting but NOT persisted in the artifact sidecar — silent activation mismatch on reload — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-179 |
| Tier | 2 |
| Source | /falsify "regression head + mask now 100% correct" round 2 (2026-06-26) — Finding A, SOFT |
| Trigger | Reload (eval/forecast/replay) a model trained with an **explicit `reg_activation` override**, or a pre-#178 relu-trained `hurdle_shrinkage`/`hurdle_lognormal` artifact, while the live config's activation default differs from training |
| Location | `views_hydranet/train/train_model.py:75-92` (`arch_keys` / `config_snapshot` — persists `output_distribution`, `static_channels`, but NOT `reg_activation`); `views_hydranet/utils/utils.py` `choose_model` (`reg_activation=config.get("reg_activation")`) |
| Cross-refs | C-159 (same sidecar-drift class — but that one crashed loud; this is silent), C-178 (the softplus fix this completes), ADR-063 |

The regression-head output activation `reg_activation` changes the forward function but is **absent from the persisted sidecar `arch_keys`**. On reload, `choose_model` therefore derives the activation from the *current* default (keyed off `output_distribution`, which IS persisted), **not** from what the model was trained with. Because softplus and ReLU share weight shapes, `load_state_dict` succeeds silently — so a model trained with one activation runs the forward with another, producing **wrong predictions with no error signal**. Demonstrated: a relu-trained `hurdle_shrinkage` artifact reloads as softplus (the round-2 probe hit this). **Tier 2:** silent-but-gated — it bites only when the trained activation differs from the reload-time default (explicit override, or a pre-#178 artifact); the production `hurdle_nb` path defaulted to softplus before and after, so it is unaffected. The fix mirrors the adjacent `output_distribution` line (`train_model.py:92`, whose comment already says "persist the head flag (else hurdle_nb reloads as ReLU)"): add `reg_activation` to the snapshot. Failing test: `tests/test_falsify_head_mask_round2.py::test_reg_activation_round_trips_through_sidecar`.

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `train_model.py:102-118` persists the RESOLVED reg-head activation into the `.pt.config.json` sidecar (`config_snapshot["reg_activation"]`), with `None` written for family/quantile heads whose activation is a closure reconstructed from `output_distribution` on reload; the code comment cites C-179. The silent activation-swap-on-reload path is closed.

---
### C-132: HydranetManager `_execute_model_training` override silently drops the wandb train-run lifecycle — RESOLVED

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

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `hydranet_manager.py:207-211` carries an explicit NOTE that `_execute_model_training` is intentionally NOT overridden, so the base `ForecastingModelManager` phase template retains ownership of the wandb run lifecycle (`initialize_run("train")` + `TrainingStage.finalize_training` + `finish_run`); training is customised through the `_train_model_artifact()` hook instead. The comment names C-132 as the reason. **C-133 (the general overridable-phase-template pattern risk) remains legitimately open.**

---
### C-201: self-zeroed ZINB decouples the classification (gate) head from the forecast — frozen-ruler AP/Brier then score a head the forecast ignores — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-201 |
| Tier | 2 |
| Source | /falsify (2026-07-20), P5 |
| Trigger | Scoring a self-zeroed `nb`/`zinb` family on the frozen lodestar ruler's gate metrics (AP/Brier) |
| Location | the lodestar scorer `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`; planned `distributions/` `prob_positive`; A-S11 (#178) eval |
| Cross-refs | C-199/C-200; ADR-067 (self-zeroed); F1 pre-registration; **C-211 (empirical confirmation — 300-lesson M1: count-only occurrence AP ~0.27 vs cls-gate ~0.44)** |

A self-zeroed ZINB produces its zeros from the distribution (`P(Y>0)=(1−π)·(1−NB(0))`), **not** from the classification head. But the frozen ruler computes gate quality (AP/Brier) on the cls head. So the reported gate metric describes an occurrence estimate the ZINB forecast does not use — the two can diverge silently, mis-informing the M1/M2 go/no-go. **Tier 2:** silent mis-attribution in the evaluation that gates production decisions. Fix: for self-zeroed families the ruler must score the **distribution-implied** `P(Y>0)` (family exposes `prob_positive`), or the eval must explicitly document that the cls head is decoupled and not the forecast's gate.

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `prob_positive` is now an abstract member of the `DistributionFamily` ABC (`distributions/base.py:104-106`, docstring cites C-201) and implemented by every family — `zero_inflated_negative_binomial.py:125` as `(1-pi)*(1-NB(0))`, `nb_core.py:132` via the numerically-stable `-expm1(log_prob_zero)`. A self-zeroed family can therefore supply the gate quantity the frozen-ruler AP/Brier metrics need, without an external classification head. Relocated on the register↔code reconciliation pass.

---
### C-251: mixture `prob_positive` numeric cancellation if built from the direct `prob_zero` — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-251 |
| Tier | 2 |
| Source | expert-code-review (2026-08-01, upfront design review of Epic #230 S2 #232) |
| Trigger | Implementing the mixture `prob_positive` literally as `1 - (w·NB1(0) + (1-w)·NB2(0))` using the direct `NBCore.prob_zero` (`(theta/(theta+mu))**theta`) |
| Location | planned `views_hydranet/distributions/mixture_negative_binomial.py` (`prob_positive`); contrast `nb_core.py:120-124` (direct `prob_zero`) and `zero_inflated_negative_binomial.py:124-130` (the stable `-expm1(log_prob_zero)` form) |
| Cross-refs | C-201 (self-zeroed gate scoring / the stable `prob_positive` lesson NB & ZINB already apply) |

`NBCore.prob_zero` cancels catastrophically for small `mu` / large `theta` (`(θ/(θ+μ))**θ → 1` minus a tiny number), which is exactly why `NegativeBinomialFamily`/`ZINBFamily` compute `prob_positive` via the stable `-expm1(NBCore.log_prob_zero(...))`. A literal mixture `prob_positive` re-introduces the cancellation, silently miscalibrating the gate/occurrence metrics the mixture is scored on. **Tier 2 (silent metric miscalibration on the occurrence side; not the primary CRPS but feeds the verdict).** Fix: `prob_positive = w·(-expm1(NBCore.log_prob_zero(mu1,theta1))) + (1-w)·(-expm1(NBCore.log_prob_zero(mu2,theta2)))`; test precision at small `mu`/large `theta`.

> **✅ RESOLVED (pre-empted by construction) — verified in source 2026-08-15 (review-rr strategic).** `mixture_negative_binomial.py:122-123` builds each component's `P(Y>0)` from the stable `-expm1(log_prob_zero)` rather than the direct `prob_zero`, with the cancellation hazard named in the docstring. Pinned by `tests/distributions/test_mixture_negative_binomial.py:216-234` ("mixture prob_positive must use -expm1(log_prob_zero), not the direct prob_zero"). The anticipated defect was never built.

---
### C-195: dual authority over "what is an event" — mask threshold vs binary-derivation threshold can drift — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-195 |
| Tier | 3 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Martin/Kleppmann/ADR-046 |
| Trigger | Changing the binary-target derivation threshold (`config['derivations']['binary'][...]['threshold']`) without changing the mask's hardcoded `> threshold` |
| Location | mask literal in `training_engine.py:263/349` vs `config_initializer.py:53` (`derivations`) |
| Cross-refs | C-193/C-194, ADR-046 (Transformations vs Derivations), ADR-003 Law 6 |

"A cell is an event where `y > 0`" is defined in **two** places — the binary-target derivation (config `derivations`) and the mask threshold in the training loop. They can silently diverge, so `by_*` labels and the body mask would disagree on which cells are events. **Tier 3:** maintainability/consistency hazard, no current corruption (both are 0 today). Fix: the mask sources its event threshold from the derivation config (single authority).

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** `body_supervision.py:75-76` `event_threshold_from_config` is now the single authority for "what is an event", sourcing the threshold from the binary-target derivation config, and `training_engine.py:307,680-684` reads it from there rather than a literal. Both the module docstring (`body_supervision.py:22`) and the call site cite C-195. The dual-authority split is closed.

---
### C-146: likelihood conflation — "ZINB" vs "hurdle-NB" are different models — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-146 |
| Tier | 2 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Trigger | Implementing `ZINBLoss` (#99) without first committing to ONE likelihood and writing its exact NLL |
| Location | dossier `2026-06-10_zinb_distributional_head_dossier/02_design.md §0/§2`; issue #99 |
| Cross-refs | C-137 (count-head likelihood-spec), D-08 (unified-NLL decision) |

The design names the head both "**ZINB**" (zeros from a Bernoulli gate **and** the NB's own zero mass — Lambert 1992) and "**zero-truncated NB on positives / hurdle_nb**" (zeros **only** from the gate, truncated positive body — Cragg 1971 / Mullahy 1986). **These are distinct likelihoods** with distinct NLLs and identifiability: in ZINB a zero has two explanations → π and the NB zero-prob are partially confounded; the hurdle factorizes cleanly but needs the truncated-NB normaliser. Implementing the wrong NLL for the intended model is a silent spec error (wrong gradients, wrong calibration). **Commit to one and write its exact NLL before #99.** **Tier 2:** structural mis-specification feeding everything downstream.

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** The likelihood commitment this entry demanded was made and is recorded at the head of the implementing module: `distributions/zero_inflated_negative_binomial.py:7` — "Committed likelihood (C-146): ZINB (Lambert 1992), NOT hurdle" — followed by the exact per-cell pmf. `hurdle_nb` survives as a separate, separately-named legacy `output_distribution`, and `validate_family_legacy_disjoint` (C-197) enforces that the two name-spaces cannot collide.

---
### C-196: `body_mask='none'` refactor must be byte-identical to the current foundation — else silent drift — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-196 |
| Tier | 3 |
| Source | expert-code-review (2026-07-18, `body_mask` design) — Feathers/Beck |
| Trigger | Refactoring the two-knob mask into `body_mask` without a characterization net |
| Location | `training_engine.py` masking path; `tests/` (no end-to-end characterization test today) |
| Cross-refs | C-193/194/195, ADR-005 |

The foundation (all-cell MSE gated) is the lodestar baseline. If the `body_mask` refactor changes the masked cell-set at `none` even slightly, the foundation shifts silently and every comparison to it is invalidated. There is currently **no** config→behaviour characterization test. **Tier 3:** regression risk on a load-bearing baseline. Fix: a characterization test snapshotting the current masked-cell-set for all three legacy knob-combos BEFORE the refactor, asserted identical after.

> **✅ RESOLVED — verified in source 2026-08-15 (review-rr strategic).** The characterization net exists: `tests/test_body_supervision_contract.py:6` asserts `body_supervision='all'` equals the all-cell foundation to numerical equality, and refers to the concern in the past tense ("was C-196"). The two-knob mask refactor has since been superseded a second time (`body_mask` → `body_supervision`, ADR-065 amend. 2026-07-28) with the byte-identity requirement pinned throughout.

---
### C-184: BatchNorm runs inside the recurrent loop — running stats accumulate T× per window over temporally-correlated steps — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-184 |
| Tier | 2 (upgraded from 4 on 2026-06-27 — see narrative) |
| Source | /falsify ConvLSTM (2026-06-26) P5; **CONFIRMED root cause via BN-recal experiment (2026-06-27)** |
| Trigger | FIRES NOW on every training run: ~40% of seeds land BN running-stats that over-amplify at eval → gate saturates → composed E[y] explodes (the seed-bimodality + much of C-113). Acute on any retrain. |
| Location | `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` (`bn_enc_conv0/1`, `bn_bottleneck_conv`, the `bn_dec_conv*` head BNs — all invoked inside the per-timestep `forward`, which the engine calls T times per window) |
| Cross-refs | C-114 (undocumented recurrent-regularization surface), C-183, C-113 (rollout dynamics) |

The encoder/bottleneck/decoder `BatchNorm2d` layers are inside the single-timestep `forward`, called T× per window over **temporally-correlated** activations (cf. Cooijmans 2016 recurrent BN). Originally logged Tier-4 ("stable design choice"). **⬆ UPGRADED Tier-2 — CONFIRMED ROOT CAUSE (2026-06-27).** The 2026-06-26 perf program found the production floor is **seed-bimodal (~40% of seeds collapse: saturated gate π̄≈0.1–0.36, rollout MCR_pos 30–260×)**. Triangulated the cause: NOT the loss (pos_weight sweep flat), NOT the weights (per-layer spectral norms + gate-head bias identical good-vs-bad), NOT the training trajectory (good/bad train-time gate-logit identical, because **training uses batch-stats BN**). The decisive test (`bn_mode_probe.py`): every seed is calibrated under **train-mode BN** (π̄≈0.002–0.005) but saturates under **eval-mode BN** (π̄ 0.4–0.998), worst for the bad seeds (which have lower BN `running_var` → eval BN over-amplifies). **FIX CONFIRMED + UNIVERSAL:** recompute BN running stats post-training (forward-only over real windows, reset BN + `momentum=None`) flips **6/6 bad seeds BAD→GOOD and preserves 2/2 good** — bad-basin rate ~40%→0%, rollout MCR_pos collapses to 2.5–8.3× (e.g. seed 201: step-1 CRPS 33.8→0.24, MCR 259→5.8). So this is **silent eval-time model-output corruption on ~40% of trained models** (Tier-2: not Tier-1 only because it surfaces as loud explosions, not a quiet wrong answer, and is now fixable). **Resolution paths:** (a) post-training BN-recal pass before artifact save [cheapest, validated], (b) fix the recurrent-BN momentum/update at the root, (c) GroupNorm/LayerNorm (no train/eval gap; needs retrain). Opt-in `bn_recal_from` flag in `training_engine.py` (uncommitted) implements the test. Tools: `/tmp/run_bn_recal_all.sh`, `/tmp/bn_mode_probe.py`, `/tmp/recal_all_score.py`. Cross-ref C-113 (this is a large part of the eval-explosion), C-147 (gate-calibration), the perf program.

> **✅ RESOLVED WITH RESIDUAL — status corrected 2026-08-15 (review-rr strategic).** This entry's text still describes the fix as an "opt-in `bn_recal_from` flag in `training_engine.py` (uncommitted)". That is **stale**: the fix is committed and **default-ON** — `training_engine.py:1084` runs `_recalibrate_bn` under `config.get("bn_recalibrate", True)`, snapshotting the BN buffers first and restoring them on any failure so a recal error can never lose a completed training run (`:1085-1099`). The seed-bimodal eval collapse is therefore mitigated by default on every new artifact. **Residual tracked as C-273** (the recalibration windows are drawn from the curriculum's high-intensity head, not a representative slice) — that entry remains open in §Open Concerns.

---
### C-138: Stale test import breaks suite collection — `test_eval_integration_toy` imports a removed `views_evaluation` module — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — F-Z1: submodule-level `importorskip` → a bare `pytest` collects clean (#95 gone).**

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

**Update 2026-07-31 (/falsify, F-Z1 — precise cause):** the file *already has* `pytest.importorskip("views_evaluation")` on **line 4**, but it guards the **top-level package** while line 6 imports the **submodule** `views_evaluation.evaluation.evaluation_manager`. The installed `views_evaluation` top package imports fine, so importorskip is a **no-op**, and line 6's submodule import still raises → collection hard-errors despite the guard. **Precise fix: `importorskip("views_evaluation.evaluation.evaluation_manager")`** (skip at the granularity of the thing actually imported). Falsification stub: `tests/test_falsify_zero_surprises.py::test_P2_plain_pytest_collects_without_ignore`.

---
### C-234: emit_family_core rollout is half-wired — emit uses the large π-stripped core, AR feedback uses the small self-zeroed body → silent verdict corruption — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S1: `_sample_feedback` is core-aware (mirrors `_emit_magnitude`); regression test.**

| Field | Value |
|-------|-------|
| ID | C-234 |
| Tier | 1 |
| Source | code-review (max, 2026-07-31; F0 — verified against source) |
| Trigger | Trusting any h≥2 / horizon or bloom readout of a `{th_,}gated_ZINBcore` run (`emit_family_core=True`), OR shipping emit_family_core more broadly, before `_sample_feedback` is made core-aware |
| Location | `views_hydranet/utils/hydranet_inference.py` (`_emit_magnitude`:253 uses `mean_core`; `to_cube_samples(core=)`:793 uses `sample_core`; but `_sample_feedback`:311 draws `fam.sample` — self-zeroed) |
| Cross-refs | C-113 (AR feedback carrier), C-239 (training-side twin), C-240 (compose guard), C-242 (validator message) |

`_emit_magnitude` and the scored D×K cube correctly switch to the π-stripped **core** under `emit_family_core` (`mean_core`/`sample_core`), but `_sample_feedback` — the DEFAULT AR feedback for family heads (`rollout_feedback` auto-resolves to `'sample'`, :101) — was **not** updated and still draws the **self-zeroed** `fam.sample`. So a th_gated_ZINBcore rollout **emits/scores the large core but feeds back the small self-zeroed body**: every horizon h≥2 is conditioned on a history the model never emitted. `_sample_feedback`'s own docstring ("Mirrors `_emit_magnitude`'s family branch") is the evidence this is an unintended miss, not a choice. **Tier 1 (silent verdict incorrectness):** no error fired; it silently invalidated the h≥2 half of the E3 th_gated_ZINBcore verdict (`2026-07-29_v2_scoreboard_dossier/07` — F2 "no bloom" is untrustworthy, the "stability" may be an artifact of the too-small feedback). h=1 readouts (incl. the decisive F1 crps_events @h1) are unaffected (no feedback at h=1). Fix: mirror the `emit_family_core` branch in `_sample_feedback` (draw `sample_core`), TDD it, re-emit the 3 banked zinb seeds, re-derive F2 + the horizon curve. **Scope: `emit_family_core` defaults False → the shipping gated_NB/ZINB are unaffected** (experiment-only bug).

---
### C-235: data-backed static channel leaves silent 0-holes for cells/months absent from the df (geometry statics fill the full grid; data-backed does not) — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S4: a data-backed static fails loud on NaN/inf coverage holes (no silent 0-hole).**

| Field | Value |
|-------|-------|
| ID | C-235 |
| Tier | 2 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Wiring a data-backed `static_channels` covariate (e.g. datafactory `ln_pop`) whose panel is sparser than the conflict panel — a cell/month present in conflict rows but absent from the covariate |
| Location | `views_hydranet/utils/volume_handler.py` (~250, data-backed static fill into a zeros volume) |
| Cross-refs | C-228 (same seam, placement defect), C-229 (covariate taxonomy), C-236/C-237/C-238 (sibling data-backed-static gaps) |

The data-backed static path writes the covariate only at observed `(cell, month)` df rows into a zeros volume, so any study cell or month absent from the df keeps a **silent 0** — unlike geometry statics, which fill the full grid at all months. `ln_pop` (the stated near-term use) can enter the panel later than conflict or cover a subset of cells: a cell with conflict at months 100–500 but population only at 300–500 gets `ln_pop=0` (population≈1) for 100–299, digested as a real covariate and — if the origin slice is 0 — re-injected as 0 across the whole rollout. **Tier 2 (silent, realistic near-term trigger):** no error; corrupts a covariate the model treats as real. `test_data_backed_static_channel.py` uses a fully-dense grid so it cannot catch the hole. Fix: fill-completeness contract (forward/mean-fill or fail-loud on missing support) + a sparse-panel test.

---
### C-236: data-backed static channel bypasses both FeatureScaler guards → raw-magnitude / NaN reaches the encoder unscaled and unchecked — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S4/S5: NaN guard + magnitude sanity rail; deeper model-side scaling deferred (C-244/#229).**

| Field | Value |
|-------|-------|
| ID | C-236 |
| Tier | 2 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Declaring a data-backed `static_channels` covariate with raw (unlogged) magnitude — e.g. population in [0, 5e7] — without also listing it under `transformations`/`features` |
| Location | `views_hydranet/utils/volume_handler.py` (~250, static fill); `views_hydranet/utils/feature_scaler.py` (`configured_columns`:51 iterates transform cols; unmapped/gradient guard:79–83 iterates `config['features']`) |
| Cross-refs | C-235/C-237/C-238 (sibling gaps), C-228 (same seam) |

Neither FeatureScaler guard covers `static_channels`: the NaN/Inf guard iterates transform columns and the unmapped/gradient-explosion guard iterates `config['features']` — a static channel is in neither. So a raw-magnitude static (population ~1e7) reaches the encoder **unscaled**, dominating every log1p-scaled feature (gradient explosion — the exact failure the unmapped-feature guard exists to prevent), and on a bare `from_df` call (the CoordConv A/B harness/tests) even a **NaN** in the static is uncaught. **Tier 2 (structural, silent-to-loud):** trigger is a specific future covariate wiring; surfaces as training instability, not a clean error. Fix: route static channels through (or parallel to) the scaler's scaling + NaN/unmapped guards.

---
### C-237: geometry-vs-df static precedence silently reclassifies a registered geometry static as data-backed when a same-named df column exists — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S3: registry-authoritative role classification; registry∧df collision fails loud.**

| Field | Value |
|-------|-------|
| ID | C-237 |
| Tier | 2 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Declaring a `static_channels` name that also appears as a df column — a datafactory covariate name matching a `STATIC_CHANNEL_DERIVATIONS` registry key, or names like `row`/`col` (literal df spatial_cols) |
| Location | `views_hydranet/utils/volume_handler.py` (~246, `geom_static = [n for n in static_channels if n not in df.columns]`) |
| Cross-refs | C-235/C-236/C-238 (sibling gaps), C-228, C-230 (raw concat primitive) |

`geom_static = [n for n in static_channels if n not in df.columns]` silently reclassifies a registered geometry static as data-backed whenever a same-named df column exists — df-column precedence, no warning. Old code always called `derive(name)` (e.g. a coordinate normalized to [-1,1]); new code silently fills from the raw df column (raw indices / arbitrary units) instead, feeding the model a **different, unnormalized channel** with zero diagnostic. **Tier 2 (silent wrong-channel):** trigger is a realistic name collision under the datafactory covariate namespace. Fix: make the geometry-vs-data-backed classification explicit/authoritative (registry wins, or fail-loud on collision), not df-column-presence.

---
### C-238: no invariant enforces "static = constant per cell across time" — a time-varying df column declared static is fed varying in history but pinned in rollout — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S6: constant-per-cell static invariant enforced (fail-loud on time-varying).**

| Field | Value |
|-------|-------|
| ID | C-238 |
| Tier | 2 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Declaring a genuinely time-varying df column (e.g. monthly population, shdi) as a `static_channel` |
| Location | `views_hydranet/utils/volume_handler.py` (~250, `from_df` writes per-observed-month with no constancy check); `views_hydranet/utils/hydranet_inference.py` (history digest :447 vs rollout pin :522–524); ADR-060 (I3) |
| Cross-refs | C-229 (covariate-taxonomy root: static seam re-injects time-varying as constant), C-235/C-236/C-237 (sibling gaps) |

`from_df` writes `df[col].values` per observed month with **no constancy check**, so a time-varying "static" is digested with its true varying trajectory during history (t<origin) but pinned to the origin month during the AR rollout — the same channel treated two different ways in one inference, silently violating ADR-060 I3, with no validator rejecting a non-constant static. **Tier 2 (structural, silent):** distinct from C-229 (which is the taxonomy/design gap) — this is the concrete missing *enforcement*. Fix: a validator that rejects a non-constant-per-cell static (or routes it to a proper dynamic-covariate path once C-229 is designed).

---
### C-239: training-time family feedback/target is not core-aware → once C-234 is fixed, ZINBcore train exposure diverges from eval exposure — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — accepted — th_gated_ZINBcore DROPPED, so no core arm ships → the train/eval exposure mismatch is moot.**

| Field | Value |
|-------|-------|
| ID | C-239 |
| Tier | 2 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Fixing C-234 (core-aware eval feedback) and then trusting a th_gated_ZINBcore verdict, OR training a zinb with scheduled sampling intending to evaluate it as a core-emit arm |
| Location | `views_hydranet/train/training_engine.py` (~229, `_family_feedback_log1p` / `_family_target_log1p_mean` — self-zeroed, not core-aware); `emit_family_core` is eval-only / not persisted |
| Cross-refs | C-234 (eval-side twin), C-99 (reg feedback path), C-113 (AR exposure) |

The training-time family feedback and target use the self-zeroed sample/mean and are not core-aware. `emit_family_core` is an eval-only re-interpretation (not persisted to the artifact), so training cannot consult it. Once C-234 makes the *eval* feedback core-consistent, a scheduled-sampling-trained zinb was exposure-trained on sparse self-zeroed feedback but rolled out on dense core feedback — reintroducing the exposure-bias drift the sample-feedback mechanism (ADR-070) was built to remove. **Tier 2 (structural, latent):** an inherent train/eval mismatch the re-emit-banked-artifact approach introduces; must be named in any finalized ZINBcore verdict and decided (retrain a true core model vs accept the caveat). Fix scope tied to C-234.

---
### C-240: to_cube_samples has no guard for the invalid core=True + composition='self_zeroed' combo → silent ungated core draw — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S1: `to_cube_samples` fails loud on core=True + self_zeroed.**

| Field | Value |
|-------|-------|
| ID | C-240 |
| Tier | 3 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Calling `to_cube_samples(..., composition='self_zeroed', core=True)` from a hand-built dict / ad-hoc driver that bypasses HydraNetConfig validation |
| Location | `views_hydranet/distributions/sampling.py` (~65–74; the guard rejects a gated composition missing its gate, not the inverse) |
| Cross-refs | C-234 (same feature), C-242 (config validator message) |

The `to_cube_samples` guard only rejects a gated composition missing its gate, not `core=True + self_zeroed`. That combo silently draws the ungated, un-self-zeroed NB core (zeros nowhere) → ~85% nonzero / mean~5 on a ~99.7%-zero field, a silent ~8× over-forecast, instead of failing loud. HydraNetConfig validation rejects it upstream, so only an ad-hoc dict bypass hits this. **Tier 3 (fail-loud gap, upstream-mitigated):** add the symmetric guard.

---
### C-241: canonicalize_config_grid_name does not dedup → a config listing both grid aliases collapses to a duplicate, tripping a false Index Contract Violation — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S8: `canonicalize_config_grid_name` dedups both aliases (no false Index Contract Violation).**

| Field | Value |
|-------|-------|
| ID | C-241 |
| Tier | 3 |
| Source | code-review (max, 2026-07-31; on the #144/#216 grid fix) |
| Trigger | A config whose `index_names`/`identity_cols` list BOTH `priogrid_gid` and `priogrid_id` (e.g. a hand-merged migration config) |
| Location | `views_hydranet/utils/grid_naming.py` (~54, maps every alias member to `grid` with no dedup); manager guard checks the DATA not the config |
| Cross-refs | GH #144/#217 (grid-naming), C-228 |

`canonicalize_config_grid_name` maps every `GRID_ID_ALIASES` member to `grid`, so `['month_id','priogrid_gid','priogrid_id']` collapses to `['month_id','priogrid_id','priogrid_id']`. The manager guard checks the data (`len(_grid_present)==1`), not the config, so it does not prevent this; downstream `standardize_raw_df` builds a length-3 expected index that no longer prefix-matches the length-2 data index → a false "Index Contract Violation" on otherwise-valid data. **Tier 3 (fail-loud false-positive on an unusual config):** dedup while preserving order, or fail-loud if both aliases are present.

---
### C-242: config validator emits a factually-wrong message for zinb + emit_family_core=True + self_zeroed ("zinb is not self-zeroed") — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S1: validator message corrected for zinb + emit_family_core.**

| Field | Value |
|-------|-------|
| ID | C-242 |
| Tier | 4 |
| Source | code-review (max, 2026-07-31) |
| Trigger | Enabling `emit_family_core` for the first time and leaving `forecast_composition='self_zeroed'` (forgetting the required external gate) |
| Location | `views_hydranet/utils/config_initializer.py` (~854, rule (2) message) |
| Cross-refs | C-234, C-240 |

The validator correctly rejects zinb + emit_family_core + self_zeroed but with the wrong reason: "output_distribution=zinb is not self-zeroed…". zinb **is** self-zeroed — it is `emit_family_core` that stripped π — so the message contradicts the glossary/ADRs and misdirects debugging. **Tier 4 (DX, no correctness impact):** reword to "emit_family_core strips zinb's π; add a gate (soft/threshold) or drop emit_family_core".

---
### C-243: VisualDiagnostics is constructed with the config BEFORE grid canonicalization → viz holds the stale grid alias on renamed (priogrid_id) data — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — S8: `VisualDiagnostics` config refreshed after grid canonicalization.**

| Field | Value |
|-------|-------|
| ID | C-243 |
| Tier | 4 |
| Source | code-review (max, 2026-07-31; on the #144/#216 grid fix) |
| Trigger | Running on datafactory `priogrid_id` data with a legacy `priogrid_gid` config and reading the diagnostic biopsies |
| Location | `views_hydranet/manager/hydranet_manager.py` (~220/303 viz built with `self.configs`; canonicalization at ~106) |
| Cross-refs | GH #144/#217, C-241 |

`VisualDiagnostics` is built with `self.configs` before/independent of the pipeline's grid-key canonicalization, so viz internally holds the pre-canonicalization alias on renamed data; any grid column it reads mislabels or misses the grid. **Tier 4 (diagnostic-only, does not corrupt pipeline output):** the biopsy plots are silently wrong on migrated data. Fix: build viz from the canonicalized config, or have viz resolve the grid via `grid_id_col`.

---
### C-247: `test_score_v2_horizons.py` is non-portable — hardcoded absolute machine path + runtime-loads gitignored `reports/` tools; green ONLY on this machine — RESOLVED

**✅ RESOLVED (merged to `development` via PR #216) — F-Z2: repo/platform-relative paths + skip-if-absent guards → CI-portable (both test files).**

| Field | Value |
|-------|-------|
| ID | C-247 |
| Tier | 2 |
| Source | /falsify (2026-07-31, F-Z2) |
| Trigger | Running the suite in **CI** or on a **fresh clone** (or any machine other than this one) — the test errors (path/file absent), not skips |
| Location | `tests/test_score_v2_horizons.py:20` (`_HN = Path("/home/simon/Documents/…/views-hydranet")`), `:22`/`:27` (`spec_from_file_location` on `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py`), `:99–100` (`sys.path.insert` + `import lodestar_score` from `reports/2026-07-17_lodestar_eval_dossier/tools`) — both `reports/` paths are **gitignored** (absent in a clone). **Second instance:** `tests/test_falsify_8sample_readiness.py:12` hardcoded `/home/simon/…/views-models/…/config_hyperparameters.py` — an absolute path into the **sibling views-models repo** (worse: cross-repo). |
| Cross-refs | C-138 / C-165 (test-suite collection integrity / CI `--ignore` masking), C-159 (dossier tools not self-validating) |

A **tracked, committed** test hardcodes an absolute path to one developer's machine and, with **no `exists()`/skip guard**, runtime-loads two tool files that live under the **gitignored `reports/`** tree (research-dossier tooling, not part of the shipped package). It therefore passes **only because those files happen to exist locally**; in CI or any fresh clone it raises (path missing / file-not-found), not skips. **Silent false-green:** reported "full suite green / 1254 passing" this session was partly propped up by machine-local state — this test provides **zero** portable/CI coverage while looking like it does. **Tier 2 (structural fragility, clear trigger = CI/clone, false-confidence):** not model-output corruption, but it degrades the suite's value as a gate and will fail the moment CI runs it. Fix: repo-relative path (`Path(__file__).resolve().parents[1]`) + a module-level skip when the gitignored tool is absent (or relocate the scorer tool under the tracked package so the test is real in CI). Falsification stubs: `tests/test_falsify_zero_surprises.py::test_P5a_*` / `test_P5b_*`.

---

### C-256: `exante_stratum` index-keyed-frame branch was untested — RESOLVED

> **RESOLVED 2026-08-03 (this PR).** Added `test_gw_stratified.py::test_exante_stratum_accepts_index_keyed_frame` — feeds an index-keyed frame (the real v2 truth parquet layout) and asserts the stratum matches column-form. Guards the reactive `reset_index` fix against a silent regression.

| Field | Value |
|-------|-------|
| ID | C-256 |
| Tier | 3 |
| Source | code-review max (2026-08-03, PR mixture-nb-head → development) |
| Trigger | Refactoring `exante_stratum` such that the `reset_index` guard is dropped/broken |
| Location | `reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py` (`exante_stratum`) |
| Cross-refs | C-248 (the stratum leakage guard), C-247 (gitignored-tool test skip) |

The `reset_index` branch (the fix for the real v2 truth parquet, whose `month_id`/`priogrid_id` are the INDEX not columns) had no regression test — every fixture in `test_gw_stratified.py` was column-form. A refactor could silently break real-parquet stratification while all tests stayed green — the exact reactive-bug class that crashed the finisher this session. **Tier 3 (test-coverage guarding a silent-nullifier).** Fixed in-PR (see above).

---

### C-257: `score_gw_v2` repo-root path resolution (parents[3]) was unexercised — RESOLVED

> **RESOLVED 2026-08-03 (this PR).** Added `test_gw_stratified.py::test_score_gw_v2_repo_root_resolves_frozen_primitives` — asserts the `parents[3]` repo-root locates `lodestar_score.py` + `rollout_skill_score.py`. Guards the path fix so a dossier-depth change is caught in tests, not as a finisher-time `ModuleNotFoundError`.

| Field | Value |
|-------|-------|
| ID | C-257 |
| Tier | 4 |
| Source | code-review max (2026-08-03, PR mixture-nb-head → development) |
| Trigger | Changing the dossier directory depth or the `sys.path` inserts in `score_gw_v2` |
| Location | `reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py` (`score_gw_v2`) |
| Cross-refs | C-256 (peer coverage gap), C-247 |

`score_gw_v2`'s `parents[3]` repo-root resolution + the frozen-primitive imports were never exercised (tests skip when the gitignored `reports/` tree is absent). **Tier 4 (loud failure, not silent — a break surfaces as a clear import error at run time).** Fixed in-PR (see above).

---

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
| Cross-refs | C-202 (θ gradient bound); *v1-review "C-3" (generator determinism) — NOT register C-03*; A-S12 (#179) CRPS M2 comparison |

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
| Cross-refs | C-199 (the *dead*-gradient extreme — this is the *exploding*-gradient counterpart at the opposite end of the `θ` range); C-200; *v1-review "C-8"*-style per-cell θ instability |

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

> ⚠️ **Resolution PARTIALLY INVALIDATED 2026-08-15 (repo-assimilation reachability sweep) — see C-271.** The mechanism exists and is tested, but grep finds **no caller in `views_hydranet/`**: `audit_manifest` is invoked only by `tests/test_genome_audit.py`, despite its own docstring stating "Must be called before `lock_entropy()` and `training_loop()`". This closure was granted on the mechanism's *existence*, not its *reachability*. Tracked as **C-271** (Tier 3, §Open) rather than reopened here; an external caller in `views-models` has not been ruled out.

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

### C-136: Magnitude/output fixes judged on a rollout-confounded test — Arm-1 was mischaracterized as a clean failure — RESOLVED

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

**Update 2026-07-27 — CLOSEABLE (the axis-split discipline is now institutionalized, not a per-readout reminder).** The confound this entry names — judging a magnitude fix on rollout-confounded metrics — is now structurally prevented by two frozen instruments built since: the **T=0 frozen lodestar ruler** (`2026-07-17_lodestar_eval_dossier`, identical months/cells/truth) isolates the magnitude/calibration axis at step-1, and the **per-horizon rollout-skill ruler** (`2026-07-25_t0_rollout_skill_dossier`, ADR-070) scores the rollout axis separately with a free-vs-oracle exposure-bias split. Magnitude changes are now scored at T=0 with the CRPS-primary + spatial-sharpness rule (C-167 close-out) — a proper score with CI, not step-1 `MCR_pos`. The remediation is delivered; **eligible for §Resolved in the next tidy.** Kept open only pending the physical move.

---

### C-148: "softplus dissolves the autoregressive explosion by construction" is unproven — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-148 |
| Tier | 2 |
| Source | expert-method-review (ZINB Pass-2, 2026-06-10) |
| Trigger | Treating C-113 as solved / soft-pedaling the #102 explosion-check on the strength of the "by construction" prose |
| Location | dossier `00_README.md`, `02_design.md §0` |
| Cross-refs | C-142 (the gate), C-113 (the explosion), C-151 (the empirical post-run clamp-confound sibling), C-152 (the load-bearing-analogy sibling) |

The claim that softplus `E[y]` feedback dissolves the C-113 runaway "**by construction**" is unproven and contested by dynamical-systems theory (Mikhaeil 2022 / Hess 2023 / Durstewitz: the blow-up is a property of the recurrent **operator's gain** / Jacobian spectral radius, not the output nonlinearity alone). `02_design §6` correctly gates on `diagnose_io_gain`, but the "by construction" prose elsewhere invites skipping the gate. Delete the over-claim; treat the explosion-check as the **load-bearing** test. **Tier 2:** an over-claim that, if believed, wastes the build and re-explodes.

**Update 2026-07-27 — CLOSEABLE (the over-claim was empirically settled, in the entry's own favor).** The bloom epic replaced "by construction" prose with a measured verdict: the 36-step runaway is a **feedback / exposure-bias bug**, not an output-nonlinearity property — exactly as this entry (and Mikhaeil/Durstewitz) argued. The counted bloom verification (`2026-07-25_t0_rollout_skill_dossier/06_bloom_verification_verdict.md`, 9/9 arms) showed mean-feedback blooms while `rollout_feedback=sample` (ADR-070) is bounded — softplus alone does **not** dissolve it, the feedback content does. The load-bearing explosion-check is now the frozen per-horizon rollout-skill ruler (free-vs-oracle gap), not prose. The concern is resolved in the direction it warned of; **eligible for §Resolved in the next tidy.** Kept open only pending the physical move.

---

### C-167: no spatial-sharpness / resolution metric — evaluation is calibration-only (resolution-blind) — RESOLVED

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

**Update 2026-07-27 — residual CLOSED for the magnitude effort (instrument validated on the CURRENT family cube + integrated into the pre-registered decision rule).** The thin residual above blocked starting the magnitude effort resolution-blind. Now closed: (1) `sharpness_scorecard.py` was **stale on the current data** — the GH#144 grid rename (`priogrid_gid`→`priogrid_id`) made it raise `ValueError: raw parquet missing 'priogrid_gid'` on every family cube; fixed with a `_grid_col` alias resolver + 2 tests (10 pass). (2) Validated end-to-end on a fresh nb gated_NB-42 family D×K cube — FSS@{1,3,5,11}/area_ratio/conc1% compute on the `origin_*/y_pred.npy` format. (3) **Foundation STEP-1 baseline recorded** (nb gated): FSS@1 ≈ 0.00–0.01, area_ratio 0.1–0.2× (timid/under-firing), conc1% 0.49–0.58, MCR 0.004–0.015 — the reference the magnitude effort must improve without degrading. (4) **Wired into the pre-registered, FAO-02-compliant magnitude decision rule** (`2026-07-20_distributional_head_dossier/05_analysis_plan.md` §"Magnitude decision rule" + falsifier **F2b**; recipe on that dossier's `00_README`): a magnitude change WINS iff crps_events improves [proper, selects] AND size_ratio→1 [magnitude] AND FSS@1/conc1% do NOT degrade [spatial guard, corroborates — never selects alone]. The resolution-blind mistake this entry warns of can no longer certify a smeared body as a win. **Eligible for relocation to §Resolved in the next register tidy** (kept in Open only pending that physical move).

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps in numbering indicate merged or resolved entries
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `tech-debt-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** `Total Concerns` and `Open Concerns` in the register header are manually maintained — update them whenever a concern is added or resolved
- **Foreign numbering schemes:** an `/expert-code-review` run emits its own `C-1`, `C-2`, … finding list that is **not** this register's numbering. Unpadded citations (`C-3`, `C-6`, `C-8`) in some narratives refer to that v1-review list, **not** to register entries `C-03`/`C-06`/`C-08`, which are unrelated concerns. When quoting a foreign finding, italicise it and say so (`*v1-review "C-6"*`); register IDs are always zero-padded below 10. *(Convention added 2026-08-15 after a review-rr cross-reference audit found 4 such collisions.)*
- **Demotion:** Tier-4 + mechanical + single-file/single-developer entries are tagged `[DEMOTED]` in their §Open title, given a demotion banner, and indexed in §Tech-Debt Backlog. They stay physically in §Open for traceability but are **not** counted as active risks — the header carries a separate demoted count. Keep the tag and the index in sync.
- **Governed by:** ADR-048
