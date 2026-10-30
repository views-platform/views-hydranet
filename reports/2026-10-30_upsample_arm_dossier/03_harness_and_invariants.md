# 03 — Harness and invariants

**2026-10-30.** ⭐ **Discovered by auditing the repository, not templated.** Every row names the
file it was found in. ⛔ This is the document whose absence let three process bypasses through — see
`07`'s postmortem.

---

## A · Invariant taxonomy

### A1 · HARD — never break; any experiment violating one is invalid

| invariant | where it lives |
|---|---|
| **The incumbent stays the default and byte-identical.** Candidate 7 adds a subclass and two registry names; `HydraBNUNet06_LSTM4` is not modified | `registry.py`; proven by `tests/architectures/test_architecture_registry.py` running the contract over **every** registered name |
| **One variable per arm.** The only config key differing between an arm and its control is `model` | `make_arch_arm.py` (`arm_identity`), re-asserted independently by `verify_bakeoff.py` step 1, and checked against the own-seed control by `preflight_exp02.py` |
| **Fail loud, never silently degrade.** Unknown architecture → `ValueError` naming what is registered; a stage that is not the defect → `ValueError`; no silent CPU fallback | `registry.py:69-74`; `upsample_interp.py`'s build guard; `preflight_exp02.py` device gate |
| ⛔ **BatchNorm is never put in train mode by a diagnostic.** MC-dropout is enabled with `set_locked_dropout(True)`, never a bare `model.train()` | `HydraBNrecurrentUnet_06_LSTM4.py:741-755`; asserted in `gf_phase_signature.py` (`G-24`). **C-384 Tier 1** — a bare `train()` put 15 BN layers into train mode, overwrote delivered running statistics, and **inverted a headline's sign** |
| **Artifact identity (`F6`).** A checkpoint is the delivery you think it is: admitted model / family / dropout sets, sha256, PROVENANCE present, `MIN_LESSONS`, known-africa hashes denied | `vehicle.py:25-63, 163-250` |
| **Data matches its own configuration.** A cached pull whose queryset digest differs is **refused**, not served | `views_pipeline_core` `provenance_builder.py:181` (**C-61**) |
| **Reproducibility: entropy locked before any measurement** | `ReproducibilityGate.lock_entropy(np_seed, torch_seed)`, used by `d4_mu_on_the_grid.py` and `gf_phase_signature.py` |
| **Full suite + lint green; working tree clean before a queue starts** | `test_falsification_repo_clean.py`, `test_risk_register_integrity.py`, `test_cic_drift_detection.py`; enforced at launch by `launch_exp02.sh` |
| ⛔ **Measurement scripts are committed BEFORE they run** (D-5), and the free parameter is swept before a scalar is reported (D-4) | repository discipline; `gf_phase_signature.py` swept the remedy |
| ⛔ **FAO-02 is LOCKED.** CRPS primary, QS99/Brier/MCR guardrails, validation partition. ρ(d) and the gap statistic are **DIAGNOSTICS** | `reference_fao02_locked_eval_framework`; stated in `PREREG.md` §4 |

### A2 · DELIBERATELY CHANGED by this programme — behind a name, default off

**Replaced:** the decoder's twelve `ConvTranspose2d(k=2, s=2)` stages, for `NearestConvUp`.
⭐ **The "default off" mechanism here is the registry itself**, which is stronger than a flag: the
incumbent class is not touched at all, so selecting it is byte-identical to today and reverting is
changing one config value. `state_dict` keys differ (`…reg.weight` → `…reg.conv.weight`), and
`load_state_dict` **refuses a cross-load in both directions** — verified.

⚠️ **What reviewers should not defend:** that the upsample stage must be learned-per-phase. That is
the defect, not a feature.

### A3 · RESPECT WHILE CHANGING — not targeted, breakable in passing

| | |
|---|---|
| **The gate's placement through the autoregressive loop.** EXP-01: *"the body is untouched; this is entirely the gate's placement degrading."* ⇒ all twelve stages change, or the comparison is meaningless |
| **The contractive recurrence.** A 35.8× state collapse over 30 free steps is why `WideMemory` was null and why freezing the cell helps. Nothing here touches it, and nothing should |
| **BatchNorm running statistics**, recalibrated post-training on 32×32 windows and applied to 360×720 — a live lead for the *gate's* dispersion defect, untouched here |
| **Emit-path shapes.** `forward` is inherited verbatim and output shapes are drop-in, so `torch.cat`, the composer and the scorer's view of the cube are unchanged — asserted by the shape test and exercised for real by smoke |
| **EXP-01's committed instruments.** `preflight.py`, `smoke.sh`, `launch_bakeoff.sh` and its sentinels are **not edited**; EXP-02 has siblings |

---

## B · The standing harness — what already exists (audited, ~85% in place)

| mechanism | found at | state |
|---|---|---|
| **Extension seam (OCP)** | `registry.py:45` `ARCHITECTURE_REGISTRY`, lazy imports | ✅ used |
| **Name-agnostic contract test** over every registered architecture | `tests/architectures/test_architecture_registry.py`, per `ArchitectureRegistry` CIC §3/§6 | ✅ candidate 7 covered automatically |
| ⭐ **Mechanism-pinning tests** | `tests/architectures/test_candidate_mechanisms.py` — exists *because* a `/falsify guard` pass found **every** EXP-01 candidate's mechanism could be removed, inverted or neutered with the suite green | ✅ **28 collected cases** for candidate 7, after the audit |
| **Independent arm-identity verifier** | `verify_bakeoff.py` step 1 — re-reads `model` from the arm's own config rather than trusting the builder's declaration | ✅ reused unchanged |
| **Per-arm postflight** | `scripts.arm_postflight.audit_arm` via the verifier; 8 tests in `tests/test_arm_postflight.py` (missing/empty artifact, floor-gate token, failed gate, NaN score, missing horizon, support mismatch) | ✅ reused unchanged |
| ⭐ **Real-pipeline smoke** — the only honest footprint measurement; exercises the emit path, the scorer's view of the cube, BN recalibration and the data pipeline at 2 lessons | `smoke.sh` → `smoke_exp02.sh` | ✅ sibling |
| **Device gate** + measured cost projection | `preflight.py` → `preflight_exp02.py`, importing `measure`, `_resolve` and the **measured** anchor `INCUMBENT_ARM_HOURS` | ✅ sibling, anchor imported not restated |
| **Shared scheduler**, one heavy job at a time, verify-after-every-arm, resumable | `2026-08-18_lesson_curve_dossier/tools/run_queue.sh` via `RES_DIR`/`VERIFIER` | ✅ reused unchanged |
| **Entropy lock / determinism** | `ReproducibilityGate.lock_entropy`; `G-25` asserts two forwards bit-identical | ✅ |
| **Artifact identity** | `vehicle.py` `F6` + Amendments 1, 4, 5, **6** | ✅ amended for candidate 7 (`C7-1`) |
| **Data/config provenance** | `C-61` `StaleCacheError` | ✅ live |
| **Fast cheap readout before expensive** | `gf_phase_signature.py` — CPU, 283 s, selected the arm before any GPU time | ✅ built |
| **Negative-result discipline** | `07_experiment_log` here and in EXP-01's dossier; `PREDICTION_LEDGER.md` (120 W / 58 R) | ✅ |
| **Run discipline** | `setsid` for long runs, `WANDB_MODE=offline`, artifacts per-arm, logs isolated | ✅ |
| **Locked evaluation baseline** | `lodestar_score.py` (FROZEN) + `score_v2_horizons.py` at h1…h36 + oracle ceiling + retention | ✅ ⚠️ see `C`4 |

---

## C · Gaps this programme must close — these ARE the pre-experiment work

1. ⛔⛔ **The candidate is not in a released wheel.** Every arm trains against the **installed**
   `views_hydranet`; all six EXP-01 candidates were in the published package before their arms ran.
   ⇒ **0.1.8 must ship.** This is the gap the three bypasses were routing around.
2. ✅ **CLOSED 2026-10-30.** The `/falsify guard` audit ran in a clean-context subagent and returned
   **FALSIFIED — 13 survivors**: every guard asserted a type, a shape or a parameter count and none
   asserted what the code computes, so the defect could be reinstated verbatim with the suite
   green. All 13 are closed and each fix is mutation-verified; the instrument was re-run and every
   number reproduced. ⇒ `07` EXP-02b. ⭐ **The rule it leaves behind: a guard that asserts a proxy
   is not a guard on the mechanism.**
3. ⛔ **The four `*fullzero*` arm dirs do not exist.** `make_arch_arm.py --lessons 300` builds them;
   smoke builds `*tinyzero*` dirs, which are **different directories**.
4. ⚠️ **No σ̂ for ΔAP at h1 or h36.** `C-357` has fired twice on arms sized against a σ̂ that does not
   exist for the quantity measured. ⇒ `PREREG.md` §6 claims **no significance**; the verdict is a
   **sign test across two seeds**, which is what n=2 supports.
5. ⚠️ **`C7-1`'s amendment is unproven end to end.** The predicate is verified; the real load happens
   the first time `stage_roughness.py` reads the arm's checkpoint.
6. ⚠️ **EXP-02 has no read-out.** `verify_bakeoff.py:118` derives completion from EXP-01's
   `ARCHES × SEEDS`, so a 4-arm queue never reports COMPLETE and the comparison table never prints.
   Identity and postflight — the parts that **stop the queue** — are per-arm and unaffected.
7. ⚠️ **The arm builder is co-owned** (`C7-6`): EXP-02's three `_ARCH_TAG` rows live in EXP-01's
   dossier tool. Sanctioned by that tool's own error message, but it is now shared and is recorded
   here as such rather than silently.
8. ⚠️ **The data cache is stale and the refusal is live** (`C-61`). Resolved by a **deliberate,
   logged** refetch with its vintage caveat declared in `05` — ⛔ never by a flag.
9. ⚠️ **Odena et al. 2016 is not held** and has been cited in four places (`01` §3).

---

## D · PRE-FLIGHT CHECKLIST — must be green before the FIRST arm

- [x] the mechanism implemented, and the numerically delicate part (shape/phase) tested first
- [x] registered via the extension seam, **not** by modifying the dispatcher
- [x] the incumbent untouched and byte-identical when the candidate is not selected
- [x] fail-loud guards: unknown name, non-conforming stage, channel mismatch, device
- [x] the cheap readout ran and **selected the arm** before any GPU time (`G-F`)
- [x] ⛔ **the selection is a SENTINEL, not four transcriptions.** `G-F` writes `results/SELECTED_ARM`; `launch_exp02.sh`, `smoke_exp02.sh`, `preflight_exp02.py` and `verify_exp02.py` all READ it and refuse without it, and the launcher applies the same staleness rule to it as to `SMOKE_OK`. A `/falsify guard` audit measured the cost of the four hand-typed copies that stood here before: one word changed in the launcher and four arms trained, scored, and were never identity-checked — gate `F1` reading as having run when it had not. A refuted `G-F` run now DELETES the sentinel.
- [x] parameter counts asserted **relationally**, with no literal
- [x] ⛔ three process bypasses removed; no `PYTHONPATH`, no gate-set fork
- [x] `ruff` clean; **2409 passed, 0 failed**; CIC-drift and registry-contract tests green
- [x] a pre-registration exists with falsifiers, a verdict rule, and endpoints **h1 and h36 — never h18**
- [x] ⛔ **`/falsify guard` ran independently and its 13 survivors are closed** (gap C2)
- [ ] ⛔ **0.1.8 released; the candidate resolvable from site-packages with no `PYTHONPATH`** (gap C1) — **blocker**
- [ ] the four `*fullzero*` arm dirs built and single-variable (gap C3)
- [ ] `gf_phase_signature.py` → `SELECTED_ARM`; `preflight_exp02.py` → `PREFLIGHT_OK`; `smoke_exp02.sh` → `SMOKE_OK`
- [ ] the refetch decision taken and logged with its vintage caveat (gap C8)
- [ ] `C7-6` / `C7-7` routed through `register-risk` (gap C7)
- [ ] an EXP-02 read-out exists (gap C6)

⛔ **`status` must refuse to call this programme ready while any unchecked box remains.**

---

## E · Rules of engagement

- **One variable at a time.** `model`, and nothing else.
- **Pre-register, then run.** Hypothesis, falsifiers and the verdict rule committed before numbers.
- **Cheap readout before expensive.** `G-F` cost 283 s of CPU and chose between three remedies.
- **Falsifier honesty.** A pre-registered falsifier that fires kills the hypothesis. ⛔ `PREREG.md`
  §6 device 3: if a negative arrives and I want to argue the test was inadequate, that argument is
  a `W-` row against myself **first**.
- ⛔ **A gate that refuses is information, not an obstacle.** Three refusals were correct and I built
  a bypass for each. The rule this dossier adds: **a refusal is either respected, or amended in
  writing before the run with its reasoning — never flagged past in flight.**
- **Improvement by representation, not by masking.** The remedy removes a mechanism; it adds no
  clamp, cap or post-hoc correction.
