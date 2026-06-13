# 04 — Roadmap (the epic)

**Date:** 2026-06-11 · Linear sub-issues. One box at a time; do → log → tick → advance. Issue numbers
assigned when the epic is opened on GitHub (this dossier is the design-of-record meanwhile).

## Epic
**Coordinate grounding for HydraNet** — implement the ADR-060 static-channel seam, add coordinate channels
(ADR-061), run the one-variable experiment, decide (ship / escalate).

> **Planning prerequisites (folded in from the 2026-06-13 `/falsify` audit — C-142, C-153, C-154, C-155).**
> The audit confirmed the *design* is feasible (the architecture supports `input>output`; `e0s` is a real
> full-res skip) but the *readiness* was overstated on four counts, now baked into the boxes below. RED
> gates: `tests/test_falsification_epic_planning_readiness.py`.

## Sub-issues (in order)
1. **Static-channel seam (ADR-060 contract) — CROSS-CUTTING, not a localized arch tweak (C-153/P1).**
   Config surface for a `static_channels` block + validator (fail-loud on coord-in-targets, I1). The seam
   must be coordinated across **all** of these touch-points (the falsify P1 finding):
   - **Architecture** — `enc_conv0` in-channels + all **16 LSTM `Wx*` convs** (`input_channels` 3→5); the
     **6 decoder `dec_conv1` top-skips** (concat raw coords sliced from the input).
   - **Inference rollout** (`hydranet_inference.py`) — inject static channels into the **seed-input slice**
     (currently built from `config["features"]`, L246) **and re-append them every autoregressive step**
     (L292 feeds back only the 3-ch prediction — ADR-060 I3).
   - **Training feedback** (scheduled-sampling `_process_sequence`, ADR-056) — the **same** re-injection.
   - **FeatureScaler** — static channels bypass `transformations` (no `log1p`).
   - **Resolve coords-vs-`features`** — decide the mechanism by which a channel *not* in `config["features"]`
     reaches the model input (a dedicated `static_channels` block consumed by the seam), since the input is
     currently assembled *from* `features`.
   - **Explosion-check prerequisite (C-142/P4)** — validate/extend `diagnose_io_gain` /
     `rollout_diagnostics.free_running_attractor` for the hurdle-NB **count-space** output (it is log-space
     and unvalidated for this head); **flag/close C-142 in `03`**. The coords "Bounded?" readout depends on it.
   *Done when:* I1–I6 green, toggle-off **bit-identical** (I5), **and** the explosion-check is validated for
   the count-space head. **ADR-060 → Accepted.**
2. **Coordinate channels (ADR-061).** Derive `(row,col)` over the full grid, normalize `[-1,1]`, inject at
   the input (3→5) and the top-level full-res skip (`e0s`, raw). *Q1 (out==in) and Q2 (clean top-skip) are
   **RESOLVED** by the 2026-06-13 audit — head out-channels are decoupled from `input_channels`, and `e0s`
   (`HydraBNrecurrentUnet_06_LSTM4.py` L451) is the full-res skip. Q3–Q5 remain.*
   *Done when:* range/shape/corner checks pass; flag on/off both run.
3. **Experiment (pre-registration `05`).** The bounded hurdle-NB **S1** config **+coordinates, nothing
   else**, ≥2 seeds. Readout = gate forensic + rollout biopsy + MCR + FAO (`03`). **Prerequisites:**
   - **Disk budget (C-154/P3)** — ~2.5 GB/prediction-dir; ≥2 seeds + diagnostics + a baseline re-run ≈
     **10–15+ GB**; the dev volume is **~97% full**. Pre-run **headroom check + cleanup**; abort if free
     space < budget. (The 6-run sweep already truncated S3_seed4 on disk-full.)
   - **Baseline provenance (C-155/P5)** — pin the comparator = `config_hyperparameters.py` (hurdle_nb) +
     the recorded per-arm env (`HN_THETA_INIT`, pos_weight) + seed + the **C-42 reproducibility lock**;
     **quarantine/align the stale `config_sweep.py` (tobit)** so coords aren't benchmarked against Tobit;
     confirm `feedback_clamp` was **off** (C-151).
   Log to `../RESULTS_LOG.md` and `07_experiment_log`.
4. **Decide.**
   - Validated (prediction in `05` holds, multi-seed stable) → **ship**; ADR-061 → Accepted.
   - Falsified (blobs persist/relocate; gate still floods) → **drop the toggle** (baseline preserved by
     I5) and **escalate to static covariates** (the next instance of the ADR-060 seam) — *not* loss tinkering.

## The two exits (from `00_README`)
- **Ship coordinates** (box 4, validated).
- **Drop coordinates → escalate to covariates** (box 4, falsified). The hurdle-NB baseline is never reverted here.

## Parking Lot (ideas — NOT acted on during the epic)
- **Placement ablation** (input-only vs input+top-skip vs per-encoder-block) — **the highest-priority
  follow-on once coords-on beats baseline**: *no cited paper backs our input+top-skip scheme* (El Jurdi =
  per-encoder-block; Ding = bottleneck), so the placement is our least-supported single choice (review,
  2026-06-13). Keep the first run input+top-skip (per ADR-061), but ablate placement before hardening it.
- **Land/water mask** (derivable from `priogrid_gid > 0`, from ADR-029) — the **cheapest first covariate**
  escalation (kills "coastline hallucination"; near-architectural, no fetched raster). The first rung if
  coordinates underdeliver.
- Static covariates (population/urban/terrain) — the richer escalation, but its own experiment.
- **Input dropout on the context (static) channels** (from ADR-029) — mitigation for channel-dilution /
  the shortcut risk; held in reserve, not enabled in the one-variable run.
- Fourier-feature encoding of the coordinates (Tancik 2020).
- Per-layer CoordConv / coordinate attention (Ding & Gao 2025).
- Relaxing the hidden-state freeze (`execute_freeze_h_option`, ADR-029) once coordinates are in.
- Combining coordinates with scheduled sampling (ADR-056) — the two halves of the blob-bloom, but compose only after each is isolated.
