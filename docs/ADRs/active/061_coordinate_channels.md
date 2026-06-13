# ADR-061: Coordinate Channels for Spatial Grounding (CoordConv)

**Status:** Accepted
**Date:** 2026-06-13
**Deciders:** Simon (chair), Claude (pair)
**Depends on:** ADR-060 (Static & Architectural Input Channels), ADR-027 (autoregressive inference)
**Supersedes:** ADR-029 (Geographic Anchors — jointly with ADR-060)
**Related:** ADR-056 (scheduled sampling — complementary lever), ADR-046 (feature lifecycle)
**Dossier:** `reports/2026-06-11_coordinate_grounding_dossier/`

---

## 1. Context

**Problem.** Tonight's bounded hurdle-NB sweep (6 runs) diagnosed a *bounded-but-drifting* rollout. The
forensics are consistent across seeds:
- the onset **gate over-fires 4–16×** and **worsens through training** (worse at higher `pos_weight`);
- the magnitude head over-predicts (~40–50× on training windows) and **rises** over lessons;
- the autoregressive rollout **blooms localized "blobs"** of predicted conflict in places ground truth
  never has it (full-horizon MCR 2.5–13, median-ratio 9–47).

This reframes C-113: the old global **explosion to ~1e17** is gone; the distributional head turned a
divergence into a **bounded, localized, slowly-blooming hotspot**. What remains is not an explosion —
it is **spatial over-firing + exposure bias**: the model hallucinates growing conflict in
structural-zero regions.

**Assumption no longer true.** That a translation-invariant ConvLSTM U-Net is appropriate for a
**spatially near-degenerate** process. Conflict is structurally absent from the vast majority of cells
for domain reasons (no people, no roads, no cities). A standard CNN is translation-invariant *by
construction*, so it is **largely blind to absolute position** (Liu et al. 2018 showed conv fails to
generalize the coordinate transform; it leaks only *limited* position via zero-padding borders — C5). It
cannot easily learn "this location is structurally peaceful," only "this neighbourhood *looks* like
patterns that have fired." The diagnosed spatial over-firing is the signature of that blindness.

**Urgency.** The hurdle-NB head **bounded** the explosion (the hard part), exposing spatial grounding as
the next lever. The bound is recent and the failure is now visible; we attack it before adding any other
moving part.

---

## 2. Decision

**Statement.** *We will make HydraNet coordinate-aware by injecting **static, geometry-derived
coordinate channels** at the model input **and** at the final full-resolution feature map — never
learned, never predicted — so the network can condition on absolute position.* This is the first
concrete instance of the **Static / Architectural** channel class defined in **ADR-060**.

The locked choices:
- **Mechanism.** CoordConv-style **static coordinate channels** (Liu et al. 2018) — *not* learned
  positional embeddings and *not* attention.
- **Two injection points.** (a) the model's first-conv **in-channels 3→5**; and (b) **raw** coordinate
  channels concatenated onto the **top-level full-resolution skip tensor** feeding the final decoder
  layer. The per-pixel decision is made at full resolution; injecting only at the input risks the
  absolute signal washing out through down/up-sampling.
- **Derived from geometry.** Computed from the full tensor's grid index, **not** the dataframe
  `row`/`col` features; *derive-then-slice* gives global alignment by construction (ADR-060 I4).
- **Normalization hardcoded to `[-1, 1]`** (the canonical CoordConv range; zero-centred). **Not** a
  config knob and **not** a sweep axis — promote to config only if proven necessary.
- **Static/exogenous (ADR-060 §2.1).** Re-injected as **true** values at every autoregressive step;
  `output_channels` stays at **3**; coordinates never enter the prediction frame.
- **Toggle.** On/off via a config / model-variant flag; **bit-identical** to baseline when disabled
  (ADR-060 I5).

- **In-scope:** the coordinate channels, their two injection points, and the validating experiment.
- **Out-of-scope:** static covariates (population / terrain / ocean — *enabled* by ADR-060, a separate
  experiment); **per-layer CoordConv / coordinate attention** (escalation — Ding & Gao 2025);
  **Fourier-feature encoding** of the coordinates (escalation if the smooth proxy underdelivers —
  Tancik et al. 2020); scale as a config knob.

---

## 3. Rationale & Integrity Impact

- **Logic.**
  - *Liu et al. 2018* — convolution provably cannot solve the coordinate transform; coordinate channels
    grant absolute position at the cost of a couple of lines, and **reduce mode collapse** in generative
    settings (our blob-bloom is a spatial mode-collapse analog).
  - *El Jurdi et al. 2021* — CoordConv in a **U-Net** stabilizes training and evades local minima when a
    pixel-wise base loss is augmented with **anatomically-constrained spatial-prior regularizers**
    (size/shape terms), and is inert with a plain loss. **We flag the analogy as partial:** their "prior"
    is an *added spatial regularizer* whose two-term interchange CoordConv stabilizes; our hurdle-NB is a
    *distributional likelihood*, not an added spatial term, so there is **no equivalent interchange**.
    This is a **plausibly analogous regime** (a non-vanilla, instability-prone loss where stabilization
    *may* help), **not "exactly" it** — and the analogy could fail *because* our prior is distributional,
    not spatial. That disanalogy is part of why **§5's experiment, not the literature, is the real test**;
    it is also a reason the head only became a *candidate* moment now, not a guarantee.
  - *Domain* — conflict base rates are near-zero across most of the grid for structural reasons; a
    position-aware model can learn that prior, a translation-invariant one cannot. Coordinates are the
    **minimal, principled** first instance of the ADR-060 Static/Architectural class.
- **Complementary lever.** ADR-056 (scheduled sampling) attacks the **exposure-bias** half of the
  blob-bloom (the model amplifying its own feedback); coordinates attack the **spatial-grounding** half
  (firing in the wrong *places*). Orthogonal and composable — this ADR isolates the coordinate lever.
- **Fortress State.** Coordinates are an **architectural property** (always derivable; aligned by
  construction, ADR-060 I4), not a fragile data feature; and they are a **static anchor the rollout
  cannot corrupt** (re-injected true every step, I3) — a fixed spatial reference at step 36 as at step 1.
- **Fail-Loud.** Governed by ADR-060 I1 (a coordinate channel appearing in a target list raises).

---

## 4. Consequences

### ✅ Positive (Benefits)
- Absolute **spatial grounding** the architecture currently lacks.
- A **static rollout anchor** that cannot drift.
- **Minimal** (+2 input channels; a couple of lines) and the first concrete exercise of the ADR-060 seam.

### ⚠️ Negative (Costs)
- Breaks **pure translation invariance** — *intended*: conflict is not translation-invariant.
- +2 input channels (negligible compute / memory).
- **Smooth-proxy risk** — networks fed raw low-dimensional coordinates are biased toward
  low-frequency (smooth) functions of position (Tancik et al. 2020, *a coordinate-MLP / NTK result*); by
  analogy, plain `(row,col)` channels **may** under-capture the **sharp** geography of where people live,
  so coordinates could under-deliver or null ambiguously. The escalations (covariates; Fourier features;
  coordinate attention) are **named and deferred**, not adopted.
- **Shortcut risk** (from ADR-029) — the model may overfit to geographic *averages* and stop
  learning dynamics. Watched via the rollout biopsy (does it still respond to inputs, or just paint the
  base rate?).
- **Channel dilution** (from ADR-029) — two static channels could drown the sparse conflict signal.
  Mitigation available if observed: **input dropout on the context (static) channels** (forces the model
  to keep using the conflict signal while acknowledging geography). Held in reserve — *not* enabled in
  the one-variable run.

---

## 5. Validation

*Pre-registered — see the dossier's `05_analysis_plan.md` + `07_experiment_log.md`.*

> **Acceptance semantics.** This ADR is **Accepted** as the committed design-of-record. The experiment
> below is its pre-registered **falsifier**, not a precondition for acceptance: **validation keeps it;
> falsification supersedes it** (escalate to the covariate ADR). The contract it instantiates (ADR-060)
> is independent of the outcome.

- **One-variable test.** The bounded hurdle-NB **S1** config (θ=1.0, `pos_weight`=10, frozen balancer,
  scheduled sampling off, 40 lessons), **+ coordinate channels, nothing else**, vs the baseline just run.
- **Referee.** The *same* forensics: the classification-gate "detection bias" plot, the autoregressive
  rollout biopsy, and the MCR readout (step-1 + full).
- **Pre-registered prediction** (if coordinates are the lever): the gate event-ratio **stops climbing**
  to 4–16×; the rollout blobs **stop blooming in structural-zero regions** specifically; **FULL MCR
  moves toward 1**.
- **Failure mode.** If the blobs **persist or merely relocate**, or the gate still floods, coordinates
  are *not* the lever ⇒ this ADR is **superseded** by the covariate escalation (ADR-060 enables) —
  **not** a return to loss-level tinkering.
- **Invariants.** Inherits ADR-060 I1–I6: `output_channels` unchanged (I1), coords absent from the
  frame (I2), re-injected true every step (I3), aligned by construction (I4), bit-identical when off
  (I5), and flipped in sync under spatial augmentation (I6).

---

## 6. Implementation Notes

- **Location.** the network class (`HydraBNUNet06_LSTM4` — the two injection points; output head
  unchanged); `HydraNetConfig` + validator (the toggle + the static-channel block, per ADR-060);
  `VolumeHandler` / `VolumeSampler` (derive coords on the full grid, slice with the dynamic window —
  ADR-060 I4); `FeatureScaler` ignores coords (pre-normalized, never `log1p`'d).
- **To confirm during the build (open questions, not blockers):** (Q1) does the model predict every
  input channel today (`output_channels == input_channels`)? → fixes how output stays 3 when input goes
  to 5; (Q2) is there a clean top-level full-resolution skip tensor to concat raw coords onto?; (Q3)
  where are windows sliced, so coords *derive-then-slice*?; (Q4) do coords bypass `FeatureScaler`
  entirely?; (Q5) does a network/model CIC exist or must one be created?
- **Replaces a tactical mitigation (from ADR-029).** Coordinate grounding aims to reduce/replace the
  existing hidden-state freeze (`execute_freeze_h_option`, the "Brain Locks") — a fixed positional
  reference should lessen the need to manually freeze parts of the hidden state to preserve the "map of
  the world." Worth measuring whether the freeze can be relaxed once coordinates are in.
- **First covariate escalation.** If coordinates underdeliver, the cheapest next rung is the **land/water
  mask** (derivable from `priogrid_gid > 0`, per ADR-060) — it kills "coastline hallucination" (ADR-029)
  with a near-architectural channel, before any fetched raster.
- **Placement is our own reasoning, not paper-validated.** El Jurdi injects CoordConv at the *first conv
  of each encoder block* (the encoding path); Ding's attention sits at the ResNet50 *bottleneck*. **No
  cited paper uses our input + top-skip (decoder) scheme** — it follows from "the per-pixel decision is
  made at full resolution" (§2), not from a published result. El Jurdi backs the *prior-loss synergy*,
  not the placement; the build should treat placement as a design choice to validate, with El Jurdi's
  per-block encoder injection and Ding's attention as fallbacks if input+top-skip underdelivers.
- **References:** the five papers below; ADR-060 (the contract), ADR-056 (complementary lever), the ZINB
  distributional-head dossier (loss context), and the coordinate-grounding dossier.

## Literature

> All five papers are registered in the library (verified against their `_meta` sidecars + curated key
> passages, 2026-06-13); cited below by their `papers/<filename>.pdf` path.

| ID | Citation | Relevance |
|----|----------|-----------|
| C1 | Liu, R., Lehman, J., Molino, P., Petroski Such, F., Frank, E., Sergeev, A. & Yosinski, J. (2018). "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution." *NeurIPS.* `papers/Liu2018_CoordConv.pdf`. | **Foundational.** Conv fails to generalize the coordinate transform (largely position-blind); coordinate channels fix it; reduces GAN mode collapse. The mechanism we adopt. |
| C2 | El Jurdi, R., Petitjean, C., Honeine, P. & Abdallah, F. (2021). "CoordConv-Unet: Investigating CoordConv for Organ Segmentation." *IRBM.* `papers/ElJurdi2021_CoordConvMedSeg.pdf`. | **Closest analog (partial).** CoordConv in a U-Net stabilizes training / evades local minima *under prior-based losses*, and is *"insignificant if there is no prior constraint problem."* But their "prior" = an added *spatial* shape-regularizer; ours = a *distributional* likelihood — a **plausibly analogous, not identical, regime** (see §3). Placement is per-encoder-block, not our input+top-skip (see §6). |
| C3 | Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T. & Ng, R. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *NeurIPS.* `papers/Tancik2020_FourierFeatures.pdf`. | **The smooth-proxy risk + escalation.** Coordinate-MLPs have a "rapid frequency falloff" (NTK), preventing them from representing high-frequency content; Fourier-encoding the inputs fixes it. *An MLP/NTK result* — applied here by analogy to name our §4 risk + the principled escalation. |
| C4 | Ding, J. & Gao, S. (2025). "GCA-ResUNet: Medical Image Segmentation Using Grouped Coordinate Attention." *arXiv:2512.23990.* `papers/Ding2025_GCAResUNet.pdf`. | **Escalation path.** Direction-aware coordinate *attention* as a plug-and-play U-Net module addressing CNN locality — if input+top-skip coordinates underdeliver. |
| C5 | Islam, M. A., Jia, S. & Bruce, N. D. B. (2020). "How Much Position Information Do Convolutional Neural Networks Encode?" *ICLR.* `papers/Islam2020_PositionEncoding.pdf`. | **Nuance (from ADR-029).** CNNs *do* implicitly encode absolute position — delivered by **zero-padding at the borders** and propagated inward, but **interfered with by semantic content**. Qualifies Liu's "cannot represent position": the implicit signal exists yet is indirect and border-anchored, motivating *explicit* coordinate injection over reliance on the padding artifact. |
