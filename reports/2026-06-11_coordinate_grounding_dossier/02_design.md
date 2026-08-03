# 02 — Design (the locked design)

**Date:** 2026-06-11 · This is the source of truth for the build. It mirrors **ADR-061 §2** and obeys the
**ADR-060** Static/Architectural contract. **Decided; not re-opened** (brake word CIRCLE).

## The change
Two static, geometry-derived coordinate channels, on the **bounded hurdle-NB S1 config, nothing else.**

### Mechanism
CoordConv-style **static coordinate channels** (Liu et al. 2018). **Not** learned positional embeddings;
**not** attention.

### Two injection points
1. **Input** — the first-conv **in-channels 3 → 5** (the three conflict histories + row + col).
2. **Top-skip** — **raw** coordinate channels concatenated onto the **top-level full-resolution skip
   tensor** feeding the final decoder layer. The per-pixel decision is made at full resolution; injecting
   only at the input risks the absolute signal washing out through down/up-sampling. *Raw* coords (not
   the already-convolved skip features) so the output layer sees clean absolute position.

### Source — derived from geometry
Computed from the **full** tensor's grid index `(i, j)` over the 180×180 grid — **not** the dataframe
`row`/`col` columns. *Derive-then-slice*: build coords for the whole grid, then slice with the **same**
window indices as the conflict channels ⇒ global alignment **by construction** (ADR-060 I4). This makes
coordinate-awareness an **architectural property**, not a data feature that can be misaligned or dropped.

### Normalization
**Hardcoded `[-1, 1]`** (canonical CoordConv; zero-centred — plays well with conv init / BatchNorm). Not
raw (row≈87–267 would swamp init/BN), not `[0,1]` (carries a constant 0.5 bias). **Not** a config knob and
**not** a sweep axis — promote to config only if proven necessary.

### Lifecycle — static/exogenous (ADR-060 §2.1, I1–I3)
- **Never predicted, never learned.** Injected at train *and* inference.
- **Re-injected as true values every autoregressive step** — a static anchor the rollout cannot corrupt.
- **`output_channels` stays 3.** Coordinates are never in `regression_/classification_targets`, never
  inverse-transformed, never in the prediction frame.

### Toggle
On/off via a config / model-variant flag. **Bit-identical** to baseline when disabled (ADR-060 I5).

## Out of scope (escalations — named, deferred, not adopted)
- **Static covariates** (population / urban / terrain / ocean) — *enabled* by the ADR-060 seam; the
  pre-registered escalation if coordinates underdeliver (`05` failure mode). The richer, sharper base-rate
  signal — but a separate one-variable experiment.
- **Fourier-feature encoding** of the coordinates (Tancik et al. 2020) — escalation for the smooth-proxy risk.
- **Per-layer CoordConv / coordinate attention** (Ding & Gao 2025) — escalation if input+top-skip is too weak.
- **Scale as a config knob.**

## Open questions to confirm during the build (not design re-opens)
- **Q1** Does the model predict every input channel today (`output_channels == input_channels`)? → fixes how output stays 3 when input goes to 5.
- **Q2** Is there a clean top-level full-resolution skip tensor to concat raw coords onto?
- **Q3** Where are windows sliced, so coords *derive-then-slice* (ADR-060 I4)?
- **Q4** Do coords bypass `FeatureScaler` entirely (derived in-model, never `log1p`'d)?
- **Q5** Does a network/model CIC exist or must one be created?
