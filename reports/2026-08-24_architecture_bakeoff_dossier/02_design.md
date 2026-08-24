# 02 — Design: the six candidates (2026-08-24)

## The incumbent, measured

`HydraBNrecurrentUnet_06_LSTM4`: **905,289** parameters.

| block | params | share |
|---|--:|--:|
| ConvLSTM (4 cells × 4 hidden channels) | **4,160** | **0.5%** |
| encoder + bottleneck | 98,080 | 11% |
| decoders (3 heads × reg+cls) | **803,049** | **89%** |

Forward path, traced: input `x` (3 dynamic ⧺ 0 static) drives four ConvLSTM cells; then
`x = cat([x, hs_1..hs_4])` enters `enc_conv0` at full resolution → `e0s`; `pool0` → `enc_conv1` →
`pool1` → bottleneck at **¼ resolution**; each of six decoder paths upsamples twice, taking `e1s` at ½
and `e0s_topskip` at full. `e0s_topskip == e0s` because `static_channels: []`.

**Two observations the numbers force.** The model is overwhelmingly a **per-frame decoder with a very
small recurrent memory** — and that memory is what must carry placement across 36 autoregressive steps.
And the only full-resolution path from input to output is a **single conv block** (`enc_conv0 → e0s →
dec_conv1`).

## Candidates

Each is one file in `views_hydranet/architectures/`, registered in the `choose_model` registry, selected
by `'model':` in `config_hyperparameters.py`. **Everything else in the config is held identical to the
`fullzero_*` controls.**

| # | name | change | params vs incumbent | evidence |
|---|---|---|---|---|
| 1 | **AntiAliasedPool** | both `MaxPool2d(2,2)` → `MaxBlurPool` (dense max, then blur, then subsample) | **+0** | `Zhang2019_AntiAliasedCNN` |
| 2 | **DynamicTopSkip** | raw-concat the 3 dynamic input channels onto `e0s` feeding every `dec_conv1` | **+0.6%** | ADR-061 seam, dynamic content |
| 3 | **FiLMSkip** | same information as **learned modulation** — a small conv predicts per-channel (γ, β) from the dynamic input, applied to `e0s`; **zero-init, so it starts as the identity** | **+0.2%** | C-230's "learned modulation, not raw concat" |
| 4 | **ShallowPool** | drop `pool1`; bottleneck at **½** not ¼; `bottleneck_conv` dilated to hold receptive field | **−16.3%** | DeepLab atrous; `Dumoulin2016` |
| 5 | **DualStream** | a parallel full-resolution stream (no downsampling) fused into the top-skip before each head | **+1.9%** | `Sun2019_HRNet` (lite) |
| 6 | **WideMemory** | ConvLSTM state widened 32 → 128 (16 per cell, not 4); conv stack unchanged | **+6.0%** (LSTM ×10.6) | the 0.5% measurement above |

### Why these six and not others

(1) is the **cleanest possible experiment in the whole space**: a named mechanism, a remedy with zero
added parameters, and no interaction with any retired seam. If spatial precision is lost at
downsampling, it should move.

(2) and (3) are a **matched pair on one variable — the primitive**. Same information, raw concat vs
learned modulation. C-230 concluded raw concat is the wrong primitive from a test that used *static*
content; this separates primitive from content, which that test could not.

(4) and (5) are two different answers to "stop destroying resolution": remove a downsampling, versus
keep a stream that never had one. (4) is nearly free in parameters; (5) is not.

(6) is the odd one out and deliberately so: it does not touch resolution at all. It asks whether the
thing carrying placement across the rollout has enough capacity to carry anything. Given the state
freeze is our only win, a 4,160-parameter memory is a live suspect.

## The capacity confound — MEASURED, and smaller than assumed

Written first as "(5) and (6) add substantial parameters". **Measured after implementation, that was
wrong**, and the correction is recorded rather than quietly edited:

| candidate | params | Δ vs incumbent |
|---|--:|--:|
| incumbent | 905,289 | — |
| (1) AntiAliasedPool | 905,289 | **+0.0%** |
| (3) FiLMSkip | 907,081 | +0.2% |
| (2) DynamicTopSkip | 910,473 | +0.6% |
| (5) DualStream | 922,217 | +1.9% |
| (6) WideMemory | 959,241 | +6.0% |
| (4) ShallowPool | 757,449 | **−16.3%** |

**No arm is capacity-confounded in the way first feared.** (1) is exactly zero by construction — the
blur filter is a buffer, not a parameter. (5) is +1.9%, not "large", because the fusion returns to
`base` channels so the six decoder paths are untouched. (6) widens the LSTM **10.6×** (4,160 →
44,288) for **+6%** of total parameters, precisely because the memory was such a small share to begin
with — which is the measurement that motivated it.

**(4) is the one asymmetric case**: it has **16% FEWER** parameters, because `upsample0` becomes a 1×1
channel projection. So a (4) **win** cannot be capacity; a (4) **loss** might be. Stated in that
direction, not glossed.

Parameter counts are still reported beside every result.

## What is deliberately NOT changed

Training loop, loss, `output_distribution: nb`, `forecast_composition: soft_gate`, `body_supervision`,
the curriculum, `ss_epsilon_max: 0.0`, the queryset, the partition, the scoring protocol. The **only**
config key that differs between an arm and its control is `model`.

## Evaluation

The full battery already emitted by `score_v2_horizons.py` at **h1/6/12/18/24/30/36**, both sides:

* **gate** — `AP`, `Brier`, `precision_at_k`, `act_pred`/`act_true`/`act_ratio`, `n_false_pos`
* **body** — `crps_all`, `crps_events`, `crps_none`, `size_ratio`, `mcr_all`/`mcr_none`/`pos_mcr`,
  `mag_on_false_pos`, `mag_on_true_pos`

plus the **oracle** (`use_real`) arm per model, which separates *"the model got worse"* from *"the
rollout got worse"* — the distinction that made M45 interpretable. Retention (`free/oracle`) is reported
per arm.
