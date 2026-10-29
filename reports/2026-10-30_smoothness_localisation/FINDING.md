# Q1/B2b — WHERE the body's spatial smoothness comes from

**2026-10-30.** Script `3d354f2` (`stage_roughness.py`), committed before it ran (D-5). Output
`stage_roughness.out`. **6.6 min CPU, zero GPU**, peak RSS as printed. **All 5 gates PASS.**
⛔ **All four of my pre-registered predictions were refuted** — §5.

---

## ⭐⭐⭐ THE ANSWER IN THREE LINES

**1 · The detail exists and the decoder throws it away.** The encoder's full-resolution features
are **ROUGHER than the truth field**: ρ(1) = **+0.443** (`enc_conv0`) and **+0.314** (`e0s`, the
skip) against a truth field at **0.499**. ⇒ ⛔ **The smoothing is NOT the encoder, and it is not
the receptive field.**

**2 · ⭐⭐⭐ Both upsampling stages are `ConvTranspose2d(kernel=2, stride=2)`, which performs NO
spatial mixing at all.** With kernel = stride there is zero overlap, so **every output cell is a
fixed per-channel scalar multiple of exactly one input cell** — a coarse cell replicated into a
2×2 block with four learned numbers. The signature is unmistakable in the measurement:
**ρ(1) = +0.123 but ρ(2) = +0.921** at `up1`, with channel-wise ρ(1) running **−0.725 … +0.719**.

**3 · The final 3×3 conv smooths across that 2×2 phase pattern** and produces the delivered field:
ρ(1) **+0.706** at `dec1` → **+0.958** at `dec_conv4_head1_reg`. ⇒ **The body's smoothness is
manufactured in the decoder, in two stages that cannot create spatial information and one that
averages away the phase pattern they leave behind.**

---

## 1 · What is measured (C-351, every slot)

| slot | value |
|---|---|
| population | one teacher-forced sweep of months **421–469** over the full **360×720** raster; ρ accumulated over the **13 out-of-sample months 457–469** |
| truth | ⛔ **not scored here.** This measures **ACTIVATIONS**, not forecast error |
| estimator | Pearson ρ of lag-`d` **land–land neighbour pairs**, E–W and N–S pooled, per channel; reported as the **channel median with min/max beside it** |
| **inference mode** | ⛔ **TEACHER-FORCED** (`vehicle.py:301-307`), `C-399`. Same loop as `d4_mu_on_the_grid.py` (`W-98`) |
| free parameters | **dropout OFF (primary) and ON**, both run; lag `d` ∈ {1,2,4,8,16,32} (D-4) |
| masking | ⛔ non-land is **zero** in the production raster, so a constant ocean would drive ρ→1. Only pairs where **both** cells are land are scored; at /2 and /4 a block counts as land only if **every** cell in it is land |

---

## 2 · ⭐⭐ The profile, dropout off

| stage | res | ρ(1) | ρ(2) | ρ(4) | ρ(32) | ch min…max at d=1 |
|---|---|---|---|---|---|---|
| `enc_conv0` | /1 | **+0.443** | +0.293 | +0.150 | +0.005 | +0.028 … +0.785 |
| `dropout["e0s"]` ← **the skip** | /1 | **+0.314** | +0.257 | +0.156 | +0.015 | −0.037 … +0.600 |
| `pool0` | /2 | +0.579 | +0.423 | +0.275 | +0.102 | +0.408 … +0.711 |
| `dropout["e1s"]` | /2 | +0.627 | +0.425 | +0.245 | +0.093 | +0.309 … +0.956 |
| `pool1` | /4 | +0.520 | +0.283 | +0.139 | −0.001 | +0.264 … +0.862 |
| `bottleneck_conv` | /4 | **+0.780** | +0.514 | +0.244 | −0.013 | +0.523 … +0.941 |
| `upsample0_head1_reg` | /2 | ⛔ **+0.149** | ⭐ **+0.954** | +0.883 | +0.659 | **−0.624 … +0.744** |
| `dec_conv0_head1_reg` | /2 | +0.764 | +0.928 | +0.810 | +0.367 | −0.006 … +0.970 |
| `upsample1_head1_reg` | /1 | ⛔ **+0.123** | ⭐ **+0.921** | +0.970 | +0.779 | **−0.725 … +0.719** |
| `dec_conv1_head1_reg` | /1 | +0.706 | +0.939 | +0.930 | +0.438 | −0.181 … +0.936 |
| `dec_conv4_head1_reg` | /1 | ⭐⭐ **+0.958** | +0.929 | +0.839 | +0.247 | +0.949 … +0.968 |

⚠️ **ρ(1) is NOT comparable across resolutions** — at /2 one lag is two full-resolution cells
(`W-51`). The rows to compare are the **/1 rows against each other**, and that comparison is the
finding.

---

## 3 · ⭐⭐⭐ The mechanism, read off the architecture after the measurement pointed at it

`HydraBNrecurrentUnet_06_LSTM4.py:216-226`:

```python
self.upsample0_head1_reg = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2, ...)
self.upsample1_head1_reg = nn.ConvTranspose2d(base * 2, base,     2, stride=2, ...)
```

⭐⭐ **kernel = 2, stride = 2 ⇒ the kernel windows do not overlap.** Each coarse input cell `x`
maps to a 2×2 output block `[w₀₀·x, w₀₁·x, w₁₀·x, w₁₁·x]`. ⇒ **No spatial mixing occurs.** The
stage does not interpolate; it **replicates and rescales.**

Two consequences, both measured:

- ⛔ **Adjacent output cells are deterministic multiples of the same number**, so their rank
  correlation reflects the sign and ratio of two fixed weights — not the field. Where
  `w₀₁/w₀₀ < 0` the channel is **anti**-correlated at lag 1: the measured channel floor is
  **−0.725**.
- ⭐ **Lag 2 compares the same phase in neighbouring blocks**, which is the coarse field's own
  lag-1 structure. ρ(2) = **+0.921**, and the curve stays above **+0.65 out to d = 32**.

⇒ ⭐⭐ **This is the classic transposed-convolution checkerboard (Odena et al. 2016) in its purest
form** — and the `kernel = stride` case is worse than the uneven-overlap case Odena describes,
because here there is no overlap to average at all.

### ⭐⭐ And at lag 2 the skip is swamped completely

| | ρ(1) | **ρ(2)** |
|---|---|---|
| `e0s` — the skip | +0.314 | **+0.257** |
| `up1` — the upsampled branch | +0.123 | **+0.921** |
| `dec1` — the post-concat 3×3 conv | +0.706 | **+0.939** |

⇒ ⛔ **At lag 2, `dec1` tracks the upsampled branch (0.939 vs 0.921) and bears no resemblance to
the skip (0.257).** The skip arrives at full resolution carrying genuine 1-cell detail and the
post-concat convolution does not use it.

⚠️ **Stated honestly: I predicted this at ρ(1) and ρ(1) refuted it** (`W-117`). The ρ(2) reading
is a **new observation from the same run**, not a rescued prediction — switching statistics after
the fact is the `W-51` sin and is not claimed as a win here.

---

## 4 · ⇒ What this means for Simon's two Q1 follow-ups

### ⭐ "Do we understand why?" — **YES, now.**

The body is smooth because the decoder reconstructs a full-resolution field from a **/4**
bottleneck using two stages that **cannot create spatial information**, and the detail that would
supply it — present at full resolution in `e0s`, rougher than the truth field — is not used.

### ⭐ "Can we do anything?" — a concrete lever, NOT proposed as work here

The standard fix for exactly this defect, from the paper that named it: **replace
`ConvTranspose2d(k=2, s=2)` with `Upsample(scale_factor=2) → Conv2d(3×3)`**, or widen the
transposed kernel to **4** with stride 2 and padding 1 so the overlap is even.

⚠️ **This is an ARCHITECTURE change, not an emitting family** — so `D-3`'s scope lock does not
cover it, but it is a retrain and therefore has a cost and needs Simon's word.

⚠️ `views_hydranet/architectures/anti_aliased_pool.py` **already exists** in the repo, and ⛔ **it
addresses the wrong side** — aliasing on the *downsampling* path. This finding is about
*upsampling*. ⇒ **Do not reach for it as the fix.**

### ⛔ And three things this measurement shows are NOT the solution

| non-solution | why, measured |
|---|---|
| **A bigger receptive field / more context** | the encoder's full-res features are **rougher than truth** (0.314–0.443 vs 0.499), and prior art puts the *effective* RF far below the theoretical one (interior `rel_abs_diff` **0.0006** vs **0.10** pooled, a **142×** gap, `2026-08-23_state_range_dossier`) |
| **A sharper loss / a structure penalty on the output** | the output stage is already the smoothest thing in the network (ρ(1) **0.958** after `dec1`'s 0.706); penalising the output asks the final 3×3 conv to undo a phase pattern it exists to remove |
| **Anti-aliased pooling** | wrong side of the network, per above |

---

## 5 · ⛔ Four predictions, four refuted — and two were badly posed

| | declared | measured | verdict |
|---|---|---|---|
| **P-12** | input ρ(1) ≈ 0.5, rising **monotonically** through encoder → bottleneck | ⛔ I **never hooked the input**, so half of it was untestable; and the profile is **non-monotone** (0.443 → 0.314 → 0.579 → 0.627 → 0.520 → 0.780) | ⛔ **WRONG**, `W-114` |
| **P-13** | `up1` ρ(1) exceeds `e0s` ρ(1) by **≥ +0.15** | **+0.123 vs +0.314** — LOWER by 0.19 | ⛔ **REFUTED IN SIGN**, `W-115` |
| **P-14** | bottleneck ρ(1) > **0.90** at /4 | **+0.780** | ⛔ **WRONG**, `W-116` |
| **P-15** | `dec1` closer to `up1` than to `e0s` in ρ(1) | \|0.706−0.123\| = **0.583** vs \|0.706−0.314\| = **0.392** → closer to the **skip** | ⛔ **WRONG as stated**, `W-117` |

⭐ **The pattern, and it is the useful part:** every prediction assumed the upsampling would
*blur* — i.e. raise ρ(1). ⛔ **It phase-stripes instead**, which lowers ρ(1) while raising
everything else. **I had the right suspect and the wrong mechanism, so every statistic I chose to
catch it pointed the wrong way.** `P-12` additionally names a quantity the script does not
collect, which is a prereg defect, not a measurement one.

---

## 6 · ⚠️ Limits

- ⛔ **TEACHER-FORCED** (`C-399`). Says nothing about the delivered autoregressive regime.
- ⛔ **This LOCALISES; it does not adjudicate.** A smooth internal representation may be correct.
  The output is *which stage* — which is what was unknown.
- ⛔ ρ on activations is **not comparable to d12b's 0.926**, a 336-month temporal mean of `log mu`
  on land cells (`W-51`). **No copy gate is claimed against it**, and the closeness of `out`'s
  **0.958** to it is a consistency remark, not a reproduction.
- ⚠️ **Dropout ON depresses every ρ**, as expected from `d12c_draw_noise.out` (11.2% of the body
  field's spatial variance is our own 4-draw MC noise). The ordering and the phase signature are
  unchanged: `up1` ρ(1) **+0.136** / ρ(2) **+0.836**; `out` ρ(1) **+0.872**.
- ⚠️ /2 and /4 land masks require the **whole block** to be land, so those rows are interior
  statistics and drop coastal cells.
- **n=1 checkpoint, the H1-reg (`sb`) head only, one sweep per condition, no σ̂.**
