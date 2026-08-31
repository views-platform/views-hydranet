# 05 — Pre-analysis plan: does the emitted field blur across the rollout? (#301)

**Status: LOCKED 2026-08-31, before any emit.**

## Provenance

`results/` is empty; no cube has been emitted and nothing has been scored. The instrument
(`scripts/field_sharpness.py`) and its tests are committed at `aa19017`, deliberately **before**
this plan, because the tests changed what this plan could sensibly say — see §3.

## 1. Question

**M48** confirmed that clamping the ConvLSTM long-term ("cell") state during free-running improves
gate skill on 4/4 seeds, with the gain rising from exactly 0.0000 at h1 to +0.0591 at h36. **We do
not know why.** M43 refuted the obvious explanation: the drifted state does not leave its
in-distribution range.

The candidate story: the model's own emitted field is spatially *blurrier* than real data, so
feeding it back accumulates a progressively blurrier picture of **where** conflict is — and
placement is what matters (M32/M45: scrambling locations costs 81% of the oracle, thinning events
costs 3%). Clamping would then work by refusing to overwrite the real-data picture with blur.

**H: during free-running, the emitted field loses spatial structure with each step.**

### The premise that does NOT support this, stated so it is not leaned on

C-190 has been cited as having measured the model as a strong low-pass (0.32 → 0.07). It did not.
C-190 is a **Tier 4** entry about the **skip-connection path** being throttled by BatchNorm and
dropout; the figures come from a **synthetic broadband impulse** through a **single forward pass**,
there is no rollout or horizon axis in it, and **the producing code does not exist** in the repo or
its history. Nothing about "the model's predictions are blurrier than reality" is established. That
is why §4 begins by measuring the baseline rather than assuming it.

## 2. The one variable

`feedback_field_transform`: `identity` (the model's own field fed back — ordinary free-running)
versus `use_real` (the real field fed back at every step). Same model, same artifact, same rollout
code, same origins. **Under `use_real`, blur cannot accumulate by construction**, because no
self-generated content ever enters the state.

Comparing the model to truth alone would confound *"the model is blurry"* with *"the model
**becomes** blurrier"*, and only the second is the claim.

## 3. Metrics — corrected by measurement before this plan was written

`tests/test_field_sharpness.py` simulates three failure modes on synthetic fields with known
answers. Measured:

| statistic | perfect | blurred σ=2 | displaced 3 | thinned 50% |
|---|---|---|---|---|
| **`moran_i`** | 0.636 | **0.968** | 0.636 | 0.292 |
| `conc1pct` | 0.0277 | **0.0967** | 0.0277 | 0.0274 |
| `fss_ratio` | 1.000 | 0.897 | **0.267** | 0.876 |

**`fss_ratio` (fss@1 / fss@11) was the planned primary and is NOT a blur detector.** It falls
further for a *displaced* field — identical sharpness, wrong place — than for a heavily blurred one.
Used as intended it would have reported displacement as blur.

**Primary: `moran_i`** — intrinsic to the prediction, so a wrong-place forecast cannot move it. It
rises monotonically with blur, is unmoved by displacement, and falls under thinning, separating all
three modes **by sign**.
**Secondary: `conc1pct`** — also intrinsic. **Rises** under blur; the opposite direction was
predicted before measuring and was wrong, so the measured direction is what is registered.
**Context only: `fss_1`, `fss_11`, `fss_ratio`** — agreement with truth, explicitly not sharpness.

## 4. Design and stages

Seed 42, `sb`, calibration, L=300, `fullzero_fortytwo`'s existing artifact. **Emit only, no
training.** Runners reused unchanged: `run_realism_arms.py` (`--arms identity,use_real
--keep-cubes`) and, at Stage 2, `run_freeze_arms.py` for the `cell` arm.

* **Stage 0 — is there anything to explain?** At **h1**, compare each arm's field to truth. This
  measures the baseline discrepancy C-190 was wrongly assumed to have established.
* **Stage 1 — does it accumulate?** `moran_i` and `conc1pct` across h1…h36, `identity` vs
  `use_real`.
* **Stage 2 — does the clamp slow it?** Only if Stage 1 survives. Adds the `cell` arm.
* **Stage 3 — a second seed.** Only if 0–2 all survive.

**Budget: 1 hour of GPU, hard stop.** This explains an already-shipped result; it gates nothing.

## 5. Harness check, free and load-bearing

At **h1** both arms predict from **real** data — the seed step, before any fed field is used. Their
h1 fields must therefore be **identical**. If they are not, the arms are not what they claim and
every number downstream is void. Checked first, before any curve is read.

## 6. Falsifiers — pre-committed

* **S1** — `identity` `moran_i` does not rise with horizon → **H dead.**
* **S2** — `use_real` rises comparably → the effect is not accumulation through the rollout, and
  **H is dead regardless of what `identity` does.** This is the falsifier that matters most: a rise
  under `identity` alone is exactly what a lazy version of this experiment would report as success.
* **S3** — `moran_i` and `conc1pct` disagree in sign → **UNRESOLVED, published as UNRESOLVED.**
  Selecting the supporting one afterwards is the C-305/C-306 failure, twice on the register.
* **S4** — the difference is present at h1 and flat thereafter → a static property of the model, not
  accumulation. Report as such and **do not attach it to M48.**
* **S5** — `moran_i` falls rather than rises under `identity` → the model goes *quiet*, not blurry
  (the thinning signature). A real finding, but a different one, and it must not be relabelled.

## 7. Prediction, on the record

I expect `identity` `moran_i` to **rise** with horizon and `use_real` to stay flat. I hold this
weakly: the mechanism is a story assembled after the fact from M43/M48, its supposed C-190 support
does not exist, and the last two things I predicted in this programme (the `conc1pct` direction, the
`fss_ratio` discriminator) were both wrong.

## 8. What this cannot establish

It can show *whether* spatial structure degrades through the rollout and *whether* the clamp slows
it. **It cannot show that blur causes the AP loss** — that needs an intervention on sharpness
itself, which is #301's own territory. The write-up must say so rather than letting a correlation
read as a mechanism.
