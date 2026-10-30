# 01 — Literature

**2026-10-30.** Holdings verified against `~/brain/9_library/papers/` by listing it, not from
memory. ⛔ One source is load-bearing and **not held** — §3.

## 1 · Held, and what we take from each

| source | what it is | what we take |
|---|---|---|
| **`Dumoulin2016_ConvolutionArithmetic`** | the reference text on conv/transposed-conv output shapes and overlap | ⭐ The arithmetic behind the defect: with `kernel_size == stride` the transposed convolution's windows **do not overlap**, so output `(2i+a, 2j+b) = W[:,:,a,b] · x[i,j]` and each of the four in-block phases carries its own free weight. It also gives the shape algebra for the two remedies (`k=2` needs one-sided padding after a ×2 upsample; `k=1` needs none) |
| **`Zhang2019_AntiAliasedCNN`** | subsampling breaks shift-equivariance; a low-pass filter between dense max and subsample restores it | ⚠️ Not our mechanism — it is the **downsampling** path, and `anti_aliased_pool.py` already implements it as candidate 1 (INCONCLUSIVE, −0.0084/+0.0008 ΔAP@h18). ⭐ What we take is its **method standard**: a remedy with **zero added parameters**, which is why the dossier calls it *"the cleanest possible experiment in the whole space"* and why `UpsampleConv2` (±0 parameters) is the arm rather than the 3×3 variant (+33.9%) |
| **`Aceituno2025_TemporalHorizons`** | Thm 4.6: minima found on short horizons do **not** generalise to long ones, with the gap scaling `O(e^{λ(T_h−T_l)})` | ⛔⛔ **The prior that predicts this arm loses.** We train at one step (ε=0, teacher-forced) and evaluate at thirty-six, so anything improving the per-frame objective is under no obligation to help at h36 — and EXP-01 measured that it *hurts*, monotonically, in proportion to how much it sharpens that map. This is the reason the arm is framed as a **prospective test of EXP-01's post-hoc finding** rather than as a fix |
| **`MillerHardt2019_StableRecurrentModels`** (cited in EXP-01's log) | a contractive recurrence sits in the stable regime, where stable recurrent models are approximable by truncated feed-forward models | ⚠️ Context for why memory capacity was not the lever (`WideMemory`'s null) and why the **effective** receptive field is far below the theoretical one — which is the measured fact that narrowed `M92`'s search to the local stack |

## 2 · Prior art inside this repository, which outranks the literature here

| | |
|---|---|
| `2026-08-24_architecture_bakeoff_dossier/07_experiment_log.md` | **EXP-01**: 6 candidates × 2 seeds, nothing promoted, `ShallowPool` REJECT, and the monotone horizon gradient (12/12 negative at h30 **and** h36, `2⁻¹²` under independent signs) |
| `2026-08-23_state_range_dossier/07_experiment_log.md:63` | the **effective** receptive field is not the theoretical one — interior `rel_abs_diff` **0.0006** against **0.10** pooled, a **142×** gap, because the recurrence **contracts** (65.6 → 1.8 over 30 free steps) |
| `2026-10-12_underfire_diagnosis/d12b_curve_readout.out` | the model's `log mu` field at ρ(1) **0.926**; truth at **0.638** (336-mo mean) / **0.499** (36-mo window) |
| `M92` / `M93` | the per-stage localisation, and the correction that the right comparator is **single-month `log1p` truth at +0.344**, not either temporal mean |

## 3 · ⛔⛔ GAPS TO FETCH — and one is a live citation problem

**`Odena, Dumoulin & Olah (2016), "Deconvolution and Checkerboard Artifacts"` (Distill) — NOT HELD.**

⇒ ⛔ **This is the primary source for the mechanism the entire programme is built on**, and it has
been cited as `Odena2016_Deconvolution` in `views_hydranet/architectures/upsample_interp.py`,
`PREREG.md`, `FINDING_GF.md` and GitHub issue **#491** — **as though it were a holding. It is not.**
That citation key does not exist in the library.

⚠️ **What this does and does not threaten.** The *mechanism* is not taken on authority: it is
**derived** from `Dumoulin2016`'s arithmetic (held) and **measured** here (`M92`'s ρ(1) **+0.123**
against ρ(2) **+0.921**, channel floor **−0.725**; `G-F`'s untrained control gap **+1.257**). ⇒ The
science stands without the paper. ⛔ **What is not acceptable is the citation**, which attributes a
named result to an unverified source — the class the 2026-10-29 claims audit exists to catch.

**Action, in this order:** (1) mark the four citations **unverified** until fetched; (2) fetch via
`/library add`; (3) `/library verify` the specific claim attributed to it — that the remedy
*reduces* checkerboarding rather than abolishing it, which is the sentence `G-F`'s "not a
tautology" argument rests on.

⚠️ Also not held: **`Shi2016_SubPixel`** (PixelShuffle / ICNR), the parameter-exact third family of
remedy. Not needed for this arm; fetch only if `UpsampleConv2` is promoted and ICNR initialisation
becomes a question.
