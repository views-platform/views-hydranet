# 01 — Literature

**2026-10-30.** Holdings verified against `~/brain/9_library/papers/` by listing it, not from
memory. ⛔ One source is load-bearing and **not held** — §3.

## 1 · Held, and what we take from each

| source | what it is | what we take |
|---|---|---|
| **`Odena2016_Deconvolution`** — Odena, Dumoulin & Olah (2016), *"Deconvolution and Checkerboard Artifacts"*, **Distill**, DOI **10.23915/distill.00003** | ⭐ **NOW BEING INGESTED** (Simon supplied the source 2026-10-30; it was NOT held when this programme's citations were written). The named mechanism | ⛔ **Read against the source, two of my claims were WRONG — see §4.** What it actually supports: the artifact comes from **uneven overlap**, *"when the kernel size ... is not divisible by the stride"*; making it divisible *"helps"* but **"models with even overlap often learn kernels that cause similar artifacts"* and *"it's still very easy for even overlap deconvolution to cause artifacts."* ⇒ ⭐ `M92` is an **empirical instance of that warning in a production model.** The remedy is **resize-convolution** (nearest or bilinear resize, then convolution; nearest *"worked best"*), and its mechanism is given in the source's own words — *"Where deconvolution has a unique entry for each output window, resize-convolution is implicitly weight-tying"* — which **matches the derivation in `02_design` §2 independently** |
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

## 3 · Gaps to fetch

⚠️ **`Shi2016_SubPixel`** (PixelShuffle / ICNR) — **not held.** ⛔ And correcting an earlier note
here that called it *"the parameter-exact third family of remedy"*: Odena states that
kernel-divisible-by-stride deconvolution is **"equivalent to 'sub-pixel convolution'"** ⇒ plain
PixelShuffle **is the incumbent's defect**, not an alternative to it. What would make it a remedy is
**ICNR initialisation**. Fetch only if `UpsampleConv2` is promoted and initialisation becomes a
question.

---

## 4 · ⛔⛔ WHAT I GOT WRONG BY CITING FROM MEMORY

The source was **not in the library** when this programme's citations were written, and I cited it
anyway — in `views_hydranet/architectures/upsample_interp.py`, `PREREG.md` §4, `FINDING_GF.md`,
`gf_phase_signature.py` and GitHub **#491**. Two claims did not survive contact with it. **All five
sites are corrected**, in place and dated.

**1 · ⛔ "`kernel == stride` is Odena's worst case, because there is no overlap to average."**
**Backwards.** Odena's artifact is **uneven** overlap, which happens when kernel size is **not**
divisible by stride; `k=2, s=2` is divisible — the case Odena **recommends as a mitigation**.
⭐ The defensible claim, and the one the candidate now rests on, is the source's explicit caveat
that even overlap is **not sufficient** because the learned kernels still diverge. That is narrower
and better supported than what it replaced.

**2 · ⛔ "Odena's fix *reduces* checkerboarding; it does not abolish it."**
The source is **stronger**: *"they do fix the checkerboard artifacts"*, qualified only by *"we
don't necessarily think that either approach is the final solution to upsampling."* ⇒ The
"a phase pattern CAN survive the remedy" argument — which `G-F`'s claim to be a real test rather
than a tautology rested on — is **my own derivation** from window parity over a 2×2 block grid, and
it should never have been hung on a citation. ⭐ `G-F` then settled it **in Odena's favour**: the
gap went to **−0.1984**.

⭐⭐ **What did NOT change: the mechanism and every number.** It is derived from `Dumoulin2016`
(held) and measured here — `M92`'s ρ(1) **+0.123146** against ρ(2) **+0.920900**, channel floor
**−0.724900**; `G-F`'s untrained control gap **+1.257220**. ⇒ The science stands; the attribution
did not.

⚠️ **The transferable rule.** A citation is an artifact like any other, and *"never describe an
artifact you have not opened"* applies to papers. The claims audit of 2026-10-29 was built for
exactly this class and it did not index `.md` prose against the library — a gap this entry records.
