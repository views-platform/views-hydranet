# D5 — pre-registration: is the over-smoothing ARCHITECTURAL or LEARNED?

**2026-10-20, written and committed BEFORE either measurement runs.** Free, CPU, no pod, no
training, no fetch. Scripts: `2026-10-12_underfire_diagnosis/d5_why_smooth.py`.

## The question D4 opened and did not answer

`2026-10-20_d3_static_channel_ceiling.md` §9 measured both heads emitting fields far smoother than
the process: body `log mu` lag-1 autocorrelation **0.8594 / 0.7636** in sample and **0.9266 / 0.9227**
out of sample, gate logit **0.9598 / 0.9574** and **0.9498 / 0.9464**, against the truth's
noise-controlled reference of **0.6419 / 0.6261**. §9 states explicitly that **why** is untested, and
lists four live candidates: the decoder's receptive field, the recurrent BatchNorm recalibration
(C-184, fitted on 32×32 windows and applied to 360×720), `weight_decay: 0.1` (global —
`utils.py:276-281` applies it to `unet.parameters()`, and the three class heads are **not** in the
`weight_decay: 0.0` exclusion groups at `training_engine.py:113,136`), and the objective's own
preference.

⛔ **Three of those four are training-side and one is not.** One measurement separates them.

## D5-A — the same architecture, untrained

Build `choose_model(cfg)` from the **same sidecar config** and do **not** load the state dict. Run
the identical 49-step sweep over the identical land mask. Measure lag-1 spatial autocorrelation of
`log mu` and the gate logit.

### Pre-committed readings — one of these three, chosen now

| init autocorrelation | reading |
|---|---|
| **≥ 0.85** (inside the trained range 0.86–0.96) | **The architecture imposes the smoothness floor.** The trained value is then **not** evidence of a learned pathology, and all three training-side candidates are **demoted** — they would be explaining something that was already there. |
| **≤ 0.65** (at or below the truth's noise-controlled 0.64) | **The architecture does not impose it ⇒ the smoothness is LEARNED.** BN recal, global weight decay and the objective stay live, and the next experiment is E6, which separates the first two for free on CPU. |
| **0.65 – 0.85** | ⛔ **INCONCLUSIVE.** Reported as inconclusive with no mechanism claimed. No post-hoc reinterpretation of the band. |

### ⛔ VOID conditions, committed now

1. `sd(log mu) < 0.01` or `sd(gate logit) < 0.01` at init. An almost-constant field makes lag-1
   autocorrelation numerically meaningless, and a random-init network through `softplus` is a real
   candidate for exactly that. A degenerate field is **VOID, not smooth.**
2. Any non-finite value in either field.

### ⚠️ The limit, stated before the number exists

**C-325** records that two of three earlier mechanism tests were void because they measured a network
**at initialisation**, where an instability that *training builds* cannot appear. D5-A is **not** that
shape — it measures a **floor the architecture imposes before training**, which is precisely what init
can measure — but the asymmetry must be respected in both directions:

- A **smooth** init shows the architecture **permits and defaults to** smoothness. It does **not**
  show training added nothing; training may have made it smoother still.
- A **rough** init shows the architecture does **not force** smoothness. It does **not** prove any
  particular training-side cause; it only keeps all three alive.

Neither outcome identifies a mechanism. D5-A **narrows the candidate set**; it does not close the
question, and under **C-307** an n=1 screen with no falsifier on the outcome closes nothing anyway.

## D5-B — how far does one input cell reach in one step? (trained weights, descriptive)

With the **trained** vehicle, run one step on the real volume, then re-run with a single land cell's
`sb` input raised by +1 in `log1p` space, and measure `|Δ log mu|` against distance from that cell.
Repeated over a handful of well-separated cells and pooled.

⛔ **No falsifier and no threshold is pre-committed for D5-B, because I have no prior for what the
number should be.** It is reported as a description — the radius at which the response falls to 10%
of its peak — and **may not be converted into a pass/fail after the fact.** It avoids C-325 entirely
by using the trained weights, which is why it is here at all.

## What D5 cannot do

It cannot tell us whether the over-smoothing *costs* anything. That needs the composed forecast
scored, which is `crps`/AP on a rollout, not a forward pass. Nothing here touches the C-152 class
statement: a proposal that sharpens the field still inherits that null unless it changes the joint or
the sampler.

---

## ⛔ Amendment A1 — committed BEFORE the run, after the `library` seat answered

I asked the `library` seat whether the corpus holds anything on over-smoothing in likelihood-trained
dense predictors. It does, and it **names a fifth candidate I had not listed — which is now the
leading one.** Recorded here rather than in the write-up so it cannot be mistaken for a post-hoc
explanation of whatever D5-A returns.

| held | what it says | level |
|---|---|---|
| **`Price2023_GenCast`** (C-2868) | *"MSE-trained deterministic models like GraphCast produce blurry forecasts closer to the ensemble mean"*, while a diffusion objective matches ERA5 power spectra | **empirical** (§3), weather on a sphere |
| **`Ribeiro2025_FlowCast`** (C-2909) | *"Deterministic nowcasting models trained with pixel-wise losses like MSE produce overly smooth forecasts at longer lead times **because they average over possible futures**"* | **empirical/introductory**, precipitation |
| **`BondTaylor2022_DeepGenerativeModelling`** (C-438) | direct likelihood optimisation penalises missing modes more than implausible samples — the theoretical reason GANs are sharper than VAEs | theoretical, generative models generally |
| **`Tancik2020_FourierFeatures`** (C-187) | spectral bias: the NTK eigenvalue spectrum decays rapidly with frequency, so standard nets cannot learn high-frequency content | mechanistic, but **MLPs with positional inputs, not conv/U-Net** |

### What this changes

**Candidate 5 — the objective emits a conditional mean, and a conditional mean of a point process is
a smooth intensity field.** My §"four live candidates" listed "the objective's own preference" as a
vague fourth item. The literature names it precisely, in two papers, in a domain shaped like ours
(gridded geophysical fields, rare intense events, rollout horizons) — and it is the **standard**
explanation for exactly this signature in exactly this class of model.

⇒ **The `≤ 0.65` branch's stated next step is amended.** If the smoothness is learned, the leading
candidate is no longer BN recal or global weight decay; it is the objective averaging over futures,
which is **literature-grounded** where those two are repo-grounded hunches. E6 remains worth running
(it is free) but is demoted from "the next experiment" to "the cheap control that rules out two
confounds first."

### ⛔ And it converges with Simon's own framing, which was prior to all of it

His objection was that the hidden and cell state may encode *"the first stochastic process, the latent
risk map or risk field"* well and *"the second stochastic process, which is like a marked spatial
point process"* badly. The literature's statement is that a likelihood-trained deterministic predictor
emits **the conditional mean**, which **is** the latent intensity field, and cannot emit a realisation
of the point process on top of it. **These are the same claim.** Three independent routes — his
architectural intuition, D4's lag-1 measurement, and the weather/nowcasting literature — land on one
place, and his came first.

It also closes a loop with **C-152**, whose class statement already said *"any future proposal that
improves per-cell marginals inherits this null unless it also changes the joint or the sampler."* The
literature's remedy class is exactly that: not a sharper field, but a different objective or a drawn
sample. C-152 was the right rule before anyone knew why.

### ⚠️ The gap, and it is a decision for Simon, not an action I will take

The corpus holds the **phenomenon** and not the **mechanism or the diagnostic**. The `library` seat
reports nothing on spectral bias in convolutional or U-Net architectures (Tancik is MLP-only), nothing
treating NLL→blurriness as a studied defect with a remedy beyond "use a GAN or diffusion objective",
and **none** of the canonical dense-prediction references held — `Ledig2017` (SRGAN), `Wang2018`
(ESRGAN), `Isola2017` (pix2pix). It also reports that our lag-1 comparison (model 0.86–0.96 against
truth 0.51–0.59) is **a cleaner quantification of this than anything in the corpus**. ⛔ An
acquisition costs money and is Simon's call; nothing is fetched here.

---

## ⛔ Amendment A2 — D6, and the falsifier §9.3 did not have

**Committed before D6 runs.** The `opponent` seat attacked §9.3's comparator two ways (its §13,
commit 885d6e8). One objection — the one *I* nominated as my weakest joint — **fails**. A different
one **lands**, and it is the fourth instance of today's error class.

### What lands: the four rows are not on the same scale

`log mu` and the gate logit are continuous and atom-free. The truth is a **raw rate with an atom of
mass 0.986 at exactly zero**. Lag-1 Pearson is invariant under *affine* maps, not under `log` or
`logit`, and the truth cannot be put on the model's scale because `log(0)` does not exist for 98.6%
of land cells. On a synthetic field of known smoothness the seat measured my estimator understating
the true log-intensity lag-1 by **0.073 → 0.146 as the silent fraction rises**, monotone in the atom,
with our panel more silent than any row it tested ⇒ **at least ~0.15 of my reported 0.22–0.32 gap is
reproducible with no model involved.**

⛔ **Not to be fixed by transforming the truth.** The seat tried `log((n+0.5)/337)` and it is *worse*
— 0.64/0.63 against a ground truth of 0.97 — because logging a quantised count injects noise into
near-silent cells. One field reads 0.97, 0.85 or 0.65 depending only on which monotone map it is
measured on. ⚠️ That its wrong fix lands on our reported 0.6419/0.6261 is coincidence in its
synthetic, and a warning.

### D6-C — the pre-registered falsifier, reading committed now

Push **the model** through **the truth's estimator**, not the truth through the model's scale. Both
rows then become one functional of two objects of the same kind.

Per land cell, from `d4_result.npz` (already holds `mu`, `theta`, `gate` for all 64,818 cells; **no
new forward pass**): draw 336 months of the gated-NB generative model, build the empirical base-rate
field **exactly as §9.3 builds the truth's**, take lag-1 of its log, and replicate. Exactly, not by
brute force: `k ~ Binomial(336, gate)` then `S ~ NB(k·theta, p)`, since a sum of `k` i.i.d.
`NB(theta, p)` is `NB(k·theta, p)`.

| outcome | pre-committed reading |
|---|---|
| **0.642 falls OUTSIDE** the posterior-predictive distribution of the statistic | §9.3's finding **STANDS** — the model's field is smoother than the truth's by more than the estimator can explain |
| **0.642 falls INSIDE** it | §9.3's finding is **WITHDRAWN** — the gap is the estimator, and the truth's intensity is consistent with being as smooth as the model's |

⛔ No third branch, no post-hoc reinterpretation of a near miss. Two-sided, α = 0.05, ≥ 200 replicates.

### D6-A — the objection I nominated, independently re-derived

The seat fitted classical attenuation `ρ_obs(T) = ρ_S/(1 + v/T)` on my three truth windows,
over-identified it, and the held-out 36-month point landed to **0.0004** (`v` = 3.4612, `ρ_S` =
**0.6405**) ⇒ the 336-month reference is attenuated by **1.02%**. ⛔ **I am not taking that on its
word.** D6-A refits it from the parquet on **more windows than it used**, holding out different
points, because a two-parameter fit through three points that predicts the middle one is impressive
only if the middle one is not doing the work.

### D6-B — the lag PROFILE, which separates what lag-1 cannot

The seat's sharpest point, conceded on sight: **§9.2 and §9.3 are in tension.** §9.2 says too
*little* large-scale amplitude (`sd(log mu)` flat 1.714 → 1.705 while the truth's grows 1.946 →
2.987); §9.3 says too *much* local correlation. A receptive-field blur produces the second and **the
opposite of the first** — blurring preserves regional contrast and destroys local contrast. So
"diffuse" does not cover both and they may be two defects.

D6-B computes `ρ(lag)` for lag 1…32 on both heads instead of lag 1 alone. **A receptive-field limit
has a hard length scale and shows a knee at its radius, identical in both heads (same decoder depth);
a shrinkage or normalisation cause is scale-free and shows none.** Descriptive — ⛔ no threshold is
pre-committed, because I have no prior for the knee's location and will not invent one after seeing it.

### Two concessions and one contest, recorded before the numbers

1. ✅ **CONCEDED: *"two heads that share only the trunk"* is false.** They also share the input
   channels, the data, the optimiser, the loss weighting, the BatchNorm regime and the weight decay.
   "Upstream of both" names **everything shared**, not the trunk — and my own ⚠️ list four lines
   below named BatchNorm and weight decay, so the sentence contradicted its own caveat. §9.4 is
   corrected to "upstream of both heads, in anything they share".
2. ⚠️ **CONTESTED, precisely: candidate 4 is not dead, only its narrow form is.** The seat argues
   *"the NLL's own preference"* cannot cause a defect in both heads because the gate trains on
   weighted BCE. **Right about the NB NLL specifically, wrong about the candidate as A1 restated it.**
   A1's candidate is not "the NB likelihood's preference" but *"a pixel-wise proper-scoring objective
   emits a conditional mean"* — and that covers **both** heads, because BCE's conditional mean is `p`
   exactly as the NB's is `mu`. `Ribeiro2025_FlowCast`'s wording is *"pixel-wise losses"*, not "the
   NLL". So the count is **three live plus one, not three**: BN recal, global weight decay, receptive
   field, **and** conditional-mean averaging, which remains the leading one and is the only candidate
   that predicts the defect appearing identically on two heads with different losses.
3. ✅ **ACCEPTED without reservation:** §9.3 is **suspended, not withdrawn**, until D6-C runs. The
   half of §9.4 that carries *"upstream of both"* never used the truth comparator and is unaffected,
   and the S1 verdict never depended on §9.3 at all — it is carried by §2/§3's bounds and C-152.

---

## ⛔ OUTCOME — logged 2026-10-20, after D5–D10 ran

**Full results: `2026-10-20_d3_static_channel_ceiling.md` §10, with the decision table at §10.1.**
This section exists because a pre-registration without an outcome link is a broken record, and
`status` is supposed to refuse to call a programme ready while one is missing. Verdict **against the
readings committed above**, not against anything written afterwards:

| pre-committed item | fired? | outcome |
|---|---|---|
| **D5-A** — ≥0.85 architectural / ≤0.65 learned / between inconclusive | ⛔ **none** | **VOID** by this plan's own guard (§10.6): at random init `sd(log mu)` = **0.0017** and `sd(gate logit)` = **0.0016** against the committed **0.01** floor, mu spanning 0.4798–0.5328. ✅ The guard worked. The question is not answerable by an init measurement on this architecture. |
| **D5-A VOID condition** — "below a 0.01 floor the test says nothing" | ✅ **FIRED** | Logged as a correct pre-commitment (`PREDICTION_LEDGER` R-16). |
| **D5-B** — one-step impulse reach, descriptive | ✅ ran | Below 10% of peak at **10 cells**, exactly **0** at ≥16. ⚠️ Only **3** probe cells met `MINSEP=60`, not 6 — the top-100 active cells are clustered. Recorded as a limitation. |
| **D6-A** — the noise objection I nominated as my own weakest joint | ⛔ **did not land** | Dead, and more thoroughly than the seat argued: nine windows, LOO, `ρ_S = 0.6329`, `v = 2.8131`; T=336 observed 0.6340 is **−0.17% past** the asymptote ⇒ attenuation indistinguishable from zero (§10.2). |
| **D6-B** — the lag profile | ✅ ran | `ρ(model)/ρ(truth)` rises monotonically **1.34 → 4.08** (log mu) and **1.50 → 4.79** (gate), never returning to 1. Excludes a **one-step decoder blur** — one candidate of five, no more (§10.5). ⚠️ The seat later withdrew its own knee-based discriminator: a ConvLSTM's reach grows ~one kernel radius per carried step, so recurrent reach predicts **no knee**. |
| **D6-C** — the falsifier, two branches committed in advance | ✅ **FIRED FOR §9.3** | The model reads **0.7053/0.6824** through the truth's estimator against 0.8594/0.7636 directly; 95% interval [0.6922, 0.7193] / [0.6683, 0.6962]; truth **0.6419/0.6261** **outside** it, 3.71 / 3.03 half-widths below the lower edge. **§9.3 STANDS** — but the estimator eats **0.154/0.081**, so the honest gap is **0.06, not 0.22** (§10.3). ⚠️ A **simulation** interval, not posterior-predictive: Monte Carlo only, and the 4 dropout passes were averaged rather than propagated. |

### ⛔ What this plan did not foresee

1. **§9.2 was an artifact nobody had attacked.** Testing D6 properly exposed it: the coarsening admits
   only blocks with `tm > 0`, moving the admitted fraction **2.2% → 22.4%**, and on a nested
   population the trend reverses. **WITHDRAWN** (§10.4, `W-33`).
2. **D8, D9 and D10 are not in this plan at all.** They were built in response to the peer seat's
   later rounds. D8's **both** readings were withdrawn (`W-34`, `W-35`); D9 settled the sign null-free
   (**215.0** vs **536.3** effective cells, model **more** concentrated); D10 showed D9 was
   **conservative** and corrected the seat's debias by 1.74× (`R-15`).
3. **The leading candidate named in Amendment A1 — conditional-mean averaging — is still the leading
   one, and still untested.** A1 predicted it alone explains the defect appearing identically on two
   heads with different losses. Nothing since has touched that prediction. **E6** remains the free CPU
   control separating BN recal from global weight decay.

### The settled finding this programme produced

> **The model emits a few large smooth blobs where the truth has many small sharp spots.**

**0.06** smoother than the variance decomposition permits (local), **2.49×** too few effective places
(global), while under-firing **4.18×** per cell-month. ⛔ The two headline figures share no scale and
must never be combined. ✅ **The S1 verdict did not move in any round.**
