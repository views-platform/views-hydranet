# 01 — Literature

## Load-bearing

**`SanchezGonzalez2020_GraphNetworkSimulators`** (ICML 2020) — the source of the intervention.
*Taken:* corrupt training inputs so the model tolerates its own errors; **random-walk noise beat
i.i.d.** because rollout error accumulates; trained on 600 steps → coherent at **5,000**; and the
supplement's statement that target adjustment *"happens implicitly when the loss is defined directly
on next-step ground-truth position"* — our case, so **no target adjustment**.
*Not taken:* **σ = 3e-4**. Their inputs are dense, standardised, unit-variance particle velocities.
Ours are `log1p(counts)`, unstandardised, **~99.94% exactly zero**. The number does not transfer; the
**principle** — *"so the training distribution is closer to the distribution generated during
rollouts"* — does. Hence S1 measures before S2 designs.

**`Aceituno2025_TemporalHorizons`** (TMLR 2025) — why the previous five attempts failed.
*Taken (all proven):* minima found on long horizons generalise to short, but not the reverse; the
gradient scales `O(e^{λT})` with the training horizon; the loss landscape roughens exponentially with
it. Together: **a performance–learnability trade-off**, and #308's measured explosion (133,465 → 9.4e9,
float32 exhausted at lesson 48, control *falling* 859 → 312) is an instance of it.
*Caveat the paper states itself:* derived for autoregressive feedforward MLPs; extension to RNNs is
discussed but **not proven**. Suggestive for our ConvLSTM, not settled.

**`Brandstetter2022_MessagePassingPDE_pushforward`** — already implemented here as
`pushforward_weight` (#289), never run. *Taken:* the arm itself. Also names Sanchez-Gonzalez's
Brownian-motion noise as the related approach, and proposes **temporal bundling** (predict K steps at
once) — a mild form of #310, the next epic.

## Context

`Bengio2015_ScheduledSampling` — the family this project has already exhausted (M30–M33: ε=0.5 costs
−0.0426 AP; #308 the same lever again). Both arms here run **ε=0** for that reason.

## Gaps to fetch

- Nothing blocking. **`Zhuang2025_HorizonForcing`** is held (4 claims) and reports ETT+HF beating
  teacher forcing, scheduled sampling *and* Professor Forcing — relevant to **#309/#310**, not to this
  epic.
- Not held: any treatment of noise injection for **sparse / zero-inflated count fields**
  specifically. The library's zero-inflation holdings (`Gao2024_UncertaintyProbGNNCrash`,
  `Gao2025_DiffusionZeroInflatedPrecipitation`, `Jiang2023_SpatiotemporalTweedieUncertainty`,
  `Lambert1992_ZeroInflatedPoisson`) are about **modelling** zero inflation, not about **augmenting**
  it. **This gap is the reason S1 exists** — there is no paper to copy the design from, so it has to be
  measured.
