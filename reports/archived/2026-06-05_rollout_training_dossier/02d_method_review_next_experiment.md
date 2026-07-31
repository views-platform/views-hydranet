# 02d — Expert Method Review: what's the next experiment? (value-of-information)

**Date:** 2026-06-07 · **Skill:** `expert-method-review` · **Chair:** simon (Bayesian; not seated)
**Trigger:** two data points in hand — SS-ceiling **0.25 → explode**, **1.0 → collapse-to-zero**. Which next step has the highest information-per-cost: (a) confirm-collapse eval, (b) an intermediate-ceiling run, (c) build the unroll-K/GTF fix?

---

## 1. Decision under review & the DGP
Pure **experiment-design / VoI** question. Evidence: along the *scheduled-sampling ceiling* axis (the dial already ramps low→ceiling over training), **0.25 explodes** (too little own-prediction practice → its feedback amplifies) and **1.0 collapses to 0** (too much → it learns the trivial zero fixed point). DGP that makes this sharp: **~95% zero cells** — so "predict zero everywhere" is a strong degenerate attractor the loss happily rewards. The question: is there a *stable-nonzero* regime in between, and is finding out worth ~80 min?

## 2. Panel
| Seat | Why | Fault line |
|---|---|---|
| **Gelman** (Bayesian workflow / VoI) | "which experiment would change the decision?" — the chair's tribe | run-the-cheap-bracket vs not |
| **Sutton** (bitter lesson) | don't burn runs sweeping a crude knob; build the general method | (b) vs (c) |
| **DL-engineer** | ~80 min/run budget; marginal-run worth | cost |
| **Hochreiter** (dynamics) | does explode∧collapse already prove per-step SS has no good regime? | knife-edge vs Goldilocks |

## 3. Library grounding
**Held:** `Brandstetter2022` (pushforward = the unroll-K of (c); trains near the data manifold), `hess23a` (GTF — the **α-anneal scheduler** the chair is intuiting: interpolate the fed-back *state* + **bound the gradient**, a *different and more principled* scheduler than an SS Bernoulli ceiling), `NIPS-2015 scheduled-sampling` (the dial we're sweeping). **Gaps → fetch:** `Huszár2015` (SS is a **biased estimator** — known secondhand via Lamb 2016 / Professor-Forcing, which we hold; load-bearing here), `Mikhaeil2022` (chaotic-DS ill-posedness).

## 4. Independent critiques

### Gelman — *value of information*
- **(a) is low-VoI: skip it as a standalone.** It would mostly confirm what we already believe (flatline-to-0, qualitatively unlike healthy pink) and **doesn't change the decision** — we won't ship the 1.0 model regardless. Don't run an experiment whose every outcome leaves the decision unchanged.
- **(b) is the high-VoI move *because* the panel can't resolve it from theory.** We have two endpoints; the middle is genuinely unknown (Goldilocks or knife-edge?). One ~80-min run **resolves that uncertainty cheaply** — and either *cheaply solves it* or *kills the plain-SS family with evidence*. That's textbook "run the experiment that splits your hypotheses."
- **But fold the real eval into (b)'s readout.** Don't judge (b) on the synthetic probe alone (same gap that makes (a) tempting). One retrain → read **both** `diagnose_io_gain` (stability) **and** a real eval (skill/collapse). That single run answers explode-vs-collapse-vs-Goldilocks **and** probe-vs-real in one shot — subsuming (a) into (b).

### Sutton — *bitter lesson*
- **The SS axis is a crude knob; stop sampling it.** We already know own-prediction training is the lever and that plain per-step SS brackets explode/collapse. Sweeping its ceiling is tuning a weak method. **Build the general one (c)** — unroll-K with controlled depth (and let K/strategy be learned, per the chair's own brainstorm).
- **80 min on (b) is 80 min not on (c).** If the durable answer is the structured lever, the bracket-point is a detour.
- Concedes: if (b) is *truly* near-free and could cheaply settle it, it's not unreasonable — but don't let a Goldilocks-SS-win **distract** from building the real thing, because (see Huszár) it'd be a biased patch.

### DL-engineer — *cost*
- **(b) is cheap: zero code, one config value, ~80 min** — and the harness (GPU-enforced driver, diagnose_io_gain, baselines) already exists. (c) is **days** (training-loop code + the K-strategy + tests + runs).
- So **sequence by cost**: the ~free bracket-point first; commit the expensive (c) build *after* it tells us whether plain-SS is dead. Cheapest decisive evidence before the costly build.

### Hochreiter — *dynamics / knife-edge*
- **explode@0.25 + collapse@1.0 likely brackets a *knife-edge*, not a plateau.** With ~95% zeros, strong own-feedback drives the **zero fixed point**; weak own-feedback leaves the operator gain >1 → explode. A stable-**nonzero** attractor may be measure-zero on this axis — so **(b) probably fails**, and its information is "confirm no good SS regime."
- **The real lever isn't the operating-point distribution (what SS/pushforward shift) — it's the Jacobian product.** GTF's `(1−α)` scaling bounds *that* (Hess Eq. 8); a ceiling sweep doesn't touch it. So the chair's "scheduler" instinct is right but its principled form is **GTF (c/B2)**, not an SS ceiling.
- **Huszár:** SS is biased — *no* ceiling is guaranteed to recover the right model. Another reason (b)'s upside is capped.

## 5. Key disagreements
- **The crux — Goldilocks vs knife-edge:** Gelman/DL-engineer say *we cannot know without the cheap run* (→ run (b)); Hochreiter says *theory + the bracket + Huszár already predict no stable-nonzero SS regime* (→ (b) is low-information, go (c)). **Merit:** Hochreiter is probably right on the prior, **but a Bayesian doesn't skip a near-free experiment that would update a genuinely uncertain belief** — and if Hochreiter is right, (b) *converts his prediction into evidence*, which is what lets you commit to (c) without second-guessing.
- **(b) vs (c) (Sutton vs Gelman):** parsimony/momentum (build the real method) vs evidence-discipline (one cheap datum first). The 80-min cost is small enough that this is close.
- **Everyone agrees:** skip standalone (a); never declare a run "works" on the synthetic probe alone (pair with a real eval) — that's the C-126/C-128 calibration discipline.

## 6. Synthesis & recommendation
**Run exactly one more cheap experiment, designed to be decisive, then commit to (c) regardless.**
1. **(b′) — one retrain at an intermediate ceiling (~0.5), read BOTH `diagnose_io_gain` *and* a real eval** (stability + skill/collapse in one ~80-min shot; this *subsumes* (a)). Pre-register the split: *stable & nonzero & nontrivial skill* → plain-SS has a regime (cheap win, surprising); *explode, collapse, or stable-but-zero-skill* → **plain SS has no good regime — confirmed, not assumed.**
2. **Then build (c) — the controlled unroll-K / GTF** — *informed* by (b′). Note the chair's "start-low scheduler" instinct is **GTF's α-anneal**, so (c) should be **B2-flavoured** (bounded cross-step gradient), not just a deeper pushforward.
3. **Skip standalone (a)** — its VoI is folded into (b′)'s real eval.

**Strongest dissent to keep live (Sutton + Hochreiter):** the bracket + Huszár's bias result may already be enough — (b′) risks being an 80-min confirmation of a foregone conclusion, and the honest VoI may be ~zero if your prior on "no SS regime" is already high. *If the chair's posterior is already confident plain-SS is dead, skip (b′) and go straight to (c).* The decision genuinely turns on **how uncertain you actually are** about the knife-edge — which only you can price.

## 7. Methodological risks (register-compatible)
| ID | Tier | Trigger | Location | Narrative |
|----|------|---------|----------|-----------|
| **RT-knifeedge** | 3 | Reading a "stable" (b′) result off the synthetic probe and concluding plain-SS works | `diagnose_io_gain`; `05` | explode@0.25 ∧ collapse@1.0 may bracket a knife-edge with no stable-*nonzero* regime; a middle ceiling can look "in-range" on the synthetic probe while being zero-skill (the 1.0 collapse already reads "healthy (in-range)" at 0.00). Always pair (b′) with a real eval (skill/MCR), never the probe alone. Ties C-126/C-128. |
| **RT-VoI** | 4 | Running confirmatory-only experiments (e.g. standalone (a)) that can't change the decision | dossier `07` | Discipline: run the experiment that *splits hypotheses*, not the one that *confirms the expected*. (a) is low-VoI; (b′) is only worth it if the knife-edge uncertainty is genuinely live. |
| **RT-biasedSS** | 3 | Committing to *any* plain-SS ceiling as the durable fix | `02_design §4.3`, `05` | Huszár 2015: scheduled sampling is a biased estimator — no ceiling is guaranteed to recover the true model. Treat (b′) as a *gate*, not a candidate endpoint; the durable fix is the (c) family (GTF's α-anneal is the principled scheduler the chair is intuiting). Fetch Huszár2015 + Mikhaeil2022. |
