# Pre-analysis plan — does removing the double-applied zero process recover rollout skill?

**Status: LOCKED** — committed alone, before `tools/` contained anything. `git log` proves the ordering.

---

## 1. Where this comes from

**M44 (today, PR #299)** measured on four converged L=300 models that the occurrence process is
**applied twice**. `forecast_composition='soft_gate'` emits `body_sample × bernoulli(gate)`
(`composition.py:55-59`); the `nb` body carries **its own mass at zero**, so a cell survives only if
the gate fires **and** the NB draw is non-zero — re-answering the question the gate was trained on.

At step 1, where the input is entirely real data and no feedback has occurred: **mean `gate/truth` =
1.280** (the gate *over*-predicts occurrence by 28%), attenuated **×0.332** to **0.424** of truth,
reproducing the independently observed emitted ratio on **4/4 seeds**.

`TruncatedNBFamily` (`truncated_negative_binomial.py`, commit `d3a2626`) was written **for this exact
defect** — its docstring names it and cites #258. **It has never been scored on a converged vehicle.**
Its only appearance is `truncated_smoke` at **40 lessons**, marked `# SMOKE (not a scored result)` —
the floor-limited vehicle declared **VOID** by `postmortem_floor_limited_vehicle.md` (C-299).

## 2. Hypothesis and the one variable

**H:** removing the body's zero mass restores emitted occurrence to the gate's level and improves
free-running gate skill (AP) at horizon.

**The one variable: `output_distribution: 'nb' → 'truncated_nb'`.** Nothing else changes.

**`body_supervision` deliberately stays `'all'`.** `TruncatedNBFamily.nll` folds its own `y>0` mask in
(`truncated_negative_binomial.py:96-105`), so `'active'` is **numerically identical** when
`event_threshold == 0`, which is our case. `truncated_smoke` changed it *as well*, and that is one of
the three simultaneous axes that made it unreadable. **We do not repeat that.**

## 3. Skepticism ledger — the prior points the WRONG way

* **M21:** *"the residual 0.068 → 0.02 is what `truncated_nb` and `body_supervision` contribute"* —
  i.e. the only existing hint is that this family **hurts**. It is confounded between two keys and
  measured on a vehicle whose control sat at AP 0.0196 = **2.16× prevalence (FG-A FAIL)**. It is not
  evidence for or against, but it is **not** an encouraging prior and is recorded as such.
* **The family's own author named a residual risk** (`d3a2626`): *"truncating the body gives the
  gate's false positives full magnitude"*, and noted `crps_all` is **blind** to it.
* The composed-activation claim (`0.03× → 1.00× the gate`) was measured **at the distribution level**,
  never through training.
* **No config-integration, head-wiring or loss-wiring test parametrizes `truncated_nb`** — they stop
  at `nb`/`zinb`. That gap is closed before launch (§6), not assumed away.

## 4. Design

4 arms, **L=300, ε=0.0, seeds 42/43/44/45**, paired against the **existing** `fullzero_*` controls,
which are the identical config minus the one key. **No control is retrained.** `sb`, horizons
1/6/12/18/24/30/36.

## 5. Endpoints and decision rule — registered before any arm runs

**Primary: AP@h18**, treatment vs its own-seed control.

Two instruments, both reported:

* **Exact one-sided permutation** over the 8 arms — floor `p = 1/C(8,4) = 0.0143`. A 4v4 can reach
  significance; the ITF pilot's 2v2 (floor 0.167) could not, and that is why 4 seeds were chosen.
* **Paired origin-block CI**, `scripts/ap_block_bootstrap.ap_diff_origin_block_ci`, per seed. It
  refuses rather than silently intersecting when supports or origin sets differ, and refuses when
  `has_gate` differs — the #282/C-293 class.

**Four states, not two** (`ss_retention` §7 precedent — this is what makes "no effect"
distinguishable from "couldn't tell"):

| state | condition |
|---|---|
| **EFFECT** | `p ≤ 0.05` **and** mean ΔAP ≥ 3·MDE_AP(h18) **and** all four seeds agree in sign |
| **NULL** | `p > 0.05` **and** the CI on the mean difference **excludes** a 30% relative effect |
| **UNDERPOWERED** | `p > 0.05` **and** the CI **includes** it |
| **VOID** | any falsifier in §6 fires |

**No branch may be overridden by an argument not written here** (C-305: a registered rule fired, was
overridden on grounds it did not contain, and was written up as "no branch matched").

### Magnitude guardrails — an AP win alone is NOT a win

The family's own named risk is that a truncated body gives the gate's **false positives full
magnitude**, and `crps_all` is blind to it. Therefore `crps_all`, `size_ratio`, `mag_on_false_pos`
and `n_false_pos` (all already columns in `score_*.csv`) are reported **at every horizon**, and:

> **A gain in AP accompanied by a regression in `crps_all` is reported as a TRADE, not a win**, and
> the ship recommendation is withheld pending a magnitude-aware decision.

This is registered now precisely because it is the finding we would be most tempted to soften later.

## 6. Falsifiers — pre-committed

* **F1 — the contrast is one key.** Each arm's resolved config must differ from **its own-seed
  `fullzero_*` control** in **exactly `{output_distribution}`**. Checked by symmetric difference of
  the two exec'd dicts, before any GPU work. *(The ancestor tool only diffs against the floor
  `violet_visitor`; that would leave the actual experimental contrast unverified.)*
* **F2 — the mechanism must engage.** Step-1 emitted occurrence must move from ~0.42× toward ~1.0× of
  the gate. **If AP moves but this does not, the family is not doing what it claims and the AP result
  is uninterpretable** — reported as such, not as a win.
* **F3 — floor gate PASS** on every arm (`FG-A ≥ 5× prevalence`, thresholds md5
  `6d5714d5ceda147ed16f53143abe7e37`). A FAIL means the vehicle cannot show the effect, which is
  exactly how `truncated_smoke` wasted three days.
* **F4 — h1 sanity.** No arm may lose h1 AP by more than the control seed sd. h1 is nearly
  teacher-forced; a large h1 loss means the change broke the model, not the rollout.
* **F5 — no NaN/inf** in any score row.

## 7. False-negative mode and reopen trigger (C-307), written before the result

**Registered false-negative mode:** this tests `truncated_nb` **under `soft_gate` with a gate trained
alongside it**. If the gate's 28% over-prediction is itself a response to the body's zero mass, then
retraining with a truncated body changes the gate too, and a null would **not** isolate the
composition. **A NULL here closes "swapping the body fixes rollout skill", NOT "the double-zero
diagnosis was wrong"** — M44's decomposition stands on its own measurement either way.

**Reopen if:** the ZINB `self_zeroed` cross-check (free, existing artifacts) shows the step-1
shortfall is absent there; or anyone wants `threshold_gate` instead of `soft_gate`.

## 8. Scope

4 seeds, one architecture, `sb`, L=300, calibration partition. **No curriculum change. No
`body_supervision` change.** The ZINB cross-check is deliberately **not** bundled here — it is a
different question and would confound the write-up.

---

# AMENDMENT 1 — §5's permutation test could not express a NEGATIVE effect

## ⚠️ Disclosure first

**This defect was found AFTER the four arms were scored, and I had already seen the numbers.** It is
disclosed here rather than quietly corrected. The justification below is taken from §5's own
registered prose, not from the observed values.

## The defect

`verify_trunc._permutation_p` computes the exact one-sided p for **`treatment > control`**. §5's
EFFECT row requires `p ≤ 0.05`. **A strongly NEGATIVE effect therefore scores `p = 1.0` and can never
reach the EFFECT branch** — it falls into `NULL / UNDERPOWERED` instead.

That is what happened: mean ΔAP@h18 = **−0.2376**, which is **7.5×** the `3×MDE` bar of 0.0317, with
all four seeds agreeing in sign — and the rendered verdict read *"NULL / UNDERPOWERED, p=1.0000"*.
**As implemented, the rule was incapable of reporting the result the experiment actually produced.**

## Why the fix is the registered rule, not a new one

Two pieces of the locked text show the intent was **two-directional** and only the implementation was
one-directional:

* §5's EFFECT row reads *"…**and all four seeds agree in sign**"*. A clause about *which* sign the
  seeds agree on is meaningless if only one sign can ever fire.
* The code itself contains `state = "EFFECT" if mean_d > 0 else "EFFECT (NEGATIVE)"` — a branch I
  wrote and which was **unreachable**, because the `p ≤ 0.05` guard above it excludes `mean_d < 0`.

**Fix:** the permutation p is computed **in the observed direction** (equivalently, a two-sided test
with the direction reported). `3×MDE` is compared on `|mean ΔAP|`, which it already was. No threshold
moves: `p ≤ 0.05`, `3×MDE`, and sign agreement are all unchanged.

**This makes the rule harder to satisfy in no direction and easier in none** — it makes a previously
unreachable branch reachable. It cannot be used to convert this result into a favourable one: the
effect is negative either way, and the change is what allows it to be *called* an effect rather than
misreported as a null.

## Consequence for the verdict

In the observed direction `p = 0.0143` — the exact `1/C(8,4)` floor, i.e. **maximum separation: all
four treatment arms fall below all four controls.** With `|mean ΔAP| = 0.2376 ≥ 3×MDE = 0.0317` and
unanimous sign, §5 returns **EFFECT (NEGATIVE)**.
