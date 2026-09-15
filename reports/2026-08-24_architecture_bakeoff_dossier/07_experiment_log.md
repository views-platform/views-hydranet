# 07 — Experiment log (append-only)

Pre-registration `05_analysis_plan.md`, **LOCKED `32f73a9`** before any scored arm existed (with the
provenance caveat stated in its §0: unlike earlier dossiers, `tools/` was not empty at lock time).

---

### EXP-01 · 2026-08-25/26 · **NO PROMOTION — and the finding is not the endpoint**

12 arms, 6 architectures × 2 seeds, L=300, paired against the existing `fullzero_*` controls.
**12/12 scored, 12/12 oracles, 12/12 floor gates PASS, zero failed arms.**

#### §4 verdicts — as pre-registered

| candidate | Δ AP@h18 (s42 / s43) | verdict |
|---|--:|---|
| AntiAliasedPool | −0.0084 / +0.0008 | INCONCLUSIVE |
| DynamicTopSkip | +0.0011 / +0.0031 | INCONCLUSIVE |
| FiLMSkip | −0.0157 / +0.0082 | INCONCLUSIVE |
| DualStream | −0.0124 / −0.0013 | INCONCLUSIVE |
| WideMemory | −0.0111 / +0.0089 | INCONCLUSIVE |
| **ShallowPool** | **−0.0178 / −0.0150** | **REJECT** |

**Nothing promotes.** §7 predicted "five of six INCONCLUSIVE means the screen worked" — that is what
happened. The named bet, **WideMemory, did not pay.**

#### The actual finding: a monotone sign gradient across the horizon

Δ AP vs own-seed control, counted over all 12 arms:

| horizon | h1 | h6 | h12 | h18 | h24 | h30 | h36 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **better** | **9** | 9 | 5 | 5 | 2 | **0** | **0** |
| worse | 3 | 3 | 7 | 7 | 10 | **12** | **12** |

**12 of 12 negative at h30 and at h36**, from six unrelated mechanisms — anti-aliased pooling, a raw
skip, learned modulation, one fewer downsampling, a parallel full-resolution stream, and a 10×-wider
recurrent state. Under independent signs a 12/12 column is `2^-12`. The gradient is **monotone**: the
count of improving arms falls 9 → 9 → 5 → 5 → 2 → 0 → 0.

**And the damage is ordered by how aggressively each architecture intervenes.** The two that go
furthest toward preserving spatial resolution are the worst at long horizon — DualStream reaches
**−0.0567** at h36 (seed 42) and ShallowPool **−0.0475** (seed 43), 3–4σ, the largest effects in the
table. The light-touch candidates cost least.

#### The models are not worse — the rollout is

| | oracle @h18 (s42 / s43) |
|---|--:|
| control | 0.4974 / 0.5014 |
| AntiAliasedPool | 0.4991 / 0.5017 |
| DynamicTopSkip | 0.4993 / 0.4944 |
| FiLMSkip | 0.5004 / 0.4990 |
| ShallowPool | 0.4909 / 0.4991 |
| DualStream | 0.4928 / 0.5065 |
| **WideMemory** | **0.5077 / 0.5094** |

Handed a real field, **every candidate matches or slightly beats the control's ceiling** — WideMemory
is the best forecaster in the set. None of that converts into rollout skill.

**And there is no trade to weigh**: Δ`crps_all` at h18 is within ±0.006 for every arm. The body is
untouched; this is entirely the gate's placement degrading through the autoregressive loop.

#### Why — and the literature said so first

`Aceituno2025_TemporalHorizons`, Theorem 4.6: *"Loss minima found by training on long forecast
horizons generalize well to short-term forecasts, but minima found on short horizons do **not**
generalize to long-term predictions"*, with the gap scaling `O(e^{λ(T_h − T_l)})`.

**We train at one step (ε=0, teacher-forced) and evaluate at thirty-six.** Anything that improves the
per-frame objective we actually optimise is under no obligation to help at h36 — and empirically it
hurts, monotonically, in proportion to how much it sharpens the per-frame map.

**This reframes three prior programmes.** Scheduled sampling (M30–33), ITF (M42) and `truncated_nb`
(M45) all changed **what the model is fed**. **None changed what is optimised** — the loss stayed
single-step in every one. Aceituno's result is about the **horizon of the loss**, not the inputs. So
long-horizon training has never actually been tried here, despite three experiments that resembled it.

`MillerHardt2019_StableRecurrentModels` explains WideMemory's null: a contractive recurrence is in the
**stable regime**, where *"stable recurrent models can be approximated by truncated feed-forward
models"*. We measured a **35.8× state collapse** and an effective influence radius far below the
theoretical receptive field. A stable recurrence does not use memory capacity, so widening it 10×
(4,160 → 44,288 LSTM parameters) buys nothing — and it explains why *freezing* the cell state helps
(M38): it pins a state that would otherwise contract away.

#### A design error of mine, stated plainly

**h18 was the worst available primary endpoint.** It sits almost exactly at the sign change (5 better /
7 worse) and reports "nothing is happening" for a pattern that is unmistakable at h1 and h36. I chose
it by programme convention, not because it discriminates. The pre-registration is otherwise intact —
the verdicts above are the registered rule applied without alteration — but the endpoint choice cost
this experiment most of its resolution, and #301 carries the lesson forward.

#### Status of the cross-arm pattern

**Post-hoc.** It cannot promote anything under §4 and is not offered as a verdict. It is a hypothesis
with an unusually strong sign test (2⁻¹² at two independent horizons), a mechanism, and prior
literature that predicted it before it was looked for.

## Falsifiers

| | result |
|---|---|
| **F1** identity | **PASS** — every arm's `model` re-asserted by `verify_bakeoff.py` independently of the builder's declaration |
| **F2** floor gate | **PASS** 12/12 |
| **F3** setup integrity | **PASS** — `arm_postflight` clean on all 12, including support match to control |
| **F4** seed-matched controls | **PASS** — controls reproduced 0.3298 / 0.3318 |
| **F5** h1 sanity | **PASS** — largest h1 loss is −0.0024 (DualStream s42), well inside σ |
| **F6** mechanism | **PASS** — pinned by `test_candidate_mechanisms.py`, verified to fire against 7 neutering mutations |

## Run notes and process defects

* **`/falsify guard` before launch found 21 of 28 mutations surviving** — verdict FALSIFIED. Every
  candidate's mechanism could have been neutered while the suite stayed green (`WideMemory` at factor
  1 is byte-identical to its control); `choose_model` could be reverted with all 1754 tests green;
  and **two guards had zero call sites** while their docstrings claimed otherwise. All closed and
  re-verified by re-applying each surviving mutation. **That audit is the reason these 24 hours
  measured anything.**
* **`dualfullzero_fortythree` failed instantly** — its directory was missing `artifacts/`, `logs/`,
  `notebooks/`, `reports/`, so `ModelPathManager` raised at import. Its **config was entirely
  correct**, which is why the queue's identity check passed it: **`ensure_arm`'s reuse path validates
  the config but never checks the directory is structurally complete.** Cost was 4 seconds of compute
  but the overnight slot. Rebuilt from scratch and re-run; the queue skipped the 11 completed arms,
  which is the crash-resilience property working in anger rather than in a test. **What emptied that
  directory is unknown and is not reconstructed here.**
* Runtimes 97–154 min/arm against a projected 1.79–2.33 h; total ~24 h, inside the 48 h window.
* Smoke measured real peak memory at **4,136–4,739 MiB** where the synthetic preflight probe reported
  ~20 MiB — a 200× understatement, and the reason that gate was replaced rather than tuned.

## Consequence

**Per-frame spatial sharpness is not the binding constraint.** The placement gap is real
(`spatial_scramble` costs 81% of the oracle) but it is not reachable through the encoder–decoder, and
sharpening the map makes long-horizon rollout worse in proportion.

**The next lever is the horizon of the loss, not the architecture and not the feedback content.**
`Brandstetter2022`'s pushforward trick is the cheapest form — unroll two steps, **cut the gradient
after the first**, take the loss against ground truth at the pushforward time — and its ablations
document two traps in advance: pushforward *with* gradients is **less** stable than without, and
Gaussian-noise injection was **worse than doing nothing**. That is #289. `Zhuang2025`'s Horizon
Forcing (#291) is the fuller version and reports beating teacher forcing, scheduled sampling and
Professor Forcing on chaotic benchmarks.

**Keep the arms.** Five of six are statistically indistinguishable from the incumbent with genuinely
different inductive biases — which is what ensemble diversity requires. ShallowPool is the one to drop.
