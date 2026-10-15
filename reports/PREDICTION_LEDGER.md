# Prediction ledger — the assistant's own base rate

**Started 2026-10-15**, at Simon's request: *"why almost 100% of the times your predictions are
wrong? should we be logging this because I might be ignoring when you are right, but the amount of
times you are wrong is remarkable."*

He is right that the base rate was invisible. Corrections get announced; silent successes do not. So
this file exists to replace both of our characterisations with a count.

**Scope.** One row per **checkable claim stated before it was checked**. Not opinions, not
recommendations, not hedged statements — claims with a truth value that a measurement, a file read or
a derivation could settle. Both outcomes are logged. A ledger that records only errors is a
confession, not an instrument.

**Protocol.** When I state a checkable claim without having checked it, it gets a row the same day
with `outcome: OPEN`. When it is settled, the row is closed. Rows are never deleted or softened;
newest at the bottom of each block. If the ledger is not being maintained, that is itself a finding
about the workflow — the same rule the experiment log lives under.

---

## The base rate so far

| | count |
|---|---|
| **WRONG** | **22** |
| **RIGHT** | **12** |
| OPEN | 1 |

⚠️ **That raw ratio is the least useful number in this file.** The errors are not randomly
distributed, and the split by *kind* is the actionable result:

| class of claim | right | wrong | read |
|---|---|---|---|
| **Derivations / algebra / arithmetic** | **10** | **1** | Closed-form results have held (three checked line by line by an adversarial panel). ⚠️ The one miss, **W-21**, was not algebra but a **statistic**: applying a per-step identity to a step-averaged quantity. |
| **Claims about what exists in the repo** (a log, a test, a prior experiment, a paper's contents) | 0 | **6** | ⛔ **The worst class, and the cheapest to prevent** — every one would have been caught by a single `grep` or file read taking seconds. |
| **Causal narratives about system behaviour** | 0 | **8** | ⛔ Wrong in **sign or magnitude**, not in kind. Coherence was being mistaken for evidence. |
| Operational / tooling | 0 | 4 | Minor, self-caught, cost only wall clock. |
| **The programme's central diagnosis** | **1** | 0 | Survived its hardest adversarial test (see R-6). |
| **Pre-registered predictions about a measurement's own behaviour** | **2** | 0 | R-11, R-12 — both committed before the data and both held. |

### The two findings that matter

**1. ⛔ Six errors were file reads I did not do.** The per-task log sink, the test that pins the
positive-cell denominator, Kendall's Appendix B, `results_balancer_bisect.md`, C-119's resolved
status, the Cragg PDF. **This is the single largest cluster and the cheapest to eliminate**: check the
artifact before describing it, always, because the check costs seconds and the error costs a
committed document. Two of these reached a committed pre-registration.

**1b. ⛔ Mean-of-ratios vs ratio-of-means, three times** (W-2, W-8, W-21). Each time I collapsed a
distribution to one summary statistic and then used it as if it were a multiplier. **W-21 is the
clearest: the identity was exactly right per step and wrong after averaging**, and it failed worst on
the rarest target — the case where a random denominator is most often 0 or 1. Rule: before projecting
with a ratio, check whether the denominator varies; if it does, compute the projection on the raw
per-unit values, never on their median.

**2. ⛔ The narrative arrived before the measurement, and it was wrong in sign.** "The body went
quiet" against a measured 35× louder. "μ̂ biased low" against truncation raising it. "The denominator
is a defect" against a deliberate, tested guarantee. The shape is always: form a coherent mechanism →
state it at the confidence the *story* deserves → measure → discover the sign is inverted.

### The rule, adopted 2026-10-15

> When I have a mechanism but not a measurement, the word **unverified** goes in the sentence, not in
> a caveat three messages later. Preferred output is *"I don't know yet — here is the measurement that
> would settle it"*, not a mechanism.

### ⛔ Amended 2026-10-15, after W-22 broke the rule the same day it was written

The rule above was not enough, because W-22 was not a mechanism claim — it was a **readiness** claim:
*"the pod can run as soon as X lands."* A statement that something is ready to execute asserts the
state of every link in its chain. **So: before saying any work is ready to run, open the thing that
will run it.** Not the config, not the library — the launcher, the installer, the pin. W-22 was
caught only because briefing a peer forced me to describe the deployment path precisely, which is an
accident, not a process.

### What actually caught the errors

| caught by | count |
|---|---|
| my own later measurement or derivation | 9 |
| the 7-seat adversarial method review | 6 |
| **Simon** | **1** (W-8) |
| self-caught immediately | 4 |

The method review found **three of the four most serious errors** for the cost of seven agents.
External adversarial checking beats self-review here, and it is cheap relative to a wasted arm.

---

## WRONG

| ID | claim, as stated | error type | settled by | note |
|---|---|---|---|---|
| W-1 | "The offline wandb logs will have the per-task loss split." | existence | my own read of `training_engine.py` | They do not. `lesson_reg`/`lesson_cls` go only to the opt-in `trajectory_log_path`, unset on both runs; `run.log` carries the total only. |
| W-2 | "The family swap amplified the body term **290×**." | population | my own recomputation | 0.003448 is the all-land positive rate; training reduces over activity-seeking windows (median 0.004508 → 1/222) and the realised loss ratio is **65.5×**. |
| W-3 | "The body had nothing to say to the trunk." | **sign** | r2 | Inverted. The body term became **35.2×** the gate term, not quiet. |
| W-4 | "μ̂ will be biased low for the job we are giving it." | **sign** | truncation algebra | Truncation *raises* the location to μ/(1−p₀). |
| W-5 | "The positive-cell denominator is a defect." | existence | reading `test_truncated_negative_binomial.py:104-126` | Deliberate, documented as "the defining guarantee #3", pinned by a test, and implemented independently twice. |
| W-6 | "B carries a heavy comparability cost." | magnitude | my own re-read of C-112/C-119 scope | Unfreezing on one experimental model changes no roster model and no logged roster result. |
| W-7 | "B relies on untested machinery." | existence | Kendall Appendix B | Init-robust over log σ² ∈ [−2,5], noise terms converge in ~100 iterations. Our C-312 trouble was the guard bug. |
| W-8 | Reported pooled task-loss magnitudes as if they characterised the three targets. | aggregation | ⛔ **Simon** — *"Have you taking into account here the difference between the three targets"* | I had not. r3 then showed a 2.3× spread the pooled figure hid. |
| W-9 | Changed position on A-vs-B three times in one conversation (A → B → A). | process | self | Each move had a named cause, but the oscillation signalled the decision was not well-determined and I presented each position with more confidence than that warranted. |
| W-10 | "Logging the `log_var` gradients is a free diagnostic that reveals the direction B would move." | redundancy | my own algebra, while writing the prereg | At v=0 the gradient is `−L/(r+1) + 1/2` — a deterministic function of L, which `trajectory_log_path` already records. No information. Also the raw `.grad` is a never-zeroed cumulative sum. |
| W-11 | "The joint hurdle NLL is unreachable under the frozen balancer without a second variable." | existence | Kendall seat | With `is_regression=False` the coefficient is exactly 1.0, so A + that flag **is** the exact joint hurdle NLL. |
| W-12 | "B is the pre-registered fallback; our prior trouble was the guard bug, not the method." | ⛔ existence | operational seat | `reports/results_balancer_bisect.md` is a pre-registered, device-matched, 3-seed GPU bisect finding that **un-freezing the balancer IS the cause of the C-113 runaway**. The register's disposition is *"regularise, not permanent freeze"*. I had cited Kendall's Table 1 over our own experiment. |
| W-13 | Sized falsifier F8 on C-119's "~20% training variance". | stale | operational seat | C-119 is **RESOLVED**; that figure is the pre-fix measurement. Reached a committed pre-registration. |
| W-14 | Sized F2/P-3 on the 0.0413 month-to-month sd. | magnitude | operational seat | `OUTCOME.md` already demonstrates ±0.010 paired-by-month resolution — 4× finer. Reached a committed pre-registration. |
| W-15 | "A leaves `lr_ns_best`'s magnitude effectively unsupervised, ~9× below the control." | framing | my own V3/V7 computation | Backwards. Zeros supply 51.7%/88.0%/67.8% of the *control's* body term, so the control's near-equal per-target ratios are the artefact; A's spread is the likelihood being honest about information content. |
| W-16 | Cited **Cragg (1971)** for hurdle-likelihood separability. | miscitation | Murphy seat, then my own PDF read | Cragg never states separate estimation (only *"ML estimates may be calculated by iterative methods"*, on the full likelihood) and his model II is a continuous **truncated normal**. **Mullahy (1986)** states separability verbatim three times and is the right model class. |
| W-17 | "The term scale is the allocation rule for shared capacity" — implying the shared trunk is most of the model. | magnitude | my own state-dict analysis | Only **11.29%** is shared (102,240 of 905,289). Body and gate have separate ~400k-param decoders, and under AdamW a per-task constant is *exactly* invisible on private parameters. |
| W-18 | Launched a long job under `conda run` expecting streamed output. | operational | self | `conda run` buffers until exit; the monitor could not have fired. 25 min of blind wall clock. |
| W-19 | Ordered the BPTT chunk sweep K = (1, 8, 24). | operational | self | Backwards. Cost is `(steps/K)` chunks × 6 backwards, so K=1 is ~23× the calls of K=24 **and** least faithful to production. Ran the most expensive, least useful block first. |
| W-20 | On the control's `cos = +0.7244`: *"the pre-committed trigger fired"* (i.e. → N). | premature | the arm block, 20 minutes later | The arm gave `cos = −0.2033`, 75% of windows negative. A control-only reading would have sent the fork to N and been wrong. ⚠️ Mitigated: I did explicitly withhold the verdict pending the arm — but I framed the trigger as fired. |
| W-21 | A's per-target magnitudes under the new denominator: **0.295 / 0.058 / 0.166** (prereg §4 P-2). | ⛔ **statistic** | the pre-flight, r5 | **`ns` missed by +202%** (0.058 predicted, **0.1753** measured). The projection used `body_A = body_positives × (P/N)` with the median positive *fraction*. That identity holds **per step**, not after averaging over steps: `positives` divides by a per-step count that is often 0 or 1, so `mean(S/P)` is dominated by small-P steps while `mean(S/N)` is not. **Mean-of-ratios vs ratio-of-means — the third instance of this error class in this programme** (cf. W-2, W-8). It bit hardest on the target where P is smallest, which is exactly where the two diverge most. ⚠️ `sb` +1% and `os` +23% were fine, and the **pooled** P-1 held (+23%), so only the rarest target was badly wrong. |
| W-22 | *"The pod can run as soon as that [views-models config] edit lands."* | ⛔ **existence** — the 7th in that class | reading `tools/podrun/pod_run_model.sh` while briefing the `models` session | **Wrong, and it would have cost a silent no-op arm.** `:171-173` installs `views-hydranet~=0.1.1` **from PyPI**. The new key exists only on an unpushed local branch, `HydraNetConfig` sets `extra="allow"`, and `reject_near_miss_keys` only catches keys within one edit of a real field — so `body_nll_denominator` would be **silently accepted and ignored**, and the arm would come back byte-identical to its own control. A $2.25 confounded "null" with the C-324/C-331 signature. Also `:236` clones views-models at `development`, so a feature-branch config edit would not reach the pod either. ⚠️ **I asserted pod-readiness without opening the pod runner** — the same failure mode as W-1, W-5, W-7, W-12, W-13, W-16. |

---

## RIGHT

| ID | claim, as stated | settled by | note |
|---|---|---|---|
| R-1 | The shared fraction of the model is ~11.3%. | the live model's `named_parameters()` | **11.29%.** Predicted from state-dict key names, confirmed by a different method. |
| R-2 | Under AdamW a per-task loss constant is invisible on that task's private parameters. | direct numerical test | **max\|difference\| = 0.0** over 60 steps at scale 1.0 vs 65.5. Exact, not approximate. |
| R-3 | The balancer's stationary point is `v* = log(2L/(r+1))`, giving `coeff* = 1/(2L)` and a weighted contribution of exactly 0.5 nats per task, so B's fixed point equalises all six. | Kendall seat, independently | *"Your derivations 1, 2, 4, 5 are all algebraically correct; I checked them."* |
| R-4 | A's projected pooled ratio is 0.225. | recomputation from the raw `r3_result.npz` arrays | Reproduced to 0.2254. |
| R-5 | If the `+0.72` cosine were a common-mode artifact, **all** pairs would be positive. | the pairwise table | Off-diagonals came back **negative**, as predicted — so the cosine structure is real signal. A risky prediction that could have gone the other way. |
| R-6 | ⛔ **The central diagnosis: the loss-magnitude imbalance propagates into shared-trunk allocation.** | r4, against three seats arguing N | Loss ratio 0.593 → 35.2 (59×); **shared-trunk gradient-norm ratio 0.312 → 10.497 (34×)**; `cos(∇body,∇gate)` **+0.72 → −0.20**, 0% → 75% of windows in conflict. Survived its hardest adversarial test. |
| R-7 | The 0.5 body coefficient is structural and a regression task can never receive 1.0. | `mtloss.py:65` | `1/((is_regression+1)·1)` = 0.5 for `True`, 1.0 for `False`. |
| R-8 | `weighted_nll_mean` returns a graph-connected exact 0 when no cell is supervised. | `nb_core.py:62-79` and its docstring | Confirmed verbatim. |
| R-9 | The positives denominator is frequently 0 or 1 for the rarest target. | measurement from the delivered arm's own per-window fractions | `lr_ns_best`: **median 0.88 positive cells per window**, 61% of windows averaging under 1. |
| R-10 | Lower NB `θ` means **more** over-dispersion, so θ̂ 0.154 vs 0.432 is over-dispersion. | `nb_core.py:3-5` | `θ` is `total_count`, `Var = μ + μ²/θ`. Used to refute a seat's claim that had the direction backwards. |
| R-11 | ⛔ **O-1 (was open): the `cos(∇body,∇gate)` sign is stable across BPTT chunk length.** Pre-registered before the run, with "a sign that moves with K voids the test". | the K=8 block | **Confirmed, and far tighter than required.** Tripling BPTT depth moved the cosine by **0.003** on the control (+0.7244 → +0.7216) and **0.023** on the arm (−0.2033 → −0.1807); norm ratios 0.312→0.325 and 10.497→9.436. Not a truncation artifact. |
| R-12 | Dropping the K=1 block costs nothing. | the K=24-vs-K=8 agreement | Stated as a recommendation *before* K=8's arm block landed, i.e. at risk. A 3× BPTT change perturbing the cosine by 0.003 means the remaining variation K=1 could show was not worth ~3 hours of 13 cores. |

---

## OPEN

| ID | claim | stated | settles when |
|---|---|---|---|
| O-2 | `pos_weight → 1.0` is a better-motivated single-variable arm than A. | 2026-10-15, from the Gneiting seat's G&Ranjan Thm 1 derivation plus P2's measured implied 1.0365 | Not yet tested; not in any pre-registration. |
