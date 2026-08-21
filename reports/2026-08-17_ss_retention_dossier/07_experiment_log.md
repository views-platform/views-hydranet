# Experiment log — scheduled sampling and rollout retention

Append-only. Falsifier verdicts recorded **before** predictions are read.

---

### EXP-01 · the sweep at L=300 · 2026-08-21 02:01 → 14:34 · **SS MAKES THE ROLLOUT WORSE**

- **Plan:** `05_analysis_plan.md` LOCKED 2026-08-17, AMENDMENT 1 (L=160 → 300) and AMENDMENT 2 (the §4
  guard implemented) 2026-08-21. Rule `scripts/ss_sweep_gate.py`, md5
  `d1432db9a7611cf349f1009225365027`, 22 tests.
- **Arms:** 4 × ε=0 and 4 × ε=0.5, seeds 42–45, L=300, one variable. `fullzero_fortytwo` came from the
  lesson curve and was not retrained. 7 arms in 12.5 h, **zero failures**.

#### Falsifiers — recorded first

F3 `N`=170430 and 13 origins on every row. F4 no shared AP. F5 no shared weight hash. F6 one
`views_hydranet` tree hash (`ca41c3f5`) across all arms. Post-hoc floor gate: **all four controls PASS
FG-A**. Single dose, single lesson count. **0 blocking.**

#### Result

| seed | AP h18 ε=0 | AP h18 ε=0.5 | retention ε=0 | retention ε=0.5 |
|--:|--:|--:|--:|--:|
| 42 | 0.3298 | 0.3064 | 0.6902 | 0.6805 |
| 43 | 0.3318 | 0.2687 | 0.6951 | 0.6060 |
| 44 | 0.3058 | 0.2528 | 0.6372 | 0.5622 |
| 45 | 0.3352 | 0.3044 | 0.7107 | 0.6723 |
| **mean** | **0.3257** | **0.2831** | **0.6833** | **0.6303** |

**Exact one-sided permutation p = 0.0286.** mean ΔAP(h18) **−0.0426**, mean Δretention **−0.053**,
endpoints agree in sign. **All four seed-pairs move down on both endpoints.**

**The anchor guard PASSES**: mean ΔAP(h1) = **−0.0277** against a limit of 0.0440 (63%). So this is a
retention result, not a traded failure — the clause exists precisely to make that statement earned.

#### Verdict: UNDERPOWERED — and the reason matters

`p ≤ α` **but the drop does not clear `3 × MDE` (0.0426 vs 0.0541)**. The DIRECTION is established; the
MAGNITUDE sits inside the resolution this design can assert. Not an EFFECT by the pre-registered rule,
and not a NULL either — a null requires `p > α`.

*(The verdict text originally printed the NULL branch's "the interval does not exclude a 30% effect"
for this case — a false explanation of the same class the lesson-curve gate had. Fixed to name the
real blocker before the number was reported to anyone.)*

#### The finding nobody predicted

**Scheduled sampling largely FIXED the zero collapse, and skill fell anyway.**

| h | act_ratio ε=0 | act_ratio ε=0.5 | ratio | ΔAP |
|--:|--:|--:|--:|--:|
| 1 | 0.382 | 0.470 | 1.2× | −0.029 |
| 6 | 0.103 | 0.273 | 2.7× | −0.040 |
| 18 | 0.0093 | 0.0875 | **9.4×** | −0.050 |
| 36 | 0.0007 | 0.0204 | **28×** | −0.032 |

`act_ratio` ≪ 1 *is* the #258 collapse. SS moved it 28× toward calibration by h36 and **lowered AP at
every horizon**. That directly corrects **M15**'s framing — *"the model does not need to be nearly
right; it needs to answer"* — which has been load-bearing since 2026-08-17. It answered, and got worse.

---

### EXP-02 · placement probe · 2026-08-21 15:49 → 16:47 · **the damage splits ~50/50**

Inference-only, 6 arms on two frozen seed-42 artifacts differing in exactly `ss_epsilon_max`. Four
predictions registered before any arm ran; **P1 and P2 both FAIL**, and that is the informative outcome
— they were written with a deliberate gap (>60% vs <30%) so that "both, roughly equally" could not be
retrofitted into either.

| model | control | ceiling | `occurrence_real_magnitude_model` | `spatial_scramble` | `thin:0.75` |
|---|--:|--:|--:|--:|--:|
| ε=0.0 | 0.3298 | 0.4974 | 0.4888 | 0.0925 | 0.4807 |
| ε=0.5 | 0.3064 | 0.4825 | 0.4756 | 0.0953 | 0.4709 |

```
total rollout gap (eps=0 - eps=0.5) at h18      +0.0234
  residual once BOTH get perfect occurrence     +0.0132  (56%)  the model itself is worse
  attributable to the field SS emits            +0.0102  (44%)  worse placement
drop in the CEILING (oracle h18)                +0.0149  <- independently ~the same number
```

Two routes to the residual — hand-it-perfect-occurrence, and the oracle ceiling — agree to 0.002. SS
made the model slightly worse *as a model*, and that part travels with it whatever field it is given.

**P3 HOLDS** — `spatial_scramble` falls below both controls (−0.237, −0.211): destroying placement is
worse than either model's own output (M12 replicated on a third pair).

**P4 HOLDS, and it kills a hypothesis.** `thin:0.75` recovers **90% (ε=0)** and **93% (ε=0.5)** of each
model's own gap. **The SS model uses a good field slightly BETTER than the control.** So *"SS damaged
the model's ability to use its input"* is **falsified**. Hand it a well-placed field and it is fine.

#### What it adds up to

**Scheduled sampling traded quiet-and-wrong for loud-and-wrong.** It fixed the under-firing symptom,
placed the extra cells worse, and placement is what the rollout runs on. Necessary-but-not-sufficient:
*answering* is required, but **where** you answer is what decides the score.

#### Theory this is consistent with

Huszár (2015), *How (not) to Train your Generative Model*: scheduled sampling is a **statistically
inconsistent** estimator — its objective is not minimised by the true data distribution even with
infinite data. Two mechanisms specific to this setting: the SS target is partly **unlearnable** (the
model is asked for the real next month from its own degraded field, and that information is not in the
input), and the **compounding regime is wrong** — a Bernoulli mask at ε=0.5 gives an expected run of
2 consecutive synthetic steps against the 18–36 inference faces.

#### Scope

One seed for the probe (the sweep's significance rests on four); one vehicle; one dose; target `sb`;
h\*=18; calibration partition. `spatial_scramble` inherits C-291's confound. Per §3.1 none of this
settles what the roster showed — those models trained under a configuration C-259 now forbids.

---

