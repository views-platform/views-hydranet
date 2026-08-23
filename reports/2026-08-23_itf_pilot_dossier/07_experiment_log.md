# Experiment log — the ITF pilot (#287)

Pre-registration: `05_analysis_plan.md`, **LOCKED `4cb8953`** before any code existed;
**AMENDMENT 1 `f76e685`** (the "reversed twin" framing corrected) before any arm ran;
**AMENDMENT 2 `ed53dd0`** (F6 fired between arms) recorded when it happened.

---

### EXP-01 · increasing teacher forcing at L=300 · 2026-08-23 · **ITF FAILS TOO**

**Both seeds land below their control by more than 1σ, and both PASSED the floor gate.**

| seed | control | ITF | Δ | σ | floor |
|--:|--:|--:|--:|--:|---|
| 42 | 0.3298 | 0.3125 | −0.0174 | **−1.30** | PASS |
| 43 | 0.3318 | 0.3105 | −0.0213 | **−1.59** | PASS |

§4's registered rule — *both seeds ≤ control − 1σ ⇒ ITF fails too* — returns **ITF FAILS TOO**.

**This is not Teutsch's premature-termination failure.** §5 registered that possibility and made the
floor gate its arbiter: an arm failing FG-A "did not train" and is never reported as "ITF is worse".
**Both arms passed.** ITF trained properly and performed worse.

**The anchor guard did not fire.** h1 moved −0.0018 (seed 42) and −0.0173 (seed 43) — ITF did not trade
one-step skill for horizon skill. It lost a little everywhere.

---

## The reading the pre-registration did not anticipate: it looks like DOSE, not direction

§1 framed this as a test of **curriculum direction**. Placing all three arms side by side suggests the
variable is **mean exposure**, and that direction may be irrelevant.

| | control | **ITF** | SS |
|---|--:|--:|--:|
| ε schedule | 0 throughout | 0.5 → 0 across 300 lessons | 0 → 0.5 over 15, then held |
| **mean ε over training** | **0.000** | **0.251** | **0.487** |
| AP@h18 (2-seed mean) | **0.3308** | 0.3115 | 0.2876 |
| act_ratio@h18 | 0.0087 | 0.0124 | 0.0702 |

**Both endpoints are monotone in mean ε, and ITF sits between the other two on each.** ITF's mean
exposure is almost exactly **half** SS's, and its AP damage (−0.019) is roughly half SS's (−0.043).

Per-horizon, the ordering `control > ITF > SS` on AP and `control < ITF < SS` on `act_ratio` holds at
**h6, h18 and h36 on both seeds — 12 of 12 orderings.** One exception, at h1: `act_ratio` orders
`ITF (0.3711) < control (0.3981) < SS (0.4978)` on seed 43. h1 has no feedback, so nothing in the
mechanism predicts an ordering there.

**And the AP ordering already holds AT h1** — 0.4779 / 0.4761 / 0.4502 (seed 42) and
0.4774 / 0.4601 / 0.4435 (seed 43). **h1 is the step with no feedback at all**, so the three arms
differ there only by their trained weights. The curricula did not merely destabilise the rollout;
they left the model measurably worse at the one-step task it was still teacher-forced on. Whatever
exposure buys, it is charged against the weights, not just against the trajectory.

**This extends M31 rather than contradicting it.** M31: scheduled sampling *fixed* the zero collapse
(`act_ratio` up 9.4× at h18) and lost AP at every horizon. ITF does the same thing at **half the dose**:
`act_ratio` up ~1.4×, AP down about half as much. **The more the model trains on its own output, the
more it fires and the worse it places** — and reversing the ramp changes the *amount* of exposure, not
the sign of its effect.

### ⚠️ This is a hypothesis the data suggests, NOT a finding

* **The dose comparison is confounded.** ITF and SS differ in mean ε **and** in schedule shape
  (ramp vs constant) — AMENDMENT 1 already demoted ITF-vs-SS to descriptive for exactly this reason.
  Three points that happen to be monotone in one covariate do not isolate it.
* **Two seeds**, one vehicle, one lesson count.
* Mean ε is a crude summary of a schedule; two arms can share it and differ everywhere.

**The cheap test that would settle it:** an SS arm at `ε_max ≈ 0.25` — constant, same mean exposure as
ITF, opposite shape. If it lands where ITF landed, dose explains the ordering and direction is
irrelevant. **~5 GPU-h per seed**, and it needs no new code: `ss_epsilon_max=0.25` with `reverse=False`
is the existing path.

---

## §7 (C-307) — false-negative mode and reopen trigger, registered BEFORE the result

**False-negative mode.** **ε started at 0.5, not 1.0** — a softened ITF chosen in §5 to protect against
the training risk. **This null cannot distinguish *"ITF fails"* from *"we did not run real ITF"***, and
it is half the paper's method while the paper's 16–81% gains are reported for the full one.

**Reopen trigger — any of:**

* **the dose hypothesis above is tested and dose does NOT explain the ordering** ⇒ direction is back on
  the table and full ITF (ε=1.0, with §5's floor-gate abort) becomes the real test;
* **#294** proceeds — aGTF anneals α **downward from 1**, which *is* an increasing-TF curriculum, so the
  two are the same experiment from different directions;
* anyone wants the training-time counterpart of **M38/M39/M41**.

**"CLOSED" means *screened out, here is what brings it back* — not settled.**

## Scope

2 seeds, one vehicle, L=300, `sb`, AP and `act_ratio` only. The `crps_all` ARTIFACT verdict (#263) is
untouched. **A 2v2 is a screen, not a measurement** — its exact one-sided permutation floor is 0.167,
which is why §4 is a direction-and-magnitude rule and the rendered verdict says so.

## Run notes

* Arm 1 completed in 55 min. **Arm 2 was aborted by F6** because I committed the arm-naming fix while
  the queue was running; the `views_hydranet` tree hash was identical at both HEADs, recorded in
  AMENDMENT 2 rather than silently overridden.
* Arm 2's first attempt then **died at lesson 223 with CUDA Xid 62 → 45** — a GPU micro-controller halt
  while the laptop was physically moved under load. **No partial artifact was left**, which is the
  dangerous case. Re-run clean from scratch; no new Xid events since the machine came to rest.
