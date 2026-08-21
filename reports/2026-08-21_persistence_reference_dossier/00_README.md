# Is persistence still the thing to beat? — a re-reference at L=300

**Status: SCAFFOLD.** No arm has run. The pre-registration is not written, and it must not be
written until §3 is settled.

## The question

**M1** is the hardest line in the ledger: *on occurrence (AP), persistence beats every arm from h6 on.*

| | h6 | h18 | h36 |
|---|--:|--:|--:|
| persistence (M1) | 0.112 | **0.108** | 0.083 |
| best held arm (M1) | 0.087 | 0.091 | 0.069 |
| free-running (M1) | 0.028 | **0.007** | 0.008 |

It is why the ledger's §*What is NOT established* says the long-horizon rollout has **not**
demonstrated value, and why further rollout work is supposed to be justified as research toward a
future capability rather than as improving something that works.

**But M1's free-running control is 0.007 at h18, and our L=300 control is 0.3298** — 47× higher.
M1 was measured on a different vehicle with a different origin set, so the two columns cannot be
placed side by side. **We do not currently know whether a 300-lesson model beats persistence.**

That single fact decides whether the last ~40 GPU-hours were remediation or research.

## What this is NOT

⚠️ **Not "just run the scorer with `--persistence`".** That was my first framing and it is wrong
twice over.

### 1. The cubes are gone

Score-then-delete (adopted after the disk-fill scar) means **zero prediction cubes survive** —
`find views-models -type d -name 'predictions_*'` returns nothing. The **weights do** survive
(e.g. `models/fullzero_fortythree/artifacts/calibration_model_20260821_045948.pt`, `weight_sha256`
recorded in each `arm_*.json`), so this is a **re-emit**, not a re-train: roughly 10 min/arm on the
observed oracle-pass rate, ~1–2 h for the control set. Cheap, but not free, and it needs the GPU.

### 2. The reference's S must match the arms' — and `_persistence_gathered` is S=1

This is the **#263 method rule**, already paid for once:

> the reference's S MUST equal the arms' cube width — `crps_ensemble`'s `2/(m*m)` bias does not
> cancel across unequal S, and **AP is quantised to S+1 rank levels**.

`_persistence_gathered` (in `rollout_skill_score.py`) builds exactly a **1-sample "distribution"**.
`scripts/rollout_ruler_core.py:65` `assert_sample_cube` exists to refuse this, and says so:

> ``_persistence_gathered`` builds exactly such a 1-sample "distribution", so the degenerate path
> is reachable from the existing ruler and has never had a test.

Our arms are S=16 cubes. **A naive persistence comparison puts a 2-rank-level forecast against a
17-rank-level one and calls the difference skill.** On CRPS the same mismatch already produced the
**ARTIFACT** verdict in Epic #263.

**So the first deliverable is a decision, not a number:** how to reference persistence at matched S.
Candidates, unassessed — (a) a degenerate S=16 cube repeating the persistence value, which fixes the
AP quantisation but not the CRPS bias term; (b) an empirical persistence *distribution* over recent
history; (c) restrict the claim to a rank-invariant statistic where S cannot bite. None is obviously
right and the choice must be locked before any arm runs.

## Sequencing — #281 does NOT gate this. Measure first.

An earlier draft of this file said the design might be too underpowered to run the experiment,
citing "arm-vs-persistence gaps of ~0.02". **That number was imported from the wrong experiment** —
it is M1's *best held arm* vs persistence on the *collapsed* vehicle (0.091 vs 0.108). The
comparison here is **L=300 free-running vs persistence on our own origins**, and the arm side is
already measured at **AP h18 = 0.3257**. If persistence is anywhere near M1's 0.108 the gap is
**~0.22, four times the 0.0541 MDE.** Power is not the binding constraint.

**Measure persistence at S=1 first.** The S=1 handicap (2 rank levels ⇒ tie-heavy ⇒ AP understated)
biases *in our favour*, which is what makes the cheap version informative:

| outcome | what it costs us next |
|---|---|
| **we LOSE to a handicapped persistence** | decisive. No matched-S work, no #281. The programme's framing is settled and the news is bad |
| **we win NARROWLY** | both the matched-S fix and #281 bite. Do them |
| **we win HUGELY** | direction is safe; matched-S only sizes how much of the margin is real |

One measurement says which of the three worlds we are in, and only one of them needs #281.

**Cost: ~10 min of GPU, not zero.** `_support_keys` reads `identifiers.npz` from the deleted
prediction dirs, so persistence cannot be built from truth alone — it needs **one control arm
re-emitted** to recover the support set, after which persistence is CPU-only. That re-emit also
regenerates a cube for any later matched-S work, so nothing is wasted.

1. Re-emit one ε=0 L=300 control → recover support. 2. Score persistence at S=1 against it.
3. **Branch on the table above.** 4. Pre-register whatever step 3 selects. 5. Log, negatives included.

## Related

- Ledger **M1**, §*The persistence re-reference, 2026-08-17*, §*What is NOT established*
- **#280** — floor-limited vehicles (M1's own vehicle is in scope)
- **#281** — the design cannot resolve effects this size
- Epic **#263** / `scripts/rollout_ruler_core.py` — where the matched-S rule comes from
