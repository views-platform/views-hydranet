# 05c — Pre-analysis plan, EXP-3 (LOCKED)

**Written 2026-09-03, before the run.** Locked by the commit that introduces it.

## 1. The confound this exists to break

M52: clamping the cell state preserves gate–body alignment (66.6 → 69.3 vs 66.6 → 4.3) *and* raises
AP (M48, +0.032 at h18, 4 seeds). Two hypotheses explain that equally well and, as of now, make
**identical** predictions:

* **H-place** — the clamp works because the anchor carries a *map* of where conflict will be large,
  and free-running loses it.
* **H-scale** — the clamp works because it steadies the state's *magnitudes*, and preserved
  alignment is a side effect.

## 2. The intervention

Roll the anchor spatially before pinning it. `torch.roll` is a **permutation**: the arm clamps just
as hard, to a state with identical norm, mean, variance, per-channel distribution and internal
spatial structure — differing only in *where* that structure sits relative to the geography.

Under the roll the two hypotheses predict **opposite** things. That is the whole point.

**One variable:** `freeze_anchor_roll ∈ {3, 15, 90}`, added to the EXP-2 arm
(`identity` + `freeze_recurrent='cell'`). Seed 42, same artifact, same 13 origins, emit-only.

## 3. The dose, and the trap it defends against

A rolled state is off-distribution. A collapse at roll=90 alone could mean "we broke it", not
"placement matters". So the roll is a **dose**: 3 cells (~local), 15, and 90 (half the grid).

* skill degrading **smoothly with distance** → placement is genuinely the carrier
* skill collapsing **identically at every distance** → an off-distribution artifact, and the arm
  proves nothing

Without the dose this experiment could not tell a result from a broken model.

## 4. Readouts and baselines

Baselines already measured on this exact vehicle: unclamped `AP@h18 = 0.3298395823400329`,
clamped `AP@h18 = 0.3621885544392029`. Retained-benefit fraction:

```
B(roll) = (AP_roll − 0.32984) / (0.36219 − 0.32984)
```

`B = 1` means the benefit fully survives a wrong map; `B = 0` means it is entirely destroyed.
**Manipulation check:** alignment in the rolled arms should fall well below the clamped 69.3 —
if it does not, the roll did not do what it claims and no reading follows.

## 5. Pre-registered predictions

| ID | prediction |
|---|---|
| **P1** | `B(90) ≤ 0.3` — most of the benefit is destroyed by a wrong map |
| **P2** | `B(3) > B(15) > B(90)` — monotone in roll distance |
| **P3** | alignment at h36 falls below 30 in the roll-90 arm (manipulation check) |
| **P4** | h1 is identical across all arms — the anchor is set at the origin and read only after |

## 6. Falsifiers (pre-committed)

| ID | fires when | consequence |
|---|---|---|
| **FR-1** | `B(90) ≥ 0.7` | **H-place DEAD.** The clamp's benefit survives a wrong map ⇒ placement content is not the carrier, and M52's story is a side effect. M48 returns to unexplained. |
| **FR-2** | `B(3) ≤ 0.1` **and** `B(15) ≤ 0.1` **and** `B(90) ≤ 0.1`, with no ordering | **INCONCLUSIVE — no claim.** Total collapse at every distance including the smallest is the signature of an off-distribution artifact, not of a placement effect. |
| **FR-3** | h1 differs between any two arms | **HALT.** The roll is acting before the origin, where it must not. |
| **FR-4** | alignment in the roll-90 arm is **not** reduced vs clamped | **HALT.** The manipulation did not take; nothing downstream is interpretable. |

**Grey zone, pre-committed:** `0.3 < B(90) < 0.7` is **PARTIAL** — reported as neither, not resolved
by choosing. Reading order: FR-3, FR-4, then the dose curve, then `B(90)` last.

## 7. What this still cannot do

Rolling breaks **every** spatial correspondence in the anchor at once, not only the gate–body one.
So a null result would establish that the anchor's *spatial content* is load-bearing — ruling out
H-scale, which is the live rival — but would **not** isolate alignment as the sole carrier. That is
a real advance over EXP-2 and is still short of a mechanism.

Seed 42 only.
