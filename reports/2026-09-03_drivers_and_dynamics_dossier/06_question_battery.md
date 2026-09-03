# 06 — Question battery for the driver/dynamics programme (FOR REVIEW, not locked)

**Drafted 2026-09-03.** Nothing here is committed until the chair reviews it and it is split into
locked pre-analysis plans. Questions are written so that **each has an outcome that would refute it**,
or at minimum a result that moves the posterior in a stated direction.

## The three drivers and the two heads

At each rollout step the model holds three things, and emits two:

| driver | what it is |
|---|---|
| **I** | the fed-back input — what the model itself emitted at *t−1* |
| **H** | the hidden state (short-term half) |
| **C** | the cell state (long-term half) |

| head | what it emits | truth-referenced score |
|---|---|---|
| **gate** | `P(y>0)` — *whether* | AP, Brier |
| **body** | `mu` — *how much* | `crps_events`, `size_ratio` |

**A rule this programme adopts because of C-319:** every question below declares whether its measure
is **INTERNAL** (a property of the emitted field) or **TRUTH-REFERENCED**. An internal measure can
never close a causal claim about a score. EXP-3 proved that the hard way: alignment, occurrence and
magnitude were all identical between a good forecast and one displaced 90 cells.

---

## Block 0 — Gating checks. If these fail, the blocks that depend on them are not interpretable.

**Q0.1 — Does the model have any escalation skill to preserve?**
*Measure (TRUTH-REFERENCED):* among cells active at h1, the rank correlation between predicted change
`mu(h) − mu(1)` and true change, per horizon, free-running.
*Why it gates:* Block C asks whether freezing *suppresses* escalation. If there is no escalation skill
in the first place, "does freezing destroy it" is unanswerable and must be reported as such rather
than as a null.
*Prediction:* weak but non-zero at h6, decaying. *Refuted if:* |rho| < 0.05 at every horizon — then
Block C is reported UNANSWERABLE, not negative.

**Q0.2 — Is a single-step swap large enough to measure against noise?**
*Measure (both):* the effect of swapping **all three** drivers at once, versus the seed-to-seed and
donor-to-donor spread.
*Why it gates:* every attribution in Block A is normalised by this. If the all-swap effect is not
comfortably above the donor spread, no component share is resolvable.
*Refuted if:* the all-swap ΔAP is within the donor interquartile range of zero.

---

## Block A — Attribution. Which driver moves the result, per head, per horizon?

**Method for the whole block:** at step *h*, replace exactly one driver with the same driver from a
**different origin at the same step**, recompute that one step, and score it. No propagation, so no
compounding and no confound. The donor is a real state the model produced, so nothing is
off-distribution. 13 origins ⇒ 12 donors per step ⇒ an interval, not a point.
**Normalised share:** `Δ_component / Δ_all`, so the three are comparable despite different units.
**Reported at h = 1, 6, 12, 18, 24, 36.**

**Q A.1 — For the GATE: does the fed-back input or the recurrent state dominate, and does that change with horizon?**
*Measure:* ΔAP (TRUTH-REFERENCED) and Δ gate-field (INTERNAL), swapping **I** vs **(H,C) together**.
*Prediction:* the state dominates and its share **grows** with horizon — M54 found the emitted field
follows the rolled memory at r ≈ 0.90 while the input was never rolled.
*Refuted if:* the input's share ≥ the state's at any horizon ≥ 12. That would overturn M54's reading.

**Q A.2 — For the BODY: same question — and is the answer *different* from the gate's?**
*Measure:* Δ`crps_events` (TRUTH-REFERENCED) and Δ`mu` field (INTERNAL).
*Why it is the most interesting question here:* the body reads the *last magnitude* directly, so it
could be **input-dominated while the gate is state-dominated**. A dissociation would say the two
heads fail for different reasons and need different fixes — and would explain why every
state-intervention so far has moved the gate and left the body at `size_ratio` 0.
*Prediction (weak, stated as such):* the body is more input-dominated than the gate at every horizon.
*Refuted if:* the body's input share is ≤ the gate's input share at every horizon.

**Q A.3 — Within the state: hidden or cell, for the GATE?**
*Prediction:* cell dominates — M39 attributes 89% of the freeze effect to it.
*Refuted if:* the hidden share ≥ the cell share at h ≥ 12.

**Q A.4 — Within the state: hidden or cell, for the BODY?**
*Genuinely open.* No prior result speaks to it.
⚠️ **Carries the C-292 caveat**: `hs = o ⊙ tanh(cl)`, so hidden is a *readout* of cell and swapping
one while holding the other creates a pair the model would never itself produce. **A.3 and A.4 are
therefore weaker than A.1/A.2 and must be reported as the caveated half.**

---

## Block B — What does freezing each half actually mean?

**Method:** four full rollout arms — `none`, `hidden`, `cell`, `all` — scored per head per horizon.
Two are already in hand (`none`, `cell`); two are new.

**Q B.1 — Does the cell's advantage over hidden hold for the BODY, or is it gate-only?**
*Measure (TRUTH-REFERENCED):* `crps_events` and `size_ratio` per horizon, per arm.
*Prediction:* the advantage is gate-only. M55 already shows the cell clamp buys **nothing** for the
body (`size_ratio` 0→0; `crps_events` +0.6% at h36).
*Refuted if:* any freeze arm moves `size_ratio` off 0 at h18 or h36.

**Q B.2 — Is freezing BOTH halves better than the best single half, or redundant?**
*Prediction:* `all ≈ cell` — M39 puts cell at 89% of the effect.
*Refuted if:* `all` exceeds `cell` by more than the seed spread, which would mean the halves carry
complementary information and the M39 decomposition is incomplete.

**Q B.3 — Does freezing ever HURT, and where?**
*Measure (TRUTH-REFERENCED):* per-horizon, per-head deltas vs `none`, looking for negatives.
*Why:* everything so far reports freezing as free. A pinned state cannot track genuine change, so a
cost should exist somewhere — most likely in the body, or at short horizons.
*Refuted if:* no arm is worse than `none` on any head at any horizon beyond the seed spread. That
would itself be a finding: the clamp is strictly dominant and should ship.

---

## Block C — Escalation and de-escalation (the new axis)

*Gated on Q0.1.*

**Q C.1 — Does the model express differential dynamics at all, or does the whole field move together?**
*Measure (INTERNAL):* the spread across cells of `log(mu(h)/mu(1))`. Near-zero spread means one
global trend and no per-cell escalation.
*Refuted if:* spread is indistinguishable from zero — then the model has no dynamics to freeze, and
C.2/C.3 are moot.

**Q C.2 — Does freezing flatten the dynamics?**
*Measure (INTERNAL):* the C.1 spread, per arm.
*Prediction:* freezing the cell pins the long-term map, so predicted trends flatten.
*Refuted if:* spread under freezing ≥ spread under `none`.

**Q C.3 — If it flattens them, does that cost skill — and is the trade worth it?**
*Measure (TRUTH-REFERENCED):* direction-of-change skill (Q0.1's rho) per arm, set against the AP gain.
*Why this is the decision question:* freezing demonstrably buys placement. If it pays for that by
destroying the ability to say *which* places are getting worse, that is a real product cost and it
belongs in the ship decision, not in a footnote.

**Q C.4 — The gate's version: does freezing buy CONTINUATION at the cost of ONSET?**
*Measure (TRUTH-REFERENCED):* split the truth-positive cells at horizon *h* into **continuing**
(active at the origin) and **new** (not active at the origin); compute AP on each subset separately.
*Why this may be the most important question in the battery:* if the clamp works by pinning a map of
where conflict already is, it should be excellent at conflict that *persists* and blind to conflict
that *starts*. Onset is the part of conflict forecasting anyone actually needs. A headline AP gain
that is entirely continuation would be close to worthless for the product, and nothing measured so
far could tell the difference.
*Prediction:* the clamp's gain is disproportionately on continuing cells.
*Refuted if:* the gain on new cells is at least as large as on continuing cells — which would make
the clamp far more valuable than currently claimed.

---

## Block D — Robustness

**Q D.1 — Does any of this replicate across seeds?**
Seeds 42/43/44/45. *None of M51–M55 is replicated*; the pre-registrations make replication a
condition, so any headline from this battery is provisional until this is answered.

**Q D.2 — Does it hold for `ns` and `os`, or only `sb`?**
Everything in this dossier is `sb`. A driver ordering that flips by target would matter.

---

## Cost sketch

| block | GPU |
|---|---|
| A (swap battery, 1 seed) | ~30 min — single-step forwards, not rollouts |
| B (2 new arms × 4 seeds) | ~1 h |
| C | none — analysis on B's dumps |
| D.1 (A and C across 4 seeds) | ~1 h |

**~2.5 h GPU total.** As ever the real cost is the capture/replay harness and its tests, and on this
session's evidence that is also where the errors live.

## Open questions for the chair

1. **Is Q C.4 (onset vs continuation) promoted to the top?** It is arguably the only question here
   that could change the ship decision.
2. **Is Block A worth its harness cost**, given that Block B answers a coarser version of the same
   thing using machinery that already exists?
3. **Seeds now or later** — 1 seed for the whole battery first, or 4 seeds on a narrower set?
