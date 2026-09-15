# 07 — Experiment log

Append-only. Every entry links its pre-registration and states which falsifiers fired.
Negatives are entered in full, with a postmortem, and are never softened or omitted.

---

## EXP-1 — silence vs fade, seed 42 · **C1 FALSIFIED** · 2026-09-02

**Pre-registration:** [`05_analysis_plan.md`](05_analysis_plan.md), locked `db20868`, amended `8e3b99c`.
Both committed before any GPU second. **One variable:** `feedback_transform`
(`identity` = free-running treatment, `use_real` = real-data control). Seed 42,
`fullzero_fortytwo`, `calibration_model_20260818_221401.pt`, 13 origins, emit-only.

### The claim under test

> *"The model doesn't make smaller forecasts as it goes forward. It makes fewer of them. The handful
> it still makes are the same size as at month one."*

### Result — it is false

Free-running, h1 → h36, primary target `sb`, over every cell with no conditioning:

| quantity | h1 | h36 | ratio |
|---|---|---|---|
| **occurrence** (mean gate) | 4.146e-03 | 1.492e-04 | **0.036** — 28× fewer |
| **magnitude, plain mean of `mu`** | 0.0961 | 0.0213 | **0.222** — 4.5× smaller |
| **magnitude, gate-weighted** | 6.4035 | 0.0924 | **0.0144** — 69× smaller |

The model fires **far less often AND predicts substantially smaller amounts**. Both halves collapse.
"Fewer, not smaller" is wrong.

**Control (`use_real`)**: occurrence ×1.19, gate-weighted magnitude ×0.91, plain magnitude ×1.12 —
flat across all 36 horizons. So the collapse is a property of free-running, not of the readout.
The two arms are *identical* at h1 (4.146378e-03 / 6.4035 / 0.0961), as they must be before any
feedback has acted — a free internal consistency check that passed.

### A third finding, not predicted by anyone

The gate's **alignment with magnitude** decays. `gate-weighted mu / plain-mean mu` measures how
concentrated firing is on the cells where the body predicts a lot:

| h | 1 | 6 | 12 | 18 | 24 | 30 | 36 |
|---|---|---|---|---|---|---|---|
| concentration | **66.6×** | 73.1× | 46.8× | 21.5× | 11.3× | 7.3× | **4.3×** |

At h1 the model fires where it expects a lot. By h36 that alignment has largely dissolved: the gate
becomes close to uninformative about where the large values are. This is why the gate-weighted
magnitude falls so much harder (69×) than the plain mean (4.5×) — **the cells that keep firing are
the small ones.** That is survivorship running *downward*, the opposite of the rival R1 predicted.

### Verdict against the pre-registered falsifiers

| ID | fired? | value |
|---|---|---|
| **F1** — magnitude ratio < 0.5 | **FIRED** | 0.0144, far below |
| F2 — unweighted falls while gate-weighted holds | no | both fell; the rival's *specific* signature is absent |
| **F3** — instruments disagree > 10% | **FIRED** | see below; recorded as fired, not amended |
| F4 — control drifts | no | control flat, 0.91 |
| F5 — occurrence does not collapse | no | 0.036, it collapses |
| F7 — anchor | passed | `AP@h18 = 0.3298395823400329`, exact |
| F9 — conditioned magnitude rises with tau | **undefined** | no support at h36; see below |

**Decision rule applied (05 §7):** F1 ⇒ C1 **falsified**; retract M50's mechanism sentence; correct
what rests on it.

### Why the original claim looked true — the tau sweep answers it

The old statistic conditioned on cells that fired. Its support at h36:

| tau | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| cells above tau at **h1** | 3840 | 1417 | 740 | 379 | 125 |
| cells above tau at **h36** | **1** | **0** | **0** | **0** | **0** |

At h36 the conditioned set is empty or a single cell. A mean over one arbitrary cell — or over none,
which is what produced the `-1.0` sentinel — is not a measurement. **This is the direct, measured
explanation of C-318**, and of why the corrected version still misled: filtering the sentinel out
left `n=2 of 156`, and two cells cannot speak for a field.

F9 could not be evaluated: with zero support at four of five tau values, the dose-response has
nothing to respond with. The sweep did its job anyway — it showed the conditioned statistic has no
support, which is the finding.

### F3 — fired, and left fired

The cube reference disagreed with the dump on magnitude by up to 949% at late horizons. Diagnosed as
**noise, not bias**: the dump lies inside the cube's ±2 s.e. at every horizon, per-origin ratios
scatter symmetrically about 1, and the reference's nonzero-draw count collapses 599 (h1) → 13 (h18)
→ 1 (h30) → **0** (h36). It cannot measure what is not there.

Re-running the check on the **control**, where the field stays busy, gives the decisive comparison:

| | treatment | control |
|---|---|---|
| gate, exact check | 0.000e+00 | 0.000e+00 |
| occurrence, worst | 3.5% | 0.8% |
| **mass at h36** | 949% off, **0 draws** | **5.4% off, 796 draws** |

Same code, same horizon index — the dump's magnitude computation is correct at h36. But **F3 fires on
the control too** (2 of 36 horizons, 13.1% and 10.7%, ~620 draws each): the 10% band is tighter than
the reference's own sampling noise at S=16, so it fires on a case that is demonstrably fine.

That is a defect in the pre-registration, demonstrated by measurement rather than argued. The
statistically correct form — agreement within the *reference's own* uncertainty — was available when
the plan was drafted and was not written; the dump passes it 4/4 on the treatment. Adopting it after
seeing results would be the post-hoc override already on the register, so **F3 stands fired** and
the chair authorised proceeding with it on the record.

### What now replaces the claim

> During free-running the emitted field collapses on **both** axes: the model fires ~28× less often
> **and** the amount it would predict falls ~4.5×. Beyond that, the *alignment* between firing and
> magnitude decays 67× → 4.3×, so what survives is disproportionately the small stuff.

**Artifacts:** `results/EXP1_READOUT.txt`, `results/g1/g1_identity_TREATMENT.txt`,
`results/g1_control/g1_use_real_CONTROL.txt`, `results/bodymean_fullzero_fortytwo_{identity,use_real}/`.
**Not yet replicated** — seed 43 is the next step, and 05 §7 makes replication a condition of the
finding, not an optional extra.

---

## EXP-2 — does the cell clamp restore the alignment? · **H SURVIVES (weakly, by design)** · 2026-09-03

**Pre-registration:** [`05b_analysis_plan_exp2.md`](05b_analysis_plan_exp2.md), locked `31f14af`
before the run. **One variable:** `freeze_recurrent='cell'` added to the `identity` arm. Seed 42,
same artifact, 13 origins, emit-only, 355.6 s.

### Gates, in the registered order

* **F-B (anchor)** — `AP@h18 = 0.3621885544392029`, the archived M48 `cell` value, **exact**. PASS.
* **F-C (h1 identity)** — occurrence, gate-weighted and plain magnitude all **identical to 9 s.f.**
  between clamped and unclamped at h1. The clamp acts only for `t > origin`, as it must. PASS.
* **F-A** — **did not fire.**

### Result

`A(h) = gate-weighted mu / plain-mean mu`, the alignment between where the model fires and where it
predicts large values:

| h | 1 | 6 | 12 | 18 | 24 | 30 | 36 |
|---|---|---|---|---|---|---|---|
| unclamped | 66.6 | 73.1 | 46.8 | 21.5 | 11.3 | 7.3 | **4.3** |
| **clamped** | 66.6 | 67.1 | 66.1 | 66.0 | 62.6 | 64.2 | **69.3** |

**Recovery fraction R = 1.042** — fully restored, against a pre-registered bar of 0.25 for support
and 0.07 for death. The clamped curve does not decay at all; it is flat within noise across 36
horizons and ends marginally above where it started.

Both collapses are also substantially arrested (P2 confirmed):

| h1 → h36 | occurrence | plain magnitude | gate-weighted magnitude |
|---|---|---|---|
| unclamped | ×0.0360 (28× fewer) | ×0.2219 (4.5× smaller) | ×0.0144 (69×) |
| **clamped** | **×0.7152** | **×0.6510** | **×0.6765** |

**What the arithmetic actually says.** Alignment is a *ratio*, so "fully preserved" means the two
magnitudes decline **proportionally** under the clamp (0.677 / 0.651 = 1.04). Without it, the
gate-weighted magnitude falls **15× further** than the plain mean (0.0144 / 0.2219 = 0.065). So the
claim is precisely: **with the cell held, the gate keeps tracking magnitude; without it, the gate
loses track of where the large values are.** The clamp does *not* stop the field fading — it still
loses ~29% of its firing and ~35% of its magnitude — it stops the field firing in the *wrong places*.

### What this does NOT establish — registered in §2 before the run, restated unchanged

1. **Not causation.** AP was already known to rise (M48, 4 seeds). Now alignment is also known to
   hold. That is **two known effects of one intervention**, not evidence that one produces the other.
2. **AP is not evidence here** and was used only as the identity check. Reading it as support would
   be circular.
3. **This was the expected outcome, which is what makes it weak.** `hs = o ⊙ tanh(hl)` (C-292) means
   holding the cell can bound the hidden half — M50 measured that. A restored alignment is therefore
   unsurprising if the story is right, so it discriminates poorly. Had F-A fired it would have been
   decisive; surviving is worth much less.
4. **Seed 42 only**, against M48's four.

Per `05b` §5 this is recorded as **consistent with**, and is **not** promoted further.

### What would actually settle it

An intervention that **restores alignment without clamping**, or **clamps without restoring
alignment**. Until one exists, "the clamp works by preserving alignment" and "the clamp does
something else that happens to preserve alignment too" fit this data equally well.

---

## EXP-3 — the rolled anchor · **H-scale REFUTED; FR-4 FIRED and taught the real lesson** · 2026-09-03

**Pre-registration:** [`05c_analysis_plan_exp3.md`](05c_analysis_plan_exp3.md), locked `25efb8b`
before the run. **One variable:** `freeze_anchor_roll ∈ {3, 15, 90}` on the EXP-2 arm. Seed 42,
13 origins, emit-only, 427/776/613 s.

### Result

| arm | occurrence ratio | plain magnitude ratio | alignment @h36 | **AP@h18** | B |
|---|---|---|---|---|---|
| unclamped | 0.0360 | 0.2219 | 4.3 | 0.32984 | 0.00 |
| clamp | 0.7152 | 0.6510 | 69.3 | **0.36219** | 1.00 |
| roll 3 | 0.7264 | 0.6526 | 60.6 | **0.04040** | −8.95 |
| roll 15 | — | — | 67.6 | **0.01062** | −9.87 |
| roll 90 | 0.7177 | 0.6279 | 61.5 | **0.00753** | −9.96 |

**The roll arms are indistinguishable from the clamp on every field statistic in this dossier, and
have 1/48th of its skill.** They are also far *below* the unclamped baseline — rolling is much worse
than not clamping at all.

### Verdict against the pre-registered falsifiers

| ID | fired? | |
|---|---|---|
| **FR-1** — `B(90) ≥ 0.7` would kill H-place | **no** | `B(90) = −9.96` |
| **FR-2** — total collapse with no ordering ⇒ inconclusive | **no** | the dose is monotone: −8.95 > −9.87 > −9.96 |
| **FR-3** — h1 must be identical | **passed** | identical to 9 s.f. across all five arms |
| **FR-4** — alignment must fall in the roll-90 arm | **FIRED** | 61.5, essentially the clamp's 69.3 |

### The primary question is answered, and it does not depend on FR-4

**H-scale is refuted.** Holding a state with *identical* scale, structure and marginals but the wrong
geography does not merely forfeit the clamp's benefit — it is **catastrophically worse than not
clamping**. The anchor's **spatial content is load-bearing**; steadying the state's magnitudes is not
what the clamp buys. That conclusion rests only on the AP comparison and the h1 identity check, both
of which are clean.

### FR-4 fired because the check was wrong, and that is the bigger finding

I predicted alignment would fall when the map was wrong. It did not — and the reason is that
**alignment is an *internal* statistic.** It measures whether the gate and the body agree *with each
other*, not whether the cells they agree on are the right ones. A rolled model is perfectly
self-consistent about the wrong places.

The same is true of **occurrence** and **body magnitude**. All three survive a 90-cell roll unchanged
while the forecast is destroyed. So:

> **No statistic in this dossier can distinguish a good forecast from the same forecast in the wrong
> place.**

This directly qualifies **M52**. "The clamp preserves alignment" is true and is *not an explanation*:
the roll arms preserve alignment just as well and are useless. Alignment is **not sufficient** for
skill. It may still be necessary — this does not refute H-place, it refutes the sufficiency reading I
put on it.

This is the second mis-specified gate in this dossier (after F3's band). Both share a cause: a
falsifier written from an assumption about what a statistic measures, never itself tested. The
glossary now carries the measured warning (`0d4d079`).

### The caveat that remains, and the cheap test for it

Whether the roll destroys skill because **placement matters** or because a rolled state is simply
**off-distribution** is not settled. For placement: the dose is monotone, and AP on a ~0.4%-positive
field is extremely sensitive to exact cell match, so a 3-cell shift plausibly misses most hotspots.
For off-distribution: a 3-cell shift already costs 8×, which is a lot for so small a displacement.

**A decisive test costs no GPU:** re-score the roll-90 arm against a truth field rolled by the same
90 cells. If the model is producing an intact forecast in the wrong place, rolling the truth to match
recovers clamp-level AP. If it is producing rubbish, it does not. That single re-score separates
"displaced" from "broken", and it is the next thing to run.

**Artifacts:** `results/score_fullzero_fortytwo_identity_freezecell_roll{3,15,90}.csv`,
`results/bodymean_fullzero_fortytwo_identity_freezecell_roll{3,15,90}/`.

---

## EXP-3b — displaced or broken? · **DISPLACED, measured** · 2026-09-03

**No new GPU data** — a re-analysis of EXP-3's field dumps with a new instrument
(`tools/roll_diagnosis.py`), validated on synthetic fields before touching real data (9 tests,
**7/7 mutations caught**).

### Why the obvious test was refused

EXP-3 proposed "roll the truth back and re-score". That is **not valid here**, and would have
produced a confident wrong answer three separate ways: `torch.roll` wraps on a torus while only
13,110 of 32,400 cells are study cells, so rolling truth wraps land onto ocean; the model's *input*
was never rolled, only its memory, so a clean displacement was never guaranteed; and AP scores study
cells only, so a displaced forecast could score zero merely by landing outside the mask.

Circular cross-correlation of the **fields** avoids all three — it is the operation exactly matched
to `torch.roll`, and involves no mask, no truth, and no score.

### Result — the clamp's forecast, moved

| roll | h | peak offset | peak r | r at (0,0) |
|---|---|---|---|---|
| 3 | 6 / 18 / 36 | **(3,3)** every horizon | 0.907 / 0.922 / 0.911 | 0.19 / 0.17 / 0.18 |
| 15 | 6 / 18 / 36 | **(15,15)** every horizon | 0.927 / 0.926 / 0.933 | 0.004 / 0.002 / 0.001 |
| 90 | 6 / 18 / 36 | **(90,90)** every horizon | 0.906 / 0.928 / 0.896 | −0.010 / −0.010 / −0.010 |

**The peak sits at exactly the roll distance, at every horizon, for every dose, at r ≈ 0.90–0.93 —
while correlation at zero offset is ~0.** At h2 all arms are still identical to the clamp (r = 1.000
at offset 0), as they must be: the blend applies to the state carried *forward*, so its effect on the
output appears one step later.

**Control:** clamp vs *unclamped* peaks at **(0,0)** with r = 0.71–0.78 — two genuinely different
forecasts in the *same place*. So the instrument distinguishes "different" from "displaced", which
is the discrimination the whole test depends on.

### Verdict

**The rolled model is not broken. It produces a structurally intact forecast, displaced by exactly
the distance its memory was displaced.** The off-distribution reading of EXP-3 — that a rolled state
simply breaks the model — is **refuted by measurement**, not argued away.

That closes the chain M48 → M53:

> The cell state functions as a **map**. Move the map by 90 cells and the forecast moves by 90 cells
> (r ≈ 0.9) while skill collapses 48×. The clamp helps because it supplies a *correct* map that
> free-running otherwise loses. Nothing about the state's scale is doing the work.

### A second reading, offered as indicative only

The peak is ~0.90, not 1.00, and the input was never rolled — so roughly a tenth of the emitted
field's spatial structure follows the (unrolled) input while the rest follows the (rolled) memory.
That says **the recurrent state dominates the emitted field's spatial structure**. ⚠️ A correlation
of 0.9 does not decompose linearly into "90% memory", and this was not pre-registered; it is a
direction for a designed experiment, not a result.

### It also explains EXP-3's fired gate

FR-4 expected alignment to fall when the map was wrong. It did not, because a displaced forecast is
**internally perfectly coherent** — the gate and body still agree with each other, about the wrong
cells. This is the mechanism behind the blindness the glossary now records.

---

## EXP-3c — is the clamp just buying persistence? · **NO** · 2026-09-03

**No new GPU, and not pre-registered** — a comparison against the fair-persistence baseline computed
on **2026-08-21**, i.e. before this dossier existed. Support verified identical at every horizon
(`N = 170430`, and `n_event` matching 1343/1336/1379/1547/1590/1655/1779), so the two are scored on
the same cells against the same truth.

### The worry

The clamp pins the state built from the **last real observations**. Conflict is highly persistent,
so "where conflict was at the origin" is already a good guess at "where it is 36 months later". The
clamp could therefore be buying a **persistence prior on placement** rather than model skill — a fix
that works by stopping the model doing something stupid, not by it doing something right.

### Result

| h | fair persistence | unclamped | clamped | clamped ÷ persistence |
|---|---|---|---|---|
| 1 | 0.2364 | 0.4779 | 0.4779 | 2.02× |
| 6 | 0.1675 | 0.4008 | 0.4118 | 2.46× |
| 12 | 0.1667 | 0.3770 | 0.3944 | 2.37× |
| 18 | 0.1416 | 0.3298 | **0.3622** | **2.56×** |
| 24 | 0.1234 | 0.2967 | 0.3350 | 2.71× |
| 30 | 0.1082 | 0.2631 | 0.3142 | 2.90× |
| 36 | 0.0951 | 0.2208 | **0.2828** | **2.97×** |

**The clamped model beats fair persistence by ~3× at h36, and its advantage GROWS with horizon**
(2.02× → 2.97×). If the frozen state were merely persistence-equivalent, the ratio would be flat or
falling. It rises.

The decay rates say the same thing more directly — fraction of h1 skill retained at h36:

| | persistence | unclamped | **clamped** |
|---|---|---|---|
| retained | 40.2% | 46.2% | **59.2%** |

**The clamped model degrades more slowly than the observed field its anchor was built from.** A pure
persistence fallback cannot do that.

### Verdict

**The clamp does not work for a bad reason.** The frozen state carries substantially more about
where conflict will be than "where it was", and that surplus widens with horizon. Combined with M54
(the state is a map) the reading is: the cell state holds a *learned* spatial prior, not a copy of
the last observation, and free-running destroys it.

⚠️ **Not pre-registered.** The baseline predates the dossier and the support was verified identical
before the numbers were read, so there is no researcher degree of freedom in the comparison — but it
was not a committed prediction and is not reported as one. Seed 42, `sb`, AP only.
