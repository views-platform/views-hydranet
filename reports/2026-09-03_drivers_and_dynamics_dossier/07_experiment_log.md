# 07 — Experiment log (Wave 1)

Append-only. Negatives and unanswerables are entered in full.

---

## INTERIM — seed 42 only. **SUPERSEDED by the n=4 entry below**; kept as the record of what n=1 looked like.

Four arms complete for `fullzero_fortytwo` (`none`, `hidden`, `cell`, `all`), 13 origins each,
posterior-mean dumps (`n_passes = 4`). Seeds 43/44/45 still running. The programme's own decision
rule requires 4/4 seed agreement, so **nothing below is established.**

**Both reproduction falsifiers are EXACT** — `none` reproduces `AP@h18 = 0.3298395823400329` and
`cell` reproduces `0.3621885544392029`, to the last digit, against the archived values. The dump
change did not touch the cube path and demonstrably did not perturb it.

### Q C.4 — onset vs continuation: the clamp is NOT a continuation-only trick

Cells split by their state at the origin month. `sb`, pooled over 13 origins.

| | h18 none → cell | h36 none → cell |
|---|---|---|
| AP, all cells | 0.3298 → 0.3622 | 0.2208 → 0.2828 |
| AP, **continuation** (active at origin) | 0.5927 → 0.6392 | 0.4724 → 0.5469 |
| AP, **onset** (quiet at origin) | 0.1763 → 0.1936 | 0.1211 → **0.1691** |
| base rate, continuation / onset | 0.353 / 0.0063 | 0.323 / 0.0079 |

**The clamp improves onset skill substantially — +40% relative at h36** — and the model has real
onset skill to begin with (AP 0.1211 against a 0.0079 base rate is **15× chance**). The product
worry that motivated this question — that the headline AP gain is entirely conflict that was
already there — **is not supported**.

⚠️ **The falsifier as written was ambiguous, and this is the third such defect in the programme.**
`06_question_battery.md` said the prediction was refuted if the gain on new cells is "at least as
large" as on continuing cells, without specifying the measure. It matters:

| measure (h36) | continuation | onset | says |
|---|---|---|---|
| absolute AP gain | 0.0745 | 0.0480 | continuation |
| relative AP gain | 0.158 | **0.396** | **onset** |
| gain in skill-above-base | 0.499 | 0.424 | continuation |

All three are reported rather than one chosen. Two of three favour continuation, but the margins
are narrow and the onset gain is large on any reading — so the defensible statement is **"the clamp
helps both, and is not continuation-only"**, not a ranking between them.

### Q0.1 / C.1–C.3 — freezing flattens the dynamics but does not cost direction skill

Cohort fixed by TRUTH at h1 (n = 1343 cells), so no arm can move its own denominator.

| h | rho: none | hidden | cell | all | | disp: none | hidden | cell | all |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.009 | 0.025 | 0.025 | 0.041 | | 0.69 | 0.76 | 0.53 | 0.50 |
| 18 | 0.040 | 0.042 | 0.054 | 0.060 | | 0.96 | 1.14 | 0.51 | 0.52 |
| 36 | 0.145 | 0.142 | 0.142 | 0.137 | | 1.20 | 1.26 | **0.51** | **0.53** |

**Q0.1 (gate):** direction skill is weak but non-zero and *rises* with horizon (rho 0.009 → 0.145).
It clears the 0.05 unanswerable-threshold from h18 on, so C.1–C.3 are answerable at long horizons
and **marginal at h6–h12** — reported as such.

**C.2 — yes, freezing flattens dynamics, and it is the CELL that does it.** Dispersion of the
predicted per-cell change grows 0.69 → 1.20 unclamped, and is **flat at ~0.51 under the cell clamp**
(and under `all`). Freezing *hidden* does not flatten it (0.76 → 1.26, like unclamped). Same
cell-versus-hidden asymmetry M39 found for AP, now visible in the dynamics.

**C.3 — the flattening costs no direction skill.** rho at h36 is 0.145 / 0.142 / 0.142 / 0.137
across none / hidden / cell / all — indistinguishable. So the clamp compresses the *amplitude* of
predicted change ~2.3× while leaving the *ordering* of which places worsen intact.

That is the same shape as everything else in this programme: **freezing fixes placement and ordering,
and does nothing for amplitude.** ⚠️ Note the honest caveat — rho ≈ 0.14 is weak skill, so "no cost"
is partly because there was little to lose.

### Still open at this timestamp

Seeds 43/44/45; the `all`-vs-`cell` redundancy question (B.2); whether any arm ever *hurts* (B.3).

---

## WAVE 1 — 4 arms × 4 seeds · **COMPLETE, 16/16, zero failures** · 2026-09-03

**Pre-registration:** [`05_analysis_plan.md`](05_analysis_plan.md), locked at commit `0a76a6a`
before the first arm. **One variable:** `freeze_recurrent` ∈ {*none*, `hidden`, `cell`, `all`}.
Seeds 42/43/44/45, 13 origins, emit-only, 167 min unattended. No arm failed, no guard fired, no
stall. **Decision rule:** 4/4 sign agreement **and** |mean| > seed sd; no p-value claimed, because a
paired sign-flip at n=4 floors at 1/16 = 0.0625.

**Both reproduction falsifiers EXACT** — seed-42 *none* and `cell` reproduced their archived
`AP@h18` to the last digit, so the posterior-mean dump change provably did not perturb the cube path.

### The headline: C.4 — the clamp is NOT a continuation-only trick

`cell` minus *none*, AP within each origin-state universe, **4/4 seeds in both universes**:

| | h18 | h36 |
|---|---|---|
| **continuation** (active at origin) | **+0.0437** SUPPORTED | **+0.0614** SUPPORTED |
| **onset** (quiet at origin) | **+0.0224** SUPPORTED | **+0.0459** SUPPORTED |

The question was whether the clamp's headline AP gain is entirely conflict that was *already there* —
which would make it close to worthless for the product. **It is not.** Onset improves on every seed
at both horizons, by roughly **38% relative** at h36 against an onset base rate of ~0.008.

The absolute gain is larger on continuation, so the pre-registered prediction ("disproportionately
on continuing cells") survives on that measure — but the defensible claim is **"it helps both"**,
because the measure choice was never pinned (see the C.4 defect in `05`).

### B.1 — the clamp does something small for the body, and nothing for its calibration

| | verdict |
|---|---|
| `size_ratio` | **NO EFFECT — all 16 deltas exactly 0.** The median conflict cell gets nothing, in every arm, every seed, every horizon. The pre-registered falsifier did not fire. |
| `crps_events` | **improves 4/4 seeds in every arm** — `cell` −6.7/−6.1/−6.5/−4.1 % at h18, but only −0.62/−0.55/−0.67/−0.49 % at h36 |

⚠️ **This corrects the seed-42 interim above, and M55.** I wrote that the clamp does "nothing" for
the body. That was too strong: `crps_events` improves *consistently* — unanimous across seeds — just
by a small relative amount that fades to ~0.6% by h36. The accurate statement is **the clamp buys a
small, replicated improvement in the body's proper score and moves its calibration not at all.**

### B.2 — `all` is redundant; the cell carries it

AP at h36: `cell` **+0.0591**, `all` **+0.0578** (seed sd 0.007). Dispersion at h36: `cell` −0.659,
`all` −0.667. Indistinguishable. Freezing both halves buys nothing over freezing the cell — **M39's
89% attribution confirmed at n=4**, and its falsifier (that `all` exceeds `cell` beyond seed spread)
did not fire.

`hidden` alone is real but minor: AP +0.0186 at h36 (4/4) against `cell`'s +0.0591, and **contested
at h18** (2 seeds negative). So the long-horizon benefit is roughly **3× larger from the cell**.

### B.3 — does freezing ever hurt? Not consistently

The only negatives are `hidden` at h18 on two of four seeds (−0.0090, −0.0002), which the rule calls
CONTESTED. No arm is consistently worse than *none* on any head at any horizon. The clamp appears
**strictly non-harmful** here — which `05` pre-committed to treating as a finding in its own right.

### Q0.1 / C.1–C.3 — freezing flattens the dynamics and costs no direction skill

**Q0.1 (the gate on the rest):** direction skill is weak but real and *rises* with horizon —
rho ≈ 0.14 at h36 — clearing the 0.05 unanswerable-threshold from h18 on. So C.1–C.3 are
answerable at long horizons and **marginal at h6–h12**.

| `cell` minus *none* | h18 | h36 |
|---|---|---|
| **dispersion** of predicted per-cell change | **−0.4563** SUPPORTED (4/4) | **−0.6593** SUPPORTED (4/4) |
| **rho** (direction skill) | **+0.0137** SUPPORTED (4/4) | −0.0020 CONTESTED |

**C.2 — yes, and it is specifically the cell.** Freezing the cell compresses the spread of predicted
change on every seed; freezing *hidden* has no consistent effect (contested at both horizons). The
same cell-versus-hidden asymmetry M39 found for AP, now visible in the dynamics.

**C.3 — the compression costs nothing.** Direction skill is *slightly better* under the clamp at h18
(4/4 seeds) and indistinguishable at h36. So the trade-off this question was designed to catch —
placement bought at the price of dynamics — **does not exist**.

⚠️ Honest caveat: rho ≈ 0.14 is weak skill in absolute terms, so "no cost" partly reflects that
there was little to lose.

### What Wave 1 establishes

> The cell clamp is a **placement and ordering** fix. At n=4 it improves the gate on *both* new and
> continuing conflict, compresses how much the model says magnitudes will change without disturbing
> *which* places it says will worsen, buys a small consistent gain in the body's proper score, and
> moves magnitude calibration **not at all**. Freezing the hidden half is a weak shadow of it, and
> freezing both adds nothing to freezing the cell.

### Not answered by Wave 1

Attribution proper (Block A) — whether the fed-back **input** or the **state** drives each head, and
in what share. Wave 1 answers it only coarsely, through interventions. `hs = o ⊙ tanh(cl)` (C-292)
also means the hidden/cell split here is the caveated half throughout. And everything is `sb`.

---

## D.2 — does any of it hold for `ns` and `os`? · **YES, and it is STRONGER there** · 2026-09-03

**Zero GPU.** The body-mean dumps carry all three targets (`mu` is `[36, 3, 180, 180]`); only the
*score CSVs* were sb-only, because the wave scored with `--targets=sb`. So this is pure re-analysis
of data already on disk, with the same instruments that were mutation-tested before it landed.

**Validity check first:** after parameterising the tools by target, `sb` reproduces its previous
numbers **identically** (+0.0614 / +0.0459 at h36), so the refactor is provably inert on the
established results.

### C.4 — the onset gain is not an `sb` artefact

`cell` minus *none*, mean over 4 seeds. **Every one of these twelve lines is SUPPORTED at 4/4.**

| target | h18 cont | h18 onset | h36 cont | h36 onset |
|---|---|---|---|---|
| `sb` | +0.0437 | +0.0224 | +0.0614 | +0.0459 |
| `ns` | **+0.1700** | +0.0423 | **+0.1401** | +0.0357 |
| `os` | +0.0505 | +0.0353 | **+0.1582** | +0.0579 |

The clamp improves **both** universes on **all three targets**, and its effect on continuation is
**2.3–2.6× larger** on `ns`/`os` than on `sb`.

### C.3 — on `ns` and `os` the clamp *improves* direction skill

| target | rho at h36, `cell` − *none* | verdict |
|---|---|---|
| `sb` | −0.0020 | CONTESTED |
| `ns` | **+0.0177** | **SUPPORTED (4/4)** |
| `os` | **+0.0598** | **SUPPORTED (4/4)** |

On `sb` the flattening cost nothing. On `ns` and `os` it is better than free: the clamp compresses
the predicted amplitude of change (dispersion −0.90 / −0.87, 4/4) **and improves the ordering of
which places worsen**, unanimously.

### C.2 and cell-dominance both generalise

Dispersion falls 4/4 under `cell` on every target. `hidden` also flattens `ns`/`os` (−0.34 / −0.51,
4/4 — unlike `sb`, where it was contested) but always **less than the cell**: −0.34 vs −0.90 on `ns`.
And on onset at h36 the cell beats hidden on every target — `sb` 0.0459 vs 0.0183, `ns` 0.0357 vs
0.0203, `os` 0.0579 vs 0.0411 — though the margin narrows on `os`.

### What this changes

The **"`sb` only"** caveat carried by M56, M57 and M58 is **lifted**: the placement finding, the
onset gain, the dispersion flattening and the cell-over-hidden ordering all hold on three targets at
n=4. The one thing that changes with target is the *sign of the dynamics trade-off* — neutral on
`sb`, positive on `ns`/`os`.

---

## WAVE 2 — attribution by per-step roll · **THE CELL DRIVES PLACEMENT; THE INPUT DOES NOT** · 2026-09-03

6 arms (cell / input / hidden × seeds 42, 43), 13 origins each, 78 min, **6/6 OK**, no failures.
Roll ONE live driver by 90 cells at **every** step, then ask — with the cross-correlation instrument
validated in EXP-3b — whether the emitted gate field followed it.

### Result: fraction of the 26 origin-seed pairs whose field peaks at the roll offset

| driver | h2 | h4 | h6 | h12 | h24 | h36 |
|---|---|---|---|---|---|---|
| **cell** | **26/26** | **26/26** | **26/26** | **26/26** | 16/26 | 9/26 |
| **input** | **0/26** | **0/26** | **0/26** | **0/26** | **0/26** | **0/26** |
| **hidden** | 0/26 | 0/26 | 0/26 | 0/26 | 0/26 | 10/26 |

**Displace the cell state and the forecast moves with it — every origin, both seeds, out to h12.
Displace what the model is fed and the forecast does not move, at any horizon, ever.** The
input-rolled arm stays 0.79–0.93 correlated with the unrolled baseline at zero offset: rolling the
input barely perturbs the emitted field at all.

Hidden's lone 10/26 at h36 sits at r = 0.47 — the weakest correlation in the table, on one seed
only, where no coherent displacement survives in any arm. It is not read as a signal.

### This answers A.1 and A.2

**The emitted field's spatial pattern is set by the recurrent CELL state, not by the input.**
Wave 1 said the cell matters more than hidden through an *intervention*; this says it directly, and
adds the part Wave 1 could not: the fed-back input contributes essentially **nothing** to where the
model points on the current step.

**That is not a contradiction of `use_real`,** which restores skill dramatically (M51: occurrence
×1.19 against ×0.036). The reconciliation is that the input acts **through the state over time**, not
on the present step's placement: corrupt the input and the state degrades across steps, which is the
feedback loop this whole programme has been chasing. Displacing the input for one step — or even
every step — does not move the placement the state is already holding.

### Correction to my own reading, recorded because it was wrong in the direction of a finding

Reading **a single origin** (335) mid-run, I saw `(0,0)` peaks at h18/h36 and concluded the 90-cell
shift was **aliasing** — 90 + 90 = 180 ≡ 0 on a 180-cell grid, so cumulative displacement returns to
the origin. **That was wrong.** The 13-origin aggregate shows 26/26 following at h2, h4, h6 *and*
h12 — horizons whose roll counts differ in parity, which no aliasing rule can produce. The decay
after h12 is the rolled pattern losing **coherence** through repeated rolling and the network's own
non-linearity, not an arithmetic artefact.

The lesson is the plainer one: **a single origin is not a measurement.** I drew a mechanism from
n=1 when the instrument's own design pools 13.

### Limits

Seeds 42 and 43 only (Wave 1's four were not available inside the window). One shift (90) — a shift
coprime with 180 would separate coherence-decay from residual aliasing more cleanly, and remains
worth running. Gate field, `sb`. And the h24/h36 decay means this attribution is established for
**short-to-mid horizons**, not across the whole rollout.

---

## ARCHITECTURE READ — what the hidden state actually is · 2026-09-03

Prompted by the chair asking "what does the hidden state even do? is it working? is it implemented
correctly?" Answered from `HydraBNUNet06_LSTM4.forward` (lines 568–603), not from inference behaviour.

**It is a textbook ConvLSTM and it is correct.** Per block, four times over:

```
i_t = σ(Wxi(x) + Whi(hs));  f_t = σ(Wxf(x) + Whf(hs));  h̃l = tanh(Wxc(x) + Whc(hs))
hl  = f_t * hl + i_t * h̃l          ← the CELL accumulates
o_t = σ(Wxo(x) + Who(hs))
hs  = o_t * tanh(hl)                ← the HIDDEN is RECOMPUTED from the cell
h   = cat([hs_1..4, hl_1..4])       ← repack order matches blend_recurrent_state's split
x   = cat([x, hs_1..4])             ← hidden ALSO feeds the encoder directly
```

**Hidden does two real jobs:** it drives all four gates at the next step, and it is concatenated onto
the U-Net input, so it shapes what the encoder sees. It is not a passive readout.

**But it has no memory of its own.** Every step `hs` is overwritten by `o_t * tanh(hl)`. Only the
cell persists, through `f*hl + i*h̃l`. That is the textbook long-term/short-term division, correctly
implemented.

### This qualifies two of my own results

**M60's "hidden 0/26" is architecturally guaranteed, not a measurement of hidden's importance.** A
displaced `hs` is regenerated from the *undisplaced* cell on the very next step — the perturbation
self-heals. The roll test would return 0/26 no matter how much work hidden does. What survives is the
narrower, still-useful claim: **hidden carries no spatial information across steps**, so it cannot be
the thing that drains during free-running.

**M56's `hidden` freeze arm is an architecturally abnormal operation.** Holding `hs` at the anchor
while the cell evolves feeds a stale hidden to the gates and the encoder, breaking the LSTM's own
recurrence. Its +0.019 AP is real and replicated (4/4) but should not be read as "what the hidden
state contributes" — it is "what happens when you jam the readout".

**The C-292 caveat, which `05` carried throughout, is now concrete rather than a formula:** `hs` is a
function of `hl`, so hidden-vs-cell was never a clean two-way split, and the cell-side results are
the trustworthy half.
