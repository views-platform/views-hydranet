# 05 — Pre-analysis plan, EXP-1 (LOCKED)

**Written 2026-09-02, before any run.** Locked by the commit that introduces this file. Any change
after the lock must be a **new, dated amendment section** at the bottom, never an edit in place.

## 1. Hypothesis

**C1.** During free-running rollout, the emitted field's decline is **occurrence**, not magnitude:
the gate factor collapses over the horizon while the body factor holds.

Falsifiable form, on the identity `mean(g·mu) = mean(g) × [Σ g·mu / Σ g]`:

> `OCCURRENCE(h36)/OCCURRENCE(h1)` is small, **and** `MAGNITUDE(h36)/MAGNITUDE(h1) ≈ 1`.

**R1 (rival, survivorship).** `mu` falls too; earlier flatness was selection on cells that fired.
**R2 (rival, instrument).** Neither holds of the model; the earlier reading was a recorder artifact.

## 2. Intervention — the one variable

**`feedback_transform`**: `identity` (free-running, treatment) vs `use_real` (real observations,
control). Everything else — model, artifact, seed, origins, `S`, composition — is held fixed.

## 3. Skepticism ledger (what I expect to be accused of, answered in advance)

| Objection | Answer |
|---|---|
| "You conditioned on active cells again." | Nothing is conditioned. The identity runs over every cell at every horizon. |
| "Gate-weighting hides a non-uniform collapse." | Which is why the **unweighted** `mean_cells(mu)` is a co-primary readout (P3/F2), not an afterthought. |
| "You reused the broken CSV." | The CSV is read only to explain the earlier number. No claim rests on it. |
| "One seed." | Seed 42 establishes; seed 43 replicates. A result that does not replicate is reported CONTESTED, not published. |
| "You picked the band after looking." | The bands are in §5, in the locking commit, and the control's own ratio is reported next to them as an empirical check on whether they were sane. |
| "The instrument perturbed the run." | C.2 parity test is blocking, and the identity anchor `AP@h18 = 0.3298395823400329` must reproduce exactly. |

## 4. Pre-registered predictions

Primary readout, decided in advance: **`MAGNITUDE(h36)/MAGNITUDE(h1)` on the treatment arm.**

| ID | prediction |
|---|---|
| **P1** | `OCCURRENCE(h36)/OCCURRENCE(h1) < 0.1` (prior evidence suggests ~1/36 to ~1/600) |
| **P2** | `MAGNITUDE(h36)/MAGNITUDE(h1) ∈ [0.7, 1.4]` (gate-weighted, from I1) |
| **P3** | unweighted `mean_cells(mu)(h36)/…(h1) ∈ [0.7, 1.4]` (from I2) |
| **P4** | control (`use_real`): **both** ratios in `[0.7, 1.4]` |
| **P5** | I1 and I2 agree within 10% at every horizon |

## 5. Falsifiers (pre-committed)

| ID | fires when | consequence |
|---|---|---|
| **F1** | `MAGNITUDE(h36)/MAGNITUDE(h1) < 0.5` | **C1 FALSIFIED.** The model does make smaller forecasts. M50's mechanism sentence is retracted a second time. |
| **F2** | unweighted ratio `< 0.5` while gate-weighted `>= 0.7` | **C1 FALSIFIED via R1.** Non-uniform gate collapse — survivorship in the weighting. |
| **F3** | I1 vs I2 disagree by `>10%` at any horizon | **HALT.** No claim either way. Fix the instrument; do not pick the agreeable one. |
| **F4** | control's `MAGNITUDE` ratio outside `[0.7, 1.4]` | **HALT.** The readout drifts without any degradation to explain it. |
| **F5** | `OCCURRENCE(h36)/OCCURRENCE(h1) > 0.5` | The occurrence collapse itself is not reproduced ⇒ **R2**; the entire prior reading was an artifact. |
| **F6** | C.2 parity fails (dump-on ≠ dump-off, or dump-off ≠ committed) | **HALT.** The instrument perturbs the measurement. |
| **F7** | the `identity` arm does not reproduce `AP@h18 = 0.3298395823400329` exactly | **HALT.** The arm is not what it claims to be. |
| **F8** | seed 43 contradicts seed 42 in **direction** on the primary readout | **CONTESTED.** Not established; reported as unresolved. |

**Grey zone, pre-committed:** a primary ratio in `[0.5, 0.7)` is **CONTESTED** — neither C1 nor R1.
It is reported as an unresolved result. It is *not* resolved by choosing a reading.

## 6. Method

1. Build the `mu` dump (`03` §C.1), default off. Green `03` §D checklist.
2. Emit two arms, seed 42, `fullzero_fortytwo`, `--keep-cubes`: `identity` and `use_real`.
3. **In this order, and no other:** (a) F7 anchor, (b) F6 parity, (c) F3 G1 identity check,
   (d) the control arm, (e) the treatment arm. The treatment's primary number is read **last**, so
   the instrument cannot be tuned against it.
4. Report all three curves over `h = 1..36`, plus the fixed-cohort arm (`02` §5) as corroboration.
5. Reproduce the old `mean_magnitude_on_active` number and state which of C1/R1 explains it.
6. Replicate on seed 43 only if no falsifier fired.

## 7. Decision rules

| outcome | action |
|---|---|
| P1–P5 hold, no falsifier | C1 **supported** at seed 42 → replicate seed 43 → if it holds, ledger row **M51**, promoted as a core diagnostic |
| F1 or F2 | C1 **falsified** → ledger row recording the falsification; retract M50's mechanism sentence; correct the #262/#258 comments that now rest on it |
| F3, F4, F6, F7 | **HALT** — instrument defect, register it, no scientific claim in either direction |
| F5 | prior reading was an artifact → C-318's scope widens; ledger row |
| grey zone or F8 | **CONTESTED** — logged in `07` and the ledger as unresolved |

Every outcome above, including "falsified" and "contested", produces a ledger row. There is no path
where this experiment runs and nothing is written down.

## 8. Budget

Emit-only, no training. Two arms × ~12 min, seed 42. Seed 43 only on survival. Hours, not days.

---

## AMENDMENT A1 — 2026-09-02, before any run

**Raised by the chair:** the repo has both gate compositions as a config axis
(`forecast_composition` ∈ {`self_zeroed`, `soft_gate`, `threshold_gate`}, with `gate_threshold` τ).
Why is the rival only defended against, and not measured?

**Confirmed first:** both vehicles under test are `forecast_composition: soft_gate`,
`output_distribution: nb`, `gate_threshold` absent — so the §2 identity applies to them unchanged.
Nothing already locked is invalidated by this amendment; it **adds** a test.

### A1.1 Why this strengthens the design

`threshold_gate` composes as `(gate >= τ) · body`. Its "active set" is exactly `{cells : g >= τ}`,
so its magnitude statistic is **conditioned on clearing a bar** — the precise shape of the statistic
that produced the suspect claim, and the precise mechanism R1 (survivorship) proposes. R1 was
previously defended against by construction (the §2 identity conditions on nothing). It can instead
be **measured**, by varying the strength of the selection and watching what the statistic does.

### A1.2 Zero marginal cost

Composition is applied at **emit time** (ADR-069), after the gate field and the body params exist.
Both are dumped by this program already (I1's gate cube, I2's `mu` field). Every τ is therefore a
pure **offline** function of fields from a single emit. No extra GPU, no extra arm.

### A1.3 Scope — what the sweep does and does not test

* **Does test:** whether "flat magnitude among active cells" is an artifact of *selection*, by
  varying the selection strength on a fixed trajectory.
* **Does NOT test:** what a model actually *fed back* under `threshold_gate` would do. Composition
  changes what is fed back, hence the trajectory itself. That is a different experiment and is not
  claimed here.

This distinction is load-bearing and is the kind of conflation that produced C-318. Any write-up
that reports the sweep must state it.

### A1.4 Additional pre-registered prediction

| ID | prediction |
|---|---|
| **P6** | Across τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, the conditioned magnitude ratio `MAG_τ(h36)/MAG_τ(h1)` is **flat in τ** — no monotone trend, spread across τ smaller than the `[0.7, 1.4]` band's half-width. |

### A1.5 Additional falsifier

| ID | fires when | consequence |
|---|---|---|
| **F9** | `MAG_τ(h36)/MAG_τ(h1)` **rises monotonically with τ** across the five values, and the spread from τ=0.1 to τ=0.9 exceeds 0.3 | **R1 CONFIRMED as a live mechanism.** Selection is measurably propping the statistic up. Any claim of flat magnitude that rests on a conditioned statistic — including the current M50 sentence — is unsupported, *independently* of what the unconditioned §2 identity shows. |

**Interaction with F1/F2, pre-committed now:** F9 firing while the unconditioned identity still shows
flat magnitude is **not** a contradiction. It would mean the model's magnitude genuinely holds *and*
the old conditioned statistic was propped up anyway — i.e. C1 is right for the wrong reason, and the
evidence previously offered for it was invalid. That combination must be reported in exactly those
words, not collapsed into "C1 confirmed".

### A1.6 Reading order

The τ sweep is read **after** the treatment arm's primary number (`05` §6.3 step e), as step (f).
It cannot be used to select the primary reading.
