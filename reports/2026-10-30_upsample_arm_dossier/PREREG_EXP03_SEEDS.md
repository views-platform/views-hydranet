# PREREG — EXP-03, four more seeds, written BEFORE they run

**2026-11-03.** Committed before any of the eight new arms exists.

⛔⛔ **This document does NOT change `PREREG.md` §6.** §6 is the verdict rule for `P-23`, it was
written before EXP-02's numbers existed, and §6 device 6 forbids changing it after them. Every
branch condition in §6 carries over **verbatim**. This document does exactly three things §6 could
not do for itself:

1. **Specifies** what §6's phrase *"both seeds"* means at n=6. §6 was written for n=2 and the
   phrase has no defined reading at n=6. ⇒ Pinned below as **all seeds, strictly** — the stringent
   reading, chosen so this specification cannot weaken the rule.
2. **Discharges** §6's own MIXED-branch order: *"names the seed count needed, **derived** from the
   observed spread rather than assumed."* EXP-02 returned MIXED, so that clause is live.
3. **Pre-registers** the new predictions, including one that predicts this experiment will not
   resolve `P-23` either.

⚠️ **The reason this document exists at all is a failure of mine, logged as `W-124`.** I first told
Simon that §6 was the wrong instrument and needed replacing with a mean endpoint. It is not: §6 is
stringent, not inadequate, and it reports MIXED because the h36 effect measures ≈ 0. The argument
ran toward the conclusion I wanted — a confirmable endpoint. **§6 stands unaltered.**

---

## 1 · What EXP-02 measured, and why it cannot answer its own question

ΔAP = AP(UpsampleConv2) − AP(its own-seed control), `sb`, **gate** AP, frozen
`lodestar_score.average_precision`, paired origin-block bootstrap over **13 origins**,
**842,634 cell-months** per endpoint, S = 16 on all four arms.

| seed | ΔAP @h1 | ΔAP @h36 |
|---|---|---|
| 42 | **+0.004890** | **−0.003549** |
| 43 | **+0.004700** | **+0.003405** |

⇒ **§6 verdict: MIXED** — *"seeds disagree at h36."* Not CONFIRMS (requires < 0 at h36 on both);
not FALSIFIES (requires ≥ 0 at h36 on both, and seed 42 is negative).

⭐⭐ **The asymmetry is the finding, and it needs no error bar to state.** At h1 the two seeds agree
to **4%**. At h36 they are on **opposite sides of zero**. Between-seed sd: **0.00013** at h1 against
**0.00492** at h36 — a factor of **38**.

⛔ **Why n=2 cannot name the seed count §6 asks for.** That h36 sd is estimated from **one degree of
freedom**. Its own 90% χ² interval is **[0.00251, 0.07842]** — a factor of **31**. Carried into a
power calculation at 90% confidence / 80% power for a mean effect of 0.005:

| σ used | seeds required |
|---|---|
| σ as measured (0.00492) | **6** |
| σ at the top of its own CI (0.07842) | **1,521** |

⇒ ⛔ **The data cannot distinguish these.** That is the honest discharge of §6's clause as of
EXP-02, and it is the entire justification for EXP-03. **EXP-03 buys σ at 5 dof instead of 1.**

---

## 2 · The intervention — seeds only. One variable, and it is n.

Eight new arms: seeds **44, 45, 46, 47** × {`HydraBNUNet06_LSTM4` (control), `UpsampleConv2`}, at
`total_lessons=300`, `ss_epsilon_max=0.0` — ⛔ **identical in every respect to EXP-02's four except
the seed.**

Binding, and each verified rather than asserted:

- ⛔ **One data pull.** `queryset_digest` **must** equal
  `b32a221b70e5255d3d85e683b483d9398ae08ed3b5e9d1be71c6e8f7aa13422c` on all eight, and the raw
  parquet **md5 must match EXP-02's** `d3636988b1085af74a08555c347ec5fe`. Four arms on two pulls
  makes this a cross-population comparison (`C-351`), which is what `P-30`'s falsification already
  did to `P-27` once.
- ⛔ **S must equal 16 on all eight**, read off `y_pred.npy.shape[1]` in each arm's own cube, not
  off a manifest field. AP is quantised to S+1 rank levels; unequal S makes AP incomparable across
  arms (epic #263: *"the reference's S MUST equal the arms' cube width"*).
- ⛔ **`n_origins` must equal 13 on all eight**, and each `upctwo` arm is differenced against the
  control **of its own seed**.
- ⛔ **No `h18` number appears anywhere in this document's verdict**, in either direction
  (§6 device 2, carried).

ℹ️ Cost, for the record: ~**$22** and ~6 h wall clock at EXP-02's measured rate (~$11 / 5h45m for
four arms, four pods in parallel).

---

## 3 · ⛔ THE VERDICT RULE AT n=6 — §6, specified, not amended

§6's three branches, **verbatim**, with the single substitution `both seeds → all six seeds`:

| branch | condition at n=6 |
|---|---|
| **CONFIRMS** | ΔAP > 0 at h1 **and** < 0 at h36, **all six seeds** |
| **FALSIFIES** | ΔAP **≥ 0 at h36 on all six seeds** |
| **MIXED** | seeds disagree at h36 |

⛔ **"All six", not "five of six", not "the mean".** The strict reading is chosen deliberately: a
specification written after seeing EXP-02's numbers must not be able to turn a MIXED into a
CONFIRMS, and only the strictest reading is immune to that charge. If achieved, unanimity at n=6
carries a binomial **p = 0.0156** on its own.

⚠️ **A mean test on ΔAP is registered here as SECONDARY and ⛔ CANNOT produce CONFIRMS.** It is
reported alongside, never instead, and the primary verdict is read off the table above. Stated now
so that preferring it later is visibly a change to this document.

---

## 4 · Pre-registered predictions

| id | prediction | threshold, declared now |
|---|---|---|
| **P-31** | the h36 between-seed sd, re-estimated at 5 dof, lands near its n=2 point estimate rather than anywhere in the 31× band — i.e. within a factor of ~1.6 either way of **0.00492** | σ̂(h36) ∈ **[0.0030, 0.0080]**, against an n=2 interval of [0.0025, 0.0784] |
| **P-32** | ⭐⭐ **THE ONE THAT MATTERS.** The h1/h36 sd asymmetry is **real and not an n=2 artifact** — the arm effect is near-deterministic at h1 and seed-dominated at h36 | **σ̂(h1) < σ̂(h36) / 3** |
| **P-33** | ΔAP @h1 is **positive on all six seeds** | sign test, all six |
| **P-34** | ⛔ **This experiment does NOT resolve `P-23`.** h36 signs are **not** unanimous ⇒ §6 returns **MIXED** a second time | sign test, stated as a failure in advance |
| **P-35** | the six-seed mean ΔAP @h36 is **not distinguishable from zero** | \|mean\| **<** its own between-seed 90% half-width |

⛔⛔ **`P-34` is declared deliberately, and it is the anti-rescue device of this document.** I am
predicting, before spending the money, that my own headline prediction will again fail to resolve.
⇒ If MIXED arrives I may **not** report it as a surprise, as *"under-powered therefore
inconclusive therefore still promising"*, or as grounds for a third seed purchase. Its measured
probability from EXP-02's own numbers is **~97%**: P(a seed lands negative at h36) = **0.506**.

---

## 5 · ⛔ The seed-count derivation, formula pre-committed

Reported from EXP-03's σ̂ with **no** post-hoc choice of estimator:

```
n_required = (z_0.95 + z_0.80)^2 * sigma_hat^2 / mu^2        # = 6.1865 * sigma^2 / mu^2
```

evaluated at **three** pre-named μ, and reported at σ̂ **and** at both ends of σ̂'s own χ²(5)
interval, so the answer is a range and is never quoted as a scalar
(`feedback_sweep_before_reporting_a_scalar`):

- **μ = 0.0050** — the effect size actually observed at h1
- **μ = 0.0035** — the larger of EXP-02's two h36 magnitudes
- **μ = 0.0400** — `P-24`'s own pre-registered threshold

⚠️ **And the pre-committed consequence of the answer.** If `n_required` at σ̂ exceeds **20 seeds**,
⛔ **the seed route is declared CLOSED on measurement** and no further seeds are bought — the
h36 endpoint is then not resolvable by replication at any affordable n, and that is the result, not
a reason to keep buying. ⚠️ What would reopen it: a change that reduces between-seed variance
itself, not more draws of it.

---

## 6 · What would make EXP-03 unreadable, named now

- ⛔ **A second data pull.** Any digest or md5 differing from §2's ⇒ the eight arms are not
  comparable to EXP-02's four and the pooled n=6 does not exist. **Verified per arm before scoring.**
- ⛔ **S ≠ 16 on any arm** ⇒ that arm's AP is on a different rank lattice. Excluded, and the
  exclusion reported.
- ⛔ **The gate not coming home** at both endpoints ⇒ verdict is **COULD NOT BE READ**, which is the
  honest outcome and **not** "the arm failed" (carried from the EXP-02 plan).
- ⚠️ **No guard enforces slicing the gate before the pod is destroyed.** It is prose in
  `runpod_run_guide.md:419-432` and an `xfail` stub. This remains the highest-consequence manual
  step in the programme.

---

## 7 · Devices carried forward from `PREREG.md` §6, unaltered

1. ⛔ An oracle-only / T=0-only improvement is a **FAIL**, declared in advance, and may not be
   reported as encouraging.
2. ⛔ No `h18` verdict, in either direction.
3. ⛔⛔ If the result is negative and I then wish to argue the test was inadequate, that argument is
   logged as a `W-` row against myself in `PREDICTION_LEDGER.md` **FIRST**. ⚠️ **This device has
   already fired once on this programme — `W-124`, 2026-11-03 — and it fired one message late.**
4. ⚠️ No branch above says **"closed"**, except §5's measured seed-route clause, which names in the
   same sentence what would reopen it.
5. ⛔ This document may not be changed after EXP-03's numbers exist.
