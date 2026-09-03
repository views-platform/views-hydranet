# 07 — Experiment log (Wave 1)

Append-only. Negatives and unanswerables are entered in full.

---

## INTERIM — seed 42 only, 2026-09-03 03:40. **Not a result: n=1 of a planned 4.**

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
