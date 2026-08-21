# Experiment log — the persistence re-reference

---

### EXP-01 · does a 300-lesson model beat persistence? · 2026-08-21 21:28 → 21:37 · **YES, at every horizon**

**Vehicle:** `fullzero_fortythree`, L=300, ε=0, seed 43, artifact `calibration_model_20260821_045948.pt`.
One emit-only pass (7 min GPU), then arm and persistence scored **in one call on one support**, so the
two sides share their origin/cell set by construction — the exact thing whose absence made M1 and our
L=300 numbers incomparable.

#### Falsifier first — did the re-emit reproduce the archived control?

ε=0 arms are bit-reproducible (M22), so `identity` must land on the archived score.

| h | AP | expected | Δ | crps_all | expected | Δ |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.4774 | 0.4774 | 2.9e-05 | 0.1246 | 0.1246 | 3.7e-05 |
| 6 | 0.4071 | 0.4071 | 4.7e-05 | 0.1120 | 0.1120 | 3.5e-05 |
| 18 | 0.3318 | 0.3318 | 2.5e-05 | 0.1341 | 0.1341 | 2.2e-05 |
| 36 | 0.2287 | 0.2287 | 1.3e-06 | 0.8747 | 0.8747 | 2.7e-05 |

`N = 170430` on every row. **PASS** — the vehicle is what we think it is.

#### Result

| h | L=300 AP | persistence AP | ratio | Δ | Δ / MDE |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.4774 | 0.1461 | **3.27×** | +0.3313 | 6.1× |
| 6 | 0.4071 | 0.1122 | **3.63×** | +0.2949 | 5.5× |
| 12 | 0.3596 | 0.1295 | **2.78×** | +0.2302 | 4.3× |
| 18 | 0.3318 | 0.1077 | **3.08×** | +0.2241 | 4.1× |
| 24 | 0.3027 | 0.0966 | **3.13×** | +0.2061 | 3.8× |
| 30 | 0.2748 | 0.0844 | **3.25×** | +0.1904 | 3.5× |
| 36 | 0.2287 | 0.0834 | **2.74×** | +0.1453 | 2.7× |

**The model beats persistence at every horizon out to 36 months**, by 2.7×–3.6×, with gaps of
**2.7×–6.1× the MDE**. #281 was never the binding constraint — as the decision table predicted.

#### The independent cross-check nobody asked for

Persistence on **our** origins lands on M1's persistence column to ~1%:

| h | here | M1 | 
|--:|--:|--:|
| 6 | 0.1122 | 0.112 |
| 18 | 0.1077 | 0.108 |
| 36 | 0.0834 | 0.083 |

Persistence is a truth-only baseline, so this is what it *should* do if the origin and cell sets are
comparable — and it says they are. **That legitimises the comparison M1 could not make.** M1's
persistence column was right. What changed is the model: free-running h18 went **0.007 → 0.3318**, a
factor of **47**.

#### Verdict against the pre-registered decision table

`00_README.md` registered three worlds before the run. This is **"win HUGELY"**: direction is safe,
and matched-S work now sizes how much of the margin is real rather than deciding whether there is one.

#### ⚠️ The margin is NOT trustworthy — and the reason is worse than S alone

The #263 rule says the reference's S must match the arms'. Reading the scorer, the mismatch here is
sharper than a width difference:

```python
p = (np.array([g[(m0,h,u)][1] for ...])      # our arm HAS a gate -> continuous probability
     if has_gate else (cs > 0).mean(1))      # persistence has NONE -> S=1 -> p in {0.0, 1.0}
```

`_persistence_gathered` returns `(np.array([last]), None)` — **no gate**. So persistence's AP is
computed from a **two-level binary score** while the arm's is computed from a finely-ranked gate
probability. AP is rank-based (sklearn `average_precision_score`), so persistence cannot express any
ordering *within* its predicted-positive set. It is handicapped to the maximum degree, **in our
favour**.

Two consequences, and the second is the interesting one:

1. **Our 3× margin is an upper bound.** A fairly-scored persistence can only score higher.
2. **M1 inherits the same bias** — its persistence column was computed the same way. So M1
   *understated* persistence, which makes its original "persistence beats every arm" claim
   **stronger** than it was written, not weaker.

**The cheap fix, and the obvious next step:** rank persistence by the persisted **value**
`truth[m0-1]` instead of by the binary indicator. Strictly more information from identical data, no
GPU, minutes of CPU. Until that runs, the honest claim is **direction, not size**.

#### Defects found in this run

* **My own scorer invocation used `--out <path>`** where the parser is `a.split("=", 1)[1]` and
  requires `--out=<path>`. This is the *documented* trap — `run_realism_arms.py`'s own docstring warns
  about it for `--targets`. Cost: nothing, because score-then-**keep** left the cube on disk and the
  re-score took 12 seconds. The guard earned its keep on its first outing.
* **The script's identifier-preservation line destroyed what it preserved.** `cp "$PRED"/origin_*/lr_*/identifiers.npz "$RES/"`
  collapsed 13 origins onto one filename. Support is the set of `(origin, unit)` pairs present at
  *every* horizon and cannot be rebuilt from one origin. Fixed to write one file per origin and to
  refuse to delete the cube if none survive — but **this run's support is not reconstructible**, so
  step 2 needs a 7-minute re-emit. Recorded rather than quietly re-run.

#### Scope

One seed, one vehicle, one target (`sb`), calibration partition. The other three ε=0 seeds are scored
but their cubes are deleted; extending to 4 seeds is 4 × 7 min. No CRPS claim is made here — the
`crps_all` comparison against a 1-sample reference is the degenerate path `assert_sample_cube` exists
to refuse (C-220), and Epic #263 already ruled that class ARTIFACT.

---

### EXP-02 · scoring persistence FAIRLY · 2026-08-21 22:10 → 22:35 · **the win survives, at 2.0–2.5× not 3×**

EXP-01 left one claim unearned: the margin. Persistence was scored from a two-level binary signal
while the arm got a continuous gate probability. This gives persistence its ranking back, and found a
second, independent way the baseline was being understated.

#### Two corrections, both in the same direction

**1. Rank by the persisted VALUE, not the indicator.** `p = (cs > 0).mean(1)` at S=1 can only be 0.0
or 1.0. Ranking by `truth[m0-1]` uses the same data with more of its information; AP is invariant to
monotone transforms, so the raw value *is* the ranking — nothing is scaled or calibrated.

**2. `score_v2_horizons` never loads month `m0-1`** (→ **#282**). It builds
`months = {m0 + h - 1}`, while `_persistence_gathered` reads `truth_map.get((m0 - 1, u), 0.0)`.
That month is present only *by accident* — the origins are consecutive, so each origin's history is
its predecessor's h=1 forecast month. **The first origin (457, needing 456) has no predecessor**, so
its persistence forecast is silently all zeros. Reproduced to four decimals before being believed:

| h | scorer | with `m0-1` loaded |
|--:|--:|--:|
| 1 | 0.1461 | 0.1632 |
| 18 | 0.1077 | 0.1152 |
| 36 | 0.0834 | 0.0870 |

#### The honest table

| h | L=300 arm | persistence (fair) | ratio | Δ | Δ/MDE | *EXP-01 ratio* |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.4774 | 0.2364 | **2.02×** | +0.2410 | 4.5× | *3.27×* |
| 6 | 0.4071 | 0.1675 | **2.43×** | +0.2396 | 4.4× | *3.63×* |
| 12 | 0.3596 | 0.1667 | **2.16×** | +0.1929 | 3.6× | *2.78×* |
| 18 | 0.3318 | 0.1416 | **2.34×** | +0.1902 | 3.5× | *3.08×* |
| 24 | 0.3027 | 0.1234 | **2.45×** | +0.1793 | 3.3× | *3.13×* |
| 30 | 0.2748 | 0.1082 | **2.54×** | +0.1666 | 3.1× | *3.25×* |
| 36 | 0.2287 | 0.0951 | **2.41×** | +0.1336 | 2.5× | *2.74×* |

Persistence at h18 rises **0.1077 → 0.1416 (+31%)** once scored properly. **The margin falls from
~3.1× to ~2.3×, and the verdict does not change**: the model beats persistence at every horizon out
to 36 months, with every gap **2.5×–4.5× the MDE**.

`N = 170430`, `n_event` 1343 / 1547 / 1779 at h1/18/36 — **identical** between the dossier's path and
the scorer's, which is what localised the discrepancy to the score rather than to the support or the
truth.

#### Falsifier: is the value-ranking simply a better-informed reference, or a thumb on the scale?

Pre-committed direction: collapsing a score to its indicator can only lose ranking information, so
value-ranking must be **≥** binary. Asserted over 200 random draws
(`test_binary_never_beats_value_on_a_random_sweep`), not on one hand-made example — **0/200
failures**. That is what licenses reporting EXP-01's 3× as an *upper bound* rather than as a result.

#### Both numbers stay in the table

The binary column is what **M1** was built on. Dropping it would hide the size of the correction
rather than show it — and the correction runs *against* us, which is the reason to keep it visible.

#### What this does to M1

M1's persistence column inherited **both** defects. So M1 **understated persistence twice over**: its
claim that persistence beat every arm from h6 on was *stronger* than it was written. M1 was **true**
for the vehicle it measured — and is **false** for a converged one.

#### Scope

Unchanged from EXP-01: one seed, one vehicle, `sb`, calibration partition. The fair reference is
CPU-only and reusable — `results/identifiers/` now holds all 13 origins, so extending to the other
three ε=0 seeds costs 3 × ~7 min of emit and no new method.

---

### EXP-03 · all four ε=0 seeds · 2026-08-21 23:12 → 23:54 · **n=1 → n=4, and it holds**

EXP-01/02 rested on **one seed**. The ledger's own standard, in the section M34 edits, is
*"positive findings at n=1 have historically evaporated on proper runs"* — and **M8 was demoted on
exactly that count** hours earlier (#280). A good result does not get a pass a bad one was denied.

Three more emit-only passes on existing weights (seeds 42, 44, 45), ~14 min each, **no failures**.

#### The question the summary asks

Not *"does the mean beat persistence"* — a mean can be carried by one lucky draw, which is what the
n=1 warning is actually about — but **"does the WORST seed beat persistence at every horizon."**

| h | mean | sd | **worst** | persistence | worst/persist | (worst−p)/MDE |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.4767 | 0.0035 | 0.4716 | 0.2364 | **1.99×** | 4.3 |
| 6 | 0.3961 | 0.0123 | 0.3786 | 0.1675 | **2.26×** | 3.9 |
| 12 | 0.3565 | 0.0189 | 0.3312 | 0.1667 | **1.99×** | 3.0 |
| 18 | 0.3257 | 0.0134 | 0.3058 | 0.1416 | **2.16×** | 3.0 |
| 24 | 0.2993 | 0.0092 | 0.2881 | 0.1234 | **2.33×** | 3.0 |
| 30 | 0.2657 | 0.0099 | 0.2528 | 0.1082 | **2.34×** | 2.7 |
| 36 | 0.2250 | 0.0122 | 0.2108 | 0.0951 | **2.22×** | 2.1 |

**The worst seed beats persistence at every horizon**, by **2.1×–4.3× the MDE**. Seed sd is
0.0035–0.0189 — an order of magnitude below the gap it would have to cross.

#### Why this is n=4 and not 4×n=1

`aggregate_seeds.py` **refuses to aggregate unless the support is identical across seeds**. It is:
`N = 170430` on all four. Persistence is a **truth-only** baseline, so given one support it returns
one number — which is why the persistence column is a single value rather than a mean, and why its
constancy *is* the evidence that the seeds are comparable. Had it varied, the seeds would not have
been measuring the same thing and no summary of them would have meant anything.

Independent consistency check: seed 42's h18 AP of **0.3298** reproduces its archived value from the
SS dossier exactly, on a run that shares no code path with the one that produced it.

#### Verdict

**M34 is upgraded from n=1 to n=4** and the ratio settles at **~2.0–2.3×** (the n=1 seed-43 draw read
slightly high at h36: 2.41× against a 4-seed worst of 2.22×). The claim survives the standard that
demoted M8.

#### Cost and estimate error

**41 minutes wall-clock** for three seeds; I predicted 21. The emit is ~6 min but the surrounding
score/setup is ~8 more, and I had costed only the emit. **Fourth consecutive run-time estimate that
came in low.** The standing correction — state estimates as ranges — did not help here because I
quoted a point again.

#### Trap left by these runs, recorded rather than left as a landmine

`results/run.log` is **append-only across runs**, so it still contains the `ABORT` line from the
EXP-01 `--out` bug. Grepping that file for `ABORT` reports a failure that was fixed hours ago — it
misled *me* mid-run. The script never relies on it: it signals through **exit codes** and per-seed
**`PERSIST_DONE_<model>`** sentinels. Any caller checking these runs must do the same.

#### Scope

Unchanged otherwise: one vehicle, `sb`, calibration partition, **AP only**. The `crps_all` ARTIFACT
verdict stands untouched. Retention is still 0.69.

---

