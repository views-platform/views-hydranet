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
