# Post-Mortem — Training Non-Determinism Root-Caused to Init-Time RNG Drift (C-119)

**Date:** 2026-06-14
**Companion to:** risk register C-119 (escalated Tier 3→1), C-160 (parity gate), C-79 (missing reproducibility test), C-42/C-43 (reproducibility gate); the channel-role side-quest plan + ADR-062; coordinate experiment #110.
**Status:** **ROOT CAUSE FOUND + FIX VALIDATED.** The fix (Path A) is *not yet applied* — to be planned and implemented next.
**Method:** read-only code audit + a controlled empirical bisection using short (1-lesson) trainings on **frozen data** (`--saved`, identical parquet), identical code, and the same locked seed (42). Determinism judged by **weight-tensor hash** (numpy `tobytes`), training **loss**, and **MCR** — never the `.pt` file sha (see §4).

---

## 1. How it surfaced (the trigger)

The channel-role side-quest (fixing the #108 static-channel seam) adopted a discipline before refactoring the 451-edge `VolumeHandler` (C-36): run the bounded hurdle-NB **no-coords** baseline *before* and *after* the refactor and require them to match — a parity gate (C-160).

The "before" run didn't reproduce the validated Jun-11 baseline. Suspecting data drift, we re-ran the **identical** no-coords config (frozen data, seed 42) — and it **didn't match the first run either**: different trained weights, ~20% different eval (FULL MCR sb **3.69 vs 2.99**, os **6.78 vs 8.47**; CRPS ±~20%). Two identical runs diverging is what exposed the non-determinism. Every unit-level guard was green — the I5 "byte-identical when off" seam test, the reproducibility gate's "Entropy locked" log. **Only the real end-to-end two-run comparison revealed it.**

## 2. The suspect space (hypotheses)

Non-deterministic op (CUDA, permitted by `use_deterministic_algorithms(..., warn_only=True)`); `ConvTranspose2d`/pooling backward atomics; multi-threaded parallel float-reduction; `PYTHONHASHSEED` set/dict ordering; dropout RNG; data/window sampling; weight initialization RNG.

*(The first investigation agents over-asserted here: one named `ConvTranspose2d` backward as the cause and called weight-init "unseeded"; another called the curriculum a "crash." These were verified individually and several were wrong — see §5. Conclusions below rest on the runs, not the agents.)*

## 3. What happened — the bisection

| Test (short run, frozen data, seed 42) | Result | Eliminates |
|---|---|---|
| `use_deterministic_algorithms(True, warn_only=False)` | did **not** raise | op-level CUDA non-determinism (all ops have deterministic impls) |
| CPU vs GPU (2 CPU runs) | differ | CUDA-specific (atomics/cuBLAS) |
| **verified** single-thread (`torch.get_num_threads()==1`) | differ | threading / parallel float-reduction |
| `PYTHONHASHSEED=0` | differ | hash-ordering of sets/dicts |
| `dropout_rate=0` | differ | the forward RNG (dropout) |
| sampled-window **data hashes** per window | **identical** | data / window sampling |
| `make()` in isolation (lock → make immediately) ×2 | **identical** | the init logic itself |
| **init-weights hash at training start, real pipeline ×2** | **differ** | → **init-time RNG drift = the cause** |
| **fix:** re-seed in `make()` before init, real config (GPU, threads, dropout=0.15) ×2 | **bit-identical** weights | confirms the fix |

**Mechanism.** The manager locks the seed (`hydranet_manager.py:279`), then the pipeline does work (data load, diagnostics, …) that advances the torch RNG by a **non-deterministic amount** before `make()` (`training_engine.py:~70-74`) runs `model.apply(init_fn)`. So initialization draws from a drifted RNG state → **different initial weights every run** → everything downstream diverges. The re-seed at `training_engine.py:494` happens *after* init — too late. `make()` *in isolation* (lock→make with nothing between) is deterministic, which is exactly why the unit-level view never caught it.

## 4. The decisive evidence — and a measurement trap we nearly fell into

The clean discriminator: **`make()` in isolation is deterministic, but the real-pipeline init-weights hash differs run-to-run.** That localizes the fault to RNG consumed *between the seed-lock and init*, not to any op, device, thread, or data path. Re-seeding immediately before init then produced **bit-identical** weights (`5c8413bd…`) on the full production config, with training loss identical to 5 decimals.

**The trap (recorded as a method standard):** the saved **`.pt` file sha256 is not a valid weight-identity check.** torch's `.pt` is a zip archive that embeds file mtimes, so two saves of *identical* weights produce *different* file bytes. Early in the bisection the `.pt` sha was the signal — which would have shown "still non-deterministic" even *after* a correct fix. The tell was that the training **loss** became bit-identical while the `.pt` sha still differed; switching to the **weight-tensor hash** resolved it. **Always judge determinism by weight-tensor hash / loss / MCR, never the `.pt` file sha.**

## 5. Intellectual-honesty audit (biases weighed)

- **A wrong leading hypothesis, refuted by its own test.** Going in, the strongest hypothesis was `warn_only=True` permitting `ConvTranspose2d` non-determinism. The `warn_only=False` probe **did not raise** — refuting it. We accepted the refutation rather than rescuing it; the bisection continued.
- **Agent output was unreliable; primary runs were trusted instead.** The fan-out agents produced contradictory and incorrect verdicts (init-order, the curriculum "crash," `ConvTranspose2d`). Each was checked directly against code and runs. This is the second time this session that agent static-analysis verdicts proved wrong (cf. the channel-role census) — reinforcing "verify, don't trust."
- **A near-false-negative caught.** The first "single-thread" runs set `OMP_NUM_THREADS=1` but I had **not verified** torch honored it; only after checking `torch.get_num_threads()==1` was the threading hypothesis cleanly eliminated. Had I trusted the env var, I might have wrongly excluded threading.
- **Scope honesty (what we did *not* run).** The fix is validated at **1 lesson (3 windows)**, not a full 40-lesson run. Determinism compounds (identical state + identical step → identical next state), so 1-lesson bit-identity strongly implies full-run bit-identity — but a full-run confirmation has not been executed. We also did **not** name *which* operation consumes the RNG between `:279` and `make()`; the fix (re-seed before init) makes init independent of it, so the exact consumer is not load-bearing for the fix, but it remains unidentified.

## 6. What is / isn't established

**Established:** training non-determinism is caused by init-time RNG drift; re-seeding before init yields bit-identical weights on the real config (GPU + multi-thread + dropout); the `.pt` file sha is an invalid determinism signal. C-119's prior "non-deterministic CUDA kernels / cannot force bitwise determinism" hypothesis is **wrong** — it *is* forceable, with a one-line ordering fix.

**Not established:** full 40-lesson bit-identity (not run); the identity of the pre-`make()` RNG consumer; whether genuine multi-seed variance (different seeds) is acceptable for the science (a separate question from this fixed-seed bug).

## 7. Disposition & program-wide implications

- **C-119:** escalated Tier 3→1, root-caused, fix recorded. Resolves when Path A lands **and** a pipeline determinism regression test (C-79) pins two-run weight-tensor identity.
- **Path A (the fix, recommended):** re-seed (`lock_entropy`) immediately before model init in `make()`; add the determinism regression test; optional hardening — flip `warn_only=True→False` (fail-loud; the probe confirmed nothing currently raises) and set `PYTHONHASHSEED`. *To be planned and implemented next.*
- **Every single-run comparison to date is confounded.** The coordinate experiment **#110 "coords made it worse" verdict cannot stand** — the no-coords baseline alone swings 2.99–3.69 run-to-run, comparable to the claimed coord effect. The FAO eligibility table and RESULTS_LOG single-run deltas are likewise suspect. After Path A, the coordinate experiment (and any comparison we intend to trust) must be **re-run on the deterministic pipeline**, and/or made multi-seed/distributional.
- **The parity discipline worked.** A two-run determinism check — not the unit suite — caught this. It becomes standard before risky refactors, and the missing pipeline-determinism test (C-79) is exactly the cheap guard that would have caught this far earlier.

## 8. Meta-lesson

This sits under a long arc — the C-113 autoregressive-explosion work, the coordinate experiment, and three mid-flight patches — **all conducted on a model that was silently not reproducible at a fixed seed.** Comparative conclusions drawn from single runs across that period inherit this confound. The fix is one line; the cost was weeks of comparisons resting on ~20% phantom variance. The discipline that prevents a repeat is cheap and now mandatory: **pin reproducibility with a test before drawing any comparative conclusion.**
