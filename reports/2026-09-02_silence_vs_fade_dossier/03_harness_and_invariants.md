# 03 — Harness and invariants

**2026-09-02** · audited against the repo as of `36b78d2`

## A. Invariant taxonomy

### A.1 Hard — never break (violating one invalidates the experiment)

| Invariant | Where it is enforced |
|---|---|
| A run with the new diagnostic **off** is byte-identical to today | to be asserted by the new parity test (§C.2) |
| `identity` feedback ≡ `None` feedback, byte-exact | `tests/test_feedback_transform_seam.py::test_F3_none_is_byte_identical_to_identity` |
| Seed lock: same seed + `pass_index` ⇒ byte-identical cube | `to_cube_samples` per-`(pass, step)` sub-generator (ADR-070) |
| Fail loud, never silently substitute a value | repo-wide; the direct cause of C-318's severity |
| No comparison across seeds | discipline (see A.3) |
| Full suite green + `ruff` before any commit | `ship-it` gates |

### A.2 Deliberately changed by this program

**One thing only:** a new default-off diagnostic that writes the body-mean field `mu` per step
(§C.1). It adds an output; it changes no computation. If it changes a number, that is a bug and the
parity test (§C.2) must fail.

### A.3 Respect while changing

* **Seed hygiene.** State numbers in the falsifier-checks dossier are **seed 43**; field numbers are
  **seed 42**. These were nearly compared as one chain once already. Every comparison here is
  **within** a vehicle.
* **BatchNorm buffers.** The pushforward audit found an extra forward pass in `train()` mode silently
  writing running stats. Any new forward introduced here must be inside `eval()`, or it confounds
  everything downstream. This program adds **no** forward pass — it dumps a tensor that already exists.
* **Sampler coupling.** The per-`(pass, step)` seeding exists because a single streamed generator
  coupled `h=1`'s draws to later steps. Do not touch generator construction.

## B. The standing harness — what already exists (reuse, do not reinvent)

| Mechanism | Present? | Detail |
|---|---|---|
| Default-off flag pattern | **yes** | `--freeze`, `--keep-cubes`, `return_params`, `_record_gate_probe` are all this shape |
| Parity / byte-identity gates | **yes** | the `identity ≡ None` seam test is the exact template for §C.2 |
| Reproducibility | **yes** | `torch_seed` + per-`(pass, step)` sub-generators; deterministic cube |
| Fast retrain-free readout | **yes** | `run_realism_arms.py` emit-only arms; ~12 min per arm |
| Evaluation comparability | **yes** | archived seed-42 `identity` and `use_real` results, `AP@h18 = 0.3298395823400329` as an exact identity anchor |
| Run discipline | **yes** | `setsid` daemons (background jobs get reaped), sentinel files, manifests, disk preflight |
| Negative-result discipline | **yes** | `07_experiment_log` + `RESULTS_LEDGER.md`; negatives are ledger rows (M47, M49) |
| Hardware gate | **yes** | fail-loud on missing accelerator |

### B.1 — CORRECTION: the instrument this dossier was opened to use does not exist as described

The opening premise was that the emitted cube is `log1p(compose_mean(mu, gate))` and therefore
`mu = expm1(lr)/gate`. **That is wrong**, and it would have produced a wrong answer quietly.

| path | composer | operation | what it governs |
|---|---|---|---|
| `_emit_magnitude` (`hydranet_inference.py:614`) | `compose_mean` | `gate * mean` — **multiplicative** | the **autoregressive feedback** |
| `to_cube_samples` (`sampling.py:23`) | `compose_samples` | `torch.bernoulli(gate)` mask — **stochastic** | the **written cube** |

The cube for a family model holds `B · log1p(y)`, `B ~ Bernoulli(g)`, `y ~ Family(params)` —
**draws, not means**. So per draw, `expm1(lr)/g = B·y/g`: zero for a `(1-g)` fraction of draws and
`y/g` otherwise. It is unbiased for `mu` but its variance is inflated by `1/g`, which at
`g ~ 1e-4` is unusable — and `g ~ 1e-4` **is the late-horizon regime the whole claim is about**.

This is the third time an instrument-level premise here has been wrong before use (C-318's sentinel;
the `train_time.py` tqdm scrape; this). The lesson the harness should carry: **read the writer, not
the reader.** Verified by reading `compose_samples`, `compose_mean`, and both `_emit_magnitude` call
sites, not by inference from names.

### B.2 — Conditioning of the estimator

`MAGNITUDE = Σ(g·mu) / Σ(g)` is a ratio of field-wide sums. Relative error on the numerator scales as
`1/sqrt(S · Σg)` rather than `1/sqrt(S · g)`, so aggregating across ~13k cells recovers the precision
that the per-cell division destroys. **Per-cell division by the gate is banned in this program.** It
is the exact shape of the mistake C-318 recorded: an arithmetically unstable quantity averaged as if
it were stable.

### B.3 — Known-faulty instrument, retained only as an explanandum

`_record_feedback_stats` writes `-1.0` as an in-band UNDEFINED sentinel for
`mean_magnitude_on_active`, `persistence`, and `neighbour_pairs_per_active` when `n_active == 0`.
The code comment warns against averaging it; the first M50 averaged it anyway. It is read in this
program **only** to reproduce and explain the earlier number, never as evidence for a claim.

## C. New harness this program needs (gates the first run)

### C.1 — The `mu`-field dump (the only new code)

Write `family.mean(params)` per step as a `[T, n_reg, H, W]` float32 `.npy`, from the params the
family path already computes at `hydranet_inference.py:1273`. Default off, behind an explicit path
argument. No new forward pass, no `train()` mode, no generator touched.

### C.2 — Parity test (**blocking**)

A run with the dump **off** must be byte-identical to today's committed result, and a run with it
**on** must produce the identical cube as with it off. Modeled directly on
`test_F3_none_is_byte_identical_to_identity`. Without this, the instrument could perturb the thing it
measures.

### C.3 — Identity check G1 (**blocking**)

I1's `MAGNITUDE` (ratio of sums, from cubes) and I2's gate-weighted `mu` (from the dump) compute the
same quantity by different routes. They must agree within **10%** at every horizon. Disagreement
means at least one instrument is wrong, and the program halts rather than picking the agreeable one.

### C.4 — Sentinel guard

Any statistic derived from a field that can be empty must assert no sentinel value survives into an
aggregate. A direct, mechanical descendant of C-318.

## D. Pre-flight checklist — all must be green before EXP-1

- [ ] `05` pre-registration written **and committed** (locked before any GPU second)
- [ ] C.1 dump implemented, default off
- [ ] C.2 parity test green — dump off ≡ today, dump on ≡ dump off
- [ ] C.4 sentinel guard in the analysis script, with a test
- [ ] `ruff` + full `pytest` green
- [ ] Working tree clean (no commits while a queue runs — the runner aborts on HEAD drift)
- [ ] Disk preflight for `--keep-cubes` (cubes are large; a prior run filled the disk)
- [ ] Identity anchor confirmed: the rerun `identity` arm reproduces `AP@h18 = 0.3298395823400329`

**G1 (C.3) is checked after EXP-1 emits but before any claim is read off it.**
