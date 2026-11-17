# Class Intent Contract: PosteriorCubeSampler (`views_hydranet/distributions/sampling.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-08-14
**Related ADRs:** ADR-067 (distribution families / D×K posterior, A-S8), ADR-068 (arm naming),
ADR-069 (forecast_composition axis), ADR-070 (per-`(pass,step)` seeding /
T=0-neutrality)

> **Amended 2026-11-17 (issue #628, ADR-079 §4):** the `zinb` family and the bodies `hurdle_nb`, `hurdle_lognormal`, `hurdle_shrinkage` and `dense_nb` are deleted, with everything only they used (`sample_core`, `mean_core`, `parameter_penalty`, the `self_zeroed` marker, `emit_family_core`, the π penalty). Below, statements of current guarantees are corrected; mentions in history and measurements are left as they were. Kept under the git tag `archive/unused-bodies-2026-11-17`.

---

## 1. Purpose

> Turn **one** MC-dropout pass's activated distribution-family params into the K per-cell posterior
> draws that fill that pass's slice of the `[T,H,W,n_reg,S]` cube — drawing in **count space** via
> the family, converting to **log1p** (emit) space, then composing with the gate per the
> `forecast_composition` arm. This is the D×K sampler (D = MC-dropout passes, K = per-cell head
> draws) behind ADR-067 A-S8 posterior emission.

The single public function is `to_cube_samples(...)` (`sampling.py:23`). It bridges the space
boundary the whole emit path shares: a family samples in **count space** (`family.sample`,
`base.py:71`), but the cube — like `_emit_magnitude` — lives in **log1p space** so the downstream
`inverse_transform` (`expm1`) recovers counts (`sampling.py:3-8`, `:103`). Determinism rides on the
caller-supplied seeded `torch.Generator` (S2 #121 gate, `sampling.py:7`).

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** own family math. It calls `family.sample`
  and never itself parameterises NBCore, activates a head, or knows a family's `n_params` beyond
  reading it (`sampling.py:59`).
- Does **not** make the RAM-preflight decision. The oversize-cube guard is
  `disk_guard.assert_cube_fits` (`disk_guard.py:64`), called by the orchestrating loop **before**
  allocation (`hydranet_inference.py:697`); `to_cube_samples` allocates only its own single-pass
  `[t,H,W,n_reg,k]` slice (`sampling.py:78`).
- Does **not** own gate-composition **policy**. It dispatches to `ForecastComposer.compose_samples`
  (`sampling.py:17`, `:105`); the per-arm masking semantics live there (see the ForecastComposer
  CIC).
- Does **not** validate the config cross-field arm rules (self-zeroed-may-not-be-gated,
  nb-requires-a-gate) — that is `HydraNetConfig` at the config boundary (ADR-069). It fails loud only
  defensively on its own arguments (`sampling.py:61`, `:70`, `:83`).
- Does **not** produce the gate. The classification head does; this consumes the first `n_reg`
  channels of the caller's gate cube (`sampling.py:88`).
- Does **not** run the D loop or MC-dropout. The caller (`generate_posterior_samples`) owns the D
  passes and calls `to_cube_samples` once per pass (`hydranet_inference.py:772-806`).

---

## 3. Responsibilities and Guarantees

- **D×K slice assembly.** Returns `np.ndarray` `[T, H, W, n_reg, k]` float32 in **log1p space**,
  non-negative (log1p of non-negative counts) (`sampling.py:55-56`, `:78`, `:103`). One call fills
  one MC-dropout pass's `K`-column slice.
- **Per-`(pass, step)` sub-generator seeding (ADR-070, `66a95ea`).** Each timestep `tt` draws from
  its **own** sub-generator seeded from `base + pass_index*1_000_003 + tt*10_007` (masked to 63 bits)
  where `base = generator.initial_seed()` (`sampling.py:97-100`). A single generator streamed across
  the 36-step trajectory would couple a step's draws to LATER steps' params (torch's batched Gamma
  rejection re-draws across the whole tensor), so `rollout_feedback` — which only changes h≥2 params
  — would perturb the SCORED h=1 cube. Independent per-step streams make each step's draw depend only
  on its own params, so the T=0 (h=1) cube is **byte-invariant** to `rollout_feedback`. This is the
  T=0-neutrality the sample-feedback default rests on (`sampling.py:89-96`).
- **Reproducibility (S2 #121).** Same seed + same `pass_index` ⇒ byte-identical cube; a different
  seed differs (`sampling.py:94-96`; test `test_to_cube_samples_is_generator_deterministic`,
  `test_sampler_dxk.py:93`). For a single step (`T=1`) the per-step scheme reproduces the pre-fix
  draw exactly (LOCKED golden anchors preserved, `sampling.py:95-96`).
- **Composition dispatch (ADR-069, #183).** `self_zeroed` is passthrough — byte-identical to the
  pre-ADR-069 path, no gate touched (`sampling.py:80-82`). `soft_gate` / `threshold_gate` mask the
  log1p draws via `compose_samples`, using the **same** per-step generator `gen_t` so the Bernoulli
  is deterministic (`sampling.py:104-105`).

---

## 4. Inputs and Assumptions

Signature (`sampling.py:23-34`):

```
to_cube_samples(params_zstack, family, k, generator, n_reg,
                gate=None, composition='self_zeroed', threshold=None,
                pass_index=0) -> np.ndarray  # [T, H, W, n_reg, k] float32, log1p space
```

- `params_zstack`: **activated** params `[T, n_reg*n_params, H, W]` (torch or numpy), target-major;
  coerced to `float32` (`sampling.py:37`, `:58`).
- `family`: the resolved `DistributionFamily` (owns `n_params`, `sample`)
  (`sampling.py:38`, `:59`).
- `k`: per-cell head draws (K) (`sampling.py:39`).
- `generator`: seeded `torch.Generator` whose `initial_seed()` is the seed base; **may be `None`**,
  in which case `base = 0` (`sampling.py:41`, `:97`).
- `n_reg`: number of regression targets = the `n_reg` axis width (`sampling.py:40`).
- `gate`: per-cell `P(y>0)` `[T,H,W,n_cls]`; only the first `n_reg` channels are used. Required when
  `composition` gates the body; ignored for `self_zeroed` (`sampling.py:43-44`, `:88`).
- `composition`: `self_zeroed` (default) / `soft_gate` / `threshold_gate` (`sampling.py:45-47`).
- `threshold`: τ ∈ (0,1) for `threshold_gate`; forwarded to `compose_samples` (`sampling.py:48`,
  `:105`).
- `pass_index`: MC-dropout pass index (D axis), folded into the per-`(pass,step)` seed
  (`sampling.py:49-50`, `:99`).

**Assumed valid by construction** (guaranteed upstream by `HydraNetConfig`, re-checked only
defensively here): the arm keyword and τ's range for `threshold_gate`.

---

## 5. Outputs and Side Effects

- **Produces** one `np.ndarray` `[T, H, W, n_reg, k]` float32, **log1p space**, non-negative
  (`sampling.py:55-56`, `:78`, `:106`). No in-place mutation of inputs; `params_zstack` is copied via
  `torch.as_tensor(np.asarray(...))` (`sampling.py:58`).
- **Stochastic:** the draws are random, seeded deterministically by
  `(generator.initial_seed(), pass_index, step)` (`sampling.py:97-100`). Under `soft_gate` the
  per-draw Bernoulli mask is likewise seeded from the same per-step generator (`sampling.py:105`).
- **No I/O, no logging, no config reads, no global state.** The caller allocates the full cube and
  writes this slice into it (`hydranet_inference.py:790`, `:783`).

---

## 6. Failure Modes and Loudness

All guards raise `ValueError` — fail loud, never silent fallback (ADR-003):

- **Channel-dim mismatch** (`sampling.py:61-65`): `params` channel `c != n_reg * n_params` raises
  `"to_cube_samples: params channel dim ... != n_reg*n_params"`. Test
  `test_to_cube_samples_rejects_channel_dim_mismatch` (`test_sampler_dxk.py:236`).
- **Gated arm without a gate** (`sampling.py:82-86`): `composition != 'self_zeroed'` with
  `gate=None` raises `"... needs a gate, got None"`. Test
  `test_to_cube_samples_gated_arm_requires_gate` (`test_sampler_dxk.py:245`).

**Documented OPEN risk (NOT guarded here):** the C-266 symmetric case (the mirror of C-240) is
guarded **upstream** by `HydraNetConfig`, not by this sampler — do not read a sampler guard into it.

---

## 7. Boundaries and Interactions

- **Calls:** `family.sample` and
  `compose_samples` (`sampling.py:17`, `:105`). It treats the family and the composer as opaque
  authorities over count-space draws and gate policy respectively.
- **Called by:** `HydraNetInference.generate_posterior_samples` — once per MC-dropout pass `d`, with
  `pass_index=d`, the pass's gate, and the config's `forecast_composition` / `gate_threshold`. That method (via
  `inference_orchestrator.py:68`) owns the D loop, the RAM preflight (`assert_cube_fits`,
  `hydranet_inference.py:697`), and cube allocation.
- **Sibling on the point path:** `_emit_magnitude` composes the **mean** (count space) via
  `compose_mean` (`hydranet_inference.py:253-268`); `to_cube_samples` composes **draws** (log1p
  space) via `compose_samples`.
- **Must not depend on:** disk/RAM guards, config objects, or the D-loop orchestration — those are
  the caller's.

---

## 8. Examples of Correct Usage

Self-zeroed nb draw (default arm), deterministic:

```python
fam = resolve_family("nb")
params = _activated_params(fam, t=2, h=4, w=4, n_reg=3)  # [t, n_reg*n_params, h, w]
cube = to_cube_samples(params, fam, k=5, generator=torch.Generator().manual_seed(7), n_reg=3)
assert cube.shape == (2, 4, 4, 3, 5)  # log1p space, non-negative
```
(`test_to_cube_samples_shape_and_log1p_space`, `test_sampler_dxk.py:82`.)


---

## 9. Examples of Incorrect Usage

- **Gated arm, no gate** — `to_cube_samples(..., composition="soft_gate", gate=None)`: raises. A
  gating arm needs the classifier `P(y>0)` (`test_sampler_dxk.py:245`).
- **Streaming one generator across steps to save a `Generator` alloc.** Do not "optimise" away the
  per-`(pass,step)` sub-generator — it is the T=0-neutrality invariant (ADR-070), not a micro-opt.
  Reusing one stream re-couples h=1's draws to h≥2 feedback (`sampling.py:89-96`).
- **Passing a raw (unactivated) param stack, or one whose channel width ≠ `n_reg*n_params`.** Raises
  the channel-dim guard (`test_sampler_dxk.py:236`).

---

## 10. Test Alignment

- `tests/distributions/test_sampler_dxk.py` — the D×K sampler unit + guards:
  `test_to_cube_samples_shape_and_log1p_space` (shape + log1p non-negativity, `:82`),
  `test_to_cube_samples_is_generator_deterministic` (same seed ⇒ equal, diff seed ⇒ differ, `:93`),
  `test_to_cube_samples_rejects_channel_dim_mismatch` (`:236`),
  `test_to_cube_samples_gated_arm_requires_gate` (`:245`), plus the end-to-end
  `test_generate_posterior_samples_dxk_fill` / `_family_deterministic` / `_legacy_width_unchanged`
  (`:205`, `:217`, `:225`).
- Regression to protect: the T=0 byte-invariance to `rollout_feedback` (per-`(pass,step)` seeding)
  and single-step golden equivalence (`sampling.py:89-96`); green determinism; red fail-loud guards.

---

## 11. Evolution Notes (Optional)

- **Stable:** the `[T,H,W,n_reg,k]` log1p-space return; the per-`(pass,step)` seeding scheme and its
  seed arithmetic (changing the multipliers or fold order breaks LOCKED golden anchors and the
  T=0-invariance); the two fail-loud guards.
- **Expected to change:** new families (they arrive via `family.sample`, no sampler
  edit) and new composition arms (added in `ForecastComposer`, dispatched here unchanged).
- **Would require revisiting this contract:** moving any config-cross-field validation into the
  sampler (currently the config boundary's job), adding a sampler-side guard for C-266, or making the
  sampler own the D loop / RAM preflight.

---

## End of Contract

This document defines the **intended meaning** of `to_cube_samples` (`views_hydranet/distributions/sampling.py`).

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
