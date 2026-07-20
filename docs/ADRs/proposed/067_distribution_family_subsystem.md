# ADR-067: A distribution-family subsystem + per-cell sampleable NB/ZINB heads

**Status:** Proposed
**Date:** 2026-07-20
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## Summary (read this first — self-contained)

The model's forecast for each map cell is built from a **gate** (chance of *any* violence) and a **body**
(guess for *how many* deaths). Which *kind* of body is used is chosen today by one config string,
`output_distribution`, that is read and branched at ~11 places across 7 files — so adding a new body kind
means editing all of them. Worse, the count-body *experiments* tried on this branch (`dense_nb`,
`hurdle_nb` — never shipped) never actually produced a distribution you can *draw samples from*: they emit
a single average number per cell, and their spread parameter is one global constant shared by every cell.

This ADR does two things: (1) introduces a small **`distributions/` subsystem** where each body kind is
one self-contained file registered in one table, so adding a kind touches one place, not eleven; and
(2) builds two real per-cell **count distributions we can sample from** — **NB** (negative binomial) and
**ZINB** (NB with an extra spike of zeros) — that emit a per-cell average *and* a per-cell spread, and
draw samples into the evaluation pipeline. Existing body kinds keep behaving bit-for-bit identically.

*(A glossary of the technical terms is at the end.)*

---

## 1. Context

**Setting.** HydraNet forecasts monthly conflict deaths per grid cell — data that is **~99.7% zeros with a
heavy right tail**. Every forecast must answer two things: *whether* any violence occurs (usually not) and
*how many* deaths if it does. The existing models split this into a **gate** (whether) × a **body** (how
many); the new count families instead fold both into **one distribution**, which must therefore pile
probability on 0 *and* keep a heavy tail. That is what makes a **count** family — especially one with an
explicit zero-spike (**ZINB**) — the natural fit, and why getting its *per-cell* spread right is what this
ADR is about.

**Problem.**
- **No registry.** `output_distribution` is a bare string read and branched at **~11 places across 7
  files** — the emit/sampler logic, the head's activation and channel-sizing, the config valid-list, the
  loss wiring, and the saved-model round-trip. Adding a body kind means editing all of them: an
  **Open/Closed violation** (a new kind should be *added* as new code, not force *edits* to existing code)
  eleven times over. By contrast `loss_reg`/`loss_class` already use clean registries (`utils.py`
  `LOSS_REG_REGISTRY`) — the good pattern exists in-repo; `output_distribution` just never got one. (The
  full file:line census is in `reports/2026-07-20_distributional_head_dossier/02_design.md`.)
- **No sampleable per-cell count head.** Investigation (2026-07-20) confirmed the branch's count-body
  experiments (`dense_nb`, `hurdle_nb` — never shipped to production) emit a single average number per cell
  (`log1p` of the mean); the dispersion `theta` (the spread) is a
  **single global constant per target** shared by every cell; and there are **zero `.sample()` calls** in
  the repo — all inference randomness comes from **MC-dropout** (Monte-Carlo dropout: repeated forward
  passes with dropout left on), not from any distribution. **Why this matters:** evaluation scores an
  *ensemble of posterior samples per cell* (CRPS, calibration), so a mean-only head cannot feed a real
  per-cell distribution into scoring — it just re-scores the same mean under dropout noise.

**Urgency.** The volatility-ceiling program showed per-cell *spread* is predictable — per-cell volatility
is rank-predictable (Spearman ≈ 0.79 on active cells), and conditional quantiles come out 48% sharper
*than the unconditional ones* while staying calibrated — but the current one-spread-for-all head cannot
express it. Testing that hypothesis requires a real
per-cell distributional head, and we do not want to build it on top of the 11-site switch and breed more
of it.

---

## 2. Decision

**Statement.** We will add a `views_hydranet/distributions/` subsystem exposing one `DistributionFamily`
abstraction and an explicit registry, and implement `nb` and `zinb` as per-cell, **sampleable** families
routed through it — leaving all existing body kinds untouched.

**In-Scope.**
- **The abstraction — one interface every body kind implements** (`base.py`, `DistributionFamily`). Its
  methods: `n_params` (how many numbers the head emits per cell), `activate` (turn those raw numbers
  into valid parameters), `nll` (the training loss), `sample` (draw outcomes), `mean` (the expected
  value), `prob_positive` (`P(Y>0)`, used for gate scoring — see §5), and `initial_raw_bias`
  (per-family informed-init biases the head applies — C-199/C-203); plus the `needs_latent` flag.
  *Why the wider head is safe:* the head emits `n_params` channels per target, but only the single `mean`
  is fed back into the autoregressive rollout — so the rollout width (and the autoregression invariant) is
  unchanged.
- **The registry — one table, one place to look things up** (`registry.py`). An explicit `name -> factory`
  map (torch-free at import, mirroring `LOSS_REG_REGISTRY`); the list of valid names (`FAMILY_NAMES`) is
  derived from its keys, so there is one source of truth. Two accessors: `get_family()` (raises on an
  unknown name) and `resolve_family()` — the **single place dispatch happens**, returning the registered
  family, or `None` for any name that is **not** a registered family (the existing `output_distribution`
  values, which keep their own code path).
- **The families — `nb` and `zinb`, both self-zeroed** (`negative_binomial.py`,
  `zero_inflated_negative_binomial.py`). `nb` emits 2 parameters per cell (`mu`, `theta`); `zinb` emits 3
  (`mu`, `theta`, `pi`). Both are built on a shared `NBCore` (composition, not inheritance).
  **Self-zeroed** means the count distribution produces its own zeros over all cells, so these families
  need **no separate gate** — unlike the existing `hurdle_*` bodies, which multiply a positive body by a
  gate. And `theta`/`pi` are emitted **per cell** — the whole point of the change.
  *Why ZINB's `pi` is not `1 − gate`:* `pi` is the **structural** zero-inflation probability (the extra
  always-off mass), whereas the gate's complement is the *total* zero probability —
  `1 − gate = π + (1−π)·NB(0)` — which mixes structural zeros with the NB's own zeros. Setting `pi = 1 −
  gate` would double-count. A model where the gate owns **all** zeros (with a **zero-truncated** NB body)
  is a **hurdle**, a distinct family (`C-146`) — so a sampleable `hurdle_nb` is a candidate **third arm**
  for the M2 comparison (`nb` vs `zinb` vs `hurdle_nb`), added like any family (one file + one registry entry).
- **Sampling — the D×K posterior.** Keep the **D** MC-dropout passes (`n_posterior_samples`, *model*
  uncertainty) and, within each pass, draw **K** per-cell samples (`n_head_samples`, *outcome*
  uncertainty). Together they fill the existing `[T,H,W,C,S]` sample cube with `S = D×K`, using a **seeded
  generator** so inference stays deterministic (verified by the determinism gate, S2 #121).
- **Config — two knobs.** `output_distribution` now also accepts the registered family names; a new field
  `n_head_samples` sets **K** (default **1**; `K>1` is valid only for the new families, so every existing
  config is unaffected).
- **Integration — the strangler-fig seam.** At each existing switch-site: `fam = resolve_family(name)`; if
  it is a family, delegate to it, otherwise fall through to the existing code. New families add **zero** new
  branches; existing branches are untouched.

**Out-of-Scope (deliberately — deferred to the Epic-B draft, #181).**
- Migrating the existing families (`standard`/`hurdle_*`/`dense_nb`/`quantile`) onto the abstraction and
  deleting their switch-sites.
- Relocating the loss modules out of `utils/`.
- The `output_distribution` → `body` rename — an optional cosmetic follow-up, deferred to **Epic B**
  (#181). (The earlier ADR-066 that proposed this has been collapsed into this ADR.)

---

## 3. Rationale & Integrity Impact

**Logic.** Four choices, each for a concrete reason:
- **Registry, not more switch-branches** — the registry pattern is already proven in-repo for losses, so
  extending it to distributions is the consistent, low-surprise choice.
- **Explicit map, not a decorator** — chosen (in the v1 `/expert-code-review`) for **fail-loud** discovery:
  an unregistered name raises with the list of valid names rather than silently going missing, and it keeps
  one source of truth.
- **One cohesive interface, not four narrow protocols** — keeps the module *deep* (a small surface hiding
  real math) instead of fragmenting it into shallow pieces.
- **ZINB composes a shared `NBCore`, not the whole NB family** — because ZINB's likelihood is a
  zero-inflation mixture, not NB's, so only the count core is shared.

**Fortress State.**
- **Reproducibility:** the sampler uses a **seeded `torch.Generator`** so inference stays deterministic
  (preserves the S2 #121 determinism gate — a critical, easily-missed integrity risk, since the new
  `.sample()` calls are the first non-dropout randomness in inference).
- **Numerical stability:** per-cell `theta`/`pi` are riskier than a global scalar, so the families apply
  boundary sanity-guards; `NBCore` clamps out-of-range parameters (e.g. a spread of exactly zero) that
  would otherwise make the likelihood blow up.
- **Resource safety:** `S = D×K` can blow up the cube, so `n_head_samples`/`posterior_S` are **bounded by
  a config-validated ceiling** (exact limit set in A-S8) and **disk/RAM-preflighted** (`disk_guard`) before
  allocation.
- **Parameterization & the transform boundary:** distribution parameters live in their **natural space**
  via link functions — `mu`,`theta` = softplus (count/positive), `pi` = sigmoid — and are **never**
  `log1p`/`expm1`'d (`expm1` on a prediction is the C-113 explosion direction). A count likelihood needs
  *raw counts*, so the loss recovers them from the target using the **config's declared inverse**
  (`config_initializer.TRANSFORMS[method]`: `log1p→expm1`, `asinh→sinh`, `identity`) via a torch/GPU mirror
  of that inverse — **never** a hardcoded `expm1` (today `to_raw_counts` hardcodes it, silently assuming
  `log1p`). Fail-loud at config load if an `nb`/`zinb` body is paired with a target transform whose inverse
  is not count-compatible. (Closes **C-198**.)
- **Estimation robustness (the sparse-count regime):** with ~99.7% zeros, a `/falsify` simulation showed the
  information identifying per-cell `θ`/`π` is concentrated in the ~1% positive cells (98.8% of the `π`
  Fisher information) and the gradients are weak at the operating point (`π`≈0.99, and `θ` in the large / Poisson-limit
  regime). The head
  therefore **requires informed initialization** — `π` ≈ the empirical zero-rate, `θ` ≈ the global-`θ`
  baseline (not default/zero) — and the `nll` **offers active-cell weighting**; `π` carries a mild prior
  (deep-zero cells leave it non-identified — the `π/μ` ridge). Without these, `θ`/`π` collapse to
  constants, silently reducing to the global-θ baseline this ADR exists to beat. (Closes **C-199/C-200**.)

**Fail-Loud.** Unknown family names raise; `n_head_samples > 1` on a non-distributional head raises;
oversize `D×K` is rejected before allocation; and a registered family name that **collides with an existing
`output_distribution` value is rejected** — the byte-identical guarantee depends on the registry and existing
name-sets staying **disjoint**, or an existing config would silently route to the new family. All at
config-load or pre-run, never silently.

---

## 4. Consequences

### ✅ Positive
- [ ] A new body kind = **one file + one registry line**, zero consumer edits (Open/Closed for the future).
- [ ] The model finally has a **per-cell, sampleable** count distribution to test the predictable-spread
  hypothesis.
- [ ] **Inference *compute* is ~unchanged:** the `S = D×K` samples still cost ~D forward passes — the K
  per-cell draws are cheap elementwise ops, not extra passes. (*Memory* is the separate axis: the sample
  cube grows with `S = D×K`, which is bounded + preflighted per §3 — not "free".)
- [ ] The per-cell design **removes** two coupling sites (the *sidecar* save + reload — the model's
  `.pt.config.json` companion file): `theta`/`pi` now come from the head, not from a saved scalar.
- [ ] Consumers depend on an abstraction (`DistributionFamily`), not on concrete NB/ZINB.

### ⚠️ Negative
- [ ] This change *adds* the registry-first guard at ~11 existing sites now; the Open/Closed payoff is for
  future families, not this one.
- [ ] Two dispatch paths (registry vs existing) coexist until the Epic-B migration; each wiring site needs a
  K=1 byte-identical parity test.
- [ ] Inference gains a new randomness source that must be seeded and guarded.
- [ ] **A win is not guaranteed.** Whether `nb`/`zinb` actually beat the current foundation is an open
  empirical question — the M1 kill-gate (§5, story A-S11) may return a null result. The *subsystem* is
  worth having regardless (it unblocks cheap future families), but the *distributional-head hypothesis*
  may not pay off.

---

## 5. Validation

**Invariants (must remain bit-perfect).**
- Every existing `output_distribution` value produces a **byte-identical forecast** (K=1) before/after each
  wiring story, checked against an anchor captured in story A-S2.
- The `(N,S)` PredictionFrame/CRPS contract (`prediction_frame_assembler.py`) is preserved.
- **Self-zeroed occurrence is scored from the distribution, not the gate head:** for `nb`/`zinb`, the
  frozen-ruler gate metrics (AP/Brier) must use the family's own `P(Y>0) = (1−π)·(1−NB(0))` (exposed as
  `prob_positive`), because a self-zeroed family's zeros come from the distribution, not from the
  classification head — scoring the cls head would report an occurrence the forecast ignores.
  (Closes **C-201**.)

**Tests (ADR-005 Red/Beige/Green).**
- *Green:* register/lookup/`FAMILY_NAMES`==keys; per-cell `theta` varies across cells (std/mean > 0.02);
  NB/ZINB NLL finite +
  differentiable to per-cell params; sampler fills `(…, D×K)` and passes the `(N,S)` invariant; sampler
  deterministic under a fixed seed.
- *Red:* the M0 red tests (`tests/test_nb_dist_head.py`, relocating to `tests/distributions/`) fail until
  the subsystem exists.
- *Beige:* `n_head_samples > 1` on an existing head raises; oversize `D×K` is rejected.
- *Open/Closed proof:* a throwaway dummy family, registered, flows through head-sizing + loss + sampler
  with no consumer edits.

**Failure mode that would trigger reconsideration.** If per-cell `θ`/`π` collapse to ~constant, **`π`
degenerates** (→ 0 ⇒ plain NB, or → 1 ⇒ dead cells), or the `θ`/`π` fields are **seed-unstable** (all per
the pre-registration's extended F1 falsifier, `05_analysis_plan.md`); or the M1 step-1 evaluation shows no
calibration/sharpness gain over the foundation on the frozen lodestar ruler — then the *sampleable head*
hypothesis is falsified (the subsystem still stands, but NB/ZINB are not the win).

---

## 6. Implementation Notes

- **Location.** New package `views_hydranet/distributions/` (`base.py`, `nb_core.py`, `registry.py`,
  `negative_binomial.py`, `zero_inflated_negative_binomial.py`, `sampling.py`); wiring at the ~11 sites via
  `resolve_family`; config in `config_initializer.py`; tests under `tests/distributions/`; CICs
  (Component Interface Contracts) `docs/CICs/DistributionFamily.md` + `DistributionRegistry.md`.
- **References.** Epic **#167** (stories A-S1…A-S13, #168–#180); tracking **#182**; Epic-B draft **#181**.
  Related ADRs: **063/064** (head output activations — this ADR's per-param activation extends them),
  **059** (predictive-uncertainty representation), **065** (`body_mask` — the clean single-setting pattern
  copied here), **006** (CICs), **008/009** (fail-loud config). **Supersedes #97/#104** (the mean-emit
  hurdle-NB program). The earlier **ADR-066** (a cosmetic `output_distribution`→`body` rename) has been
**collapsed into this ADR** (both were same-day branch drafts); the rename itself is deferred to Epic B (#181).
- **Precedent & order.** The existing **quantile head** (`views_hydranet/utils/quantile_head.py` + the
  `_is_quantile` path in the architecture) is the multi-channel-per-target precedent to copy: it widens the
  reg heads by a per-target channel count while keeping `output_channels = 1` for the AR feedback — do the
  same for the per-cell distribution params. Implement in the **story order of tracking issue #182** — the
  `distributions/` core (A-S2) first, then the families, then the config + wiring stories (A-S5–A-S8).
- **Evidence trail.** `reports/2026-07-20_distributional_head_dossier/` (design, LOCKED pre-registration,
  experiment log); the v1 `/expert-code-review` — its amendments (explicit registry not a decorator; a
  single `DistributionFamily` ABC; the seeded sampler; bounded D×K; the centralized `resolve_family` seam;
  `NBCore` composition) are folded into §2–§3. The concrete risks surfaced by the reviews and the `/falsify`
  audits are registered as **C-197–C-201** (name-collision, transform-`expm1`, estimation robustness,
  `π/μ` identifiability, self-zeroed eval contract).

---

## Glossary

- **gate / body** — the two parts of each cell's forecast: probability of any violence (gate) × how many
  deaths (body).
- **Negative Binomial (NB)** — a probability distribution over whole-number counts (0,1,2,…); naturally
  puts weight on 0.
- **ZINB (zero-inflated NB)** — an NB plus an explicit extra spike of probability at exactly 0.
- **mu / theta / pi** — per-cell parameters the head emits: `mu` = mean, `theta` = dispersion (spread),
  `pi` = extra probability of a structural zero (ZINB only).
- **per-cell / homoscedastic** — per-cell = each map cell gets its own spread; homoscedastic = one shared
  spread for all cells (today's limitation).
- **D×K sampling** — S posterior samples = D MC-dropout passes (model uncertainty) × K draws from each
  cell's own distribution (outcome uncertainty).
- **strangler-fig** — a migration style: new code runs beside the old behind one switch; the old is
  replaced gradually, not rewritten at once.
- **registry / `resolve_family`** — the one table of body kinds and the one function that looks a kind up;
  the single place dispatch happens.
- **frozen lodestar ruler** — the fixed T=0 scorer (`reports/2026-07-17_lodestar_eval_dossier/`) every
  model is judged on.
