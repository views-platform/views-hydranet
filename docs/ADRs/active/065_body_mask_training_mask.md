# ADR-065: `body_mask` — a first-class, validated point-body training-mask setting

**Status:** Active (promoted from Proposed 2026-08-14 — the body_supervision resolver is shipped; see `docs/CICs/BodySupervisionResolver.md`)
**Date:** 2026-07-18
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## 1. Context
**Why are we doing this now?**
- **Problem:** *Which cells the point body (MSE/MAE) trains on* is currently expressed as a **combination of two
  implicit knobs**, not a single legible setting:
  - `hurdle_threshold: float | None` (`config_initializer.py:104`) — a real validated field: `None` = all cells,
    `0` = mask on.
  - `hurdle_mask_mode` — read **raw and UN-validated** via `config.get("hurdle_mask_mode", "per_step")`
    (`training_engine.py:549`). A typo (`active-window` vs `active_window`) **silently** degrades to `per_step`
    (risk-register **C-194**).
  The three real states (all-cell / per-step-positive / active-window) are a 2×knob cross-product. A config
  reader **cannot see the active mask from any single field** — the exact opposite of a boundary contract.
- **Assumptions (now false):** that a point-body mask is a niche internal detail. It is now a **first-class
  modelling lever** (the all-cell + MSE foundation vs. positives-only bodies is the live experimental axis), and
  it is planned to **grow** (hurdle variants, a distributional head). A patchy two-knob scaffold does not survive
  that growth.
- **Silent failure modes it carries:** the mask **only applies to point bodies** (`not use_latent`); under a
  latent/likelihood loss (NB/lognormal/tobit) it is **silently ignored** with only a warn-once (**C-193**,
  arguably an ADR-008 violation). And the mask's "what is an event" (`target > 0`) **duplicates** the binary-target
  derivation's `threshold = 0` (**C-195**) — two authorities over one fact.
- **Urgency:** the next experiments toggle this setting repeatedly. Before that toggling hardens into more ad-hoc
  config combinations, the setting must become **one validated front door** with a fail-loud boundary contract
  (ADR-009), so nothing silently degrades and `none` is provably byte-identical to today's foundation.

---

## 2. Decision
**The new Law of the Land.**
- **Statement:** "We will introduce **one validated config field `body_mask ∈ {none, pos_cells, pos_timelines}`**
  as the **sole front door** for how the point body is masked during training. A **pure resolver function** maps
  the keyword to the concrete cell-set mechanism; the two legacy knobs (`hurdle_threshold`, `hurdle_mask_mode`)
  are **retired** from the public config surface."
- **Taxonomy (precise cell-set semantics):**
  | `body_mask` | Cell set the point-body loss is computed on | Equivalent legacy state |
  |---|---|---|
  | `none` | **All** cells (dense). Default. Byte-identical to today's foundation. | `hurdle_threshold=None` |
  | `pos_cells` | **Per-step positive:** cells where `y > thr` at *that* timestep. | `threshold=0`, `mode=per_step` |
  | `pos_timelines` | Cells **active anywhere in the curriculum window** (the existing `_active_window_mask`), **including their decay-zero steps**. | `threshold=0`, `mode=active_window` |
  where `thr` is the **single** event threshold defined by the binary-target **derivation** config (ADR-046), not
  a literal baked into the mask code.
- **In-Scope:** the point-body (`not use_latent`) training mask; its config field + validators; the pure resolver;
  the single-authority threshold sourcing; the fail-loud coupling checks; retirement of the two legacy knobs.
- **Out-of-Scope:** the **latent/likelihood** masking path (NB/lognormal/tobit own their masking via truncation,
  ADR-054/055/059) — unchanged; the **gate/classification** mask; the mask **computation** itself
  (`_active_window_mask` stays); any change to `none`'s numerics.

### 2.1 Recorded decisions on the four live disagreements
- **D-1 — Enum vs. registry → ENUM NOW, registry deferred.** `body_mask` is a validated `Literal`/enum resolved by
  a **pure function**, *not* an ADR-049-style registry. Rationale (ADR-003 Boring / YAGNI): three closed, known
  values with no third-party extension point do not justify a registry's indirection today. The resolver is written
  so the **registry seam stays open** — promote to a registry only when a **4th** mask (or a plugin need) actually
  arrives, not speculatively.
- **D-2 — Delete vs. deprecate the legacy knobs → RETIRE (clean break), gated by a characterization net.**
  `hurdle_threshold` + `hurdle_mask_mode` are removed as public fields; `body_mask` is the sole authority. Keeping
  both alive would preserve exactly the **dual-authority tangle** this ADR removes (ADR-003). A legacy config that
  still sets the old keys **fails loud** at validation with a message pointing to `body_mask` (ADR-009) — it does
  **not** silently shim. The break is only made **after** the S2 characterization net proves `body_mask=none` is
  byte-identical to the current foundation. *(Alternative — deprecate-in-place with a translating shim — rejected:
  it leaves two doors open and re-introduces the un-validated `mode` read.)*
- **D-3 — Threshold ownership → the DERIVATION config is the single authority.** "What counts as an event" is
  defined **once**, by the binary-target derivation (`threshold = 0`, ADR-046). `pos_cells` sources `thr` from that
  derivation config; it must **not** hard-code `0` or introduce a second `body_mask`-local threshold literal
  (closes C-195).
- **D-4 — Raise vs. warn on the latent coupling → RAISE (hard error).** Requesting `pos_cells`/`pos_timelines`
  together with a **latent/likelihood** loss is a **contract violation**, not a soft condition: the point-body mask
  is a **no-op** there and the request is meaningless. The config validator raises a `ValueError` (ADR-008/009),
  replacing the current warn-once (closes C-193). `body_mask=none` + latent stays legal (it *is* the all-cell case).

### 2.2 Migration strategy (named here; executed by S5)
1. **S2** captures a characterization net over the *current* two-knob behaviour (the before-anchor).
2. **S3/S4** add `body_mask` + validators + resolver, wired so `none` reproduces the foundation.
3. **S5** deletes `hurdle_threshold` + `hurdle_mask_mode` from the public surface, migrates the ~35 test references
   and any on-disk configs to `body_mask`, and adds a fail-loud validator rejecting the legacy keys.
4. **S6** proves `none` byte-identical + contract-tests each mode end-to-end; **S7** greens the full suite + smoke.

---

## 3. Rationale & Integrity Impact
- **Logic (Correctness > Convenience):** a boundary contract must be **legible from one field** and **validated at
  the boundary** (ADR-009). Collapsing a 2-knob cross-product (one of whose knobs is read un-validated) into a
  single validated enum removes both the illegibility and the silent-typo degradation. A **pure resolver** keeps
  the keyword→mechanism mapping in one testable place instead of scattered `if threshold/if mode` branches.
- **Fortress State (Reproducibility):** `none` is contractually **byte-identical** to the current foundation
  (guarded by the S2 net + the S6 byte-identical proof, C-196), so introducing the field regresses nothing. Each
  mode's cell set is defined unambiguously and reproducibly — the semantics do not drift under the ongoing
  pandas→views-frames migration because the mask operates on **training tensors**, not the output/prediction path
  (no pandas/polars introduced; ADR-047 direction respected).
- **Fail-Loud:** the two worst behaviours today are **silent** — the `mode` typo (C-194) and the latent no-op
  (C-193). This ADR converts both into **hard validation errors** (ADR-008), which is the whole point of promoting
  the setting to a first-class contract rather than leaving it to per-config discipline.

---

## 4. Consequences

### ✅ Positive
- [ ] One legible, validated front door for the point-body mask (ADR-009 boundary contract).
- [ ] Silent `mode`-typo degradation (C-194) and silent latent no-op (C-193) become fail-loud (ADR-008).
- [ ] Single authority for "what is an event" (C-195 closed) via the derivation config (ADR-046).
- [ ] `none` byte-identical to the foundation ⇒ zero regression; the registry seam stays open for real growth.

### ⚠️ Negative
- [ ] Breaking change: `hurdle_threshold`/`hurdle_mask_mode` are removed → ~35 test references + any on-disk
      configs must be migrated to `body_mask` (S5). Mitigated by the fail-loud rejection message + the S2 net.
- [ ] One new validated field + resolver + coupling checks (small, principled boilerplate; the ADR-003 cost of a
      real contract over an implicit cross-product).

---

## 5. Validation
- **Invariants:** `body_mask=none` ⇒ point-body loss computed over **every** cell, **byte-identical** to the
  pre-change foundation. `pos_cells`/`pos_timelines` + latent loss ⇒ **`ValueError`** (never a silent no-op).
  Unknown `body_mask` value ⇒ `ValueError` at validation. `thr` always sourced from the derivation config.
- **Tests (ADR-005 Green/Beige/Red):**
  - *Green* — config→behaviour **contract** tests: `body_mask=X ⇒ body loss computed on cell-set Y` end-to-end,
    one per mode (S6); `none` byte-identical proof (S6, C-196).
  - *Beige* — validator rejects unknown values, rejects `pos_*`+latent, rejects the retired legacy keys (S3/S5).
  - *Red* — the S2 characterization net pins the current two-knob behaviour so the refactor cannot silently alter
    any mode.
- **Failure Mode (reopen this ADR if):** a `body_mask` mode silently trains on a different cell set than its
  contract states; or `none` ceases to be byte-identical; or a 4th mask/plugin need arrives (⇒ revisit D-1, promote
  to an ADR-049 registry).

---

## 6. Implementation Notes
- **Location (enforcement):**
  - *Config boundary* — `views_hydranet/utils/config_initializer.py`: new validated `body_mask` field (Literal enum)
    + a model-validator raising on `pos_*`+latent and on the retired legacy keys.
  - *Resolver* — a pure `resolve_body_mask(body_mask, ...) → mask spec` function (its own module/tested unit),
    consumed by `training_engine.py` in place of the raw `hurdle_threshold`/`config.get("hurdle_mask_mode")` reads
    (`training_engine.py:343-372, 549`). `_active_window_mask` (`:168`) is reused unchanged.
  - *Threshold authority* — `thr` read from the binary-target derivation config (ADR-046), not a literal.
- **CIC (ADR-006):** the resolver + the training-loop masking step warrant a small Intent Contract (single input
  contract: keyword+loss-type+derivation-threshold → deterministic cell set) — added in S6.
- **References:** risk register **C-193/C-194/C-195/C-196**; disagreements **D-1…D-4** (8-expert review,
  2026-07-18); Epic **#158**, stories **#159–#165**; `reports/GLOSSARY.md` (locked vocabulary).
  Governed by / cross-references: **ADR-003** (Boring Architecture), **ADR-005** (Testing), **ADR-006** (Intent
  Contracts), **ADR-008** (Error Propagation), **ADR-009** (Boundary Contracts & Config Validation), **ADR-046**
  (Transformations vs Derivations), **ADR-047** (Pandas-Free Output direction), **ADR-049** (Sampling Strategy
  Registry — the deferred seam), **ADR-054/055/059** (latent/likelihood masking, out of scope here).

---

## Amendment 2026-07-28 — generalize the keyword to a graded, asymmetric **supervision window** (`body_supervision` + `onset_lead` / `cessation_lag`)

**Status of amendment:** Proposed · **Deciders:** Simon Polichinel von der Maase (chair) · **Source:** the
magnitude effort (dossier `2026-07-20_distributional_head_dossier/08`) + the magnitude-fix expert-method-review
(2026-07-28, C-224…C-227, D-13).

### A1. Why amend now
The three-keyword taxonomy (`none` / `pos_cells` / `pos_timelines`) proved too coarse for the magnitude lever:
- **`pos_cells` over-cooks and is boundary-blind.** `body_mask` masks the **loss, not the input** (`training_engine`
  runs `model(t0_input)` on the full volume; the mask is a `weight=` on `family.nll`). So the ConvLSTM still *sees*
  onset/cessation — but the body **head gets zero gradient off the positive support**, so (i) it drifts high on
  nearby zeros (the measured true-zero over-cook: `E[y]>100` on ~46 ns / 67 os true-zero cells, dossier 08) and
  (ii) it is never taught to **ramp down toward zero** approaching cessation. These are one root: the body is
  unsupervised exactly where conflict begins and ends.
- **`pos_timelines` is timid.** Supervising *every* zero in an ever-active cell's history — including long dead
  stretches far from any episode — drags μ back toward 0 (measured: size_ratio stays 0).
- The right region is **neither**: supervise the body on conflict episodes **and their temporal boundaries**
  (the onset run-up, the cessation decay), delegating only the **deep structural background** to the gate — the
  chair's stated division of labour ("the gate subdues the obvious structural zeros; the body gets the conflict
  right and may be wrong elsewhere").

### A2. Decision (the amended Law)
Retire `body_mask ∈ {none, pos_cells, pos_timelines}`. Replace with **one graded, asymmetric supervision window**:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `body_supervision` | `Literal['all','active']` | `'all'` | `all` = every cell (old `none`); `active` = a temporal window around conflict activity |
| `onset_lead` | `int ≥ 0` | `0` | months **before** onset the body is supervised (the run-up); `active` only |
| `cessation_lag` | `int ≥ 0` | `0` | months **after** cessation the body is supervised (the decay); `active` only |

**Semantics (asymmetric temporal dilation of the per-step-positive mask).** For a cell, a timestep `t` is
supervised iff there exists an active month `t'` (`y_{t'} > thr`) with

  **`t − cessation_lag ≤ t' ≤ t + onset_lead`**

i.e. dilate the active months `onset_lead` steps *backward in supervised-time* (run-up) and `cessation_lag`
steps *forward* (decay), per cell, in the T dimension. (`onset_lead` reaches to *future* active months → it
supervises the run-up; `cessation_lag` reaches to *past* active months → it supervises the decay.) Values
`≥ T−1` saturate to the whole window.

### A3. Taxonomy mapping (byte-identical endpoints — the retirement gate, S2)
| old `body_mask` | new (`body_supervision`, `onset_lead`, `cessation_lag`) | parity |
|---|---|---|
| `none` | `('all', —, —)` | all-True; unchanged foundation |
| `pos_cells` | `('active', 0, 0)` | `t'∈[t,t]` ⇒ `y>thr` at `t` — **byte-identical** to old pos_cells |
| `pos_timelines` | `('active', ≥T−1, ≥T−1)` | window covers all T ⇒ active-anywhere broadcast — **byte-identical** to old pos_timelines |

The clean-break retirement of the three keywords is gated on a **byte-identical parity test** at both endpoints
(same discipline as the S2 characterization net that gated the original ADR-065 retirement of the legacy knobs).

### A4. Rationale
- **Turns Philosophy A vs B into an empirical knob.** `(0,0)` is the pure-hurdle body (occurrence is wholly the
  gate's job — Philosophy B); larger radii give the body ownership of the boundary dynamics (Philosophy A). We do
  not argue A vs B; we sweep `(onset_lead, cessation_lag)` and let the frozen ruler pick. `pos_cells` and
  `pos_timelines` become the two endpoints of that dial, not competing keywords.
- **Hypothesis (falsifiable):** an interior radius lifts the bulk-magnitude downward bias (µ un-collapsed, since
  the deep background is excluded) *without* the over-cook (the near-activity zeros teach ramp-down and constrain
  the drift) — beating both endpoints. A best radius of `(0,0)` is a real, informative null.
- **Asymmetry is deliberate** (chair): onset (from zero) and cessation (to zero) need not be the body's job to the
  same degree — the pre-onset quiet is arguably the gate's, the decay is more clearly the body's. `onset_lead=0,
  cessation_lag>0` is a first-class, expressible setting.

### A5. Scope guards (do not let this masquerade)
- **Point-body only.** Still `not use_latent`; NB/ZINB are `needs_latent=False`, so `active` applies to them
  (the C-193 latent no-op RAISE is retained for a latent loss + `active`). Latent-truncation losses unchanged.
- **NOT a tail fix.** This lever changes *where the body is supervised*, not whether the family can *reach* the
  ξ≈0.8 surge. The `tail_scorecard` will still show `q90≈0` on the top magnitude bin for every radius — expected,
  and the evidence that this axis and the heavy-tail axis (C-149/C-224) are orthogonal. Success = bulk-magnitude
  downward-bias only.
- **`onset_lead` uses future-active truth to *select training cells*** — a training-time label/weight
  construction (like the curriculum's window selection), **not** an inference-time feature; the model is never
  given future info at emit. Noted so no reader mistakes it for leakage.
- **Retirement is fail-loud.** A config still setting `body_mask=` raises at validation with the migration hint
  (`none→all`; `pos_cells→active,0,0`; `pos_timelines→active,W,W`) — never a silent shim (ADR-009).

### A6. Enforcement locations (delta from the base ADR)
- `config_initializer.py`: replace the `body_mask` Literal with `body_supervision` Literal + `onset_lead` /
  `cessation_lag` int fields (`≥0` validators); the latent-no-op RAISE keys on `body_supervision=='active'`; add
  the retired-`body_mask` fail-loud.
- `body_mask.py`: `resolve_body_supervision(onset_lead, cessation_lag, event_threshold) → window→BoolTensor`
  (asymmetric temporal dilation; `_active_window_mask` reused for the saturated endpoint). Old
  `resolve_body_mask` retired.
- `training_engine.py`: `apply_body_mask = body_supervision=='active' and not use_latent`; resolver call updated.
- CIC (`DistributionFamily` / training docs) + `reports/GLOSSARY.md` (locked term; retired aliases).

**References (amendment):** risk register **C-224…C-227**, **D-13**; dossier
`2026-07-20_distributional_head_dossier/08`; the `tail_scorecard` diagnostic; supersedes the `pos_cells` /
`pos_timelines` rows of §2's taxonomy (kept above for provenance).
