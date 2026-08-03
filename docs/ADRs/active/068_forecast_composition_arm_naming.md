# ADR-068: A naming convention for forecast-composition arms (`[th_]gated_<body>[core]`)

**Status:** Active
**Date:** 2026-07-24 (accepted 2026-07-24)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## Summary (read this first — self-contained)

The model's forecast for a map cell is built from a **body** (how many deaths, if any) and an **occurrence
rule** (how the cell's zeros are produced). Once we have per-cell distributional bodies (ADR-067: **NB**,
**ZINB**), the *same trained heads can be composed into a forecast several different ways* — the body can
self-zero (ZINB's structural π), or be gated by the classification head **softly** (`Bernoulli(gate)×body`)
or via a **hard threshold** (`gate ≥ τ`); and a **ZINB-trained** body can be reused with its π stripped and
an external gate instead. Each of these is an **arm** we score head-to-head on the frozen ruler.

We were naming these ad-hoc (`gated_NB`, `masked_NB`, …). `masked_NB` is ambiguous (masks *what*, *how*?),
and the ad-hoc scheme neither scales to future bodies (gamma, lognormal) nor prevents a real correctness
trap (re-gating an already-self-zeroed body **double-counts the zeros**).

This ADR fixes a **systematic naming convention** for arms: **`[th_]gated_<bodymodel>[core]`** for a
*composed* forecast, and the **bare distribution name** (`ZINB`, `ZIgamma`) for a *self-zeroed standalone*.
The one load-bearing word is **`core`**, defined precisely as "*the positive body of a zero-inflated
distribution with its structural π removed*." The convention is registered as canonical vocabulary in
`reports/GLOSSARY.md` §2c; this ADR records **why** it is shaped this way.

*(A glossary of the technical terms is at the end.)*

---

## 1. Context

**Setting.** HydraNet forecasts monthly conflict deaths per grid cell — ~99.7% zeros with a heavy tail.
ADR-067 gave us per-cell **bodies** we can sample from (NB, ZINB). Separately we have a sharp
classification **gate** `P(Y>0)`. A single trained (body + gate) can be turned into a scored forecast by
more than one **composition rule**, and we intend to compare these rules empirically:

- **ZINB** — the body self-zeroes via its own structural π; forecast `E[y]=(1−π)μ`, **no external gate**.
- **gated_NB** — the NB body, soft-gated per draw by `Bernoulli(gate)`.
- **th_gated_NB** — the NB body, hard-gated: full body where `gate ≥ τ`, zero where `gate < τ` (τ a fixed,
  a-priori probability).
- **gated_ZINBcore** — the **ZINB-trained** body `NB(μ,θ)` with **π dropped**, soft-gated by the cls head.
  Motivated because ZINB's π absorbs the zeros *during training*, freeing μ to be the *conditional*
  magnitude (un-timid) rather than plain NB's zero-diluted μ.

**Problem.** The naming was accreting ad-hoc:

- **Ambiguity.** `masked_NB` does not say what is masked or by what — a hard gate threshold is not obviously
  a "mask".
- **A hidden correctness trap.** "ZINB × gate" reads as a reasonable arm but is *wrong*: ZINB's output is
  already self-zeroed by π, so multiplying by the gate zeros twice — `(1−π)μ × gate` **double-counts**. A
  name that does not distinguish "the self-zeroed distribution" from "its π-stripped core" invites this bug.
- **A misleading asymmetry.** Left ad-hoc, `gated_NB` vs `gated_ZINBcore` looks like a `full`-vs-`core` body
  distinction — but both gate a *bare* `NB(μ,θ)`; the only difference is which training produced `(μ,θ)`.
- **No room for growth.** Adding gamma / lognormal bodies (and possibly ZI-continuous) would each invite a
  fresh ad-hoc name.

The repo already treats **vocabulary as load-bearing** (`reports/GLOSSARY.md` is a locked-vocabulary
document; ADR-003 gives declarations authority over inference). A composition taxonomy this error-prone
deserves a locked convention, not per-run coinage.

---

## 2. Decision

Adopt a **two-knob** naming convention for forecast-composition arms, registered in `GLOSSARY.md` §2c.

**Knob 1 — body source:** which model's parameters supply the body (`NB`, `ZINB`, later `gamma`,
`lognormal`, `ZIgamma`, …). Same *form* of body; different *training objective* ⇒ different `μ`.

**Knob 2 — occurrence rule:** how the zeros arise —
- **self** — a self-zeroed distribution's own structural π (no external gate);
- **soft gate** — prefix **`gated_`** — per draw `Bernoulli(gate) × body`;
- **threshold gate** — prefix **`th_gated_`** — full body where `gate ≥ τ`, zeroed where `gate < τ`, for a
  **fixed a-priori** τ.

**The pattern:**
- a *composed* forecast is **`[th_]gated_<bodymodel>[core]`**;
- a *self-zeroed standalone* is the **bare distribution name** (`ZINB`, `ZIgamma`).

**The `core` suffix (locked definition):** *the positive body of a zero-inflated distribution with its
structural π removed*, so it can be composed with an external gate instead of self-zeroing.
- `core` appears **only** on a body derived from a ZI model (e.g. `ZINBcore`). A body with **no** structural
  π (NB, gamma, lognormal) **never** carries it — there is nothing to strip.
- The **presence** of `core` signals "a π was stripped here"; its **absence** means "no ZI model was
  involved" — **not** "the full/non-core body".
- **Guardrail:** a `core` must **never** be re-multiplied by its own π. `(1−π)μ × gate` double-counts; once
  `core`, the gate is the *only* zeroing mechanism.

**The current arms:**

| name | exact forecast |
|---|---|
| `ZINB` | self-zeroed standalone: `E[y]=(1−π)μ`, π-masked sampling, no external gate |
| `gated_NB` | `Bernoulli(gate) × NB(μ,θ)`, `(μ,θ)` from the **nb** model |
| `th_gated_NB` | full `NB(μ,θ)` where `gate ≥ τ`, else 0; `(μ,θ)` from the **nb** model *(renames `masked_NB`)* |
| `gated_ZINBcore` | `Bernoulli(gate) × NB(μ,θ)`, `(μ,θ)` from the **zinb** model, **π dropped** |

**Extension (no new convention):** continuous bodies have no zero mass, so they *must* be gated:

| body model | self-zeroed standalone | soft-gated | threshold-gated |
|---|---|---|---|
| NB | — | `gated_NB` | `th_gated_NB` |
| ZINB | `ZINB` | `gated_ZINBcore` | `th_gated_ZINBcore` |
| gamma | — | `gated_gamma` | `th_gated_gamma` |
| lognormal | — | `gated_lognormal` | `th_gated_lognormal` |
| ZI-gamma *(if ever)* | `ZIgamma` | `gated_ZIgammacore` | `th_gated_ZIgammacore` |

**Retired alias:** `masked_NB` → **`th_gated_NB`**.

---

## 3. Rationale & Integrity Impact

**Why `core` is defined as "π-stripped ZI body", not "the bare body" generically.** Two candidate schemes
were considered and rejected:

- **Uniform `core` everywhere** (`gated_NBcore` / `gated_ZINBcore`): visually symmetric, but *false* — it
  implies the two arms are the same kind of thing differing only cosmetically, hiding that **only one
  involved a π-stripping**. It also puts a redundant `core` on NB/gamma/lognormal, which have no π. Uniform
  tagging launders away the very distinction the reader needs.
- **Fully ad-hoc** (`masked_NB`, `ZINB×gate`): ambiguous and trap-prone as in §1.

Defining `core` as a *signal that a π was removed* makes the asymmetry `gated_NB` vs `gated_ZINBcore`
**informative**: the tag's presence is exactly the flag for the one non-obvious, double-count-prone
operation. Common cases stay short; the rare, dangerous case is marked. This is the honest encoding of the
actual structure.

**Integrity impact (the double-count trap).** The convention makes the correctness rule legible in the
name: an arm named `…core` has had its π removed, so it *needs* a gate and must *not* be re-π'd; an arm
named `ZINB` is already self-zeroed and must *not* be gated. A reviewer can catch a mis-composition from the
identifier alone. This directly serves ADR-004 (evaluation integrity) and the locked-glossary discipline.

**Why threshold τ is a-priori.** Naming `th_gated_*` foregrounds that a threshold exists; the governing
pre-registration (`reports/2026-07-20_distributional_head_dossier/05_analysis_plan.md`) fixes τ to
uncontroversial a-priori values (0.5, per-target base rate) so τ is never fit on the frozen-ruler months
(Goodhart). The name and the pre-registration together keep the knob honest.

---

## 4. Consequences

### ✅ Positive
- **One convention for all arms**, present and future (gamma/lognormal/ZI-continuous) — no per-run coinage.
- **The correctness rule is in the name:** `core` ⇒ π removed ⇒ gate is the only zeroing; a bare ZI name ⇒
  already self-zeroed ⇒ do not gate. Kills the `(1−π)μ × gate` double-count at the identifier level.
- **The `gated_NB`/`gated_ZINBcore` asymmetry becomes informative**, not a source of confusion.
- **Descriptive over cryptic:** `th_gated_NB` states the mechanism; `masked_NB` did not.
- Aligns with the existing locked-glossary discipline (ADR-003), so it is enforceable ("drift").

### ⚠️ Negative
- **`core` must be learned.** Its meaning is precise but non-obvious; a reader who skips the definition can
  still misread the asymmetry. Mitigation: the definition is stated verbosely in GLOSSARY §2c and here.
- **Mild verbosity** (`gated_ZINBcore` vs a terse codename). Accepted deliberately — the codename era was
  retired (GLOSSARY §6) precisely because opaque names hid choices.
- **A rename cost:** `masked_NB` references (scratch drivers, plot labels, any scoring templates) must be
  updated. Small and mechanical; no shipped artifact depends on the string.

---

## 5. Validation

- **Canonical source:** `reports/GLOSSARY.md` §2c holds the locked names, the `core` definition, the
  double-count guardrail, and the extension table. This ADR records the rationale; the glossary is the
  enforced vocabulary ("drift").
- **Consistency check:** every arm name must parse as either `[th_]gated_<bodymodel>[core]` or a bare
  self-zeroed distribution name; `core` appears iff the body came from a ZI model. Any identifier that
  fails this (e.g. a resurrected `masked_NB`, or a `gated_ZINB` without `core`) is a drift to correct.
- **Cross-references:** ADR-067 (the family subsystem that produces the bodies these arms compose);
  `05_analysis_plan.md` (pre-registers the arms + the a-priori τ); register C-146 (ZINB-vs-hurdle
  commitment), C-201/C-211 (self-zeroed vs gated scoring), C-212 (the ZINB NaN, why the ZINB body is
  reliable to reuse).

## 6. Implementation Notes

- **Scope.** This is a *vocabulary* decision. Arms are a mix of (a) trained-family config values
  (`output_distribution=nb|zinb`, ADR-067) and (b) **score-time compositions** of already-trained heads
  (`gated_NB`, `th_gated_NB`, `gated_ZINBcore`). The naming spans both; it does **not** mandate that each
  arm be a distinct `output_distribution`. `gated_NB`/`th_gated_NB`/`gated_ZINBcore` are scoring compositions
  over stored cubes/artifacts, not new trained models.
- **Where names surface:** scoring drivers and their output columns, plot/biopsy labels, dossier prose,
  register entries. Adopt the convention in each as they are touched; retire `masked_NB` on sight.
- **`gated_ZINBcore` feasibility note (not a naming issue):** it needs the *bare* ZINB core, but a ZINB
  run's stored samples are already π-masked (`ZINBFamily.sample`), so it requires a cheap **re-inference**
  on the trained ZINB artifact emitting `NB(μ,θ)` without the π mask — inference only, no retrain. Tracked
  with the arm, gated on the 3-seed ZINB result confirming the magnitude-vs-locality premise.

---

## Glossary

- **arm** — a specific rule for turning a trained body + gate into the one scored forecast (a compose/score
  choice, layered on the ADR-067 training choice).
- **body** — the part of the model predicting *how many* deaths (per-cell `μ`, and for count families a
  per-cell spread `θ`).
- **gate** — the classification head's `P(Y>0)` per cell (whether *any* violence occurs).
- **NB / ZINB** — negative-binomial body / NB with a structural zero-inflation spike (parameter **π**); see
  ADR-067.
- **self-zeroed** — a body that produces its own zeros (ZINB's π) and needs no external gate.
- **core** — the positive body of a zero-inflated distribution with its structural π **removed**, so it can
  be composed with an external gate; appears only on ZI-derived bodies (e.g. `ZINBcore`).
- **soft gate (`gated_`)** — occurrence via `Bernoulli(gate) × body` per draw.
- **threshold gate (`th_gated_`)** — occurrence via a hard cut: full body where `gate ≥ τ`, else 0.
- **τ** — the fixed, a-priori threshold probability of a `th_gated_` arm.
- **double-count** — the error of zeroing twice: `(1−π)μ × gate`, i.e. gating an already-self-zeroed body.
