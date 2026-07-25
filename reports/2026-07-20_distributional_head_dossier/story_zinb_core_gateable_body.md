# Story — `zinb-core` as a gateable body (emit-time, ensemble-diversity arm)

**Lineage:** follow-on to Epic #183 (ADR-069 forecast-composition axis). **Status:** SCOPED, not started.
**Date scoped:** 2026-07-25. **Kind:** eval-only re-emission (NO retrain). **Precedes:** the bloom epic.

---

## 1. Why (the one-paragraph case)

Every current composed arm uses a **timid body**: plain all-cell `nb` has size-ratio ≈ 0.02 (zero-dilution
shrinks μ). The ZINB **core** — the same NB core with the structural π **stripped** — fires *large*
(size-ratio ≈ 1.06), because during training π absorbed the zeros and licensed the core to explain only
positives. Gating that large core externally (soft or hard) gives a genuinely **different body magnitude**
under a sharp gate. Measured at emit-time (seed 44, the 2026-07-25 re-test): **gated_ZINBcore 0.152 sb,
th_gated_ZINBcore (τ=0.5) 0.148 sb** — viable, marginally the weakest cluster on standalone crps-all, but
that is *expected and fine*: **its only justification is ensemble diversity, not standalone score.** This
story wires it as a first-class, honestly-scored arm so the ensemble decision is made on 3 seeds and an
actual ensemble-payoff test, not a single-seed number.

## 2. The two-axis picture this closes (body × composition)

| body ↓ / comp → | self_zeroed | soft_gate | threshold_gate |
|---|---|---|---|
| **nb** (timid) | *foundation* | gated_NB ✓ 3-seed | th_gated_NB ✓ 3-seed |
| **zinb** (self-zeroing) | ZINB ✓ 3-seed | ✗ double-counts π+gate | ✗ double-counts |
| **zinb-core** (large, π-stripped) | = the original kill (ungated large body) | **gated_ZINBcore** ← this story | **th_gated_ZINBcore** ← this story |

The story adds exactly the **bottom row's two gated cells**. `zinb-core + self_zeroed` stays forbidden (it
IS the corrupted-knowledge kill: a large ungated body on 99.2% true-zero cells → crps-none 0.87).

## 3. What already exists (verified 2026-07-25 — no hand-waving)

- **`sample_core`** — present on `ZINBFamily` (`zero_inflated_negative_binomial.py:98`) and the ABC
  (`base.py:73`). S5 (`62d19ae`) retired only the *wiring* (`emit_family_core` config flag +
  `to_cube_samples(core=…)` param), NOT the method. The capability to draw the π-stripped core is intact.
- **The composer** (`compose_samples` / `compose_mean`) already applies soft_gate/threshold_gate to any
  body cube — proven on the zinb-core cube in the re-test. No composer change needed.
- **The validator** (`config_initializer.py:789` `validate_forecast_composition`) keys off
  `SELF_ZEROED_FAMILIES = frozenset({"zinb"})` (`registry.py:47`). Rule (1): a self-zeroed family must NOT
  be gated. Rule (2): a non-self-zeroed family MUST be gated. **This is the single lever**: zinb-core is
  *not* self-zeroed, so it must land outside `SELF_ZEROED_FAMILIES` and it will be *required* to declare a
  gate — exactly right.
- **Artifacts:** ≥6 `zinb` calibration artifacts on disk (violet_visitor/artifacts); the s44
  `calibration_model_20260724_092826.pt` was re-emitted core-mode in the re-test. **Task 0 pins the exact
  S8 3-seed triplet** from the S8 experiment-log entry — these are the re-emission inputs. **No retraining.**

## 4. The one open design decision (name it, don't bury it)

How does a **trained `zinb` artifact** get emitted as **`zinb-core`** at eval time, cleanly (the "completely
correct" bar from th_gated_NB — real config path + validator + tests + CIC, not a temp flag)?

- **Option A — new emit-only registered family `zinb_core`.** `.sample = core`, `.mean = μ` (ungated core
  mean, *not* (1−π)μ), `n_params=3`, NLL identical to zinb; NOT in `SELF_ZEROED_FAMILIES`. The composition
  axis then gates it with **zero special-casing** (validator rules already do the right thing). Needs an
  **emit-time "load a zinb artifact as zinb_core" override** (the artifact's metadata says
  `output_distribution="zinb"`), since we re-emit existing artifacts rather than retrain.
- **Option B — emit-mode on zinb.** One family `zinb`, add `zinb_emit ∈ {self_zeroed, core}` (default =
  today's ZINB). `core` ⇒ `.sample`→core AND zinb is treated as non-self-zeroed for the validator so gate
  compositions become legal. Simpler artifact story (same family), but adds a stateful flag + a validator
  branch — the special-case the composition axis was built to avoid.

**Leaning: Option A** (keeps the validator clean, fits the registry, the composition axis needs zero new
branches) — but this is a real decision for a short `expert-method-review` or an ADR-069 amendment, NOT
settled here. The re-emission-override detail is the crux either way.

## 5. Pre-registered decision rules (commit before running — negatives first-class)

- **G1 — faithfulness guardrail (must pass):** emitting a zinb artifact in `self_zeroed` mode must
  reproduce that seed's banked ZINB crps-all (the re-test already showed this to 4 dp on s44). If it
  drifts, the re-emission is wrong — STOP.
- **F1 — standalone floor (soft, expected to pass):** 3-seed emit-time crps-all(sb) of the better
  zinb-core composition ≤ **0.16** (s44 gave 0.148–0.152). If it blows past this, the seed-44 number was a
  fluke → drop.
- **F2 — the real kill (ensemble payoff):** adding zinb-core to the {gated_NB, ZINB} ensemble must
  **improve** ensemble crps-all on **≥2 of 3 targets** vs the 2-arm ensemble. **If diversity does not pay,
  DROP zinb-core as redundant** — standalone-weakest is only acceptable if it diversifies. This is the
  pre-committed negative.
- **F3 — diversity mechanism (confirmatory):** zinb-core's per-cell forecast must be materially different
  from gated_NB's (size-ratio gap ≫ 0, and forecast-error correlation < ~0.9) — proving it's a different
  *body*, not a relabelled gate. If it's not actually diverse, F2 will fail and this explains why.

## 6. Tasks (eval-only; small)

0. **Pin inputs** — from the S8 log, identify the exact 3 zinb seed artifacts (paths + seeds); confirm on
   disk + sha256. (No retrain.)
1. **Design lock** — decide Option A vs B (short method-review / ADR-069 amendment); write it down.
2. **Wire (TDD)** — implement the chosen path: registry/validator change so `zinb-core` is a legal,
   *gate-required*, non-self-zeroed body; restore the emit-time core route (clean, not the temp
   `emit_family_core`); the re-emission override. Red tests first (validator matrix rows, parity of the
   self_zeroed sanity path, core-vs-self_zeroed cube difference).
3. **Re-emit + score** — 3 zinb artifacts × {soft_gate, threshold_gate τ=0.5} on the frozen lodestar
   ruler; bank crps-all/events/none + size-ratio per seed/target. Include the self_zeroed G1 sanity row.
4. **Ensemble test (F2/F3)** — combine zinb-core with {gated_NB, ZINB} 3-seed sets; score the ensemble
   crps-all vs the 2-arm ensemble; measure the diversity metric.
5. **Log** — dossier `07_experiment_log` entry linked to §5 falsifiers (which fired); CIC/glossary/register
   updates if the family surface changed; ship with the ritual.

## 7. Scope boundaries (what this story is NOT)

- **NOT training** — pure re-emission of existing zinb artifacts. If any step wants a retrain, STOP and
  re-scope.
- **T=0 calibration only** — the bloom (T>0) is the *next* epic, deliberately after this.
- **No π-ridge / no gate-calibration changes** — the gate cube and the zinb π are frozen as trained.
- **Stealth protocol** — the violet_visitor floor config is never committed/pushed (md5
  `6c28bdb1390fc413d43b2d74d87251f8`); every driver trap-restores it.

## 8. Cost

~S8-scale: 3 artifacts × 2 compositions × (re-emit + score) + one ensemble-combine pass. GPU-eval-minutes,
no training. Short enough that ask-before-long-batches is satisfied by this scoping note.

## 9. Recommendation

Real candidate, honestly framed: **the decision hinges on F2 (ensemble payoff), not on standalone crps.**
If zinb-core diversifies the ensemble it earns a permanent seat as the one large-body arm; if not, we
drop it with a clean 3-seed negative and the two-axis matrix is closed either way. Then: **the bloom.**
