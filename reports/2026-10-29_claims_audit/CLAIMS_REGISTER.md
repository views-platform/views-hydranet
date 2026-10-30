# CLAIMS REGISTER — what this week's written record can be stood on

**2026-10-29.** ⭐ **This is the one page to read.** It exists because a retraction (`W-107`) and an
attribution error (`W-108`) made *"what do I actually know?"* unanswerable, and the answer should
not depend on my memory or my good faith. **Every row names a file and a number you can check.**

Scope: the **5,416 lines of markdown this window added or changed** since 2026-10-26, across **38
documents** — per `git log`, not per memory. Of those, **714 are asserting claims carrying a
measurement**; **173 (24.2%)** were flagged for a human read by
`audit_claims.py` (`65ae1c6`, amendments `6f720f9` / `888af23` / +2), whose output is
`audit_claims.out`. I then read the flags. ⛔ The script triages; it does not judge.

---

## ⭐⭐ THE ANSWER IN FOUR LINES

1. ⭐ **No new claim of the `W-107` / `W-108` shape exists.** All 7 CROSS-SOURCE flags are false
   positives. The two known failures are the only two.
2. ⛔ **One live uncorrected instance of an already-falsified claim was found, and is now fixed** —
   `C-391` still said the `nb` checkpoint was *deleted* six days after `W-99` recorded that it is
   on disk.
3. ⛔⛔ **One new error, and it is mine from today, in the document I called the index of what we
   know** — `W-109`. I stated a **relayed** number as a measured one.
4. ⚠️ **One provenance gap, and it sits under Q2** — the 2026-10-18 gate dossier **captured no run
   output at all.**

---

## 1 · The claims that CHANGED — corrections applied

| claim | document | verdict | why |
|---|---|---|---|
| *"the published B-1a 0.391 / B-1b 0.6267 are `nb` @ 0.15 on a **deleted** checkpoint"* | `technical_risk_register.md` **C-391** | ⛔ **RESCOPE → corrected in place** | **False, and known false since 2026-10-28** (`C-394`, `W-99`). The checkpoint is on disk, sha `4acbf4d3…`, exactly the sha inside `d4_result.npz`. ⚠️ **C-391's scope limit itself is unchanged and still binding** — that checkpoint has no replicate, so σ̂ for `nb` @ 0.15 stays **unmeasured**. The artifact existing does not supply one |
| *"gate-weighted mean θ = **0.0902** — **24,000×** the grid median"* | `2026-10-29_body_state_of_knowledge/FINDING.md` | ⛔⛔ **RESCOPE → corrected + `W-109` logged** | **0.0902 appears in NO committed `.out` in this repository.** Relayed from ADR-073 §0's prose, never produced by a run, and stated flatly by me as a measured correction. ⭐ The **direction** stands — θ on the grid is not θ where the model emits — but **the factor is not a measured number here** |

### ⭐ The one I nearly broke by not reading

`2026-10-28_sigma_hat/FINDING_RP2_SIGMA.md:112` carries the same false *"DELETED checkpoint"*
sentence — and a `⛔⛔ CORRECTED 2026-10-28` block **two lines beneath it**. Verdict **STANDS**.
⇒ Acting on the flag without reading the context would have "corrected" an already-corrected
document. **That is the over-disclaiming harm this register is supposed to avoid**: a false row
teaches distrust of sound work exactly as a missing row hides bad work.

---

## 2 · The claims that STAND — and the evidence that they do

| finding | count | what it means |
|---|---|---|
| **CROSS-SOURCE** — two numbers compared whose producing `.out` files are disjoint | **7, all false positives** | ⭐⭐ **The load-bearing reassurance.** Every one is an arrow transition between a *control* and an *arm* output — different files, like-for-like by design (`0.400 → 0.090`, `2.5323 → 2.6155`). **No third instance of the `W-107`/`W-108` shape exists in this week's work** |
| **DECLARED-ONLY** — a number appearing in `.out` only as a declared expectation | **28, and the record already handles them** | Every `0.039327` occurrence sits inside a sentence saying so: *"reproduced by neither implementation"*, *"remains reproduced by no"*, *"still reproduced by no implementation"*. ⭐ The habit `W-98` installed is working |
| **documents whose new claims carry no flag at all** | **10 of 38** | Verdict **STANDS**, no edit. Listed in §5 |

---

## 3 · MODE-UNNAMED (97) — a class-level verdict, not 97 edits

97 claims compare against a rival — a baseline, climatology, persistence, what ships — without
naming the inference mode on the line.

⚠️ **But the mode slot was created on 2026-10-29, by `C-399`, as a consequence of `W-107`.** A claim
written before the requirement existed cannot be faulted for omitting it, and **97 retroactive
banners would be exactly the over-disclaiming this register is against.** What matters is that both
halves are already handled:

- every document carrying the **retracted** cross-regime numbers has a banner, and ADR-073 §0 a
  correction block — verified by reading all 38, not asserted;
- **new** claims carry the slot: M89 and `2026-10-29_delivered_forecast/` name the inference mode
  per row.

⇒ **Verdict: STANDS as a class.** The remedy is the habit going forward, enforced mechanically by
`audit_claims.py`, not a retrospective edit of work that was correct under the rules it was written
under.

---

## 4 · ⚠️ The provenance gap, and it is under the question Simon actually cares about

**`reports/2026-10-18_gate_underdispersion_prereg/` captured no run output at all.** Its numbers —
including the gate slope **1.146** control / **1.419** arm, the evidence that *the gate* is
under-dispersed — live in `RESULTS.md` prose, labelled `MEASURED` beside their scripts
(`results/e2_meanlogit.py`). They are not fabricated and the label is honest. ⛔ **But they cannot
be verified without re-running that script.**

This matters because it is the only measured dispersion defect anywhere in the record, and
`2026-10-29_body_state_of_knowledge/FINDING.md` uses it to draw the *"do not conflate the gate with
the body"* boundary. ⇒ **The boundary stands; the gate's numbers are evidenced in prose, not in a
capture.** Flagged, not fixed — re-running `e2_meanlogit.py` is cheap and belongs with Q2.

---

## 5 · Documents whose new claims carry no flag — STANDS, untouched

```
2026-10-12_underfire_diagnosis/ARTIFACT_MANIFEST.md
2026-10-12_underfire_diagnosis/d4_result_CONTROL_nb_4acbf4d3.PROVENANCE.md
2026-10-21_structure_penalty_arm_prereg.md
2026-10-24_control_vehicle_prereg.md
2026-10-27_hypothesis_h/FINDING.md
2026-10-27_hypothesis_h/QUANTILE_HEAD_STRUCTURAL.md
2026-10-27_measurement_floor/ARGUMENT.md
2026-10-28_candidate_ranking_DRAFT.md
2026-10-28_guardrail_coverage/FINDING_M5.md
2026-10-28_rp3_l5/FINDING_L5.md
```

---

## 6 · ⛔ What this register does NOT cover — stated, not buried

1. ⛔ **Pre-existing claims in files we touched but did not change.** Scope is the diff. The
   alternative is auditing the repository's whole history. ⇒ The one case that matters —
   pre-existing claims *invalidated* by this week's retraction — was checked separately by reading
   all 38 documents for a banner, and §2 reports the result.
2. ⛔ **A wrong number that is internally consistent.** If a run produced it and a document quotes
   it faithfully, nothing here fires. Only re-measurement catches that.
3. ⛔ **Prose claims carrying no number.** The instrument is numeric.
4. ⚠️ **`.out` files only** as the evidence corpus — 85 tracked, 26 on disk untracked (their own
   lesser class, `UNTRACKED-ARTIFACT`, 9 flags). ⭐ **Deliberately not widened to `.md`: a claim in
   prose cannot serve as its own evidence.** §4 is the cost of that choice, reported rather than
   engineered around.
5. ⚠️ **The instrument took four amendments** — 56% → 39.5% → 25.4% → 24.8% → **24.2%** flag rate
   against a stop threshold of 30% declared before the first run. Two were my scope errors (whole
   files instead of the diff; treating `→` as a comparison) and two were defects in my own regex
   (an unmatched exponent read `9.983e-01` as `9.983`; a trailing `%` read `99.6567%` as
   `99.6567`). ⛔ **G-0's four gates were unchanged through all four and passed every time** —
   tightening is only legitimate if it still catches everything known to be bad.

---

## 7 · How to check any row

| to verify | run |
|---|---|
| the scope is the real diff | `git log --since=2026-10-26 -U0 -p --all -- '*.md'` |
| `0.0902` has no artifact | `grep -rl "0\.0902" --include="*.out" reports/` → **nothing** |
| `0.039327` was never produced | `grep -rn "0\.039327" --include="*.out" reports/` → four hits, **every one** a failed gate's expectation or a quoted ledger row |
| the checkpoint is on disk | `sha256sum ~/Documents/views_hydranet_vehicle_20261012/calibration_model_20261012_033112.pt` → `4acbf4d3…` |
| the gate dossier has no captures | `ls reports/2026-10-18_gate_underdispersion_prereg/**/*.out` → **none** |
| the whole triage | `reports/2026-10-29_claims_audit/audit_claims.out` |

⭐ **The point of this table is that you never have to take my word for a single row above it.**
