# Candidate 7 — the decoder's upsampling path

**Opened 2026-10-30.** Living spine; every other doc here is point-in-time and dated in its header.

## 1 · Purpose

The body's forecast field is **spatially too smooth at short range** — measured, model-free and
threshold-swept (`M92`, `M93`). This programme asks whether that is caused by the decoder's two
`ConvTranspose2d(kernel=2, stride=2)` stages, which **cannot mix spatially at all** (kernel =
stride ⇒ no window overlaps ⇒ each output cell is a fixed per-channel scalar multiple of exactly
one input cell), and whether replacing them changes **rollout skill** — not just the per-frame map.

## 2 · Relationship to prior work

⛔ **This is candidate 7 of the `2026-08-24_architecture_bakeoff_dossier`, not a new programme.**
That dossier ran **six** spatial-precision candidates × 2 seeds at L=300, promoted **nothing**, and
produced a post-hoc finding this programme exists to test prospectively: **12 of 12 arms negative
at h30 and h36**, monotone across the horizon, with the damage **ordered by how aggressively each
architecture sharpens the per-frame map** — while *"handed a real field, every candidate matches or
slightly beats the control's ceiling."*

⇒ EXP-02's read-out appends to **that dossier's** `07_experiment_log` as well as to this one.
⭐ `ShallowPool`'s docstring named `M92`'s mechanism **two months before it was measured**.

**ADRs:** none is created by this programme. The six prior candidates have **no CIC and no ADR**,
and the `ArchitectureRegistry` CIC states a **name-agnostic** contract already enforced over every
registered name. ⇒ An ADR is the **exit ramp on `promote`**, if the arm wins — not an entry fee.

## 3 · Document index

| # | file | status |
|---|---|---|
| 00 | `00_README.md` | **living** |
| 01 | `01_literature.md` | ⚠️ **drafted with a load-bearing gap** — the mechanism's primary source is not held |
| 02 | `02_design.md` | drafted |
| 03 | `03_harness_and_invariants.md` | ⭐ **drafted from a repo audit, not a template** |
| 04 | `04_roadmap.md` | drafted |
| 05 | `05_analysis_plan.md` | drafted — the pre-registration itself is `PREREG.md` (absorbed, below) |
| 06 | `06_glossary.md` | drafted |
| 07 | `07_experiment_log.md` | **append-only**; `G-F` (EXP-02a), the guard audit (EXP-02b), and a postmortem of this programme's own process failure |

**Absorbed artifacts** (this directory was a flat `reports/2026-10-30_upsample_arm/` until it became
this dossier; nothing was duplicated):

| file | role |
|---|---|
| `PREREG.md` | the pre-registration + amendment 1. `05` links it rather than restating it |
| `gf_phase_signature.py` / `.out` | the `G-F` mechanism probe and its captured output |
| `FINDING_GF.md` | `G-F`'s read-out |
| `tools/` | `preflight_exp02.py`, `smoke_exp02.sh`, `launch_exp02.sh` |

## 4 · Harness at a glance

⭐ **Most of it already existed.** The audit in `03` found the registry extension seam, a
name-agnostic contract test, mechanism-pinning tests created *because* a `/falsify guard` pass
found every prior candidate could be neutered silently, an independent arm-identity verifier, a
per-arm postflight, a real-pipeline smoke gate, a device gate, an entropy lock, artifact identity
(`F6`), and a data/config provenance guard.

⛔ **The gaps are not in the harness — they are in this programme's position relative to it.** The
blocking one: **the candidate is not in a released wheel**, and every arm in this repository trains
against the installed package.

## 5 · Current state & next actions

- [x] mechanism measured (`M92`, `M93`) and localised to a named defect
- [x] remedy implemented as a registry candidate; **28 mechanism test cases** (12 → 28 after the audit); `G-F` selected the arm
- [x] ⛔ **three process bypasses removed** — see `07`'s postmortem
- [x] code to review standard: `C7-1` (`F6` Amendment 6), `C7-2`, `C7-3`, `C7-4`
- [x] ⛔ **`/falsify guard` in a clean-context subagent — FALSIFIED, 13 survivors, all closed and
      mutation-verified; the instrument re-run and every number reproduced** (`07` EXP-02b)
- [ ] ⛔ **release 0.1.8** via the rehearsal path; review the 0.1.6/0.1.7 content that ships with it
- [ ] build the four `*fullzero*` arm dirs, then `preflight_exp02.py` → `smoke_exp02.sh` → `launch_exp02.sh`
- [ ] ⚠️ route `C7-6`/`C7-7` through `register-risk` (do not hand-edit the register)
- [ ] ⚠️ fetch Odena et al. 2016 — `01` §gaps
- [ ] ⚠️ EXP-02 needs its own read-out: `verify_bakeoff.py:118` derives completion from EXP-01's
      `ARCHES × SEEDS`, so a 4-arm queue never reports COMPLETE. Identity and postflight — the
      parts that stop the queue — are per-arm and unaffected.

## 6 · Conventions

Numbered docs, dated in-header; `00` living, `07` append-only. `reports/` is gitignored, so
everything here is **`git add -f`** with named files (`C-395`). Measurement scripts are **committed
before they run** (D-5). On close the directory moves to `reports/archived/`.
