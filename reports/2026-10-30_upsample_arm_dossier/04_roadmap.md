# 04 — Roadmap, phased and gated

**2026-10-30.** Each phase's gate must be green before the next begins. ⛔ The gates are the point:
this programme's one process failure was crossing three of them in a night.

## Phase A — the code to review standard · ✅ DONE

`C7-1` (`F6` Amendment 6 — else the arm's own checkpoint is undiagnosable), `C7-2` (widths read off
the module replaced + a fail-loud build guard), `C7-3` (`preflight_exp02.py`, so the gate set does
not fork), `C7-4` (relational parameter assertions).
**Gate:** `ruff` clean; 2409 passed, 0 failed; registry-contract and CIC-drift tests green. ✅

## Phase B — the dossier · ✅ DONE (this directory)

**Gate:** `03`'s pre-flight checklist written from a **repo audit**, with the gaps named. ✅

## Phase C — guard audit · ⛔ NEXT · no GPU

`/falsify guard` over the 12 new mechanism guard cases and `G-F`'s `G-17`/`G-23`/`G-24`/`G-25`/`G-26`,
⛔ **in a clean-context subagent** — the auditor may not be the author.
**Gate:** every guard **HOLDS**. Any DECORATIVE or WEAK verdict is fixed here, not later.
⭐ One guard is already mutation-verified (the non-conforming-stage refusal); the rest are not.

## Phase D — release 0.1.8 · ⛔ BLOCKER for everything after it

The step all six EXP-01 candidates took and this one had not. Rehearsal-first, Trusted Publishing.
⚠️ **It ships the 0.1.6 and 0.1.7 content too** — reviewing that diff is part of this phase.
**Gate:** rehearsal green; `version("views-hydranet") == "0.1.8"`; `UpsampleConv2` resolvable from
**site-packages with no `PYTHONPATH`**.

## Phase E — arm construction and the two sentinels · GPU, minutes to ~1 h

1. `make_arch_arm.py --lessons 300 --eps 0.0 --seed {42,43} --variant {HydraBNUNet06_LSTM4,UpsampleConv2}`
   → four `*fullzero*` dirs.
2. The refetch decision, **logged in `05` with its vintage caveat** before it is taken.
3. `preflight_exp02.py --out results` → `PREFLIGHT_OK`.
4. `smoke_exp02.sh` → `SMOKE_OK` (2 lessons per architecture; the real footprint, emit path, BN recal).

**Gate:** both sentinels written, and `03` §D fully green.

## Phase F — the queue · ~7–10 GPU-h local, sequential

`launch_exp02.sh`: **control first, both seeds**, then the arm.
⚠️ **No local concurrency is possible** — EXP-01's measured footprints are 4.1–4.7 GiB of an 8.2 GiB
card, so exactly one run fits. 4 pods in parallel (~2.5 h) becomes possible once Phase D ships.
**Gate:** continuous — `verify_bakeoff.py` runs after **every** arm and a non-zero exit stops the
queue.

## Phase G — read-out and write-up

Full gate+body battery at **h1/6/12/18/24/30/36**, oracle ceiling, retention, parameter counts
beside every result. ⛔ **Falsifiers evaluated before any verdict is written.**
⚠️ Needs an EXP-02 read-out first (`03` §C6).
Outputs: `07` here **and** an EXP-02 entry in EXP-01's `07`; a `RESULTS_LEDGER` row; `S-1`, `S-2`,
`P-20`…`P-27` all closed; the body index updated; #491 corrected.

## Decision points

| point | question | consequence |
|---|---|---|
| after C | does any guard fail to hold? | fix before the release; a decorative guard would let a neutered arm report a false null |
| after D | is the 0.1.6/0.1.7 diff acceptable to ship? | if not, the release narrows and this phase repeats |
| after E | does the control reproduce EXP-01's AP@h18 (**P-27**)? | ⛔ if not, the code and/or data moved since August and every cross-reference to EXP-01's **absolute** numbers becomes indicative. Arm-vs-control stays valid |
| after F | does `P-23` hold (ΔAP > 0 at h1, < 0 at h36, both seeds)? | **CONFIRMS** → the architecture axis closes on evidence and the loss-horizon follow-up is promoted. **FALSIFIES** → spatial architecture is live again. **MIXED** → under-powered; closes nothing |
