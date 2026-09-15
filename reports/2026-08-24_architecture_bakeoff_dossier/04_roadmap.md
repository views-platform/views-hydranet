# 04 — Roadmap (2026-08-24), phased and gated

Every phase has an explicit gate. **A red gate stops the programme; it does not get argued past.**

## Phase 0 — harness (no GPU) · gate: `03` §D fully green

G1 registry → G2 byte-identity → G7 six architectures + tests → G3 queue identity → G5 postflight audit
(validated on today's completed arms) → G4 smoke script → G6 CUDA assert.

**Gate:** `03` §D checklist green, full suite + lint green, tree committed.

## Phase 1 — smoke (≈1 GPU-h) · gate: six sentinels + a cost projection that fits

Each architecture trains 2 lessons, emits, scores. Records **peak GPU memory** and **wall-clock per
lesson**, and projects the 300-lesson cost.

**Gate, all four:** every architecture produced a scored smoke; peak memory leaves headroom on 8,188
MiB; projected 300-lesson time is inside the 12.5 h timeout **with margin**; total projected queue time
fits the window. **An architecture that fails any of these is dropped from the queue, not "watched".**

## Phase 2 — pre-registration · gate: `05` committed alone

Locked before any 300-lesson arm; `tools/` for it empty at lock time so `git log` proves the ordering.

## Phase 3 — the queue (≈29 GPU-h) · gate: continuous

12 arms — six architectures × seeds 42, 43 — sequential, `setsid`, `RES_DIR` in-repo. The postflight
audit runs inside the verify hook after **every** arm; a non-zero exit stops the queue.

**Ordering is deliberate: candidate (1) AntiAliasedPool first**, both seeds. It is the zero-parameter
arm with the clearest mechanism — if the harness is going to fail, it fails on the cheapest arm, and if
(1) moves the needle we learn the most per hour. Then (6), (2), (3), (4), (5) — capacity-adders last,
because they are the slowest and the most likely to hit memory.

**On crash:** relaunch the identical command. Completed arms are skipped on the strength of their
`score_*.csv` + `score_*_use_real.csv`. Nothing already finished is lost or re-run.

## Phase 4 — read-out and write-up · gate: falsifiers before verdicts

Full gate+body battery at all seven horizons, oracle ceiling and retention per arm, parameter counts
beside every result. Falsifiers evaluated **before** any verdict is written.

## Decision points

| point | question | consequence |
|---|---|---|
| end of Phase 1 | does any architecture fail smoke? | dropped, recorded, queue re-projected |
| end of Phase 1 | does the projected total exceed the window? | cut arms **by pre-registered priority order**, never by early results |
| after arm 2 | did the postflight audit pass on a real 300-lesson arm? | if not, stop — the harness is wrong, and 10 arms are still unspent |
| Phase 4 | did any candidate beat the control on AP **without** a body regression? | that is the only outcome that promotes; an AP gain with a `crps_all` regression is a **trade** |
