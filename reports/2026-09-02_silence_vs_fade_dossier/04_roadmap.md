# 04 — Roadmap

**2026-09-02**

| Phase | Gate to enter | Deliverable |
|---|---|---|
| **P0 — Lock** | — | `05` committed. No GPU second before this. |
| **P1 — Instrument** | P0 | `mu`-field dump, default off (`03` §C.1); parity test (§C.2); sentinel guard (§C.4). `ruff` + `pytest` green. |
| **P2 — Pre-flight** | P1 | `03` §D checklist all green, incl. the `AP@h18` identity anchor. |
| **P3 — EXP-1 (seed 42)** | P2 | Two arms emitted. Readouts in the order fixed by `05` §6.3. G1 checked **before** any claim is read. |
| **P4 — Replication (seed 43)** | P3 with **no falsifier fired** | Same two arms, seed 43. |
| **P5 — Write-up** | P3 (either way) | `07` entry, ledger row **M51**, and corrections to anything that rested on the claim. |

**Hard stop:** if F3, F4, F6 or F7 fires, the program halts at that gate. The defect is registered
and no scientific claim is made in either direction.

**P5 is unconditional.** It runs on falsification exactly as it runs on support.
