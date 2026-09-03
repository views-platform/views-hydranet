# 03 — Harness and invariants (Wave 1)

**2026-09-03.** What makes an 8-hour unattended run trustworthy, and what a red-team review found
wrong with the first version of it.

## A. Invariants

### A.1 Hard — violating one invalidates the wave
| invariant | enforced by |
|---|---|
| all arms run on the same code | git-HEAD guard aborts an arm if the repo moved |
| all arms run on the same instrument | every dump records `n_passes`; the verifier asserts 4 |
| the cube path is unchanged by the dump | reproduction falsifier against two archived `AP@h18` values |
| the clamp acts only for `t > origin` | h1 identity across arms within a seed |
| an arm is complete or it failed | 13 origins required; a partial dump is a failure, not a short arm |
| fail loud, never substitute | CUDA-availability gate; no silent CPU fallback |

### A.2 Deliberately varied
`freeze_recurrent` only. Nothing else.

### A.3 Respect while changing
The **posterior-mean dump** was changed immediately before this wave (pass-0 → mean over all D
passes). It touches no computation the cube uses, which the reproduction falsifier verifies rather
than assumes.

## B. What the red-team found — the first launcher would have died

A read-only review of the queue, run before launch, found one night-ending bug and four expensive
ones. The queue had **already been started** with the bad value and was killed four minutes in.

| finding | consequence had it stood |
|---|---|
| **`ARM_TIMEOUT` 1800 s on a 7–12 min estimate** | the `none` arm is **measured at 2127.8 s**; the timeout kills arm 1, and a killed arm leaves a 2.5 GB cube that makes the seed's other **three** arms die instantly on "refusing to start". One constant, four arms, per seed. |
| no `WANDB_MODE=offline` | C-163: `wandb.finish()` DNS-hung **38 minutes** on this box |
| no CUDA gate | device selection falls back to CPU **silently at DEBUG level**; 16 arms become a 16-hour grind that looks like slow progress |
| no foreign-GPU gate | `ollama serve` runs on this machine; if invoked it loads a model into VRAM and OOMs the arm |
| no `_pf_staging` cleanup | pipeline-core removes it **only on the success path**, it is indexed by sequential `origin_i` so a stale tree is shape-compatible with the next arm's, and nothing in this repo globs it |
| score CSV as skip token | it is written **before** cube deletion, so a mid-write kill marks a broken arm done |

Fixed, plus: a **stall watchdog** (the runner's stdout does not grow during an arm — it writes the
child's log only after the child exits — so the live signal is the dump directory, one file per
origin); a per-arm `--tag` so sentinels are per-arm; a circuit breaker at **3** consecutive failures
(2 would trip on one bad seed); a 6.5 h deadline on *starting* arms so the finisher runs inside the
window; and a finisher that **must never raise**, since partial state is its normal input.

## C. Instruments, all mutation-tested before the data landed

| tool | tests | mutations | what a mutation found |
|---|---|---|---|
| `wave1_data.py` | (shared) | — | exists so the **C-322 grid flip lives in one place**: naive placement correlates 0.026, the flip 1.0000, and the failure is silent |
| `subset_ap.py` | 6 | **7/7** | — |
| `escalation.py` | 7 | **7/7** | the dispersion cohort was **unpinned**; on a 99%-empty field that measures the background, not the conflict |
| `roll_diagnosis.py` (predecessor) | 9 | **7/7** | — |

The discipline that matters: each instrument recovers a **planted** signal before it is allowed near
real data, and each is required to produce the answer that would *kill* the hypothesis, not only the
one that confirms it.

## D. Operating scar recorded during the run

**Committing moves HEAD and the guard aborts the queue.** I committed analysis tools mid-wave and
had to re-baseline `repo.head` by hand. The guard is right — it exists to catch *another session*
changing code mid-run — but it cannot tell my dossier-tool commit from a change to the inference
path. Procedure for the rest of the wave: commit, then immediately re-baseline and log it in
`ANOMALIES.txt`. Every re-baseline is recorded with its justification.
