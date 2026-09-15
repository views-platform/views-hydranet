# Decay dial — PAUSED 2026-08-22 10:00, waiting for the machine

**Nothing is running.** No emit processes, no partial cubes, GPU idle, 97 GB free.

## Why paused

The machine is shared and was carrying ~11 cores of other work:

| process | elapsed at 09:57 | CPU |
|---|--:|--:|
| `library_rebuild` (a `/library` run, not this session's) | 40 min | **978%** |
| `pytest` from another Claude session (different shell snapshot) | 3 min | 78% |
| this sweep's emit | — | **93.9% — starved to one core** |

Measured under that load: **240 s/origin → 52 min/arm**. The overnight hard-freeze arms ran the
same vehicle at ~55 s/origin. The 4× is **contention, not code** — see below.

## To resume

```bash
# check the machine first
uptime; ps -eo pcpu,args --sort=-pcpu | head -4

cd /home/simon/Documents/scripts/views_platform/views-hydranet
D=reports/2026-08-22_state_freeze_l300_dossier
DIAL_ARMS="cell@0.5" setsid nohup bash $D/tools/run_decay_dial.sh \
  > $D/results/dial_launcher.log 2>&1 < /dev/null &
```

`DIAL_ARMS` defaults to `cell@0.25,cell@0.5,cell@0.75`. **Start with `cell@0.5` alone** — the
midpoint is the single most informative arm, and it answers the shape question by itself:

| `cell@0.5` at h18 | reading |
|---|---|
| **> 0.3709** | a **dial** with an interior optimum — sweep 0.25 and 0.75 next |
| between **0.3318** and **0.3709** | **monotone ⇒ a switch**; the hard freeze is the answer, ship it |
| **< 0.3318** | non-monotone in the other direction; the mechanism is not what we think |

Endpoints already measured on this exact vehicle (seed 43): `none` **0.3318**, `cell` **0.3709**.

Confirm any interior win on seed 42 before believing it — one seed has decided nothing in this
programme.

## The diagnosis I got wrong, recorded

I attributed the 4× slowdown to the new blend arithmetic and killed the run on that basis. **The
microbenchmark had already ruled it out** — the blend adds ~5 ms/call over the hard-freeze path,
which at ~36 rollout steps is *seconds* per origin, not the 185 s observed. Only after that did I
check the machine and find the 10-core rebuild.

**Third time this session I diagnosed from a measurement whose conditions I had not checked** (a
25-second lesson rate; one `nvidia-smi` line; now this). The standing rule — measure over an
interval — is not sufficient on a **shared** machine: the interval has to be one where you know
what else is running. `uptime` and `ps --sort=-pcpu` before any timing claim.

The `torch.lerp`-on-half change **stays**: 36.2 → 12.3 ms/call is strictly better and the previous
version did half its work for nothing. But it was a fix to a non-problem, and the code comment now
says so rather than crediting it with a regression it did not cause.
