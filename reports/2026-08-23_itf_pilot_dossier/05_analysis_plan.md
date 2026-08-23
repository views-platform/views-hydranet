# Pre-registration — the ITF pilot (#287)

> **LOCKED before any arm runs.** Committed alone; `git log` proves the ordering against every result
> that follows. Unlike the #290/#291 checks — which were decision memos, because exploration had
> surfaced their values first — **nothing here has been measured yet.**

## §1 The question, and why it is well-controlled

**M30–M33**: scheduled sampling made the free-running rollout **worse** (AP@h18 0.3257 → 0.2831,
p=0.0286, 4v4). We ramped ε **0 → 0.5**, i.e. *increasing* the model's own output = **decreasing**
teacher forcing.

**Teutsch et al. 2022** report the opposite curriculum wins: *increasing* teacher forcing — start
free-running, add TF — beats both baselines **and** decreasing curricula across six chaotic systems
(NRMSE 16–81%), and warns that the decreasing direction *"can lead to premature termination of training
when applied to time series forecasting"* — **unlike NLP, where SS was proposed.**

**The pilot runs ε 0.5 → 0: the direction-reversed twin of an arm we have already run, at the same peak
dose.** Direction is therefore the single variable, and both comparison sets already exist at L=300:

| arm set | ε schedule | status |
|---|---|---|
| `fullzero_*` | ε = 0 throughout | **trained, 4 seeds** |
| `fullhalf_*` | ε: 0 → 0.5 (decreasing TF) | **trained, 4 seeds** |
| `itf_*` | **ε: 0.5 → 0 (increasing TF)** | this pilot |

## §2 The one variable

`ScheduledSamplingMixer` gains a **`reverse`** flag: `ε = ε_max · (1 − raw)` instead of `ε_max · raw`.
`reverse=False` is the default and **must be byte-identical** to today's behaviour. Nothing else changes
— same `ε_max = 0.5`, same `linear` schedule, same `warmup_lessons`, same seeds, same L=300.

⚠️ **This is `views_hydranet/` training code**, not a dossier probe. Every defect found today
(C-303/C-305/C-307/C-308) was in a probe, where the blast radius is one experiment. This changes what
the model learns.

## §3 Endpoints

* **Primary:** free-running gate **AP@h18**, target `sb`, calibration partition.
* **Secondary:** AP@h6 and h36; `act_ratio` (M31 showed SS *fixed* the zero collapse while losing AP —
  the same could happen here and would be the interesting result again).
* **Reference:** the seed-matched `fullzero_*` control, and the seed-matched `fullhalf_*` SS arm.

## §4 Decision rule — NUMERIC, and it is not a significance test

**A 2v2 cannot reach significance.** The exact one-sided permutation test's floor is
`1 / C(4,2) = 0.167`. Presenting a 2v2 p-value as evidence would be theatre. The pilot is a
**direction-and-magnitude screen**, and it is labelled as such.

Measured control seed sd at h18 (n=4, `fullzero_*`): **σ = 0.0134**.

| outcome | verdict |
|---|---|
| **both** ITF seeds ≥ their control **+ 1σ (+0.0134)** | **PROMOTE** — extend to 4v4 for a real test |
| **both** ITF seeds ≤ their control **− 1σ** | **ITF fails too** — direction is not the answer; #287 gets its false-negative mode and reopen trigger and closes |
| anything else | **INCONCLUSIVE** — report and decide; do **not** promote and do **not** close |

σ is the *measured* spread of the thing being compared, chosen before any ITF number exists.

**Anchor guard (§4 of the SS pre-registration, carried):** if `|ΔAP(h1)| > 3 × MDE_AP(h1)`, ITF traded
one-step skill for horizon skill — report as a **traded failure**, not a retention result.

**Pairing:** ITF-vs-control at a fixed seed differs in one flag, so per-seed differences use
`scripts/ap_block_bootstrap.ap_diff_origin_block_ci` — **not** the between-seed MDE. **C-306** is
precisely the error of judging one contrast with another's yardstick.

## §5 Abort criterion — decided now, not when it looks wrong

Teutsch warns ITF can terminate training early, and **#287 registers the specific risk**: ITF starts
free-running, our gate collapses under free-running (M16), so a model that never sees a good state may
never train off the floor.

**An ITF arm that fails `scripts/floor_gate.py` FG-A (`AP_ctrl(h*) ≥ 5 × prevalence`) did not train, and
is reported as "did not train" — never as "ITF is worse".** The floor gate is an existing, tested
instrument with a pinned threshold md5; using it removes the judgement call.

**M28** is the precedent: two 40-lesson arms failed FG-A and were correctly classed as smoke, not as
evidence.

## §6 Falsifiers

1. **`reverse=False` must be byte-identical** to the current mixer, asserted over the full lesson range —
   otherwise this change silently perturbs every existing arm.
2. **ε must actually decrease.** A test asserts `ε(0) ≈ ε_max` and `ε(L−1) ≈ 0` under `reverse=True`; a
   flag that transposes nothing would produce an "ITF" arm identical to SS.
3. **One `views_hydranet` tree hash** across the pilot's arms (F6), so ITF and control differ only in the
   flag.
4. **The ε=0 controls are reused, not re-trained** — they must reproduce their published AP@h18
   (0.3298 / 0.3318 / 0.3058 / 0.3352) or the comparison is not seed-matched.

## §7 False-negative mode and reopen trigger (C-307, recorded up front)

**This is a pilot, and a null here is weak evidence.** Registered *before* the result:

**False-negative mode.** ε starts at **0.5, not 1.0** — a softened ITF chosen to protect against the
training risk in §5. A null therefore cannot distinguish *"ITF fails"* from *"we did not run real ITF"*.
It is **half the paper's method**, and the paper's gains are reported for the full one.

**Reopen trigger — any of:** a null at ε=0.5 while the arms train cleanly (⇒ try ε=1.0 with the §5 abort
criterion, which is then the real test); **#294** proceeding, since aGTF anneals α **downward from 1**
and is the same curriculum from the other direction; or anyone wanting the training-time counterpart of
M38/M39/M41.

## §8 Scope

**2 seeds (42, 43)**, one vehicle, L=300, `sb`, **AP only**. ~5 h/arm ⇒ **~10 h**. The `crps_all`
ARTIFACT verdict (#263) is untouched. A 2v2 screens direction; it does not measure an effect.
