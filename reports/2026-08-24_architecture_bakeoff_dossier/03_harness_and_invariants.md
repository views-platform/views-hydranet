# 03 — Harness and invariants (2026-08-24) — **THE GATING DOCUMENT**

A 12-arm queue is ~**29 GPU-hours**. Three separate runs have already been destroyed by small harness
faults (three arms SIGKILLed by an under-sized timeout, ~12 GPU-h; one 7-hour run lost to a `RES_DIR`
inside a wiped `/tmp`; one queue aborted mid-flight by a commit moving HEAD). **This document exists so
the next 29 hours are not lost the same way.** Nothing here is an assurance; every row names the
mechanism and how it is tested.

---

## A. Invariant taxonomy

### A1. Hard invariants — never break (violation ⇒ the experiment is invalid)

| invariant | enforced by |
|---|---|
| The incumbent architecture stays the default and **byte-identical** when no new `model` is selected | new test: registry path vs direct construction, fixed seed, `torch.equal` on `reg`/`cls`/`h_next` |
| An arm differs from its control in **exactly one config key** (`model`) | arm builder's symmetric-difference check against the **control**, not just the floor |
| Every arm is scored on the **same support and origins** | `score_v2_horizons.py` intersects support across arms; `ap_diff_origin_block_ci` **refuses** on support/origin/gate mismatch |
| Full suite + `ruff check` + `ruff format` green before launch | CI + local |
| No silent NaN/Inf in any emitted score | postflight audit (new) |
| Fail-loud over silent degradation | no new `try/except` that swallows |

### A2. Deliberately changed by this program

The **architecture** — behind `config['model']`, which defaults to the incumbent. Each candidate is a
separate file and a separate registry entry; **none modifies the incumbent's file**.

Also, deliberately: the **ADR-061 top-skip seam**, retired for *static* content by C-228/C-230, is
re-tested here for *dynamic* content. That retirement is not being overturned by assertion — candidates
(2) and (3) are the test, and (3) is the form C-230 recommends.

### A3. Respect while changing

* **Parameter count** — (5) and (6) change it substantially; recorded per arm, never compared as if held
  constant (see `02` §capacity confound).
* **GPU memory** — 8,188 MiB total; the incumbent peaks ~3.6 GiB in training. (5)/(6) must be measured
  in smoke before a 300-lesson arm is allowed.
* **Wall-clock** — `TRAIN_TIMEOUT = 150 s × lessons` = 12.5 h at L=300. A slower architecture must be
  projected from smoke, not assumed.
* **The `hs_1..hs_4` split** — `blend_recurrent_state` and the state-freeze diagnostics assume
  `channels % 8 == 0` (4 short-term + 4 long-term groups). Candidate (6) changes the width; it **must**
  keep divisibility by 8 or it silently breaks the freeze machinery.

---

## B. The standing harness — audited, not templated

Discovered in the repo and **reused as-is**:

| mechanism | where | what it actually does |
|---|---|---|
| Single-scheduler lock | `run_queue.sh` | `flock -n` on `.queue.lock`; a second queue exits 11. No `pgrep` anywhere. |
| **Resumability** | `run_queue.sh` | an arm whose `score_<arm>.csv` **and** `score_<arm>_use_real.csv` are both non-empty is **SKIPPED** and counted done. **This is the crash-resilience property** — relaunching the identical command continues. |
| Disk preflight | `run_queue.sh` + `run_lesson_arm.sh` + `run_realism_arms.py` | 25 GB at three layers; the queue **pauses and re-checks** every 10 min rather than failing; unparseable `df` aborts. |
| HEAD-drift abort (F6) | `run_queue.sh:assert_head` | HEAD captured at start, re-checked before every arm; a change aborts the queue. ⇒ **no commits while running.** |
| Train timeout | `run_lesson_arm.sh` | `150 s × lessons`; there is **no mid-training checkpoint**, so a killed arm is lost entirely. |
| Cube hygiene | `run_lesson_arm.sh` | refuses to start if `predictions_*` exists; deletes the cube **only after** every artifact is confirmed present. |
| **Cube-deletion interlock** | `run_lesson_arm.sh` | if any of `ap_ci`/`ret_ci`/`arm_*.json` is missing or non-zero rc, the cube is **kept** and the arm exits 10 — so a failed bootstrap costs a re-score, not a retrain. |
| Floor gate | `scripts/floor_gate.py` | FG-A ≥5× prevalence, FG-B vs climatology, FG-C vs 3×MDE; thresholds md5-pinned `6d5714d5ceda147ed16f53143abe7e37`; verdict in the **filename**. |
| Paired CIs | `scripts/ap_block_bootstrap.py` | origin-block; refuses <3 origins, non-finite AP, support/origin/gate mismatch. |
| Verify-after-every-arm | `run_queue.sh` | runs `$VERIFIER` after each arm and **checks its exit code** — a crashed verifier stops the queue. **This is the hook the postflight audit plugs into.** |
| Detachment | launch discipline | `setsid` (plain background jobs are reaped); `RES_DIR` **inside the repo** (a `/tmp` scratchpad was wiped mid-run and took a live run's log with it). |
| Arm-builder verification | `make_*_arm.py` | execs both configs and requires the symmetric difference to equal exactly the intended key set; refuses to overwrite an existing arm. |
| Provenance | artifact sidecar | `.pt.config.json` records `model`, `output_distribution`, `static_top_skip`, widths — so a scoring run cannot silently rebuild the wrong architecture. |

---

## C. New harness this program needs — **the gaps, and the build order**

| # | gap | why it matters here | the build |
|---|---|---|---|
| **G1** | `choose_model` is a hardcoded `if/else` with one branch | six new architectures would mean six edits to a dispatcher — the OCP violation `harness.md` warns about | a **registry** keyed by name; unknown name raises with the available set |
| **G2** | **baseline byte-identity is unproven** across the registry refactor | the refactor touches the path every existing run uses | test: registry vs direct construction, fixed seed, `torch.equal` on all three outputs |
| **G3** | the queue's arm-identity tuple is `lessons:seed:eps:ss_reverse` — **`model` is absent** | on the resume path an arm built on architecture A could be silently reused when B was asked for. With six architectures this is the single most dangerous gap. | extend the tuple to include `model`; test both the match and mismatch paths |
| **G4** | **no preflight smoke** | an architecture that fails to instantiate, emits wrong shapes, or OOMs would burn a queue slot ~2.4 h in | a 2-lesson smoke per architecture that trains, emits, scores, and records **peak GPU memory and wall-clock**; the queue **refuses to start** unless every smoke sentinel exists |
| **G5** | **no postflight setup audit** | today the verifier checks the *result*; nothing checks the *setup* — that every expected artifact exists, is non-empty, has matching `N`/support, and carries no NaN | an audit run inside the existing verify hook, so a broken setup stops the queue at arm 2 rather than arm 12 |
| **G6** | device gate **warns, does not raise** | a CPU fallback would not fail — it would burn the 12.5 h timeout, twelve times | hard `assert torch.cuda.is_available()` in the preflight; queue refuses to launch |
| **G7** | per-architecture unit tests absent | (obviously — the architectures do not exist yet) | per architecture: instantiates, forward runs, `reg` width `= n_targets × n_params`, `cls` width `= n_targets`, `h_next` shape `== h`, channels divisible by 8, no NaN |

**Build order is G1 → G2 → G7 → G3 → G5 → G4 → G6**, because G4's smoke *uses* the registry and the
tests, and G6 belongs to G4's script.

---

## D. Pre-flight checklist — **every line green before the first 300-lesson arm**

- [ ] **G1** registry implemented; unknown name raises naming the available set
- [ ] **G2** incumbent byte-identical through the registry (fixed seed, `torch.equal`) — **blocker**
- [ ] **G7** six architectures implemented, each with unit tests green
- [ ] channels-divisible-by-8 asserted for candidate (6) — the state-freeze machinery depends on it
- [ ] **G3** queue identity tuple includes `model`; match **and** mismatch paths tested
- [ ] **G5** postflight audit written **and validated against today's completed `truncfullzero_*` arms
      as a positive control** — it must pass on known-good arms before it is trusted to gate anything
- [ ] **G4** smoke: all six train + emit + score at 2 lessons; peak memory and wall-clock recorded
- [ ] per-architecture projected 300-lesson cost **< the 12.5 h timeout with margin**; total projected
      queue time fits the window
- [ ] **G6** CUDA asserted; queue refuses on CPU
- [ ] `ruff check` + `ruff format` + full suite green
- [ ] **05 pre-analysis plan locked** (committed alone, `tools/` for it empty)
- [ ] working tree clean and **committed** — F6 aborts the queue on any HEAD change

## E. Known failure modes carried into the risk register

C-310 (wall-clock estimates from too little evidence — **this program's smoke phase is the mitigation**),
C-303/C-309 (claims outrunning measurement), C-308 (a probe measuring the wrong regime).
