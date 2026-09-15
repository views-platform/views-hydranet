# 03 — Harness and invariants

**Audited:** 2026-09-05, against the repo at `c2aca78`. Not templated — every row below was checked.

## A. Invariant taxonomy

### A1. Hard invariants — never break

| invariant | enforced by |
|---|---|
| The stable baseline stays default and **byte-identical when the flag is off** | a parity test this dossier must write; precedent `tests/test_input_noise.py` |
| **No shadow defaults** — `config.get("flag")` with no second argument | AST test, `tests/test_falsification_magic_numbers.py:31` |
| Fail-loud on NaN/Inf | `IntegrityGuardian` — it is what caught #308's explosion |
| Full suite + `ruff` green, tree clean at every story boundary | `tests/test_falsification_repo_clean.py` |
| **The BN-recalibration passes see clean data** | `training_augmentation=False`; **C-328**, four instances |
| Config field count matches the CIC | `tests/test_falsification_loss_param_validation.py:36` — currently **96**; PF's fields raise it |

### A2. Deliberately changed by this program (behind a default-off flag)

The generator's loss gains an adversarial term computed from a **free-running** forward whose
hidden states are compared against the teacher-forced ones. Concretely: training currently performs
one teacher-forced pass per window; PF adds a second, self-fed pass and a discriminator.
**Nobody should defend the single-pass property** — replacing it is the experiment.

### A3. Respect while changing — not targeted, breakable in passing

| thing | why it is fragile here |
|---|---|
| **BatchNorm running statistics** | **C-328, four instances, two of them created by fixes for it.** PF's extra forward *must* run under `_batchnorm_eval` or it writes stats that ship in the artifact. The pushforward branch already does this — copy it, do not re-derive it. |
| **Gradient stability through a free-running unroll** | #308 died at lesson 48; **M61** CREEP, pre-clip norm 133,465 → 9.4e9 over 33 lessons, while `loss_reg` *fell* the whole time. **The loss is blind to it.** |
| The dropout stream | the pushforward comment records that an extra forward consumes dropout draws, so two arms differing only in the flag do not share a stream. Same applies to PF. |
| `multitaskloss` balancer | **C-312**: `log(stds)` goes negative once a task fits and both ADR-014 guards keyed on that sign. Every arm must pin `freeze_multitask_balancer: True` (**the config default is False**). |

## B. The standing harness — what already exists (reuse, do not reinvent)

| mechanism | status | where |
|---|---|---|
| Default-off feature flags | **exists** | `config_initializer.py`, `random_flips` / `pushforward_weight` precedent |
| Reject-if-ignored validators | **exists** | `reject_pushforward_without_a_family` and the three added by #311 |
| Parity / regression gates | **exists** | full suite, 2036 tests |
| Floor gate (**FG-A**, **FG-C**) | **exists, tested in CI, under-invoked** | `scripts/floor_gate.py` — C-299 |
| Potency pre-flight | **exists as a pattern; each dossier writes its own** | `scripts/potency_check.py` — C-324, and C-325 requires it **at a trained checkpoint** |
| Arm post-flight | **exists** | `scripts/arm_postflight.py` |
| Arm identity check | **exists** | `scripts/arm_identity_check.py` |
| Screen verdict | **exists** | `scripts/screen_verdict.py` |
| Weight-hash post-condition | **exists**, inline in launchers | arms must differ before any score is read |
| Recursive process kill | **exists** | C-326 fix — watchdogs previously killed the subshell, not the tree |
| **The gradient stabiliser** | **exists, 9/9 mutations caught, 80 lessons clean** | `ss_feedback_grad_clip` |
| **The self-fed-forward seam** | **exists** | `training_engine.py:780-833` — the pushforward branch |

> **Structural finding, unchanged since #311 and worth repeating: no gate in this repo is
> repo-wide.** Each is opted into by a dossier launcher, and a dossier that forgets one gets no
> warning. #311 was the first dossier since August to invoke `floor_gate` at all. **The launcher
> this dossier writes must wire every row above**, and `04_roadmap` treats that as a story, not a
> detail.

### The seam, concretely

The pushforward branch already does four of the five things PF needs:

```
training_engine.py:780   if pushforward_weight > 0.0 and family is not None and ...
        :786             fed = _family_feedback_log1p(...).detach()   # self-fed field
        :804             pf_in = _attach_static_channels(fed, t1, idx)
        :805             pf_h  = h.detach() if pushforward_detach_state else h
        :806             with _batchnorm_eval(model):                 # C-328 discipline
        :807                 pf_out = model(pf_in, pf_h)
```

PF differs in **what it does with the result**: pushforward scores `pf_out.reg` against `y_{t+2}`;
PF ignores the emission and takes **`pf_out.h_next` as the free-running behaviour sequence**, to be
discriminated against the teacher-forced `h`. It also must **not** `.detach()` the fed field — the
adversarial gradient has to reach the generator.

**This is the single largest cost saving available and the single largest risk.** Reuse the
structure; the `.detach()` on line 793 is exactly the line whose removal reproduces #308.

## C. New harness this program needs — the gaps

1. **The discriminator** — a module, its optimiser, and its own tests. New code, no precedent in
   this repo.
2. **A stability gate that can call a run VOID.** PF is adversarial training on a vehicle with
   known BatchNorm seed-bimodality (**C-184**) and ~20% training variance (**C-119**). A null
   produced by a discriminator that collapsed or saturated is **uninterpretable** and must be
   pre-committed as **VOID, not negative** — the #308 lesson (**C-320**, whose fifth instance was a
   decision rule with no failure branch, so when arm B died no branch could be evaluated).
   The gate needs a *measured* quantity: discriminator accuracy over training, with a
   pre-registered band. At 0.5 it has learned nothing; at 1.0 it has won and the generator gets no
   gradient. **Both extremes are VOID, and both must be logged every lesson, not read afterward.**
3. **The free-running segment length as a config field**, not a constant — **C-85**, whose open debt
   is the hardcoded `0.5` flip probability, and a second instance is not acceptable.
4. **A mutation-tested guard for each described invariant.** **C-303 has twelve occurrences** —
   prose asserting a check the code does not implement, the most habitual defect in the register.
   Every claim in this dossier that *describes* a check must have a test that goes red when the
   check is deleted.

## D. Pre-flight checklist — must be green before the first GPU spend

- [ ] `02_design` written and put through `expert-method-review` — **blocker**
- [ ] `05_analysis_plan` pre-registered, with the stability gate and an explicit **VOID branch**
- [ ] Discriminator implemented + unit-tested; the numerically delicate part first
- [ ] Behind a default-off flag; **baseline byte-identical with it off** (a test, not a claim)
- [ ] `ss_feedback_grad_clip` **wired from day one**, not added after the first explosion
- [ ] `freeze_multitask_balancer: True` pinned on every arm (C-312)
- [ ] Every new forward audited for BatchNorm writes (C-328) and dropout-stream effects
- [ ] Cross-field validators reject every combination the wire cannot honour
- [ ] Mutation testing to exhaustion, **committed before mutating** (a lesson learned the hard way)
- [ ] Adversarial audit **in a clean context by a non-author**
- [ ] Full suite + lint green; CIC field count updated
- [ ] Launcher wires **every** gate in §B — floor gate on the **control arm before any treatment
      arm runs**, at zero extra GPU
- [ ] Potency proven on the arm's own config **and at a trained checkpoint** (C-324/C-325)
- [ ] Every verdict branch fired on synthetic fixtures before real data exists

## E. Rules of engagement

* **One variable at a time.**
* **Pre-register, then run.**
* **Cheap readout before expensive** — the gradient-trajectory probe cost 36 minutes and identified
  #308's failure family; the equivalent here is discriminator accuracy per lesson.
* **Read the control arm's score the moment it lands, not after the queue drains.** That single
  habit would have saved the three days the floor-limited post-mortem cost.
* **Falsifier honesty** — a pre-registered falsifier that fires kills the hypothesis. No ad hoc
  rescue.
