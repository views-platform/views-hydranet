# 04 — Roadmap

**Written:** 2026-09-05. Phased and **gated**: no phase starts until the prior gate is green.

```
S0 dossier + harness audit  ──DONE
   │
S1 design + method review ──────► GATE: a seated panel has ruled on F1-F6
   │
S2 pre-registration ────────────► GATE: falsifiers, stability gate and a VOID branch committed
   │
S3 implement (flag default-off) ► GATE: byte-identical when off, proven by a test
   │
S4 adversarial audit ───────────► GATE: mutation testing to exhaustion, clean-context non-author
   │
S5 smoke + potency ─────────────► GATE: potent on the arm's own config AND at a trained checkpoint
   │
S6 screen: control vs PF ───────► GATE: floor gate on the CONTROL before the treatment arm runs
   │
S7 score + locked rule
   │
S8 disposition
```

| S | title | substance |
|---|---|---|
| **S0** | Dossier + harness audit | **done** — this directory |
| **S1** | Design + `expert-method-review` | settle F1–F6 in `02_design`. **The panel is seated before pre-registration, not after** — the seam the skill defines. |
| **S2** | Pre-registration | `05_analysis_plan`: hypothesis, the one variable, skepticism ledger, pre-registered predictions, falsifiers, the **stability gate**, and an explicit **VOID branch**. C-320's fifth instance was a rule with no failure branch; this one gets one. |
| **S3** | Implement | the flag, the discriminator, its optimiser, the config fields and their reject-if-ignored validators. `ss_feedback_grad_clip` wired **from day one**. CIC field count 96 → N. |
| **S4** | Adversarial audit | mutation testing to exhaustion, **committed before mutating**; `/falsify guard`; review in a clean context by a non-author. |
| **S5** | Smoke + potency | 2-lesson smoke; potency on the arm's own config **and at a trained checkpoint** (C-324/C-325); discriminator accuracy logged per lesson from the first smoke. |
| **S6** | Screen | control vs PF, 300 lessons, n=1, one seed. ~4 h/arm estimated against pushforward's measured 3.2 h. Launcher wires **every** gate in `03_harness` §B. |
| **S7** | Score + locked rule | `screen_verdict.py`; weight-hash post-condition read **first**; `AP@h18`, `sb`, 13 origins. |
| **S8** | Disposition | ledger M-entry, register, and either a proposed ADR or an honest close. **A null closes nothing** (C-307). |

## Decision points

* **After S1** — the panel may rule PF not worth building on this vehicle. That is a legitimate
  outcome and costs one review instead of ~8 GPU-hours.
* **After S5** — if the knob is not potent at a trained checkpoint, stop. C-325 exists because two
  #308 mechanism tests measured a network at initialisation and were recorded as ruling a mechanism
  out; the arm's gradient did explode, 33 lessons later.
* **After S6** — if the stability gate is breached, the run is **VOID** and is re-run or abandoned,
  **not** recorded as a negative.

## Budget

One screen, two arms, 300 lessons, n=1. Not a confirmation. A 4-seed run is a **separate** decision
with a pre-registered trigger, per the #311 precedent where the trigger was Δ ≥ +0.02 and the
measured Δ was −0.196, so seeds were correctly not bought.
