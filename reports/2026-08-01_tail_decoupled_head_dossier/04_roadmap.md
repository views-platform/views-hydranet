# 04 — Roadmap (phased, gated)

Each milestone gates the next. Cheap-before-expensive; no 3-seed run until the pre-flight checklist
(`03 §D`) is green and `05` is pre-registered.

| M | Milestone | Gate to pass | Output |
|---|-----------|--------------|--------|
| **M0** | Design + method review + risk intake | `02` reviewed (`expert-method-review`, or accept the `08` panel); C-MR1..C-MR6 → `register-risk` | reviewed `02`, registered risks |
| **M1** | Build the mixture family (TDD) | **C1** NLL numerics + ordered-means unit-tests green (delicate part first); `@register`; **C4** parity anchor (baseline byte-identical when off); full suite + ruff green | `distributions/mixture_nb.py` + tests |
| **M2** | Sampler + inference wiring | **C2** mixture sampler + determinism test; D×K cube contract intact; `_emit_magnitude`/`_sample_feedback` handle the family | sampler + emit path + tests |
| **M3** | Ruler extension | **C3** stratified-proper column + GW test added to `score_v2_horizons.py`; h=1==T=0 anchor still holds; GW stat unit-tested | scorer + tests |
| **M4** | Single-tile overfit smoke (fast readout) | can fit one known heavy-conflict cell (distinguishes "can't train" from "no tail signal"); `record_params` shows a live component 2 | smoke result in `07` |
| **M5** | Pre-register | `05` written (hypothesis + falsifiers + GW decision rule) **before** the run | `05_analysis_plan.md` |
| **M6** | 3 seeds × 300 lessons on v2 foundation | hardened driver (one-at-a-time, trap-restore, inline-score-delete-cube); `crps_none` bloom guardrail live | artifacts + scores |
| **M7** | Score + verdict | GW conditional test on the high-risk stratum vs falsifiers; `log` (incl. negative postmortem) | `07` verdict |
| **M8** | Disposition | **positive** → `promote` to a proposed ADR; **negative** → close the magnitude axis, update the amount-ceiling record | ADR *or* closed axis |

**Decision points:**
- After **M4**: if the single-tile smoke can't fit a heavy cell *and* `record_params` shows the head
  can't route to component 2, stop — that's a build/identifiability failure, not a science result; fix
  before spending the 3-seed run.
- After **M7**: the binary verdict (within-family vs real) is the ship-relevant output either way.

**Dependency note:** the training driver lives in `views-models` (never committed from here). M6 is
GPU-gated and runs there; this dossier holds the design, harness, prereg, and scores.
