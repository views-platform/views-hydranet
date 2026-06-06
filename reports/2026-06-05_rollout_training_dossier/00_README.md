# Rollout-Training Dossier — Axis B ("train the model the way it is used")

**Opened:** 2026-06-05 · **Status:** design under review · **Owner:** simon
**Branch context:** `development` (C-113 autoregressive-runaway program)

---

## 1. Purpose

`HydraBNUNet06_LSTM4` is trained one-step-ahead but **run 36 steps free-running** at
inference. The prediction→input feedback loop — the exact operator the io-gain
diagnostic identified as the runaway carrier (`scripts/diagnose_io_gain.py`,
`reports/results_freezeh_ablation.md`) — receives **zero gradient** during training
(`training_engine.py:200`, `prev_pred = t1_pred.detach()`). This dossier designs the
fix at the level the literature says it must live — the **training algorithm**, not the
architecture and not an inference-time hard-prior hack. We call it **Axis B**:
*train against the rollout the model will actually perform.* Two graded
implementations are on the table (pushforward, GTF) plus an adversarial maximal
option (Professor Forcing), all governed by one config hyperparameter,
`rollout_horizon` (K).

## 2. Relationship to prior work / ADRs

- **Complements / supersedes parts of:** ADR-056 (scheduled sampling — the existing,
  *detached + Bernoulli-masked + biased* cousin of these methods); ADR-028 §2
  (deterministic-recurrence stabilizers — clamps, the symptom-level fallback);
  ADR-057 (variational/locked dropout — the *posterior* fix, orthogonal to the
  *dynamics* fix designed here).
- **Builds on the C-113 evidence base:** `reports/results_balancer_bisect.md`,
  `reports/preanalysis_balancer_sweep.md` (the C-111 balancer is the *acute*
  trigger; this dossier addresses the *chronic* train/inference mismatch that makes
  the model fragile enough for the balancer to tip it over).
- **Will graduate to:** a proposed ADR (`02_design` → ADR-058 candidate) once the
  method review + a pre-registered experiment clear it.
- **Distinct from:** the ZITD distributional-head dossier
  (`reports/2026-06-05_distributional_head_dossier/`) — that fixes *what the head
  emits* (chronic MCR); this fixes *how the recurrence is trained*. They compose.

## 3. Document index

| # | File | Role | Status |
|---|------|------|--------|
| 00 | `README` | spine (this file) | living |
| 01 | `literature` | the three papers + recurrent-stability neighbours, annotated | drafted 2026-06-05 |
| 02 | `design` | the Axis-B design + the `rollout_horizon` HP + GPU-cost analysis | drafted 2026-06-05 — **reviewed (02b); revised 2026-06-06 (R1–R7 folded, see §10)** |
| 02b | `method_review` | expert-method-review panel verdict + methodological risks | done 2026-06-05 |
| 03 | `harness_and_invariants` | invariants (hard / changed / respect) + standing harness + §3 new-harness gaps + pre-flight checklist | **seeded 2026-06-06** |
| 04 | `roadmap` | gated implementation/experiment sequence | TODO |
| 05 | `analysis_plan` | first pre-registered experiment (B1 MVP, active balancer; resolves falsify P2/P4) | **seeded 2026-06-06** |
| 06 | `glossary` | exposure bias, pushforward, GTF, α, K, zero-stability… | TODO |
| 07 | `experiment_log` | append-only ledger | TODO |

## 4. Harness at a glance

- **Already exists:** `scripts/diagnose_io_gain.py` (retrain-free free-running
  attractor + operator-gain probe — the primary cheap readout); the GPU-enforced
  sweep driver pattern (`views-models/scripts/run_balancer_sweep.sh`);
  `ReproducibilityGate.lock_entropy`.
- **To build (→ `03`):** a training-time rollout-stability invariant test (the
  fed-back operator's gradient is non-zero and finite under the new path); a parity
  guard that the *one-step* training path is byte-identical when `rollout_horizon=1`;
  a per-step-loss monotonicity/curvature check across K.

## 5. Current state & next actions

- [x] Read the three papers in full (`01`).
- [x] Pin the exact current behaviour: BPTT flows through `h`; the prediction
      feedback is **detached**; SS is Bernoulli@`ss_epsilon` (`training_engine.py:178–200`).
- [x] Draft `02_design`.
- [x] **Run `expert-method-review` on `02_design`** (Hochreiter / DL-engineer /
      Sutton / Gneiting / +Shi / +Operational) → `02b`. Verdict: **design sound, layer
      right, B1→B2→B3 ordering right; NOT yet experiment-ready** until the readout
      measures calibration (not just attractor magnitude), the proper-score quarantine
      is operationalised, and the chaos-premise + `seq_len`-vs-36 checks are done.
- [ ] **Fold the 6 review fixes into `02`** before pre-registration: gradient clipping
      (Hochreiter); calibration/sharpness readout + annealed stability weight
      (Gneiting); explicit feedback-space spec (Shi); measured K=12 memory + checkpoint
      plan (DL-engineer); `seq_len`-vs-36 check + a direct-multi-horizon baseline number
      (Sutton); promote the chaos-premise to a B2-blocker.
- [ ] `register-risk` the 6 methodological risks (M-RT1…M-RT6 in `02b` §7).
- [ ] Kill `freeze_h` (Element 1) — `freeze_h="none"`, remove the inference-time
      state-freeze; **gate behind the `rollout_horizon=1` parity guard + golden_hour
      re-eval** (Operational). Grounded in `reports/results_freezeh_ablation.md` (inert).
- [x] `05` pre-registered first experiment (B1 pushforward MVP, **active** balancer, K=12;
      resolves falsify P2/P4 → C-125 note / C-129). Decisions: R6 satisfied; active is the
      primary arm; Axis B sequenced **before** ZITD.
- [x] `03` harness + pre-flight checklist — invariants, standing harness, and the §3 new-harness
      gaps to build before B1 (the last *doc* gate).
- [ ] **Build the §3 harness gaps (TDD)** then **B1 (#78)** — the pushforward path behind
      `rollout_horizon` (K=1 parity), the feedback-gradient-liveness test, annealed weight +
      uncontaminated CRPS, gradient clipping, the full-36 boundedness + calibration readouts.
      *(Training-loop change + GPU — the real build; gated on this checklist going green.)*
- [x] ~~Sequence after the C-111 balancer verdict closes~~ — **verdict in** (sweep: freeze
      seed-fragile, exposure-bias is root); R6 satisfied, proceed.
- **Strongest live dissent (D5):** fixing the point runaway may not fix — and could
  worsen (mean-hedging/blurring) — the chronic MCR/zero-rate calibration problem. Carry
  as a falsifier, not a footnote.

## 6. Conventions

Numbered, dated docs; `00_README` living, the rest point-in-time. git-tracked via
`git add -f` (reports/ is gitignored). On close, move to `reports/archived/`.
Risks go to the register, not here.
