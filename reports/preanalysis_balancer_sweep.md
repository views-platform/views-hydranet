# Pre-Analysis Plan — Balancer × Seed sweep (C-124 / C-113 acute, multi-seed confirmation)

**Date:** 2026-06-05 (pre-registered *before* the sweep) · **Risk:** C-124, C-113, C-111
**Upgrades:** the single-seed Stage-1 result in `preanalysis_balancer_benefit.md` (frozen healthy, active exploded, violet/42) to a **multi-seed factorial**, killing the single-seed caveat.

---

## 1. Hypothesis
**H:** The C-111 active MultiTaskLoss balancer is a **reliable, seed-independent blow-up lever**: across seeds, *active* → out-of-range free-running attractor (runaway), *frozen* → in-range (stable). If so, `freeze_multitask_balancer=True` is the robust acute fix for C-113 and the learnable balancer does not earn its place.

## 2. Design (the ONE base config; vary only seed × balancer)
**3 × 2 factorial = 6 from-scratch trains**, all on the **violet base config** (isolates the two variables; not confounded by pink/blue's other config diffs):
- **seed** ∈ {42, 4, 99} (the project's three seeds; set `np_seed = torch_seed = seed`)
- **`freeze_multitask_balancer`** ∈ {False (active), True (frozen)}

## 3. Readout
Per fresh artifact: **`scripts/diagnose_io_gain.py`** free-running attractor (retrain-free, ~30 s) — in-range (≲ log 13) vs out-of-range (≳ log 20). This is the validated proxy (matched the real eval for pink/violet). Optional: a confirmatory `--evaluate` on the frozen cells for real CRPS.

## 4. Pre-registered predictions
| | seed 42 | seed 4 | seed 99 |
|---|---|---|---|
| **active** | out-of-range | out-of-range | out-of-range |
| **frozen** | in-range | in-range | in-range |
(3/3 active explode; 3/3 frozen stable.)

## 5. Falsifiers (pre-committed)
- **F1 — active not always explosive:** ≥1 active cell is in-range ⇒ the explosion is **seed-conditional**, not a deterministic balancer effect ⇒ "freeze always" is over-strong; the active balancer is sometimes fine ⇒ nuance the fix (freeze as safe default, active opt-in).
- **F2 — frozen not always stable:** ≥1 frozen cell is out-of-range ⇒ **freezing is not sufficient** ⇒ the balancer is not the sole cause ⇒ re-open the diagnosis (other driver).
- **F3 — diagnostic vs reality:** if a confirmatory eval on a frozen cell explodes despite an in-range attractor ⇒ the proxy is unfaithful for that artifact ⇒ trust the eval.

## 6. Method
- GPU-enforced driver (CUDA pre-flight gate + on-GPU PID verification — no silent CPU fallback, C-115); one model at a time (~80 min each → ~8 h total); config trap-restored after each cell and at exit.
- Capture the newest artifact per cell; run `diagnose_io_gain` on it; record to `logs/balancer_sweep_RESULTS.txt`.
- Seeds set via `np_seed`/`torch_seed`; balancer via `freeze_multitask_balancer`.

## 7. Decision rules
- **3/3 active explode + 3/3 frozen in-range (predicted):** robust confirmation ⇒ **ship `freeze_multitask_balancer=True`** as the C-113 acute fix across the golden_hour models; **resolve C-124** ("balancer does not earn its place; equal weighting is the stable, no-worse choice"); close the C-113 acute track (chronic ZITD work continues separately).
- **F1:** freeze is a safe default but document the seed-conditionality; consider active-opt-in.
- **F2:** re-open — freezing insufficient; the balancer is not the (sole) cause.
- Any outcome logged; negatives recorded plainly.

---

## RESULT (2026-06-05) — **F2 FIRED**: freezing is seed-fragile → re-open

Sweep `20260605_132248`, **5/6 cells** (`seed99_frozen` crashed: `CUDA error: unspecified launch failure` — almost certainly the lid-suspend `nvidia_uvm` wedge, not a model failure; train_exit=1, no artifact). Readout = `diagnose_io_gain` free-running attractor on **synthetic** seeds (noisy — intra-cell verdicts flip across U[0,s] seeds, so read the `inf`/out-of-range explosions, not the borderline rows).

| | active | frozen |
|---|---|---|
| **seed 42** (bisect, reused) | PATHOLOGICAL ~log16 | healthy ~log4–5 |
| **seed 4** | PATHOLOGICAL (≤~log14.5; 2e5–7e5) | **PATHOLOGICAL → `inf`** (step48 ~9.5k in log) |
| **seed 99** | **PATHOLOGICAL → `inf`** | — (CUDA crash, no artifact) |

- **Active: 3/3 explode** — prediction held; the active balancer reliably destabilises.
- **Frozen: NOT 3/3 stable** — seed 42 healthy, but **`seed4_frozen` exploded to `inf`** (*worse* than `seed4_active`), seed 99 missing. **F2 fired.**

**Verdict (per §5/§7): F2 — freezing is not sufficient; the balancer is not the sole cause ⇒ re-open the diagnosis.** "Ship `freeze_multitask_balancer=True` as the robust acute fix" is **falsified** — freezing is seed-fragile and on seed 4 actively harmful. Corroborates the chronic train/inference **exposure-bias** mismatch (Axis-B rollout training, register **C-125**; `reports/2026-06-05_rollout_training_dossier/`) as the root, with the balancer one trigger among seeds. **C-124 stays OPEN** (does *not* resolve as "freeze ships").

**Follow-ups:** (a) optionally re-run `seed99_frozen` after clearing the CUDA/`nvidia_uvm` wedge to complete the factorial — *the F2 conclusion is already robust without it*; (b) the durable fix is the rollout-training program, sequenced per C-125/C-126.
