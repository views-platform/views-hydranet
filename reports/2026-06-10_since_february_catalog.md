# Catalog — what's been done since February 2026

**Date:** 2026-06-10 · **Purpose:** a sober stock-take after the C-113 explosion crisis — the git
timeline (what was *built*) and the R&D dossier record (what was *tried*, each classified
**success / regression / neutral**). Curated to decide where to go from a clear head.

---

## TABLE 1 — Git history (high level; *what*, no judgment)

| Period | What was built |
|--------|----------------|
| **Feb** | ADR-compliance / architecture-hardening wave — FeatureScaler (ADR-019), VolumeHandler/Sampler (ADR-007/009), fail-fast config, Symmetric Feature Lifecycle (ADR-046), forensic diagnostics (ADR-045/037), legacy-code isolation |
| **Mar** | Pandas-free output path (ADR-047 PredictionFrame), memory hygiene, typed config consolidation, dtype float32, decoder-head aliasing fix, −2100 LOC dead-code purge |
| **Apr** | **Loss expansion begins** — loss *registry* (`5c3088f`), **LogNormal NLL** (`ffcb746`), **cluster E**: Pareto + **hurdle masking (C-45)** + QS99 (`aba45bc`) |
| **May** | **Tobit** censored loss (ADR-054, `56194d2`), per-target sigma, **scheduled sampling** (ADR-056), **locked dropout** (ADR-057) |
| **Jun** | **C-111 balancer unfreeze** (`8a3c135`) → **C-113 explosion crisis** → root-cause investigation → 3 R&D dossiers → **wandb fix (C-132, `a324e4b`)** → experiments R1–R4 → reconciliation |

Last known-stable lineage: **`e029e63` (2026-06-02)** — one commit *before* the C-111 unfreeze.

---

## TABLE 2 — R&D dossier: each point, classified

✅ Success · 🔴 Regression · ⚪ Neutral (diagnostic / negative-result / process)

| # | Point | Verdict | Why |
|---|-------|:-------:|-----|
| 1 | **C-111 balancer unfreeze** | 🔴 | The confirmed **trigger** of the explosion. Stable for years with it frozen. *The regression to undo.* |
| 2 | **C-113 root-cause investigation** (freeze_h ablation + io_gain + balancer bisect) | ✅ | Found the carrier (prediction→input feedback, not the recurrent state) and the C-111 trigger. Genuine understanding. |
| 3 | **`diagnose_io_gain` tool** | ✅ | Reusable 30-s stability probe. Keeper. |
| 4 | **wandb training-logging fix (C-132)** | ✅ | Real fix — training observable again. Keeper. |
| 5 | **Phase-0 magnitude diagnosis** | ✅ | Localized the collapse to positive cells; ruled out balancer-drowning; showed post-hoc gating doesn't fix. |
| 6 | **`mcr_readout.py` tool** | ✅ | Durable, guarded magnitude/score readout. Keeper. |
| 7 | **R1 — clean SS-off baseline (80L)** | ⚪ | Diagnostic: explosion is **SS-independent**. (Model itself explodes.) |
| 8 | **R2 / R3 — dropout/lr sweep trials** | ⚪ | Diagnostic: explosion persists across dropout/lr; exploding head varies; `os` always healthy. |
| 9 | **Sweep OOM (C-135)** | ⚪ | Operational finding: OOM is a *symptom* of the explosion, not a sweep bug. |
| 10 | **40-lesson collapse baseline** | ⚪ | Diagnostic: mapped the **other** failure mode (collapse, MCR≈0 — stable but useless). |
| 11 | **ADR-057 locked dropout** *(as the explosion fix)* | ⚪ | Falsified — still exploded. Cleared a hypothesis. |
| 12 | **In-domain feedback clamp** | ⚪ | Falsified-as-fix — bounds-then-pins (MCR ~56k). Safety rail, not a cure. |
| 13 | **Freeze-balancer-as-ship-fix** | ⚪ | Falsified — seed-fragile (seed-4 frozen exploded *worse*). |
| 14 | **Arm-1 (hurdle, lognormal)** | ⚪ | Mixed/informative: un-collapsed magnitude one-step ✓ **but** rollout exploded ✗ → revealed magnitude↔rollout **coupling**. |
| 15 | **R4 (hurdle + SS 0.5)** | ⚪ | Negative result: exploded; SS made no difference → **closed the SS dial** (0.25/0.5/1.0 all fail). |

---

## The honest net

- **✅ Real keepers (survive any revert):** the root-cause understanding (C-111 → feedback-loop explosion), two diagnostic tools (`diagnose_io_gain`, `mcr_readout`), the wandb fix (C-132), the Phase-0 diagnosis.
- **🔴 Exactly one true regression:** the **C-111 balancer unfreeze** — the thing that broke a working model. Singular, revertible.
- **⚪ Everything else is neutral** — diagnostics and *negative results* that **bracketed the problem**: SS is exhausted, three "easy" fixes are falsified, both failure modes are mapped, the magnitude↔rollout coupling is understood.

**Conclusion:** not "three weeks of nothing" — **one regression to undo, a handful of genuine keepers, and a now-small, well-mapped problem.** The real fix is narrowed to rollout-training-with-gradient (GTF) or a count head, and we know exactly why the cheap things failed.
