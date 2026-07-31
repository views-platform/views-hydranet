# 04 — Roadmap (implementation + experiment sequencing)

**Date:** 2026-06-05 · **Status:** seeded · **Dossier:** [00_README](00_README.md)
**Depends on:** [02_design](02_design.md) (the ZITD head), [03_harness_and_invariants](03_harness_and_invariants.md) (the gates).

This turns the design into an **ordered, gated** build. Sequencing principles (from `03 §4`): **one variable at a time**; **cheapest informative experiment first**; **pre-register before running** (`05_analysis_plan`); each step gates the next (tests → fast readout → eval). Nothing trains until the harness pre-flight (`03 §5`) is green.

---

## 1. Phases

| Phase | Goal | Key steps | Exit gate | Depends on |
|------:|------|-----------|-----------|-----------|
| **P0 — Unblock** | remove the literature/implementation blockers | (a) verify/fetch **Gao-Zhu 2024 STZITD** + **Dunn & Smyth** Tweedie-density (`01 §7`); (b) pick the density route (Jiang lower bound vs `tweedie` pkg vs Dunn-Smyth series) | density route chosen + reference in hand | — |
| **P1 — Loss & distribution (no model, no training)** | a correct, tested Tweedie/ZITD NLL + sampler | TDD `utils/tweedie_loss.py` (`ZITDLoss`) and `utils/tweedie_distribution.py` (sample, mean, `P(Y>0)`, quantiles); register in `LOSS_REG_REGISTRY` (OCP) | unit tests green: known-value vs reference, finite grads `ρ∈(1,2)`, exact-zero handling, NaN/Inf guards (`03 §3.1–3.2`) | P0 |
| **P2 — Head & plumbing (behind a flag)** | model can emit ZITD params; baseline untouched | add `π,μ,φ,ρ` decoders + `ModelOutput` fields; `output_distribution` config flag (`"point"` default / `"zitd"`); `volume_handler` derives `E[y]=(1−π)μ`, `P(Y>0)`; `inference` autoregressive feedback = `log1p(E[y] or sample)` (`02 §0.2`); adapt `diagnose_io_gain` to read `E[y]` | **parity gate**: flag off ⇒ baseline byte-identical (`03 §3.6`); full suite + ruff green; CIC synced | P1 |
| **P3 — First experiment (MVP)** | does ZITD bound the runaway *and* score well? | train **violet** (the exploder) with ZITD, **fixed `ρ≈1.5`**, **mean rollout** (simplest cut); fast readout first | `diagnose_io_gain` attractor **in-range** (≲ log 13), then eval: CRPS / MCR / calibration vs `s0` baseline (pre-registered in `05`) | P2 |
| **P4 — Iterate / ablate** | find the right configuration | one variable at a time: learned `ρ`; sampled rollout; per-target `φ,ρ`; classifier-derivation transition; balancer handling (informed by C-111 bisect) | each ablation pre-registered; logged in `07_experiment_log` | P3 |
| **P5 — Decide & graduate** | commit or fall back | if wins: **ADR** (`docs/ADRs/proposed/`), roll to pink/blue, ensemble; if not: postmortem → Path C (DEMM/GPD tail) or renewal (`01 §2`) | ADR adopted *or* documented negative result; dossier archived | P4 |

## 2. Dependency graph

```
P0 unblock ─▶ P1 loss+sampler (tested) ─▶ P2 head+plumbing (flag, parity) ─▶ P3 MVP (violet, fixed ρ, mean rollout)
                                                                                   │
                                                                                   ├─▶ P4 ablations (learned ρ · sampled rollout · per-target φ,ρ · classifier transition)
                                                                                   └─▶ P5 decide → ADR + roll out  |  or → postmortem → Path C / renewal
C-111 bisect (in flight, orthogonal) ───────────────────────────────────────────▶ informs P4 balancer handling
```

## 3. The first experiment (MVP) — why this shape

Smallest change that tests the core claim. **One model** (violet — the clean exploder, so a win is unambiguous), **fixed `ρ`** (avoids the `ρ∈(1,2)` constraint/saturation risk, `02 §7.2`, on the first try), **mean rollout** (deterministic, simplest feedback). Readout is the **retrain-free `diagnose_io_gain`** first (≈30 s: is the free-running `E[y]` in-range?), *then* a full eval for CRPS/MCR/calibration. Expected (pre-register in `05`): attractor in-range **by construction** (sub-exponential link), CRPS ≤ baseline, MCR moving toward 1. Falsifier: if violet's `E[y]` still leaves the data range, the head's link/parameterization is wrong — rethink before scaling.

## 4. Decision points (the open choices from `02 §0.6`)

| Choice | First cut | Decide at | Note |
|--------|-----------|-----------|------|
| `μ` link | **softplus** | P1 | smooth; ReLU is the original §3.2 fallback |
| `ρ` | **fixed ≈1.5** | P3→P4 | learned per-cell `ρ` only after fixed-ρ works |
| autoregressive feedback | **mean** | P3→P4 | sampled rollout = the honest posterior; test after mean works |
| `φ,ρ` sharing | **shared across cells, per-target** | P4 | per-cell `φ,ρ` is the Jiang/Gao-Zhu richness, costs stability |
| classification heads | **keep during transition**, derive `P(Y>0)` in parallel | P4 | flip to derived-only once parity confirmed |
| MTL balancer | keep, `freeze_multitask_balancer` available | P4 | informed by the C-111 bisect (`02 §0.4`, §7.5) |

## 5. Relationship to other work

- **C-111 bisect (orthogonal, in flight):** acute regression vs chronic mis-specification. ZITD proceeds on **either** outcome (`02 §0.4`); the bisect only informs P4's balancer handling.
- **Tobit / ADR-054 (Path A, shipped):** the incumbent likelihood baseline. ZITD must beat *both* the log1p-point baseline (`s0`) **and** Tobit on CRPS/MCR/calibration to justify adoption.
- **Fallbacks (`01 §2`, `02 §9`):** if the Tweedie tail underfits extremes → Path C (DEMM, explicit GPD tail); if Tweedie density proves too unstable → renewal (Türkmen) or ZINB.

## 6. Milestones / definition of done

- **M1 (P1):** `ZITDLoss` + sampler, tests green, registered — *the blocker cleared*.
- **M2 (P2):** model emits ZITD behind a flag; **baseline byte-identical with flag off**; suite green.
- **M3 (P3):** violet ZITD trained; `diagnose_io_gain` in-range; first CRPS/MCR/calibration vs `s0` recorded in `07`.
- **M4 (P4):** best configuration selected via pre-registered ablations.
- **M5 (P5):** ADR proposed (adopt) **or** postmortem (fall back) — and the dossier archived under `reports/archived/`.

> Cost note: P0–P2 are CPU/test work (cheap, no GPU). The first GPU cost is P3 (one ~2h train + 30 s readout + one ~40 min eval). Ablations (P4) are the main GPU spend — one model at a time, n ≤ 64, pre-registered, per `03 §4`.
