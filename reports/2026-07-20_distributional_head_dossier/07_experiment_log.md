# 07 — Experiment log (append-only; negatives first-class)

Each entry links its pre-registration (`05_analysis_plan`), names the ONE variable, records the readout,
states the verdict against the pre-registered falsifiers (which fired / none), and the decision.

---

## (seed) 2026-07-20 — M0 scaffold
- Pre-reg: `05_analysis_plan` locked.
- Variable: none (setup).
- Readout: dossier created; red TDD tests pending; ground truth verified (`02_design`).
- Verdict: n/a. Decision: proceed to red tests, then M1 smoke on user go-ahead.

## (seed) 2026-07-20 — M0 red tests
- Pre-reg: n/a (TDD scaffolding).
- Variable: none.
- Readout: `tests/test_nb_dist_head.py` — 12 tests, all RED for the right reasons (config rejects
  nb/zinb; `n_head_samples` field missing; head emits 1 ch/target not 2/3; `nb_dist_loss` module
  absent). Contract pinned: per-cell (mu,theta[,pi]) activated head channels; NBDistLoss/ZINBDistLoss
  gradient reaches per-cell theta/pi; nb_to_samples/zinb_to_samples → (…,K) counts; D×K = S.
- Verdict: n/a. Decision: implement M1 (config → head → loss → sampler), turning tests green.

## (seed 42/43/44) 2026-07-24 — ZINB 3×300 (self-zeroed body) — M2 result
- Pre-reg: `05_analysis_plan` §Forecast-composition arms, arm 1 (ZINB self-zeroed = `(1−π)μ`, no ×gate).
- Variable: body family **foundation (global-θ point body) → per-cell ZINB head (D×K sampled)**; the C-212
  `log_prob_zero` NaN fix (`b5a87d5`) was in place, all 3 seeds trained clean (was 2/3 crashing pre-fix).
- Readout (frozen ruler, T=0, count-only, N=170430):

  | target | crps-all ZINB | foundation | white_ranger | size-ratio |
  |--------|--------------:|-----------:|-------------:|-----------:|
  | sb | 0.141 | 0.137 | 0.191 | 0.02 → **0.25** |
  | ns | 0.084 | 0.083 | 0.088 | (timid → moved) |
  | os | 0.040 | 0.028 | 0.030 | — |

  Seed-stable across 42/43/44. The **timid body is fixed** (size-ratio 0.02→0.25). crps-all is at
  parity with the foundation on sb/ns and beats white_ranger on sb/ns; **os is the localized weak
  spot** (0.040 > both foundation 0.028 and WR 0.030).
- Verdict vs falsifiers: **none fired.** F1 no θ/π collapse, no seed-instability, sampler non-degenerate.
  F2 not triggered (crps-all not worse on ≥2 targets vs foundation — sb/ns parity). F3 deferred (M3
  validation). **NOT a KILL.** Decision: keep ZINB as the crps-all/magnitude front-runner; os under-fit
  noted (localized, not global). Proceed to the composition-arm head-to-head.

## (seed 42/43/44) 2026-07-24 — gated_NB (soft cls-gate re-score) — composition arm 2
- Pre-reg: `05_analysis_plan` §arm 2 (forecast = NB body × cls occurrence gate; pure re-score of the
  preserved nb 3×300 cubes, ZERO extra GPU).
- Variable: composition **ZINB self-zeroed → gated_NB (nb body × gate)** — same underlying head family
  question, different zero mechanism (external gate vs structural π).
- Readout (frozen ruler, T=0):

  | target | crps-all | AP |
  |--------|---------:|---:|
  | sb | 0.159 | **0.447** |
  | ns | 0.091 | **0.385** |
  | os | 0.046 | **0.259** |

  ZINB wins **crps-all** on all 3 targets; gated_NB wins **AP** (locality/occurrence) on all 3. A clean
  **magnitude (ZINB) vs locality (gated_NB) tradeoff** — the sharp cls gate localizes better; the
  structural-π body scores the count magnitude better.
- Verdict: no falsifier; both arms viable on different axes. Decision: the tradeoff motivates testing
  whether the two axes **fuse** → build + score gated_ZINBcore (arm 4).

## (seed 44) 2026-07-24 — gated_ZINBcore — composition arm 4 — **NEGATIVE (hypothesis falsified)**
- Pre-reg: ADR-068 + `05_analysis_plan` composition arms. **Pre-committed prediction:** gated_ZINBcore
  (ZINB's π-stripped NB core × external cls gate) fuses **ZINB's crps-all** (better body magnitude) with
  **gated_NB's AP** (sharp gate) → best-of-both. Code: `sample_core` (drops π) + `emit_family_core` flag.
- Variable: zero mechanism for the ZINB **core** — **structural π → external cls gate** (π dropped, not
  stacked; avoids the `(1−π)μ × gate` double-count). Single-seed proof (seed 44), same basis as the
  ZINB/gated_NB comparison.
- Readout (frozen ruler, T=0):

  | target | crps-all | (ZINB) | (WR) | AP | crps-none |
  |--------|---------:|-------:|-----:|---:|----------:|
  | sb | **0.981** | 0.141 | 0.191 | 0.438 | 0.870 |
  | ns | **0.488** | 0.084 | 0.088 | 0.389 | 0.415 |
  | os | **0.462** | 0.040 | 0.030 | 0.256 | 0.440 |

- Verdict vs falsifiers: **F2 FIRED** (crps-all worse than everything on all 3 targets — 5–15× ZINB, and
  worse than the white_ranger baseline). It DID take gated_NB's AP (0.438 ≈ 0.447, same gate, expected)
  but crps-all **exploded**. **Worst-of-both, not best-of-both.** (Precision: the frozen ruler's crps-all
  is **gate-independent** — computed on the raw body samples; the gate feeds only AP/Brier. So 0.981 is the
  *ungated* ZINB-core body, crps-none-dominated; a soft external gate cannot touch crps-all by construction.
  See `postmortem_gated_zinbcore`.) Decision: **KILL the arm.** Do NOT spend
  3-seed compute — the failure is structural (see postmortem), so seed variation cannot rescue it. → see
  `postmortem_gated_zinbcore.md`.

## (seed 42/43/44) 2026-07-24 — th_gated_NB — composition arm 3 — **POSITIVE (clears; strongest all-round)**
- Pre-reg: `05_analysis_plan` §arm 3 (hard cls-gate threshold: full nb body where `gate ≥ τ`, zeroed
  where `gate < τ`; two FIXED a-priori τ = 0.5 and per-target base rate). Gate to run was met: the ZINB
  vs gated_NB split confirmed a real magnitude-vs-locality tradeoff (arm not moot).
- Enabling work: the frozen lodestar scorer computes crps-all on the raw body samples (gate-independent),
  so a hard-threshold body composition had to be **added** to the ruler. Done TDD, **byte-identical** for
  all existing arms (HEAD vs extended scorer max|Δ| = 0.00e+00), `--selftest` re-frozen (`apply_threshold_gate`).
- Variable: composition **soft gated_NB → hard th_gated_NB** (τ zeros the body); pure re-score of the
  preserved nb 3×300 cubes, ZERO GPU. τ pre-registered a-priori (no Goodhart).
- Readout (frozen ruler, T=0, mean of seeds 42/43/44; per-seed rock-stable):

  | target | arm | crps-all | AP | crps-events | crps-none |
  |--------|-----|---------:|---:|------------:|----------:|
  | sb | gated_NB soft | 0.159 | 0.447 | 15.62 | 0.038 |
  |    | **th_gated_NB@0.5** | **0.139** | 0.447 | 15.73 | **0.017** |
  |    | ZINB (banked) | 0.141 | — | — | 0.042 |
  | ns | gated_NB soft | 0.091 | 0.385 | 22.20 | 0.014 |
  |    | **th_gated_NB@0.5** | **0.080** | 0.385 | 22.35 | **0.003** |
  |    | ZINB (banked) | 0.084 | — | — | — |
  | os | gated_NB soft | 0.046 | 0.259 | 6.24 | 0.020 |
  |    | **th_gated_NB@0.5** | **0.031** | 0.259 | 6.30 | **0.005** |
  |    | ZINB (banked) | 0.040 / WR 0.030 | — | — | — |

  th50 beats soft gated_NB on crps-all on **all 3 seeds × 3 targets**; **≥ ZINB** (ties sb, wins ns+os —
  ZINB's weak targets); keeps gated_NB's **AP** (0.447/0.385/0.259 ≫ WR 0.334/0.223/0.158). The fusion
  gated_ZINBcore failed at. τ=**baserate** ≈ no-op (base rates ~0.4–0.8% too low to zero anything).
- Verdict vs falsifiers: **none fired.** The crps-all win is entirely **crps-none** (−55…−78%, confident
  zeros on true-zero cells); **crps-events flat-to-+1%** (few real events below τ=0.5). Honest limits:
  th50 is better at aggregate score + occurrence, **not** at sizing — **size-ratio DROPS** (sb 0.29→0.13),
  so the prereg's "th_gated wins size-ratio" expectation is **falsified** (the body was already unshrunk
  in this gate-independent ruler; thresholding only adds zeros, incl. on false-negative events). **os still
  loses crps-all to white_ranger** by a hair (0.0305 vs 0.0299) and on crps-events. Decision: th_gated_NB
  @ τ=0.5 is the **strongest all-round arm** (best crps-all + AP, seed-stable) — candidate for the M3
  validation-partition graduation. Its edge is decisive occurrence, not magnitude.

