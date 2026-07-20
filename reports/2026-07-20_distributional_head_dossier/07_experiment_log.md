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

