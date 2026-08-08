# 06 — Glossary (2026-08-08)

Primary vocabulary is **`reports/GLOSSARY.md`** (locked; gate, body, gated forecast, crps-all/events/none,
size-ratio, baseline, etc.). Program-specific terms this dossier introduces or leans on:

- **member** — one HydraNet model dir = one `(family, composition, seed)` on the v2 foundation.
- **`concat` pool** — equal-weight posterior sample-pooling: constituents' D×K cubes concatenated on the sample
  axis (`PredictionFrameEnsembleManager._aggregate_prediction_frames`). Pooled draws = 8×`S`.
- **`S`** — per-member total sample count = `n_posterior_samples` (D, MC-dropout passes) × `n_head_samples` (K,
  per-cell family draws). Pooled ensemble width = 8×`S`.
- **roster** — the pre-registered set of 8 `(family, composition, seed)` members (05).
- **foundation config** — the v2 `gated_NB` key block (nb/soft_gate/mse/300L/…); reconstructed at S1.
- **D×K-vs-contract wrinkle** — the config-time `test_ensemble_configs` reads `n_posterior_samples` (=D), but
  the runtime produces D×K; `expected_samples_per_model` must equal the produced D×K (reconciled at S4).
- **light_strider** — the datafactory climatology baseline the ensemble is scored against.
- Families/compositions: **gated_NB** (nb×soft_gate), **th_gated_NB** (nb×threshold_gate τ=0.5), **mixture_NB**
  (mixture_nb×soft_gate). **ZINB** excluded (blooms).

If a needed word is missing from `reports/GLOSSARY.md`, EDIT it there (don't invent a synonym).
