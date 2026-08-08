# tools/ — banked references

## `foundation_gated_nb.py` — the v2 `gated_NB` foundation config (S1 #244)
The canonical, reconstructed **v2 `gated_NB` foundation** `config_hyperparameters.py` — the config the winning
v2 scoreboard run used, which was lost as uncommitted scratch. Regenerated deterministically from
`violet_visitor`'s committed config via the `smoke_mutate` transform (`scratchpad/smoke_mutate.py`), then
validated.

**Key block:** `output_distribution=nb` · `forecast_composition=soft_gate` · `n_head_samples=4` ·
`n_posterior_samples=4` (→ D×K = **16**) · `total_lessons=300` · `loss_reg=mse` · `loss_class=weighted_bce` ·
`loss_class_pos_weight=2.0` · `reg_activation=softplus` · `body_supervision=all` · `rollout_feedback=sample` ·
`bn_recalibrate=True` · log1p on the 3 lr_ targets · `torch_seed=np_seed=42`. Preserves the model-specific
grid / region / data / targets from the source config.

**Validation:**
- Loads + passes `ConfigInitializer` (roundtrip OK: od=nb, comp=soft_gate, K=4, D=4, lessons=300, loss_reg=mse).
- **Already trained + emitted cleanly** in the 2026-08-04 plumbing smoke (EXP-00) at 40 lessons — the 300-lesson
  version differs only in `total_lessons`, so no new smoke is needed for S1.

**How S3 uses it:** this is the shared block propagated to all 8 members; each member overrides only
`(output_distribution, forecast_composition[, gate_threshold], torch_seed, np_seed)` per the LOCKED roster (05):
- gated_NB → `nb` + `soft_gate`, seeds 42/43/44
- th_gated_NB → `nb` + `threshold_gate` + `gate_threshold=0.5`, seeds 45/46
- mixture_NB → `mixture_nb` + `soft_gate`, seeds 42/43/44

**Commit:** per the epic's standing rule, committing this onto `models/violet_visitor/` (and propagating at S3)
is the **user's** action — this file is the banked reference, not a committed model config.
