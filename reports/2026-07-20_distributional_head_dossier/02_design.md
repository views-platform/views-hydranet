# 02 — Design (per-cell NB/ZINB sampleable head)

> **PIVOT (2026-07-20) — clean-architecture subsystem.** The authority for the design is now **ADR-067**
> (`docs/ADRs/proposed/067_distribution_family_subsystem.md`) and **Epic #167** (stories A-S1…A-S13). The
> earlier "just add `nb`/`zinb` to the `output_distribution` valid-set + a `utils/nb_dist_loss.py`" sketch
> below is **superseded** by a proper `views_hydranet/distributions/` subsystem: a single
> `DistributionFamily` abstraction, an explicit `name→lazy-factory` registry with one `resolve_family()`
> dispatch seam (strangler-fig — legacy families untouched), NB + ZINB built on a shared `NBCore`
> (composition), a **seeded** D×K sampler with `posterior_S = D*K` bounded + disk-preflighted, and the
> `n_head_samples` config knob. Hardened by the v1 `/expert-code-review` (explicit map not decorator; one
> ABC not four Protocols; seeded generator; D×K memory guard). The **ground truth, head math, loss,
> sampler semantics, and config behaviour below still hold** — only the *module layout* changed (a
> `distributions/` package instead of `utils/nb_dist_loss.py`, dispatch via `resolve_family` instead of
> new switch branches). Read ADR-067 for the current structure.

## Verified ground truth (read-only investigation, 2026-07-20)

- **Emit path** `views_hydranet/utils/hydranet_inference.py:207-237` `_emit_magnitude` — every branch
  returns a mean; `dense_nb` → `log1p(mu)` (`:224-227`, ignores θ); `hurdle_nb` → `log1p(E[y])` via
  `hurdle_nb.py:28-48`. No distribution object at inference.
- **θ global scalar** — learnable `nn.Parameter` per target `dense_nb_loss.py:58` / `truncated_nb_loss.py:59`;
  sidecar snapshot `train_model.py:107-111`; loaded as attr `model_artifact_fetcher.py:117`; broadcast
  `[1,C,1,1]` `hydranet_inference.py:178`.
- **Head width** `architectures/HydraBNrecurrentUnet_06_LSTM4.py:112`
  `reg_out_ch = n_quantiles if _is_quantile else output_channels` (=1); reg convs
  `dec_conv4_head{1,2,3}_reg = Conv2d(base, reg_out_ch, ...)` `:164/:202/:240`. **Quantile head precedent**:
  widen ONLY reg heads, keep `output_channels`=1 → AR invariant `input_channels == 3*output_channels +
  static` (`config_initializer.py:277`) untouched (`:94-99`).
- **Sample cube** `hydranet_inference.py:515-534` `[T,H,W,C,S]`, `S = n_posterior_samples`
  (`config_initializer.py:136`). Two paths: dropout loop `:592-609`; quantile Path A `:551-590` (one pass,
  fill S analytically, **bypass** dropout). `(N,S)` invariant guarded `prediction_frame_assembler.py:183-188`
  + `test_prediction_frame_assembler.py:146`.
- **Config precedents**: `body_mask` clean field — tuple `:25`, field `:108`, `field_validator` `:405-414`,
  `model_validator` `:658-681`, clean-break `reject_retired_hurdle_knobs` `:625-640`; registry validator
  `validate_loss_reg_params` `:578-593`.

## The head (architecture)
Each reg head emits **P params/target** (not 1 mean), following the quantile decoupling:
- `nb` → **2 ch/target**: `mu` (softplus), `theta` (softplus, per-cell dispersion).
- `zinb` → **3 ch/target**: `mu`, `theta`, `pi` (sigmoid, per-cell zero-inflation).
`reg_out_ch = n_params_per_target`; `output_channels` stays 1 (AR feedback). Feedback channel = per-cell
mean E[y] (`(1-pi)*mu` for ZINB) via a new `_emit_magnitude` nb/zinb branch (mirrors `dense_nb` but from
the emitted per-cell `mu`).

## The loss
New per-cell NB/ZINB NLL reading θ (and π) **from the head channels**, not a global scalar (the key
departure from `DenseNBLoss`). Register in `LOSS_REG_REGISTRY`; wire per-target `choose_loss`
(`utils.py:181-191`). θ no longer sidecar-persisted for these heads (emitted).

## The sampler (D×K)
New branch in `generate_posterior_samples` beside quantile Path A. Keep the dropout loop (D =
`n_posterior_samples` passes, epistemic); in each pass the head emits per-cell params → build
`torch.distributions.NegativeBinomial` (× `Bernoulli(pi)` zero-mask for ZINB) → draw K =
`n_head_samples` samples/cell → write S columns `d*K:(d+1)*K`. S = D×K, log1p space, `"S"` axis preserved.

## Config (NO rename — deferred to Epic B)
- Add `nb`, `zinb` to `output_distribution` valid set (`config_initializer.py:391-398`).
- Add `n_head_samples: int = Field(default=1, ge=1)` (K). S = D×K.
- `model_validator`: `n_head_samples>1` requires `output_distribution ∈ {nb,zinb}` (fail-loud); legacy
  K=1 → byte-identical.
- `dense_nb`/`hurdle_nb` keep their mean-emit meaning (distinct legacy values).

## Explicitly out of scope
- The `output_distribution`→`body` rename + pretty names (deferred to Epic B; cosmetic).
- Quantile head (dropped — inference cost).
- Multi-step rollout quality / the bloom (T=0 only; M2 rollout sanity separate).
