# Class Intent Contract: BodySupervisionResolver (`views_hydranet/utils/body_supervision.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-07-28
**Related ADRs:** ADR-065 + amendment 2026-07-28 (body_supervision window), ADR-046 (Transformations
vs Derivations), ADR-008 (Error Propagation), ADR-009 (Boundary Contracts & Configuration Validation)

---

## 1. Purpose

> Turn the validated `body_supervision` config (`all` / `active` + `onset_lead` + `cessation_lag`)
> into the exact set of cell-timesteps the **point body** (MSE/MAE and other non-latent regression
> losses) is supervised on during training.

It is the single "resolve once" mechanism the training loop calls instead of an inline
`if mode == …` ladder. The config is validated at the boundary
(`HydraNetConfig.validate_body_supervision` + `validate_body_supervision_latent`); this module is the
pure function it maps to. (Renamed 2026-07-28 from the retired `body_mask.py`.)

---

## 2. Public surface

- `resolve_body_supervision(onset_lead, cessation_lag, event_threshold) -> (window -> BoolTensor)`
  for the `active` mode. `body_supervision='all'` takes the loop's own unmasked branch.
- `_active_window_mask(window, threshold)` — the saturated-radii endpoint (active-anywhere).
- `event_threshold_from_config(config)` — sole authority for "what counts as an event" (the binary
  derivation threshold; ADR-046 / C-195), never a literal in the loop.

## 3. Responsibilities & guarantees

- **Asymmetric temporal dilation.** A timestep `t` in a cell is supervised iff an active month `t'`
  (`y_{t'} > event_threshold`) exists with `t − cessation_lag ≤ t' ≤ t + onset_lead`. `onset_lead`
  reaches to future active months (the onset run-up); `cessation_lag` reaches to past active months
  (the cessation decay). Pure, deterministic, shape-preserving `[B,T,n_reg,H,W] → bool[same]`.
- **Byte-identical endpoints (retirement gate).** `active,0,0` ≡ the old per-step-positive
  (`pos_cells`); `active,≥T−1,≥T−1` ≡ the old active-anywhere broadcast (`pos_timelines`);
  `all` ≡ the old all-cell foundation (`none`). Pinned in `tests/test_body_supervision.py` and end-
  to-end (masked loss value) in `tests/test_body_supervision_contract.py`.
- **Point-body only.** Applies under a non-latent loss (`not use_latent`); a latent likelihood owns
  its own zero handling, so `active` there is a silent no-op and is REJECTED loud at config validation.

## 4. Failure modes

- Unknown `body_supervision` value / negative radius → fail loud at the config boundary (ADR-009).
- The retired `body_mask` / `hurdle_threshold` / `hurdle_mask_mode` keys → fail loud with the
  migration hint (never a silent shim).
- Ambiguous binary-derivation thresholds → `event_threshold_from_config` raises (ADR-008).

## 5. Test alignment

- `tests/test_body_supervision.py` — resolver truth-table (asymmetric dilation), endpoint parity,
  threshold authority, config validators.
- `tests/test_body_supervision_contract.py` — end-to-end masked-loss-value contract + the all-cell
  byte-identical proof.
