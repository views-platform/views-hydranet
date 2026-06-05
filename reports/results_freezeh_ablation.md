# Results — `freeze_h` Channel-Isolation Ablation (C-113)

**Date:** 2026-06-04 (autonomous overnight run)
**Pre-registered plan:** `reports/preanalysis_freezeh_ablation.md` (written *before* execution)
**Verdict:** **H1 CORROBORATED.** The autoregressive divergence is driven by the **prediction→input feedback loop**, *not* the recurrent hidden/cell state. `freeze_h` cannot fix C-113 in any mode — confirmed empirically.

---

## 1. Data

`violet_visitor`, same artifact (`calibration_model_20260603_180015.pt`), n=16, LockedDropout active, only `freeze_h` varied. Primary endpoint **`step-wise/lr_sb_best/CRPS`** (healthy ≈ 0.13; BOUNDED < 1, EXPLODED > 1e3).

| Arm | `hs` upd | `hl` upd | **lr_sb (primary)** | lr_ns | lr_os | verdict |
|------|:-:|:-:|--------------------|----------|---------|---------|
| `hl` (baseline) | ✓ | ✗ | **2.13e17** | 2.78e9 | 54.5 | EXPLODED |
| `all` (**KEY**) | ✗ | ✗ | **5.13e17** | 3.88e11 | 31.3 | EXPLODED |
| `hs` | ✗ | ✓ | **6.98e17** | 5.01e15 | 212.6 | EXPLODED |
| `none` | ✓ | ✓ | **2.86e17** | 6.70e14 | 7929.7 | EXPLODED |

(step-wise CRPS. `lr_os` time-series-wise stays ~0.04 in all arms; the explosion is on `lr_sb`/`lr_ns`.)

**Reproduction control passed:** the `hl` arm reproduced the I2 baseline to full precision — `2.132242375124143e+17` (ablation) `== 2.132242375124143e+17` (I2). The harness is correct and the run is deterministic.

## 2. Reading against the pre-registered decision rules

> §4 risky prediction: H1 predicts the `all` arm **still explodes** despite a fully frozen recurrent state; the recurrent-driven hypothesis predicts it **becomes bounded**.

- **`all` EXPLODED (5.13e17)** — with the *entire* recurrent state (`hs` **and** `hl`) frozen for all 36 steps. The risky prediction is **confirmed**.
- All four arms land within ~half an order of magnitude on `lr_sb` (2.1–7.0 ×10¹⁷). **The recurrent-state configuration is irrelevant to whether — and to how much — the model explodes.**
- `hl` (2.13e17) vs `all` (5.13e17): additionally freezing `hs` does **not** reduce divergence ⇒ the short-term channel (C2) is not necessary.
- `all` (5.13e17): fully frozen state still diverges ⇒ neither C2 nor C3 is necessary.
- `none` (2.86e17) ≈ `hl` (2.13e17): letting `hl` evolve does not materially amplify the endpoint ⇒ C3 is not a dominant amplifier here.

**Conclusion:** the only feedback channel left live in every arm — **C1, the prediction→input loop** — is sufficient to produce the explosion, and it is the driver. No `freeze_h` mode touches C1, so none of them can fix it. The `freeze_h="hl"` setting shipped in all configs is, against this failure mode, **inert**.

## 3. Secondary observation (timing, not magnitude)

The per-step sparklines differ by arm even though the endpoints match:
- `hl` / `all` (cell state frozen): `lr_sb` spikes **only at the final steps** (`…▁▁▃█`).
- `hs` / `none` (cell state `hl` live): additional **mid-horizon** spikes (`…▆▅▁▁▁▂▅█▄…`).

So an evolving cell state changes *when* instability appears across the horizon, but not the terminal blow-up. The input loop guarantees the explosion regardless of state handling. (Consistent with June-3: step-1 normal → late-horizon catastrophe.)

## 4. What this means for the fundamental fix

The fix must target the **input→output map** — the gain the model applies to its own fed-back prediction — not the recurrent gates:

1. **Axis-0 diagnostic — corrected quantity.** Measure the **input→output Jacobian gain** `‖∂pred/∂x_input‖` over one autoregressive step (pink < 1, violet > 1 expected), *not* the recurrent hidden-to-hidden spectral radius. The latter is the wrong operator — `freeze_h` neutralizes the recurrent state and the explosion is unchanged.
2. **Axis-A — retargeted.** Spectral-norm / Lipschitz constraints belong on the **input-to-hidden convs `Wx*` and the U-Net encoder/decoder convs** (the input→output path), not (only) on the recurrent `Wh*` gates.
3. **Axis-B promoted.** Pushforward / GTF training attacks the fed-back-error channel directly — now the best-matched near-term lever.
4. **In-domain feedback-input clamp** (bound the fed-back `x` to the training input domain) is the surviving, magnitude-neutral stabilizer — it bounds the *input copy*, not the emitted output, so it does not fight MCR.
5. **ADR-028 §2 cell-state clamp is pre-falsified.** `freeze_h="hl"`/`"all"` are the extreme of a cell-state clamp; total freezing didn't help, so a softer `clamp(hl, ±C)` won't either.

## 5. Threats / honesty notes

- **Exit code 137 on all four arms — OOM in teardown, not in evaluation.** Each `run.sh` was SIGKILLed *after* the `wandb:` run-summary dumped and synced, during a post-eval `Publishing/Fetching queryset pg_metadata` step (`dmesg`: `Out of memory: Killed process (python) anon-rss ~12 GB`). The metrics were finalized before the kill — proven by the exact-to-16-digits reproduction of the I2 baseline (§1). **The CRPS values are valid.** Separate operational issue: the metadata-publish teardown peaks at ~12 GB RSS and OOMs on this box (note for the pipeline; out of scope here).
- **Single model.** This localizes the channel for `violet`. `blue_stranger` (the other exploder) was not run (per prior instruction). The universal claim across weight configs is not tested, but the mechanism (C1 sufficiency) is weight-agnostic.
- **`all` also freezes `hs`** — it shows C1-alone-with-static-state suffices; the `hs`/`none` arms supply the rest of the 2×2 and agree.

## 6. Disposition

- **C-113** diagnosis sharpened: driver = prediction→input feedback gain > 1 (input→output map), `expm1`-amplified. Recurrent-state control (and `freeze_h`) is the wrong lever.
- **`freeze_h`** confirmed inert against C-113. Candidate for retirement (it also introduces a train/inference mismatch — training evolves the full state, inference freezes `hl` — and blunts long-term memory during rollout). Set `freeze_h="none"` once a real fix lands; do not rely on it as a stabilizer.
- Options catalogue corrected (Axis-0 quantity, Axis-A target) — see `reports/options_catalogue_autoregressive_stability.md`.
