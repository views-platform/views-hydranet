# Class Intent Contract: ScheduledSamplingMixer

**Status:** Active
**Owner:** Training Strategy
**Last reviewed:** 01.06.2026
**Related ADRs:** ADR-056

---

## 1. Purpose

The `ScheduledSamplingMixer` computes the epsilon schedule for binary scheduled sampling (Bengio et al. 2015). It determines, at each training lesson, the probability that the model's own prediction replaces the ground-truth input at each timestep. This closes the train/inference distribution gap (exposure bias) for autoregressive models.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform the actual input mixing — that logic lives in `_process_sequence()` in `training_engine.py`.
- This class does **not** interact with the model, optimizer, or loss functions.
- This class does **not** manage training state or lesson counting.
- This class is **not** an `nn.Module` — it has no learnable parameters.

---

## 3. Responsibilities and Guarantees

- **Schedule Authority:** Computes a deterministic epsilon value for any given `lesson_idx`. Same inputs produce same outputs.
- **Monotonic Non-Decreasing:** Epsilon never decreases as `lesson_idx` increases — the model is exposed to progressively more of its own predictions.
- **Bounded Output:** `0 ≤ get_epsilon(lesson_idx) ≤ epsilon_max ≤ 1.0` for all inputs.
- **Three Schedules:** Supports `linear`, `inverse_sigmoid`, and `exponential` schedules, each implementing the complement of the corresponding Bengio et al. (2015) §2.4 decay.
- **Warmup Period:** During warmup lessons, epsilon is 0 (for non-linear schedules) or ramps linearly from 0 (for linear schedule).
- **Fail-Loud Construction:** Rejects invalid parameters at `__init__` time with descriptive `ValueError`.

---

## 4. Inputs and Assumptions

- **Construction:** Requires `schedule` (str), `epsilon_max` (float). Optional: `warmup_lessons` (int, required for linear), `k` (float, required for inverse_sigmoid and exponential).
- **Query:** `get_epsilon(lesson_idx: int)` — assumes non-negative integer.
- **Convention:** Epsilon is the probability of using the **model's prediction**, not ground truth. This is the complement of Bengio et al.'s convention (their ε_i is probability of ground truth).

---

## 5. Outputs and Side Effects

- **`get_epsilon(lesson_idx)`:** Returns a float in `[0, epsilon_max]`.
- **Logging:** Logs schedule parameters at construction time via `logger.info`.
- **No side effects:** Pure computation, no state mutation after construction.

---

## 6. Failure Modes and Loudness

- **Invalid schedule name:** Raises `ValueError` listing valid options.
- **epsilon_max out of range:** Raises `ValueError` if not in [0, 1].
- **Exponential k ≥ 1:** Raises `ValueError` — the schedule diverges instead of converging.
- **Inverse-sigmoid k < 1:** Raises `ValueError` at construction — Bengio 2015 inverse-sigmoid decay requires `k ≥ 1` (`k < 1` is a wrong schedule shape; `k = 0` would divide by zero in `get_epsilon`). Symmetric to the exponential guard.
- **Missing k for non-linear schedules:** Not caught by the mixer (caught by `HydraNetConfig.validate_scheduled_sampling_params` instead).

All failures are Fail-Loud at construction time. No silent degradation.

---

## 7. Boundaries and Interactions

- **Constructed by:** `training_loop()` in `training_engine.py`, from config fields `ss_schedule`, `ss_epsilon_max`, `ss_warmup_lessons`, `ss_k`.
- **Called by:** `training_loop()` once per lesson: `epsilon = mixer.get_epsilon(lesson_idx)`.
- **Consumed by:** `_process_sequence()` via the `ss_epsilon` parameter.
- **Config validation:** `HydraNetConfig.validate_scheduled_sampling_params` provides the first line of defense — the mixer sees only valid parameters.
- **Independent of:** Model architecture, loss functions, data pipeline.

---

## 8. Usage Example

```python
from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

mixer = ScheduledSamplingMixer(
    schedule="linear",
    epsilon_max=0.5,
    warmup_lessons=10,
)

for lesson_idx in range(80):
    epsilon = mixer.get_epsilon(lesson_idx)
    # lesson 0: epsilon=0.0 (pure teacher forcing)
    # lesson 5: epsilon=0.25 (halfway through warmup)
    # lesson 10+: epsilon=0.5 (full scheduled sampling)
```

---

## 9. Literature

Bengio, S., Vinyals, O., Jaitly, N. & Shazeer, N. (2015). "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." *NeurIPS 2015*, pp. 1171-1179.

---

## 10. Test Alignment

| Test File | Coverage |
|-----------|----------|
| `tests/test_scheduled_sampling.py` | 24 tests: 7 schedule computation (green), 3 validation (red), 5 config acceptance (green), 4 config rejection (red), 4 integration (green), 1 backward compat (green) |
