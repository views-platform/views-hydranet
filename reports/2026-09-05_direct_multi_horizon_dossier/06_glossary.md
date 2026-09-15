# 06 — Glossary

Repo-wide vocabulary is `reports/GLOSSARY.md` and **that file governs**. Terms this programme adds:

| term | definition |
|---|---|
| **direct multi-horizon** | Predicting horizon `k` in one forward pass from the origin state, for every `k`, with no prediction fed back. Contrast **recursive**, the current 36-step loop. |
| **forking sequences** | MQRNN's training scheme (C-526): every history step is a fork point, and at each fork the decoder predicts the next K horizons. Prevents a direct head from wasting training positions. |
| **global decoder** | MQRNN's `f(c_t, x_{t+k})` (C-527): one decoder, shared across all horizons, taking the context and horizon-specific covariates. **Not** 36 decoders and **not** 36 models. |
| **horizon covariate** | The input channel(s) carrying which horizon is being predicted. The mechanism by which one shared decoder emits different answers for `k=1` and `k=36`. |
| **the origin state** | `h` after the encoder has digested history, before any forecasting. In a direct head it is the *only* state used; it never evolves. |
| **marginal horizons** | The cost of a direct head: each horizon has its own predictive distribution, with nothing enforcing joint coherence across `k`. |
| **the clamped control** | The comparison baseline from 2026-09-05: `freeze_recurrent='cell'` (ADR-027 §2.1), AP@h18 0.3624 / h36 0.2841 — **not** the unclamped rollout. |
