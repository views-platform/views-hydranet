# Dense-MSE mechanics grid — pre-registration (predictions before looking) · 2026-07-17

**Goal (exploratory, understanding — not a hypothesis test):** isolate the effect of reg_activation and
loss-space on the ALL-CELL plain point body (output_distribution='standard', mse/count_mean, wBCE pw2,
observed target, no cap). NO pinball / hurdle / NB / distributional head. Deliverable = forensic plots +
bulk_score reads + my evaluation, for the user to review.

**Design:** OFAT-ish 2x2x2 = act{softplus,relu} x space{log=mse, count=count_mean} x seed{42,43} = 8 runs,
40L, train+eval, priogrid_id. (winsor-cap + running-average target factors = the code-needed follow-up,
built tomorrow.)

**Pre-registered expectations (so tomorrow's read is honest):**
- **relu + log (mse):** DEAD body — reproduces the P0 dense-mse anchor (ratio_med ~0.000; 97% of positive
  cells emit exactly 0; dead-ReLU C-178). CRPS artificially low (dead wins the 99.7% zeros).
- **softplus + log (mse):** ALIVE but TIMID — all-cell MSE pulls toward the ~0 mean of a 99.7%-zero target,
  so ratio_med low (predict ~0.02-0.13; possibly LOWER than tonight's positives-only A0p because the
  all-cell zero-pull is stronger than the hurdle-masked body). The honest revived front-runner.
- **count-space (count_mean), either act:** DIVERGES — exp-gradient amplification (verified unstable in the
  production ConvLSTM, project_count_mean_fails_oos). Expect NaN/Inf, garbage predictions, or huge CRPS.
  The landmine demo (runs last; per-run timeout guards a hang).
- **seed 42 vs 43:** reproducibility check (not the production BatchNorm bimodality — that was the floor).
- **Throughout:** within2x / spearman LOW (the body can't rank cells — amount-ceiling WALL).

**What would surprise me (worth flagging in the read):** softplus+log NOT timid (ratio_med >0.3) without
CRPS blow-up; count_mean NOT diverging; a large seed split.

## AMENDMENT (02:15) — eval crash → TRAIN-ONLY / in-sample
First launch: sp_log_s42 TRAINED fine (~32min) then CRASHED at eval with "Input contains infinity" — the
alive all-cell 'standard' softplus body blooms in the 36-step rollout eval (C-113; 'standard' emit has no
clamp; tonight's hurdle runs survived only because the gate suppresses the bloom). Per user scope
([[feedback_no_clamp_no_rollout_scope]]: step-0/in-sample, feedback-clamp REJECTED), switched the grid to
TRAIN-ONLY (-t, no -e). Read the body IN-SAMPLE from training forensics + the saved artifact (I can probe
in-sample T=0 body magnitude per arm in the morning from the artifacts). No rollout, no clamp. OOS T=0
ratio_med (needs a working eval — a 1-step eval or the user's preferred method) is a tomorrow-with-user
decision. relu arms still train (dead body visible in-sample); count arms expected to NaN in training
(the divergence, visible in the loss curve).
