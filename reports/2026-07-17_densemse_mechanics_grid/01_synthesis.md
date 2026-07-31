# Dense-MSE mechanics grid — synthesis + evaluation · 2026-07-17

Exploratory understanding grid (user-requested): all-cell PLAIN point body (no pinball / hurdle / NB /
distributional head), `output_distribution='standard'`, wBCE gate pw2, observed target, no cap. Flipped
`reg_activation {softplus, relu}` × space `{log = mse, count = count_mean}` × seed `{42,43}` = 8 runs, 40L.
Pre-registration + amendment: `00_prereg.md`. Read IN-SAMPLE (teacher-forced 1-step, month 440→441, no
36-step rollout) via `tools/insample_probe.py` — because (a) the alive all-cell body blooms in the rollout
eval and (b) the auto-forensic biopsies didn't regenerate. Metric per arm/target: `frac_alive` (frac of
positive-truth cells with E[y]>0.1), `ratio_med` (median E[y]/truth on positives), `ey_max` (divergence
check). E[y] = expm1(reg) (standard emit = identity). Full table: `results/insample_probe.txt`; plot:
`results/insample_probe_frac_alive.png`.

## Headline finding (from the RUN itself, before any probe)
**The alive all-cell `standard` body is UNUSABLE out-of-sample as-is.** sp_log_s42 trained fine (40L) then
**crashed the 36-step rollout eval with "Input contains infinity"** — the C-113 bloom (`expm1` of a
free-running log field → inf). The `standard` emit has NO clamp; tonight's hurdle runs only survived because
the gate suppresses most cells. The dead-ReLU front-runner "worked" precisely *because* it was dead (0 → no
bloom). ⇒ an ungated all-cell point body needs a gate/hurdle **or** a clamp to be OOS-viable. This is *why*
the hurdle structure exists. (We did NOT paper over it with `feedback_clamp` — user rejected that; read
in-sample per the step-0 scope.)

## In-sample body-magnitude read (the 3 mechanics)
| arm (sb) | frac_alive | ratio_med | ey_mean_pos | ey_max |
|---|---|---|---|---|
| sp_log s42 | 0.55 | 0.012 | 2.97 | 145 |
| sp_log s43 | **1.00** | **0.127** | 5.37 | 206 |
| relu_log s42 | 0.00 | 0.000 | 0.00 | 0.24 |
| relu_log s43 | 0.00 | 0.000 | 0.00 | 0.00 |
| sp_count s42 | 0.11 | ~0 | 0.06 | 1.9 |
| sp_count s43 | 0.14 | ~0 | 0.07 | 3.4 |
| relu_count s42/43 | 0.00 | 0.000 | 0.00 | <0.6 |

1. **softplus REVIVES the all-cell body; relu is DEAD.** sp_log frac_alive 0.55–1.0 (E[y]≈3–5 on positive
   cells) vs relu_log ≡ 0 (dead-ReLU, C-178). Confirmed on the *actual all-cell front-runner*, not just the
   hurdle positives. **softplus is the necessary fix.** (Matches prereg.)
2. **⚠️ SURPRISE #1 — aliveness is strongly SEED-dependent.** sp_log s42 frac_alive 0.55 / ratio_med 0.012
   vs s43 1.00 / 0.127 — a ~10× swing between two seeds (and relu s42 flickers to ey_max 0.24 while s43 is a
   hard 0). This is the BatchNorm seed-bimodality ([[project_perf_program_anchor0]], C-184) expressing as
   body aliveness. **2 seeds is barely enough** to characterize this body; "does softplus revive" is
   "yes, but by a seed-dependent amount."
3. **⚠️ SURPRISE #2 — count-space did NOT diverge at T=0; it went near-DEAD.** Prereg predicted an
   exp-gradient blow-up. Instead, in-sample T=0, `count_mean` collapsed timid (frac_alive 0.11–0.14,
   ey_max 1.9–3.4, weights finite — `clip_grad_norm` held). So the count_mean *instability* is a
   rollout/training-dynamics phenomenon, **not** a T=0 explosion; at step-0 it's just another flavor of
   dead/timid (count-MSE is dominated by fitting the 99.7% zeros → predicts ~0). My prereg expectation was
   wrong here — recorded honestly.
4. **Even revived, the body is deeply TIMID + heteroscedastic.** Best `ratio_med` is 0.127 (predicts ~13%
   of truth median) — all-cell MSE's zero-pull keeps it timid (the amount-ceiling WALL). Yet `ey_max`
   145–206 on a few cells: the body under-fires the median while over-firing a handful — the same
   rescale-not-calibrate over-fire signature as the τ-dial ([[project_body_knob_quest]]).

## Evaluation / what it means for the next step
- **softplus is necessary but not sufficient.** It revives the body, but the body is then timid + seed-
  bimodal + heteroscedastic + OOS-explosive without a gate. None of that is fixed by the activation.
- **The all-cell zero-pull is the timid-maker.** This is exactly what the code-needed follow-up factors
  attack: the **winsor-cap** (clip the tail so MSE isn't dragged) and the **running-average target** (smooth
  the spiky target). The grid shows the baseline they must beat: revived-but-timid `ratio_med` ≈ 0.01–0.13,
  seed-dependent.
- **Measurement caveat:** in-sample teacher-forced 1-step, month 440→441; ±1-month alignment doesn't affect
  the alive/dead/timid conclusions. OOS T=0 `ratio_med` (needs a working eval — a true 1-step eval or the
  user's preferred path, NOT feedback-clamp) is still owed and is a with-user decision.

## Status
All 8 arms trained (exit 0), floor restored (md5 OK), views-models clean. Code UNCOMMITTED (standing rule);
dossier git-tracked. Next (with user awake): build winsor-cap + running-average target transforms (TDD),
then run those cells vs this revived-timid baseline.
