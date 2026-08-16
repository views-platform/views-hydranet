# 07 — Experiment log (append-only)

Every run and outcome, **including negatives and postmortems**. Each entry links its pre-registration
(`05_analysis_plan.md`) and states its verdict against the pre-committed falsifiers. No success-only drift.

---

## EXP-00 — build + guard verification (no arms run yet) · 2026-08-15

**Story:** the mechanism, before any experiment. **Code:**
`views_hydranet/utils/hydranet_inference.py::blend_recurrent_state` +
`HydraNetInference(freeze_recurrent=...)`; driver `tools/{run_freeze_arms,freeze_arm_entry}.py`.

### What was built

`freeze_recurrent ∈ {None, "hidden", "cell", "all"}`, an explicit argument on `HydraNetInference`, forwarded
by `InferenceOrchestrator`. Default `None` = today's behaviour, untouched. **Not a config key**, so no model
config can enable it and ADR-027's retirement of `freeze_h` stands;
`tests/test_inference_logic.py::test_freeze_h_option_retired` is green.

The hold is applied at `t > origin` only, against an anchor captured at the end of the seed step
(`t == origin`) — the last state built from real observations.

### Guards verified by sabotage, not by assertion

A guard that cannot fail is not a guard (EXP-09 of the ruler dossier). Three distinct leaks were introduced
deliberately and each was caught:

| sabotage | caught by |
|---|---|
| anchor captured one step late (first free-running step, not the seed) | `test_freezing_all_makes_the_state_stop_contributing_new_information` |
| blend applied at the seed step's forward | same test — h=1 is structurally immune, see below |
| blend leaking into the history-digestion branch (`t < origin`) | **`test_h1_is_byte_identical_across_every_mode`** |

**A correction to the plan's own wording.** It called the h=1 identity check "the load-bearing self-test" and
implied it catches any leak. It does not: h=1 is produced by the seed step's forward pass, which reads the
state built during digestion, so a blend placed at or after that forward cannot move it. What h=1 actually
guards is leakage into **digestion**. Narrower than advertised, still worth having, and now documented
accurately in the test itself.

### A mock that would have made the experiment vacuous

The first `_StateSensitiveModel` gave the two memory halves equal effective rates (short `+1.0/step`, long
`+0.1/step` behind a `×10` coefficient). Under it, `hidden` and `cell` produce **byte-identical** rollouts —
two of the four arms would have silently been the same arm. Caught by
`test_each_mode_produces_a_distinct_rollout_beyond_h1`, which exists for exactly this. Rates are now `+1.0`
vs `+0.5`.

### One pre-registration risk closed in the driver

`truncated_smoke` carries **two** calibration artifacts, and the more recent
(`calibration_model_20260814_061215.pt`) is the ε=0.1 scheduled-sampling arm — **not** the EXP-SS-2 artifact
F2 requires. Letting the pipeline default to the latest would have scored a different model and made the
reproduction control meaningless while looking fine. `--artifact` is therefore **required**, not defaulted,
in both driver scripts, and the file's existence is checked before the run starts.

### Status

**No arm has been run.** 1421 tests pass, ruff clean. Next: the `none` arm on `truncated_smoke`, timed, to
establish the per-arm cost before committing to the remaining seven runs.

---

## EXP-01 — the `none` control arm · 2026-08-15 · **F2 CLEARED, P1 CONFIRMED**

**Pre-registration:** `05_analysis_plan.md` (LOCKED) — P1, falsifier F2.
**Vehicle:** `truncated_smoke`, artifact `calibration_model_20260814_003058.pt` (the EXP-SS-2 artifact).
**Run:** emit-only, 13 origins, `rollout_feedback='sample'`, `freeze_recurrent=None`. **25.8 min** to
generate, ~1 min to score. Cubes deleted after scoring.

### F2 — does the harness reproduce the probe it will be compared against?

| h | EXP-SS-2 free-`sample` (AP / act_ratio) | this run |
|--:|---|---|
| 1 | 0.298 / 1.41 | **0.2979 / 1.4081** |
| 18 | 0.007 / 0.29 | **0.0070 / 0.2913** |
| 36 | 0.008 / 0.27 | **0.0083 / 0.2663** |

Three to four significant figures on both metrics at all three horizons. **F2 does not fire.** The freeze
arms are comparable to EXP-SS-1/2 and to the v2 board. **P1 CONFIRMED.**

### The full curve — new, and it relocates the collapse

Every prior probe reported h = 1/18/36 only. At seven horizons the shape is not a decay across the horizon;
it is a **cliff between h1 and h6**:

| h | 1 | 6 | 12 | 18 | 24 | 30 | 36 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **gate AP** | **0.2979** | **0.0284** | 0.0064 | 0.0070 | 0.0068 | 0.0071 | 0.0083 |
| `act_ratio` | 1.408 | 0.385 | 0.320 | 0.291 | 0.289 | 0.278 | 0.266 |
| `size_ratio` | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `crps_all` | 0.151 | 0.118 | 0.113 | 0.135 | 0.134 | 0.169 | 0.875 |

**90% of the AP loss happens by h6**, and it is essentially flat from h12 to h36 (0.006–0.008). The model
does not degrade gradually — it falls off a cliff in the first few free-running steps and then sits there.
`size_ratio` is 0.0000 from h6 on: the body stops emitting magnitude entirely, immediately.

**This sharpens what any fix has to do.** A mechanism that explains "slow drift over 36 months" is the wrong
shape of explanation; whatever goes wrong, goes wrong almost at once.

### An aside the ruler dossier predicted

`crps_all` is **lowest at h12 (0.113)** — better than at h1 (0.151) — while gate AP is 47× worse there. The
C-231 zero-domination trap, visible in flight: the score improves as the model goes quiet. Anyone reading
`crps_all` alone would conclude the rollout gets *better* until h18.

### Ops notes
- **Three driver bugs, all found by running it** — `parse_args()` takes no argv list; every arm writes to the
  same prediction dir (named after the artifact, not the run) so a leftover would contaminate the next arm;
  and `score_v2_horizons` requires `--targets=sb`, not the space form its own docstring shows.
- That last one cost the control its scoring pass — but **score-then-delete meant the 2.5 GB of cubes
  survived**, so it was re-scored instead of regenerated. The ordering earned its keep by accident.

**Next:** `hidden`, `cell`, `all` on the same artifact. P2/P3 and F1/F3 are read only after all four exist.

---

## EXP-02 — the three freeze arms · 2026-08-15 · **VERDICT: STATE-IMPLICATED**

**Pre-registration:** `05_analysis_plan.md` (LOCKED) — P2, P3, falsifiers F1, F3, decision rule.
**Vehicle:** `truncated_smoke`, artifact `calibration_model_20260814_003058.pt`. Emit-only, 13 origins,
`rollout_feedback='sample'`, ~26 min/arm. Cubes score-then-deleted.

### Gate AP — the primary metric

| h | `none` | `hidden` | `cell` | `all` | best Δ |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.2979 | 0.2979 | 0.2979 | 0.2979 | +0.0000 |
| 6 | 0.0284 | 0.0507 | 0.0716 | 0.0865 | +0.0580 |
| 12 | 0.0064 | 0.0393 | 0.0729 | 0.0806 | +0.0743 |
| **18** | 0.0070 | 0.0342 | 0.0821 | **0.0912** | **+0.0842** |
| 24 | 0.0068 | 0.0329 | 0.0715 | 0.0851 | +0.0783 |
| 30 | 0.0071 | 0.0299 | 0.0745 | 0.0731 | +0.0673 |
| **36** | 0.0083 | 0.0253 | 0.0671 | **0.0693** | **+0.0609** |

### The pre-registered rule, applied as written

```
max(AP_held) − AP_none @h18 = +0.0842  ≥ 0.05  ✓
max(AP_held) − AP_none @h36 = +0.0609  ≥ 0.05  ✓
⇒ STATE-IMPLICATED
```

**The threshold was not touched.** 0.05 is FAO-02's superiority margin, fixed in the LOCKED plan before any
arm ran, and it is cleared at both required horizons.

- **P1 CONFIRMED** (EXP-01). **P2 CONFIRMED** — held arms hold AP materially above the control at h18 and h36.
- **P3 CONFIRMED** — `size_ratio` is **0.0000 at h≥18 in every arm**, held or not. Freezing recovers
  occurrence and does nothing for magnitude. The two ceilings are separate, as pre-registered.
- **F1 did not fire** — h=1 is byte-identical across all four arms on AP, `act_ratio`, `crps_all`, Brier and
  precision@k. The arms share a common history.
- **F2 did not fire** (EXP-01). **F3 did not fire** — the arms are far outside seed noise.

### ⚠️ CORRECTED — the arm ordering does NOT localise the damage to the cell state

**Original claim (retained for the record):** `hidden` +0.0272, `cell` +0.0751, `all` +0.0842 ΔAP at h18,
therefore "the damage accumulates in the cell state; `cell` carries 89% of the effect; this is a specific
channel, not the recurrent state in general."

**That claim is withdrawn.** `/code-review medium` on PR #277 checked the arms against the ConvLSTM's own
equations (`HydraBNrecurrentUnet_06_LSTM4.forward:527-529`):

```
hl = f_t * hl + i_t * hl_tilde        # the cell update
hs = o_t * tanh(hl)                   # the hidden state is a READOUT of the cell
```

Because `hs` is derived from `hl`, pinning `hl` to the anchor also constrains `hs` — the short-term half is
re-derived from the anchored cell every step. **`cell` therefore structurally approximates `all`, whatever
the truth about where damage accumulates.** The reverse does not hold: under `hidden`, `hl` still integrates
freely, which is why that arm is the only clean single-channel intervention in the set.

So the observed `cell ≈ all` is predicted by the architecture and is **not evidence** that the cell state is
the locus.

**What survives:** holding the recurrent state recovers ~23% of the oracle gap at h18/h36 (`all` +0.0842 /
+0.0609), and freezing the short-term half alone recovers the least (+0.0272). The state path is a real
mediator. **Which half carries it is not established by this design**, and separating them needs an arm that
holds `hl` *and* recomputes `hs` from the anchored `hl` (or the mirror) — no arm does that. Registered as
C-292.

### Corroborating signal — the fired cells are real

`precision_at_k` at h18: `none` **0.0045** → `all` **0.1655**, a **37×** improvement. At h36: 0.0084 → 0.1450.
And `act_ratio` improves in the same order (h36: 0.266 → 0.305), so the AP gain is **not** bought by firing
less — the gate fires on more genuinely-active cells.

### How much of the problem this is

The oracle ceiling at h36 is AP ≈ 0.271 (EXP-SS-2 `teacher_forced`).

| arm | AP @h36 | share of the oracle gap recovered |
|---|--:|--:|
| `hidden` | 0.0253 | 6% |
| `cell` | 0.0671 | 22% |
| `all` | 0.0693 | **23%** |

**Real, substantial, and not the whole story.** 77% of the gap survives a total state freeze, so state
corruption is *a* mechanism, not *the* mechanism. #262's distributional-gap thesis is not overturned — it is
now sharing the explanation with a channel it had ruled out.

### `crps_all` could not see any of this

At h18 the four arms score `crps_all` **0.1353 / 0.1352 / 0.1350 / 0.1346** — identical to three decimals —
while gate AP spans **13×** and precision@k **37×**. At h36 they agree to four decimals (0.8753–0.8755).

This is the strongest live demonstration yet of the Epic #263 finding: on this DGP `crps_all` is not merely
noisy about occurrence, it is **blind** to it. Any programme that had judged these arms on the FAO-02 primary
metric would have reported "no effect" and closed the question.

### What this overturns

**#262's ruling that the collapse is "NOT hidden-state / recurrent drift (overturns the prior C-222-based
bet)" is wrong**, and wrong for the reason C-222 records: the oracle probe varies the *input* while the state
evolves normally, so it can only show the state is healthy when never polluted. It cannot speak to the
polluted case. C-222's confound was real and should be reopened.

The 2026-06 retirement of `freeze_h` is **not** contradicted — that ablation asked whether freezing stopped
the C-113 *bloom* (it did not, and still does not: `crps_all` at h36 is unmoved). It never asked about the
gate.

### Scope — carried into any downstream claim

40 lessons, seed 42, one origin set, one target (`sb`), one vehicle. **Indicative.** The direction and the
hidden-vs-cell asymmetry are large enough to survive a lot of noise; the magnitudes are not rankable.
`violet_visitor` (160 lessons, 13 origins) is the confirmation vehicle and has not been run.

### What this does NOT license

Reinstating `freeze_h`. A hard freeze is a train/inference mismatch and buys 23% of the gap. What the result
argues for is a **soft prior on the cell state** — decay or confidence-weight its update while the model is
feeding on its own output — which is a new pre-registration, not an extension of this one.

---
