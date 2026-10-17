# Raw evidence — pos_weight arm readout
# Measured by the views-models session, 2026-10-17. Handed to the views-hydranet session to write up.
# This file is PRIMARY DATA plus provenance. It is deliberately not prose.

## 0. Arm identity — verified from the library's own log, not from config text

| run | role | weights sha256 | built-loss pos_weight | evidence |
|---|---|---|---|---|
| `amber_kestrel` | control | `2a5efc2a67c08eac05b8e32b7634e1c0a7eba04c3938781ad7264d702bdf93dd` | **2.0** | `amber .../calibration_delivery_20261013/run.log:430` (2026-10-12 22:43:18) |
| `copper_kestrel` | arm | `f286e4a3ac506215f1b0aecf3473da52aedf160d855fdfa46eddec67199ca705` | **1.0** | `copper .../calibration_delivery_20261016/run.log:95` (2026-10-16 09:52:39) |

Both log lines come from `views_hydranet/utils/weighted_bce_loss.py:41` and report the value on the
CONSTRUCTED criterion, which is what the prereg's free pre-flight asked for. Both shas match the
`sha` field stored inside the corresponding `p2_result_*.npz`, so the harness measured these weights
and no others.

### TRAP — do not cite the `pos_weight` field inside the npz
`p2_result_amber.npz` and `p2_result_copper.npz` BOTH store `pos_weight = 2.0`. That field is the
harness's hardcoded `SHIPPED_POS_WEIGHT` constant (violet_visitor's value), not a value read from the
model. `p2_gate_calibration.py:185` assigns `pw = SHIPPED_POS_WEIGHT` under verdict D2. It is
meaningless for these two arms. Cite the run.log lines above instead.

### Provenance asymmetry to state in the write-up
`copper` ships `configs_as_run/` (6 files) inside its delivery; `amber` does not — its delivery holds
only `artifact draws gate_h1 MANIFEST metricframes parquet run.log STAGE STATUS wandb_offline`. So the
control's as-run config must be read from git at `73c3f42` (per prereg §2), while the arm's is captured
in-band. That is a difference in what was ARCHIVED, not in what RAN, and the run.log lines are
authoritative for both. Worth naming because it means the two arms are not symmetric as records.

## 1. P-1 — gate calibration error, median |gate − p|

| run | local sweep | pod's own gate slice |
|---|---|---|
| control (w=2) | **0.0154** | 0.0175 |
| arm (w=1)     | **0.0303** | 0.0298 |

Prediction: ≤0.010 AND no worse than control. Measured: 1.97× control. **F1 FIRES → KILL.**

Two independent paths agree to ~0.001–0.002, so the number is not an artefact of either path.

## 2. P-2 — median implied pos_weight (the effective tilt the trained gate actually shows)

`implied_w = s(1−p) / (p(1−s))`, median over calibration bins with n ≥ 1000 and 0 < p < 1.
Honest gate ⇒ 1.0.

| run | trained at | local sweep | pod slice | ratio implied/trained (local) |
|---|---|---|---|---|
| control | 2.0 | **1.3090** | 1.3571 | 0.655 |
| arm     | 1.0 | **0.6252** | 0.6047 | 0.625 |
| violet_visitor (published, for scale) | 2.0 | 1.0365 | — | 0.518 |

Prediction: [0.90, 1.15]. Both arms outside, on OPPOSITE sides. **The sign flipped.**

### Derived, and NOT pre-registered — flag clearly as an extrapolation from two points
Interpolating to implied_w = 1.0:
- linear in w, local numbers:  1 + (1.000−0.6252)/(1.3090−0.6252) = **1.548**
- linear in w, slice numbers:  1 + (1.000−0.6047)/(1.3571−0.6047) = **1.525**
- log-linear in w, local:      **1.55**

So the data brackets the honest value in (1, 2) and points at **≈1.5**. It does NOT support 2.0,
and it rules out the direction of 2.5. Two points, n=1 each: this is a prediction to be tested,
not a result. See §8 for why acting on it would still be a hack.

## 3. P-3 — gate AP (ranking score), paired by month

| run | pooled AP |
|---|---|
| control | **0.5324199835833106** |
| arm     | **0.5384912009101893** |

Paired by month (13 origins, months 457–469), arm − control: **+0.0075**,
bootstrap CI **[+0.0014, +0.0136]**, arm higher in **11 of 13** months.

Per-month AP, control:
`0.518086 0.497207 0.466338 0.500436 0.526224 0.497208 0.532072 0.537580 0.527599 0.587742 0.596930 0.586652 0.567399`

Per-month AP, arm:
`0.524430 0.498265 0.480155 0.521486 0.532970 0.498807 0.537173 0.534982 0.544637 0.615079 0.606739 0.594107 0.550156`

Prediction (prereg §3.2): structurally UNCHANGED, ±0.010, because `pos_weight` applies a strictly
monotone tilt in p and AP depends on the gate only through RANKS. **HELD** — but note the CI excludes
zero and its upper bound (+0.0136) sits just outside the ±0.010 band. Report as held-with-a-note, not
as a clean null. The honest reading: the structural argument is confirmed in magnitude; the small
positive residual is unexplained and is a candidate for seed noise.

## 4. P-4 — CRPS, each arm from ITS OWN metricframes

| target | control | arm | change |
|---|---|---|---|
| sb | 0.1594 | 0.1010 | −37% |
| ns | 0.1027 | 0.0173 | −83% |
| os | 0.0265 | 0.0109 | −59% |

Comparability checked: both arms' metricframes carry `data_version = 504` and
`scoring_code_version = 2.0.0`, selected on `metric == "CRPS"` and `group_id == "mean"`.

Prediction: no worse, and ≤2% better. Measured: 37–83% BETTER. **MISSED UPWARD.**

**This is not skill.** `activation_metrics.py` states the mechanism in its own words:
"crps_all / MCR go BLIND when the field is ~99% zero: a forecast that almost never fires still
[scores well]… This is the falsifier crps_all cannot see." A sparser, smaller forecast on a
99.7%-zero field scores better by construction. §5 and §6 are the same fact seen without the blind
spot. The write-up must not present P-4 as a gain anywhere, including in a summary table, without
the blindness stated in the same visual unit.

## 5. P-5 — emitted positive fraction, pre-registered 100-cell set, h=1

| run | fraction |
|---|---|
| control | **0.4999038461538461** |
| arm     | **0.3946634615384615** |

Prediction: [0.55, 0.85]. **Both arms outside the band**, so F5 fires literally but cannot
discriminate. The band was sized on the composition arm's expectation, which that arm itself missed
(its OUTCOME.md records P-1 predicted [0.62, 0.76], measured 0.4999, MISSED). Control reproduces
`2026-10-13_composition_arm_outcome/OUTCOME.md`'s published 0.4999 exactly, which is the validation
that the harness is wired correctly, not a coincidence.

## 6. P-6 — act_ratio at h18 = P(emitted > 0 over draws) / P(truth > 0)

| run | P(pred>0) | P(truth>0) | act_ratio |
|---|---|---|---|
| control | 0.009397985958316422 | 0.003995803634792804 | **2.3519639144639144** |
| arm     | 0.001743268726398412 | 0.003995803634792804 | **0.4362748737748737** |

1.0 = honest. Control OVER-fires 2.35×. Arm UNDER-fires at 0.44× — the #258 collapse signature.

Prediction: ≤ control. **HELD** — and this is the pre-registered guardrail that exposes §4 as an
artefact. The prereg included it precisely because CRPS cannot see firing rate.

**Also closes an endpoint the control's own write-up left open.**
`2026-10-13_composition_arm_outcome/OUTCOME.md:22` records: "**F3 / F4** act_ratio, crps_all |
**not measured** — h≥2 gate posterior deliberately not retrieved (#558) | open". That was wrong about
what act_ratio needs. It reads the EMITTED COMPOSED draws (`lr_sb_best`, shape (2333448, 16) =
64,818 cells × 36 months × 16 draws, `time` 457–492), which came home in `draws/`. Only the raw
`by_*` gate posterior at h≥2 died with the pod, and act_ratio does not need it. The control's
act_ratio at h18 is 2.3520. That number should be back-filled into the composition arm's OUTCOME.md.

## 7. Sparsity at h=1 (where all endpoints land)

| | control | arm |
|---|---|---|
| P(emitted > 0) | 0.00404 | 0.00216 |
| total predicted mass | 6138 | 4099 |

Firing roughly halves; total mass drops ~33%.

## 8. Interpretation — the part that matters more than the verdict

Gneiting & Ranjan (2011) Thm 1: weighting a proper score by a function of the OUTCOME makes it
improper. `weighted_bce` at pos_weight w has optimum `s* = wp / (1 + (w−1)p)`. So **w ≠ 1 is
mathematically guaranteed to yield a miscalibrated gate**, and w = 1 is the principled value.

Therefore at w = 1 the gate SHOULD have been calibrated. It is not: 0.0303, systematically
UNDER-stating (implied 0.63 < 1). That under-statement has a cause that is NOT the pos_weight tilt —
removing the tilt is what revealed it. And w = 2 was pushing predictions back up, partially cancelling
it (implied 1.31 > 1).

**So the finding is not "1.0 is worse than 2.0". It is: there is a second, independent downward bias
in the gate, and the platform has been unknowingly compensating for it with an improper loss tilt.**

Consequences to state:
- tuning w to ≈1.55 (§2) would ALSO be a hack: good calibration numbers, underlying defect intact,
  cancellation valid only at the data distribution it was tuned against, silent failure under shift.
- the principled treatment is post-hoc recalibration (Platt / temperature scaling), which by the
  §3 rank argument provably cannot damage AP — monotone transformations are invisible to a ranking
  score. Cost: a one-parameter fit on arrays already in hand, seconds on CPU, no GPU run.
  Limit: applying it across all 36 horizons needs the gate posterior retained at h≥2, which is
  exactly what #558 / C-161 is about. At h=1 it is possible today from `gate_h1/`.
- `pos_weight` cannot be LEARNED by gradient descent: it lives inside the loss, so minimising that
  loss w.r.t. it is degenerate.
- the real target is the downward bias itself. Treatments are secondary.

## 9. Falsifiers

| | verdict | note |
|---|---|---|
| F1 | **FIRES → KILL** | P-1 worse than control (0.0303 vs 0.0154) |
| F2 | no | |
| F3 | no | |
| F4 | no | |
| F5 | fires literally | control also outside the band; cannot discriminate — see §5 |
| F6 | **tripped literally; RULED NOT VOID by Simon, 2026-10-17** | see §10 |
| F7 | pass | |
| F8 | no divergence | 300/300 lessons, zero hard failure signatures |

## 10. F6 — the prereg contradicts itself, and the ruling

The two runs differ in TWO config values, not one:
1. `loss_class_pos_weight` 2.0 → 1.0 — the variable under test.
2. `trajectory_log_path` — absent in the control, set in the arm
   (`copper .../configs_as_run/config_hyperparameters.py:233`, via `_trajectory_log_path()` at :53).

F6 voids the experiment on any second difference. But **prereg §6 MANDATES setting
`trajectory_log_path`**. The same document both requires the second difference and voids the run for
having it.

Substantive reading — NOT substantive, from reading the code:
`views-hydranet/views_hydranet/train/training_engine.py:1423-1470` registers a read-only forward hook
on `dec_conv4_head{1,2,3}_class` that accumulates `_o.detach().mean().item()` and writes a CSV. No RNG
draw, no gradient path, no parameter touched. The comment at :1422 states
"config.trajectory_log_path=None (default) ⇒ no hooks, no file, byte-unchanged."

**Ruling: Simon ruled the experiment NOT VOID on 2026-10-17**, after being shown the conflict and the
option to rerun the control with logging enabled (~12 h, ~$16). Record it as a dated human ruling with
the conflict visible — the point of a pre-registration is defeated if an inconvenient clause is
quietly dropped after the results are known. The §6/F6 contradiction should also be FIXED in the
prereg's own text so the next arm does not inherit it.

## 11. Threshold-sizing flaw — a defect in the prereg, not in the run

- P-1's ≤0.010 was sized on violet_visitor's 0.0054, a smaller-extent vehicle. The CONTROL fails it
  (0.0154).
- P-5's [0.55, 0.85] was sized on an expectation the composition arm itself missed. The CONTROL
  fails it (0.4999).

So F1 and F5 as WRITTEN cannot distinguish arm from control. What discriminates is the paired
contrast — same script, same data, same stack, 0.0303 vs 0.0154 — and that is unambiguous. State this
plainly; the verdict is correct but one of its stated justifications is not, and anyone re-checking
the arithmetic will find it.

## 12. Validation performed before any interpretation

| check | result |
|---|---|
| P2 statistics block, applied verbatim to the stored violet array | reproduces published 0.0054 / 0.0573 / 1.0365 exactly |
| control gate AP vs composition-arm OUTCOME.md | 0.5324 — exact |
| control emitted positive fraction vs OUTCOME.md | 0.4999 — exact |
| local inference vs the pod's own gate slice, P-1 | agree to ~0.002 |
| weights shas in npz vs delivery `.sha256` | both match |
| CRPS comparability | both `data_version` 504, `scoring_code_version` 2.0.0 |

## 13. Method caveats that must appear in the write-up

- **n = 1 per arm. No seed replication.** The prereg pre-commits that a SCREEN cannot license a
  fleet change. Valid use: killing an idea. Invalid use: asserting the cancellation mechanism in §8
  as established. §8's mechanism is a reading of the pattern, strongly supported by the Gneiting &
  Ranjan theorem but not separated from seed noise by measurement.
- **Stack skew.** Pod torch 2.14.1, local torch 2.10.0+cu128. numpy matches at 1.26.4. Mitigated two
  ways: both arms measured on the SAME local stack (so the contrast is internally consistent), and
  cross-validated against the pod's own gate slice (§1, §2).
- **Library version matched.** Pod ran `views-hydranet==0.1.3`; the local checkout is `0.1.3` and
  `git log <bump>..HEAD -- views_hydranet/` is empty, so no library drift since release.
- **One harness override.** `vehicle.py` hardcodes `EXPECTED_SIDECAR["output_distribution"] = "nb"`
  (violet_visitor's value) and VOIDS on anything else; both kestrels are `truncated_nb`. A scratchpad
  copy overrides that ONE value (`run_p2.py`) after asserting the committed file's sha256
  (`a77044813c50b8f8052094bb0c913b7d7ab23839d5d4d67041bcb49f2466a4d8`) is unchanged. Nothing else
  was altered. The committed `p2_result.npz` was backed up before anything ran. This is a latent
  generality defect in `vehicle.py` worth its own issue: the instrument refuses every arm it was
  built to compare.
- **Calibration method, unchanged from P2:** posterior-mean gate over D=4 locked-dropout draws,
  bin edges `[0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01]`, medians over bins
  with n ≥ 1000, `NP_SEED = TORCH_SEED = 42`, `BASE_ORIGIN = 456`, `N_ORIGINS = 13`,
  `LEADIN_START = 421`, CPU. Positive fraction is a POOLED MEAN, never a median over cells (draws
  quantise at 1/16).

## 14. Files in `evidence/`

| file | what |
|---|---|
| `p2_result_amber.npz` | control gate posterior (4, 13, 64818) + truth + sha. NOTE the `pos_weight` trap, §0 |
| `p2_result_copper.npz` | arm, same shape |
| `readout_result.npz` | P-3/P-5/P-6 per-arm values and per-month AP |
| `run_p2.py` | runs the committed harness with the single documented override |
| `stats_only.py` | P2's statistics block applied to a stored npz — the reproduction check |
| `slice_calib.py` | calibration from each arm's POD-PRODUCED gate slice — the cross-validation |
| `readout.py` | P-3 / P-4 / P-5 / P-6 |
| `sparse.py`, `inv.py`, `mfmeta.py` | sparsity at h=1, npz inventory, metricframe metadata check |
| `readout_summary.txt` | the compact table as first computed |

NOT copied: `p2_gate_calibration.py` and `vehicle.py` (committed unchanged at
`reports/2026-10-12_underfire_diagnosis/`), and `p2_result_ORIGINAL_violet.npz` (the backup of the
committed file, which is unchanged on disk).
