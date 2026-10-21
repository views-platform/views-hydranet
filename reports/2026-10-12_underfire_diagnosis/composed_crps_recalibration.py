"""§4.3 — does recalibrating to the SCORED metric buy anything the trained objective missed?

The seven-seat method review (`reports/2026-10-15_loss_balance_method_review.md` §4.3) ranked this
third and costed it at "$0, CPU minutes", and it was never run. Its purpose, in its own words: it
*"either quantifies the NLL-vs-CRPS divergence the prereg names as a limitation, or retires it as
measured-and-negligible for the price of an afternoon."*

THE QUESTION. We train the body by minimising an NB NLL and we score under CRPS (FAO-02, LOCKED).
Machete (2012) establishes that the two prefer different directions of departure, so the parameters
that minimise the training objective need not minimise the score. That is named as a limitation in
two pre-registrations and has never been measured. ⇒ Take the DELIVERED weights, change nothing
about training, and ask how much composed CRPS a one-parameter recalibration recovers.

  arm 1   theta' = s * theta          (the body's dispersion)
  arm 2   logit(g') = a * logit(g)    (a Platt SLOPE on the gate, no intercept)

⛔ ONE PARAMETER EACH, SWEPT, and reported as a curve rather than a scalar — this repo has withdrawn
three claims that crossed the value they were testing as a single free parameter moved.

⛔ WHAT THIS IS NOT. It is not a proposal to ship a recalibration, and it does not touch FAO-02: it
optimises the locked metric rather than replacing it (Gneiting & Raftery 2007 §9.1, optimum-score
estimation). Whether any map earns config surface is Simon's decision and is not asked here.

THE SLOTS (C-351), stated once:
  population  all 64,818 land cells x the 13 OUT-OF-SAMPLE months 457-469. ⛔ The VALIDATION
              partition, because FAO-02 mandates it. No admission filter; zeros are data.
  transform   none — count space, matching the frozen ruler
              (`2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`), which scores raw
              `lr_{t}_best` counts with no log1p anywhere.
  estimator   `crps_ensemble` COPIED EXACTLY from that ruler (the sorted-sample energy form with
              the biased `2/(m*m)` spread term), so this number is comparable to every historical
              row. ⛔ All arms use m = 16, so the `2/(m*m)` bias cancels in every comparison made
              here — the C-263 trap only bites across UNEQUAL ensemble sizes.

THE ENSEMBLE, matching the delivered protocol exactly: the cube carries 4 posterior draws of
(mu, theta, gate); 4 NB draws from each gives the deployed 16 members
(`n_posterior_samples` 4 x `n_head_samples` 4). Composition is `soft_gate` via the repository's own
`compose_samples`, called in log1p space as its contract requires and inverted afterwards.

⛔ PRE-STATED BEFORE RUNNING, so this is a test and not an exercise:
  * I expect the theta sweep's optimum to sit ABOVE 1.0 — M6 measured the CRPS-optimal theta at
    0.4298 against the model's 0.1544 on the body alone, so the scored metric wants LESS dispersion
    than the model emits, which is a scale of about 2.8x.
  * I expect the CRPS gain from it to be SMALL — under 3% — because the field is 97.8% zeros and
    the gate, not the body's dispersion, decides almost every cell's score.
  * I expect the gate slope's optimum to sit near 1.0 (the gate is calibrated: median |gate - p| =
    0.0054 over 842,634 cell-months) and to buy less than the theta arm.
  ⇒ If either arm recovers more than 5%, the NLL-vs-CRPS divergence is MATERIAL and the two
    pre-registrations that name it as an unquantified limitation need amending.
  ⇒ If both are under 1%, it is retired as measured-and-negligible and nobody need raise it again.

Reads the frozen control cube and the cached pull. Cost printed.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, pull  # noqa: E402

from views_hydranet.distributions.composition import compose_samples  # noqa: E402

SP = Path(__file__).parent
CUBE = "d4_result_CONTROL_nb_4acbf4d3.npz"   # the frozen nb control; its checkpoint is deleted
OOS_LO, OOS_HI = 457, 469                    # the validation partition
HEAD_DRAWS = 4                               # x 4 posterior draws = the deployed 16 members
SEED = 42
t_start = time.time()


def crps_ensemble(truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """COPIED VERBATIM from `lodestar_score.py:26-33`. ⛔ Do not 'improve' it — its biased
    `2/(m*m)` spread term is what every historical row was computed with."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    a = np.abs(s - truth[:, None]).mean(1)
    w = 2 * np.arange(m) - m + 1
    b = (2.0 / (m * m)) * (w[None, :] * s).sum(1)
    return a - 0.5 * b


npz = np.load(SP / CUBE)
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {m: i for i, m in enumerate(months)}
j = [mcol[m] for m in range(OOS_LO, OOS_HI + 1)]

MU = npz["mu"][:, j, :].astype(np.float64)        # [D, T, C]
TH = npz["theta"][:, j, :].astype(np.float64)
GT = npz["gate"][:, j, :].astype(np.float64)
D, T, C = MU.shape

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= OOS_LO) & (df.month_id <= OOS_HI)]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[list(range(OOS_LO, OOS_HI + 1))].to_numpy(np.float64)).T  # [T, C]

print("=" * 96)
print("§4.3 — COMPOSED-CRPS RECALIBRATION ON THE DELIVERED WEIGHTS. Nothing retrained.")
print("=" * 96)
print(f"  vehicle      {str(npz['sha'])[:16]}…  (the frozen nb control; its checkpoint is DELETED)")
print(f"  partition    OUT OF SAMPLE {OOS_LO}-{OOS_HI}, {T} months x {C:,} land cells "
      f"= {T * C:,} cell-months")
print(f"  ensemble     {D} posterior draws x {HEAD_DRAWS} head draws = {D * HEAD_DRAWS} members")
print(f"  truth        {TARGETS[SB]}, count space, {100 * (Y == 0).mean():.2f}% exact zeros")


def composed_crps(theta_scale: float, gate_slope: float) -> float:
    """One sweep point. Same seed every time, so the sweep isolates the parameter."""
    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    th = TH * theta_scale
    if gate_slope == 1.0:
        gt = GT
    else:
        lg = np.log(np.clip(GT, 1e-9, 1 - 1e-9) / (1 - np.clip(GT, 1e-9, 1 - 1e-9)))
        gt = 1.0 / (1.0 + np.exp(-gate_slope * lg))
    members = []
    for d in range(D):
        # NB in the (total_count, prob) parameterisation the family uses: p = mu/(mu+theta)
        p = MU[d] / (MU[d] + th[d])
        draws = rng.negative_binomial(
            np.repeat(th[d][None], HEAD_DRAWS, 0), 1.0 - np.repeat(p[None], HEAD_DRAWS, 0)
        ).astype(np.float64)                                   # [K, T, C]
        # compose through the REPOSITORY's own composer, in the log1p space its contract names
        comp = compose_samples(
            torch.from_numpy(np.log1p(draws)).permute(1, 2, 0),   # [T, C, K]
            torch.from_numpy(np.repeat(gt[d][None], 1, 0)[0]),    # [T, C]
            "soft_gate", generator=gen,
        )
        members.append(np.expm1(comp.numpy()))
    ens = np.concatenate(members, axis=-1).reshape(T * C, D * HEAD_DRAWS)
    return float(crps_ensemble(Y.reshape(-1), ens).mean())


base = composed_crps(1.0, 1.0)
print(f"\n  BASELINE composed CRPS, the model's own parameters   {base:.6f}")

for name, grid, call in (
    # ⛔ GRIDS WIDENED 2026-10-21 AFTER THE FIRST RUN, and the reason is recorded rather than
    # the first run being quietly discarded: BOTH optima landed on a grid edge, so the edge guard
    # below fired and neither number was reportable. The first run's bracket was theta 0.25-8.0
    # (monotone down to -8.42% at the edge) and gate slope 0.4-2.2 (monotone down to -1.52%).
    # Widening a bracket that failed to contain its optimum is not the same as moving a threshold
    # after seeing a result — no decision rule changed, and the §4.3 reading rule (1% / 5%) is as
    # committed. theta -> inf is the Poisson limit; gate slope -> inf is a hard 0.5 threshold,
    # i.e. `threshold_gate`, which is why the upper end must be reached rather than guessed.
    ("theta scale  (body dispersion)", np.geomspace(0.25, 1.0e4, 17),
     lambda v: composed_crps(v, 1.0)),
    ("gate slope   (Platt, no intercept)", np.geomspace(0.4, 64.0, 17),
     lambda v: composed_crps(1.0, v)),
):
    print(f"\n  SWEEP — {name}")
    print("    %-10s %14s %12s" % ("value", "composed CRPS", "vs baseline"))
    vals = []
    for v in grid:
        c = call(v)
        vals.append(c)
        mark = "  <-- model's own" if abs(v - 1.0) < 1e-9 else ""
        print("    %-10.4f %14.6f %11.2f%%%s" % (v, c, 100 * (c - base) / base, mark))
    vals = np.asarray(vals)
    k = int(vals.argmin())
    print("    ⇒ optimum at %.4f, CRPS %.6f, a %.2f%% improvement on the delivered parameters"
          % (grid[k], vals[k], 100 * (base - vals[k]) / base))
    if grid[k] == grid[0] or grid[k] == grid[-1]:
        print("    ⛔ THE OPTIMUM IS AT A GRID EDGE — the sweep does not bracket it and the "
              "number above is a bound, not an optimum.")

print("""
  ⛔ READING. Under 1% on both arms retires the NLL-vs-CRPS divergence as measured-and-negligible.
  Over 5% on either makes it material, and the two pre-registrations that name it as an
  unquantified limitation need amending. ⚠️ Either way this is ONE vehicle, ONE target (`sb`) and
  n=1, and a recalibration that helps out of sample here is not evidence it would help on a
  retrained model.""")

print("\n" + "=" * 96)
print("COST, measured")
print("=" * 96)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
