"""Is MC dropout worth 4x the forward passes? — the C-128 cost test, at matched sample count.

Simon: *"Since we are not working on the uncertainty now I would switch MC dropout off and just
sample more from the head."* This measures it instead of deciding it, and the test has been an open
owner's-decision item since C-128 was written.

THE TWO CONDITIONS, both emitting exactly 16 members so the comparison is of QUALITY at fixed
ensemble size, never of ensemble size:

  A  as deployed      4 MC-dropout passes x 4 head draws       4 forward sweeps of the grid
  B  dropout off      1 deterministic pass x 16 head draws     1 forward sweep

⇒ B costs a QUARTER of A's forward work. If B scores no worse, the dropout posterior is being paid
for and not used.

⛔ WHY THIS IS NOT OBVIOUSLY A WIN FOR A. MC dropout contributes EPISTEMIC spread that head sampling
cannot produce — C-128 measured it at 11.2% of the body field's spatial variance, and spatially
correlated (lag-1 +0.27…+0.46), because dropout removes channels and the convolutions spread the
hole. The case for keeping it is that this is real uncertainty. The case against is that we have
separately measured the body as already OVER-dispersed, so epistemic spread stacked on top of
excess aleatoric spread may be actively harmful.

⛔ PRE-STATED BEFORE RUNNING:
  * I expect B to score EQUAL OR BETTER on composed CRPS, and I will call a difference under 1%
    indistinguishable at n=1.
  * The sharper, riskier half: I expect B to be strictly BETTER, because §4.3 measured that this
    body wants LESS dispersion, not more — a theta scale of ~26 recovered 9.58% held out. Removing
    a second source of spread should point the same way.
  * ⇒ If A is better by more than 1%, the dropout posterior is earning its cost and Simon's
    proposal should be declined on the evidence.

⛔ THE VEHICLE IS THE ONE WE ARE MOVING OFF, and that is stated here rather than discovered later.
The only checkpoint on disk is `truncated_nb` (`2a5efc2a…`); the `nb` control was deleted. The
question is about the INFERENCE machinery rather than the family, so it should transfer — but it is
n=1 on a vehicle the programme has just decided against, and it must be re-confirmed on R1.

THE SLOTS (C-351):
  population  all 64,818 land cells x the 13 OUT-OF-SAMPLE months 457-469 (FAO-02's partition)
  transform   none — count space, matching the frozen ruler
  estimator   `crps_ensemble` copied verbatim from `lodestar_score.py:26-33`; both arms m = 16, so
              its biased 2/(m*m) spread term cancels in the comparison
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
OOS_LO, OOS_HI = 457, 469
SEED = 42
t_start = time.time()


def crps_ensemble(truth, samples):
    """Verbatim from `lodestar_score.py:26-33`."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    a = np.abs(s - truth[:, None]).mean(1)
    w = 2 * np.arange(m) - m + 1
    return a - 0.5 * (2.0 / (m * m)) * (w[None, :] * s).sum(1)


def load(tag):
    z = np.load(SP / f"d4_result{tag}.npz")
    j = [i for i, m in enumerate(list(z["months"])) if OOS_LO <= m <= OOS_HI]
    return (z["mu"][:, j, :].astype(np.float64), z["theta"][:, j, :].astype(np.float64),
            z["gate"][:, j, :].astype(np.float64), z["land_cells"], str(z["sha"]))


MU_A, TH_A, GT_A, land, sha_a = load("_tnb")        # 4 dropout passes
MU_B, TH_B, GT_B, land_b, sha_b = load("_tnbNODROP")  # 1 deterministic pass
if sha_a != sha_b or not np.array_equal(land, land_b):
    raise SystemExit("VOID: the two conditions are not the same vehicle or the same cells.")
if MU_B.shape[0] != 1:
    raise SystemExit(f"VOID: the dropout-off cube has {MU_B.shape[0]} passes, expected 1.")

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= OOS_LO) & (df.month_id <= OOS_HI)]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land)[list(range(OOS_LO, OOS_HI + 1))].to_numpy(np.float64)).T
T, C = Y.shape

print("=" * 96)
print("IS MC DROPOUT WORTH 4x THE FORWARD PASSES? — matched at 16 members")
print("=" * 96)
print(f"  vehicle      {sha_a[:16]}…  ⚠️ truncated_nb, the family the programme is moving OFF")
print(f"  population   {T} months x {C:,} land cells = {T * C:,} cell-months, out of sample")


def ensemble(MU, TH, GT, head_draws):
    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    mem = []
    for d in range(MU.shape[0]):
        p = MU[d] / (MU[d] + TH[d])
        dr = rng.negative_binomial(np.repeat(TH[d][None], head_draws, 0),
                                   1.0 - np.repeat(p[None], head_draws, 0)).astype(np.float64)
        comp = compose_samples(torch.from_numpy(np.log1p(dr)).permute(1, 2, 0),
                               torch.from_numpy(GT[d]), "soft_gate", generator=gen)
        mem.append(np.expm1(comp.numpy()))
    return np.concatenate(mem, -1).reshape(T * C, -1)


ens_a = ensemble(MU_A, TH_A, GT_A, 4)    # 4 passes x 4 = 16
ens_b = ensemble(MU_B, TH_B, GT_B, 16)   # 1 pass  x 16 = 16
assert ens_a.shape == ens_b.shape, (ens_a.shape, ens_b.shape)

yv = Y.reshape(-1)
ca = float(crps_ensemble(yv, ens_a).mean())
cb = float(crps_ensemble(yv, ens_b).mean())
obs = float((yv > 0).mean())

print("\n  %-34s %14s %12s %12s" % ("condition", "composed CRPS", "pos frac", "fwd sweeps"))
print("  %-34s %14.6f %12.4f %12d" % ("A  4 dropout passes x 4 draws", ca, (ens_a > 0).mean(), 4))
print("  %-34s %14.6f %12.4f %12d" % ("B  dropout OFF, 1 pass x 16", cb, (ens_b > 0).mean(), 1))
print("  %-34s %14s %12.4f" % ("observed", "—", obs))
print("\n  ⇒ B vs A: %+.2f%% on CRPS, at ONE QUARTER of the forward cost"
      % (100 * (cb - ca) / ca))

# Where A's spread actually comes from — the thing the cost buys.
mu_bar = MU_A.mean(0)
between = MU_A.var(0, ddof=1).mean()          # across dropout passes
print("\n  WHAT THE EXTRA THREE PASSES BUY, measured on the body's own parameter")
print("    mean across-pass variance of mu            %.6e" % between)
print("    mean mu                                    %.6e" % mu_bar.mean())
print("    across-pass sd as a share of mu            %.4f"
      % float(np.sqrt(between) / mu_bar.mean()))
print("    dropout-off mu vs the 4-pass mean, median ratio  %.4f"
      % float(np.median(MU_B[0] / np.clip(mu_bar, 1e-12, None))))

print("""
  ⛔ READING. A difference under 1% is indistinguishable at n=1 and makes B the cheaper equal.
  Only A being better by more than 1% would say the dropout posterior earns its cost.""")

print("\n" + "=" * 96)
print("COST, measured")
print("=" * 96)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
