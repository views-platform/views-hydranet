"""Train as `nb`, emit zero-truncated — the arithmetic test, before any code is built.

Pre-registered in `reports/2026-10-21_emit_time_split_prereg.md`. ⛔ Predictions are committed
there; this file must not restate them loosely.

THE DESIGN, and the one thing that makes it sharp: COMMON RANDOM NUMBERS. Both emission rules are
inverse-CDF draws from the SAME uniform `u`:

    NB emission          y = F^-1(u)                 u ~ U(0, 1)
    zero-truncated       y = F^-1(p0 + u*(1 - p0))   the same u, rescaled onto (p0, 1)

⇒ Every member is the *same draw* pushed above the zero atom. The comparison therefore isolates the
truncation exactly, with no sampler variance between arms at all. ⛔ Rejection sampling would have
been the obvious implementation and would have been both slower and noisier — its cost explodes
precisely where `p0 -> 1`, which is most of this grid.

The 2x2 is emission rule x theta scale, because §4.3 showed the theta lever is worth 9.58% held out
and two levers that are tested apart cannot show an interaction (F4).
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, pull  # noqa: E402

from views_hydranet.distributions.composition import compose_samples  # noqa: E402

SP = Path(__file__).parent
CUBE = "d4_result_CONTROL_nb_4acbf4d3.npz"
OOS_LO, OOS_HI = 457, 469
HEAD, SEED = 4, 42
THETA_SCALE = 21.9921          # §4.3's IN-SAMPLE-fitted scale — never the out-of-sample optimum
t_start = time.time()


def crps_ensemble(truth, samples):
    """Verbatim from `lodestar_score.py:26-33`."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    a = np.abs(s - truth[:, None]).mean(1)
    w = 2 * np.arange(m) - m + 1
    return a - 0.5 * (2.0 / (m * m)) * (w[None, :] * s).sum(1)


# ⛔ Everything below runs in `main()`, NOT at import. These filenames match pytest's default
# `python_files = *_test.py`, so a plain `pytest --collect-only` IMPORTS them — and the `.npz`
# cubes they read are gitignored, so on CI the import raised `FileNotFoundError` and broke
# collection for the whole suite (`test_falsify_zero_surprises.py::test_P2`). A measurement
# driver should do its work when RUN, never when imported.
def main() -> None:
    npz = np.load(SP / CUBE)
    land, months = npz["land_cells"], list(npz["months"])
    j = [i for i, m in enumerate(months) if OOS_LO <= m <= OOS_HI]
    MU, TH, GT = (npz[k][:, j, :].astype(np.float64) for k in ("mu", "theta", "gate"))
    D, T, C = MU.shape

    df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
    w = df[(df.month_id >= OOS_LO) & (df.month_id <= OOS_HI)]
    Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
         .reindex(land)[list(range(OOS_LO, OOS_HI + 1))].to_numpy(np.float64)).T
    yv = Y.reshape(-1)
    is_zero = yv == 0

    print("=" * 96)
    print("EMIT-TIME SPLIT — train as nb, emit zero-truncated. Arithmetic, nothing retrained.")
    print("=" * 96)
    print(f"  vehicle      {str(npz['sha'])[:16]}…  nb, the delivered violet_visitor config")
    print(f"  population   {T} months x {C:,} land cells = {T * C:,} cell-months, out of sample")
    print(f"  truth        {100 * is_zero.mean():.2f}% exact zeros, observed positive fraction "
          f"{1 - is_zero.mean():.4f}")
    print("  draws        COMMON RANDOM NUMBERS: one u per member, shared by both arms")

    # One fixed uniform cube, drawn once and reused by every arm.
    rng = np.random.default_rng(SEED)
    U = rng.random((D, HEAD, T, C))


    def ensemble(theta_scale: float, truncated: bool) -> np.ndarray:
        gen = torch.Generator().manual_seed(SEED)   # the SAME gate mask for every arm, too
        mem = []
        for d in range(D):
            th = TH[d] * theta_scale
            p_succ = th / (th + MU[d])                       # numpy/scipy's success probability
            u = U[d]                                         # [HEAD, T, C]
            if truncated:
                p0 = np.power(p_succ, th)                    # P(Y = 0) = (theta/(theta+mu))^theta
                u = p0[None] + u * (1.0 - p0[None])          # rescale onto (p0, 1): the same draw,
                #                                              pushed above the zero atom
            y = nbinom.ppf(u, np.broadcast_to(th, u.shape), np.broadcast_to(p_succ, u.shape))
            comp = compose_samples(torch.from_numpy(np.log1p(y)).permute(1, 2, 0),
                                   torch.from_numpy(GT[d]), "soft_gate", generator=gen)
            mem.append(np.expm1(comp.numpy()))
        return np.concatenate(mem, -1).reshape(T * C, D * HEAD)


    print("\n  %-38s %13s %11s %10s %12s %12s" % (
        "cell", "composed CRPS", "vs base", "pos frac", "CRPS on y=0", "CRPS on y>0"))
    base = None
    rows = {}
    for tname, tscale in (("model theta", 1.0), ("theta x %.1f" % THETA_SCALE, THETA_SCALE)):
        for ename, trunc in (("NB emission", False), ("zero-truncated", True)):
            ens = ensemble(tscale, trunc)
            per = crps_ensemble(yv, ens)
            tot = float(per.mean())
            if base is None:
                base = tot
            rows[(tname, ename)] = tot
            print("  %-38s %13.6f %10.2f%% %10.4f %12.6f %12.6f" % (
                f"{ename:<16s} | {tname}", tot, 100 * (tot - base) / base,
                float((ens > 0).mean()), float(per[is_zero].mean()), float(per[~is_zero].mean())))

    print("\n  THE 2x2, read as the prereg's F4 asks — does the theta effect")
    print("  survive the emission rule?")
    for ename in ("NB emission", "zero-truncated"):
        a = rows[("model theta", ename)]
        b = rows[("theta x %.1f" % THETA_SCALE, ename)]
        print("    %-16s theta rescale moves CRPS %+.2f%%" % (ename, 100 * (b - a) / a))
    for tname in ("model theta", "theta x %.1f" % THETA_SCALE):
        a = rows[(tname, "NB emission")]
        b = rows[(tname, "zero-truncated")]
        print("    %-16s truncation moves CRPS   %+.2f%%" % (tname, 100 * (b - a) / a))

    print("\n" + "=" * 96)
    print("COST, measured")
    print("=" * 96)
    print("  wall time %.2f s" % (time.time() - t_start))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print("  peak RSS  %.2f GiB" % peak)


if __name__ == "__main__":
    main()
