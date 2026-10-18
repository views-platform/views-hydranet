"""E6f - re-estimate BatchNorm statistics on the INFERENCE distribution and re-score.

THE DISCRIMINATOR (PREREG Amendment 8): E6f dominates E6 because it is PAIRED on the identical
global population, so none of the seven window confounds apply. F1-quad is an EQUIVALENCE test.

Locked method (Amendment 9), all ten:
  1  scratch copy in memory
  NEVER written back (C-112/C-119)
  2  re-estimation months 121-456 (TRAINING months
  the eval window would make the statistics
     origin-dependent, a different and larger remedy)
  3  full-grid 360x720 passes, NOT windows - the inference geometry is the point
  4  _reset_bn_stats semantics: reset_running_stats() + momentum=None (cumulative average)
  5  dropout ON (ADR-057: inference runs MC-dropout, so statistics must match)
  6  training_augmentation=False - clean data, no flips, no input noise (C-328 inst. 4/5)
  7  torch.no_grad() throughout
  no gradients, no optimiser
  8  PAIRED: b BEFORE on this same pipeline must reproduce 1.1875 bin-free (F3' precondition)
  9  both arms, amber first
 10  cost measured and reported, never claimed.  --smoke N measures cost before committing.

MANDATORY GUARD: _assert_bn_buffers_finite after re-estimation. With momentum=None one
non-finite activation poisons all 15 layers' buffers and they never recover; b returns a number
and nothing complains (training_engine.py:1266-1292).

USAGE (not run without Simon's explicit authorisation):
  python3 e6f_bn_reestimate.py --smoke 8        # time 8 months, extrapolate, run nothing else
  python3 e6f_bn_reestimate.py --arm amber      # full E6f
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
DIAG = HN / "reports/2026-10-12_underfire_diagnosis"
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(HN))
import vehicle  # noqa: E402
from vehicle import SB, TARGETS, build_volume, grid_rowcol, load_vehicle, pull  # noqa: E402

# vehicle.EXPECTED_SIDECAR hardcodes violet_visitor's 'nb'; these arms are 'truncated_nb'.
# Identical accommodation to run_p2.py, which states: "The ONLY change is
# vehicle.EXPECTED_SIDECAR['output_distribution'] ... Every statistic is the committed file."
# The gate refuses LOUDLY on a mismatch, which is why this is explicit and asserted, not silenced.
assert vehicle.EXPECTED_SIDECAR["output_distribution"] == "nb", "gate already changed; stop"
vehicle.EXPECTED_SIDECAR["output_distribution"] = "truncated_nb"

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _assert_bn_buffers_finite,
    _reset_bn_stats,
)

ART = {"amber": MODELS / "amber_kestrel/data/generated/calibration_delivery_20261013/artifact"
                 / "calibration_model_20261013_010739.pt"}
POSTERIOR_D, NP_SEED, TORCH_SEED = 4, 42, 42
BASE_ORIGIN, N_ORIGINS, LEADIN_START = 456, 13, 421
RECAL_FROM, RECAL_TO = 121, 456
B_EXPECT = 1.1875          # bin-free, retargeted in Amendment 4. F3' precondition.
SD_B = 0.0107              # measured on the full population; F1-quad band is 2*sd = 0.021
EPS = 1e-9


def lg(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def ml_slope(x, yy, iters=80):
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        W = p * (1 - p)
        r = yy - p
        H = np.array([[W.sum(), (W * x).sum()], [(W * x).sum(), (W * x * x).sum()]])
        d = np.linalg.solve(H + 1e-9 * np.eye(2), np.array([r.sum(), (r * x).sum()]))
        a += d[0]
        b += d[1]
        if np.abs(d).max() < 1e-11:
            break
    return a, b


def set_bn_train(model, on):
    """BN accumulates only in train() mode. Nothing else is put into training mode."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train(on)
            n += 1
    return n


def score(model, cfg, vol, months, lri, lci):
    """P2's scoring sweep: D locked-dropout passes, posterior-mean gate on land cells."""
    model.eval()
    model.set_locked_dropout(True)
    set_bn_train(model, False)
    g = np.zeros((POSTERIOR_D, N_ORIGINS, len(lri)), dtype=np.float32)
    with torch.no_grad():
        for d in range(POSTERIOR_D):
            model.reset_locked_dropout()
            h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
            for t, fed in enumerate(months):
                out = model(vol[:, t], h)
                h = out.h_next
                if fed >= BASE_ORIGIN:
                    g[d, fed - BASE_ORIGIN] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()
    return g.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="amber")
    ap.add_argument("--smoke", type=int, default=0,
                    help="time N re-estimation months and exit without re-scoring")
    a = ap.parse_args()

    ReproducibilityGate.lock_entropy(np_seed=NP_SEED, torch_seed=TORCH_SEED)
    model, cfg, sha = load_vehicle(ART[a.arm])            # LOCK 1: in-memory scratch only
    print(f"{a.arm}  sha {sha[:16]}…")
    for m in model.modules():
        if "BatchNorm" in type(m).__name__ and m.training:
            raise SystemExit("VOID: BN already in train() mode before we start")

    df = pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
    cells = np.sort(df.priogrid_id.unique())
    r, c = grid_rowcol(cells)
    lri, lci = r - 1, c - 1
    truth = np.zeros((N_ORIGINS, len(cells)), dtype=bool)
    bym = {m: g for m, g in df.groupby("month_id")}
    for n in range(N_ORIGINS):
        gg = bym[BASE_ORIGIN + 1 + n].set_index("priogrid_id").reindex(cells)
        truth[n] = gg[TARGETS[SB]].to_numpy() > 0
    y = truth.ravel().astype(np.float64)

    recal_months = list(range(RECAL_FROM, RECAL_TO + 1))
    if a.smoke:
        recal_months = recal_months[:a.smoke]
    score_months = list(range(LEADIN_START, BASE_ORIGIN + N_ORIGINS))

    # ---- LOCK 8: PAIRED baseline on THIS pipeline, before touching anything ----
    if not a.smoke:
        t0 = time.time()
        vol_s, _, _ = build_volume(df, score_months)
        gb = score(model, cfg, vol_s, score_months, lri, lci)
        _, b_before = ml_slope(lg(gb.ravel().astype(np.float64)), y)
        print(f"PAIRED BASELINE  b_before = {b_before:.4f}   expect {B_EXPECT:.4f}   "
              f"({time.time()-t0:.0f}s)")
        if abs(b_before - B_EXPECT) > 0.002:
            raise SystemExit(f"⛔ F3' PRECONDITION FAILS: {b_before:.4f} != {B_EXPECT:.4f}. "
                             "The pipeline does not reproduce the finding; nothing downstream "
                             "is interpretable. STOP.")

    # ---- re-estimate BN on the inference geometry (LOCKS 2,3,4,5,6,7) ----
    vol_r, _, _ = build_volume(df, recal_months)
    n_bn = _reset_bn_stats(model)                          # LOCK 4
    model.eval()
    model.set_locked_dropout(True)           # LOCK 5: dropout ON
    set_bn_train(model, True)                              # BN accumulates
    model.reset_locked_dropout()
    print(f"re-estimating {n_bn} BN layers over {len(recal_months)} full-grid "
          f"{vol_r.shape[-2]}x{vol_r.shape[-1]} months (dropout ON, no augmentation)")
    t0 = time.time()
    with torch.no_grad():                                  # LOCK 7
        h = torch.zeros(1, cfg["total_hidden_channels"], *vol_r.shape[-2:])
        for t, _fed in enumerate(recal_months):
            out = model(vol_r[:, t], h)
            h = out.h_next
            if t and t % 25 == 0:
                print(f"    {t}/{len(recal_months)}  {time.time()-t0:.0f}s elapsed")
    dt = time.time() - t0
    per = dt / len(recal_months)
    print(f"re-estimation: {dt:.0f}s for {len(recal_months)} months ({per:.2f}s/month)")

    if a.smoke:
        full = per * (RECAL_TO - RECAL_FROM + 1)
        print(f"\n⚠️  SMOKE ONLY. Projected full re-estimation: {full/60:.1f} min "
              f"({RECAL_TO-RECAL_FROM+1} months). Nothing re-scored, nothing concluded.")
        return

    set_bn_train(model, False)
    _assert_bn_buffers_finite(model, "E6f inference-distribution re-estimation")  # MANDATORY
    print("✅ BN buffers finite")

    ga = score(model, cfg, vol_s, score_months, lri, lci)
    _, b_after = ml_slope(lg(ga.ravel().astype(np.float64)), y)

    print(f"\n{'':<28}{'b':>9}{'|b-1|':>9}")
    print(f"{'before (as shipped)':<28}{b_before:>9.4f}{abs(b_before-1):>9.4f}")
    print(f"{'after BN re-estimation':<28}{b_after:>9.4f}{abs(b_after-1):>9.4f}")
    print(f"\nF1-quad EQUIVALENCE band: |b-1| <= 2*sd = {2*SD_B:.4f}")
    if abs(b_after - 1) <= 2 * SD_B:
        print("⛔ H1 IS THE CAUSE — the degenerate recal sample is the defect; "
              "remedy is a bounded one-line change.")
    elif abs(b_after - b_before) < 2 * SD_B:
        print("⛔ H1's REMEDY FAILS — b unmoved. H2 (weight_decay 0.1) or H3.")
    else:
        print("⛔ PARTWAY — improved but NOT equivalent to 1.0. Neither candidate named. "
              "This is the outcome the equivalence test exists to catch. STOP; report to Simon.")

    E = np.array([0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01])
    print("\nSIGNED per-bin curve (§7 requires signed alongside absolute)")
    print(f"  {'bin':<14}{'n':>8}{'gate':>9}{'obs':>9}{'signed':>9}  (after re-estimation)")
    sa = ga.ravel().astype(np.float64)
    for lo, hi in zip(E[:-1], E[1:]):
        m = (sa >= lo) & (sa < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f}){'':<4}{m.sum():>8}{sa[m].mean():>9.4f}"
              f"{y[m].mean():>9.4f}{sa[m].mean()-y[m].mean():>+9.4f}")


if __name__ == "__main__":
    main()
