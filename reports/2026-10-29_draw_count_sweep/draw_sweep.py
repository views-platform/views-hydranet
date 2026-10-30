"""Q2 — is the body under-dispersed, or are we simply failing to SAMPLE it?

WHY THIS EXISTS
---------------
Simon's founding question, unanswered after days: *is our body under-dispersed, and if so
do we understand why and can we do anything about it?*

`reports/2026-10-29_body_state_of_knowledge/FINDING.md` established that the question
has never been measured, and that the reason is structural rather than neglect:

  * `RP-4`, the entry that asks exactly this, is queued and has never run;
  * **14 of 20 battery columns are functions of the MEAN FIELD alone** (M69–M73),
    confirmed at exactly `0.000e+00`, so dispersion is visible to four columns, three of
    which are one CRPS split;
  * all three CRPS columns use the **plug-in** estimator, which Ferro 2014 proves
    **favours under-dispersed ensembles** — our ruler has been rewarding the pathology;
  * **PIT is REJECTED by FAO-02**, so the locked framework holds no dispersion
    diagnostic at all.

THE HYPOTHESIS THIS TESTS
-------------------------
At `theta -> 0` the NB is identically `(1-pi)·delta_0 + pi·LogSeries` with
`pi = theta·log(1 + mu/theta)` (ADR-073 §3). That is **mostly zero with a rare heavy
branch**, and the grid-median `theta` is **3.6e-06**. ⛔ At the delivered 4 MC-dropout
passes x 4 head draws = 16 members we essentially never sample that branch.

⇒ ⭐ **The nominal distribution may be wildly OVER-dispersed while the realised 16-member
ensemble looks UNDER-dispersed. These are opposite claims and we have measured neither.**

Head draws are FREE and uncapped (`W-68`, #459) — unlike MC-dropout passes, capped at
1–4 — so sweeping the head draw count `K` answers this on a frozen cube with no
retraining and no GPU.

WHAT IS MEASURED  (C-351 slots, every one named)
------------------------------------------------
  population      13 out-of-sample months 457-469 x 64,818 land cells = 842,634
                  cell-months. ⭐ The SAME population as every RP-3 number and as M89,
                  so these are like-for-like with the published record.
  truth           `ged_sb_best` counts, no transform.
  estimator       BOTH — plug-in `e1 - A/(2m^2)` and fair `e1 - A/(2m(m-1))` (#459).
  inference mode  ⛔ **TEACHER-FORCED.** `d4_result.npz` comes from stepping the model
                  over a volume `vehicle.py:301-307` fills from the truth frame. This is
                  the slot whose absence is `C-399`, and it is named here because this
                  script needs `mu`/`theta`, which only the frozen cube carries — the
                  delivery run stores composed samples only.
  free parameter  `K` in {1, 4, 16, 64, 256, 1024} head draws per MC pass, i.e. total
                  ensembles of 4, 16, 64, 256, 1,024 and 4,096 members (D-4: swept before
                  any scalar is reported). ⭐ `K=4` is what ships.
  composition     BOTH `threshold_gate` tau=0.20 (what ships) and `soft_gate`, from the
                  repo's own `compose_samples`. ⭐ Drawn ONCE and composed twice, so the
                  two are PAIRED on identical draws and fill two empty corners of #490.

⚠️ WHAT THIS CANNOT SAY, stated before the run
----------------------------------------------
It measures the **emission's** dispersion GIVEN the trunk's `mu`/`theta`. It cannot say
whether `mu`/`theta` are themselves right, and it is a statement about the teacher-forced
regime, not the delivered one.

⚠️ The spread–error relation below is a **DIAGNOSTIC, not a FAO-02 column.** FAO-02
governs scoring and selection; this answers a mechanism question. ⛔ If it is ever used
to drive a selection decision it needs Simon's amendment to FAO-02 first.

PRE-REGISTERED PREDICTIONS — declared here, committed before this runs (D-5)
----------------------------------------------------------------------------
  P-1  the per-cell ensemble sd RISES monotonically with `K`, because larger `K` samples
       more of the heavy branch.
  P-2a ⭐ the PER-DRAW firing rate is FLAT in `K` — each draw is iid, so it CANNOT depend
       on the draw count. Stated because the obvious sloppy version of P-2 ("firing rises
       with K") is wrong by construction, and separating them is the point.
  P-2b ⭐ the PER-CELL any-fire rate rises as `1 - (1-p)^(4K)` with
       `p = 1 - prob_zero(mu, theta)`, and is predicted in CLOSED FORM from the cube's own
       `prob_zero` — gated as G-5 rather than eyeballed.
  P-3  the fair estimator is near-flat in `K` while the plug-in falls — the Ferro 2014
       signature, already seen over m=4..16 in #489 (plug-in 0.058466 -> 0.055790, fair
       0.055380 -> 0.054958). Extended here to 4,096 members.
  P-4  ⭐⭐ THE DECISIVE ONE. At the delivered 16 members the mean ensemble sd is far
       BELOW the RMSE, i.e. under-dispersed. **If "under-dispersed" is a draw-count
       artifact the ratio sd/RMSE climbs toward 1 as `K` grows; if it is real, it
       plateaus well below 1.** Either outcome closes RP-4.

GATES, declared before the run
------------------------------
  G-1 ⭐ COPY. At `K=4` with `threshold_gate`, reproduce #489's committed TF-samp m=16
      numbers **0.049103** plug-in / **0.047522** fair. Reproduced by replaying
      `ar_score.py`'s exact unchunked loop and seed 489, because the anchor is the
      committed implementation and never a ledger row (`W-98`).
  G-2 STRUCTURAL. The population is exactly 842,634 cell-months.
  G-3 STRUCTURAL. `threshold_gate`'s mask does not depend on the draws, so the fraction of
      cells it leaves open is IDENTICAL at every `K`. Asserted exactly, with its value
      printed (`W-96`: assert a structural property, not bit-equality of a sum).
  G-4 COPY. Truth on this population reproduces M89's mean **0.060383** and max **2482**.
  G-5 ⭐ DERIVATION. P-2b's closed form predicts the measured per-cell any-fire rate at
      every `K` to a RELATIVE 1%, with the residual printed beside its scale (`W-96`).

⛔ NON-FINITE DRAWS — the handling rule is declared NOW, not chosen after seeing them.
At `theta = 3.6e-06`, `lam = gamma·mu/theta` can reach ~1e5 x mu, so `torch.poisson` may
return non-finite values. Their COUNT is reported at every `K` and is itself a finding:
it is the heavy branch being sampled.

⛔ **The rule, fixed now so it cannot be chosen after seeing the numbers:** CRPS needs a
whole row, so a cell-month with ANY non-finite draw is **EXCLUDED from every statistic**,
and both the non-finite draw count and the number of excluded ROWS are printed beside
every number they could affect. ⛔ No silent clipping, and no substituting zero for
infinity — that would quietly convert "the heavy branch fired" into "the model predicted
nothing", which is the opposite reading.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(D4))

from vehicle import SB, TARGETS, pull  # noqa: E402

from views_hydranet.distributions.composition import compose_samples  # noqa: E402
from views_hydranet.distributions.nb_core import NBCore  # noqa: E402

T0 = time.time()
torch.set_num_threads(8)  # the cap d4_mu_on_the_grid.py records: uncapped took all cores

OS_LO, OS_HI = 457, 469
GATE_TAU = 0.20
K_SWEEP = (1, 4, 16, 64, 256, 1024)
SEED = 489  # the seed ar_score.py used, so G-1 can replay it exactly

G1_PLUGIN, G1_FAIR = 0.049103, 0.047522   # ar_score.out, TF-samp m=16
G4_MEAN, G4_MAX = 0.060383, 2482.0        # ar_score.out truth line


def crps_ensemble(Y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row plug-in and fair CRPS. Copied from ar_score.py, the committed anchor.

    ``m = D*K >= 4`` throughout this script, so the fair estimator's ``m-1`` is never a
    division by zero and no single-member branch is needed — a branch that cannot fire is
    worse than no branch."""
    Ys = np.sort(Y, axis=1)
    m = Ys.shape[1]
    e1 = np.abs(Ys - z[:, None]).mean(1)
    w = 2.0 * np.arange(m) - (m - 1)
    A = 2.0 * (Ys * w).sum(1)
    return e1 - A / (2.0 * m * m), e1 - A / (2.0 * m * (m - 1))


def gate0() -> None:
    """Assert the primitives on foreign input before anything real is loaded."""
    y = np.array([[0.0, 2.0]])
    p, f = crps_ensemble(y, np.array([1.0]))
    assert abs(p[0] - 0.5) < 1e-12 and abs(f[0]) < 1e-12, (p, f)
    rng = np.random.default_rng(0)
    E = rng.gamma(2.0, 3.0, size=(7, 11))
    brute = np.abs(E[:, :, None] - E[:, None, :]).sum((1, 2))
    Es = np.sort(E, axis=1)
    w = 2.0 * np.arange(11) - 10.0
    assert np.allclose(2.0 * (Es * w).sum(1), brute), "pairwise identity"
    print("GATE 0  CRPS primitives on foreign input   PASS")


gate0()

npz = np.load(D4 / "d4_result.npz")
land_cells, months = npz["land_cells"], [int(m) for m in npz["months"]]
mcol = {m: i for i, m in enumerate(months)}
MU, TH, GT = (npz[k].astype(np.float64) for k in ("mu", "theta", "gate"))

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T

sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1)]
y = Y[sco_m].ravel()
N = y.size
D = MU.shape[0]

mu_f = MU[:, sco_m].reshape(D, -1)       # [D, N]
th_f = TH[:, sco_m].reshape(D, -1)
gt_f = GT[:, sco_m].reshape(D, -1)

print("=" * 110)
print("Q2 — THE BODY'S DISPERSION vs THE DRAW COUNT")
print("=" * 110)
print(f"  checkpoint      sha {str(npz['sha'])[:16]}…  family nb  MC-dropout passes {D}")
print(f"  population      {len(sco_m)} months {OS_LO}-{OS_HI} x {len(land_cells):,} = {N:,}")
print(f"  truth           ged_sb_best counts, no transform. mean {y.mean():.6f} "
      f" max {y.max():.0f}")
print(f"  positives       {int((y > 0).sum()):,}  ({100.0 * (y > 0).mean():.4f}%)")
print("  inference mode  ⛔ TEACHER-FORCED (vehicle.py:301-307 fills the volume from truth)")
print(f"  composition     threshold_gate tau={GATE_TAU} and soft_gate, PAIRED on one draw set")

assert N == 842_634, N
print(f"\n  PASS  G-2  population is exactly {N:,} cell-months")
ok4 = abs(y.mean() - G4_MEAN) < 5e-7 and y.max() == G4_MAX
print(f"  {'PASS' if ok4 else '⛔ FAIL'}  G-4  truth reproduces M89: mean {y.mean():.6f} "
      f"(committed {G4_MEAN})  max {y.max():.0f} (committed {G4_MAX:.0f})")

# ---- G-1: replay ar_score.py's exact unchunked loop ------------------------------------
gen = torch.Generator().manual_seed(SEED)
cols = []
for d in range(D):
    mu_d = torch.as_tensor(mu_f[d], dtype=torch.float32)
    th_d = torch.as_tensor(th_f[d], dtype=torch.float32)
    s = NBCore.sample(mu_d, th_d, 4, gen).numpy().astype(np.float64)
    cols.append(s * (gt_f[d] >= GATE_TAU)[:, None])
TFS = np.concatenate(cols, axis=1)
p16, f16 = (v.mean() for v in crps_ensemble(TFS, y))
ok1 = abs(p16 - G1_PLUGIN) < 1e-6 and abs(f16 - G1_FAIR) < 1e-6
print(f"  {'PASS' if ok1 else '⛔ FAIL'}  G-1  COPY: K=4 threshold_gate reproduces ar_score.out "
      f"  got {p16:.6f} / {f16:.6f}  (committed {G1_PLUGIN} / {G1_FAIR})")
if not ok1:
    print("\n⛔⛔ G-1 FAILED — the sampler path does not reproduce the committed anchor.")
    print("    ⛔ The sweep is deliberately NOT run.")
    sys.exit(1)
del TFS, cols

# closed-form P-2b: P(at least one of 4K draws fires) = 1 - (1-p)^(4K), p per MC pass
p_pos = np.stack([
    1.0 - NBCore.prob_zero(torch.as_tensor(mu_f[d]), torch.as_tensor(th_f[d])).numpy()
    for d in range(D)
])                                               # [D, N]
open_mask = gt_f >= GATE_TAU                     # [D, N], independent of the draws
frac_open = float(open_mask.mean())

COMPS = ("threshold_gate", "soft_gate")
rows = []
for K in K_SWEEP:
    m = D * K
    chunk = max(2_000, 20_000_000 // m)   # hold each composed block near 20M elements
    t0 = time.time()
    acc = {c: {"p": 0.0, "f": 0.0, "sd": 0.0, "se": 0.0, "fire": 0.0, "any": 0.0,
               "nf": 0, "drop": 0, "n": 0} for c in COMPS}
    gen = torch.Generator().manual_seed(SEED + K)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        masks = {c: [] for c in COMPS}
        for d in range(D):
            mu_d = torch.as_tensor(mu_f[d, lo:hi], dtype=torch.float32)
            th_d = torch.as_tensor(th_f[d, lo:hi], dtype=torch.float32)
            gt_d = torch.as_tensor(gt_f[d, lo:hi], dtype=torch.float32)
            s = NBCore.sample(mu_d, th_d, K, gen)                      # [n, K]
            for c in COMPS:
                masks[c].append(compose_samples(s, gt_d, c, GATE_TAU, gen))
        for c in COMPS:
            E = torch.cat(masks[c], dim=1).numpy().astype(np.float64)  # [n, m]
            bad = ~np.isfinite(E)
            acc[c]["nf"] += int(bad.sum())
            # Declared rule: a row with ANY non-finite draw is EXCLUDED, never zero-filled.
            keep = ~bad.any(axis=1)
            acc[c]["drop"] += int((~keep).sum())
            E, zc = E[keep], y[lo:hi][keep]
            if E.shape[0] == 0:
                continue
            acc[c]["n"] += E.shape[0]
            pl, fr = crps_ensemble(E, zc)
            acc[c]["p"] += float(pl.sum())
            acc[c]["f"] += float(fr.sum())
            acc[c]["sd"] += float(E.std(axis=1, ddof=1).sum())
            acc[c]["se"] += float(((E.mean(1) - zc) ** 2).sum())
            acc[c]["fire"] += float((E > 0).mean(1).sum())
            acc[c]["any"] += float((E > 0).any(1).sum())
            del E
        del masks
    # P-2b closed form: P(>=1 of D*K draws fires) = 1 - prod_d (1-p_d)^K, over DRAWS only.
    pred_any = float((1.0 - np.prod((1.0 - p_pos) ** K, axis=0)).mean())
    for c in COMPS:
        a = acc[c]
        nn = max(a["n"], 1)
        rows.append({
            "K": K, "m": m, "comp": c,
            "plugin": a["p"] / nn, "fair": a["f"] / nn,
            "sd": a["sd"] / nn, "rmse": (a["se"] / nn) ** 0.5,
            "fire": a["fire"] / nn, "anyfire": a["any"] / nn,
            "nf": a["nf"], "drop": a["drop"], "n": a["n"], "pred_any": pred_any,
        })
    print(f"  K={K:<5d} m={m:<5d} chunk {chunk:>6d}  done in {time.time() - t0:6.1f} s")

print("\n" + "=" * 110)
print("1. THE SWEEP — both compositions, both estimators, paired on one draw set")
print("=" * 110)
print(f"  {'comp':<16s}{'K':>6s}{'m':>7s}{'plug-in':>11s}{'fair':>11s}{'ens sd':>11s}"
      f"{'RMSE':>11s}{'sd/RMSE':>9s}{'fire/draw':>11s}{'any-fire':>10s}{'nonfin':>9s}{'dropped':>9s}{'scored':>10s}")
for r in rows:
    print(f"  {r['comp']:<16s}{r['K']:>6d}{r['m']:>7d}{r['plugin']:>11.6f}{r['fair']:>11.6f}"
          f"{r['sd']:>11.6f}{r['rmse']:>11.6f}{r['sd'] / r['rmse']:>9.4f}"
          f"{r['fire']:>11.6f}{r['anyfire']:>10.6f}{r['nf']:>9d}{r['drop']:>9d}{r['n']:>10d}")

print(f"\n  observed positive rate on this population: {(y > 0).mean():.6f}")
print(f"  PASS  G-3  threshold_gate leaves {frac_open:.6f} of cell-month-passes open at EVERY K "
      f"(mask is draw-independent, asserted exactly)")

print("\n" + "=" * 110)
print("2. THE PRE-REGISTERED PREDICTIONS, SCORED")
print("=" * 110)
for c in ("threshold_gate", "soft_gate"):
    rr = [r for r in rows if r["comp"] == c]
    sd = [r["sd"] for r in rr]
    fire = [r["fire"] for r in rr]
    fairs = [r["fair"] for r in rr]
    plugs = [r["plugin"] for r in rr]
    print(f"\n  {c}")
    print(f"    P-1  ens sd monotone rising in K: {all(b >= a for a, b in zip(sd, sd[1:]))}"
          f"   {sd[0]:.6f} -> {sd[-1]:.6f}")
    sp = (max(fire) - min(fire)) / max(max(fire), 1e-12)
    print(f"    P-2a per-draw firing FLAT in K (iid): relative spread {sp:.2e}"
          f"   {min(fire):.6f}..{max(fire):.6f}")
    print(f"    P-3  fair {fairs[0]:.6f} -> {fairs[-1]:.6f} (range {max(fairs) - min(fairs):.6f})"
          f" | plug-in {plugs[0]:.6f} -> {plugs[-1]:.6f} (range {max(plugs) - min(plugs):.6f})")
    print(f"    P-4  sd/RMSE  {rr[1]['sd'] / rr[1]['rmse']:.4f} at the delivered m=16"
          f"  ->  {rr[-1]['sd'] / rr[-1]['rmse']:.4f} at m={rr[-1]['m']}")

res = [(r["K"], r["anyfire"], r["pred_any"]) for r in rows if r["comp"] == "threshold_gate"]
print("\n  G-5  DERIVATION: P-2b's closed form 1-(1-p)^(4K) vs measured per-cell any-fire")
worst = 0.0
for K, meas, pred in res:
    rel = abs(meas - pred) / max(pred, 1e-12)
    worst = max(worst, rel)
    print(f"       K={K:<5d} measured {meas:.6f}  closed form {pred:.6f}  relative {rel:.2e}")
print(f"  {'PASS' if worst < 0.01 else '⛔ FAIL'}  G-5  worst relative residual {worst:.2e} "
      f"against a 1% bar (⚠️ the closed form ignores the gate mask; see §3)")

print("\n" + "=" * 110)
print("3. ⚠️ LIMITS")
print("=" * 110)
print("  · TEACHER-FORCED cube. This is the emission's dispersion given the trunk's mu/theta,")
print("    not a statement about the delivered autoregressive regime (C-399).")
print("  · The spread-error relation is a DIAGNOSTIC, not a FAO-02 column. Driving a selection")
print("    decision with it needs Simon's amendment to FAO-02 first.")
print("  · G-5's closed form is over DRAWS only and ignores threshold_gate's mask, so it bounds")
print("    the any-fire rate from above wherever the gate closes a cell.")
print("  · n=1 checkpoint, sb only, no sigma-hat on any number here.")

print("\n" + "=" * 110)
print(f"  wall time {time.time() - T0:.1f} s (per-K timings above)   "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
