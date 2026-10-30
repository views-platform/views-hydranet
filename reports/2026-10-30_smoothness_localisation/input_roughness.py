"""Q1/B2a — what is the roughness of the field the network is ACTUALLY FED?

WHY THIS EXISTS, AND IT IS A FALSIFICATION TEST OF MY OWN PUBLISHED CLAIM
-------------------------------------------------------------------------
`M92` finding (1) states: *"the encoder is NOT the culprit — its full-resolution features are
ROUGHER than the truth field: rho(1) = +0.443 (`enc_conv0`) and +0.314 (`e0s`) against a truth
field at 0.499."*

⛔ **That 0.499 is the wrong comparator and I used it anyway.** `d12b_curve_readout.out:94` labels
it *"truth log mean, window"* — a **36-month TEMPORAL MEAN** of log truth. The 0.638 beside it is a
**336-month** mean. ⛔ **The network is fed ONE MONTH at a time** (`vehicle.py:301-307` writes
`log1p` of a single month's counts into the input channels).

⇒ A single month of `ged_sb_best` is **99.66% zeros**. Temporal averaging smooths, so single-month
rho(1) must be **BELOW** 0.499 — and if it is also **below the encoder's 0.314**, then
**`M92` finding (1) is REFUTED as worded** and the encoder is *smoother* than its own input, not
rougher.

⭐⭐ **And this closes the hole `W-114` names.** `stage_roughness.py`'s `STAGES` begins at
`enc_conv0`, so it never hooked the input — a quantity P-12 declared and the instrument could not
collect. The fed volume **is** the log1p truth field, so measuring it here measures the network's
input directly.

WHAT IS MEASURED  (C-351 slots)
-------------------------------
  population      the same 360x720 raster, 64,818 land cells; months 457-469 (out of sample),
                  and the 36-month window 421-456 for the like-for-like comparison with d12b.
  truth           `ged_sb_best` counts. ⭐ **BOTH transforms reported**: raw counts and `log1p`
                  (the latter is what the network is fed, `build_volume`).
  estimator       Pearson rho of lag-`d` **land-land** neighbour pairs, E-W and N-S pooled.
                  ⛔ Non-land is zero in the production raster; a constant ocean would drive
                  rho -> 1, so only pairs with BOTH cells on land are scored.
  inference mode  ⭐ **NOT APPLICABLE — no model is loaded.** Data read only, zero forward passes.
  free parameter  ⭐ **the temporal aggregation**, swept: single month (13 of them, reported
                  individually and as a mean), the 13-month mean, and the 36-month window mean.
                  This is the parameter whose value I quoted without sweeping (D-4), and it is
                  the whole reason this script exists.
                  Plus the lag `d` over {1,2,4,8,16,32}.

PRE-REGISTERED PREDICTIONS — declared here, committed before this runs (D-5)
----------------------------------------------------------------------------
  P-16  single-month `log1p` truth rho(1) is **BELOW** the 36-month window's **0.499**, because
        temporal averaging smooths. Declared although near-certain: it is the premise of P-17.
  P-17  ⭐⭐⭐ **THE ONE THAT MATTERS, AND IT IS AGAINST MY OWN PUBLISHED CLAIM.** Single-month
        `log1p` truth rho(1) is **BELOW the encoder's +0.314** (`e0s`), which would **REFUTE
        `M92` finding (1) as worded**. Reasoning, stated before the run: for a near-binary field
        at base rate p = 0.0034, `rho(1) = (P(both) - p^2) / (p(1-p))`, so even strong clustering
        at `P(neighbour is event | event) = 0.15` gives only **rho(1) ~= 0.147**.
        ⇒ **I expect to have to correct `M92`.**
  P-18  The **raw-count** rho(1) is LOWER than the `log1p` rho(1) at every aggregation, because
        `log1p` compresses the extreme events that otherwise dominate the covariance.
  P-19  The 36-month window mean of `log1p` truth reproduces d12b's **+0.499** at d=1 to within
        **0.05**. ⚠️ A CONSISTENCY check, not a copy gate — d12b's land set, masking and E-W/N-S
        pooling are not verified identical to this script's, so a miss is informative about the
        comparison rather than fatal.

GATES, declared before the run
------------------------------
  G-17 ⭐⭐ POSITIVE AND NEGATIVE CONTROL on the rho estimator, reusing the hand cases that gated
       `stage_roughness.py`: ramp -> rho(1) > +0.99; checkerboard -> rho(1) < -0.99; iid noise ->
       |rho(1)| < 0.02.
  G-18 ⭐ THE MASK IS APPLIED — asserted STRUCTURALLY, on the pair count.
       ⛔⛔ **AMENDMENT 1, after G-18 FAILED on its first form (`W-118`).** It originally demanded
       that the unmasked rho(1) be *markedly higher* than the masked one. Measured: **+0.5630
       unmasked vs +0.5671 masked**, a gap of **-0.0041** — so it failed, and it could never have
       passed **on this field**. The reason is the finding: `ged_sb_best` is ~99.66% zeros **on
       land as well as on ocean**, so land and ocean are statistically alike and excluding ocean
       cannot move a correlation. ⇒ The original form asserted an **EQUALITY-shaped effect where
       only a bound exists** — the fourth instance of the `W-93`/`W-96`/`W-110` family, and the
       second time I have done it in two days.
       ⚠️ **The gate is weakened deliberately and the weakening is recorded, not hidden.** It now
       asserts the structural property that is actually load-bearing — **the mask excludes pairs**,
       i.e. `scored < total`, strictly — and the two rho values are **printed side by side with
       their gap** so the reader sees the size of the effect rather than a verdict (`W-96`).
       ⚠️ ⛔ **This does NOT retire the mask in `stage_roughness.py`.** There it excluded ~76% of
       pairs on a field where the network emits a nonzero baseline on land and something else on
       ocean, and whether it moved rho there is **unmeasured**.
  G-19 STRUCTURAL. The positive count in the scored window must reproduce `M89`/`M90`'s
       **2,905** positives over months 457-469 on the 64,818 land cells, asserted exactly.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
# ⛔ REPO FIRST (C-347): for a script `sys.path[0]` is the script's own
# directory, so `views_hydranet` would otherwise resolve to the installed
# wheel while `vehicle` comes from the working tree. `vehicle.py` now
# asserts the admitted architectures are importable and fails loudly.
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(D4))

from vehicle import TARGETS, build_volume, pull  # noqa: E402

OS_LO, OS_HI = 457, 469
WIN_LO, WIN_HI = 421, 456
LAGS = (1, 2, 4, 8, 16, 32)
D12B_WINDOW_RHO1 = 0.499
E0S_RHO1 = 0.313470          # stage_roughness.out, dropout off, the skip branch
ENC0_RHO1 = 0.443300         # stage_roughness.out, dropout off, enc_conv0
M89_POSITIVES = 2905


def pair_rho(f: np.ndarray, mask: np.ndarray, d: int) -> tuple[float, int]:
    """Pearson rho of lag-`d` land-land neighbour pairs, E-W and N-S pooled. `f` is [H, W]."""
    ew = mask[:, :-d] & mask[:, d:]
    ns = mask[:-d, :] & mask[d:, :]
    npair = int(ew.sum() + ns.sum())
    if npair == 0:
        return float("nan"), 0
    a = np.concatenate((f[:, :-d][ew], f[:-d, :][ns]))
    b = np.concatenate((f[:, d:][ew], f[d:, :][ns]))
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return (float((a * b).sum() / den) if den > 0 else float("nan")), npair


# ------------------------------------------------------------- G-17, before any data is read
rng = np.random.default_rng(15)
H, W = 64, 64
m_all = np.ones((H, W), dtype=bool)
r_ramp = pair_rho(np.tile(np.arange(W, dtype=np.float64), (H, 1)), m_all, 1)[0]
r_board = pair_rho(((np.arange(H)[:, None] + np.arange(W)[None]) % 2) * 2.0 - 1.0, m_all, 1)[0]
r_noise = pair_rho(rng.standard_normal((H, W)), m_all, 1)[0]
ok17 = r_ramp > 0.99 and r_board < -0.99 and abs(r_noise) < 0.02
print(f"{'PASS' if ok17 else '⛔ FAIL'}  G-17  rho controls: ramp {r_ramp:+.4f}  "
      f"checkerboard {r_board:+.4f}  iid noise {r_noise:+.4f}")
if not ok17:
    sys.exit(1)

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
months = list(range(WIN_LO, OS_HI + 1))
vol, land, _ = build_volume(df, months)      # vol is log1p, [1, T, 3, H, W] — what the net is fed
GR, GC = land.shape
sb = 0
LOG = vol[0, :, sb].numpy().astype(np.float64)
RAW = np.expm1(LOG)
os_idx = [i for i, m in enumerate(months) if OS_LO <= m <= OS_HI]
win_idx = [i for i, m in enumerate(months) if WIN_LO <= m <= WIN_HI]

pos = int((RAW[os_idx][:, land] > 0.5).sum())
ok19 = pos == M89_POSITIVES
print(f"{'PASS' if ok19 else '⛔ FAIL'}  G-19  positives over months {OS_LO}-{OS_HI} on "
      f"{int(land.sum()):,} land cells: {pos:,} (M89/M90 committed {M89_POSITIVES:,})")
if not ok19:
    print("⛔⛔ G-19 FAILED — this is not the population every other Q1/Q2 number was built on.")
    sys.exit(1)

m_rho, m_pairs = pair_rho(LOG[os_idx].mean(0), land, 1)
u_rho, u_pairs = pair_rho(LOG[os_idx].mean(0), np.ones_like(land), 1)
ok18 = 0 < m_pairs < u_pairs
print(f"{'PASS' if ok18 else '⛔ FAIL'}  G-18  the mask EXCLUDES pairs (structural): "
      f"{m_pairs:,} scored of {u_pairs:,} total ({100.0 * m_pairs / u_pairs:.1f}%)")
if not ok18:
    print("⛔⛔ G-18 FAILED — the mask excludes nothing, so it is not applied at all.")
    sys.exit(1)
print(f"       ⚠️ and its EFFECT on rho(1) is small on this field: masked {m_rho:+.6f} vs "
      f"unmasked {u_rho:+.6f}, gap {u_rho - m_rho:+.6f}")
print("       ⇒ ged_sb_best is ~99.66% zeros on LAND as well as ocean, so the two are")
print("         statistically alike. G-18's first form demanded a large gap and FAILED; it")
print("         could not have passed on this field. Amendment 1, logged as W-118.")

print("\n" + "=" * 110)
print("Q1/B2a — THE ROUGHNESS OF THE FIELD THE NETWORK IS FED")
print("=" * 110)
print(f"  raster          {GR}x{GC}, {int(land.sum()):,} land cells")
print(f"  truth           ged_sb_best ({TARGETS[sb]}), BOTH transforms reported")
print("  inference mode  ⭐ NOT APPLICABLE — no model loaded, zero forward passes")
print(f"  lags            {LAGS}   (D-4, alongside the temporal aggregation below)")

AGG = (
    ("single month, mean over 457-469", "per_month", os_idx),
    ("13-month mean  457-469",          "mean",      os_idx),
    ("36-month mean  421-456",          "mean",      win_idx),
)
table: dict[tuple[str, str], dict[int, float]] = {}
for label, how, idx in AGG:
    for tname, F in (("log1p", LOG), ("raw", RAW)):
        row = {}
        for d in LAGS:
            if how == "per_month":
                row[d] = float(np.mean([pair_rho(F[i], land, d)[0] for i in idx]))
            else:
                row[d] = pair_rho(F[idx].mean(0), land, d)[0]
        table[(label, tname)] = row

print("\n" + "=" * 110)
print("1. rho(d) BY TEMPORAL AGGREGATION — the parameter I quoted without sweeping")
print("=" * 110)
print(f"  {'aggregation':<34s}{'transform':>10s}" + "".join(f"{f'rho({d})':>10s}" for d in LAGS))
for label, _, _ in AGG:
    for tname in ("log1p", "raw"):
        r = table[(label, tname)]
        print(f"  {label:<34s}{tname:>10s}" + "".join(f"{r[d]:>+10.4f}" for d in LAGS))

print("\n" + "=" * 110)
print("2. ⭐⭐⭐ THE DECISIVE COMPARISON — against M92 finding (1)")
print("=" * 110)
sm = table[("single month, mean over 457-469", "log1p")][1]
wn = table[("36-month mean  421-456", "log1p")][1]
print(f"  single-month log1p truth rho(1)   {sm:+.6f}   ⇐ what the network is ACTUALLY FED")
print(f"  36-month window mean rho(1)       {wn:+.6f}   (d12b committed {D12B_WINDOW_RHO1:+.3f})")
print(f"  encoder e0s rho(1)                {E0S_RHO1:+.6f}   (stage_roughness.out, dropout off)")
print(f"  encoder enc_conv0 rho(1)          {ENC0_RHO1:+.6f}")
print(f"\n  P-16  single month below the window's {D12B_WINDOW_RHO1:+.3f}: "
      f"{'HELD' if sm < D12B_WINDOW_RHO1 else '⛔ REFUTED'}  ({sm:+.4f})")
print(f"  P-19  window reproduces d12b to within 0.05: "
      f"{'HELD' if abs(wn - D12B_WINDOW_RHO1) < 0.05 else '⚠️ MISSED'}  "
      f"(|{wn:+.4f} - {D12B_WINDOW_RHO1:+.3f}| = {abs(wn - D12B_WINDOW_RHO1):.4f})")
if sm < E0S_RHO1:
    print("\n  ⛔⛔ P-17 HELD ⇒ M92 FINDING (1) IS REFUTED AS WORDED.")
    print(f"      The fed field has rho(1) {sm:+.4f}, BELOW the encoder's e0s at {E0S_RHO1:+.4f}.")
    print("      ⇒ The encoder is SMOOTHER than its own input, not rougher than 'the truth")
    print("      field'. The 0.499 I compared against is a 36-month TEMPORAL MEAN and was the")
    print("      wrong comparator.")
else:
    print("\n  ⭐⭐ P-17 REFUTED ⇒ M92 FINDING (1) STANDS, on the RIGHT comparator.")
    print(f"      The fed field has rho(1) {sm:+.4f}, at or above the encoder's {E0S_RHO1:+.4f}.")
p18 = all(table[(lab, "raw")][1] < table[(lab, "log1p")][1] for lab, _, _ in AGG)
print(f"  P-18  raw below log1p at every aggregation: {'HELD' if p18 else '⛔ REFUTED'}")

print("\n" + "=" * 110)
print("⚠️ LIMITS")
print("=" * 110)
print("  · ⭐ NO MODEL. Data read only — nothing depends on a checkpoint or an inference mode.")
print("  · ⛔ A single month of ged_sb_best is ~99.66% zeros, so its rho is the clustering")
print("    of a very sparse point pattern. A conditional-MEAN field SHOULD be smoother than a")
print("    realisation (W-101's identity), so 'the model is smoother than one month of truth' is")
print("    NOT by itself a defect — which is exactly why the comparator matters.")
print("  · ⚠️ d12b's land set, masking rule and E-W/N-S pooling are not verified byte-identical to")
print("    this script's, so P-19 is a CONSISTENCY check and no copy gate is claimed (W-98).")
print("  · sb only.")
print(f"\n  wall time {time.time() - T0:.1f} s   peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
