"""STAGE 1b — FSS through the five-transform battery. prereg ../2026-10-26_fss_battery_prereg.md

⛔ Predictions are in the prereg (`571ea76`), committed before this file existed. They are NOT
restated here, so that this file cannot quietly become the record of what was predicted.

⚠️ DECLARED DEVIATION from prereg §4.3. The prereg says the transforms are "re-used verbatim from
instrument_battery.py". That module does its work at MODULE LEVEL, so importing it would run the
whole Stage 1 battery as a side effect. They are therefore COPIED verbatim, byte for byte, with
this note -- the same call `f4_convergence_amendment.py` made and for the same reason.

`fss` and `matched_pred_thresh` ARE genuinely re-used from production, and two controls below
prove the generalisations of them used here agree with production exactly.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).parent))
from fss_skill_threshold import bar, ideal_band, scale_min  # noqa: E402
from sharpness_scorecard import fss as prod_fss  # noqa: E402
from sharpness_scorecard import matched_pred_thresh as prod_thresh  # noqa: E402

ARM = Path(
    "/home/simon/views-diagnostics/hf_arm_20261026/plots/02_training_dynamics/"
    "biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
)
CTL = Path(
    "/home/simon/views-diagnostics/hf_control_20261025/diagnostics/251026_1043/"
    "02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
)
SCALES = list(range(1, 24, 2))          # prereg §2: n = 1,3,...,23 on a 24-wide window
SHOW = (1, 3, 5, 9, 15, 23)
DISP_WIN, DISP_PAD = 24, 4              # copied verbatim from instrument_battery.py
EPS = 1e-6


# --------------------------------------------------------- FSS, generalised from production
def thresh_ratio(pred, truth, r=1.0):
    """matched_pred_thresh generalised to f_M / f_o = r. r=1.0 MUST equal production (C-1)."""
    n_events = int(round(r * float((truth > 0).sum())))
    flat = pred.ravel()
    if n_events <= 0 or n_events >= flat.size:
        return float("inf")
    return float(np.partition(flat, -n_events)[-n_events] - 1e-12)


def fss_pooled(preds, truths, scales, r=1.0):
    """FBS pooled over fields BEFORE the ratio (prereg §2). Single field MUST equal prod (C-2)."""
    num = dict.fromkeys(scales, 0.0)
    den = dict.fromkeys(scales, 0.0)
    for p, t in zip(preds, truths, strict=True):
        prd = (p > thresh_ratio(p, t, r)).astype(np.float64)
        obs = (t > 0.0).astype(np.float64)
        for n in scales:
            pf = uniform_filter(prd, size=n, mode="constant")
            of = uniform_filter(obs, size=n, mode="constant")
            num[n] += float(np.mean((pf - of) ** 2))
            den[n] += float(np.mean(pf**2) + np.mean(of**2))
    return {n: (1.0 - num[n] / den[n] if den[n] > 0 else float("nan")) for n in scales}


def binary_stack(preds, truths, r=1.0):
    return np.stack([(p > thresh_ratio(p, t, r)) for p, t in zip(preds, truths, strict=True)])


# --------------------------------------------- I-2 (B-1) and R-1, copied verbatim from Stage 1
def gamma_iso(f, d):
    """⛔ ISOTROPIC — the mean of the E-W and N-S semivariance, NOT B-1a's estimator.

    ⚠️ RENAMED 2026-10-27b. It was called `gamma_ew` under a comment saying it was copied verbatim
    from Stage 1, and it is not: `instrument_battery.py:69` and `c366_two_instruments.py:93` are
    **E-W only**, as is the scoreboard's `d12` source. One name, two instruments — and the
    scoreboard quoted THIS one's displacement series as B-1a's, which inverted the evidence. On the
    same field and window the two give B-1(d=1) = 0.4465 here against 0.2727 there, and under
    displacement this one moves 0.96% while B-1a's own moves +22.4%. See `W-77`.

    ⛔ Kept rather than replaced, and renamed rather than quietly corrected: every number in
    `fss_battery.out` was produced by this function, so swapping it would silently restate a
    committed artifact. The B-1 column here is an isotropic comparator and is labelled as one.
    """
    a, b = f[:, :, d:], f[:, :, : f.shape[2] - d]
    c, e = f[:, d:, :], f[:, : f.shape[1] - d, :]
    return 0.5 * (np.mean((a - b) ** 2) + np.mean((c - e) ** 2))


def b1(pred, truth, d):
    """⚠️ B-1's FORM on the isotropic estimator above — not comparable to the scoreboard's B-1a."""
    return (gamma_iso(pred, d) / gamma_iso(truth, d)) / (pred.var() / truth.var())


def spearman_ref(pred, truth):
    return float(spearmanr(pred.ravel(), truth.ravel()).statistic)


# ------------------------------------------- transforms, copied verbatim from instrument_battery
SEED = 20261026
rng = np.random.default_rng(SEED)


def reseed() -> None:
    """Reset the module-level draw so each space sees the SAME permutations.

    ⛔ `global`, and that is the whole point. Both space loops live inside `main()`, so a bare
    `rng = np.random.default_rng(SEED)` there binds a LOCAL and leaves the module-level generator
    that `shuffle_in_blocks` reads untouched — an inert fix that reads as a real one. My first
    attempt at this correction was exactly that, and `ruff`'s F841 caught it, not me. Same class as
    the register's C-329 sub-form 4: the repair looked right and changed nothing.
    """
    global rng
    rng = np.random.default_rng(SEED)


def shuffle_in_blocks(f, k):
    n, h, w = f.shape
    b = (f.reshape(n, h // k, k, w // k, k).transpose(0, 1, 3, 2, 4)
          .reshape(n, h // k, w // k, k * k))
    idx = rng.permuted(np.broadcast_to(np.arange(k * k), b.shape), axis=-1)
    b = np.take_along_axis(b, idx, axis=-1)
    return b.reshape(n, h // k, w // k, k, k).transpose(0, 1, 3, 2, 4).reshape(n, h, w)


def smooth3(f, passes=1):
    for _ in range(passes):
        p = np.pad(f, ((0, 0), (1, 1), (1, 1)), mode="edge")
        f = sum(p[:, i:i + f.shape[1], j:j + f.shape[2]]
                for i in range(3) for j in range(3)) / 9.0
    return f


def window(pred, truth, s=0):
    """Prereg §2: ALL transforms on the SAME fixed 24x24 truth window. No wrap (F-3)."""
    o = DISP_PAD
    tp = truth[:, o:o + DISP_WIN, o:o + DISP_WIN]
    mp = pred[:, o + s:o + s + DISP_WIN, o + s:o + s + DISP_WIN]
    return np.ascontiguousarray(mp), np.ascontiguousarray(tp)


def load(p):
    z = np.load(p)
    return {k: z[k].astype(np.float64) for k in ("y_reg", "y_hat_reg", "y_hat_cls")}


def flatten(d, space):
    mv = lambda a: np.moveaxis(a, -1, 1).reshape(-1, 32, 32)  # noqa: E731
    gate, body, truth = mv(d["y_hat_cls"]), mv(d["y_hat_reg"]), mv(d["y_reg"])
    if space == "counts":
        return gate * np.expm1(body), np.expm1(truth)
    return np.log1p(gate * np.expm1(body)), truth


# ----------------------------------------------------------------------------------- reporting
def row(label, p, t, f_o, r=1.0):
    s = fss_pooled(list(p), list(t), SCALES, r=r)
    sm = scale_min(s, bar(f_o))
    cells = f"{' '.join(f'{s[n]:7.4f}' for n in SHOW)}"
    print(f"    {label:<24s} {cells}   smin {str(sm):>4s}   "
          f"B-1 {b1(p, t, 1):7.4f}   R-1 {spearman_ref(p, t):+7.4f}")
    return s, sm


def main() -> None:
    print("=" * 118)
    print("STAGE 1b — FSS THROUGH THE FIVE-TRANSFORM BATTERY")
    print("           prereg ../2026-10-26_fss_battery_prereg.md (571ea76, committed first)")
    print("=" * 118)

    # ---- F-1: the ideal-band control must reproduce scale_min = 2D
    ok = []
    for d in (1, 2, 3, 5, 8, 11):
        got = scale_min(prod_fss(*ideal_band(128, d), SCALES + [25, 27, 29], 0.5, 0.5),
                        bar(1 / 128))
        ok.append(got is not None and abs(got - 2 * d) <= 1)
    if not all(ok):
        raise SystemExit("VOID (F-1): the ideal-band scale_min = 2D control does not reproduce.")
    print("  F-1  ideal-band scale_min = 2D reproduces for D = 1,2,3,5,8,11 .......... PASS")

    ctl, arm = load(CTL), load(ARM)
    if not np.array_equal(ctl["y_reg"], arm["y_reg"]):
        raise SystemExit("VOID (F-2): the two windows do not carry the same truth.")
    print("  F-2  y_reg byte-identical across arms .................................. PASS")
    print(f"  F-3  T-4 crops to a FIXED {DISP_WIN}x{DISP_WIN} truth window, never wraps .. PASS")

    # ---- C-1 / C-2: the generalisations agree with production EXACTLY
    cp0, ct0 = flatten(ctl, "counts")
    g0, h0 = window(cp0, ct0)
    d1 = abs(thresh_ratio(g0[0], h0[0], 1.0) - prod_thresh(g0[0], h0[0]))
    pooled1 = fss_pooled([g0[0]], [h0[0]], SCALES)
    prod1 = prod_fss(g0[0], h0[0], SCALES, prod_thresh(g0[0], h0[0]))
    d2 = max(abs(pooled1[n] - prod1[n]) for n in SCALES)
    print(f"  C-1  thresh_ratio(r=1) vs production matched_pred_thresh: {d1:.3e} ...... "
          f"{'PASS' if d1 == 0.0 else 'FAIL'}")
    print(f"  C-2  fss_pooled(1 field) vs production fss:               {d2:.3e} ...... "
          f"{'PASS' if d2 == 0.0 else 'FAIL'}")

    f_o = float((h0 > 0).mean())
    print(f"\n  POPULATION: 18 pooled fields (6 steps x 3 targets), {DISP_WIN}x{DISP_WIN}, "
          f"{h0.size:,} cell-values, lesson 297 win 1, in-sample.")
    print(f"  f_o = {f_o:.6f} on THIS window (not the 0.003448 land rate)  =>  "
          f"bar = 0.5 + f_o/2 = {bar(f_o):.6f}")
    print(f"  events per field: min {int((h0 > 0).sum(axis=(1, 2)).min())}, "
          f"median {int(np.median((h0 > 0).sum(axis=(1, 2))))}, "
          f"max {int((h0 > 0).sum(axis=(1, 2)).max())}")

    for space in ("counts", "log1p"):
        # ⛔ RE-SEEDED PER SPACE, added 2026-10-27b. The module-level rng was never reset, so T-2
        # drew different permutations for counts and log1p and the committed .out shows the two
        # diverging (k=2: 0.2959 vs 0.3265) where the transform guarantees they cannot. The
        # divergence was the draw, not the space. See `W-78`.
        reseed()
        cp, ct = flatten(ctl, space)
        print()
        print("=" * 118)
        print(f"space = {space}      FSS at n = {SHOW}")
        print("=" * 118)
        p, t = window(cp, ct)
        base, _ = row("baseline", p, t, f_o)

        print("\n  T-1 amplify")
        for c in (0.5, 1, 2, 4, 8, 16):
            q, t1 = window(cp * c, ct)
            row(f"x{c}", q, t1, f_o)

        print("\n  T-2 shuffle within block")
        for k in (2, 8):
            q, t2 = window(shuffle_in_blocks(cp, k), ct)
            row(f"k={k}", q, t2, f_o)

        print("\n  T-3 smooth")
        for n_pass in (1, 2, 3):
            q, t3 = window(smooth3(cp, n_pass), ct)
            row(f"{n_pass} pass", q, t3, f_o)

        print("\n  T-4 displace, fixed truth window")
        for s in (0, 1, 2, 3):
            q, t4 = window(cp, ct, s)
            row(f"s={s}", q, t4, f_o)

        print("\n  T-5 inflate per-cell variance, MEAN FIELD FIXED")
        print("      ⚠️ FSS consumes a POINT field and has no variance channel at all.")
        row("tau^2 any x var(truth)", p, t, f_o)

        print("\n  threshold-rule sweep, f_M / f_o")
        for r in (0.5, 1.0, 2.0):
            afss = 2 * 1.0 * r / (1.0 + r**2)
            s_r, sm_r = row(f"r={r}  (AFSS {afss:.4f})", p, t, f_o, r=r)
            del s_r, sm_r

    # ---- P-F1: is count-matched FSS a function of RANK ORDER alone?
    print()
    print("=" * 118)
    print("P-F1 — is count-matched FSS a function of the forecast's RANK ORDER alone?")
    print("=" * 118)
    pc, tc = window(*flatten(ctl, "counts"))
    pl, tl = window(*flatten(ctl, "log1p"))
    bc, bl = binary_stack(pc, tc), binary_stack(pl, tl)
    fc, fl = fss_pooled(list(pc), list(tc), SCALES), fss_pooled(list(pl), list(tl), SCALES)
    dcell = int((bc != bl).sum())
    dfss = max(abs(fc[n] - fl[n]) for n in SCALES)
    print(f"  counts vs log1p:  differing binary cells {dcell:,} / {bc.size:,}   "
          f"max |dFSS| {dfss:.3e}")
    raw_p, raw_t = flatten(ctl, "counts")
    for c in (0.5, 2, 4, 16):
        q, tq = window(raw_p * c, raw_t)
        bq = binary_stack(q, tq)
        fq = fss_pooled(list(q), list(tq), SCALES)
        print(f"  T-1 c={c:<5}:      differing binary cells "
              f"{int((bq != bc).sum()):,} / {bc.size:,}   "
              f"max |dFSS| {max(abs(fq[n] - fc[n]) for n in SCALES):.3e}")
    print(f"\n  F-4 bar: refuted only if max |dFSS| > 1e-4. counts/log1p {dfss:.3e} => "
          f"{'HOLDS' if dfss <= 1e-4 else 'REFUTED'}")

    # ---- the arm, DESCRIPTIVE ONLY (prereg §7.5)
    print()
    print("=" * 118)
    print("THE ARM — DESCRIPTIVE ONLY, not a verdict (prereg §7.5; n=1, C-307)")
    print("=" * 118)
    for space in ("counts", "log1p"):
        # ⛔ RE-SEEDED PER SPACE, added 2026-10-27b. The module-level rng was never reset, so T-2
        # drew different permutations for counts and log1p and the committed .out shows the two
        # diverging (k=2: 0.2959 vs 0.3265) where the transform guarantees they cannot. The
        # divergence was the draw, not the space. See `W-78`.
        reseed()
        ap, at = window(*flatten(arm, space))
        cpx, ctx = window(*flatten(ctl, space))
        print(f"  {space}")
        row("control", cpx, ctx, f_o)
        row("arm", ap, at, f_o)


if __name__ == "__main__":
    main()
