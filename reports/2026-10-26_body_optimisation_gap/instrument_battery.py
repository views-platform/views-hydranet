"""Stage 1 — the instrument battery. Pre-registered in ../2026-10-26_instrument_battery_prereg.md.

Three candidate structure instruments against five controlled transforms of one forecast field.
Nothing is trained. The predictions are committed in §3 of the prereg, which was written and
committed (`c0cdd03`) BEFORE this file existed.

  I-1  `_structure_penalty`  one-sided log shortfall of within-block variance  (POSITIVE CONTROL)
  I-2  B-1                   [gamma_pred(d)/gamma_true(d)] / [var_pred/var_true]
  I-3  VS-p                  the variogram score, Scheuerer & Hamill 2015 Eq. (3), p = 2
  R-1  Spearman(model,truth) NOT one of the three. A placement-sensitive REFERENCE, so that T-4
                             has a positive control and "nothing moved" cannot be read as
                             "nothing happened".

⛔ I-1'S ROW IS NOT A PREDICTION. Three of its five answers were measured this morning
(`c3ec352`). It is here as the positive control and F-1 is the gate: if those three do not
reproduce exactly, this is not the same instrument and nothing else in the run is interpretable.

⛔ DECLARED DEVIATION (C-371). VS-p as published sums over ALL d^2 ordered pairs with
inverse-distance weights, which are positive for every pair -- 4,201,373,124 pairs at grid scale.
The weights here are TRUNCATED at radius 4. Propriety survives (the paper grants properness "for
any choice of nonnegative weights", and zero is nonnegative), but a truncated run is NOT the
method as published and is never to be cited as such.

⚠️ METHOD CLARIFICATION against prereg §2 T-5. The prereg sweeps `tau^2 / v_bar`, presupposing a
baseline per-cell predictive variance. The biopsy npz carries `y_hat_reg` (log1p of the family
mean) and `y_hat_cls` (the gate) but NOT `theta`, so there is no v_bar in this population. v is
therefore treated as a FREE PARAMETER swept from 0, expressed relative to the truth field's
variance. That is strictly more informative -- it maps the whole curve rather than one ratio --
and it makes P-2 ("an interior minimum at tau* > 0") directly testable. The prediction is
unchanged; only its parameterisation is.
"""

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARM = Path(
    "/home/simon/views-diagnostics/hf_arm_20261026/plots/02_training_dynamics/"
    "biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
)
CTL = Path(
    "/home/simon/views-diagnostics/hf_control_20261025/diagnostics/251026_1043/"
    "02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
)
BLOCKS = (2, 8)
LAGS = (1, 2, 4, 8)
VS_RADIUS = 4
EPS = 1e-6


# ------------------------------------------------------------------- I-1 and I-2, copied verbatim
def block_var(f, k):
    n, h, w = f.shape
    b = f.reshape(n, h // k, k, w // k, k)
    m1, m2 = b.mean(axis=(2, 4)), (b * b).mean(axis=(2, 4))
    return np.clip(m2 - m1 * m1, 0.0, None)


def penalty(pred, truth):
    """I-1. One-sided log shortfall of block variance, summed over k. Lower is better."""
    tot = 0.0
    for k in BLOCKS:
        short = np.log(block_var(truth, k) + EPS) - np.log(block_var(pred, k) + EPS)
        tot += float(np.clip(short, 0.0, None).mean())
    return tot


def gamma_ew(f, d):
    diff = f[:, :, d:] - f[:, :, :-d]
    return 0.5 * float((diff * diff).mean())


def b1(pred, truth, d):
    """I-2. 1.0 = the model carries its full share of structure at that scale."""
    return (gamma_ew(pred, d) / gamma_ew(truth, d)) / (pred.var() / truth.var())


# ----------------------------------------------------------------------------------------- I-3
def _offsets(r):
    """Half-plane offsets within radius r, with inverse-distance weights. Half-plane because the
    pair (i,j) and (j,i) carry the same |y_i - y_j|, so taking both would double every pair."""
    out = []
    for dr in range(0, r + 1):
        for dc in range(-r, r + 1):
            if (dr, dc) <= (0, 0) or dr * dr + dc * dc > r * r:
                continue
            out.append((dr, dc, 1.0 / np.hypot(dr, dc)))
    return out


VS_OFFSETS = _offsets(VS_RADIUS)


def vs2(mean_f, truth, v=0.0):
    """I-3. VS-p at p=2, truncated inverse-distance weights.

    S = sum_ij w_ij ( |y_i - y_j|^2  -  E|X_i - X_j|^2 )^2, and for conditionally independent
    heads E|X_i - X_j|^2 = (m_i - m_j)^2 + v_i + v_j. `v` is the per-cell predictive variance,
    here a scalar swept from 0 (see the module docstring). Normalised by total weight so values
    are comparable across radius choices. Lower is better.
    """
    num = den = 0.0
    for dr, dc, w in VS_OFFSETS:
        sl_a = (slice(None), slice(dr, None), slice(max(dc, 0), mean_f.shape[2] + min(dc, 0)))
        sl_b = (slice(None), slice(0, mean_f.shape[1] - dr),
                slice(max(-dc, 0), mean_f.shape[2] - max(dc, 0)))
        t = (truth[sl_a] - truth[sl_b]) ** 2
        m = (mean_f[sl_a] - mean_f[sl_b]) ** 2 + 2.0 * v
        num += w * float(((t - m) ** 2).mean())
        den += w
    return num / den


def spearman_ref(pred, truth):
    """R-1. The placement-sensitive reference. Not one of the three instruments."""
    return float(spearmanr(pred.ravel(), truth.ravel()).statistic)


# ------------------------------------------------------------------------------------ transforms
rng = np.random.default_rng(20261026)


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


DISP_WIN, DISP_PAD = 24, 4          # 24 is divisible by 8; PAD leaves room for s <= 4


def displace(pred, truth, s):
    """T-4. Translate the model field by s cells against a FIXED truth window. No wrap.

    ⛔ F-3: no wrap. A wrapped translation manufactures an edge discontinuity that every one of
    these instruments reads as structure.

    ⚠️ METHOD CLARIFICATION, forced by the production guard and kept because it is better.
    `_structure_penalty` RAISES unless every block size divides the grid (`training_engine.py:455`
    — "avg_pool2d would silently drop the remainder", the C-351 population failure inside a loss
    term). A 30x30 crop at s=1 is not divisible by 8, and the guard fired on the first run. The
    window is therefore FIXED at 24x24 for every s, with the truth window held still and only the
    model window sliding. That removes a confound the first design had: the population no longer
    changes with s, so a change in any instrument is the displacement and not the crop.
    """
    o = DISP_PAD
    tp = truth[:, o:o + DISP_WIN, o:o + DISP_WIN]
    mp = pred[:, o + s:o + s + DISP_WIN, o + s:o + s + DISP_WIN]
    return np.ascontiguousarray(mp), np.ascontiguousarray(tp)


# ---------------------------------------------------------------------------------------- fields
def load(p):
    z = np.load(p)
    return {k: z[k].astype(np.float64) for k in ("y_reg", "y_hat_reg", "y_hat_cls")}


def flatten(d, space):
    mv = lambda a: np.moveaxis(a, -1, 1).reshape(-1, 32, 32)  # noqa: E731
    gate, body, truth = mv(d["y_hat_cls"]), mv(d["y_hat_reg"]), mv(d["y_reg"])
    if space == "counts":
        return gate * np.expm1(body), np.expm1(truth)
    return np.log1p(gate * np.expm1(body)), truth


ctl, arm = load(CTL), load(ARM)
print("=" * 100)
print("STAGE 1 — THE INSTRUMENT BATTERY.  prereg ../2026-10-26_instrument_battery_prereg.md")
print("=" * 100)
if not np.array_equal(ctl["y_reg"], arm["y_reg"]):
    raise SystemExit("VOID (F-2): the two windows do not carry the same truth.")
print("  F-2 truth byte-identical across arms: True")
print(f"  I-3 weights: inverse distance, TRUNCATED at radius {VS_RADIUS} "
      f"({len(VS_OFFSETS)} half-plane offsets) — DECLARED DEVIATION, C-371")

for space in ("counts", "log1p"):
    cp, ct = flatten(ctl, space)
    tv = ct.var()
    print("\n" + "=" * 100)
    print(f"space = {space}")
    print("=" * 100)
    base = (penalty(cp, ct), b1(cp, ct, 1), vs2(cp, ct, 0.0), spearman_ref(cp, ct))
    print(f"  baseline   I-1 {base[0]:10.4f}   I-2(d1) {base[1]:8.4f}   "
          f"I-3 {base[2]:12.4f}   R-1 {base[3]:+7.4f}")

    def row(lab, f, t=ct, v=0.0):
        print(f"  {lab:26s} I-1 {penalty(f, t):10.4f}   I-2(d1) {b1(f, t, 1):8.4f}   "
              f"I-3 {vs2(f, t, v):12.4f}   R-1 {spearman_ref(f, t):+7.4f}")

    print("\n  T-1 amplify  (P-1 ctrl: I-1 monotone better · I-2 EXACT · I-3 interior minimum)")
    for c in (0.5, 1, 2, 4, 8, 16):
        row(f"    x{c}", cp * c)

    print("\n  T-2 shuffle within block  (I-1 exactly 0 at k=2 · I-2 moves · I-3 MAY IMPROVE)")
    for k in BLOCKS:
        row(f"    k={k}", shuffle_in_blocks(cp, k))

    print("\n  T-3 smooth  (all three must worsen)")
    for s in (1, 2, 3):
        row(f"    {s} pass", smooth3(cp, s))

    print("\n  T-4 displace, cropped to the valid overlap  (P-1: I-2 EXACTLY invariant)")
    for s in (0, 1, 2, 3):
        mp, tp = displace(cp, ct, s)
        print(f"    s={s} ({tp.shape[1]}x{tp.shape[2]}, fixed)   "
              f"I-1 {penalty(mp, tp):10.4f}   I-2(d1) {b1(mp, tp, 1):8.4f}   "
              f"I-3 {vs2(mp, tp, 0.0):12.4f}   R-1 {spearman_ref(mp, tp):+7.4f}")

    print("\n  T-5 inflate per-cell predictive variance, MEAN FIELD FIXED  (P-2 / C-370)")
    print("      ⚠️ I-1 and I-2 take a POINT field and have no variance channel at all.")
    vs_curve = []
    for ratio in (0.0, 0.0625, 0.25, 1.0, 4.0, 16.0):
        v = ratio * tv
        s = vs2(cp, ct, v)
        vs_curve.append((ratio, s))
        print(f"    tau^2 = {ratio:7.4f} x var(truth)   I-1 {penalty(cp, ct):10.4f}   "
              f"I-2(d1) {b1(cp, ct, 1):8.4f}   I-3 {s:12.4f}")
    best = min(vs_curve, key=lambda x: x[1])
    verdict = ("CONFIRMED — an interior minimum at tau* > 0" if best[0] > 0 else
               "REFUTED — VS-p worsens monotonically from tau = 0; C-370 is withdrawn")
    print(f"    => P-2 / C-370 {verdict}.  argmin at tau^2 = {best[0]} x var(truth), "
          f"VS-p {best[1]:.4f} vs {vs_curve[0][1]:.4f} at tau=0")

print("\n" + "=" * 100)
print("F-4 — is the closed form right? analytic VS-2 against a sampled estimate, K = 4096")
print("=" * 100)
cp, ct = flatten(ctl, "counts")
v = 0.25 * ct.var()
sd, K = np.sqrt(v), 4096
num = den = 0.0
for dr, dc, w in VS_OFFSETS[:6]:
    a = (slice(None), slice(dr, None), slice(max(dc, 0), cp.shape[2] + min(dc, 0)))
    b = (slice(None), slice(0, cp.shape[1] - dr), slice(max(-dc, 0), cp.shape[2] - max(dc, 0)))
    xa, xb = cp[a], cp[b]
    run = np.zeros_like(xa)                      # running mean, so K draws cost O(1) memory
    for k in range(K):
        d = (xa + rng.normal(0, sd, xa.shape)) - (xb + rng.normal(0, sd, xb.shape))
        run += (d * d - run) / (k + 1)
    ana = (xa - xb) ** 2 + 2.0 * v
    num += w * float(np.abs(run - ana).mean())
    den += w * float(np.abs(ana).mean())
rel = num / den
print(f"  mean |sampled - analytic| / mean |analytic|, 6 shortest offsets, K={K}: {rel:.4%}")
print(f"  => F-4 {'PASSES' if rel <= 0.01 else 'FIRES — the closed form is wrong and every'}"
      f"{'' if rel <= 0.01 else ' VS-p number with it is VOID'} (bar: 1%)")
