"""C-366: why do `_structure_penalty` and B-1 disagree by sign about the same model?

Both claim to measure "does this model carry its due spatial structure?". On the structure-penalty
arm the penalty improved 4.87x (1.7223 -> 0.3540) while B-1 at d=1 worsened 4.44x (0.400 -> 0.090).
Until that is resolved neither can be trusted as a target, and B-1 has been the body programme's
primary diagnostic since D12.

THE TWO DEFINITIONS, read off the code rather than recalled.

  penalty  `training_engine.py:402-476`. For k in (2, 8): the within-k-x-k-block variance of the
           composed forecast `sigmoid(gate).detach() * family.mean(params)` against that of the
           observed counts, as a ONE-SIDED difference of logs,
               sum_k mean( clamp( log(var_true + 1e-6) - log(var_pred + 1e-6), min=0 ) ).
           Lower is better. Excess dispersion costs exactly nothing.

  B-1      `body_scoreboard.py:93-111`. The classical variogram ratio at lag d, divided by the
           ratio of the two fields' total variances,
               B1(d) = [ gamma_pred(d) / gamma_true(d) ] / [ var_pred / var_true ].
           1.0 = the model carries its full share of structure at that scale.

THE HYPOTHESIS, stated before the measurement and falsifiable by it: these are not two estimates
of one quantity but measurements of DISJOINT properties, so a sign disagreement is not a
contradiction and C-366 dissolves rather than resolving. Specifically --

  H1  the penalty is HOMOGENEOUS in the field's amplitude: scaling the forecast by c scales every
      block variance by c^2, so the penalty moves with nothing rearranged.
  H2  B-1 is EXACTLY INVARIANT to that same scaling: gamma and var both scale by c^2 and the
      ratio cancels. B-1 cannot see amplitude at all.
  H3  the penalty is EXACTLY INVARIANT to permuting cells WITHIN each block: a variance depends
      only on a multiset. Permuting within 2x2 blocks also leaves every 8x8 block's multiset
      unchanged, so BOTH penalty terms are invariant while B-1 at d=1 collapses. The penalty
      cannot see arrangement at the scale it is evaluated on.
  H4  adding unstructured noise IMPROVES the penalty (var_pred rises, the shortfall shrinks) and
      RAISES B-1 toward 1/(1 - rho_true(d)) > 1, because gamma and var each gain the same tau^2
      and gamma_pred < var_pred at short lag. ⇒ The arm's B-1 FELL, so whatever the arm did it was
      NOT adding noise. That contradicts how §4c's result has been described in this programme
      ("flooded the floor"), and the contradiction is the point of including H4.

⛔ PRE-COMMITTED FALSIFIER F1. The two headline numbers were measured on different populations --
the penalty on 32x32 training windows from the trained checkpoints with 12 draws, B-1 on the
360x720 grid. This script computes BOTH on ONE shared field: the matched `lesson_297_win_1`
biopsy, which both arms wrote with BYTE-IDENTICAL truth. If the control -> arm change does NOT
disagree by sign there, then the original disagreement is a population artefact rather than a
property of the two definitions, and H1-H4 are beside the point. ⚠️ A space mismatch could also
produce that, so if F1 fires the conclusion is "population OR space", never "population".

⚠️ SPACE IS SWEPT, NOT CHOSEN. The penalty's native inputs are COUNTS; the forensic npz stores
log1p. Counts are reconstructed as `y_hat_cls * expm1(y_hat_reg)` against `expm1(y_reg)` -- which
is the penalty's own composition, since `y_hat_reg` is log1p(family.mean) with the gate ignored
(`_family_target_log1p_mean`) and `y_hat_cls` is the post-sigmoid gate. Both spaces are reported
because the choice is a free parameter and a single number across it would not be quotable.

⚠️ NO LAND MASK. The window is a curriculum window on the global grid and the npz carries no
mask, so all 1,024 cells enter. Any non-land cell is a structural zero in both fields.
"""

from pathlib import Path

import numpy as np

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
EPS = 1e-6


# ----------------------------------------------------------------------------- the two instruments
def block_var(f: np.ndarray, k: int) -> np.ndarray:
    """E[x^2] - E[x]^2 within each k x k block of [N, H, W] -> [N, H/k, W/k]."""
    n, h, w = f.shape
    b = f.reshape(n, h // k, k, w // k, k)
    m1 = b.mean(axis=(2, 4))
    m2 = (b * b).mean(axis=(2, 4))
    return np.clip(m2 - m1 * m1, 0.0, None)


def penalty(pred: np.ndarray, truth: np.ndarray) -> float:
    """`_structure_penalty` verbatim: one-sided log shortfall of block variance, summed over k."""
    tot = 0.0
    for k in BLOCKS:
        short = np.log(block_var(truth, k) + EPS) - np.log(block_var(pred, k) + EPS)
        tot += float(np.clip(short, 0.0, None).mean())
    return tot


def gamma_ew(f: np.ndarray, d: int) -> float:
    """Classical method-of-moments gamma-hat at E-W lag d: 0.5 * mean (Z(s+d) - Z(s))^2."""
    diff = f[:, :, d:] - f[:, :, :-d]
    return 0.5 * float((diff * diff).mean())


def b1(pred: np.ndarray, truth: np.ndarray, d: int) -> float:
    """[gamma_pred(d)/gamma_true(d)] / [var_pred/var_true]. 1.0 = full share of structure."""
    num = gamma_ew(pred, d) / gamma_ew(truth, d)
    den = pred.var() / truth.var()
    return num / den


# ------------------------------------------------------------------------------------- the fields
def load(p: Path) -> dict[str, np.ndarray]:
    z = np.load(p)
    return {k: z[k].astype(np.float64) for k in ("y_reg", "y_hat_reg", "y_hat_cls")}


ctl, arm = load(CTL), load(ARM)
same = np.array_equal(ctl["y_reg"], arm["y_reg"])
print("=" * 100)
print("C-366 -- TWO INSTRUMENTS, ONE SHARED FIELD. Matched biopsy lesson_297_win_1.")
print("=" * 100)
print(f"  truth byte-identical across the two arms: {same}")
if not same:
    raise SystemExit("VOID: the two windows do not carry the same truth. §4c's gate, verbatim.")


def flatten(d: dict[str, np.ndarray], space: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (composed forecast, truth) as [steps*targets, 32, 32] in `space`."""
    #: y_hat_reg is log1p(family.mean) with the gate IGNORED; y_hat_cls is the post-sigmoid gate.
    #: The penalty composes them as gate * mean, in counts. (6, 32, 32, 3) -> (18, 32, 32).
    mv = lambda a: np.moveaxis(a, -1, 1).reshape(-1, 32, 32)  # noqa: E731
    gate, body, truth = mv(d["y_hat_cls"]), mv(d["y_hat_reg"]), mv(d["y_reg"])
    if space == "counts":
        return gate * np.expm1(body), np.expm1(truth)
    return np.log1p(gate * np.expm1(body)), truth


print(f"\n  {18} fields per arm (6 steps x 3 targets), 32x32, {18 * 1024:,} cell-values")
print("\n" + "=" * 100)
print("STEP 1 -- F1: does the sign disagreement REPRODUCE on this one shared field?")
print("=" * 100)
for space in ("counts", "log1p"):
    cp, ct = flatten(ctl, space)
    ap, at = flatten(arm, space)
    pc, pa = penalty(cp, ct), penalty(ap, at)
    print(f"\n  space = {space}")
    print(f"    penalty   control {pc:10.4f}   arm {pa:10.4f}   "
          f"{'ARM BETTER' if pa < pc else 'CONTROL BETTER'}  ({pc / max(pa, 1e-12):.2f}x)")
    for d in LAGS:
        bc, ba = b1(cp, ct, d), b1(ap, at, d)
        print(f"    B-1 d={d:<2d}    control {bc:10.4f}   arm {ba:10.4f}   "
              f"{'ARM BETTER' if abs(ba - 1) < abs(bc - 1) else 'CONTROL BETTER'}")

# --------------------------------------------------------------------------------- the battery
rng = np.random.default_rng(20261026)


def shuffle_in_blocks(f: np.ndarray, k: int) -> np.ndarray:
    """Permute cells WITHIN each k x k block, independently per block and per field."""
    n, h, w = f.shape
    b = (f.reshape(n, h // k, k, w // k, k).transpose(0, 1, 3, 2, 4)
          .reshape(n, h // k, w // k, k * k))
    idx = rng.permuted(np.broadcast_to(np.arange(k * k), b.shape), axis=-1)
    b = np.take_along_axis(b, idx, axis=-1)
    return (b.reshape(n, h // k, w // k, k, k).transpose(0, 1, 3, 2, 4).reshape(n, h, w))


def smooth3(f: np.ndarray) -> np.ndarray:
    """3x3 box mean with edge replication — a pure arrangement change at fixed total mass-ish."""
    p = np.pad(f, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(p[:, i:i + f.shape[1], j:j + f.shape[2]] for i in range(3) for j in range(3)) / 9.0


print("\n" + "=" * 100)
print("STEP 2 -- THE BATTERY. Controlled transforms of the CONTROL's own forecast field.")
print("=" * 100)
print("  H1 scaling moves the penalty · H2 scaling leaves B-1 EXACT · H3 within-block shuffle")
print("  leaves the penalty EXACT · H4 noise improves the penalty and RAISES B-1.\n")

for space in ("counts", "log1p"):
    cp, ct = flatten(ctl, space)
    sd = cp.std()
    cases = [
        ("identity", cp),
        ("scale x0.5", cp * 0.5),
        ("scale x2", cp * 2.0),
        ("scale x4", cp * 4.0),
        ("shuffle within 2x2", shuffle_in_blocks(cp, 2)),
        ("shuffle within 8x8", shuffle_in_blocks(cp, 8)),
        ("+ white noise 0.5sd", cp + rng.normal(0, 0.5 * sd, cp.shape)),
        ("+ white noise 1sd", cp + rng.normal(0, 1.0 * sd, cp.shape)),
        ("+ white noise 2sd", cp + rng.normal(0, 2.0 * sd, cp.shape)),
        ("3x3 box smooth", smooth3(cp)),
    ]
    base_p = penalty(cp, ct)
    base_b = {d: b1(cp, ct, d) for d in LAGS}
    print(f"  space = {space}")
    print(f"    {'transform':22s} {'penalty':>10s} {'d penalty':>10s} " +
          " ".join(f"{'B-1 d' + str(d):>10s}" for d in LAGS))
    for name, f in cases:
        p = penalty(f, ct)
        bs = [b1(f, ct, d) for d in LAGS]
        print(f"    {name:22s} {p:10.4f} {p - base_p:+10.4f} " +
              " ".join(f"{v:10.4f}" for v in bs))
    print(f"    {'(identity B-1)':22s} {'':10s} {'':10s} " +
          " ".join(f"{base_b[d]:10.4f}" for d in LAGS))
    print()

print("=" * 100)
print("STEP 3 -- EXACTNESS. H2 and H3 claim INVARIANCE, which is a stronger claim than a trend.")
print("=" * 100)
for space in ("counts", "log1p"):
    cp, ct = flatten(ctl, space)
    print(f"\n  space = {space}")
    for c in (0.5, 2.0, 4.0, 100.0):
        dev = max(abs(b1(cp * c, ct, d) - b1(cp, ct, d)) for d in LAGS)
        print(f"    H2  B-1 under scaling x{c:<6g}  max |change| over d in {LAGS} = {dev:.3e}")
    for k in BLOCKS:
        s = shuffle_in_blocks(cp, k)
        for kk in BLOCKS:
            a = block_var(cp, kk)
            b = block_var(s, kk)
            print(f"    H3  shuffle within {k}x{k}: block variance at k={kk} "
                  f"max |change| = {np.abs(a - b).max():.3e}")
        print(f"    H3  shuffle within {k}x{k}: PENALTY change = "
              f"{penalty(s, ct) - penalty(cp, ct):+.3e}   "
              + "  ".join(f"B-1 d{d} {b1(s, ct, d):.4f}" for d in LAGS))

# ----------------------------------------------------------------------------------------- step 4
print("\n" + "=" * 100)
print("STEP 4 -- CAN THE ARM'S SIGNATURE BE REPRODUCED BY HAND, from the control?")
print("=" * 100)
print("  H1-H4 say the penalty sees amplitude and not arrangement, B-1 the reverse. The move that")
print("  improves the penalty while worsening B-1 is therefore AMPLIFY (free for B-1, good for")
print("  the penalty) plus SMOOTH (bad for both, but amplification more than pays for it).")
print("  ⛔ Neither alone will do it: scaling leaves B-1 EXACT, and smoothing alone makes the")
print("  penalty WORSE. If no (scale, smooth) cell lands near the arm, this mechanism is refuted.")
for space in ("counts", "log1p"):
    cp, ct = flatten(ctl, space)
    ap, at = flatten(arm, space)
    tgt_p, tgt_b = penalty(ap, at), [b1(ap, at, d) for d in LAGS]
    print(f"\n  space = {space}   ARM OBSERVED: penalty {tgt_p:.4f}   "
          + "  ".join(f"B-1 d{d} {v:.4f}" for d, v in zip(LAGS, tgt_b)))
    print(f"    {'scale':>7s} {'smooths':>8s} {'penalty':>9s} "
          + " ".join(f"{'B-1 d' + str(d):>9s}" for d in LAGS) + f" {'|err|':>8s}")
    best = None
    for c in (1, 2, 4, 8, 16, 32):
        f0 = cp * c
        for s in range(4):
            f = f0
            for _ in range(s):
                f = smooth3(f)
            p = penalty(f, ct)
            bs = [b1(f, ct, d) for d in LAGS]
            err = abs(p - tgt_p) / tgt_p + sum(abs(a - b) for a, b in zip(bs, tgt_b)) / len(LAGS)
            if best is None or err < best[0]:
                best = (err, c, s)
            print(f"    {c:7d} {s:8d} {p:9.4f} " + " ".join(f"{v:9.4f}" for v in bs)
                  + f" {err:8.3f}")
    print(f"    => closest: scale x{best[1]}, {best[2]} smoothing pass(es), "
          f"|err| {best[0]:.3f}")
