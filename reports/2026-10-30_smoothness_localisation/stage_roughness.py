"""Q1/B2b — WHERE inside the network does the body's spatial smoothness come from?

THE QUESTION, AND WHAT IS ALREADY ANSWERED
------------------------------------------
Q1 ("is the body spatially too smooth?") is **answered: yes**. `d12b_curve_readout.out` §4:
the model's `log mu` field has lag-1 E-W correlation **0.926** out of sample against a truth
field at **0.638** (336-month mean) / **0.499** (36-month window); model-free and
threshold-swept the field needs **3-7x** the distance to decorrelate, and the shape-corrected
share of due structure runs **0.205** at d=1 to **0.776** at d=32 => a **near-field defect that
closes by ~32 cells**.

⛔ **What is NOT answered is WHERE.** This script localises it to a module. It does not
adjudicate: a smooth internal representation may be correct. The output is *which stage*.

⭐⭐ PRIOR ART THAT CONSTRAINS THE ANSWER BEFORE THIS RUNS
---------------------------------------------------------
Found by searching `reports/` rather than rebuilt (the "superseded retires a conclusion, never
an instrument" rule, which has already cost this project a working reliability diagram):

  * `2026-08-23_state_range_dossier/07_experiment_log.md:63` + `tools/capture_regimes.py:253-257`
    — **the theoretical receptive field is NOT the effective one.** Patch-vs-full-grid
    `rel_abs_diff` is **0.0006** in the interior against **≈0.10** pooled: a **142x** gap on
    seed 42, 13x on seed 43. The recurrence **contracts** (65.6 -> 1.8, 35.8x over 30 free
    steps), so influence decays faster than it propagates.
  * RF read off the architecture = **26 cells** (3x3 convs, two 2x2 pools), margin 13.

⇒ ⭐ Context beyond ~32 cells barely reaches the output, and d12b says the defect is worst at
**d=1** and closed by **d=32**. **Those are consistent and together they narrow the question:
the smoothing happens INSIDE the local neighbourhood, not by integrating wide context.** That
excludes "the model averages over hundreds of cells" and points at the conv/pool/upsample
stack — which is what the hooks below separate.

⚠️ It does NOT exclude the encoder: a 26-cell RF is still 26x the 1-cell lag where the defect
is worst. The encoder and the decoder are both live, which is the point of measuring per stage.

THE DECISIVE STRUCTURAL READ, stated before the run
---------------------------------------------------
The H1-reg decoder's last stage is, verbatim (`HydraBNrecurrentUnet_06_LSTM4.py:634-641`):

    torch.cat([self.upsample1_head1_reg(H1_d0), e0s_topskip], 1)

⭐⭐ **The `torch.cat` happens OUTSIDE both modules**, so a forward hook can read the
**upsampled branch** and the **full-resolution skip branch** SEPARATELY. That turns a vague
"is it the decoder" into a sharp question:

  **Is the skip's detail swamped by the upsampled path?**

  * `upsample1_head1_reg` is a `ConvTranspose2d` stride 2 — it *interpolates*, so its output
    should be markedly smoother than the skip it is concatenated with.
  * If `dec_conv1_head1_reg`'s output resembles the **upsampled** branch rather than the
    **skip**, the decoder is discarding detail the network still had at full resolution, and
    the lever is the decoder.
  * If the skip is already as smooth as the upsampled branch, the detail was lost in the
    ENCODER and no decoder change recovers it.

WHAT IS MEASURED  (C-351 slots)
-------------------------------
  population      one teacher-forced sweep of months 421-469 over the full 360x720 raster;
                  rho is accumulated over the 13 OUT-OF-SAMPLE months 457-469 only.
  truth           not scored here. This measures ACTIVATIONS, not forecast error.
  estimator       Pearson correlation of lag-`d` neighbour PAIRS, E-W and N-S, per channel;
                  reported as the channel median with min/max beside it, never a bare mean.
  inference mode  ⛔ **TEACHER-FORCED** (`vehicle.py:301-307`). `C-399`. Same cube route as
                  `d4_mu_on_the_grid.py`, whose loop this copies (`W-98`).
  free parameters **dropout OFF and ON, both run** (the primary is OFF: MC-dropout noise is
                  itself spatially correlated — `d12c_draw_noise.out` puts 11.2% of the body
                  field's spatial variance in our own 4-draw noise — so it would DEPRESS rho
                  and confound a smoothness measurement); and the lag `d` over {1,2,4,8,16,32}
                  to match d12b's scale profile (D-4).

⛔ LAND MASKING, because this is the measurement's main hazard
-------------------------------------------------------------
Non-land cells are **zero** in the production raster (`config_queryset.py:27`). A large
constant ocean would drive any autocorrelation toward 1 and the result would be a map of the
coastline. ⇒ **Every rho here is computed only over neighbour pairs where BOTH cells are
land**, and at reduced resolution the land mask is downsampled by requiring **all** cells of
the block to be land (conservative). The scored-pair fraction is printed for every stage.

PRE-REGISTERED PREDICTIONS — declared here, committed before this runs (D-5)
----------------------------------------------------------------------------
  P-12  rho(1) of the FED INPUT channels is low, near d12b's truth value (~0.5), and rises
        monotonically through encoder -> bottleneck. A non-monotone profile would mean a stage
        SHARPENS, which nothing in the architecture should do.
  P-13  ⭐⭐ THE ONE THAT MATTERS. rho(1) of `upsample1_head1_reg`'s output EXCEEDS rho(1) of
        its concat partner `e0s` by at least **0.15**, because ConvTranspose2d stride 2
        interpolates and the skip does not.
  P-14  rho(1) at the bottleneck exceeds **0.90** at its own 90x180 resolution ⇒ the encoder
        contributes, and the decoder is not the sole cause.
  P-15  The post-concat `dec_conv1_head1_reg` output sits CLOSER to the upsampled branch than
        to the skip branch in rho(1) ⇒ the skip's detail is swamped.
  ⛔ NOT predicted: that the final stage reproduces d12b's **0.926**. d12b measured a
     336-month TEMPORAL MEAN of `log mu` on land cells; this measures per-month activations.
     **Not like-for-like (`W-51`)** and it is not gated as a copy.

GATES, declared before the run
------------------------------
  G-12 ⛔ **C-384.** MC-dropout is enabled with `set_locked_dropout(True)`, **never** a bare
       `model.train()`. Assert **no** `_BatchNorm` module is in train mode — a bare `train()`
       put 15 of them there and overwrote the checkpoint's delivered running statistics, which
       inverted the sign of a headline. Asserted for BOTH dropout conditions.
  G-13 Two forwards on identical input are **bit-identical** (the mask is locked and not
       reset between them; with dropout off, nothing is sampled at all).
  G-14 ⭐ **Every registered hook FIRED and produced the expected resolution.** A hook that
       silently never fires is the decorative-guard failure this repo has recorded 9 times in
       18 repositories. Asserted on the exact shape, per stage.
  G-15 ⭐⭐ POSITIVE AND NEGATIVE CONTROL on the rho estimator, on fields whose answer is known
       by construction: a linear ramp -> rho(1) > +0.99; a checkerboard -> rho(1) < -0.99;
       iid noise -> |rho(1)| < 0.02.
  G-16 The land-pair mask actually excludes ocean: scored pairs must be strictly between 0 and
       the total adjacent-pair count, printed as a fraction.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

T0 = time.time()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(D4))

from vehicle import build_volume, grid_rowcol, load_vehicle, pull  # noqa: E402

from views_hydranet.architectures.locked_dropout import LockedDropout  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402

torch.set_num_threads(8)

ART = (Path.home() / "Documents" / "views_hydranet_vehicle_20261012"
       / "calibration_model_20261012_033112.pt")
LEADIN_START, LAST_FED = 421, 469
OS_LO, OS_HI = 457, 469
LAGS = (1, 2, 4, 8, 16, 32)

#: The H1-reg path, in forward order, with the resolution DIVISOR each stage outputs.
#: `e0s` and `up1` are the two branches of the final torch.cat — the decisive pair.
STAGES: tuple[tuple[str, str, int], ...] = (
    ("enc0",   "enc_conv0",            1),
    ("e0s",    'dropout["e0s"]',       1),
    ("pool0",  "pool0",                2),
    ("e1s",    'dropout["e1s"]',       2),
    ("pool1",  "pool1",                4),
    ("bneck",  "bottleneck_conv",      4),
    ("up0",    "upsample0_head1_reg",  2),
    ("dec0",   "dec_conv0_head1_reg",  2),
    ("up1",    "upsample1_head1_reg",  1),
    ("dec1",   "dec_conv1_head1_reg",  1),
    ("out",    "dec_conv4_head1_reg",  1),
)


def resolve(model: torch.nn.Module, spec: str) -> torch.nn.Module:
    if spec.startswith("dropout["):
        return model.dropout[spec.split('"')[1]]
    return getattr(model, spec)


def pair_rho(f: np.ndarray, mask: np.ndarray, d: int) -> tuple[float, float, float, int]:
    """Channel-wise Pearson rho of lag-`d` land-land neighbour pairs, E-W and N-S pooled.

    `f` is [C, H, W]; `mask` is [H, W] bool. Returns (median, min, max, pairs). Only pairs
    whose BOTH cells are land are scored — a constant ocean would otherwise drive rho to 1.
    """
    ew = mask[:, :-d] & mask[:, d:]
    ns = mask[:-d, :] & mask[d:, :]
    npair = int(ew.sum() + ns.sum())
    if npair == 0:
        return float("nan"), float("nan"), float("nan"), 0
    out = []
    for c in range(f.shape[0]):
        a = np.concatenate((f[c, :, :-d][ew], f[c, :-d, :][ns]))
        b = np.concatenate((f[c, :, d:][ew], f[c, d:, :][ns]))
        a = a - a.mean()
        b = b - b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if den > 0.0:
            out.append(float((a * b).sum() / den))
    if not out:
        return float("nan"), float("nan"), float("nan"), npair
    o = np.array(out)
    return float(np.median(o)), float(o.min()), float(o.max()), npair


# ------------------------------------------------------- G-15, before any model is touched
rng = np.random.default_rng(15)
H, W = 64, 64
m_all = np.ones((H, W), dtype=bool)
ramp = np.tile(np.arange(W, dtype=np.float64), (H, 1))[None]
board = (((np.arange(H)[:, None] + np.arange(W)[None]) % 2) * 2.0 - 1.0)[None]
noise = rng.standard_normal((1, H, W))
r_ramp = pair_rho(ramp, m_all, 1)[0]
r_board = pair_rho(board, m_all, 1)[0]
r_noise = pair_rho(noise, m_all, 1)[0]
ok15 = r_ramp > 0.99 and r_board < -0.99 and abs(r_noise) < 0.02
print(f"{'PASS' if ok15 else '⛔ FAIL'}  G-15  rho controls: ramp {r_ramp:+.4f} (>+0.99)  "
      f"checkerboard {r_board:+.4f} (<-0.99)  iid noise {r_noise:+.4f} (|.|<0.02)")
if not ok15:
    print("\n⛔⛔ G-15 FAILED — the rho estimator does not recover known answers. Nothing below")
    print("    this line would be readable.")
    sys.exit(1)

# ----------------------------------------------------------------------------- the raster
df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
land_cells = np.sort(df.priogrid_id.unique())
lr, lc = grid_rowcol(land_cells)
months = list(range(LEADIN_START, LAST_FED + 1))
vol, land, _ = build_volume(df, months)
GR, GC = land.shape

#: Land masks per resolution divisor. ⛔ Conservative: a reduced-resolution cell counts as land
#: only if EVERY full-resolution cell in its block is land, so no block straddles a coastline.
MASKS = {1: land}
for div in (2, 4):
    b = land[: (GR // div) * div, : (GC // div) * div]
    MASKS[div] = b.reshape(GR // div, div, GC // div, div).all(axis=(1, 3))
for div, mk in sorted(MASKS.items()):
    print(f"  land mask /{div}: {mk.shape[0]}x{mk.shape[1]}  land {int(mk.sum()):,} "
          f"({100.0 * mk.mean():.1f}%)")

print("=" * 110)
print("Q1/B2b — WHERE THE SPATIAL SMOOTHNESS COMES FROM, PER STAGE")
print("=" * 110)
print(f"  population      months {LEADIN_START}-{LAST_FED} fed, rho accumulated over the "
      f"{OS_HI - OS_LO + 1} out-of-sample months {OS_LO}-{OS_HI}")
print(f"  raster          {GR}x{GC}, {len(land_cells):,} land cells")
print("  inference mode  ⛔ TEACHER-FORCED (vehicle.py:301-307)")
print(f"  lags            {LAGS}   (D-4)")

results: dict[tuple[str, str], dict[int, tuple[float, float, float, int]]] = {}
for mode in ("dropout_off", "dropout_on"):
    ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
    model, cfg, sha = load_vehicle(ART)
    if mode == "dropout_off":
        n_do = 0
        for m in model.modules():
            if isinstance(m, LockedDropout):
                m.locked = False
                m.eval()
                n_do += 1
        if n_do == 0:
            raise SystemExit("VOID: no LockedDropout modules found to disable.")
    else:
        model.set_locked_dropout(True)          # ⛔ G-12 / C-384: never a bare model.train()
    bn = [type(m).__name__ for m in model.modules()
          if "BatchNorm" in type(m).__name__ and m.training]
    if bn:
        raise SystemExit(f"⛔ VOID (G-12): {len(bn)} BatchNorm modules in train() mode — the "
                         "checkpoint's delivered running statistics would be overwritten (C-384).")
    print(f"\n  {mode}: vehicle sha {sha[:8]}… family {cfg['output_distribution']}  "
          f"PASS G-12 (no BatchNorm in train mode, {sum(1 for _ in model.modules())} modules)")

    caught: dict[str, torch.Tensor] = {}

    def _mk(key: str):
        def _h(_mod, _inp, out):
            caught[key] = out.detach()
        return _h

    handles = [resolve(model, spec).register_forward_hook(_mk(key))
               for key, spec, _ in STAGES]

    acc: dict[str, dict[int, list[tuple[float, float, float, int]]]] = {
        k: {d: [] for d in LAGS} for k, _, _ in STAGES}
    with torch.no_grad():
        h0 = torch.zeros(1, cfg["total_hidden_channels"], GR, GC)
        a = model(vol[:, 0], h0).reg
        b2 = model(vol[:, 0], h0).reg
        if not torch.equal(a, b2):
            raise SystemExit(f"⛔ VOID (G-13): two forwards differ by max "
                             f"{float((a - b2).abs().max()):.3e} — the mask is not held.")
        print(f"  PASS G-13  two forwards on identical input are bit-identical ({mode})")

        model.reset_locked_dropout()
        h = torch.zeros(1, cfg["total_hidden_channels"], GR, GC)
        for t, fed in enumerate(months):
            caught.clear()
            out = model(vol[:, t], h)
            h = out.h_next
            if fed < OS_LO:
                continue
            if fed == OS_LO:                     # ⛔ G-14, once, on the first scored month
                missing = [k for k, _, _ in STAGES if k not in caught]
                if missing:
                    raise SystemExit(f"⛔ VOID (G-14): hooks never fired for {missing} — a hook "
                                     "that silently does not fire is a decorative guard.")
                for key, spec, div in STAGES:
                    got = tuple(caught[key].shape[-2:])
                    want = (GR // div, GC // div)
                    if got != want:
                        raise SystemExit(f"⛔ VOID (G-14): {key} ({spec}) emitted {got}, "
                                         f"expected {want} at divisor /{div}.")
                print(f"  PASS G-14  all {len(STAGES)} hooks fired at their declared resolution")
            for key, _, div in STAGES:
                f = caught[key][0].numpy().astype(np.float64)
                mk = MASKS[div]
                f = f[:, : mk.shape[0], : mk.shape[1]]
                for d in LAGS:
                    acc[key][d].append(pair_rho(f, mk, d))
        for hd in handles:
            hd.remove()

    for key, _, _ in STAGES:
        results[(mode, key)] = {
            d: (float(np.mean([r[0] for r in acc[key][d]])),
                float(np.mean([r[1] for r in acc[key][d]])),
                float(np.mean([r[2] for r in acc[key][d]])),
                acc[key][d][0][3])
            for d in LAGS}
    print(f"  {mode} swept in {time.time() - T0:.0f}s")

# ------------------------------------------------------------------------------- G-16, report
tot = {1: 0, 2: 0, 4: 0}
for div, mk in MASKS.items():
    tot[div] = int((mk.shape[0] * (mk.shape[1] - 1)) + ((mk.shape[0] - 1) * mk.shape[1]))
bad16 = []
for key, _, div in STAGES:
    p = results[("dropout_off", key)][1][3]
    if not (0 < p < tot[div]):
        bad16.append((key, p, tot[div]))
print(f"\n  {'PASS' if not bad16 else '⛔ FAIL'}  G-16  land-pair masking excludes ocean "
      f"at every stage" + ("" if not bad16 else f" — offenders {bad16}"))

for mode in ("dropout_off", "dropout_on"):
    print("\n" + "=" * 110)
    print(f"{'PRIMARY' if mode == 'dropout_off' else 'SECONDARY'} — {mode}: "
          "rho(d) PER STAGE, channel median (min..max across channels)")
    print("=" * 110)
    print(f"  {'stage':<8s}{'res':>10s}{'pairs':>12s}" + "".join(f"{f'rho({d})':>10s}"
                                                                 for d in LAGS))
    for key, spec, div in STAGES:
        r = results[(mode, key)]
        print(f"  {key:<8s}{f'/{div}':>10s}{r[1][3]:>12,d}"
              + "".join(f"{r[d][0]:>+10.4f}" for d in LAGS) + f"   {spec}")
    print(f"  {'':<8s}{'':>10s}{'ch min..max at d=1':>12s}" + "".join(
        f"{'':>10s}" for _ in LAGS))
    for key, _, _ in STAGES:
        r = results[(mode, key)][1]
        print(f"  {key:<8s}{'':>10s}{f'{r[1]:+.4f} .. {r[2]:+.4f}':>22s}")

print("\n" + "=" * 110)
print("⭐⭐ THE DECISIVE PAIR — the two branches of the final torch.cat, at FULL resolution")
print("=" * 110)
for mode in ("dropout_off", "dropout_on"):
    sk = results[(mode, "e0s")][1][0]
    up = results[(mode, "up1")][1][0]
    dc = results[(mode, "dec1")][1][0]
    ou = results[(mode, "out")][1][0]
    print(f"\n  {mode}")
    print(f"    skip branch       e0s   rho(1) = {sk:+.6f}")
    print(f"    upsampled branch  up1   rho(1) = {up:+.6f}   "
          f"gap vs skip {up - sk:+.6f}  (P-13 bar: > +0.15)")
    print(f"    post-concat conv  dec1  rho(1) = {dc:+.6f}   "
          f"|dec1-up1| {abs(dc - up):.6f}  vs  |dec1-e0s| {abs(dc - sk):.6f}")
    print(f"    final pre-activ   out   rho(1) = {ou:+.6f}")
    verdict = ("UPSAMPLED branch — the skip is swamped" if abs(dc - up) < abs(dc - sk)
               else "SKIP branch — the skip survives")
    print(f"    ⇒ P-15: dec1 is closer to the {verdict}")

print("\n" + "=" * 110)
print("⚠️ LIMITS")
print("=" * 110)
print("  · ⛔ TEACHER-FORCED (C-399). Says nothing about the delivered autoregressive regime.")
print("  · This LOCALISES, it does not adjudicate. A smooth internal representation may be")
print("    correct; the output is WHICH STAGE, which is what was unknown.")
print("  · rho is computed on ACTIVATIONS in their own units, pre-activation-function where the")
print("    stage is a conv. ⛔ Stages are NOT comparable to d12b's 0.926, which is a 336-month")
print("    TEMPORAL MEAN of log mu on land cells (W-51). No copy gate is claimed against it.")
print("  · Reduced-resolution land masks require the WHOLE block to be land, so /2 and /4 drop")
print("    coastal cells and their rho is an interior statistic.")
print("  · n=1 checkpoint, the H1-reg (sb) head only, one teacher-forced sweep per condition.")
print(f"\n  wall time {time.time() - T0:.1f} s   peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
