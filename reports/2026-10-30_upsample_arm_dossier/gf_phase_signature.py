"""G-F — does either remedy remove the phase signature `M92` measured?

PRE-REGISTRATION: `reports/2026-10-30_upsample_arm_dossier/PREREG.md` §4. Committed before this runs (D-5).

WHAT IS BEING TESTED
--------------------
`M92` measured that the decoder's `ConvTranspose2d(kernel=2, stride=2)` stages **cannot mix
spatially** — kernel == stride, so no window overlaps and output at `(2i+a, 2j+b)` is
`W[:,:,a,b] @ x[i,j]`, giving each of the four in-block phases **its own free weight matrix.** The
signature at `up1` on the TRAINED checkpoint was lag-1 rho **+0.123146** against lag-2
**+0.920900** — a **gap of +0.797754** — with channel-wise lag-1 rho down to **-0.724900**.

This script asks whether the two surviving remedies (`UpsampleConv2` at +-0 parameters,
`UpsampleConv1` at -20.4%) remove that gap.

⛔ THREE MODES, UNTRAINED, ONE SEED — and untrained-vs-untrained is the only clean comparison.
The replacement convs have **no trained weights**, so comparing them against a *trained* transposed
conv would measure training rather than geometry. The control is therefore rebuilt untrained too,
from the same seed, on the same input.

⚠️ `G-F` IS A REAL TEST, NOT A TAUTOLOGY. `Upsample(nearest)` makes the conv's input blocky in 2x2
constant blocks, and a window centred at an even position covers a different arrangement of those
blocks than one at an odd position ⇒ **a phase pattern CAN survive the remedy.**
For `Conv2d(2x2)` specifically: at the `(even, even)` phase the window sees four copies of one
input cell and applies the **summed** kernel, while the other three phases straddle block
boundaries.

⛔ AMENDMENT 2 (2026-10-30, after this script had already run; the OUTPUT IS UNCHANGED and no
number here is affected). The paragraph above previously credited the caution to
`Odena2016_Deconvolution` — "its fix *reduces* checkerboarding; it does not abolish it." Read
against the source, **Odena's wording is stronger**: "they do fix the checkerboard artifacts". The
"a phase pattern CAN survive" argument is therefore **this script's own derivation**, not the
source's, and the measurement below settled it in Odena's favour (gap -> -0.1984). Recorded rather
than quietly edited, because the argument was load-bearing for calling G-F a real test.

WHAT IS MEASURED  (C-351 slots)
-------------------------------
  population      the same 360x720 raster and the same 13 out-of-sample months 457-469 as
                  `stage_roughness.py`; months 421-469 are fed so the recurrent state is warmed
                  identically for all three modes.
  truth           not scored. This measures ACTIVATIONS.
  estimator       Pearson rho of lag-`d` **land-land** neighbour pairs, E-W and N-S pooled, per
                  channel; reported as the channel median with the channel MINIMUM beside it,
                  because the minimum is where anti-correlation shows.
  inference mode  ⛔ TEACHER-FORCED (`vehicle.py:301-307`), matching `M92` so the two are
                  comparable. ⚠️ The ARM itself is scored autoregressively; that is deliberate and
                  is the whole point of the prereg (§3).
  weights         ⛔ **UNTRAINED**, `torch.manual_seed(SEED)` before each build. No checkpoint is
                  loaded and none exists for the candidates.
  free parameter  the remedy: control / `UpsampleConv2` / `UpsampleConv1` (D-4 — the design choice
                  is SWEPT, not argued).

⚠️ `pair_rho` is copied from `stage_roughness.py` rather than imported: importing would execute that
script, and an instrument whose numbers are committed must not change when a shared helper does.

PREDICTIONS — declared in PREREG.md §5b before this ran
-------------------------------------------------------
  P-20  at least one remedy removes the signature: gap `rho(2) - rho(1)` at `up1` **< 0.15** AND
        channel floor at rho(1) **>= -0.05**.
  P-21  `UpsampleConv1` shows a **smaller** residual gap than `UpsampleConv2`, because a 1x1 kernel
        applies the same matrix at every phase and cannot form a phase pattern at all, while a 2x2
        kernel can.

⛔ THE ONE VOID CONDITION, named in PREREG.md §6 device 4 before any number existed
-----------------------------------------------------------------------------------
If the **control** (untrained, same seed) does **not** show the signature — gap **< 0.40** — then
the instrument is not detecting on untrained weights what it detected on trained ones, and this run
is VOID. ⛔ **Nothing else voids it.** Any other objection to this measurement is disallowed by
PREREG.md §6 device 3, which requires a `W-` row against myself first.

⛔ AND `G-F` IS NOT A GATE ON WHETHER THE ARM RUNS. The arm's premise is `M92`, already measured.
Per Simon's **S-2**, a `G-F` failure lowers confidence and closes nothing.

GATES
-----
  G-17 POSITIVE AND NEGATIVE CONTROLS on the rho estimator, before any model is built: a ramp ->
       rho(1) > +0.99; a checkerboard -> rho(1) < -0.99; iid noise -> |rho(1)| < 0.02.
  G-23 parameter counts PRINTED and asserted against PREREG.md §2.
  G-24 ⛔ C-384 — dropout is disabled by putting `LockedDropout` in eval, never a bare
       `model.train()`; assert **no `_BatchNorm` is in train mode**, for every mode.
  G-25 two forwards on identical input are **bit-identical**, for every mode.
  G-26 ⛔⛔ **C-347. AMENDMENT 1, after the first run imported the WRONG CODE.**
       `views_hydranet` must resolve to the **working tree**, asserted on the resolved
       `__file__` and printed. For a script, `sys.path[0]` is the SCRIPT'S OWN DIRECTORY, not the
       cwd — so `python reports/.../gf_phase_signature.py` silently imported **installed 0.1.5**
       from site-packages and died on `Unknown model type: 'UpsampleConv2'`. ⚠️ **It died only
       because the name did not exist there.** Had it existed, this script would have measured a
       different build and reported clean numbers. ⭐ `pytest` passes regardless, because it puts
       the repo root on `sys.path` — so **the test suite and the measurement script were reading
       different code**, which is exactly `C-347`'s shape with the arrow pointed at reports/.
       ⭐ **VERIFIED, not assumed, before amending:** installed 0.1.5 is **byte-identical** to the
       working tree for `HydraBNrecurrentUnet_06_LSTM4.py`, `locked_dropout.py`, `nb_core.py`,
       `composition.py` and `utils/utils.py` (only `registry.py` differs, and only by the two
       lines added today) ⇒ **`M90`, `M91`, `M92` and `M93` are UNAFFECTED.**
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
# ⛔ G-26 / C-347: REPO FIRST. For a script `sys.path[0]` is the script's own directory, so
# `views_hydranet` would otherwise resolve to the INSTALLED wheel (0.1.5) rather than the working
# tree. Asserted below as well — a comment is not a guard.
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(D4))

from vehicle import build_volume, pull  # noqa: E402

import views_hydranet  # noqa: E402
from views_hydranet.architectures.locked_dropout import LockedDropout  # noqa: E402
from views_hydranet.architectures.registry import get_architecture  # noqa: E402

_vh = Path(views_hydranet.__file__).resolve()
if REPO not in _vh.parents:
    print(f"⛔⛔ VOID (G-26): views_hydranet resolved to {_vh}, which is NOT under {REPO}. "
          "This script would measure a different build than the repository under test (C-347).")
    sys.exit(1)
print(f"PASS  G-26  views_hydranet resolves to the WORKING TREE: {_vh}")

torch.set_num_threads(8)

LEADIN_START, LAST_FED = 421, 469
OS_LO, OS_HI = 457, 469
SEED = 42
LAGS = (1, 2)
#: production values, violet_visitor/configs/config_hyperparameters.py
IN_CH, HID, OUT_CH, DROPOUT = 3, 32, 1, 0.15
MODES = ("HydraBNUNet06_LSTM4", "UpsampleConv2", "UpsampleConv1")
EXPECTED_PARAMS = {"HydraBNUNet06_LSTM4": 905_289, "UpsampleConv2": 905_289,
                   "UpsampleConv1": 720_969}
#: the stage names hooked, with the resolution divisor each outputs
STAGES = (("up0", "upsample0_head1_reg", 2), ("up1", "upsample1_head1_reg", 1),
          ("out", "dec_conv4_head1_reg", 1))
VOID_GAP = 0.40          # PREREG.md §6 device 4
P20_GAP, P20_FLOOR = 0.15, -0.05
M92_TRAINED = {"up1_rho1": 0.123146, "up1_rho2": 0.920900, "up1_floor": -0.724900}


def pair_rho(f: np.ndarray, mask: np.ndarray, d: int) -> tuple[float, float, int]:
    """Channel-wise Pearson rho of lag-`d` land-land pairs, E-W and N-S pooled.

    `f` is [C, H, W]; `mask` is [H, W] bool. Returns (channel median, channel MIN, pairs).
    Copied from `stage_roughness.py` deliberately — see the module docstring.
    """
    ew = mask[:, :-d] & mask[:, d:]
    ns = mask[:-d, :] & mask[d:, :]
    npair = int(ew.sum() + ns.sum())
    if npair == 0:
        return float("nan"), float("nan"), 0
    out = []
    for c in range(f.shape[0]):
        a = np.concatenate((f[c, :, :-d][ew], f[c, :-d, :][ns]))
        b = np.concatenate((f[c, :, d:][ew], f[c, d:, :][ns]))
        a, b = a - a.mean(), b - b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if den > 0.0:
            out.append(float((a * b).sum() / den))
    if not out:
        return float("nan"), float("nan"), npair
    o = np.array(out)
    return float(np.median(o)), float(o.min()), npair


# --------------------------------------------------------- G-17, before any model is built
rng = np.random.default_rng(15)
H0, W0 = 64, 64
m_all = np.ones((H0, W0), dtype=bool)
r_ramp = pair_rho(np.tile(np.arange(W0, dtype=np.float64), (H0, 1))[None], m_all, 1)[0]
r_board = pair_rho((((np.arange(H0)[:, None] + np.arange(W0)[None]) % 2) * 2.0 - 1.0)[None],
                   m_all, 1)[0]
r_noise = pair_rho(rng.standard_normal((1, H0, W0)), m_all, 1)[0]
ok17 = r_ramp > 0.99 and r_board < -0.99 and abs(r_noise) < 0.02
print(f"{'PASS' if ok17 else '⛔ FAIL'}  G-17  rho controls: ramp {r_ramp:+.4f}  "
      f"checkerboard {r_board:+.4f}  iid noise {r_noise:+.4f}")
if not ok17:
    sys.exit(1)

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
months = list(range(LEADIN_START, LAST_FED + 1))
vol, land, _ = build_volume(df, months)
GR, GC = land.shape
MASKS = {1: land}
for div in (2, 4):
    b = land[: (GR // div) * div, : (GC // div) * div]
    MASKS[div] = b.reshape(GR // div, div, GC // div, div).all(axis=(1, 3))

print("=" * 110)
print("G-F — DOES EITHER REMEDY REMOVE THE PHASE SIGNATURE? (untrained, one seed, three modes)")
print("=" * 110)
print(f"  raster          {GR}x{GC}, {int(land.sum()):,} land cells")
print(f"  months fed      {LEADIN_START}-{LAST_FED}; rho over the out-of-sample {OS_LO}-{OS_HI}")
print(f"  weights         ⛔ UNTRAINED, torch.manual_seed({SEED}) before each build")
print("  inference mode  ⛔ TEACHER-FORCED (vehicle.py:301-307), matching M92")
print(f"  M92 reference   trained control at up1: rho(1) {M92_TRAINED['up1_rho1']:+.6f}  "
      f"rho(2) {M92_TRAINED['up1_rho2']:+.6f}  gap "
      f"{M92_TRAINED['up1_rho2'] - M92_TRAINED['up1_rho1']:+.6f}")

res: dict[tuple[str, str, int], tuple[float, float]] = {}
for mode in MODES:
    torch.manual_seed(SEED)
    model = get_architecture(mode)(
        IN_CH, HID, OUT_CH, DROPOUT, output_distribution="nb",
        n_static_channels=0, static_top_skip=True, reg_activation="softplus",
    ).float().eval()

    n_par = sum(p.numel() for p in model.parameters())
    ok23 = n_par == EXPECTED_PARAMS[mode]
    n_do = 0
    for m in model.modules():                      # ⛔ G-24 / C-384: never a bare model.train()
        if isinstance(m, LockedDropout):
            m.locked = False
            m.eval()
            n_do += 1
    bn = [type(m).__name__ for m in model.modules()
          if "BatchNorm" in type(m).__name__ and m.training]
    if bn or not ok23 or n_do == 0:
        print(f"\n⛔⛔ VOID for {mode}: BatchNorm-in-train={len(bn)}  params={n_par:,} "
              f"(expected {EXPECTED_PARAMS[mode]:,})  LockedDropout found={n_do}")
        sys.exit(1)
    print(f"\n  {mode}")
    print(f"    PASS G-23  params {n_par:,} == PREREG §2's {EXPECTED_PARAMS[mode]:,}")
    print(f"    PASS G-24  no BatchNorm in train mode; {n_do} LockedDropout set to eval")

    caught: dict[str, torch.Tensor] = {}

    def _mk(key: str):
        def _h(_mod, _inp, out):
            caught[key] = out.detach()
        return _h

    handles = [getattr(model, spec).register_forward_hook(_mk(key)) for key, spec, _ in STAGES]
    acc: dict[str, dict[int, list[tuple[float, float, int]]]] = {
        k: {d: [] for d in LAGS} for k, _, _ in STAGES}
    with torch.no_grad():
        h0 = torch.zeros(1, HID, GR, GC)
        if not torch.equal(model(vol[:, 0], h0).reg, model(vol[:, 0], h0).reg):
            print(f"⛔⛔ VOID (G-25): two forwards differ for {mode}")
            sys.exit(1)
        print("    PASS G-25  two forwards on identical input are bit-identical")
        h = torch.zeros(1, HID, GR, GC)
        for t, fed in enumerate(months):
            caught.clear()
            out = model(vol[:, t], h)
            h = out.h_next
            if fed < OS_LO:
                continue
            for key, _, div in STAGES:
                mk = MASKS[div]
                f = caught[key][0].numpy().astype(np.float64)[:, : mk.shape[0], : mk.shape[1]]
                for d in LAGS:
                    acc[key][d].append(pair_rho(f, mk, d))
        for hd in handles:
            hd.remove()
    for key, _, _ in STAGES:
        for d in LAGS:
            res[(mode, key, d)] = (float(np.mean([r[0] for r in acc[key][d]])),
                                   float(np.mean([r[1] for r in acc[key][d]])))
    print(f"    swept in {time.time() - T0:.0f}s")

print("\n" + "=" * 110)
print("1. THE PHASE SIGNATURE, PER MODE — gap = rho(2) - rho(1). A phase pattern makes it LARGE.")
print("=" * 110)
hdr = ("mode", "stage", "rho(1)", "rho(2)", "GAP", "ch floor rho(1)")
print(f"  {hdr[0]:<22s}{hdr[1]:>6s}{hdr[2]:>11s}{hdr[3]:>11s}{hdr[4]:>11s}{hdr[5]:>18s}")
for mode in MODES:
    for key, _, _ in STAGES:
        r1, f1 = res[(mode, key, 1)]
        r2, _ = res[(mode, key, 2)]
        print(f"  {mode:<22s}{key:>6s}{r1:>+11.4f}{r2:>+11.4f}{r2 - r1:>+11.4f}{f1:>+18.4f}")

ctrl = res[("HydraBNUNet06_LSTM4", "up1", 2)][0] - res[("HydraBNUNet06_LSTM4", "up1", 1)][0]
print("\n" + "=" * 110)
print("2. ⛔ THE VOID CONDITION — PREREG.md §6 device 4, named before any number existed")
print("=" * 110)
print(f"  control gap at up1 = {ctrl:+.6f}   bar: must be >= {VOID_GAP:.2f}")
if ctrl < VOID_GAP:
    print("  ⛔⛔ VOID. The instrument does not detect on untrained weights what it detected on")
    print("      trained ones. No verdict is read from this run. Nothing else voids it.")
    sys.exit(1)
print("  ✅ NOT VOID — the untrained control reproduces the signature, so the remedies are")
print("     measured against an instrument that demonstrably detects it.")

print("\n" + "=" * 110)
print("3. ⭐⭐ THE PRE-REGISTERED VERDICT")
print("=" * 110)
gaps, floors = {}, {}
for mode in MODES[1:]:
    gaps[mode] = res[(mode, "up1", 2)][0] - res[(mode, "up1", 1)][0]
    floors[mode] = res[(mode, "up1", 1)][1]
    verdict = ("✅ removes it" if gaps[mode] < P20_GAP and floors[mode] >= P20_FLOOR
               else "⛔ does not")
    print(f"  {mode:<16s} gap {gaps[mode]:+.6f} (bar < {P20_GAP})   "
          f"channel floor {floors[mode]:+.6f} (bar >= {P20_FLOOR})   {verdict}")
passers = [m for m in MODES[1:] if gaps[m] < P20_GAP and floors[m] >= P20_FLOOR]
print(f"\n  P-20  at least one remedy removes the signature: "
      f"{'HELD' if passers else '⛔ REFUTED'}   ({len(passers)} of 2)")
print(f"  P-21  UpsampleConv1's gap below UpsampleConv2's: "
      f"{'HELD' if gaps['UpsampleConv1'] < gaps['UpsampleConv2'] else '⛔ REFUTED'}   "
      f"({gaps['UpsampleConv1']:+.6f} vs {gaps['UpsampleConv2']:+.6f})")
if passers:
    # PREREG §2: among the remedies that REMOVE the signature, the lowest parameter cost takes
    # the GPU slot. "Cost" is distance from the incumbent's count, so +-0 beats -20.4%: a
    # parameter-matched arm is uninterpretable in neither direction.
    n_inc = EXPECTED_PARAMS["HydraBNUNet06_LSTM4"]
    pick = min(passers, key=lambda m: abs(EXPECTED_PARAMS[m] - n_inc))
    tightest = min(passers, key=lambda m: gaps[m])
    print("\n  ⇒ ⭐⭐ THE ARM, chosen by these numbers and not by argument (PREREG §2):")
    print(f"       smallest residual gap      : {tightest} ({gaps[tightest]:+.6f})")
    print(f"       lowest parameter cost      : {pick} "
          f"({EXPECTED_PARAMS[pick]:,}, {100.0 * (EXPECTED_PARAMS[pick] - n_inc) / n_inc:+.1f}%)")
    print(f"       ⇒ PREREG §2 selects        : {pick}")
    if tightest != pick:
        print(f"       ⚠️ {tightest} removes MORE of the signature but costs more capacity. The")
        print("          prereg's rule stands as written; the trade is recorded in the FINDING.")
else:
    print("\n  ⇒ ⛔ NEITHER remedy removes the signature on untrained weights.")
    print("     Per S-2 this lowers confidence and closes nothing; per PREREG §6 device 3, any")
    print("     argument from me that this test was inadequate must be logged as a W- row FIRST.")

print("\n" + "=" * 110)
print("⚠️ LIMITS")
print("=" * 110)
print("  · ⛔ UNTRAINED weights. This measures the stage's GEOMETRY, not what training does with")
print("    it. A trained candidate could re-learn a phase pattern a 2x2 kernel permits.")
print("  · ⛔ TEACHER-FORCED (C-399), to match M92. The arm is scored autoregressively, and")
print("    EXP-01's whole finding is that the two do not agree.")
print("  · ⚠️ One seed. The architecture difference is deterministic given the seed, but the")
print("    MAGNITUDE of each gap is a one-draw quantity with no sigma-hat.")
print("  · sb head (head1) only; up0/up1/out on the reg branch.")
print(f"\n  wall time {time.time() - T0:.1f} s   peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
