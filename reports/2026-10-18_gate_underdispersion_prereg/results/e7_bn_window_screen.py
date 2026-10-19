"""E7 — does the TRAINING window size move the BatchNorm statistics the head is trained against?

WHY THIS IS THE SCREEN AND NOT THE EXPERIMENT. E6f established that the BatchNorm statistics
carry 0.41 of the calibration slope's movement with the weights untouched, which makes the
train/deploy geometry mismatch the leading root-cause suspicion: the head is trained against
per-batch moments on 32x32 activity-selected crops and deployed on the 360x720 grid. Testing
that properly means a retrain at a larger `window_dim`. This screen asks, for free, whether a
LEVER EXISTS AT ALL before anything is spent.

THE QUANTITY. BatchNorm normalises by its input's batch moments, so the decision-relevant
quantity is the input mean and variance at each BN layer, per channel, as a function of crop
size. Reported in BN's own units: the mean shift divided by the full-grid sd (because BN divides
by that sd), and the variance ratio.

PRE-COMMITTED READING, written before the run:
  * mismatch shrinks MONOTONICALLY and materially as the crop grows  -> a lever exists with a
    measured direction; a retrain at larger `window_dim` is worth pricing.
  * mismatch flat, or non-monotone in D                              -> no lever; the hypothesis
    dies here, for free, and the root cause is not window extent.

⛔ THE FULL GRID IS NOT THE CORRECT TARGET, and this script must not be read as if it were.
E6f REPLACED the running statistics with full-grid ones and the slope crossed 1.0 and overshot
to 0.7793 — worse in absolute terms. So "closer to the full grid" is not "better". This screen
reads the SHAPE of mismatch(D) only. ⛔ GOODHART BAN (PREREG, after E6f): no window size is to
be chosen by what it does to `b`. The axis has no correct point; it has a direction or it has
none.

MATCHED BY CONSTRUCTION, then asserted. `VolumeSampler.__init__` seeds a local generator from
`config['np_seed']` and `_generate_window` consumes the same number of draws whatever `dim` is
(one anchor call, then two `rng.integers(0, dim)`). So rebuilding the sampler per size replays
the IDENTICAL anchor sequence, and comparing sizes compares extent around the SAME anchors
rather than two different sets of places. VOID rule 2 asserts it rather than assuming it.

VOID RULES
  1  any BatchNorm module in train() mode -> VOID. This script READS input moments through
     forward-pre-hooks and must never write a running buffer (C-112/C-119: scratch only).
  2  the anchor sequence not identical across sizes -> VOID, the comparison is not matched.
  3  any non-finite moment -> VOID.
  4  a crop that is not exactly (1, T, C, D, D) -> VOID.
  5  less than MEM_FLOOR_GB available at either checkpoint -> VOID. ⛔ Earned: the OOM killer
     took two runs of this script at ~11.7 GB resident. The global raster and the dataframes
     behind it are built, and the full-grid forward pass needs its own gigabytes of activations,
     so the build is FREED before any pass runs and the floor is checked either side.

USAGE
  python3 e7_bn_window_screen.py --smoke 4     # 4 windows per size, to time it
  python3 e7_bn_window_screen.py               # the screen
"""
import argparse
import gc
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
DIAG = HN / "reports/2026-10-12_underfire_diagnosis"
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(HN))
import vehicle  # noqa: E402

# Identical accommodation to e6f_bn_reestimate.py and run_p2.py: vehicle.EXPECTED_SIDECAR
# hardcodes violet_visitor's 'nb' and these arms are 'truncated_nb'. Asserted, not silenced.
assert vehicle.EXPECTED_SIDECAR["output_distribution"] == "nb", "gate already changed; stop"
vehicle.EXPECTED_SIDECAR["output_distribution"] = "truncated_nb"
from vehicle import TARGETS, grid_rowcol, load_vehicle, pull  # noqa: E402

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.utils.curriculum import CurriculumLearner  # noqa: E402
from views_hydranet.utils.feature_scaler import FeatureScaler  # noqa: E402
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402
from views_hydranet.utils.volume_sampler import VolumeSampler  # noqa: E402

ART = (MODELS / "amber_kestrel/data/generated/calibration_delivery_20261013/artifact"
       / "calibration_model_20261013_010739.pt")
SIZES = [32, 64, 128]
N_WIN, WARM, STEPS = 300, 4, 12
NP_SEED, TORCH_SEED = 42, 42
MEM_FLOOR_GB = 6.0          # measured floor, VOID rule 5


def mem_available_gb():
    """MemAvailable, the kernel's own estimate of what can be had without swapping."""
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1e6
    raise SystemExit("VOID: cannot read MemAvailable")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def need(gb, where):
    """VOID rule 5. The OOM killer took two runs of this script at ~11.7 GB resident before
    this guard existed: building the global raster and then running a full-grid forward pass in
    one process. Refuse up front rather than be killed halfway."""
    av = mem_available_gb()
    print(f"    [mem] {where}: {av:.1f} GB available, peak so far {peak_gb():.1f} GB")
    if av < gb:
        raise SystemExit(f"VOID rule 5 ({where}): {av:.1f} GB available, need {gb:.1f} GB. "
                         "Close something or lower N_WIN.")


def hook_bn(model):
    """Forward-PRE hooks capturing each BN layer's input moments. Reads only; writes nothing."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise SystemExit("VOID: no BatchNorm layers found — wrong architecture?")
    grab = {}

    def pre(mod, inp):
        x = inp[0].detach()
        grab.setdefault(id(mod), []).append(
            (x.mean(dim=(0, 2, 3)).numpy(), x.var(dim=(0, 2, 3), unbiased=False).numpy()))

    for m in bns:
        m.register_forward_pre_hook(pre)
    return bns, grab


def assert_eval_only(model, where):
    """VOID rule 1. BN accumulates in train() mode; nothing here may put it there."""
    bad = [type(m).__name__ for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training]
    if bad:
        raise SystemExit(f"VOID ({where}): {len(bad)} BatchNorm modules in train() mode")


def run_steps(model, tens, C_hidden, grab):
    """WARM + STEPS forward passes from a zero state. Returns the moments captured in STEPS."""
    grab.clear()
    h = torch.zeros(1, C_hidden, tens.shape[-2], tens.shape[-1])
    with torch.no_grad():
        for t in range(WARM):
            h = model(tens[:, t], h).h_next
        grab.clear()                      # warm-up moments discarded
        for t in range(WARM, WARM + STEPS):
            h = model(tens[:, t], h).h_next
    return {k: (np.stack([a for a, _ in v]), np.stack([b for _, b in v]))
            for k, v in grab.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="windows per size; times it and stops")
    a = ap.parse_args()
    n_win = a.smoke or N_WIN

    ReproducibilityGate.lock_entropy(np_seed=NP_SEED, torch_seed=TORCH_SEED)
    model, mcfg, sha = load_vehicle(ART)
    model.eval()
    model.set_locked_dropout(True)        # ADR-057: inference runs MC-dropout
    assert_eval_only(model, "before any pass")
    bns, grab = hook_bn(model)
    C_hidden = mcfg["total_hidden_channels"]
    print(f"amber sha {sha[:12]}…  {len(bns)} BN layers hooked  hidden {C_hidden}")

    spec = importlib.util.spec_from_file_location(
        "hp", MODELS / "amber_kestrel/configs/config_hyperparameters.py")
    hp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp)
    cfg = hp.get_hp_config()
    if cfg["window_dim"] != SIZES[0]:
        raise SystemExit(f"VOID: as-run window_dim is {cfg['window_dim']}, not {SIZES[0]} — "
                         "the baseline size is wrong and every comparison is mislabelled")

    # ---- build the training volume, take what the screen needs, then FREE the build ----
    # ⛔ The raster plus the dataframes behind it peak near 8 GB and the full-grid forward pass
    # needs its own gigabytes. Holding both is what the OOM killer took. So: the activity map
    # is computed on the FULL time axis (it is what selects the anchors, and slicing it would
    # change the training distribution), the volume is then CUT to the steps actually used, the
    # lesson sequence is materialised, and everything upstream is dropped before any pass runs.
    need(MEM_FLOOR_GB, "before building the volume")
    df = pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
    tr = df[(df.month_id >= 121) & (df.month_id <= 456)].copy()
    r, c = grid_rowcol(tr.priogrid_id.to_numpy())
    tr["row"], tr["col"], tr["c_id"] = r, c, 0
    tr = tr.rename(columns=dict(zip(TARGETS, cfg["regression_targets"])))
    scaled = FeatureScaler(cfg).fit_transform(tr)
    handler = VolumeHandler.from_df(scaled, cfg)
    planner = CurriculumLearner(cfg, handler)
    tvh = VolumeSampler(handler, cfg).get_train_volume()
    act = {n: np.count_nonzero(tvh.data[..., tvh.channel_map.index(n)], axis=0)
           for n in planner.subjects}
    lessons = [planner.get_lesson(w % 30) for w in range(n_win)]
    print(f"train volume {tvh.data.shape}  (T, H, W, C)  -> cutting to {WARM + STEPS} steps")

    # ⚠️ slice_time returns a VIEW, so the parent raster stays alive and is NOT freed here.
    # Measured and printed rather than assumed — the dataframes are the part that goes.
    print(f"    raster {tvh.data.nbytes/1e9:.2f} GB, {len(tvh.channel_map)} channels "
          f"(stays resident: slice_time is a view)")
    tvh = tvh.slice_time(0, WARM + STEPS)
    del df, tr, scaled, handler, planner
    gc.collect()
    need(MEM_FLOOR_GB, "after freeing the dataframes")

    # ---- reference: the full training extent, ONE spatial pass (there is only one grid) ----
    full_t = tvh.to_pytorch()
    t0 = time.time()
    ref = run_steps(model, full_t, C_hidden, grab)
    assert_eval_only(model, "after the full-grid pass")
    t_full = time.time() - t0
    ref_mean = {k: v[0].mean(0) for k, v in ref.items()}
    ref_sd = {k: np.sqrt(v[1].mean(0)) for k, v in ref.items()}
    ref_var = {k: v[1].mean(0) for k, v in ref.items()}
    nf = sum(int(np.sum(~np.isfinite(m))) for m in ref_mean.values())
    if nf:
        raise SystemExit(f"VOID rule 3: {nf} non-finite full-grid moments")
    print(f"full-grid reference: {tuple(full_t.shape)}  {t_full:.1f}s for {WARM}+{STEPS} steps")

    # ---- crops, matched anchors ----
    out, anchors_ref = {}, None
    for D in SIZES:
        cfg["window_dim"] = D
        sampler = VolumeSampler(tvh, cfg)              # re-seeds rng from cfg['np_seed']
        anchors, orig = [], sampler._select_anchor
        sampler._select_anchor = lambda *x, **k: anchors.append(orig(*x, **k)) or anchors[-1]
        shift, vrat = [], []
        t0 = time.time()
        for w in range(n_win):
            tgt, thr = lessons[w]
            vh = sampler._generate_window(tvh, act[tgt], thr)
            tens = vh.to_pytorch()
            if tens.shape[-2:] != (D, D) or tens.shape[0] != 1:
                raise SystemExit(f"VOID rule 4: crop {tuple(tens.shape)} at D={D}")
            got = run_steps(model, tens, C_hidden, grab)
            for k, (mu, var) in got.items():
                shift.append((mu.mean(0) - ref_mean[k]) / np.maximum(ref_sd[k], 1e-6))
                vrat.append(var.mean(0) / np.maximum(ref_var[k], 1e-12))
        dt = time.time() - t0
        assert_eval_only(model, f"after D={D}")
        if anchors_ref is None:
            anchors_ref = anchors
        elif anchors != anchors_ref:
            n_bad = sum(1 for x, y in zip(anchors, anchors_ref) if x != y)
            raise SystemExit(f"VOID rule 2: anchor sequence differs at D={D} "
                             f"({n_bad}/{len(anchors)} windows) — sizes are not matched")
        s, v = np.abs(np.concatenate(shift)), np.concatenate(vrat)
        if not (np.isfinite(s).all() and np.isfinite(v).all()):
            raise SystemExit(f"VOID rule 3: non-finite moment at D={D}")
        out[D] = dict(n_win=n_win, sec=dt, sec_per_win=dt / n_win,
                      shift_med=float(np.median(s)), shift_p90=float(np.percentile(s, 90)),
                      vrat_med=float(np.median(v)), vrat_p90=float(np.percentile(v, 90)),
                      n_chan=int(s.size))
        print(f"  D={D:<4} {n_win} win  {dt/60:5.1f} min ({dt/n_win:.2f} s/win)  "
              f"peak {peak_gb():.1f} GB  "
              f"|shift| med {out[D]['shift_med']:.4f} p90 {out[D]['shift_p90']:.4f}   "
              f"var ratio med {out[D]['vrat_med']:.4f}")

    if a.smoke:
        tot = sum(out[D]["sec_per_win"] * N_WIN for D in SIZES) + t_full
        print(f"\n⚠️  SMOKE ONLY ({n_win} windows/size). Projected full screen: {tot/60:.1f} min. "
              "Nothing concluded.")
        return

    print(f"\n{'D':>5}{'|shift| med':>14}{'|shift| p90':>14}{'var ratio med':>16}"
          f"{'var ratio p90':>16}   (BN units, vs the full extent)")
    for D in SIZES:
        o = out[D]
        print(f"{D:>5}{o['shift_med']:>14.4f}{o['shift_p90']:>14.4f}"
              f"{o['vrat_med']:>16.4f}{o['vrat_p90']:>16.4f}")

    ms = [out[D]["shift_med"] for D in SIZES]
    mono = all(ms[i] > ms[i + 1] for i in range(len(ms) - 1))
    span = ms[0] / max(ms[-1], 1e-12)
    print(f"\nmonotone decreasing in D: {mono}   ratio D={SIZES[0]} / D={SIZES[-1]}: {span:.2f}×")
    if mono and span >= 1.5:
        print("⛔ A LEVER EXISTS — mismatch falls monotonically with crop size.\n"
              "   Price the retrain. ⛔ And it is a DIRECTION, not a target:\n"
              "   E6f proved the full grid is not the correct point either.")
    elif not mono:
        print("⛔ NON-MONOTONE — no window size may be extrapolated to. Hypothesis DIES here.")
    else:
        print(f"⛔ FLAT ({span:.2f}× over a 4× crop) — window extent is not the lever. DIES here.")

    rec = dict(sha=sha[:16], sizes=SIZES, n_win=N_WIN, warm=WARM, steps=STEPS,
               n_bn=len(bns), full_shape=list(full_t.shape), full_sec=t_full,
               monotone=bool(mono), span=float(span), per_size=out)
    (HERE / "e7_window_screen.json").write_text(json.dumps(rec, indent=2))
    print(f"\nwrote {HERE / 'e7_window_screen.json'}")


if __name__ == "__main__":
    main()
