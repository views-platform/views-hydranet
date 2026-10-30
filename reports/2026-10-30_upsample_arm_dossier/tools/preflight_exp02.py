#!/usr/bin/env python3
"""Preflight for EXP-02 (candidate 7) — refuse to launch ~7 GPU-hours on an unproven setup.

A SIBLING of `reports/2026-08-24_architecture_bakeoff_dossier/tools/preflight.py`, not a copy:
the measurement machinery (`measure`, `_resolve`) and the **measured** cost anchor
(`INCUMBENT_ARM_HOURS`) are IMPORTED from it, so neither is restated. Restating a measurement is
the `C-362` defect — a derived constant pinned away from the artifact it came from.

WHY A SIBLING AT ALL, when C7-3 says a gate's requirements may not fork
----------------------------------------------------------------------
They do not fork: `launch_exp02.sh` demands `PREFLIGHT_OK` exactly as `launch_bakeoff.sh` does.
What differs is the **control semantics**, and that cannot be expressed as a flag on the parent:

  * EXP-01 pairs each arm against a PRE-EXISTING `fullzero_<seed>` control and reads its reference
    config from `fullzero_fortytwo`. ⛔ Those directories **no longer exist** (`ls views-models/
    models | grep fullzero` is empty), so the parent would fail there today for a reason that has
    nothing to do with this experiment.
  * EXP-02's control is an **arm of its own queue** (`ctlfullzero_<seed>`), built by the same
    builder on the same commit, because pairing against August's committed scores would vary the
    CODE VERSION as well as `model`.

Parameterising the parent would therefore mean changing EXP-01's control semantics, i.e. editing a
committed instrument. This file supplies EXP-02's semantics and nothing else.

⛔ IT WILL FAIL UNTIL 0.1.8 IS RELEASED, AND THAT IS THE POINT
--------------------------------------------------------------
`preflight.py` imports `views_hydranet.architectures.registry` at module scope, which resolves to
the **INSTALLED** wheel. `UpsampleConv2` does not exist in 0.1.5, so this refuses.

⭐ That refusal is correct and is **deliberately not worked around**. An earlier version of the
EXP-02 tools set `PYTHONPATH` to point the run at the working tree; that would have trained code
that was never released, on a path unique to one shell, while every other model in views-models
runs the wheel. **All six EXP-01 candidates were in the published package before their arms ran.**
Candidate 7 waits for the release. There is no `PYTHONPATH` in this file.

WHAT IT CHECKS, in order
------------------------
1. **CUDA is available.** The repo's device report only WARNS on CPU, so a CPU fallback would not
   fail — it would burn the 12.5 h `TRAIN_TIMEOUT`, four times.
2. **Every arm directory exists and is single-variable** against its own-seed control: the configs
   must differ in exactly `model`, and `model` must be the architecture the label claims.
   ⚠️ The four dirs are built by `make_arch_arm.py --lessons 300` BEFORE this runs — smoke builds
   `*tinyzero*` dirs at 2 lessons, which are different directories.
3. **Every architecture builds and runs**, with parameter count and peak memory recorded.
   ⚠️ Memory is RECORDED, not gated — `smoke_exp02.sh` is the real memory gate, on the real
   pipeline.
   ⛔⛔ **AND THE COST RATIO IS MEASURED AT THE WRONG RESOLUTION, deliberately and stated here.**
   The imported `measure()` times at `dim = hp["window_dim"]` = **32**, which its own docstring
   calls "the PRODUCTION window size" — but production inference is
   `hp["height"] x hp["width"]` = **360x720**. Measured on this candidate the two disagree in
   DIRECTION, not just magnitude: at window 32 with a 12-step BPTT backward `UpsampleConv2` is
   **0.97x** the incumbent, while a full 360x720 forward is **1.41x** (the twelve stages go
   6.37 -> 25.48 GMAC, exactly 4x).
   ⇒ **This gate therefore bounds the TRAINING cost, which is what the queue window is about, and
   says nothing about rollout or delivered-forecast cost.** That is the right scope for a launch
   gate and the wrong scope for a promotion decision, so it is named rather than silently relied
   on. The rollout cost is ~+7.5 min/arm (1,872 forwards), recorded in `02_design`.
4. **A cost projection** for the four arms, as a RATIO against the imported measured anchor, versus
   the window.

Writes `results/preflight.json` and `PREFLIGHT_OK`. `launch_exp02.sh` refuses without the sentinel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BAKEOFF_TOOLS = _HERE.parents[1] / "2026-08-24_architecture_bakeoff_dossier" / "tools"
sys.path.insert(0, str(_BAKEOFF_TOOLS))

from make_arch_arm import arm_label  # noqa: E402
from preflight import _MODELS, INCUMBENT_ARM_HOURS, _resolve, measure  # noqa: E402

#: control first, then the arm — the order `PREREG.md` section 10d pre-registers.
ARCHES = ("HydraBNUNet06_LSTM4", "UpsampleConv2")
CONTROL = "HydraBNUNet06_LSTM4"
SEEDS = (42, 43)
LESSONS, EPS = 300, 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--window-hours", type=float, default=48.0)
    #: ⛔ RESTORED. The parent declares this and raises per-arm on it; the extraction into this
    #: file kept only the AGGREGATE window check, so an arm projected over the SIGKILL could pass
    #: while the total sat under the window. `run_lesson_arm.sh:79` sets
    #: TRAIN_TIMEOUT=$((150 * L)) = 45,000 s = 12.5 h at L=300, enforced by `timeout -k 120`, and
    #: a killed arm leaves no checkpoint to resume from.
    ap.add_argument("--timeout-hours", type=float, default=12.5,
                    help="TRAIN_TIMEOUT at L=300, as run_lesson_arm.sh computes it")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    # ── 1. hard device gate ─────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print(
            "PREFLIGHT FAIL: CUDA is not available. The device report only WARNS on CPU, so a "
            "CPU run would not fail — it would burn the 12.5 h timeout on every arm."
        )
        return 1
    device = torch.device("cuda")
    total_mib = torch.cuda.get_device_properties(device).total_memory / 1024**2

    # ── 2. the four arms exist and are single-variable against their own-seed control ───────
    labels = {
        (arch, seed): arm_label(lessons=LESSONS, eps=EPS, seed=seed, variant=arch)
        for arch in ARCHES
        for seed in SEEDS
    }
    for (arch, seed), label in labels.items():
        d = _MODELS / label
        ctrl = _MODELS / labels[(CONTROL, seed)]
        cfg_p = d / "configs" / "config_hyperparameters.py"
        if not cfg_p.is_file():
            problems.append(
                f"{label}: arm directory missing — build it with "
                f"make_arch_arm.py --lessons {LESSONS} --eps {EPS} --seed {seed} "
                f"--variant {arch}"
            )
            continue
        if arch == CONTROL:
            continue  # the control is its own reference; nothing to diff against
        c_p = ctrl / "configs" / "config_hyperparameters.py"
        if not c_p.is_file():
            problems.append(f"{label}: its own-seed control {ctrl.name} is missing")
            continue
        a, c = _resolve(cfg_p), _resolve(c_p)
        diff = {k for k in set(a) | set(c) if a.get(k) != c.get(k)}
        if diff != {"model"}:
            problems.append(
                f"{label}: differs from {ctrl.name} in {sorted(diff)}, want ['model'] — this is "
                "the M45 two-key failure, and it makes the arm uninterpretable"
            )
        if a.get("model") != arch:
            problems.append(f"{label}: model is {a.get('model')!r}, want {arch!r}")

    # ── 3. every architecture builds and runs; memory RECORDED, gated by smoke ─────────────
    ref_label = labels[(CONTROL, SEEDS[0])]
    ref_p = _MODELS / ref_label / "configs" / "config_hyperparameters.py"
    if not ref_p.is_file():
        print(f"PREFLIGHT FAIL: no reference config at {ref_p} — the control arm must be built "
              "first; it is the reference every projection is a ratio against.")
        return 1
    ref = _resolve(ref_p)
    results: dict[str, dict] = {}
    for arch in ARCHES:
        try:
            m = measure(arch, ref, args.steps, device)
        except Exception as exc:  # noqa: BLE001 — a build/OOM failure is reported, not raised
            problems.append(f"{arch}: FAILED to run — {type(exc).__name__}: {exc}")
            continue
        m["headroom_mib"] = total_mib - m["peak_mib"]
        results[arch] = m

    # ── 4. the ratio projection, anchored on the IMPORTED measured arm cost ────────────────
    if CONTROL not in results:
        problems.append("the control failed to run — there is no anchor to project against")
        total_h = float("nan")
    else:
        base = results[CONTROL]["sec_per_step"]
        for m in results.values():
            m["cost_ratio_vs_control"] = m["sec_per_step"] / base
            m["projected_arm_hours"] = INCUMBENT_ARM_HOURS * m["cost_ratio_vs_control"]
        # four runs: every architecture at every seed, the control included (it is re-run)
        total_h = len(SEEDS) * sum(m["projected_arm_hours"] for m in results.values())
        #: ⛔ PER-ARM first: an arm over the SIGKILL is fatal on its own, however small the total.
        for name, m in results.items():
            if m["projected_arm_hours"] > args.timeout_hours:
                problems.append(
                    f"{name}: projected {m['projected_arm_hours']:.1f} h exceeds the "
                    f"{args.timeout_hours} h TRAIN_TIMEOUT — the arm would be SIGKILLed with no "
                    "checkpoint to resume from"
                )
        if total_h > args.window_hours:
            problems.append(
                f"projected total {total_h:.1f} h exceeds the {args.window_hours} h window"
            )

    blob = {
        "gpu_total_mib": total_mib,
        "timed_steps": args.steps,
        "arm_labels": {f"{a}:{s}": label for (a, s), label in labels.items()},
        "per_arch": results,
        "projected_total_queue_hours_4_runs": total_h,
        "problems": problems,
    }
    (out / "preflight.json").write_text(json.dumps(blob, indent=2))

    print(f"  {'architecture':<22}{'params':>10}{'peak MiB':>10}{'s/step':>9}"
          f"{'vs ctrl':>9}{'proj h/arm':>12}")
    for name, r in results.items():
        print(f"  {name:<22}{r['params']:>10,}{r['peak_mib']:>10.0f}{r['sec_per_step']:>9.4f}"
              f"{r.get('cost_ratio_vs_control', float('nan')):>9.2f}"
              f"{r.get('projected_arm_hours', float('nan')):>12.2f}")
    print(f"\n  anchor (imported, measured): one L=300 arm = {INCUMBENT_ARM_HOURS:.2f} h")
    print(f"  projected total for the 4 runs: {total_h:.1f} h (window {args.window_hours} h)")

    if problems:
        print("\nPREFLIGHT FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    (out / "PREFLIGHT_OK").write_text(json.dumps(blob, indent=2))
    print("\nPREFLIGHT OK — sentinel written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
