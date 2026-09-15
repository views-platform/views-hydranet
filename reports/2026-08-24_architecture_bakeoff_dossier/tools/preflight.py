#!/usr/bin/env python3
"""Preflight — refuse to launch a ~29 GPU-hour queue on an unproven setup (dossier `03` §D, G4/G6).

Checks, in order, and **all of them before any 300-lesson arm**:

1. **CUDA is available and usable.** The repo's device report WARNS on CPU but does not
   raise; a CPU fallback would not fail, it would burn the 12.5 h `TRAIN_TIMEOUT` — twelve
   times.
   Peak memory is RECORDED but deliberately NOT gated here — the real memory gate is `smoke.sh`.
3. **A 300-lesson cost projection** from measured per-step time, vs the timeout and window.
4. **Every arm directory exists and is single-variable** against its own-seed control.

Writes `results/preflight.json` and a `PREFLIGHT_OK` sentinel. The launcher refuses without it.

This is C-310's mitigation made concrete: wall-clock is **measured here**, not estimated, because
this project has repeatedly sized guards from one observation and lost runs to it.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import torch

from views_hydranet.architectures.registry import get_architecture

_MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
ARCHES = (
    "AntiAliasedPool", "DynamicTopSkip", "FiLMSkip",
    "ShallowPool", "DualStream", "WideMemory",
)
TAGS = {"AntiAliasedPool": "aa", "DynamicTopSkip": "dyn", "FiLMSkip": "film",
        "ShallowPool": "shal", "DualStream": "dual", "WideMemory": "wide"}
SEEDS = {42: "fortytwo", 43: "fortythree"}

#: The incumbent's MEASURED end-to-end cost for one L=300 arm on this box, from real run logs
#: (`2026-08-17_ss_retention_dossier/results/run.log`): train+emit 81 / 81 / 96 min plus oracle
#: 9 / 9 / 13 min. The worst observed pair is taken, not the mean.
#:
#: The projection is a RATIO against this anchor, never an absolute model of the pipeline. The
#: first version of this script modelled a "step" as one forward+backward over `time_steps=36` and
#: projected 0.10 h for an arm that really costs ~1.8 h — a 17x underestimate that PASSED. That is
#: precisely the failure C-310 registers: a guard sized from a wrong model of the work is worse
#: than no guard, because it grants permission. Training really loops the full training sequence
#: (~384 steps) per window and then emits; a ratio cancels all of that.
INCUMBENT_ARM_HOURS = (96 + 13) / 60.0
INCUMBENT = "HydraBNUNet06_LSTM4"


def _resolve(p: Path) -> dict:
    t = p.read_text()
    ast.parse(t)
    ns: dict = {}
    exec(compile(t, str(p), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns["get_hp_config"]()


def measure(arch: str, hp: dict, steps: int, device: torch.device) -> dict:
    """Build at the arm's real widths and time a train step at the PRODUCTION window size.

    Timed with `torch.cuda.synchronize()` around the loop — without it the measurement records
    queue submission, not compute, and would understate exactly the heavy architectures this is
    meant to catch.
    """
    torch.cuda.reset_peak_memory_stats(device)
    net = get_architecture(arch)(
        hp["input_channels"], hp["total_hidden_channels"], hp["output_channels"],
        hp["dropout_rate"], output_distribution=hp["output_distribution"],
        n_static_channels=len(hp.get("static_channels", [])),
    ).to(device).train()
    dim = hp["window_dim"]
    x = torch.randn(1, hp["input_channels"], dim, dim, device=device)
    h = net.init_hTtime(net.base, dim, dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    for _ in range(3):  # warm-up: cuDNN autotune and allocator settle
        out = net(x, h)
        (out.reg.sum() + out.cls.sum()).backward()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = net(x, h)
        (out.reg.sum() + out.cls.sum()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    per_step = (time.perf_counter() - t0) / steps

    return {
        "params": sum(p.numel() for p in net.parameters()),
        "state_width": net.base,
        "peak_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "sec_per_step": per_step,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--timeout-hours", type=float, default=12.5, help="TRAIN_TIMEOUT at L=300")
    ap.add_argument("--window-hours", type=float, default=48.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    # ── 1. G6: hard device gate ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print(
            "PREFLIGHT FAIL: CUDA is not available. The device report only WARNS on CPU, so a "
            "CPU run would not fail — it would burn the 12.5 h timeout on every arm."
        )
        return 1
    device = torch.device("cuda")
    total_mib = torch.cuda.get_device_properties(device).total_memory / 1024**2

    # ── 4. arms exist and are single-variable ───────────────────────────────────────────────
    for arch in ARCHES:
        for seed, word in SEEDS.items():
            d = _MODELS / f"{TAGS[arch]}fullzero_{word}"
            ctrl = _MODELS / f"fullzero_{word}"
            if not (d / "configs" / "config_hyperparameters.py").is_file():
                problems.append(f"{d.name}: arm directory missing")
                continue
            a = _resolve(d / "configs" / "config_hyperparameters.py")
            c = _resolve(ctrl / "configs" / "config_hyperparameters.py")
            diff = {k for k in set(a) | set(c) if a.get(k) != c.get(k)}
            if diff != {"model"}:
                problems.append(
                    f"{d.name}: differs from {ctrl.name} in {sorted(diff)}, want ['model']"
                )
            if a.get("model") != arch:
                problems.append(f"{d.name}: model is {a.get('model')!r}, want {arch!r}")

    # ── 2/3. every architecture builds, runs, and is projected ──────────────────────────────
    ref = _resolve(_MODELS / "fullzero_fortytwo" / "configs" / "config_hyperparameters.py")
    seq_hint = args.steps
    results = {}
    for arch in (INCUMBENT, *ARCHES):
        try:
            m = measure(arch, ref, args.steps, device)
        except Exception as exc:  # noqa: BLE001 - a build/OOM failure is reported, not raised
            problems.append(f"{arch}: FAILED to run — {type(exc).__name__}: {exc}")
            continue
        m["headroom_mib"] = total_mib - m["peak_mib"]
        results[arch] = m

    # ── the ratio projection, anchored on the incumbent's measured real cost ────────────────
    if INCUMBENT not in results:
        problems.append("the incumbent failed to run — there is no anchor to project against")
        base_step = None
    else:
        base_step = results[INCUMBENT]["sec_per_step"]
    for arch, m in results.items():
        if base_step:
            m["cost_ratio_vs_incumbent"] = m["sec_per_step"] / base_step
            m["projected_arm_hours"] = INCUMBENT_ARM_HOURS * m["cost_ratio_vs_incumbent"]
        # NO memory GATE here, deliberately. A `/falsify guard` audit measured this check's margin
        # at x292 — a 64x-wider recurrent state still passed it silently — because a single-window
        # forward/backward is not the training footprint. A threshold that cannot fire is worse
        # than none: it reads as verified headroom. The peak stays as INFORMATION only; the real
        # memory gate is `smoke.sh`, which allocates through the real pipeline. Footprint does not
        # depend on lesson count, so a 2-lesson arm measures a 300-lesson arm exactly.
        if base_step and m["projected_arm_hours"] > args.timeout_hours:
            problems.append(
                f"{arch}: projected {m['projected_arm_hours']:.1f} h exceeds the "
                f"{args.timeout_hours} h TRAIN_TIMEOUT — the arm would be SIGKILLed with no "
                "checkpoint to resume from"
            )

    total_h = sum(r.get("projected_arm_hours", 0.0) for r in results.values() if r)
    total_h -= results.get(INCUMBENT, {}).get("projected_arm_hours", 0.0)  # controls already exist
    total_h *= 2  # two seeds
    blob = {
        "gpu_total_mib": total_mib,
        "timed_steps": seq_hint,
        "per_arch": results,
        "projected_total_queue_hours_2_seeds": total_h,
        "problems": problems,
    }
    (out / "preflight.json").write_text(json.dumps(blob, indent=2))

    hdr = f"{'architecture':22} {'params':>10} {'peak MiB':>9} {'s/step':>8}"
    print(f"{hdr} {'vs inc':>7} {'proj h/arm':>11}")
    for name, r in results.items():
        print(
            f"  {name:20} {r['params']:10,} {r['peak_mib']:9.0f} {r['sec_per_step']:8.4f} "
            f"{r.get('cost_ratio_vs_incumbent', float('nan')):7.2f} "
            f"{r.get('projected_arm_hours', float('nan')):11.2f}"
        )
    print(
        f"\n  anchor: the incumbent's MEASURED arm cost is {INCUMBENT_ARM_HOURS:.2f} h "
        f"(96 min train+emit + 13 min oracle, worst observed)"
    )
    print(f"  projected total for the 12 NEW arms: {total_h:.1f} h (window {args.window_hours} h)")
    if total_h > args.window_hours:
        problems.append(
            f"projected total {total_h:.1f} h exceeds the {args.window_hours} h window"
        )

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
