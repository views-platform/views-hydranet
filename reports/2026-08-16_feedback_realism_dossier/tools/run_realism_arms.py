#!/usr/bin/env python3
"""run_realism_arms.py — the feedback-realism arms, one model, score-then-delete.

Pre-registration: ``05_analysis_plan.md`` (LOCKED). Runs each arm as one emit-only pass on a saved
artifact — **no retraining** — scores it on the activation-aware ruler, and records the per-step
statistics of the field the arm actually fed.

Same proven shape as ``2026-08-15_state_freeze_dossier/tools/run_freeze_arms.py``, which earned
each of these guards by tripping over the absence of one:

* **score-then-delete** — a scoring failure leaves the cubes intact for a re-score instead of
  costing a 26-minute regeneration (it did, once);
* **refuse on a leftover prediction dir** — the pipeline names it after the *artifact*, so every
  arm writes the same path and a survivor would be mixed into the next arm's score;
* **``--artifact`` required** — ``truncated_smoke`` has two calibration artifacts and the newer is
  the eps=0.1 SS arm, not the one the control must reproduce;
* **``--targets=sb`` in equals form** — ``score_v2_horizons`` parses with ``split("=", 1)[1]`` and
  raises ``IndexError`` on the space form its own docstring shows.

Usage:
    python tools/run_realism_arms.py --model truncated_smoke \\
        --artifact calibration_model_20260814_003058.pt \\
        --arms identity,use_real,wrong_month:-60
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HN = Path(__file__).resolve().parents[3]
_MODELS_ROOT = _HN.parent / "views-models" / "models"
_V2T = _HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "tools"

HORIZONS = "1,6,12,18,24,30,36"
MIN_FREE_GB = 25.0  # ~2.5 GB/arm plus headroom; refuse to start rather than fill the disk


def _safe(arm: str) -> str:
    """Filesystem-safe arm label: 'thin:0.25' -> 'thin_0.25'."""
    return arm.replace(":", "_")


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _prediction_dirs(model_dir: Path) -> set[Path]:
    return set((model_dir / "data" / "generated").glob("predictions_*"))


def run_arm(
    model: str,
    arm: str,
    artifact: str,
    models_root: Path,
    out: Path,
    keep_cubes: bool,
    length_scale: float | None = None,
    gate_probe: bool = False,
    freeze: str | None = None,
    body_mean_dump: bool = False,
    anchor_roll: int | None = None,
    per_step_roll: str | None = None,
) -> dict:
    """Run one arm end to end. Returns the manifest record; raises on any failure."""
    model_dir = models_root / model
    # An arm's identity is the spec PLUS its diagnostic knobs, so the OUTPUT NAME must carry both.
    # `--length-scale` was added 2026-08-16 without this, and the omission is silent-then-wrong: a
    # sweep over length scales runs three arms that all write `..._identity.csv`, each overwriting
    # the last, leaving one file that looks like a complete result. Caught 2026-08-17 only because
    # the arms were run one at a time. `--tag` does not help — it names the manifest and the
    # sentinel, not these files.
    label = _safe(arm)
    if length_scale is not None:
        label = f"{label}_ls{length_scale}"
    if freeze is not None:
        label = f"{label}_freeze{freeze}"
    if per_step_roll is not None:
        # In the label for the same reason every other knob is: the Wave 2 series runs input/hidden/
        # cell as three arms of one batch and they would otherwise overwrite each other.
        label = f"{label}_psr{per_step_roll.replace(':', '')}"
    if anchor_roll is not None:
        # MUST be in the label: the EXP-3 dose series runs roll 3, 15 and 90 as three arms of one
        # batch, and without this they all write `..._freezecell.csv` over each other and leave one
        # file that looks like a complete result. Same failure --length-scale caused in 2026-08.
        label = f"{label}_roll{anchor_roll}"
    before = _prediction_dirs(model_dir)

    # Every arm writes the SAME prediction path (named after the artifact, not the run), so a
    # survivor from a crash or an interrupt would be written into by this arm and two arms' cubes
    # would be scored as one. Refuse rather than contaminate.
    if before and not keep_cubes:
        raise SystemExit(
            f"refusing to start arm {arm!r}: {len(before)} prediction dir(s) already exist under "
            f"{model_dir / 'data' / 'generated'} "
            f"({', '.join(sorted(p.name for p in before))}). Delete them or restart the batch."
        )
    free = _free_gb(model_dir)
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"refusing to start arm {arm!r}: {free:.1f} GB free at {model_dir}, need "
            f"{MIN_FREE_GB} GB. Each arm writes ~2.5 GB of cubes (C-154)."
        )

    stats_csv = out / f"fedfield_{model}_{label}.csv"
    # Derived from the arm label, never taken from the caller. A flat caller-supplied directory
    # would let two arms of one batch write `bodymean_origin*.npz` over each other and leave a
    # complete-looking result that is silently half one arm and half the other — the exact
    # silent-then-wrong failure --length-scale caused before the label carried it.
    dump_dir = out / f"bodymean_{model}_{label}" if body_mean_dump else None
    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "realism_arm_entry.py"),
            "--arm",
            arm,
            "--model-dir",
            str(model_dir),
            "--artifact",
            artifact,
            "--stats-out",
            str(stats_csv),
            *(["--length-scale", str(length_scale)] if length_scale is not None else []),
            *(["--gate-out", str(out / f"gate_{model}_{label}.csv")] if gate_probe else []),
            *(["--freeze", freeze] if freeze is not None else []),
            *(["--body-mean-dump", str(dump_dir)] if dump_dir else []),
            *(["--anchor-roll", str(anchor_roll)] if anchor_roll is not None else []),
            *(["--per-step-roll", per_step_roll] if per_step_roll else []),
        ],
        cwd=str(model_dir),
        capture_output=True,
        text=True,
    )
    (out / f"{model}_{label}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"arm {arm!r} failed (rc={proc.returncode}); see {out / f'{model}_{label}.log'}"
        )
    elapsed = time.time() - t0

    new = _prediction_dirs(model_dir) - before
    if len(new) != 1:
        raise SystemExit(
            f"arm {arm!r}: expected exactly one new prediction dir, found {len(new)}: "
            f"{sorted(str(p.name) for p in new)}. Refusing to score an ambiguous artifact."
        )
    pred_dir = new.pop()

    score_csv = out / f"score_{model}_{label}.csv"
    scored = subprocess.run(
        [
            sys.executable,
            str(_V2T / "score_v2_horizons.py"),
            f"{label}|{pred_dir}|lr_{{t}}_best|by_{{t}}_best",
            "--targets=sb",
            f"--horizons={HORIZONS}",
            f"--out={score_csv}",
        ],
        capture_output=True,
        text=True,
    )
    (out / f"score_{model}_{label}.log").write_text(scored.stdout + scored.stderr)
    if scored.returncode != 0:
        raise SystemExit(f"scoring arm {arm!r} failed; see {out / f'score_{model}_{label}.log'}")

    if not keep_cubes:
        shutil.rmtree(pred_dir)  # score-then-delete: two arms' cubes never coexist

    return {
        "model": model,
        "arm": arm,
        "artifact": artifact,
        "pred_dir": pred_dir.name,
        "score_csv": score_csv.name,
        "fedfield_csv": stats_csv.name,
        "elapsed_s": round(elapsed, 1),
        "cubes_deleted": not keep_cubes,
        # Recorded because an arm's identity is the spec PLUS the diagnostic knobs. The batch-1/2
        # manifests carry only model+arm, so reconstructing which run had a correlation length set
        # meant reading the fed-field CSVs — the manifest could not answer it.
        "length_scale": length_scale,
        "gate_probe": gate_probe,
        "freeze_recurrent": freeze,
        "body_mean_dump": str(dump_dir) if dump_dir else None,
        "anchor_roll": anchor_roll,
        "per_step_roll": per_step_roll,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="views-models model dir name")
    ap.add_argument("--artifact", required=True, help="exact artifact filename to score")
    ap.add_argument("--arms", required=True, help="comma-separated arm specs")
    # #301 fade test. Composes with the feedback arm; the label carries it, for the same
    # silent-overwrite reason --length-scale does.
    ap.add_argument(
        "--freeze",
        default=None,
        choices=("hidden", "cell", "all"),
        help="clamp this half of the recurrent state to its last real-data value",
    )
    # EXP-3: comma-separated roll distances, run as one arm each (the dose series).
    ap.add_argument(
        "--anchor-rolls",
        default=None,
        help="comma-separated anchor roll distances, one arm per value (needs --freeze)",
    )
    ap.add_argument(
        "--per-step-roll",
        default=None,
        help='roll one driver each step, "<input|hidden|cell>:<shift>"',
    )
    ap.add_argument("--models-root", default=str(_MODELS_ROOT))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--keep-cubes", action="store_true", help="debug only; skips the disk guard")
    # silence-vs-fade: dump the un-composed body mean and gate per arm, so occurrence and
    # magnitude can be read apart offline. The directory is DERIVED from the arm label, so two
    # arms of one batch cannot overwrite each other.
    ap.add_argument(
        "--body-mean-dump",
        action="store_true",
        help="write the un-composed body-mean + gate fields for each arm (diagnostic)",
    )
    ap.add_argument("--tag", default="batch", help="label for the sentinel and manifest")
    # DIAGNOSTIC: the correlated feedback sampler. Omitted = production independent Bernoulli.
    # Without this the corr sweep had no reproducible entry point and was run ad hoc.
    ap.add_argument(
        "--length-scale",
        type=float,
        default=None,
        help="correlation length for the fed-back gate draw (the copula arm)",
    )
    # DIAGNOSTIC: the gate-structure probe. Opt-in and expensive — see HydraNetInference.
    ap.add_argument(
        "--gate-probe",
        action="store_true",
        help="also record the gate-structure CSV per arm (expensive)",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    # [None] = no roll dimension, which is every pre-EXP-3 batch and keeps them byte-identical.
    rolls = (
        [int(r.strip()) for r in args.anchor_rolls.split(",") if r.strip()]
        if args.anchor_rolls
        else [None]
    )
    if args.anchor_rolls and args.freeze is None:
        raise SystemExit(
            "--anchor-rolls needs --freeze: rolling an anchor nobody holds is a no-op."
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"manifest_{args.model}_{args.tag}.jsonl"

    for arm, roll in ((a, r) for a in arms for r in rolls):
        suffix = f" roll={roll}" if roll is not None else ""
        print(f"===== ARM {arm}{suffix} ({args.model}) =====", flush=True)
        rec = run_arm(
            args.model,
            arm,
            args.artifact,
            Path(args.models_root),
            out,
            args.keep_cubes,
            length_scale=args.length_scale,
            gate_probe=args.gate_probe,
            freeze=args.freeze,
            body_mean_dump=args.body_mean_dump,
            anchor_roll=roll,
            per_step_roll=args.per_step_roll,
        )
        with open(manifest_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  done in {rec['elapsed_s']}s -> {rec['score_csv']}", flush=True)

    (out / f"{args.model}_{args.tag}_DONE").write_text("ok\n")  # sentinel for the monitor
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
