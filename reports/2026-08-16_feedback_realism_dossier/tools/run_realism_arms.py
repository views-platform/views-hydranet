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
    ap.add_argument("--models-root", default=str(_MODELS_ROOT))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--keep-cubes", action="store_true", help="debug only; skips the disk guard")
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"manifest_{args.model}_{args.tag}.jsonl"

    for arm in arms:
        print(f"===== ARM {arm} ({args.model}) =====", flush=True)
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
        )
        with open(manifest_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  done in {rec['elapsed_s']}s -> {rec['score_csv']}", flush=True)

    (out / f"{args.model}_{args.tag}_DONE").write_text("ok\n")  # sentinel for the monitor
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
