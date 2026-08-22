#!/usr/bin/env python3
"""run_freeze_arms.py — the four recurrent-state arms, one model, score-then-delete.

Pre-registration: `05_analysis_plan.md` (LOCKED). Runs `none` / `hidden` / `cell` / `all` on one
saved artifact — **emit only, no retraining** — and scores each on the activation-aware ruler.

The whole driver is one override:

    class _FreezeArmManager(HydranetManager):
        def _setup_evaluation(...):
            ctx = super()._setup_evaluation(...)
            ctx.orchestrator.freeze_recurrent = self.freeze_recurrent

`_EvaluationContext` is a NamedTuple, but the orchestrator it holds is mutable, so the arm is
selected without touching the manager, the config, or the pipeline. `freeze_recurrent` is not a
config key — ADR-027's retirement of `freeze_h` is untouched and
`tests/test_inference_logic.py::test_freeze_h_option_retired` stays green.

**Disk (C-154 / the 37 GB scar).** Each arm writes ~2.5 GB of cubes. This scores each arm and
deletes its prediction dir before starting the next, so two arms' cubes never coexist, and it
preflights free space. `--keep-cubes` opts out for a single-arm debug run.

Usage:
    python tools/run_freeze_arms.py --model truncated_smoke \
        --artifact calibration_model_20260814_003058.pt [--arms none,hidden,cell,all]
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

ARMS = ("none", "hidden", "cell", "all")


_ARM_PARSER = None


def _parse_arm(spec: str):
    """The child's own parser, imported so parent and child cannot disagree about a spec.

    Imported lazily so `--help` does not pay for the torch import `freeze_arm_entry` pulls in, and
    **cached** so the validation loop does not pay for it once per arm — the first version re-exec'd
    the whole model package for every spec it checked, which defeated the laziness it claimed.
    """
    global _ARM_PARSER
    if _ARM_PARSER is None:
        import importlib.util

        _p = Path(__file__).resolve().parent / "freeze_arm_entry.py"
        _s = importlib.util.spec_from_file_location("freeze_arm_entry", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _ARM_PARSER = _m.parse_arm
    return _ARM_PARSER(spec)


def _safe(arm: str) -> str:
    """`cell@0.5` -> `cell_0.5`: `@` is legal in a filename but reads badly in globs
    and in the seed-vs-arm filename parsing `freeze_table.py` does."""
    return arm.replace("@", "_")
HORIZONS = "1,6,12,18,24,30,36"
MIN_FREE_GB = 25.0  # ~2.5 GB/arm plus headroom; refuse to start rather than fill the disk


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _prediction_dirs(model_dir: Path) -> set[Path]:
    return set((model_dir / "data" / "generated").glob("predictions_*"))


def run_arm(
    model: str, arm: str, artifact: str, models_root: Path, out: Path, keep_cubes: bool
) -> dict:
    """Run one arm end to end. Returns the manifest record; raises on a failed run."""
    model_dir = models_root / model
    if not model_dir.is_dir():
        raise SystemExit(
            f"model directory not found: {model_dir}. A mistyped --model otherwise surfaces as an "
            "unhandled FileNotFoundError from the disk preflight, because globbing a nonexistent "
            "path returns an empty set without complaint."
        )
    before = _prediction_dirs(model_dir)

    # The pipeline names the prediction dir after the ARTIFACT timestamp, not the run — so every
    # arm of this batch writes to the SAME path. That is safe only because each arm's cubes are
    # deleted before the next starts. If a dir survives (a crash, --keep-cubes, an interrupted
    # run), the next arm would write into it and silently mix two arms' cubes into one score.
    # Refuse rather than contaminate.
    if before and not keep_cubes:
        raise SystemExit(
            f"refusing to start arm {arm!r}: {len(before)} prediction dir(s) already exist under "
            f"{model_dir / 'data' / 'generated'} "
            f"({', '.join(sorted(p.name for p in before))}). Every arm writes to the same path "
            "(named after the artifact), so a leftover dir would be mixed into this arm's cubes. "
            "Delete it, or re-run the batch from the start."
        )

    free = _free_gb(model_dir)
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"refusing to start arm {arm!r}: {free:.1f} GB free at {model_dir}, need "
            f"{MIN_FREE_GB} GB. Each arm writes ~2.5 GB of cubes (C-154)."
        )

    # One subprocess per arm, arm passed by argv (never an env var — the selected arm must be
    # visible in the process table and the log, not ambient). Subprocess rather than in-process
    # because each arm allocates GPU memory and a ~2.5 GB cube set; isolation makes the release
    # unconditional and stops one arm's crash taking the batch down.
    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "freeze_arm_entry.py"),
            "--arm",
            arm,
            "--model-dir",
            str(model_dir),
            "--artifact",
            artifact,
        ],
        cwd=str(model_dir),
        capture_output=True,
        text=True,
    )
    (out / f"{model}_{_safe(arm)}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"arm {arm!r} failed (rc={proc.returncode}); see {out / f'{model}_{arm}.log'}"
        )
    elapsed = time.time() - t0

    new = _prediction_dirs(model_dir) - before
    if len(new) != 1:
        raise SystemExit(
            f"arm {arm!r}: expected exactly one new prediction dir, found {len(new)}: "
            f"{sorted(str(p.name) for p in new)}. Refusing to score an ambiguous artifact."
        )
    pred_dir = new.pop()

    score_csv = out / f"score_{model}_{_safe(arm)}.csv"
    scored = subprocess.run(
        [
            sys.executable,
            str(_V2T / "score_v2_horizons.py"),
            f"{arm}|{pred_dir}|lr_{{t}}_best|by_{{t}}_best",
            # `=` form is REQUIRED: score_v2_horizons parses with `a.split("=", 1)[1]`, so the
            # space-separated form its own docstring shows raises IndexError. Cost the first
            # control arm its scoring pass (the cubes survived — deletion is post-scoring).
            "--targets=sb",
            f"--horizons={HORIZONS}",
            f"--out={score_csv}",
        ],
        capture_output=True,
        text=True,
    )
    (out / f"score_{model}_{_safe(arm)}.log").write_text(scored.stdout + scored.stderr)
    if scored.returncode != 0:
        raise SystemExit(
            f"scoring arm {arm!r} failed; see {out / f'score_{model}_{_safe(arm)}.log'}"
        )

    if not keep_cubes:
        shutil.rmtree(pred_dir)  # score-then-delete: two arms' cubes never coexist

    return {
        "model": model,
        "arm": arm,
        "artifact": artifact,
        "pred_dir": pred_dir.name,
        "score_csv": score_csv.name,
        "elapsed_s": round(elapsed, 1),
        "cubes_deleted": not keep_cubes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="views-models model dir name")
    # Pinned by the pre-registration, never defaulted — see freeze_arm_entry.py's --artifact note.
    ap.add_argument("--artifact", required=True, help="exact artifact filename to score")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--models-root", default=str(_MODELS_ROOT))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument(
        "--keep-cubes",
        action="store_true",
        help="DEBUG ONLY. Keeps each arm's cubes AND disables the leftover-prediction-dir check, "
        "so a second arm will write into the first arm's directory and the two are scored as one. "
        "It does NOT skip the free-space preflight. Never use it to work around a disk refusal.",
    )
    args = ap.parse_args()

    # Validate here as well as in the child: the child raises per-arm inside a subprocess, so a
    # typo in arm 3 of 4 would otherwise surface only after two arms had already burned GPU time.
    # Delegates to the child's parser so the two can never disagree about what a spec means.
    arms = tuple(a.strip() for a in args.arms.split(","))
    for a in arms:
        _parse_arm(a)  # raises SystemExit on anything malformed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"manifest_{args.model}.jsonl"
    # Appended across runs, so a re-run would otherwise interleave records with no way to tell
    # which pass produced which. A batch marker makes that legible; the committed manifest already
    # carries two incompatible schemas (the `none` row was hand-assembled after an out-of-band
    # rescore) with nothing saying so.
    with open(manifest_path, "a") as fh:
        fh.write(
            json.dumps({"batch_start": True, "arms": list(arms), "artifact": args.artifact}) + "\n"
        )

    for arm in arms:
        print(f"===== ARM {arm} ({args.model}) =====", flush=True)
        rec = run_arm(args.model, arm, args.artifact, Path(args.models_root), out, args.keep_cubes)
        with open(manifest_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  done in {rec['elapsed_s']}s -> {rec['score_csv']}", flush=True)

    (out / f"{args.model}_DONE").write_text("ok\n")  # sentinel for the detached finisher
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
