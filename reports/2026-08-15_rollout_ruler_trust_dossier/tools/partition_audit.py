#!/usr/bin/env python3
"""partition_audit.py — assert the provenance invariants instead of assuming them.

Epic #263 / S2 (#266). Three cluster-16 invariants were held by *reasoning* or by *reading a
config once*, never by an assertion:

* **C-217** — the 13 origins must roll entirely inside the held-out calibration test window.
  True only because the artifacts are calibration-partition trained (train 121-456); a
  validation-partition artifact (train 121-504) would put all 13 origins in-sample.
* **C-218** — a ``mean``-feedback rollout is broken by construction (DeepAR/Salinas2020 feeds
  ancestral samples), so it must never be scored as "deployed skill".
* **C-220** — CRPS is strictly proper only on a predictive distribution; on a 1-sample cube it
  silently degenerates to absolute error.

Plus two provenance checks: the truth parquet's sha must match the pinned ``V2_TRUTH_SHA256``,
and each arm's trained artifact must be **resolvable and recorded** (it fails loud if not).

On Giacomini & White 2006 §3.2 Comment 3 ("expanding window forecasting schemes are ruled out
by assumption"): the fixed-scheme property holds **by construction** — one prediction directory
is the output of one training run, so its 13 origins necessarily share an artifact. This module
does not *verify* that (there is nothing that could differ); it makes the claim **auditable**
by pinning which artifact, and refusing to proceed when that cannot be established.

Cheap by design: it reads ``identifiers.npz`` and array *headers* only, never a 2.5 GB cube, so
a bad arm fails before any expensive load.

Usage:
    python tools/partition_audit.py <arm>=<pred_dir> [<arm>=<pred_dir> ...] \\
        [--models-root /path/to/views-models] [--run-type calibration] [--out results/]

Repo root via ``Path(__file__).resolve().parents[3]`` — never a hardcoded path (C-247).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parents[3]
_SCRIPTS = _HN / "scripts"
_V2_RULER = _HN / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_ruler.py"
_DEFAULT_MODELS_ROOT = _HN.parent / "views-models" / "models"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from rollout_ruler_core import assert_sample_cube  # noqa: E402


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emitted_months(pred_dir: Path) -> dict:
    """Per-origin (min, max, n) emitted month, read from identifiers.npz only."""
    out = {}
    for odir in sorted(glob.glob(str(pred_dir / "origin_*"))):
        ids = sorted(glob.glob(os.path.join(odir, "*", "identifiers.npz")))
        if not ids:
            raise FileNotFoundError(f"no identifiers.npz under {odir}")
        t = np.load(ids[0])["time"]
        out[os.path.basename(odir)] = (int(t.min()), int(t.max()), int(len(np.unique(t))))
    if not out:
        raise FileNotFoundError(f"no origin_* dirs under {pred_dir}")
    return out


def cube_shapes(pred_dir: Path) -> dict:
    """Shape of every y_pred.npy, via mmap headers — no cube is read into memory."""
    out = {}
    for f in sorted(glob.glob(str(pred_dir / "origin_*" / "*" / "y_pred.npy"))):
        out[str(Path(f).relative_to(pred_dir))] = tuple(np.load(f, mmap_mode="r").shape)
    return out


def resolve_artifact(model_dir: Path, pred_dir: Path, run_type: str) -> dict:
    """Map a prediction dir to its single trained artifact by the shared run timestamp."""
    m = re.search(r"predictions_\w+?_(\d{8}_\d{6})$", pred_dir.name)
    if not m:
        raise ValueError(
            f"cannot resolve the artifact for {pred_dir.name}: the directory name does not "
            "carry a run timestamp, so the scored predictions cannot be tied to a training "
            "run. Provenance unestablished — refusing to record it as audited."
        )
    ts = m.group(1)
    sha_file = model_dir / "artifacts" / f"{run_type}_model_{ts}.pt.sha256"
    if not sha_file.exists():
        raise FileNotFoundError(
            f"no .sha256 sidecar at {sha_file}. The artifact behind these predictions cannot "
            "be identified, so the Giacomini fixed-scheme claim is unauditable for this arm. "
            "Returning a None sha here would have recorded 'audited' for something unchecked."
        )
    return {"timestamp": ts, "artifact_sha": sha_file.read_text().strip(), "note": None}


def audit_arm(
    pred_dir,
    model_dir,
    *,
    run_type: str = "calibration",
    diagnostic_only: bool = False,
    truth_parquet=None,
) -> dict:
    """Audit one arm's provenance. Raises on any violation; returns the audit record."""
    pred_dir, model_dir = Path(pred_dir), Path(model_dir)

    # --- partition (C-217) ------------------------------------------------------------
    parts = _load_by_path("cfg_partitions", model_dir / "configs" / "config_partitions.py")
    partition = parts.generate()[run_type]
    train_lo, train_max = partition["train"]

    months = emitted_months(pred_dir)
    min_emitted = min(v[0] for v in months.values())
    max_emitted = max(v[1] for v in months.values())
    leak = min_emitted <= train_max
    if leak:
        raise ValueError(
            f"C-217 LEAK: {pred_dir.name} emits from month {min_emitted}, but the {run_type} "
            f"train window ends at {train_max}. Those origins are IN-SAMPLE — every T>0 skill "
            "number computed from this arm would be optimistic. Refusing to score it."
        )

    # --- rollout feedback (C-218) -----------------------------------------------------
    hyp = _load_by_path("cfg_hyper", model_dir / "configs" / "config_hyperparameters.py")
    hp = hyp.get_hp_config() if hasattr(hyp, "get_hp_config") else {}
    feedback = hp.get("rollout_feedback")
    if feedback != "sample" and not diagnostic_only:
        raise ValueError(
            f"C-218: {model_dir.name} has rollout_feedback={feedback!r}. A mean-feedback "
            "rollout is broken by construction (a probabilistic recursive model rolls out on "
            "ancestral samples, Salinas2020) — its bloom is a method artifact, not a model "
            "property. Score it only with diagnostic_only=True, never as 'deployed skill'."
        )

    # --- sample cube, not a point forecast (C-220) ------------------------------------
    shapes = cube_shapes(pred_dir)
    if not shapes:
        raise FileNotFoundError(f"no y_pred.npy under {pred_dir}")
    for rel, shp in shapes.items():
        assert_sample_cube(shp, where=f"{pred_dir.name}/{rel}")
    widths = sorted({s[1] for s in shapes.values()})

    # --- one artifact per arm (Giacomini fixed-scheme) --------------------------------
    art = resolve_artifact(model_dir, pred_dir, run_type)

    # --- truth pin --------------------------------------------------------------------
    truth = {"path": None, "sha256": None, "matches_pin": None}
    if truth_parquet is None and _V2_RULER.exists():
        truth_parquet = _load_by_path("v2_ruler", _V2_RULER).V2_TRUTH
    if truth_parquet is not None and Path(truth_parquet).exists():
        v2 = _load_by_path("v2_ruler", _V2_RULER) if _V2_RULER.exists() else None
        sha = _sha256(Path(truth_parquet))
        pin = getattr(v2, "V2_TRUTH_SHA256", None)
        truth = {"path": str(truth_parquet), "sha256": sha, "matches_pin": (sha == pin)}
        if pin is not None and sha != pin:
            raise ValueError(
                f"truth sha mismatch: {truth_parquet} hashes to {sha[:16]}… but the pin is "
                f"{pin[:16]}…. The scoring substrate changed; comparisons across it are "
                "invalid (the 'different-months bug' class)."
            )

    return {
        "arm": model_dir.name,
        "pred_dir": str(pred_dir),
        "run_type": run_type,
        "partition": {"train": [train_lo, train_max], "test": list(partition["test"])},
        "n_origins": len(months),
        "min_emitted_month": min_emitted,
        "max_emitted_month": max_emitted,
        "leak": leak,
        "rollout_feedback": feedback,
        "diagnostic_only": diagnostic_only,
        "sample_widths": widths,
        "n_cubes_checked": len(shapes),
        "artifact_timestamp": art["timestamp"],
        "artifact_sha": art["artifact_sha"],
        "artifact_note": art["note"],
        "truth_sha256": truth["sha256"],
        "truth_matches_pin": truth["matches_pin"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="arm=pred_dir pairs")
    ap.add_argument("--models-root", default=str(_DEFAULT_MODELS_ROOT))
    ap.add_argument("--run-type", default="calibration")
    ap.add_argument("--diagnostic-only", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    records = []
    for spec in args.arms:
        arm, _, pred = spec.partition("=")
        records.append(
            audit_arm(
                Path(pred),
                Path(args.models_root) / arm,
                run_type=args.run_type,
                diagnostic_only=args.diagnostic_only,
            )
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition_audit.json").write_text(json.dumps(records, indent=2))

    lines = [
        "# Partition & provenance audit",
        "",
        f"Epic #263 / S2 (#266). Run type: `{args.run_type}`. {len(records)} arm(s).",
        "",
        "| arm | train | origins | emitted | leak | rollout_feedback | S | truth pin |",
        "|---|---|--:|---|:--:|---|--:|:--:|",
    ]
    for r in records:
        lines.append(
            f"| `{r['arm']}` | {r['partition']['train'][0]}–{r['partition']['train'][1]} | "
            f"{r['n_origins']} | {r['min_emitted_month']}–{r['max_emitted_month']} | "
            f"{'**LEAK**' if r['leak'] else 'no'} | `{r['rollout_feedback']}` | "
            f"{','.join(str(w) for w in r['sample_widths'])} | "
            f"{'ok' if r['truth_matches_pin'] else 'n/a'} |"
        )
    lines += [
        "",
        "**C-217** no origin emits at or before the train window's end. **C-218** every scored "
        "arm feeds back samples, not the mean. **C-220** every cube is 2-D with S > 1. "
        "**Giacomini fixed-scheme** each arm resolves to one artifact across all origins.",
        "",
    ]
    (out / "partition_audit.md").write_text("\n".join(lines))

    print(f"wrote {out / 'partition_audit.json'} and partition_audit.md ({len(records)} arms)")
    for r in records:
        print(
            f"  {r['arm']:<18} origins={r['n_origins']:>3} "
            f"emitted={r['min_emitted_month']}–{r['max_emitted_month']} "
            f"leak={r['leak']} feedback={r['rollout_feedback']} S={r['sample_widths']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
