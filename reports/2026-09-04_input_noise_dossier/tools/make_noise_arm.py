#!/usr/bin/env python3
"""make_noise_arm.py — build Epic #311's two S5 arms as their own model directories.

Floor is `fullzero_fortytwo` (ε=0.0, 300 lessons, seed 42) — the shape S5's control takes, and the
vehicle S1 measured. Adapted from `2026-09-03_bptt_sa_dossier/tools/make_bptt_arm.py`; every guard
there is kept because each exists for a recorded failure: the floor's config is never written;
verification execs both configs and asserts the symmetric difference of the resolved dicts is
EXACTLY the intended key set, so an unintended change anywhere fails loud; the queryset identity is
pinned to the floor's so both arms share one cached parquet and cannot diverge by data vintage; and
`diagnostic_visualizations` is forced off (~6 h/emit if left on).

BOTH arms are retrained, including the control. Reusing the floor's artifact would save ~110 min by
assuming the day's code changes did not touch the ε=0 path — and that assumption is what the C-324
episode punished.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

_MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
_FLOOR = _MODELS / "fullzero_fortytwo"

#: ModelPathManager enforces ^[a-z]+_[a-z]+$ — two lowercase words, no digits.
_ARMS = {
    "control": ("noisecontrol_fortytwo", None),
    "noise": ("noisedropout_fortytwo", 0.204),
}


def _insert_after(text: str, anchor_key: str, line: str) -> str:
    pat = re.compile(rf"('{re.escape(anchor_key)}':\s*[^\n]*?,)")
    new, n = pat.subn(rf"\g<1>\n    {line}", text, count=1)
    if n != 1:
        raise SystemExit(f"make_noise_arm: anchor {anchor_key!r} not found exactly once")
    return new


def _set_lessons(text: str, lessons: int) -> str:
    """Rewrite `total_lessons` — smoke arms only."""
    pat = re.compile(r"('total_lessons':\s*)[^\n]*?,")
    new, n = pat.subn(rf"\g<1>{lessons},", text, count=1)
    if n != 1:
        raise SystemExit("make_noise_arm: 'total_lessons' not found exactly once")
    return new


def _resolve(path: Path, fn: str) -> dict:
    text = path.read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns[fn]()


def build(
    which: str, *, out_root: Path = _MODELS, label: str | None = None, lessons: int | None = None
) -> Path:
    default_label, dropout = _ARMS[which]
    label = label or default_label
    dest = out_root / label
    if dest.exists():
        raise SystemExit(f"make_noise_arm: {dest} exists — refusing to overwrite an arm")

    shutil.copytree(
        _FLOOR,
        dest,
        ignore=shutil.ignore_patterns(
            "data", "logs", "wandb", "artifacts", "reports", "notebooks", "__pycache__", "*.pyc"
        ),
    )
    for sub in ("data/generated", "data/processed", "artifacts", "logs", "notebooks", "reports"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    src_raw = _FLOOR / "data" / "raw"
    if not src_raw.is_dir():
        raise SystemExit(f"make_noise_arm: {src_raw} missing — an arm would fetch data mid-run")
    shutil.copytree(src_raw, dest / "data" / "raw")

    hp = dest / "configs" / "config_hyperparameters.py"
    text = hp.read_text()
    if dropout is not None:
        text = _insert_after(
            text, "ss_feedback", f"'input_noise_dropout': {dropout!r},  # #311 the ONE variable"
        )
    if lessons is not None:
        text = _set_lessons(text, lessons)
    hp.write_text(text)

    # Queryset identity — two legal states, both checked rather than assumed (see make_bptt_arm).
    qs = dest / "configs" / "config_queryset.py"
    qtext = qs.read_text()
    marker = "model_name = ModelPathManager.get_model_name_from_path(__file__)"
    pinned = re.search(r'^model_name = "([a-z_]+)"', qtext, re.M)
    if marker in qtext:
        qs.write_text(
            qtext.replace(
                marker,
                "# PINNED by make_noise_arm.py: the queryset identity is the FLOOR's, because both\n"
                "# arms pull identical data. Deriving it per arm would change the C-61 digest.\n"
                f'model_name = "{_FLOOR.name}"',
                1,
            )
        )
    elif pinned:
        floor_pin = re.search(
            r'^model_name = "([a-z_]+)"',
            (_FLOOR / "configs" / "config_queryset.py").read_text(),
            re.M,
        )
        if not floor_pin or floor_pin.group(1) != pinned.group(1):
            raise SystemExit(
                f"make_noise_arm: arm queryset pins {pinned.group(1)!r} but the floor pins "
                f"{floor_pin and floor_pin.group(1)!r} — the arms would read different caches"
            )
    else:
        raise SystemExit("make_noise_arm: config_queryset.py neither derives nor pins model_name")

    meta = dest / "configs" / "config_meta.py"
    mtext = meta.read_text().replace(f'"name": "{_FLOOR.name}"', f'"name": "{label}"')
    mtext = re.sub(
        r'"diagnostic_visualizations":\s*True', '"diagnostic_visualizations": False', mtext
    )
    meta.write_text(mtext)

    _verify(dest, dropout=dropout, lessons=lessons)
    return dest


def _verify(dest: Path, *, dropout: float | None, lessons: int | None = None) -> None:
    floor = _resolve(_FLOOR / "configs" / "config_hyperparameters.py", "get_hp_config")
    arm = _resolve(dest / "configs" / "config_hyperparameters.py", "get_hp_config")

    intended = {"input_noise_dropout"} if dropout is not None else set()
    intended |= {"total_lessons"} if lessons is not None else set()
    diff = {k for k in set(floor) | set(arm) if floor.get(k) != arm.get(k)}
    expected = {k for k in intended if arm.get(k) != floor.get(k)}
    if diff != expected:
        raise SystemExit(
            f"make_noise_arm: config differs from the floor in {sorted(diff)}, expected exactly "
            f"{sorted(expected)}. An unintended key changed — the arm is NOT built."
        )

    assert arm.get("input_noise_dropout") == dropout, arm.get("input_noise_dropout")
    assert arm["ss_epsilon_max"] == 0.0, "both arms run eps=0; scheduled sampling is measured harmful"
    pinned = (
        "torch_seed",
        "np_seed",
        "time_steps",
        "output_distribution",
        "forecast_composition",
        "window_dim",
        "n_posterior_samples",
        "n_head_samples",
        "pushforward_weight",
    ) + (() if lessons is not None else ("total_lessons",))
    for k in pinned:
        assert arm[k] == floor[k], f"{k} drifted from the floor: {arm[k]} vs {floor[k]}"
    if list(arm["features"]) != list(arm["regression_targets"]):
        raise SystemExit("make_noise_arm: C-260 — features must equal regression_targets in order")
    m = _resolve(dest / "configs" / "config_meta.py", "get_meta_config")
    if m.get("diagnostic_visualizations") is not False:
        raise SystemExit("make_noise_arm: diagnostic_visualizations must be False (~6 h/emit)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=sorted(_ARMS))
    ap.add_argument("--label")
    ap.add_argument("--lessons", type=int, help="smoke arms only; otherwise the floor's value")
    a = ap.parse_args()
    if a.label and not re.fullmatch(r"[a-z]+_[a-z]+", a.label):
        raise SystemExit(f"make_noise_arm: --label {a.label!r} must match ^[a-z]+_[a-z]+$")
    print(f"built {build(a.which, label=a.label, lessons=a.lessons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
