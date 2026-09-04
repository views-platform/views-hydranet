#!/usr/bin/env python3
"""make_bptt_arm.py — build the two BPTT-SA screen arms as their own model directories.

Adapted from `2026-08-17_ss_retention_dossier/tools/make_ss_arm.py`, which is the established
arm-as-directory builder in this repo. Two changes from it:

* the **floor is `fullzero_fortytwo`**, not `violet_visitor` — the L=300 vehicle every result in the
  silence-vs-fade and drivers dossiers was measured on;
* `ss_backprop_through_feedback` is a **new** key, so it is INSERTED rather than replaced.

Everything that made the original trustworthy is kept, because each guard is there for a recorded
failure: the floor's config is never written; verification execs both configs and asserts the
**symmetric difference of the resolved dicts is exactly the intended key set**, so an unintended
change anywhere fails loud; the queryset identity is pinned to the floor's so both arms share one
cached parquet and cannot diverge by data vintage; and `diagnostic_visualizations` is forced off.

The two arms differ in **exactly one boolean**. That is the whole experiment (#308).
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

#: Keys this tool may change. Anything else differing between floor and arm is a defect.
_INTENDED = {"ss_epsilon_max", "ss_feedback", "ss_backprop_through_feedback"}

#: ModelPathManager enforces ^[a-z]+_[a-z]+$ — two lowercase words, no digits.
_ARMS = {
    "detached": ("ssdetached_fortytwo", False),  # plain scheduled sampling: the wire is cut
    "attached": ("ssattached_fortytwo", True),  # BPTT-SA: the wire is connected
}
_EPS = 0.5


def _set_key(text: str, key: str, literal: str) -> str:
    pat = re.compile(rf"('{re.escape(key)}':\s*)[^\n]*?,")
    new, n = pat.subn(rf"\g<1>{literal},", text, count=1)
    if n != 1:
        raise SystemExit(f"make_bptt_arm: key {key!r} not found exactly once")
    return new


def _insert_after(text: str, anchor_key: str, line: str) -> str:
    """Insert a NEW key immediately after an existing one, so ordering stays readable."""
    pat = re.compile(rf"('{re.escape(anchor_key)}':\s*[^\n]*?,)")
    new, n = pat.subn(rf"\g<1>\n    {line}", text, count=1)
    if n != 1:
        raise SystemExit(f"make_bptt_arm: anchor {anchor_key!r} not found exactly once")
    return new


def _resolve(path: Path, fn: str) -> dict:
    text = path.read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns[fn]()


def build(
    which: str,
    *,
    out_root: Path = _MODELS,
    label: str | None = None,
    traj: str | None = None,
) -> Path:
    """Build one arm.

    `label` renames the destination directory, and `traj` turns on the engine's opt-in
    per-lesson trajectory CSV (`trajectory_log_path`). Both exist for the GRAD-TRAJ probe, which
    needs a THROWAWAY clone of an arm: re-running in the original directory would either destroy
    `ssdetached_fortytwo`'s artifact or leave its config no longer describing the run that
    produced it. A clone costs 9.5 MB of `data/raw` and keeps provenance exact.

    `trajectory_log_path` is observational — the engine registers read-only forward hooks and
    writes a CSV. It does not touch the forward math, the RNG, or the optimiser, so a clone with
    `traj` set trains the same trajectory as the arm it clones.
    """
    default_label, backprop = _ARMS[which]
    label = label or default_label
    dest = out_root / label
    if dest.exists():
        raise SystemExit(f"make_bptt_arm: {dest} exists — refusing to overwrite an arm")

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
        raise SystemExit(f"make_bptt_arm: {src_raw} missing — an arm would fetch data mid-run")
    shutil.copytree(src_raw, dest / "data" / "raw")

    hp = dest / "configs" / "config_hyperparameters.py"
    text = hp.read_text()
    text = _set_key(text, "ss_epsilon_max", repr(_EPS))
    text = _set_key(text, "ss_feedback", repr("sample"))  # C-259
    text = _insert_after(
        text,
        "ss_feedback",
        f"'ss_backprop_through_feedback': {backprop!r},  # #308 BPTT-SA: the ONE variable",
    )
    if traj:
        text = _insert_after(
            text,
            "ss_backprop_through_feedback",
            f"'trajectory_log_path': {traj!r},  # GRAD-TRAJ probe: observational, per-lesson CSV",
        )
    hp.write_text(text)

    # Queryset identity. config_queryset derives model_name from its own path and feeds the C-61
    # provenance digest, so a clone would change the DATA identity and reject the shared cache as
    # stale. But `fullzero_fortytwo` is ITSELF a clone and already carries a hardcoded pin to the
    # original floor (`violet_visitor`) whose cache every arm in this family shares. So there are
    # two legal states, and both are checked rather than assumed:
    #   - a derive-marker  -> pin it, as the original builder does
    #   - an inherited pin -> LEAVE IT. Rewriting it would point this arm at a different cache
    #                        than the floor it is being compared against.
    qs = dest / "configs" / "config_queryset.py"
    qtext = qs.read_text()
    marker = "model_name = ModelPathManager.get_model_name_from_path(__file__)"
    pinned = re.search(r'^model_name = "([a-z_]+)"', qtext, re.M)
    if marker in qtext:
        qs.write_text(
            qtext.replace(
                marker,
                "# PINNED by make_bptt_arm.py: the queryset identity is the FLOOR's, because both arms\n"
                "# pull identical data. Deriving it per arm would change the C-61 digest.\n"
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
                f"make_bptt_arm: arm queryset pins {pinned.group(1)!r} but the floor pins "
                f"{floor_pin and floor_pin.group(1)!r} — the arms would read different caches"
            )
    else:
        raise SystemExit("make_bptt_arm: config_queryset.py neither derives nor pins model_name")

    meta = dest / "configs" / "config_meta.py"
    mtext = meta.read_text().replace(f'"name": "{_FLOOR.name}"', f'"name": "{label}"')
    mtext = re.sub(
        r'"diagnostic_visualizations":\s*True', '"diagnostic_visualizations": False', mtext
    )
    meta.write_text(mtext)

    _verify(dest, backprop=backprop, traj=traj)
    return dest


def _verify(dest: Path, *, backprop: bool, traj: str | None = None) -> None:
    floor = _resolve(_FLOOR / "configs" / "config_hyperparameters.py", "get_hp_config")
    arm = _resolve(dest / "configs" / "config_hyperparameters.py", "get_hp_config")

    intended = _INTENDED | ({"trajectory_log_path"} if traj else set())
    diff = {k for k in set(floor) | set(arm) if floor.get(k) != arm.get(k)}
    expected = {k for k in intended if arm.get(k) != floor.get(k)}
    if diff != expected:
        raise SystemExit(
            f"make_bptt_arm: config differs from the floor in {sorted(diff)}, expected exactly "
            f"{sorted(expected)}. An unintended key changed — the arm is NOT built."
        )

    assert arm["ss_epsilon_max"] == _EPS, arm["ss_epsilon_max"]
    assert arm["ss_backprop_through_feedback"] is backprop
    assert arm.get("trajectory_log_path") == traj, arm.get("trajectory_log_path")
    # the screen compares ONE boolean; everything that could confound it is asserted equal
    for k in (
        "torch_seed",
        "np_seed",
        "total_lessons",
        "output_distribution",
        "forecast_composition",
        "window_dim",
        "n_posterior_samples",
        "n_head_samples",
    ):
        assert arm[k] == floor[k], f"{k} drifted from the floor: {arm[k]} vs {floor[k]}"
    if arm["ss_epsilon_max"] > 0 and arm.get("ss_feedback") != "sample":
        raise SystemExit("make_bptt_arm: C-259 — ss_feedback must be 'sample' when eps > 0")
    if list(arm["features"]) != list(arm["regression_targets"]):
        raise SystemExit("make_bptt_arm: C-260 — features must equal regression_targets in order")
    m = _resolve(dest / "configs" / "config_meta.py", "get_meta_config")
    if m.get("diagnostic_visualizations") is not False:
        raise SystemExit("make_bptt_arm: diagnostic_visualizations must be False (~6 h/emit)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=sorted(_ARMS), help="detached = plain SS; attached = BPTT-SA")
    ap.add_argument("--label", help="destination directory name (default: the arm's own name)")
    ap.add_argument("--traj", help="path for the engine's opt-in per-lesson trajectory CSV")
    a = ap.parse_args()
    if a.label and not re.fullmatch(r"[a-z]+_[a-z]+", a.label):
        # ModelPathManager enforces this and fails much later, after the copytree.
        raise SystemExit(f"make_bptt_arm: --label {a.label!r} must match ^[a-z]+_[a-z]+$")
    dest = build(a.which, label=a.label, traj=a.traj)
    print(f"built {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
