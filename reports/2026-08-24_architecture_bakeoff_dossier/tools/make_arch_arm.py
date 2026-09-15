#!/usr/bin/env python3
"""make_arch_arm.py — build one ARCHITECTURE bake-off arm (dossier 2026-08-24).

Descends from `make_trunc_arm.py` -> `make_itf_arm.py` -> `make_ss_arm.py`. Pre-registration:
`reports/2026-08-24_architecture_bakeoff_dossier/05_analysis_plan.md`.

**The one key this varies:** ``model`` — the architecture, resolved through
`views_hydranet.architectures.registry`. Everything else is held identical to the `fullzero_*`
controls, so an arm differs from its own-seed control in exactly ONE config key.

**Why `--variant`.** Six architectures share one builder, so the architecture is the queue's
optional 5th spec field and arrives here as ``--variant``. `arm_label` folds a short tag onto the
ADJECTIVE (`aafullzero_fortytwo`), never as a third part: `validate_model_name` is
`^[a-z]+_[a-z]+$` and a three-part name is rejected at pipeline startup — which is what
killed the ITF pilot's first launch after the queue had already accepted it.

**`arm_identity` is declared here**, so `run_queue.sh` checks `model` on the resume path.
Without it a resumed queue could reuse an arm built on a different architecture and score it as
this one — the hole that forced the truncated-nb programme to patch its own verifier instead.

--- original docstring follows ---

make_ss_arm.py — build one sweep arm as its own model directory.

**Arm-as-directory, not config mutation.** Every training sweep in this repo so far has overwritten
the live config and relied on an ``md5`` floor plus ``trap … EXIT`` to put it back. That works,
but it writes the tracked config of a real roster model during an experiment. Cloning instead:

* ``violet_visitor``'s config is **never written** — its md5 becomes a read-only invariant,
  which is strictly stronger than restore-after-write;
* the arm → artifact mapping is **structural** (the ``.pt`` lives inside the arm's own directory),
  which kills the class of error that produced 2026-08-14's ε=0.5 log/manifest disagreement;
* each arm gets its own wandb run name, killing the silent-overwrite class fixed in ``bde79c0``;
* logs, cubes and `ERROR.log` are isolated per arm.

The clone is ~9.6 MB excluding `data/generated`, `logs`, `wandb` and `artifacts`.

Verification is stronger than the tool it descends from
-------------------------------------------------------
``make_arm_config.py`` (2026-07-02 dossier) verifies by ``ast.parse`` → ``exec`` → assert the keys
it *set*. That cannot catch a regex which also matched something else. This module execs **both**
the floor and the arm config and asserts the **symmetric difference of the two resolved dicts is
exactly the intended key set** — so an unintended change anywhere in the file fails loud, not just
one to a key we happened to think of.

It also asserts, before any GPU work:

* ``diagnostic_visualizations is False`` — violet's is **True**, and the per-origin biopsy costs
  ~28 min/origin, i.e. **~6 h per emit**: the largest silent cost landmine in the sweep.
* ``ss_feedback == 'sample'`` whenever ``ss_epsilon_max > 0`` (C-259).
* ``features == regression_targets`` in order (C-260 rejects otherwise; fail before the GPU spins).
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

_MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
_FLOOR = _MODELS / "violet_visitor"

#: Keys this tool may change in config_hyperparameters.py. Anything else differing between floor
#: and arm is a defect, not a feature.
_INTENDED = {
    "ss_epsilon_max",
    "ss_feedback",
    "torch_seed",
    "np_seed",
    "total_lessons",
    # this dossier's one key — see the module docstring
    "model",
}

#: ``ModelPathManager.validate_model_name`` enforces ``^[a-z]+_[a-z]+$`` — exactly two lowercase
#: words, no digits, one underscore. So the arm's parameters cannot appear literally in the directory
#: name; they are encoded through these tables instead, and ``arm.json`` carries the authoritative
#: values. Deterministic both ways, so a name is still a reliable handle in a log or the process table.
_LESSON_WORD = {
    2: "tiny",
    40: "short",
    160: "long",
    300: "full",
    600: "sixhundred",
    900: "ninehundred",
}
#: 2 = the smoke rung. The ensemble dossier's harness checklist makes a 2-lesson smoke the
#: precondition for a long run; it exercises every seam in ~20 min instead of hours.
#: 600/900 break the short/long/full style deliberately: nobody will remember an invented
#: adjective for 900, and an unambiguous name beats a pretty one (lesson-curve dossier,
#: reports/2026-08-18_lesson_curve_dossier).
_EPS_WORD = {0.0: "zero", 0.1: "tenth", 0.25: "quarter", 0.5: "half"}
_SEED_WORD = {
    42: "fortytwo",
    43: "fortythree",
    44: "fortyfour",
    45: "fortyfive",
    46: "fortysix",
    47: "fortyseven",
}


#: architecture -> the short tag folded onto the adjective. Explicit table, not a slugifier: the
#: label must be deterministic in both directions and legal as `^[a-z]+_[a-z]+$`.
_ARCH_TAG = {
    "AntiAliasedPool": "aa",
    "DynamicTopSkip": "dyn",
    "FiLMSkip": "film",
    "ShallowPool": "shal",
    "DualStream": "dual",
    "WideMemory": "wide",
}


def arm_label(*, lessons: int, eps: float, seed: int, variant: str | None = None) -> str:
    """Pipeline-legal directory name: (300, 0.0, 42, AntiAliasedPool) -> `aafullzero_fortytwo`."""
    if variant not in _ARCH_TAG:
        raise SystemExit(
            f"make_arch_arm: --variant must be one of {sorted(_ARCH_TAG)}; got {variant!r}. "
            "Add a tag to the table rather than improvising a name."
        )
    for table, key, what in (
        (_LESSON_WORD, lessons, "lessons"),
        (_EPS_WORD, eps, "eps"),
        (_SEED_WORD, seed, "seed"),
    ):
        if key not in table:
            raise SystemExit(
                f"make_arch_arm: no word for {what}={key}. Add one to the table rather than "
                "improvising a name — the mapping must stay deterministic in both directions."
            )
    # The tag prefixes the ADJECTIVE. The pipeline requires EXACTLY two lowercase parts
    # (`validate_model_name`, `^[a-z]+_[a-z]+$`); a three-part name is rejected at startup, after
    # the queue has already accepted the arm — the ITF pilot's first launch died exactly there.
    return f"{_ARCH_TAG[variant]}{_LESSON_WORD[lessons]}{_EPS_WORD[eps]}_{_SEED_WORD[seed]}"


def arm_identity(*, lessons: int, eps: float, seed: int, variant: str | None = None) -> dict:
    """The config keys that constitute this arm's identity, for `run_queue.sh`'s reuse check.

    `model` is the point: without it a resumed queue could reuse an arm built on a different
    architecture and score it as this one. The rest reproduce the legacy tuple so nothing is lost.
    """
    if variant not in _ARCH_TAG:
        raise SystemExit(f"make_arch_arm: unknown variant {variant!r}")
    return {
        "model": variant,
        "total_lessons": lessons,
        "torch_seed": seed,
        "np_seed": seed,
        "ss_epsilon_max": eps,
    }


def _set_key(text: str, key: str, literal: str) -> str:
    """Replace ``'key': <value>,`` preserving any trailing comment. Raises if absent."""
    pat = re.compile(rf"('{re.escape(key)}':\s*)[^\n]*?,")
    new, n = pat.subn(rf"\g<1>{literal},", text, count=1)
    if n != 1:
        raise SystemExit(f"make_ss_arm: key {key!r} not found exactly once")
    return new


def _resolve(path: Path, fn: str) -> dict:
    """exec a config file and call its getter — the readback that makes verification real."""
    text = path.read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns[fn]()


def build(
    *, lessons: int, eps: float, seed: int, variant: str, out_root: Path = _MODELS
) -> Path:
    label = arm_label(lessons=lessons, eps=eps, seed=seed, variant=variant)
    dest = out_root / label
    if dest.exists():
        raise SystemExit(f"make_ss_arm: {dest} already exists — refusing to overwrite an arm")

    shutil.copytree(
        _FLOOR,
        dest,
        ignore=shutil.ignore_patterns(
            "data", "logs", "wandb", "artifacts", "reports", "notebooks", "__pycache__", "*.pyc"
        ),
    )
    # Mirror the floor's directory skeleton. ModelPathManager validates that each of these exists
    # and raises FileNotFoundError otherwise — so an arm missing `data/processed` dies at startup,
    # which is exactly what happened on the first attempt at 15:15.
    for sub in ("data/generated", "data/processed", "artifacts", "logs", "notebooks", "reports"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    # data/raw carries the cached parquet, so no arm fetches from the network mid-sweep.
    src_raw = _FLOOR / "data" / "raw"
    if not src_raw.is_dir():
        raise SystemExit(f"make_ss_arm: {src_raw} missing — an arm would fetch data mid-sweep")
    shutil.copytree(src_raw, dest / "data" / "raw")

    hp = dest / "configs" / "config_hyperparameters.py"
    text = hp.read_text()
    text = _set_key(text, "ss_epsilon_max", repr(eps))
    text = _set_key(text, "torch_seed", repr(seed))
    text = _set_key(text, "np_seed", repr(seed))
    text = _set_key(text, "total_lessons", repr(lessons))
    # THE one key. `_set_key` raises unless it matches exactly once, and the floor config already
    # carries `'model': 'HydraBNUNet06_LSTM4'`, so no insert path is needed here.
    text = _set_key(text, "model", repr(variant))
    if "'ss_feedback'" in text:
        text = _set_key(text, "ss_feedback", repr("sample"))
    else:
        text = text.replace(
            f"'ss_epsilon_max': {eps!r},",
            f"'ss_epsilon_max': {eps!r},\n    'ss_feedback': 'sample',  # C-259",
            1,
        )
    hp.write_text(text)

    # PIN THE QUERYSET IDENTITY TO THE FLOOR'S.
    #
    # config_queryset.py derives `model_name` from its own file path and uses it as the queryset
    # "name", which feeds the C-61 provenance digest. Cloning therefore changes the DATA identity even
    # though the data is byte-identical, and the cached parquet is rejected as stale.
    #
    # The arms genuinely share a queryset — they differ only in training hyperparameters, never in what
    # data is pulled — so pinning the name to the floor's is the honest statement, not a workaround. It
    # also makes "every arm sees identical data" structural rather than merely intended, which is what
    # a controlled sweep needs: a per-arm refetch could return a different vintage and confound the
    # comparison with a data change.
    qs = dest / "configs" / "config_queryset.py"
    qtext = qs.read_text()
    marker = "model_name = ModelPathManager.get_model_name_from_path(__file__)"
    if marker not in qtext:
        raise SystemExit(
            "make_ss_arm: config_queryset.py no longer derives model_name as expected"
        )
    qs.write_text(
        qtext.replace(
            marker,
            "# PINNED by make_ss_arm.py: the queryset identity is the FLOOR's, because every sweep arm\n"
            "# pulls the same data. Deriving it from the path would change the C-61 provenance digest\n"
            "# per arm and reject the shared cache as stale.\n"
            f'model_name = "{_FLOOR.name}"',
            1,
        )
    )

    meta = dest / "configs" / "config_meta.py"
    mtext = meta.read_text().replace('"name": "violet_visitor"', f'"name": "{label}"')
    mtext = re.sub(
        r'"diagnostic_visualizations":\s*True', '"diagnostic_visualizations": False', mtext
    )
    meta.write_text(mtext)

    # A failed verification must not leave a half-built arm behind: `build` refuses to overwrite an
    # existing directory, so the wreckage of a rejected build would block every later attempt and
    # look like "the builder is broken" rather than "the config was wrong".
    try:
        _verify(dest, lessons=lessons, eps=eps, seed=seed, variant=variant)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest


def _verify(dest: Path, *, lessons: int, eps: float, seed: int, variant: str) -> None:
    floor = _resolve(_FLOOR / "configs" / "config_hyperparameters.py", "get_hp_config")
    arm = _resolve(dest / "configs" / "config_hyperparameters.py", "get_hp_config")

    diff = {k for k in set(floor) | set(arm) if floor.get(k) != arm.get(k)}
    expected = {k for k in _INTENDED if arm.get(k) != floor.get(k)}
    if diff != expected:
        raise SystemExit(
            f"make_trunc_arm: config differs from the floor in {sorted(diff)}, expected exactly "
            f"{sorted(expected)}. An unintended key changed — the arm is NOT built."
        )

    assert arm["ss_epsilon_max"] == eps, arm["ss_epsilon_max"]
    assert arm["torch_seed"] == seed and arm["np_seed"] == seed
    assert arm["total_lessons"] == lessons
    if arm.get("model") != variant:
        raise SystemExit(
            f"make_arch_arm: model is {arm.get('model')!r}, expected {variant!r} — this arm would "
            "train a different architecture than the one requested."
        )
    if arm.get("output_distribution") != floor.get("output_distribution"):
        raise SystemExit(
            "make_arch_arm: output_distribution changed. Only `model` may differ — this program "
            "varies the ARCHITECTURE, and M45 is what a two-key experiment costs."
        )

    # ── the contrast this dossier actually claims ────────────────────────────────────────────
    # The ancestors diff only against the FLOOR. But every claim here is treatment-vs-CONTROL,
    # and the control is `fullzero_<seed>`. Verify THAT symmetric difference is exactly the one
    # key, or the single-variable claim is unproven no matter what the floor diff says.
    control_dir = _MODELS / f"fullzero_{_SEED_WORD[seed]}"
    control_hp = control_dir / "configs" / "config_hyperparameters.py"
    if control_hp.is_file():
        control = _resolve(control_hp, "get_hp_config")
        vs_control = {k for k in set(control) | set(arm) if control.get(k) != arm.get(k)}
        # A SMOKE arm deliberately differs in total_lessons too — it is never a scored result, so
        # the single-variable invariant applies to the SCORED arms only. Stated rather than
        # silently tolerated, and the smoke's own lesson count is asserted above.
        same_len = lessons == control.get("total_lessons")
        allowed = {"model"} if same_len else {"model", "total_lessons"}
        if vs_control != allowed:
            raise SystemExit(
                f"make_arch_arm: arm differs from its control {control_dir.name} in "
                f"{sorted(vs_control)}, expected exactly {sorted(allowed)}. The experiment is "
                "NOT single-variable — the arm is NOT built."
            )
    elif lessons == 300:
        raise SystemExit(
            f"make_arch_arm: control {control_dir} not found — a scored arm has nothing to be "
            "compared against."
        )

    # ── prove the mechanism, do not trust the flag ───────────────────────────────────────────
    # The ancestor built the real ScheduledSamplingMixer and checked its endpoints. The analogue
    # here: resolve the architecture through the SAME registry `choose_model` uses and instantiate
    # it at the arm's real widths. A `model` string that reaches the config but names something the
    # registry cannot build would fail ~2 minutes into a 2.4-hour arm, twelve times over.
    import torch

    from views_hydranet.architectures.registry import get_architecture

    cls = get_architecture(arm["model"])
    net = cls(
        arm["input_channels"],
        arm["total_hidden_channels"],
        arm["output_channels"],
        arm["dropout_rate"],
        output_distribution=arm["output_distribution"],
        n_static_channels=len(arm.get("static_channels", [])),
    )
    if net.base % 8:
        raise SystemExit(
            f"make_arch_arm: {arm['model']} has state width {net.base}, not divisible by 8; "
            "blend_recurrent_state would silently mis-assign short-term vs long-term memory."
        )
    n_targets = len(arm["regression_targets"])
    with torch.no_grad():
        out = net(torch.zeros(1, arm["input_channels"], 16, 16), net.init_hTtime(net.base, 16, 16))
    if out.cls.shape[1] != n_targets:
        raise SystemExit(
            f"make_arch_arm: {arm['model']} emits {out.cls.shape[1]} gate channels, expected "
            f"{n_targets} — the emitted field would not match the targets."
        )

    m = _resolve(dest / "configs" / "config_meta.py", "get_meta_config")
    if m.get("diagnostic_visualizations") is not False:
        raise SystemExit(
            "make_ss_arm: diagnostic_visualizations must be False — the per-origin biopsy costs "
            "~28 min/origin, i.e. ~6 h per emit"
        )
    if m.get("name") != dest.name:
        raise SystemExit(f"make_ss_arm: meta name {m.get('name')!r} != dir {dest.name!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lessons", type=int, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", required=True, help=f"architecture: one of {sorted(_ARCH_TAG)}")
    a = p.parse_args()
    dest = build(lessons=a.lessons, eps=a.eps, seed=a.seed, variant=a.variant)
    print(f"built {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
