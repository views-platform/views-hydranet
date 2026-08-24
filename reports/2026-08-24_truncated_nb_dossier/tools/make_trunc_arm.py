#!/usr/bin/env python3
"""make_trunc_arm.py — build one `truncated_nb` arm as its own model directory.

Descends from `2026-08-23_itf_pilot_dossier/tools/make_itf_arm.py`, itself from
`2026-08-17_ss_retention_dossier/tools/make_ss_arm.py`. Pre-registration:
`reports/2026-08-24_truncated_nb_dossier/05_analysis_plan.md`, LOCKED before this file existed.

**The one key this adds:** ``output_distribution = 'truncated_nb'`` — the zero-truncated NB body,
so the classification gate becomes the ONLY zero source under ``soft_gate`` (#258, M44).

**What is deliberately NOT changed: ``body_supervision``.** ``TruncatedNBFamily.nll`` folds its own
``y>0`` mask in, so ``'active'`` is numerically identical to ``'all'`` at ``event_threshold == 0``.
`truncated_smoke` changed it as well as the family and the lesson count — three axes at once —
which is why nothing in that record attributes anything to the family. This arm changes ONE key.

Verification is stronger than the tool it descends from
-------------------------------------------------------
The ancestors diff the arm against the FLOOR (`violet_visitor`). That leaves the actual
experimental contrast unverified, because the control is `fullzero_<seed>`, not the floor.
`_verify` here does BOTH, and requires the symmetric difference against the control to be exactly
``{'output_distribution'}`` — the contrast every claim in this dossier rests on.

It also keeps the ancestor's "prove the mechanism, do not trust the flag" discipline:
`make_itf_arm` built the real `ScheduledSamplingMixer` and checked its endpoints; this resolves the
real family and asserts the two structural properties that make the arm meaningful —
``prob_positive == 1`` and no zero in 5000 draws from the ``mu -> 0`` background. A key that
reached the config and did nothing would cost ~8 GPU-hours across four arms.

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
    "output_distribution",
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


def arm_label(*, lessons: int, eps: float, seed: int) -> str:
    """The pipeline-legal directory name for an arm, e.g. (40, 0.5, 43) -> ``shorthalf_fortythree``."""
    for table, key, what in (
        (_LESSON_WORD, lessons, "lessons"),
        (_EPS_WORD, eps, "eps"),
        (_SEED_WORD, seed, "seed"),
    ):
        if key not in table:
            raise SystemExit(
                f"make_ss_arm: no word for {what}={key}. Add one to the table rather than "
                "improvising a name — the mapping must stay deterministic in both directions."
            )
    # The pipeline requires `adjective_noun` — EXACTLY two lowercase parts
    # (`views_pipeline_core/data/model_path.py:validate_model_name`, `^[a-z]+_[a-z]+$`). So the
    # marker prefixes the ADJECTIVE, never the label: a three-part name is rejected at pipeline
    # startup, which is what killed the ITF pilot's first launch AFTER the queue had accepted it.
    #
    # The prefix is required for a second reason: `{lessons}{eps}_{seed}` is already owned by the
    # lesson-curve sweep. Without it, L=300/eps=0.0/seed=42 would be named `fullzero_fortytwo` —
    # which is this arm's own CONTROL. Same lessons, same dose, different body: the one collision
    # guaranteed to be both catastrophic and invisible.
    return f"trunc{_LESSON_WORD[lessons]}{_EPS_WORD[eps]}_{_SEED_WORD[seed]}"


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


def build(*, lessons: int, eps: float, seed: int, out_root: Path = _MODELS) -> Path:
    label = arm_label(lessons=lessons, eps=eps, seed=seed)
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
    # carries `'output_distribution': 'nb'`, so no insert path is needed here.
    text = _set_key(text, "output_distribution", repr("truncated_nb"))
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

    _verify(dest, lessons=lessons, eps=eps, seed=seed)
    return dest


def _verify(dest: Path, *, lessons: int, eps: float, seed: int) -> None:
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
    if arm.get("output_distribution") != "truncated_nb":
        raise SystemExit(
            f"make_trunc_arm: output_distribution is {arm.get('output_distribution')!r}, must be "
            "'truncated_nb' — this would be an ordinary nb arm, i.e. the control."
        )
    if arm.get("body_supervision") != floor.get("body_supervision"):
        raise SystemExit(
            "make_trunc_arm: body_supervision changed. It must NOT: the family folds its own y>0 "
            "mask into nll, so 'active' is numerically identical at event_threshold==0, and "
            "changing it alongside the family is exactly the confound that made truncated_smoke "
            "unreadable (postmortem_floor_limited_vehicle.md)."
        )
    if arm.get("forecast_composition") != "soft_gate":
        raise SystemExit(
            f"make_trunc_arm: forecast_composition is {arm.get('forecast_composition')!r}; the "
            "premise is that the GATE is the only zero source, which requires soft_gate."
        )

    # ── the contrast this dossier actually claims ────────────────────────────────────────────
    # The ancestors diff only against the FLOOR. But every claim here is treatment-vs-CONTROL,
    # and the control is `fullzero_<seed>`. Verify THAT symmetric difference is exactly the one
    # key, or the single-variable claim is unproven no matter what the floor diff says.
    control_dir = _MODELS / f"fullzero_{_SEED_WORD[seed]}"
    control_hp = control_dir / "configs" / "config_hyperparameters.py"
    if not control_hp.is_file():
        raise SystemExit(
            f"make_trunc_arm: control {control_dir} not found — the arm has nothing to be "
            "compared against and the 4v4 design is not satisfiable."
        )
    control = _resolve(control_hp, "get_hp_config")
    vs_control = {k for k in set(control) | set(arm) if control.get(k) != arm.get(k)}
    if vs_control != {"output_distribution"}:
        raise SystemExit(
            f"make_trunc_arm: arm differs from its control {control_dir.name} in "
            f"{sorted(vs_control)}, expected exactly ['output_distribution']. The experiment is "
            "NOT single-variable — the arm is NOT built."
        )

    # ── prove the mechanism, do not trust the flag ───────────────────────────────────────────
    # The ancestor built the real ScheduledSamplingMixer and checked its endpoints. The analogue
    # here: resolve the real family and assert the two structural properties that make this arm
    # mean anything. A key that reached the config and did nothing costs ~2 GPU-hours per arm.
    import torch

    from views_hydranet.distributions import resolve_family

    fam = resolve_family(arm["output_distribution"])
    if fam is None or fam.self_zeroed:
        raise SystemExit(
            f"make_trunc_arm: {arm['output_distribution']!r} did not resolve to a non-self-zeroed "
            "family; under soft_gate a self-zeroed body would double-count zeros again."
        )
    # the mu -> 0 background, where a plain nb draws 0 almost always
    _params = torch.stack([torch.full((5000,), 1e-4), torch.full((5000,), 0.5)], dim=-1)
    _draws = fam.sample(_params, 1, torch.Generator().manual_seed(3))
    if int((_draws == 0).sum()) != 0:
        raise SystemExit(
            "make_trunc_arm: the resolved body drew a ZERO on the mu->0 background. The gate is "
            "then not the only zero source and this arm does not test what it claims."
        )
    if not bool(torch.equal(fam.prob_positive(_params), torch.ones(5000))):
        raise SystemExit("make_trunc_arm: prob_positive is not identically 1 — wrong family.")
    if arm["ss_epsilon_max"] > 0 and arm.get("ss_feedback") != "sample":
        raise SystemExit(
            "make_ss_arm: C-259 — ss_feedback must be 'sample' when ss_epsilon_max > 0"
        )
    if list(arm["features"]) != list(arm["regression_targets"]):
        raise SystemExit("make_ss_arm: C-260 — features must equal regression_targets in order")

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
    a = p.parse_args()
    dest = build(lessons=a.lessons, eps=a.eps, seed=a.seed)
    print(f"built {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
