#!/usr/bin/env python3
"""make_pf_arm.py — build one PUSHFORWARD arm (dossier 2026-08-26).

Descends from `make_arch_arm.py` -> `make_trunc_arm.py` -> `make_itf_arm.py` -> `make_ss_arm.py`,
and keeps their verification discipline verbatim: exec both configs, assert the symmetric
difference is exactly the intended key set, and diff against the CONTROL as well as the floor.

**The keys this varies:** ``pushforward_weight`` and ``pushforward_detach_state`` (#289,
Brandstetter et al. 2022). Everything else is held identical to the `fullzero_*` controls.

Two things differ from every ancestor and are the reason this is a separate builder rather than a
flag on `make_arch_arm.py`:

* **The keys are NEW.** `violet_visitor`'s config predates them, so `_set_key` — which requires
  exactly one existing match — cannot be used. There is an explicit insert path, and `_verify`
  reads the value back out of the resolved dict so an insert that landed in a comment or a
  docstring fails loud.
* **`pushforward_weight > 0` is only honoured on a family head.** `_process_sequence` guards the
  term on `family is not None`, so a non-family arm would train with no pushforward and report a
  clean run. `HydraNetConfig.reject_pushforward_without_a_family` now refuses that combination at
  config load, and `_verify` checks it here too — before any GPU work, not after.

`arm_identity` declares both keys, so `run_queue.sh`'s reuse check cannot hand back an arm built at
a different weight and score it as this one.

--- ancestor docstring follows, unchanged: its reasoning about arm-as-directory, the queryset pin
    and the verification-by-readback still governs this builder ---

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
    # this dossier's keys — see the module docstring
    "pushforward_weight",
    "pushforward_detach_state",
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


#: pushforward setting -> the short tag folded onto the adjective. Explicit table, not a
#: slugifier: the label must be deterministic in BOTH directions and legal as `^[a-z]+_[a-z]+$`.
#: The variant string encodes weight and the state fork together, because they are one arm.
_PF_TAG = {
    #: The CONTROL. No tag, so the label is the plain `tinyzero_fortytwo` / `fullzero_fortytwo`
    #: the rest of the programme already uses. It writes `pushforward_weight: 0.0` explicitly
    #: rather than relying on the field default, so the control's config states its own condition.
    "0.0": "",
    #: The control-reuse GATE (05_analysis_plan §3). Identical to `0.0` in every config value; it
    #: exists only so the fresh build gets its own directory instead of colliding with the archived
    #: `fullzero_*` it must be compared against.
    "0.0-recheck": "re",
    #: The weight D1 actually tests. 0.1 damaged the base model (oracle -0.03 at every horizon,
    #: F5/F6 both fired) so its arm is VOID; 0.01 is the last weight this programme will try.
    "0.01": "pflow",
    "0.1": "pf",  # weight 0.1 — VOID, kept so the arm can still be rebuilt for the dose record
    "0.1-detach": "pfd",  # weight 0.1, state detached (Brandstetter's stateless reading)
    "0.5": "pfhalf",
}

#: variant -> (pushforward_weight, pushforward_detach_state)
_PF_SPEC = {
    "0.0": (0.0, False),
    "0.0-recheck": (0.0, False),
    "0.01": (0.01, False),
    "0.1": (0.1, False),
    "0.1-detach": (0.1, True),
    "0.5": (0.5, False),
}


def arm_label(*, lessons: int, eps: float, seed: int, variant: str | None = None) -> str:
    """Pipeline-legal directory name: (2, 0.0, 42, "0.1") -> `pftinyzero_fortytwo`."""
    if variant not in _PF_TAG:
        raise SystemExit(
            f"make_pf_arm: --variant must be one of {sorted(_PF_TAG)}; got {variant!r}. "
            "Add a tag to the table rather than improvising a name."
        )
    for table, key, what in (
        (_LESSON_WORD, lessons, "lessons"),
        (_EPS_WORD, eps, "eps"),
        (_SEED_WORD, seed, "seed"),
    ):
        if key not in table:
            raise SystemExit(
                f"make_pf_arm: no word for {what}={key}. Add one to the table rather than "
                "improvising a name — the mapping must stay deterministic in both directions."
            )
    # The tag prefixes the ADJECTIVE. The pipeline requires EXACTLY two lowercase parts
    # (`validate_model_name`, `^[a-z]+_[a-z]+$`); a three-part name is rejected at startup, after
    # the queue has already accepted the arm — the ITF pilot's first launch died exactly there.
    return f"{_PF_TAG[variant]}{_LESSON_WORD[lessons]}{_EPS_WORD[eps]}_{_SEED_WORD[seed]}"


def arm_identity(*, lessons: int, eps: float, seed: int, variant: str | None = None) -> dict:
    """The config keys that constitute this arm's identity, for `run_queue.sh`'s reuse check.

    Both pushforward keys are the point: without them a resumed queue could reuse an arm built at
    a different weight — or on the other side of the state fork — and score it as this one.
    """
    if variant not in _PF_SPEC:
        raise SystemExit(f"make_pf_arm: unknown variant {variant!r}")
    weight, detach = _PF_SPEC[variant]
    return {
        "pushforward_weight": weight,
        "pushforward_detach_state": detach,
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
        raise SystemExit(f"make_pf_arm: key {key!r} not found exactly once")
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
        raise SystemExit(f"make_pf_arm: {dest} already exists — refusing to overwrite an arm")

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
        raise SystemExit(f"make_pf_arm: {src_raw} missing — an arm would fetch data mid-sweep")
    shutil.copytree(src_raw, dest / "data" / "raw")

    hp = dest / "configs" / "config_hyperparameters.py"
    text = hp.read_text()
    text = _set_key(text, "ss_epsilon_max", repr(eps))
    text = _set_key(text, "torch_seed", repr(seed))
    text = _set_key(text, "np_seed", repr(seed))
    text = _set_key(text, "total_lessons", repr(lessons))
    # THE keys. Unlike every ancestor's, these are NEW: `violet_visitor`'s config predates #289,
    # so there is nothing for `_set_key` to match and an insert path is required. Anchored on
    # `'model'`, which the floor is guaranteed to carry (it is the arch bake-off's one key), so a
    # missing anchor fails here rather than silently producing an arm without the flag.
    weight, detach = _PF_SPEC[variant]
    anchor = "'model': "
    if anchor not in text:
        raise SystemExit(
            "make_pf_arm: no 'model' key to anchor the pushforward insert on — the floor config "
            "changed shape; fix the anchor rather than guessing a location."
        )
    for key, value in (("pushforward_weight", weight), ("pushforward_detach_state", detach)):
        if f"'{key}'" in text:
            text = _set_key(text, key, repr(value))
        else:
            line = text[text.index(anchor) :].split("\n", 1)[0]
            text = text.replace(line, f"{line}\n        '{key}': {value!r},  # #289", 1)
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
            "make_pf_arm: config_queryset.py no longer derives model_name as expected"
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
            f"make_pf_arm: config differs from the floor in {sorted(diff)}, expected exactly "
            f"{sorted(expected)}. An unintended key changed — the arm is NOT built."
        )

    assert arm["ss_epsilon_max"] == eps, arm["ss_epsilon_max"]
    assert arm["torch_seed"] == seed and arm["np_seed"] == seed
    assert arm["total_lessons"] == lessons
    weight, detach = _PF_SPEC[variant]
    # Read back out of the RESOLVED dict, not the file text: an insert that landed inside a
    # comment or a docstring would still be present as a string and must not pass.
    if arm.get("pushforward_weight") != weight:
        raise SystemExit(
            f"make_pf_arm: pushforward_weight is {arm.get('pushforward_weight')!r}, expected "
            f"{weight!r} — the insert did not take, and this arm would train WITHOUT the "
            "intervention while being scored as if it had it."
        )
    if arm.get("pushforward_detach_state") != detach:
        raise SystemExit(
            f"make_pf_arm: pushforward_detach_state is {arm.get('pushforward_detach_state')!r}, "
            f"expected {detach!r} — this arm is on the wrong side of the state fork."
        )
    # `_process_sequence` computes the term only when the head resolves to a family. A non-family
    # arm would train with no pushforward at all and report a clean run. HydraNetConfig now
    # rejects that too; checked here as well, because this is BEFORE the GPU spins up.
    from views_hydranet.distributions import resolve_family

    if weight > 0.0 and resolve_family(arm.get("output_distribution", "")) is None:
        raise SystemExit(
            f"make_pf_arm: output_distribution={arm.get('output_distribution')!r} is not a "
            "distribution family, so the pushforward term is never computed — this arm would "
            "look clean and train nothing extra."
        )
    if arm.get("model") != floor.get("model"):
        raise SystemExit(
            "make_pf_arm: model changed. Only the pushforward keys may differ — this program "
            "varies the LOSS, and M45 is what a two-key experiment costs."
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
        # Both pushforward keys count as the one variable: they are a single intervention with a
        # fork, not two. `pushforward_detach_state` shows up even at its default because the
        # control's config predates #289 and carries neither key at all — an honest diff, not noise.
        allowed = {"pushforward_weight", "pushforward_detach_state"}
        if not same_len:
            allowed = allowed | {"total_lessons"}
        if vs_control != allowed:
            raise SystemExit(
                f"make_pf_arm: arm differs from its control {control_dir.name} in "
                f"{sorted(vs_control)}, expected exactly {sorted(allowed)}. The experiment is "
                "NOT single-variable — the arm is NOT built."
            )
    elif lessons == 300:
        raise SystemExit(
            f"make_pf_arm: control {control_dir} not found — a scored arm has nothing to be "
            "compared against."
        )

    # ── prove the mechanism, do not trust the flag ───────────────────────────────────────────
    # The ancestor resolved the architecture through the registry, because a `model` string the
    # registry cannot build would die two minutes into a 2.4-hour arm. The analogue here: run the
    # REAL `_process_sequence` at this arm's weight and require the pushforward term to actually
    # change the loss. A weight that reaches the config but is silently ignored — a non-family
    # head, a guard that short-circuits, a key inserted into a comment — would otherwise produce a
    # full arm that is byte-identical to its control and be scored as a treatment.
    import torch
    import torch.nn as nn

    from views_hydranet.architectures.registry import get_architecture
    from views_hydranet.distributions import get_family
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.train.training_engine import _SequenceIndices, _process_sequence

    fam_name = arm.get("output_distribution")
    feats = list(arm["regression_targets"])
    cls = list(arm["classification_targets"])
    idx = _SequenceIndices(
        feats + cls,
        {
            "features": feats,
            "regression_targets": feats,
            "classification_targets": cls,
            "static_channels": list(arm.get("static_channels") or []),
        },
    )

    class _Sum(nn.Module):
        def forward(self, losses):
            return losses.sum()

    def _loss_at(w: float) -> float:
        torch.manual_seed(0)
        model = (
            get_architecture(arm["model"])(
                len(feats), int(arm["total_hidden_channels"]), 1, 0.0, output_distribution=fam_name
            )
            .float()
            .train()
        )
        h = model.init_hTtime(model.base, 8, 8).float()
        torch.manual_seed(1)
        n = len(feats) + len(cls)
        x = (torch.rand(1, 5, n, 8, 8) < 0.05).float() * torch.rand(1, 5, n, 8, 8) * 3
        fam = get_family(fam_name)
        return _process_sequence(
            x,
            model,
            h,
            criterion_reg=FamilyLoss(fam),
            criterion_class=nn.BCEWithLogitsLoss(),
            multitaskloss_instance=_Sum(),
            idx=idx,
            device=torch.device("cpu"),
            family=fam,
            ss_feedback=arm.get("ss_feedback", "mean"),
            forecast_composition=arm.get("forecast_composition", "soft_gate"),
            pushforward_weight=w,
            pushforward_detach_state=detach,
        )["total"].item()

    off, on = _loss_at(0.0), _loss_at(weight)
    if weight == 0.0:
        # The CONTROL proves the opposite property, and it is the one that makes every comparison
        # in this dossier meaningful: at weight 0 the term must be BYTE-identical to not having the
        # flag at all. If it were merely close, the control would itself be a weak treatment.
        if on != off:
            raise SystemExit(
                f"make_pf_arm: the control's loss at weight 0.0 is {on}, not the {off} of a run "
                "with no pushforward at all. Default-off is not inert, so this control is a "
                "treatment and every contrast against it is confounded."
            )
    elif on == off:
        raise SystemExit(
            f"make_pf_arm: pushforward_weight={weight} does not change the loss "
            f"({on} == {off}). The flag reaches the config but the term is not being computed — "
            "this arm would train exactly like its control and be scored as a treatment."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lessons", type=int, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--variant", required=True, help=f"pushforward setting: one of {sorted(_PF_SPEC)}"
    )
    a = p.parse_args()
    dest = build(lessons=a.lessons, eps=a.eps, seed=a.seed, variant=a.variant)
    print(f"built {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
