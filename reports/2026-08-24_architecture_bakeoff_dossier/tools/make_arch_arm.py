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

It asserts, before any GPU work:

* ``diagnostic_visualizations`` equals whatever ``--diagnostics`` asked for — **off by default**,
  because violet's is **True** and the per-origin biopsy is expensive. ⛔ It is never flipped
  implicitly in either direction, and the value is read back by **parse** after the rewrite, so a
  commented-out key, a non-literal such as ``True and False``, or two live copies all refuse.
  ⚠️⚠️ **Both cost figures in circulation are unverified; neither is quoted here as measured.**
  (a) "~28 min/origin, i.e. ~6 h per emit" appears in five builder docstrings and in **no
  measurement file in this repository**. (b) A relayed emit baseline of "1.9-3.6 min/origin" came
  from the `models` session on 2026-11-03, described as parsed from "``[Origin N/13]`` timestamps
  in ``exp02_arm.log``" — but that log carries **no per-line timestamps at all**, only tqdm bars
  with embedded elapsed fields, so the stated method cannot be what produced it; and an independent
  parse of the completed-bar elapsed matched only 4-13 of 13 origins per arm and returned
  **1.25 min/origin** for the arm quoted at 3.6, a ~3x disagreement. ⛔ So the emit cost with
  diagnostics ON **and** OFF are both unmeasured, and the log that would settle either is
  gitignored (``.gitignore:283``), in another repository, under a delivery path whose pods are
  destroyed. ⭐ The cheap way to settle it: run one arm with ``--diagnostics``, read the per-origin
  elapsed for origins 1 and 2, and kill it there. That is two origins, not an arm.
  ⛔ **Two things must be true on the pod, or the arm dies EARLY — not after training.**
  ⚠️ This docstring said "AFTER training" until 2026-11-03 and that was **false**:
  ``hydranet_manager.py:278-279`` hands ``VisualDiagnostics`` to ``_run_data_pipeline`` **before**
  training, and ``training_engine.py`` plots at 1454, 1755, 1884, 1896 and 2036 — **inside** it.
  Since ``visual_diagnostics.py:27-40`` imports matplotlib lazily on the first plotting call, a
  missing matplotlib raises in the data pipeline or during training. ⇒ The loss is minutes, not a
  finished run, and an operator told to "read origins 1-2" would never reach that stage.

  (a) **matplotlib must be installed**, as this package's optional ``viz`` extra. ⛔ Never hand-pin
  its range; the extra declares it and a second copy drifts without failing. ⛔⛔ **Two installs,
  toolz strictly LAST, then asserted.** Measured 2026-11-03 against the real indexes:

  * ``views-hydranet[viz]~=0.1.1`` **plus** ``toolz>=0.12.1`` on one line is **UNSATISFIABLE**:
    ``viewser==6.6.4`` caps ``toolz>=0.11.1,<0.12.0`` and ``views-pipeline-core`` needs
    ``viewser>=6.6.4``. ⇒ That is what `pod_run_model.sh:177-181`'s *"MUST stay the last install"*
    actually encodes — unsatisfiability, not ordering hygiene.
  * the extra installed into a venv already holding a good toolz **actively downgrades it**:
    ``uv pip install --dry-run`` reports ``- toolz==1.2.0`` / ``+ toolz==0.11.2``.
  * ``uv pip install`` is **additive**; it removes nothing. Verified on a throwaway venv —
    ``appwrite==13.6.1`` survived a later unrelated install. ⇒ A hand install does **not** drop
    ``views-pipeline-core[appwrite]``, so views-models#517's publish-time failure is not
    reintroduced. (Raised in review as a defect; measured false.)

  ::

      uv pip install --python "$VENV/bin/python" "views-hydranet[viz]~=0.1.1"   # downgrades toolz
      uv pip install --python "$VENV/bin/python" "toolz>=0.12.1"                # restores it, LAST
      V=$("$VENV/bin/python" -c "import matplotlib, tlz, toolz; print(toolz.__version__)")
      case "$V" in 0.1[2-9]*|0.[2-9]*|[1-9]*) ;; *) echo "toolz $V too old" >&2; exit 1;; esac

  ⚠️ The check imports ``tlz`` as well as ``toolz``, because C-151's symptom is ``tlz`` failing to
  import on py3.11, not a version string reading wrong. ⚠️ And `pod_run_model.sh:186` writes
  ``.built-by-pod-run-model`` last so its presence means "the whole block ran in order" — a hand
  install afterwards makes that marker's claim narrower than it reads.

  (b) **the plots land in a RELATIVE** ``reports/plots/diagnostics/<ts>/``, which
  `pod_run_model.sh:371`'s ``cd`` resolves to
  ``/workspace/views-models/models/<arm>/reports/...`` — **outside the runner's archive glob**, so
  they must be pulled separately (views-models ``tools/podrun/stream_diagnostics.py``).

  ⛔ **This flag is unreachable from the only programmatic launcher.** `run_queue.sh:149` invokes
  this builder with ``--lessons/--eps/--seed`` (plus ``--variant``) and **nothing else**: no
  ``--diagnostics`` and no ``--control``, and its spec grammar has no slot for either. ⇒ A
  diagnostics arm must be built BY HAND, and — because ``--control`` is also absent and the default
  ``fullzero_<seed>`` directories no longer exist — ``run_queue.sh`` cannot build **any**
  300-lesson arm at all; it reuses pre-built ones. ⚠️ So this builder is not the gate on the launch
  path, and ``arm_identity_check.py`` resolves only ``config_hyperparameters.py`` while this key
  lives in ``config_meta.py`` ⇒ it is **invisible to the reuse check**. An arm reused from an
  earlier build keeps the setting it was built with, silently. Build fresh when it matters.
⛔ **Run this with the project interpreter**, ``conda run -n views-hydranet-env``. ``_verify``
imports ``views_hydranet.architectures.registry`` to instantiate the variant on CPU, and the
``views_pipeline`` environment has no ``views_hydranet`` at all — so a non-default variant fails
only at verification, after the arm has been copied. Reported 2026-11-03 after it cost a pass.

It also asserts, before any GPU work:

* ``ss_feedback`` is **set** to ``'sample'`` whenever ``ss_epsilon_max > 0`` (C-259) — ⚠️ written
  at the ``_set_key`` step, **not** asserted in ``_verify``, unlike every other item in this list.
  It is the one write with no readback anywhere; said plainly because this docstring claimed it was
  asserted until 2026-11-03.
* ``features == regression_targets`` in order (C-260 rejects otherwise; fail before the GPU spins).
* ⛔ every directory in ``_REQUIRED_SUBDIRS`` holds a trackable file, not merely exists — the arm
  reaches the pod by ``git clone --depth 1`` and git carries files, so an empty one is absent there
  and ``ModelPathManager`` raises AFTER provisioning. Measured at **1 of 7** on all eight EXP-03
  arms built at 99fe887; views-models #685 is the same failure five days earlier.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: This tool's own name, for error prefixes. Derived, never retyped: five of this
#: file's refusal messages named a DIFFERENT dossier's tool because the name was a
#: literal copied down the make_ss -> make_itf -> make_trunc -> make_arch lineage.
_TOOL = Path(__file__).stem

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
#: name; they are encoded through these tables instead. ⚠️ This comment named ``arm.json`` as the
#: authoritative record of those values until 2026-11-03, and **nothing in either repository writes
#: one** — no ``json.dump`` in any of the five builders, and no ``arm.json`` beside any of the four
#: built arms. The authoritative record is each arm's own ``configs/*.py``, which is what
#: ``arm_identity_check.py`` and ``_verify`` actually read. ⇒ The encoding is deterministic both
#: ways, so a name is still a reliable handle in a log or the process table, but it is the ONLY
#: machine-readable summary and that is a thinner guarantee than the old comment implied.
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
#: Sentinel for "no single live boolean under this key". A plain `None` would collide with a
#: config that genuinely wrote `None`, which is one of the shapes that must refuse.
_ABSENT = object()


#: The directories ``ModelPathManager`` validates on import. ⚠️ Each must contain a TRACKABLE
#: file, not merely exist: the arm reaches the pod through `git clone --depth 1`, and git carries
#: files rather than directories.
_REQUIRED_SUBDIRS = (
    "data/raw",
    "data/generated",
    "data/processed",
    "artifacts",
    "logs",
    "notebooks",
    "reports",
)


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
    # 2026-10-30, EXP-02 (reports/2026-10-30_upsample_arm/PREREG.md section 10c). ADDITIVE: no
    # existing label changes. The tags are spelled out because `validate_model_name` is
    # `^[a-z]+_[a-z]+$` and forbids digits, so "up2" would be rejected at pipeline startup —
    # after the queue had already accepted the arm, which is how the ITF pilot's first launch died.
    #
    # ⭐ The incumbent is here for the first time, and it is load-bearing: EXP-01 paired against
    # `fullzero_*` arms built by an EARLIER programme, those model directories no longer exist, and
    # pairing against their committed scores would vary the CODE VERSION as well as `model`. The
    # control is therefore an arm of the EXP-02 queue, built by this builder, on the same commit.
    "HydraBNUNet06_LSTM4": "ctl",
    "UpsampleConv2": "upctwo",
    "UpsampleConv1": "upcone",
}


def arm_label(*, lessons: int, eps: float, seed: int, variant: str | None = None) -> str:
    """Pipeline-legal directory name: (300, 0.0, 42, AntiAliasedPool) -> `aafullzero_fortytwo`."""
    if variant not in _ARCH_TAG:
        raise SystemExit(
            f"{_TOOL}: --variant must be one of {sorted(_ARCH_TAG)}; got {variant!r}. "
            "Add a tag to the table rather than improvising a name."
        )
    for table, key, what in (
        (_LESSON_WORD, lessons, "lessons"),
        (_EPS_WORD, eps, "eps"),
        (_SEED_WORD, seed, "seed"),
    ):
        if key not in table:
            raise SystemExit(
                f"{_TOOL}: no word for {what}={key}. Add one to the table rather than "
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
        raise SystemExit(f"{_TOOL}: unknown variant {variant!r}")
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
        raise SystemExit(f"{_TOOL}: key {key!r} not found exactly once")
    return new


def _resolve(path: Path, fn: str) -> dict:
    """exec a config file and call its getter — the readback that makes verification real."""
    text = path.read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns[fn]()



def _strip_orphaned_import(qs: Path) -> None:
    """Remove the import that pinning ``model_name`` orphans, and REFUSE if it survives.

    ⛔ Pinning the name deletes the import's only executable use, so ``ruff`` fires F401 on the
    generated arm. That reddens ``ruff check .`` in views-models — ``ship-it`` Step 1 for every
    seat shipping from that repo — and the arm directory is not this tool's to revisit afterwards.

    ⭐ ``ruff`` does the removal rather than a text edit here, because ``ruff`` is what *defines*
    the failure: it is already pinned in ``pyproject.toml`` and already shelled out to by
    ``tests/test_falsification_repo_clean.py``, and it is AST-aware. A hand-rolled removal has to
    get an ``__all__`` re-export, a string annotation, a parenthesised or aliased import, an
    indented import inside ``if TYPE_CHECKING:``, and a docstring copy of the import line all
    right, and when it does not it either ships the F401 anyway or deletes a binding ``ruff``
    considers live — turning the check it exists to satisfy from green to red.

    ⛔ The post-check is the point, not a belt-and-braces extra. The failure mode of an anchored
    rewrite that misses its anchor is SILENCE: the builder prints success and the arm ships
    F401-red. So ``--fix`` is verified, never trusted.

    ``--isolated`` keeps the rule set this tool's own, rather than inheriting whatever config
    happens to sit above the arm directory in views-models.
    """
    cmd = ["ruff", "check", "--isolated", "--select", "F401"]
    try:
        # ⚠️ The --fix result is deliberately discarded: the re-check below is the verification, so
        # `check=True` here would only add an uncaught crash site without proving anything.
        subprocess.run([*cmd, "--fix", "-q", str(qs)], check=False, capture_output=True, text=True)
        after = subprocess.run([*cmd, str(qs)], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(
            f"{_TOOL}: `ruff` is not on PATH, so the import orphaned by pinning model_name in "
            f"{qs} cannot be removed and the arm would ship F401-red. Run this inside the "
            "project environment."
        ) from None
    # ruff exits 1 for findings and 2 for its own failure. Both refuse, but they are different
    # problems and a message that names the wrong one sends the operator to the wrong file.
    if after.returncode == 1:
        raise SystemExit(
            f"{_TOOL}: {qs} still fails `ruff check --select F401` after --fix:\n"
            f"{after.stdout}{after.stderr}"
        )
    if after.returncode != 0:
        raise SystemExit(
            f"{_TOOL}: `ruff` itself failed (exit {after.returncode}) on {qs}, so whether the arm "
            f"is F401-clean is UNKNOWN and it must not ship on an unverified config:\n"
            f"{after.stdout}{after.stderr}"
        )


def _meta_bool(text: str, key: str, getter: str = "get_meta_config"):
    """The literal value of ``key`` in the dict ``getter`` RETURNS, parsed — never executed.

    ⛔ Scoped to that ONE dict, and refusing anything not statically determinable. Walking every
    ``ast.Dict`` in the file false-accepted twice, both measured rather than imagined:

    * ``return {"inner": {key: True}}`` reported ``True`` while the real config has no such key
      at all, so an arm asked to run with diagnostics ON would have shipped with them OFF.
    * ``cfg = {key: False}`` followed by ``cfg[key] = True`` reported ``False`` while the value is
      ``True`` — the parse reads the CONSTRUCTION, not the value, and that is exactly the direction
      a hand-edited arm would take. ⇒ A gate that reads construction is no gate against tampering.

    ⚠️ The real configs are ``meta_config = {...}`` followed by ``return meta_config``, not a bare
    ``return {literal}`` — so the returned NAME is traced back to its single assignment, and the
    name is refused if anything in the function could have mutated it afterwards.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _ABSENT
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == getter]
    if len(fns) != 1:
        return _ABSENT
    fn = fns[0]
    #: ⛔ A decorator can replace the return value wholesale, so the body would say nothing about
    #: what the getter actually yields. Refuse rather than read the body underneath it.
    if fn.decorator_list:
        return _ABSENT
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop the docstring
    if not body or not isinstance(body[-1], ast.Return):
        return _ABSENT

    returned = body[-1].value
    if isinstance(returned, ast.Dict):
        target = returned
    elif isinstance(returned, ast.Name):
        name = returned.id
        assigns = [
            n
            for n in body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == name
        ]
        if len(assigns) != 1 or not isinstance(assigns[0].value, ast.Dict):
            return _ABSENT
        #: ⛔ Refuse if ANYTHING could have changed it after construction: a subscript store, a
        #: method call such as `.update`, or an augmented assignment. This is the false-accept.
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id == name
                and isinstance(n.ctx, ast.Store)
            ):
                return _ABSENT
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == name
            ):
                return _ABSENT
            if (
                isinstance(n, ast.AugAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == name
            ):
                return _ABSENT
        target = assigns[0].value
    else:
        return _ABSENT

    found = _ABSENT
    for k, v in zip(target.keys, target.values):
        if k is None:
            return _ABSENT  # `**` unpacking: the key may be hiding inside it
        if not (isinstance(k, ast.Constant) and k.value == key):
            continue
        if not (isinstance(v, ast.Constant) and isinstance(v.value, bool)):
            return _ABSENT
        if found is not _ABSENT:
            return _ABSENT  # two live copies: which one wins is not ours to guess
        found = v.value
    return found

def _rewrite_meta(meta: Path, *, label: str, diagnostics: bool) -> None:
    """Rewrite the arm's ``config_meta.py`` name and diagnostics keys, verify by parse, then write.

    ⛔ **The write is inside this function deliberately.** The earlier form returned text for the
    caller to write, and a mutation that called it for its refusal and then wrote the *unvalidated*
    original survived the entire suite — so did deleting the sibling name rewrite. The validated
    text must be the only text that can reach disk.

    Both rewrites check ``subn``'s count rather than discarding it, because the failure mode of a
    rewrite that misses its anchor is SILENCE.
    """
    text = meta.read_text(encoding="utf-8")

    text, n_name = re.subn(r'"name":\s*"violet_visitor"', f'"name": "{label}"', text)
    if n_name != 1:
        raise SystemExit(
            f"{_TOOL}: expected exactly one `\"name\": \"violet_visitor\"` in {meta}, found "
            f"{n_name}. The arm would ship under the floor's name, which "
            "`arm_identity_check.py` cannot see because it reads only config_hyperparameters.py."
        )

    #: ⚠️ The ``\b`` is defence in depth, not the guard. Measured 2026-11-03: removing it leaves
    #: every bad shape still refusing and the file still untouched — only WHICH refusal fires
    #: changes (count vs readback). ⇒ It is an equivalent mutant, and no test pins it, because a
    #: test whose only job is to hold a message in place is one more thing to maintain and nothing
    #: more. The readback below is what actually closes the class.
    text, n_diag = re.subn(
        r'("diagnostic_visualizations":\s*)(?:True|False)\b',
        lambda m: m.group(1) + ("True" if diagnostics else "False"),
        text,
    )
    if n_diag != 1:
        raise SystemExit(
            f"{_TOOL}: expected exactly one `diagnostic_visualizations` boolean in {meta}, found "
            f"{n_diag}. The arm would carry whatever the floor happened to say, so it must not "
            "ship on an unverified diagnostics setting. ⚠️ Fix the FLOOR's config_meta.py "
            f"({_FLOOR / 'configs' / 'config_meta.py'}); {meta} is a copy and this arm directory "
            "is removed, so a retry is safe."
        )

    got = _meta_bool(text, "diagnostic_visualizations")
    if got is not diagnostics:
        raise SystemExit(
            f"{_TOOL}: after rewriting {meta}, `diagnostic_visualizations` parses as "
            f"{'ABSENT' if got is _ABSENT else repr(got)} but {diagnostics!r} was asked for. "
            "The textual rewrite reported success and did not achieve it — a commented-out key, a "
            "non-literal value such as `True and False`, or two live copies. The arm is NOT built."
        )

    meta.write_text(text, encoding="utf-8")



def build(
    *, lessons: int, eps: float, seed: int, variant: str, out_root: Path = _MODELS,
    control: str | None = None, diagnostics: bool = False,
) -> Path:
    label = arm_label(lessons=lessons, eps=eps, seed=seed, variant=variant)
    dest = out_root / label
    if dest.exists():
        raise SystemExit(f"{_TOOL}: {dest} already exists — refusing to overwrite an arm")

    shutil.copytree(
        _FLOOR,
        dest,
        ignore=shutil.ignore_patterns(
            "data", "logs", "wandb", "artifacts", "reports", "notebooks", "__pycache__", "*.pyc"
        ),
    )
    # ⛔⛔ THE CLEANUP STARTS HERE — immediately after `dest` exists on disk, and it wraps
    # EVERYTHING. Twice now the `try` began further down and a refusal above it left a
    # half-built arm behind; the second time (2026-11-03) the surviving site was the bare
    # `raise SystemExit` for config_queryset.py, which fires AFTER
    # `config_hyperparameters.py` has been written. That combination is the dangerous one:
    # `arm_identity_check.py:117` reads ONLY config_hyperparameters.py, so it PASSES, and
    # `run_queue.sh:114` reuses the directory on `[ -d ... ]` alone ⇒ the queue trains an arm
    # whose config_meta.py still says `"name": "violet_visitor"` and still carries the
    # floor's `diagnostic_visualizations`.
    #
    # ⇒ The rule is structural, not a list of sites: once `dest` exists, every path out of
    # this function either returns it complete or removes it. Pinned by
    # `test_every_refusal_after_dest_exists_is_inside_the_cleanup`, which enumerates the
    # `raise` statements rather than a hand-written set of callee names — the two previous
    # guards were keyed to names and could not see a new site.
    try:
        # Mirror the floor's directory skeleton. ModelPathManager validates that each of these exists
        # and raises FileNotFoundError otherwise — so an arm missing `data/processed` dies at startup,
        # which is exactly what happened on the first attempt at 15:15.
        for sub in _REQUIRED_SUBDIRS:
            d = dest / sub
            d.mkdir(parents=True, exist_ok=True)
            #: ⛔⛔ A `.gitkeep` per directory, or the directory does not survive the pod's
            #: `git clone --depth 1` and `ModelPathManager` raises BEFORE training — after the pod
            #: is provisioned and paid for. git tracks files, not directories, and `mkdir` leaves
            #: these EMPTY: the floor carries all seven `.gitkeep` files but
            #: `shutil.ignore_patterns` above drops `data`, `logs`, `artifacts`, `reports` and
            #: `notebooks`, so only `data/raw` got one back — via the parquet copy, by accident.
            #: ⇒ 1 of 7, measured on all eight EXP-03 arms built at 99fe887. This is views-models
            #: #685 recurring in the tool five days after it was fixed in the instructions.
            (d / ".gitkeep").touch()
        # data/raw carries the cached parquet, so no arm fetches from the network mid-sweep.
        src_raw = _FLOOR / "data" / "raw"
        if not src_raw.is_dir():
            raise SystemExit(f"{_TOOL}: {src_raw} missing — an arm would fetch data mid-sweep")
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
                f"{_TOOL}: config_queryset.py no longer derives model_name as expected"
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

        # A failed verification must not leave a half-built arm behind: `build` refuses to overwrite an
        # existing directory, so the wreckage of a rejected build would block every later attempt and
        # look like "the builder is broken" rather than "the config was wrong".
        # Inside the cleanup deliberately: these refuse, and a refusal outside the `try`
        # would leave `dest` with config_meta.py un-rewritten — which `run_queue.sh:114`
        # reuses and `arm_identity_check.py:117` passes, training a mislabelled arm.
        #
        # ⛔⛔ `_rewrite_meta` is here for that reason and it was NOT, until 2026-11-03. The form
        # `meta.write_text(_set_diagnostics(mtext, ...))` put its refusal FIVE LINES ABOVE this
        # `try`, because Python evaluates the argument before the call — so the defect this very
        # comment was written to prevent was reintroduced one statement above the fix, and the
        # test that asserts the property could not see it because it is keyed to the name
        # `_strip_orphaned_import`. Found by `/code-review max`.
        _rewrite_meta(meta, label=label, diagnostics=diagnostics)
        _strip_orphaned_import(qs)
        _verify(dest, lessons=lessons, eps=eps, seed=seed, variant=variant, control=control,
                diagnostics=diagnostics)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest


def _assert_clonable(dest: Path) -> None:
    """Refuse unless every required subdirectory holds at least one file git can track.

    ⛔ Existence is not the property. ``ModelPathManager`` validates that these directories exist,
    but the arm reaches the pod through ``git clone --depth 1`` — and git carries FILES. An empty
    directory is simply absent on the other side, so validation raises there and not here: after
    the pod is provisioned, after the operator has paid for it, and with nothing in the build log
    to explain it.

    ⇒ Measured on all eight EXP-03 arms built at 99fe887: **1 of 7**, the one being ``data/raw``
    and only because the parquet copy happened to bring a ``.gitkeep`` with it. views-models #685
    is the same failure, fixed in the instructions five days earlier and then reintroduced by this
    tool — which is the argument for asserting it here rather than documenting it again.
    """
    bare = [s for s in _REQUIRED_SUBDIRS if not any((dest / s).iterdir())]
    if bare:
        raise SystemExit(
            f"{_TOOL}: {sorted(bare)} exist but are EMPTY in {dest}. git tracks files, not "
            "directories, so `git clone --depth 1` would not recreate them on the pod and "
            "ModelPathManager would raise there — after provisioning. The arm is NOT built."
        )


def _assert_meta(dest: Path, *, diagnostics: bool) -> None:
    """Read the BUILT arm's ``config_meta.py`` back and refuse unless it says what was asked.

    Separate from ``_verify`` so it can be executed by a test: ``_verify`` instantiates a torch
    architecture and resolves a control arm from ``_MODELS``, so nothing in the suite ran it, and
    three mutations on this check survived — ``is not diagnostics`` → ``!=``, the whole ``if`` →
    ``if False:``, and a silent revert to the old hard ``is not False`` (which would make
    ``--diagnostics`` refuse its own output).
    """
    m = _resolve(dest / "configs" / "config_meta.py", "get_meta_config")
    #: ⛔ `is not`, not `!=`. `True == 1` is true in Python, so `!=` would accept a stray `1` as
    #: equivalent to True. Verified: 1, 1.0, "True", None and [] are all refused by `is`.
    if m.get("diagnostic_visualizations") is not diagnostics:
        raise SystemExit(
            f"{_TOOL}: diagnostic_visualizations is "
            f"{m.get('diagnostic_visualizations')!r} but --diagnostics asked for {diagnostics!r}. "
            "The biopsy's cost is the reason this is checked at all, and it is NOT measured in "
            "this repository — so an arm must never acquire it by accident in either direction."
        )
    if m.get("name") != dest.name:
        raise SystemExit(f"{_TOOL}: meta name {m.get('name')!r} != dir {dest.name!r}")


def _verify(dest: Path, *, lessons: int, eps: float, seed: int, variant: str,
            control: str | None = None, diagnostics: bool = False) -> None:
    floor = _resolve(_FLOOR / "configs" / "config_hyperparameters.py", "get_hp_config")
    arm = _resolve(dest / "configs" / "config_hyperparameters.py", "get_hp_config")

    diff = {k for k in set(floor) | set(arm) if floor.get(k) != arm.get(k)}
    expected = {k for k in _INTENDED if arm.get(k) != floor.get(k)}
    if diff != expected:
        raise SystemExit(
            f"{_TOOL}: config differs from the floor in {sorted(diff)}, expected exactly "
            f"{sorted(expected)}. An unintended key changed — the arm is NOT built."
        )

    #: ⛔ `SystemExit`, not `assert`. A bare `assert` is removed entirely under `-O` /
    #: `PYTHONOPTIMIZE`, so these three arm-identity invariants — the ones `arm_identity_check.py`
    #: re-checks on every queue reuse — would silently stop being checked, while every other
    #: refusal in this function kept working. Flagged by `/code-review max`.
    for _key, _want in (("ss_epsilon_max", eps), ("torch_seed", seed), ("np_seed", seed),
                        ("total_lessons", lessons)):
        if arm.get(_key) != _want:
            raise SystemExit(
                f"{_TOOL}: {_key} is {arm.get(_key)!r}, expected {_want!r} — this arm's identity "
                "is not the one requested, and `arm_identity_check.py` would accept it on reuse."
            )
    if arm.get("model") != variant:
        raise SystemExit(
            f"{_TOOL}: model is {arm.get('model')!r}, expected {variant!r} — this arm would "
            "train a different architecture than the one requested."
        )
    if arm.get("output_distribution") != floor.get("output_distribution"):
        raise SystemExit(
            f"{_TOOL}: output_distribution changed. Only `model` may differ — this program "
            "varies the ARCHITECTURE, and M45 is what a two-key experiment costs."
        )

    # ⛔⛔ BEFORE the self-control early return below, not after. Until 2026-11-03 this readback
    # sat 57 lines further down, past `if control_dir.resolve() == dest.resolve(): return` — so it
    # was DEAD for every arm that is its own control, i.e. every `ctlfullzero_*`, which is six of
    # the twelve arms `PREREG_EXP03_SEEDS.md` pools into σ. That document rested its safety claim
    # on this exact readback ("`_verify` guarantees matches what was asked for"), and the guarantee
    # did not exist for half the scored set. Found by `/code-review max`.
    #
    # It also has nothing to do with the control contrast; it was below it only by append order.
    _assert_meta(dest, diagnostics=diagnostics)
    _assert_clonable(dest)

    # ── the contrast this dossier actually claims ────────────────────────────────────────────
    # The ancestors diff only against the FLOOR. But every claim here is treatment-vs-CONTROL,
    # and the control is `fullzero_<seed>`. Verify THAT symmetric difference is exactly the one
    # key, or the single-variable claim is unproven no matter what the floor diff says.
    #: ⛔⛔ `control` is EXPERIMENT-SPECIFIC and must be passed, not assumed. The original form
    #: hardcoded `fullzero_<seed>`, and those directories NO LONGER EXIST — so the `elif lessons
    #: == 300` branch below raised for every scored arm while the 2-lesson smoke path took neither
    #: branch and still wrote SMOKE_OK. ⇒ The whole gate chain reported ready for a launch that
    #: could not build one scored arm. Found by an independent `/code-review max` of PR #492.
    #:
    #: ⭐ Default is UNCHANGED (`fullzero_<seed>`), so EXP-01's behaviour is byte-identical.
    #: EXP-02 passes `--control ctlfullzero_<seed>`, which is an arm of its own queue.
    control_name = control or f"fullzero_{_SEED_WORD[seed]}"
    control_dir = _MODELS / control_name
    #: ⛔ When the arm IS its own control — EXP-02 re-runs the incumbent as a queue member — there
    #: is no contrast to check and `vs_control` is `set()`, which the `allowed = {"model"}` rule
    #: below would reject. The reference cannot differ from itself, and saying so is not a
    #: weakening: it is the one case where the single-variable claim is vacuously true.
    if control_dir.resolve() == dest.resolve():
        print(f"  {dest.name} IS its own control — no contrast to verify (it is the reference)")
        return
    control_hp = control_dir / "configs" / "config_hyperparameters.py"
    if control_hp.is_file():
        #: Named `control_cfg`, not `control`: rebinding the `control: str | None` parameter to a
        #: dict made the two meanings share one name for the rest of the function.
        control_cfg = _resolve(control_hp, "get_hp_config")
        vs_control = {k for k in set(control_cfg) | set(arm) if control_cfg.get(k) != arm.get(k)}
        # A SMOKE arm deliberately differs in total_lessons too — it is never a scored result, so
        # the single-variable invariant applies to the SCORED arms only. Stated rather than
        # silently tolerated, and the smoke's own lesson count is asserted above.
        same_len = lessons == control_cfg.get("total_lessons")
        allowed = {"model"} if same_len else {"model", "total_lessons"}
        if vs_control != allowed:
            raise SystemExit(
                f"{_TOOL}: arm differs from its control {control_dir.name} in "
                f"{sorted(vs_control)}, expected exactly {sorted(allowed)}. The experiment is "
                "NOT single-variable — the arm is NOT built."
            )
    elif lessons == 300:
        raise SystemExit(
            f"{_TOOL}: control {control_dir} not found — a scored arm has nothing to be "
            "compared against. Pass --control <label> naming the control arm of THIS experiment; "
            "the historical `fullzero_*` directories no longer exist."
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
            f"{_TOOL}: {arm['model']} has state width {net.base}, not divisible by 8; "
            "blend_recurrent_state would silently mis-assign short-term vs long-term memory."
        )
    n_targets = len(arm["regression_targets"])
    with torch.no_grad():
        out = net(torch.zeros(1, arm["input_channels"], 16, 16), net.init_hTtime(net.base, 16, 16))
    if out.cls.shape[1] != n_targets:
        raise SystemExit(
            f"{_TOOL}: {arm['model']} emits {out.cls.shape[1]} gate channels, expected "
            f"{n_targets} — the emitted field would not match the targets."
        )



def _parser() -> argparse.ArgumentParser:
    """The real parser, extracted from ``main`` so a test can assert against IT.

    ⛔ Three mutations survived the whole suite while this lived inside ``main``: adding
    ``default=True`` to ``--diagnostics``, forwarding a literal ``diagnostics=False``, and dropping
    the kwarg. The test could only check substrings of ``main``'s source and the default of a
    throwaway parser the TEST built — i.e. it asserted CPython's ``store_true`` semantics with no
    reference to the code under test. Found by `/code-review max`.

    ``RawDescriptionHelpFormatter`` because ``description=__doc__`` is a multi-kilobyte block whose
    bullets argparse otherwise reflows into a wall.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lessons", type=int, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", required=True, help=f"architecture: one of {sorted(_ARCH_TAG)}")
    p.add_argument(
        "--control",
        default=None,
        help="label of THIS experiment's control arm, e.g. ctlfullzero_fortytwo. Default keeps "
             "the historical `fullzero_<seed>`, which no longer exists on disk. NOTE: a CONTROL "
             "arm is built naming ITSELF here -- `--variant <incumbent> --control <its own label>` "
             "-- because the reference cannot differ from itself and _verify returns early on that "
             "branch. This was not discoverable from the help text and cost a failed build pass.",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="build the arm with `diagnostic_visualizations: True`. OFF by default. "
             "THE COST IS UNMEASURED: the '~28 min/origin' figure has no evidence file behind it, "
             "and a relayed '1.9-3.6 min/origin' baseline could not be reproduced (its stated "
             "source has no timestamps, and an independent parse disagreed ~3x) -- so run one arm, "
             "read origins 1-2, and kill it if the number is bad. The pod needs "
             "`views-hydranet[viz]` as TWO installs with `toolz>=0.12.1` strictly LAST, then "
             "asserted: the one-line combined form is UNSATISFIABLE (viewser caps toolz<0.12) and "
             "the extra alone downgrades toolz to 0.11.2, which breaks every datafactory fetch "
             "with no error. Without matplotlib the run dies in the DATA PIPELINE or during "
             "training, not after it. And the plots land outside pod_run_model.sh's archive glob "
             "-- pull them before teardown or they are lost with the pod.",
    )
    return p


def main() -> int:
    a = _parser().parse_args()
    dest = build(lessons=a.lessons, eps=a.eps, seed=a.seed, variant=a.variant,
                 control=a.control, diagnostics=a.diagnostics)
    print(f"built {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
