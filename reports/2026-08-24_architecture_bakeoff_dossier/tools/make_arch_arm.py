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

* ``diagnostic_visualizations`` equals whatever ``--diagnostics`` asked for — **off by default**,
  because violet's is **True** and the per-origin biopsy is expensive. ⛔ It is never flipped
  implicitly in either direction, and a config whose key is missing is refused rather than
  defaulted.
  ⚠️ **The cost this guard has always cited is unsourced.** "~28 min/origin, i.e. ~6 h per emit"
  appears in five builder docstrings and in **no measurement file in this repository**. Treat it as
  unverified; do not quote it as measured.
  ⭐ **What IS measured, from EXP-02's own arm logs with diagnostics OFF** (the `models` session,
  2026-11-03, parsed from `[Origin N/13]` timestamps in `exp02_arm.log`): the emit stage ran
  **3.6 min/origin** on `ctlfullzero_fortythree` (43.3 min / 13) and **1.9 min/origin** on
  `upctwofullzero_fortytwo` (22.6 min / 13). ⇒ The unsourced figure would be an **8–15× increase on
  the emit stage**, which is where the cost lands — so emit, not the whole arm, is the thing to
  compare against. ⭐ And the cheap way to settle it: run one arm with `--diagnostics`, read the
  per-origin timestamps for origins 1 and 2, and kill it there if the number is bad. That is two
  origins, not an arm.
  ⛔ **Two things must be true on the pod or the arm dies AFTER training.** (a) matplotlib must be
  installed: it is an **optional extra** of this package, so the pod must install
  ``views-hydranet[viz]`` — ⚠️ and in ONE command together with the ``toolz>=0.12.1`` override,
  because `pod_run_model.sh:177-181` records that any later install can silently revert toolz below
  0.12 and break every datafactory fetch. ⛔ Never hand-pin matplotlib's range here; it is declared
  by the ``viz`` extra and a second copy drifts without failing. (b) the plots land in a RELATIVE
  ``reports/plots/diagnostics/<ts>/``, which `pod_run_model.sh:371`'s ``cd`` resolves to
  ``/workspace/views-models/models/<arm>/reports/...`` — **outside the runner's archive glob**, so
  they must be pulled separately (views-models `tools/podrun/stream_diagnostics.py`).
  ⚠️ ``arm_identity_check.py`` resolves only ``config_hyperparameters.py``, and this key lives in
  ``config_meta.py`` ⇒ it is **invisible to ``run_queue.sh``'s reuse check**. An arm reused from an
  earlier build keeps the setting it was built with, silently. Build fresh when it matters.
* ``ss_feedback == 'sample'`` whenever ``ss_epsilon_max > 0`` (C-259).
* ``features == regression_targets`` in order (C-260 rejects otherwise; fail before the GPU spins).
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


def _set_diagnostics(mtext: str, *, diagnostics: bool, where: Path | str) -> str:
    """Return `mtext` with `diagnostic_visualizations` set to `diagnostics`, or refuse.

    Extracted from `build` for one reason: a rewrite whose failure mode is SILENCE needs a test,
    and an expression inlined in a 90-line function cannot have one.

    It matches `True` **or** `False` and writes the value that was asked for, rather than matching
    only `True` as this builder did until 2026-11-03. That older form made `--diagnostics` a silent
    no-op against a floor that already said `False` — the fail-open shape, where the tool reports
    success and changes nothing.

    `subn`'s count is checked rather than discarded, so a renamed key, a non-boolean value or a
    second occurrence refuses here, at the rewrite, instead of surfacing later as an arm whose
    diagnostics setting is whatever the floor happened to carry.
    """
    out, n = re.subn(
        r'("diagnostic_visualizations":\s*)(?:True|False)',
        lambda m: m.group(1) + ("True" if diagnostics else "False"),
        mtext,
    )
    if n != 1:
        raise SystemExit(
            f"{_TOOL}: expected exactly one `diagnostic_visualizations` boolean in {where}, "
            f"found {n}. The arm would carry whatever the floor happened to say, so it must not "
            "ship on an unverified diagnostics setting."
        )
    return out


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
    # Mirror the floor's directory skeleton. ModelPathManager validates that each of these exists
    # and raises FileNotFoundError otherwise — so an arm missing `data/processed` dies at startup,
    # which is exactly what happened on the first attempt at 15:15.
    for sub in ("data/generated", "data/processed", "artifacts", "logs", "notebooks", "reports"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
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
    mtext = meta.read_text().replace('"name": "violet_visitor"', f'"name": "{label}"')
    meta.write_text(_set_diagnostics(mtext, diagnostics=diagnostics, where=meta))

    # A failed verification must not leave a half-built arm behind: `build` refuses to overwrite an
    # existing directory, so the wreckage of a rejected build would block every later attempt and
    # look like "the builder is broken" rather than "the config was wrong".
    try:
        # Inside the cleanup deliberately: this refuses, and a refusal outside the `try`
        # would leave `dest` with config_meta.py un-rewritten — which `run_queue.sh:114`
        # reuses and `arm_identity_check.py:117` passes, training a mislabelled arm.
        _strip_orphaned_import(qs)
        _verify(dest, lessons=lessons, eps=eps, seed=seed, variant=variant, control=control,
                diagnostics=diagnostics)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest


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

    assert arm["ss_epsilon_max"] == eps, arm["ss_epsilon_max"]
    assert arm["torch_seed"] == seed and arm["np_seed"] == seed
    assert arm["total_lessons"] == lessons
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
        control = _resolve(control_hp, "get_hp_config")
        vs_control = {k for k in set(control) | set(arm) if control.get(k) != arm.get(k)}
        # A SMOKE arm deliberately differs in total_lessons too — it is never a scored result, so
        # the single-variable invariant applies to the SCORED arms only. Stated rather than
        # silently tolerated, and the smoke's own lesson count is asserted above.
        same_len = lessons == control.get("total_lessons")
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

    m = _resolve(dest / "configs" / "config_meta.py", "get_meta_config")
    # `is not` rather than `!=`, so a truthy non-bool (a stray 1) is refused rather than accepted
    # as equivalent to True.
    if m.get("diagnostic_visualizations") is not diagnostics:
        raise SystemExit(
            f"{_TOOL}: diagnostic_visualizations is "
            f"{m.get('diagnostic_visualizations')!r} but --diagnostics asked for {diagnostics!r}. "
            "The biopsy's cost is the reason this is checked at all, and it is NOT measured in "
            "this repository — so an arm must never acquire it by accident in either direction."
        )
    if m.get("name") != dest.name:
        raise SystemExit(f"{_TOOL}: meta name {m.get('name')!r} != dir {dest.name!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lessons", type=int, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", required=True, help=f"architecture: one of {sorted(_ARCH_TAG)}")
    p.add_argument(
        "--control",
        default=None,
        help="label of THIS experiment's control arm, e.g. ctlfullzero_fortytwo. Default keeps "
             "the historical `fullzero_<seed>`, which no longer exists on disk.",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="build the arm with `diagnostic_visualizations: True`. ⛔ OFF by default. ⚠️ The cost "
             "is NOT measured in this repo: the '~28 min/origin' figure has no evidence file behind "
             "it, and the measured emit baseline with diagnostics off is 1.9-3.6 min/origin, so "
             "that figure would be an 8-15x increase on emit. Read origins 1-2 and kill it if the "
             "number is bad. ⛔ The pod needs `views-hydranet[viz]` (matplotlib is an optional "
             "extra; without it the run dies AFTER training) installed in ONE command with "
             "`toolz>=0.12.1`, and the plots land outside pod_run_model.sh's archive glob — pull "
             "them separately or they are lost with the pod.",
    )
    a = p.parse_args()
    dest = build(lessons=a.lessons, eps=a.eps, seed=a.seed, variant=a.variant,
                 control=a.control, diagnostics=a.diagnostics)
    print(f"built {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
