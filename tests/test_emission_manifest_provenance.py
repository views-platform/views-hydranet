"""ADR-074 §2.1: the provenance the manifest records, and the labels it puts on what it wrote.

⛔⛔ **Nothing in this file had a test of any kind before it.** A mutation audit of commit
`3fad6e4` found that `_repo_provenance`'s two outputs — `commit` and `dirty` — were asserted
nowhere: the existing tests check only `{r.name for r in m.repos}`, the set of names, which is
fixed by the loop that builds it and cannot disagree with anything. So every one of these left
the suite green at *2504 passed*:

- inverting the toplevel check, so a correctly-anchored repository records unknown-and-dirty and
  only a **mis**-anchored one records a commit;
- deleting the toplevel check, so a wheel in a `site-packages` tree nested under an unrelated
  checkout records **that** repository's HEAD as ours;
- swapping the views-hydranet anchor to the views-models path, so both entries read one repository;
- dropping `-uno`, so `dirty` counts files the run itself created and can never be false;
- swapping the `forecast` and `gate` labels, so every manifest calls the body a gate;
- dropping the `unrecognised` fallback, so an unlabelled target vanishes from the record.

⚠️ Each is a **false provenance claim** — the precise failure ADR-074 §2.1 says the `dirty` flag
exists to prevent — and each is silent. A reader has no way to tell a recorded commit that is this
checkout's from one belonging to whatever repository happened to sit above it.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("views_pipeline_core")

from tests.test_emission_manifest_adversarial import (  # noqa: E402
    ARTIFACT,
    _ctx,
    _fake_manager,
    _write,
)
from views_hydranet.infrastructure.emission_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    read_manifest,
)
from views_hydranet.manager.hydranet_manager import HydranetManager  # noqa: E402


def _git_repo(parent: Path, name: str) -> Path:
    """A real git repository named *name*, with one commit.

    Named deliberately: the name is what the toplevel check compares against.
    """
    repo = parent / name
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "tracked.py").write_text("x = 1\n")
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "T")
    run("add", "src/tracked.py")
    run("commit", "-qm", "one")
    return repo


def _provenance(models_path):
    """`_repo_provenance` called unbound on the smallest stand-in it can read."""
    model_path = MagicMock()
    model_path.models = models_path
    return {r.name: r for r in HydranetManager._repo_provenance(MagicMock(_model_path=model_path))}


# ─── `commit`: whose revision is it? ────────────────────────────────────────


def test_the_hydranet_entry_records_this_checkout_s_head(tmp_path):
    """⭐ The recorded commit must be THIS repository's HEAD, compared against git itself.

    Catches two mutations at once: inverting the toplevel comparison (which would record
    unknown-and-dirty for the correctly-anchored repository) and swapping the views-hydranet anchor
    to the views-models path (which would make one repository answer for both).
    """
    import views_hydranet

    anchor = Path(views_hydranet.__file__).resolve().parent
    expected = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    repos = _provenance(tmp_path)

    assert repos["views-hydranet"].commit == expected, (
        f"recorded {repos['views-hydranet'].commit!r} for views-hydranet; this checkout's HEAD is "
        f"{expected!r}. A commit that is not ours is a false provenance claim, not a near miss"
    )
    assert len(expected) == 40, "the fixture did not read a real revision"


def test_a_repository_that_is_not_the_one_it_claims_records_no_commit(tmp_path, caplog):
    """⛔ The anchor sits inside SOME repository — that is not enough to claim its revision.

    `git -C` walks UP, so a non-editable wheel in a `site-packages` tree nested under any unrelated
    checkout resolves to that checkout. Deleting the toplevel verification left the suite green,
    and the manifest would then record an unrelated repository's HEAD as the one that produced the
    run.
    """
    foreign = _git_repo(tmp_path, "not-views-models")

    with caplog.at_level("WARNING"):
        repos = _provenance(foreign)

    assert repos["views-models"].commit is None, (
        f"recorded {repos['views-models'].commit!r} from a repository named "
        f"{foreign.name!r} — an unrelated checkout's revision, presented as this run's"
    )
    assert repos["views-models"].dirty is True, "unknown must degrade to dirty, not to clean"
    assert "which is not views-models" in caplog.text


def test_a_repository_that_cannot_be_read_is_unknown_and_dirty(tmp_path):
    """*Green:* a path in no repository at all is recorded as unknown, never omitted.

    A missing entry is indistinguishable from a repository that was never involved.
    """
    repos = _provenance(tmp_path / "not_a_repo")

    assert repos["views-models"] == type(repos["views-models"])(
        name="views-models", commit=None, dirty=True
    )


# ─── `dirty`: a flag that cannot be false carries no information ─────────────


def test_an_untracked_file_is_not_a_modification(tmp_path):
    """⛔ `-uno`, and it is the difference between a flag and a constant.

    The run writes its own predictions into views-models *before* this is called, so with untracked
    files counted `dirty` was structurally unable to be false for the one repository whose
    configuration was being run — a provenance qualifier that qualified every run equally.
    """
    repo = _git_repo(tmp_path, "views-models")
    (repo / "predictions_calibration_20261102").mkdir()
    (repo / "predictions_calibration_20261102" / "origin_0.parquet").write_text("not really")

    repos = _provenance(repo)

    assert repos["views-models"].commit is not None, "the fixture repository was not read"
    assert repos["views-models"].dirty is False, (
        "an untracked file the run created itself was counted as a modification of the committed "
        "state, so `dirty` is true on every run and says nothing"
    )


def test_a_tracked_modification_is_a_modification(tmp_path):
    """*Green, and the half that stops the fix becoming `dirty = False`.*"""
    repo = _git_repo(tmp_path, "views-models")
    (repo / "src" / "tracked.py").write_text("x = 2\n")

    assert _provenance(repo)["views-models"].dirty is True


# ─── `quantities`: the label on what was written ────────────────────────────


def test_each_target_is_labelled_with_the_quantity_it_holds(tmp_path):
    """⛔ Swapping the two labels left the suite green: nothing asserted them from a run.

    A reader joining a directory to its quantity would read the forecast as the gate and the gate
    as the forecast — in every manifest, with nothing anywhere to contradict it.
    """
    fake, predictions = _fake_manager(tmp_path, {})
    _write(fake, _ctx(all_targets=["lr_ged_sb", "by_sb_best"]))

    quantities = read_manifest(predictions / MANIFEST_FILENAME).quantities

    assert quantities == {"lr_ged_sb": "forecast", "by_sb_best": "gate"}, (
        f"recorded {quantities}; `lr_` is the body's forecast and `by_` is the gate"
    )


def test_a_target_matching_neither_prefix_is_labelled_not_dropped(tmp_path):
    """⛔ Dropping it makes "never requested" indistinguishable from "failed to write".

    An unrecognised quantity is a finding. Silence about it is the thing ADR-074 exists to prevent,
    and deleting the fallback branch cost nothing.
    """
    fake, predictions = _fake_manager(tmp_path, {})
    _write(fake, _ctx(all_targets=["lr_ged_sb", "something_new"]))

    quantities = read_manifest(predictions / MANIFEST_FILENAME).quantities

    assert quantities["something_new"] == "unrecognised", (
        f"recorded {quantities}; an unlabelled target was dropped from the record rather than "
        f"reported, so the manifest no longer lists every target the run emitted"
    )


# ─── The writer's two `except` clauses are two different reports ─────────────


def test_a_defect_in_the_writer_is_reported_as_a_defect(tmp_path, caplog):
    """⭐ Collapsing the two clauses into one left the suite green.

    The existing test asserts only that *an* ADR-074 error is logged, not which clause produced
    it — so a programming error could be reported in a line indistinguishable from a genuine
    serialisation failure, and with no traceback. Four tests used to pass exactly that way, which
    means the feature could have been failing on every call with the suite green.
    """
    fake, predictions = _fake_manager(tmp_path, {})

    with patch(
        "views_hydranet.manager.hydranet_manager.declared_coverage_of",
        side_effect=AttributeError("a defect, not an environment"),
    ):
        with caplog.at_level("ERROR"):
            _write(fake, _ctx())

    assert "UNEXPECTED exception" in caplog.text, (
        "an AttributeError was reported in the environmental-failure line; a defect in the writer "
        "and an unwritable disk must not read the same"
    )
    assert "Traceback" in caplog.text, "the unexpected-exception clause must log `exc_info`"
    assert not (predictions / MANIFEST_FILENAME).exists()


def test_an_environmental_failure_is_not_reported_as_a_defect(tmp_path, caplog):
    """*Green, and the other side:* a non-serialisable config is expected, and says so plainly."""
    fake, predictions = _fake_manager(tmp_path, {})

    with caplog.at_level("ERROR"):
        _write(fake, _ctx(orchestrator_config={"bad": object()}))

    assert "failed to write the emission manifest" in caplog.text
    assert "UNEXPECTED exception" not in caplog.text, (
        "an expected serialisation failure was reported as a defect in the writer"
    )
    assert not (predictions / MANIFEST_FILENAME).exists()


# ─── The predictions directory must be a DIRECTORY ──────────────────────────


def test_a_file_where_the_predictions_directory_belongs_is_refused_by_name(tmp_path, caplog):
    """⛔ `is_dir()`, not `exists()` — and `rmdir` cannot tell the two apart.

    The existing guard test deletes the directory, so `exists()` is false too and the mutation
    survives. A **file** at that path is what separates them, and it is also the shape the real
    defect took: a `MagicMock`'s `.exists()` returns a MagicMock, which is truthy, so under a
    mocked path manager the guard passed and `write_manifest` then `mkdir`-ed the mock's repr-
    derived path — 214 empty directories in the repository root, invisible to `git status`.

    ⇒ The assertion is on the MESSAGE, not merely on the absence of a manifest. Both readings end
    with nothing written; only one of them says the naming convention has moved.
    """
    fake, predictions = _fake_manager(tmp_path, {})
    predictions.rmdir()
    predictions.write_text("a file, not a directory")

    with caplog.at_level("ERROR"):
        _write(fake, _ctx())

    assert "it is not a directory" in caplog.text, (
        "a file at the predictions path was accepted as the output directory; the guard is a "
        "statement about a real filesystem, not about an object"
    )
    assert predictions.read_text() == "a file, not a directory", "the file was overwritten"


# ─── The ANCHOR: which copy of the package is being asked? ──────────────────
#
# ⛔⛔ A second audit, of commit `568a749`, found the tests above green under three further
# mutations — and the reason is the one this file was written to close, one level further out.
# `test_the_hydranet_entry_records_this_checkout_s_head` proves that *some* anchor inside this
# worktree was read; it cannot prove the anchor was the **package's own** directory, because under
# pytest the repository root is `sys.path[0]` and so the package directory, the repository root
# and the process CWD all resolve into the same worktree named `views-hydranet`. The guard was
# satisfied by the pytest layout rather than the production one, which is exactly the divergence
# `_repo_provenance` was fixed for (C-347).


def test_the_anchor_is_the_package_s_own_directory_not_the_process_cwd(tmp_path, monkeypatch):
    """⛔ `Path.cwd()` in place of the package directory left the full suite green.

    In production the two are unrelated: the package is a wheel under `site-packages` and the CWD
    is wherever `pod_run_model.sh` was invoked — a views-models checkout. Anchoring at the CWD
    would record **views-models'** HEAD under the name `views-hydranet`, which is the false
    provenance claim ADR-074 §2.1 exists to prevent, and it would do so on every pod run while
    reading correctly on every developer machine.

    ⇒ The test moves the package to a `site-packages`-shaped path inside a differently-named
    repository. The correct anchor lands there and refuses; the CWD anchor lands in this checkout
    and happily records a commit.

    ℹ️ One related mutation is NOT caught and should not be: anchoring one directory higher
    (`.parent.parent`). Measured — `views_hydranet/` holds no `.git` in the checkout and none in a
    `site-packages` install, so `git -C` walks up to the same toplevel from either. It is an
    equivalent mutant in every layout the package is actually installed in, and would stop being
    one only if somebody ran `git init` inside the package directory.
    """
    import views_hydranet

    foreign = _git_repo(tmp_path, "some-unrelated-checkout")
    installed = foreign / "venv" / "lib" / "site-packages" / "views_hydranet"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    monkeypatch.setattr(views_hydranet, "__file__", str(installed / "__init__.py"))

    repos = _provenance(tmp_path / "not_a_repo")

    assert repos["views-hydranet"].commit is None, (
        f"recorded {repos['views-hydranet'].commit!r} for a package installed under "
        f"{foreign.name!r} — provenance was read from somewhere other than the package's own "
        f"directory, so the commit belongs to whatever repository the process happened to be in"
    )
    assert repos["views-hydranet"].dirty is True, "unknown must degrade to dirty, not to clean"


def test_a_git_invocation_that_fails_is_unknown_rather_than_an_empty_revision(
    tmp_path, monkeypatch
):
    """⛔ `check=False` left the full suite green, and it fabricates a clean, known revision.

    Without `check`, a failing `rev-parse --show-toplevel` returns `""`; `Path("").resolve()` is
    the process CWD, whose basename under pytest *is* `views-hydranet`, so the name check passes.
    `rev-parse HEAD` then also fails quietly and the record becomes
    ``commit="", dirty=False`` — which reads as *clean checkout, revision known*, the strongest
    provenance claim the schema can make, from a repository that could not be read at all.

    ⚠️ The existing unknown-and-dirty test cannot see this: it anchors **views-models**, and the
    CWD's basename is not `views-models`, so that entry takes the honest branch either way.
    """
    import views_hydranet

    loose = tmp_path / "no_repository_here" / "views_hydranet"
    loose.mkdir(parents=True)
    (loose / "__init__.py").write_text("")
    monkeypatch.setattr(views_hydranet, "__file__", str(loose / "__init__.py"))

    recorded = _provenance(tmp_path / "not_a_repo")["views-hydranet"]

    assert recorded.commit is None, (
        f"recorded commit {recorded.commit!r} from a directory that is in no repository; an empty "
        f"string is not 'unknown', it is a revision field a reader will treat as read"
    )
    assert recorded.dirty is True, (
        "a repository that could not be read was recorded as CLEAN — the one reading the dirty "
        "flag exists to make impossible"
    )


def test_git_missing_from_the_path_is_an_environmental_failure_not_a_defect(
    tmp_path, monkeypatch, caplog
):
    """⛔ The `OSError` arm was credited and unreachable: no fixture could get to it.

    `git -C /nonexistent` *exits* 128, which is a `CalledProcessError` — a `SubprocessError`. The
    only way to raise `OSError` here is for the `git` executable itself to be absent — a slim pod,
    not a developer machine. Without that arm the `FileNotFoundError` escapes into
    `_write_emission_manifest`'s blanket clause and is reported as **a defect in the writer** —
    precisely the confusion the writer's two-clause split exists to prevent.
    """
    monkeypatch.setenv("PATH", str(tmp_path / "an_empty_bin"))

    with caplog.at_level("WARNING"):
        repos = _provenance(tmp_path)

    assert [r.commit for r in repos.values()] == [None, None], (
        "a revision was recorded with no git on PATH, so it did not come from version control"
    )
    assert all(r.dirty for r in repos.values()), "unknown must degrade to dirty"
    assert "could not read version control" in caplog.text, (
        "an absent `git` was not reported as an environmental failure; it would surface as an "
        "unexpected exception in the manifest writer, naming the wrong culprit"
    )


# ─── `quantities`: the prefix test is POSITIONAL ────────────────────────────


def test_a_gate_target_that_merely_contains_the_forecast_prefix_is_still_a_gate(tmp_path):
    """⛔ `LINEAR_PREFIX in target` left the full suite green: every fixture target has its prefix
    at position 0, so no test could tell `in` from `startswith`.

    A classification target whose *name* happens to contain `lr_` anywhere — `by_sb_colr_best` —
    would then be labelled `forecast`, and a reader joining the directory to its quantity would
    read the gate as the body. The prefix is a statement about where the name begins.
    """
    fake, predictions = _fake_manager(tmp_path, {})
    _write(fake, _ctx(all_targets=["lr_ged_sb", "by_sb_colr_best"]))

    quantities = read_manifest(predictions / MANIFEST_FILENAME).quantities

    assert quantities["by_sb_colr_best"] == "gate", (
        f"recorded {quantities}; the prefix was matched anywhere in the name instead of at its "
        f"start, so a gate target containing `lr_` is labelled as the body's forecast"
    )


# ─── The artifact digest is a digest, not a line of a file ───────────────────


def test_the_digest_sidecar_is_recorded_without_its_trailing_newline(tmp_path):
    """⚠️ Dropping `.strip()` left the full suite green, because the fixture wrote no newline.

    `sha256sum > file` ends with one, and so does any sidecar written by hand or by a pod script.
    The recorded digest would then be 65 characters with a `\\n` in it — not equal to the real
    digest under any comparison a reader would make, and the manifest exists to make exactly that
    comparison possible.
    """
    fake, predictions = _fake_manager(tmp_path, {})
    (tmp_path / "artifacts" / ARTIFACT).with_suffix(".pt.sha256").write_text("b" * 64 + "\n")

    _write(fake, _ctx())

    assert read_manifest(predictions / MANIFEST_FILENAME).artifact_sha256 == "b" * 64, (
        "the digest was recorded with the sidecar's trailing newline, so it compares equal to no "
        "digest anybody will compute"
    )
