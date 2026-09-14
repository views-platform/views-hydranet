"""The publication path: contracts that no other test can reach.

`.github/workflows/publish_package.yml` has never executed, and pytest cannot run it. Everything it
promises was, until S6/#359, asserted only in prose — which is the **C-303** shape, prose asserting
a guard the code does not implement, and this register's most-repeated defect.

These tests read the workflow files as text rather than parsing them: PyYAML is not a declared
dependency of this project, and a test that skips when its parser is absent is a **C-329** test
that cannot fail. The substrings below were each chosen so that the realistic regression — deleting
the step, re-inlining the check, swapping the version output — changes them.

What they do NOT prove: that the workflow runs. Only a live dispatch does that, and it is held
behind #351 (the S10 gate).
"""

import re
from pathlib import Path

REPO = Path(__file__).parent.parent
PUBLISH = REPO / ".github" / "workflows" / "publish_package.yml"
CI = REPO / ".github" / "workflows" / "ci.yml"
CONTRACT = REPO / ".github" / "scripts" / "wheel_contract.py"


class TestTheWheelContractIsOneFile:
    """CI and the release job asserted the wheel contract separately, and had already drifted:
    only CI checked `Project-URL`. One file makes the drift impossible rather than detectable."""

    def test_both_workflows_call_the_shared_script(self):
        for workflow in (PUBLISH, CI):
            assert ".github/scripts/wheel_contract.py" in workflow.read_text(), (
                f"{workflow.name} no longer calls the shared wheel contract — if it was "
                "re-inlined, the two copies will drift again (S6/#359)"
            )

    def test_neither_workflow_keeps_an_inline_copy(self):
        for workflow in (PUBLISH, CI):
            assert "Requires-Dist" not in workflow.read_text(), (
                f"{workflow.name} asserts the wheel contract inline again; it belongs in "
                f"{CONTRACT.relative_to(REPO)}, which both workflows run"
            )


class TestTheContractReadsTheCodeNotJustTheMetadata:
    """A wheel whose `.dist-info` is correct and whose Python tree is empty passes `twine check`,
    installs under `--no-deps`, and satisfies every metadata assertion. Demonstrated on a forged
    wheel in S6/#359. Importing the package is the only assertion that reads the code."""

    def test_the_contract_imports_the_package(self):
        """Matched on an executable line, not a substring: the first draft of this test passed
        with the import commented out, because the words survived in the comment."""
        lines = [ln.strip() for ln in CONTRACT.read_text().splitlines()]
        assert any(ln.startswith("import views_hydranet") for ln in lines), (
            "the wheel contract no longer imports the package, so a wheel containing zero Python "
            "modules would pass it (S6/#359)"
        )

    def test_the_import_must_resolve_to_the_installed_wheel(self):
        """Both workflows run from the repo root, where `views_hydranet/` is on the path. Without
        this assertion the import succeeds against the working tree and proves nothing."""
        assert "assert views_hydranet.__file__.startswith(purelib)" in CONTRACT.read_text(), (
            "the wheel contract no longer checks that the import resolved inside site-packages — "
            "it would pass on an empty wheel by importing the source checkout instead"
        )


class TestARehearsalCannotOccupyAReleaseVersion:
    """`workflow_dispatch` takes no branch restriction: whoever dispatches picks the ref. A
    rehearsal at the release version burned that version on TestPyPI, and the real Release then
    died at the TestPyPI step on uv's hash-mismatch check (verified: skip on identical bytes,
    exit 2 on different bytes) — a blocked release with a slot it could never reuse. The `.devN`
    stamp makes the collision impossible."""

    def test_a_non_release_run_stamps_a_throwaway_version(self):
        """Pinned to the step, not to the suffix string: the suffix also appears in the version
        step, so the first draft of this test stayed green with the stamp step deleted."""
        text = PUBLISH.read_text()
        assert "- name: Rehearsal — stamp a throwaway .devN version" in text, (
            "the .devN stamp step is gone — a manual dispatch can again upload at the release "
            "version and burn it on TestPyPI permanently (S6/#359)"
        )
        stamp = text.split("- name: Rehearsal — stamp a throwaway .devN version", 1)[1]
        stamp = stamp.split("- name:", 1)[0]
        assert "if: github.event_name != 'release'" in stamp, (
            "the stamp step is no longer rehearsal-only — it would rewrite a release's version"
        )
        assert ".dev${{ github.run_number }}" in stamp, (
            "the stamp no longer writes a run-unique .devN, so two rehearsals can collide"
        )

    def test_the_install_back_verifies_the_version_that_was_uploaded(self):
        """`steps.ver.outputs.version` is the pyproject version; `publish_version` is what the
        upload actually used. On a rehearsal they differ, and installing the former would 404."""
        assert 'steps.ver.outputs.publish_version }}"' in PUBLISH.read_text(), (
            "the install-back no longer pins the version that was uploaded"
        )


def _steps(text: str) -> dict[str, str]:
    """The workflow's steps, keyed by `- name:`, each with its own block of text. Assertions are
    made per step, not on substring counts across the file: a `count == 2` pin passed with the
    condition moved to a different step, commented out, or its `==`/`!=` swapped (#372 review)."""
    parts = text.split("- name: ")[1:]
    return {part.split("\n", 1)[0].strip(): part for part in parts}


class TestNoPathPublishesWhileSkippingTheTagGuard:
    """C-341: the real PyPI publish, the tag guard and the PyPI-version guard carry the
    *identical* release-only condition. The guide states this as an invariant; this enforces
    it per step."""

    RELEASE_ONLY = "if: github.event_name == 'release'"

    def test_the_real_publish_is_release_only(self):
        steps = _steps(PUBLISH.read_text())
        # an executed `uv publish` line, not one quoted in a comment
        publish = [
            n
            for n, b in steps.items()
            if re.search(r"^\s+(run: )?uv publish", b, re.M) and "--publish-url" not in b
        ]
        assert publish == ["Publish to PyPI (Trusted Publishing — no token)"], (
            f"expected exactly one real-PyPI publish step, found {publish}"
        )
        block = steps[publish[0]]
        assert self.RELEASE_ONLY in block and f"# {self.RELEASE_ONLY}" not in block, (
            "the real PyPI publish step lost its release-only condition — a workflow_dispatch "
            "rehearsal would `uv publish` to real PyPI (C-341)"
        )

    def test_both_guards_carry_the_same_release_only_condition(self):
        steps = _steps(PUBLISH.read_text())
        for name in (
            "Guard — pyproject version must equal the tag being released",
            "Guard — pyproject version must be newer than what is on PyPI",
        ):
            assert name in steps, f"step {name!r} is gone"
            assert self.RELEASE_ONLY in steps[name], (
                f"{name!r} is no longer release-only. The tag guard has no tag to compare on a "
                "dispatch; the PyPI-version guard fails every rehearsal from an un-bumped ref "
                "once a version is published."
            )

    def test_the_testpypi_publish_runs_on_both_events(self):
        """The rehearsal is the point of the workflow; it must never be conditioned away."""
        steps = _steps(PUBLISH.read_text())
        block = steps["Publish to TestPyPI (Trusted Publishing — no token)"]
        assert "--publish-url https://test.pypi.org/legacy/" in block
        assert "if:" not in block, "the TestPyPI publish must run on release AND dispatch"

    def test_no_dispatch_input_can_reach_real_pypi(self):
        assert "inputs." not in PUBLISH.read_text(), (
            "a workflow_dispatch input is back. The 'Stop after TestPyPI' checkbox was removed "
            "because unticking it published an untagged version permanently (C-341)."
        )
