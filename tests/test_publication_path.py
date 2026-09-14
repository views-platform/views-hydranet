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
    """`workflow_dispatch` takes no branch restriction: whoever dispatches picks the ref. Combined
    with `--check-url`, a rehearsal at the release version made the real Release skip its own
    upload and validate the stale wheel. The `.devN` stamp makes the collision impossible."""

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


class TestNoPathPublishesWhileSkippingTheTagGuard:
    """C-341: the tag guard and the real publish must carry the *identical* condition. The guide
    states this as an invariant; nothing enforced it until now."""

    def test_the_tag_guard_and_the_real_publish_share_one_condition(self):
        text = PUBLISH.read_text()
        release_only = text.count("if: github.event_name == 'release'")
        assert release_only == 2, (
            f"expected exactly two release-only steps (the tag guard and the real PyPI publish); "
            f"found {release_only}. A publish step without the guard, or a guard without the "
            "publish, reopens C-341."
        )

    def test_no_dispatch_input_can_reach_real_pypi(self):
        assert "inputs." not in PUBLISH.read_text(), (
            "a workflow_dispatch input is back. The 'Stop after TestPyPI' checkbox was removed "
            "because unticking it published an untagged version permanently (C-341)."
        )
