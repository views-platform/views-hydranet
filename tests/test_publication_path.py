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

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

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


def _contract_module():
    """Import `.github/scripts/wheel_contract.py` as a module WITHOUT running `main()`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wheel_contract", CONTRACT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _contract_ast():
    import ast

    return ast.parse(CONTRACT.read_text())


class TestTheContractReadsTheCodeNotJustTheMetadata:
    """A wheel whose `.dist-info` is correct and whose Python tree is empty passes `twine check`,
    installs under `--no-deps`, and satisfies every metadata assertion. Demonstrated on a forged
    wheel in S6/#359. Importing the package is the only assertion that reads the code.

    The contract's checks are exercised IN-PROCESS here, not matched as text: a guard audit on
    #372 left `require()` with a `pass` body, a `require()` that exited 0 after printing FAILED,
    an emptied `REQUIRED_DEPENDENCIES`, a disabled purelib condition and a try/except around the
    import all green under the previous, substring-based versions of these tests."""

    def test_the_contract_imports_the_package_and_not_inside_a_try(self):
        """An `import_module(PACKAGE)` call that is NOT wrapped in try/except: on an empty wheel
        that call raising IS the contract firing. A try/except that swallowed it passed the
        text-matching version of this test with the import line intact."""
        import ast

        tree = _contract_ast()
        main = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        imports = [
            n
            for n in ast.walk(main)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "import_module"
        ]
        assert imports, "main() no longer imports the package — an empty wheel would pass"
        inside_try = [
            n
            for t in ast.walk(main)
            if isinstance(t, ast.Try)
            for n in ast.walk(t)
            if n in imports
        ]
        assert not inside_try, (
            "the package import is wrapped in try/except — its failure is swallowed"
        )

    def test_require_exits_non_zero_with_the_reason(self):
        mod = _contract_module()
        with pytest.raises(SystemExit) as exc:
            mod.require(False, "the reason")
        assert exc.value.code != 0 and exc.value.code is not None, "require() exited 0 on failure"
        assert "the reason" in str(exc.value.code)
        mod.require(True, "never raised")  # and it is not always-raise either

    def test_the_metadata_check_rejects_a_missing_dependency(self):
        mod = _contract_module()
        good = [f"{d} (>=1)" for d in mod.REQUIRED_DEPENDENCIES]
        mod.check_metadata(good, ["Repository, https://x"])  # the real shape passes
        for dropped in mod.REQUIRED_DEPENDENCIES:
            with pytest.raises(SystemExit, match=dropped):
                mod.check_metadata(
                    [r for r in good if dropped not in r], ["Repository, https://x"]
                )
        with pytest.raises(SystemExit, match="Project-URL"):
            mod.check_metadata(good, [])

    def test_required_dependencies_are_the_runtime_dependencies_in_pyproject(self):
        """`REQUIRED_DEPENDENCIES = ()` passed everything. Pinned against the source of truth."""
        import tomllib

        mod = _contract_module()
        deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
        names = {d.split(" ")[0].split(">")[0].split("(")[0].strip() for d in deps}
        assert set(mod.REQUIRED_DEPENDENCIES) == names, (
            f"the contract requires {sorted(mod.REQUIRED_DEPENDENCIES)} but pyproject declares "
            f"{sorted(names)} — one of them drifted"
        )

    def test_the_import_must_resolve_to_the_installed_wheel(self):
        """With a PYTHONPATH or .pth that puts the checkout ahead of site-packages, the import
        succeeds against the working tree and proves nothing. Exercised, not text-matched: the
        previous version passed with the condition `or True`-d and with `purelib = ""`."""
        mod = _contract_module()
        mod.check_installed_code(
            "venv/site-packages/views_hydranet/__init__.py",
            "venv/site-packages",
            ["HydranetManager"],
        )
        with pytest.raises(SystemExit, match="outside the installed site-packages"):
            mod.check_installed_code(
                "checkout/views_hydranet/__init__.py",
                "venv/site-packages",
                ["HydranetManager"],
            )
        with pytest.raises(SystemExit, match="public API changed"):
            mod.check_installed_code(
                "venv/site-packages/views_hydranet/__init__.py",
                "venv/site-packages",
                ["HydranetManager", "utils"],
            )

    def test_the_contract_does_not_depend_on_assert_statements(self):
        """Python strips `assert` under -O / PYTHONOPTIMIZE. A contract made of asserts is green
        in an optimised interpreter with nothing checked (C-329). An AST scan, not a prefix match:
        `assert(x), "m"` has no space after the keyword and slipped past `startswith`."""
        import ast

        offenders = [n.lineno for n in ast.walk(_contract_ast()) if isinstance(n, ast.Assert)]
        assert not offenders, f"assert statements in the wheel contract at lines {offenders}"


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


def _third_party_imports() -> dict[str, set[str]]:
    """Every non-stdlib, non-first-party module `views_hydranet` imports, at ANY nesting depth.

    Depth matters: `sklearn` is imported inside `TrainingForensics._calculate_cls_metric`, so a
    module-top-only scan would have missed the one undeclared import whose failure arrives
    mid-training rather than at import.
    """
    roots: dict[str, set[str]] = {}
    for path in sorted((REPO / "views_hydranet").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which is first-party by construction.
                names = [node.module] if (node.level == 0 and node.module) else []
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in sys.stdlib_module_names or root == "views_hydranet":
                    continue
                roots.setdefault(root, set()).add(str(path.relative_to(REPO)))
    return roots


def _partition(
    found: dict[str, set[str]], declared: set[str]
) -> tuple[dict[str, set[str]], set[str]]:
    """Split the imports into (undeclared, unresolvable) against a declared-distribution set.

    Separate from the test that uses it only so the `unresolvable` branch is reachable with a
    synthetic input — it is the branch that must never be quietly widened into a skip.
    """
    undeclared: dict[str, set[str]] = {}
    unresolvable: set[str] = set()
    for module, files in found.items():
        providers = packages_distributions().get(module)
        if not providers:
            unresolvable.add(module)
            continue
        if not any(p.lower().replace("_", "-") in declared for p in providers):
            undeclared[module] = files
    return undeclared, unresolvable


class TestEveryImportIsDeclared:
    """The guard chain derived every link from `pyproject`, and so could not see this class.

    `REQUIRED_DEPENDENCIES` is pinned to `pyproject` (above), and the wheel metadata is pinned to
    `REQUIRED_DEPENDENCIES`. Every link resolves back to the declaration — never to the imports,
    which is where the truth is. So an import that nobody declared was invisible to all of it, and
    the class shipped **six times**: `wandb` (undeclared until #381), `pandas` (undeclared through
    releases 0.1.0–0.1.3), then `numpy`, `pydantic`, `tqdm` and `sklearn`, all four found on
    2026-10-20 by walking the import graph rather than re-reading the declaration.

    Each one resolved at install time anyway, through some other package's transitive choice —
    which is why nothing failed, and why nothing caught it. `sklearn` arrived via
    `views-pipeline-core -> viewser -> pyod -> scikit-learn`: three hops, the third an
    outlier-detection package this repo never imports.

    This is a permanent guard in a shared file, which the design preferences say to avoid in favour
    of a bounded local change. It is here because the bounded local change **is** the declaration,
    and the declaration was made correctly five times while the defect kept recurring: the thing
    that recurs is nobody checking. Six instances and four published wheels is the overhead earned.

    What it does NOT prove: that a declared range is *right*, or that a declared dependency is
    still used. Only that nothing load-bearing is missing.
    """

    def _declared(self) -> set[str]:
        import tomllib

        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        specs = list(project["dependencies"])
        for extra in project.get("optional-dependencies", {}).values():
            specs.extend(extra)
        return {
            re.split(r"[\s\[\(><=!;~]", spec.strip())[0].lower().replace("_", "-")
            for spec in specs
        }

    def test_the_scan_actually_found_the_imports(self):
        """Without this, a walker that silently returns {} makes the test below vacuously green —
        the C-329 shape. Asserted on torch because it is the one import that cannot go away."""
        found = _third_party_imports()
        assert len(found) >= 5, (
            f"the import scan found only {sorted(found)} — the walker is broken"
        )
        assert len(found["torch"]) > 40, (
            f"torch was found in {len(found.get('torch', ()))} files; it is imported by most of "
            "the package, so a low count means the scan is missing whole files"
        )
        # torch cannot prove the scan reaches NESTED imports — it is module-top in every one of
        # those files, so a depth-0-only walker still finds it in >40. `sklearn` is imported at
        # function level and nowhere else in the package, which makes it the one discriminator:
        # restricting the walker to `tree.body` was a mutation that left this class green until
        # this assertion existed, and it is the mutation that re-hides the sklearn instance.
        assert "sklearn" in found, (
            "sklearn was not found. It is imported inside "
            "`TrainingForensics._calculate_cls_metric` and at no module top, so its absence means "
            "the scan is not descending into function bodies — the depth this guard needs"
        )

    def test_an_uninstalled_import_is_reported_rather_than_passed_over(self):
        """The `unresolvable` branch had no test, and neutering it (`unresolvable = set()`) left
        the class green — a surviving mutation. An uninstalled import cannot be checked against
        the declaration at all, so passing over it is the one outcome that must be impossible:
        an incomplete environment and the next undeclared import look identical from here."""
        undeclared, unresolvable = _partition(
            {"a_module_no_distribution_provides": {"synthetic"}}, self._declared()
        )
        assert unresolvable == {"a_module_no_distribution_provides"}
        assert not undeclared, "an unresolvable module must not be silently counted as declared"

    def test_every_third_party_import_is_a_declared_dependency(self):
        undeclared, unresolvable = _partition(_third_party_imports(), self._declared())

        assert not unresolvable, (
            f"cannot resolve a distribution for {sorted(unresolvable)} — not installed in this "
            "environment, so whether they are declared cannot be checked"
        )
        assert not undeclared, "undeclared load-bearing imports:\n" + "\n".join(
            f"  {module}: {sorted(files)}" for module, files in sorted(undeclared.items())
        )
