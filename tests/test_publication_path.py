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
import subprocess
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
    # Git-tracked files only, not `rglob`. `views_hydranet/wandb/` is a gitignored runtime dump
    # that lives INSIDE the package directory (wandb's code-save writes `run-*/tmp/code/`), so a
    # glob would treat whatever lands there as an import site of this package — reporting its
    # imports as undeclared dependencies, or raising `SyntaxError` out of the test on a file that
    # does not parse.
    tracked = subprocess.run(
        ["git", "ls-files", "views_hydranet/*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert len(tracked) > 50, f"git ls-files returned {len(tracked)} package files — scan is blind"

    roots: dict[str, set[str]] = {}
    for path in sorted(REPO / p for p in tracked):
        # Explicit encoding: 57 of the 67 package files carry non-ASCII docstrings, so the
        # platform default decodes them wrongly wherever it is not UTF-8.
        tree = ast.parse(path.read_text(encoding="utf-8"))
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

    ⛔ The declared set is checked BEFORE the distribution is resolved, and the order is the whole
    correctness of this function. Resolving first made the guard RED on a correct repo:
    `matplotlib` is imported inside `VisualDiagnostics._load_mpl` and declared in the `viz` extra,
    which neither `pip install .` nor `environment.yml`'s pip list installs. It therefore has no
    resolvable distribution in the environment CI actually builds, and a resolve-first order filed
    a correctly declared dependency as `unresolvable`. CI passed only because `pyod` happens to
    pull matplotlib in unpinned — precisely the accidental transitive guarantor this guard exists
    to catch.
    """
    # One call: `packages_distributions()` is not cached and re-walks every distribution on
    # sys.path, so calling it per module repeated a full metadata scan ~10 times.
    providers_by_module = packages_distributions()
    undeclared: dict[str, set[str]] = {}
    unresolvable: set[str] = set()
    for module, files in found.items():
        if module.lower().replace("_", "-") in declared:
            # The import name is the distribution name. True for all but `sklearn`, and the only
            # check available for a declared dependency that is not installed.
            continue
        providers = providers_by_module.get(module)
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

    What it does NOT prove: that a declared dependency is still used, or that a declared dependency
    RANGE is right. Only that nothing load-bearing is missing. ⚠️ One half of the range gap is now
    covered by `TestRequiresPythonIsServiceable` below — `requires-python`, which is also a
    declared range and was also never validated. The dependency ranges themselves still are not.
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
        assert len(found.get("torch", ())) > 40, (
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

    def test_a_declared_dependency_that_is_not_installed_still_counts_as_declared(self):
        """`matplotlib` is the live case: imported inside `VisualDiagnostics._load_mpl`, declared
        in the `viz` extra, and installed by neither `pip install .` nor `environment.yml`'s pip
        list. A resolve-before-declared order filed it as `unresolvable` and turned the guard red
        on a correct repo — green in CI only because `pyod` pulls matplotlib in unpinned.

        Asserted on a name that is declared but certainly absent from site-packages, so the test
        does not depend on whether matplotlib happens to be installed where it runs."""
        declared = self._declared() | {"a-declared-but-absent-distribution"}
        undeclared, unresolvable = _partition(
            {"a_declared_but_absent_distribution": {"synthetic"}}, declared
        )
        assert not unresolvable, "a declared dependency must not need to be installed to pass"
        assert not undeclared

    def test_every_third_party_import_is_a_declared_dependency(self):
        undeclared, unresolvable = _partition(_third_party_imports(), self._declared())

        assert not unresolvable, (
            f"cannot resolve a distribution for {sorted(unresolvable)} — not installed in this "
            "environment, so whether they are declared cannot be checked"
        )
        assert not undeclared, "undeclared load-bearing imports:\n" + "\n".join(
            f"  {module}: {sorted(files)}" for module, files in sorted(undeclared.items())
        )


class TestRequiresPythonIsServiceable:
    """`requires-python` is a declared range too, and nothing validated it either.

    ⛔ Found 2026-10-20 by `/falsify` on the v0.1.4 release surface, in the residue
    `TestEveryImportIsDeclared` above had already confessed to. The ceiling was `<3.15`, so the
    wheel advertised 3.11 through 3.14 — while `viewser` 6.6.4's `pandas<2.0.0` against our own
    `pandas>=1.5.3` leaves **1.5.3 as the only satisfying release**, pinning pandas to one version
    whose last wheel is cp311. Three of the four advertised interpreters had no wheel for a
    dependency the package cannot start without, in every release 0.1.0 through 0.1.3.

    It is the same shape as the imports above with the arrow turned around: there, a real import
    had no declaration; here, a declaration had no reality behind it. Both were invisible for the
    same reason — every check derived from `pyproject` rather than from anything that could
    contradict it.

    ⚠️ **This pins a MEASURED CONSTANT, and that is a real limitation, not a formality.** It
    enforces the current fix and makes widening the ceiling a deliberate edit that forces a
    re-read. It does **not** track pandas' wheel matrix: if upstream lifts `pandas<2.0.0`, this
    keeps passing while the ceiling could safely widen — a false negative, in the harmless
    direction. The honest guard would resolve the tree and query wheel availability per
    interpreter, which needs the network; this suite has **zero** network tests, and a guard gated
    behind an env var is one that never runs in CI (C-324, the shape this repository keeps paying
    for). The trade is named rather than hidden.

    `packaging` is safe to import here: `pytest` declares `packaging>=22` as a hard requirement, so
    it is present wherever this suite runs. One hop from a declared test dependency — deliberately
    checked, because an earlier draft of this guard avoided it on the mistaken belief that it was
    an accidental guarantor of exactly the kind C-346 is about.
    """

    # The highest CPython minor for which EVERY pin of the currently resolved tree publishes a
    # wheel. Measured 2026-10-20 against published PyPI metadata; the binding constraint is
    # `pandas==1.5.3` (cp311 is its last wheel), and numpy==1.26.4 is the next tightest (cp312).
    HIGHEST_WHEEL_SUPPORTED_MINOR = 11

    def test_requires_python_advertises_no_interpreter_without_a_pandas_wheel(self):
        """Fail-closed by construction: a ceiling that is deleted or widened to `<4` makes
        `contains()` true for every later minor, so it fails without a special case for it."""
        import tomllib

        from packaging.specifiers import SpecifierSet

        declared = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
            "requires-python"
        ]
        spec = SpecifierSet(declared)
        # 30 rather than the current ceiling: the point is to catch a RAISED or absent ceiling, so
        # the probe must look past wherever it sits today.
        advertised = [
            m for m in range(self.HIGHEST_WHEEL_SUPPORTED_MINOR, 30) if spec.contains(f"3.{m}.0")
        ]
        assert advertised, f"requires-python {declared!r} admits no 3.x at all — read it again"

        unserviceable = [m for m in advertised if m > self.HIGHEST_WHEEL_SUPPORTED_MINOR]
        assert not unserviceable, (
            f"requires-python {declared!r} advertises Python "
            f"{', '.join(f'3.{m}' for m in unserviceable)}, but the locked dependency tree has no "
            f"wheel above 3.{self.HIGHEST_WHEEL_SUPPORTED_MINOR}: `pandas==1.5.3` is pinned to "
            f"exactly one release by viewser's `pandas<2.0.0` against our `pandas>=1.5.3`, and "
            f"cp311 is its last wheel. Published metadata would make a capability claim the tree "
            f"cannot deliver. Fix the ceiling, or lift the upstream pandas bound first and "
            f"re-measure HIGHEST_WHEEL_SUPPORTED_MINOR."
        )
