"""
Falsification audit: "the repo is clean" (2026-04-20).

Verdict: FALSIFIED (4 hard, 2 soft falsifications).

Each test below encodes a specific falsifying observation. Tests are RED
stubs — they FAIL to document the gap. When the gap is fixed, flip to GREEN.
"""

import re
import subprocess
from pathlib import Path

# ── F4-01: HARD — Uncommitted changes ────────────────────────────────────────


class TestF4_01_UncommittedChanges:
    """25 modified files and 7 untracked files sit uncommitted."""

    def test_git_working_tree_is_clean(self):
        """
        F4-01: 'Clean repo' requires a clean working tree. git status
        shows 25 modified files and 7 untracked files spanning source,
        tests, config, and reports.
        """
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.stdout.strip() == "", (
            f"HARD FALSIFICATION F4-01: Working tree is not clean.\n{result.stdout[:500]}"
        )


# ── F4-02: HARD — Lint and format violations ─────────────────────────────────


class TestF4_02_LintViolations:
    """33 ruff errors, 61 files need reformatting."""

    def test_ruff_check_passes(self):
        """
        F4-02a: ruff check reports 33 errors across 9 files, including
        unused imports, ambiguous variable names, and line length violations.
        """
        result = subprocess.run(
            ["ruff", "check", "."],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, (
            f"HARD FALSIFICATION F4-02a: ruff check found errors.\n{result.stdout[:500]}"
        )

    def test_ruff_format_passes(self):
        """
        F4-02b: ruff format --check reports 61 files needing reformatting.
        CI lint job runs this check — it would fail on push.
        """
        result = subprocess.run(
            ["ruff", "format", "--check", "."],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, (
            f"HARD FALSIFICATION F4-02b: {result.stdout.count('Would reformat')} "
            f"files need reformatting."
        )


# ── F4-03: SOFT — Dead code (unused imports/variables) ───────────────────────


class TestF4_03_DeadCode:
    """12 unused imports/variables across 9 files."""

    def test_no_unused_imports_or_variables(self):
        """
        F4-03: ruff --select F401,F841 finds 12 instances of dead code:
        unused imports (inspect, numpy, pytest, torch, importlib,
        MultiTaskLoss) and unused variables (n_reg, n_cls).
        """
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "F401,F841"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, (
            f"SOFT FALSIFICATION F4-03: Dead code found.\n{result.stdout[:500]}"
        )


# ── F4-04: HARD — Register header count mismatch ─────────────────────────────


class TestF4_04_RegisterAccuracy:
    """Header says 69 total / 19 open but actual counts differ."""

    def test_register_header_matches_actual_counts(self):
        """
        F4-04: Register header claims 69 total concerns, 19 open, 50
        resolved. Actual grep count: 68 entries (C-34 missing), 18 open,
        50 resolved. Off-by-one in total and open.
        """
        register = (
            Path(__file__).parent.parent / "reports" / "technical_risk_register.md"
        ).read_text()

        import re

        entries = re.findall(r"^### C-\d+", register, re.MULTILINE)

        # Count from headers
        total_match = re.search(r"Total Concerns\s*\|\s*(\d+)", register)
        open_match = re.search(r"Open Concerns\s*\|\s*(\d+)", register)
        resolved_match = re.search(r"Resolved Concerns\s*\|\s*(\d+)", register)

        header_total = int(total_match.group(1))
        header_open = int(open_match.group(1))
        header_resolved = int(resolved_match.group(1))

        actual_total = len(entries)
        actual_resolved = sum(
            1 for e in re.findall(r"^### C-\d+.*RESOLVED", register, re.MULTILINE)
        )
        actual_open = actual_total - actual_resolved

        assert header_total == actual_total, f"Total: header={header_total}, actual={actual_total}"
        assert header_open == actual_open, f"Open: header={header_open}, actual={actual_open}"
        assert header_resolved == actual_resolved, (
            f"Resolved: header={header_resolved}, actual={actual_resolved}"
        )


# ── F4-06: SOFT — Double separators in register ──────────────────────────────


class TestF4_06_DoubleSeparators:
    """Structural artifacts: consecutive --- separators with only blank lines between."""

    def test_no_double_separators_in_register(self):
        """
        F4-06: Two instances of consecutive --- separators (lines 91/93
        and 369/371) indicate deleted or missing entries leaving
        formatting debris.
        """
        register = (
            Path(__file__).parent.parent / "reports" / "technical_risk_register.md"
        ).read_text()

        lines = register.splitlines()
        doubles = []
        i = 0
        while i < len(lines):
            if lines[i].strip() == "---":
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and lines[j].strip() == "---":
                    doubles.append(i + 1)
            i += 1

        assert len(doubles) == 0, (
            f"SOFT FALSIFICATION F4-06: {len(doubles)} double separator(s) at line(s) {doubles}"
        )


# ── F4-07: HARD — Tracked code depending on untracked `reports/` files ───────


class TestF4_07_UntrackedDependencies:
    """A tracked test or tool must never depend on a file that is not in the repository.

    `reports/` is gitignored, but the scorers under it are force-tracked (`git add -f`) so their
    tests run in CI. When one is missed, nothing goes red: the dependent test carries a
    `pytest.skip(allow_module_level=True)` guard and silently reports success while measuring
    nothing. That has now happened three times — the Epic #263 ruler shipped with zero portable
    coverage, and `activation_metrics.py` / `score_v2_horizons.py` were both unreachable from a
    clean clone while their tests "passed". The second of those is loaded at runtime by
    `rescore_v2.py`, a tracked driver, so the shipped ruler could not run in a fresh checkout.

    The rule is derived, not listed, so a new dependency is covered without anyone remembering to
    add it here. A prose reference counts as a dependency too: `tail_index.py` cites
    `s5_tail.py::gpd_xi` as the remedy for C-284, which is a dead pointer for every reader but the
    author if the file is not in the repo.

    **Known limit:** it can only flag a file that exists on the machine running it, so it catches
    the mistake where it is made (authoring) rather than in CI, where an untracked file is simply
    absent and indistinguishable from one that was never there.
    """

    @staticmethod
    def _tracked(root: Path) -> set:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, cwd=root, check=True
        )
        return set(out.stdout.split())

    @staticmethod
    def _referenced_report_paths(text: str) -> set:
        """Literal `reports/.../x.py` paths appearing in a source file."""
        return set(re.findall(r"reports/[\w./-]+?\.py", text))

    def test_no_tracked_file_depends_on_an_untracked_report_tool(self):
        """F4-07: every `reports/*.py` reachable from tracked code must itself be tracked."""
        root = Path(__file__).parent.parent
        tracked = self._tracked(root)
        missing = []

        depends_on_reports = [
            rel
            for rel in sorted(tracked)
            if rel.endswith(".py") and (rel.startswith(("tests/", "scripts/")) or "/tools/" in rel)
        ]
        for rel in depends_on_reports:
            src = (root / rel).read_text()

            # A dossier tool named by path — how tests and drivers reach one.
            for dep in self._referenced_report_paths(src):
                if dep not in tracked and (root / dep).exists():
                    missing.append(f"{rel} -> {dep}")

            # A tool importing a sibling from its own `tools/` dir after a sys.path insert —
            # a plain `from x import y`, so the path form above cannot see it.
            if "/tools/" in rel:
                for mod in re.findall(r"^\s*from\s+(\w+)\s+import\s", src, re.M):
                    sibling = str(Path(rel).parent / f"{mod}.py")
                    if (root / sibling).exists() and sibling not in tracked:
                        missing.append(f"{rel} -> {sibling} (sibling import)")

        assert not missing, (
            "HARD FALSIFICATION F4-07: tracked code depends on untracked `reports/` file(s). "
            "Their guard tests will skip in CI and report success while measuring nothing. "
            "Fix with `git add -f <path>`.\n  " + "\n  ".join(sorted(set(missing)))
        )
