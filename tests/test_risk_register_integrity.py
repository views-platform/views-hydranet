"""
Risk register structural integrity tests.

Ensures the register's header counts, section placement, and formatting
stay consistent as entries are added, resolved, and moved.

Green taxonomy only — all tests verify structural invariants (ADR-005).
"""

import re
from pathlib import Path

REGISTER_PATH = Path(__file__).parent.parent / "reports" / "technical_risk_register.md"


def _read_register() -> str:
    return REGISTER_PATH.read_text()


def _parse_header(text: str) -> dict[str, int]:
    """Extract Total/Open/Resolved counts from the header table."""
    counts = {}
    for label in ("Total Concerns", "Open Concerns", "Resolved Concerns"):
        m = re.search(rf"\|\s*{label}\s*\|\s*(\d+)\s*\|", text)
        assert m, f"Header missing '{label}' row"
        counts[label] = int(m.group(1))
    return counts


def _extract_section(text: str, heading: str) -> str:
    """Extract text between a ## heading and the next ## heading."""
    pattern = rf"(^## {re.escape(heading)}\s*$)(.*?)(?=^## |\Z)"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    assert m, f"Section '## {heading}' not found in register"
    return m.group(2)


def _entry_headers_in(section_text: str) -> list[tuple[str, bool]]:
    """Return list of (entry_id, is_resolved) from ### headers in a section."""
    entries = []
    for m in re.finditer(r"^### (C-\d+|D-\d+):.*?(— RESOLVED)?$", section_text, re.MULTILINE):
        entry_id = m.group(1)
        is_resolved = m.group(2) is not None
        entries.append((entry_id, is_resolved))
    return entries


# ---------------------------------------------------------------------------
# GREEN TEAM — structural invariants
# ---------------------------------------------------------------------------
class TestGreen:
    def test_header_counts_match_actual_entries(self):
        """Header Total/Open/Resolved must match actual entry counts."""
        text = _read_register()
        header = _parse_header(text)

        all_ids = set()
        for m in re.finditer(r"^### (C-\d+):", text, re.MULTILINE):
            all_ids.add(m.group(1))

        open_section = _extract_section(text, "Open Concerns")
        open_entries = _entry_headers_in(open_section)
        open_ids = {eid for eid, resolved in open_entries if not resolved}

        resolved_section = _extract_section(text, "Resolved Concerns")
        resolved_entries = _entry_headers_in(resolved_section)
        resolved_ids = {eid for eid, _ in resolved_entries}

        resolved_in_open = {eid for eid, resolved in open_entries if resolved}
        open_count = len(open_ids)
        resolved_count = len(resolved_ids) + len(resolved_in_open)

        assert header["Total Concerns"] == len(all_ids), (
            f"Header Total={header['Total Concerns']} but found {len(all_ids)} concern entries"
        )
        assert header["Open Concerns"] == open_count, (
            f"Header Open={header['Open Concerns']} but found {open_count} open entries"
        )
        assert header["Resolved Concerns"] == resolved_count, (
            f"Header Resolved={header['Resolved Concerns']} "
            f"but found {resolved_count} resolved entries"
        )

    def test_no_resolved_entries_in_open_section(self):
        """Entries marked RESOLVED must not sit in the Open Concerns section."""
        text = _read_register()
        open_section = _extract_section(text, "Open Concerns")
        misplaced = [
            eid for eid, resolved in _entry_headers_in(open_section) if resolved
        ]
        assert misplaced == [], (
            f"Resolved entries misplaced in Open section: {misplaced}"
        )

    def test_no_open_entries_in_resolved_section(self):
        """Entries without RESOLVED marker must not sit in the Resolved section."""
        text = _read_register()
        resolved_section = _extract_section(text, "Resolved Concerns")
        misplaced = [
            eid for eid, resolved in _entry_headers_in(resolved_section) if not resolved
        ]
        assert misplaced == [], (
            f"Open entries misplaced in Resolved section: {misplaced}"
        )

    def test_no_duplicate_ids(self):
        """Every C-xx / D-xx ID must appear exactly once as a ### header."""
        text = _read_register()
        ids = [m.group(1) for m in re.finditer(r"^### (C-\d+|D-\d+):", text, re.MULTILINE)]
        dupes = [eid for eid in ids if ids.count(eid) > 1]
        assert dupes == [], f"Duplicate entry IDs: {set(dupes)}"

    def test_section_ordering(self):
        """Sections must appear in governance order."""
        text = _read_register()
        expected_order = [
            "Tier Definitions",
            "Open Concerns",
            "Disagreements",
            "Resolved Concerns",
            "Register Conventions",
        ]
        positions = []
        for heading in expected_order:
            m = re.search(rf"^## {re.escape(heading)}", text, re.MULTILINE)
            assert m, f"Missing section: ## {heading}"
            positions.append(m.start())
        assert positions == sorted(positions), (
            f"Section order wrong. Expected: {expected_order}"
        )
