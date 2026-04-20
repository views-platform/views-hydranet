"""
CIC Section 6 drift detection (C-55).

Ensures CIC "Failure Modes and Loudness" sections stay in sync with
actual validators and error paths in source code.
"""

import re
from pathlib import Path

CIC_DIR = Path(__file__).parent.parent / "docs" / "CICs"


def _read_section6(cic_name: str) -> str:
    """Extract Section 6 text from a CIC markdown file."""
    path = CIC_DIR / f"{cic_name}.md"
    text = path.read_text()
    match = re.search(
        r"## 6\. Failure Modes.*?\n(.*?)(?=\n## \d|\Z)",
        text,
        re.DOTALL,
    )
    assert match, f"Section 6 not found in {path}"
    return match.group(1)


class TestHydraNetConfigSection6:
    """HydraNetConfig CIC must document all field_validators."""

    def test_documents_loss_reg_validator(self):
        s6 = _read_section6("HydraNetConfig")
        assert "loss_reg" in s6.lower(), (
            "CIC drift: HydraNetConfig Section 6 missing loss_reg validator (added in C-05)"
        )

    def test_documents_loss_class_validator(self):
        s6 = _read_section6("HydraNetConfig")
        assert "loss_class" in s6.lower(), (
            "CIC drift: HydraNetConfig Section 6 missing loss_class validator (added in C-05)"
        )

    def test_documents_aggregate_method_validator(self):
        s6 = _read_section6("HydraNetConfig")
        assert "aggregate_method" in s6.lower(), (
            "CIC drift: HydraNetConfig Section 6 missing aggregate_method validator"
        )


class TestIntegrityGuardianSection6:
    """IntegrityGuardian CIC must document current threshold and monitor_numpy."""

    def test_documents_prediction_ceiling_1000(self):
        s6 = _read_section6("IntegrityGuardian")
        assert "1,000" in s6 or "1000" in s6, (
            "CIC drift: IntegrityGuardian Section 6 still says 10,000 but code uses 1,000 (C-51)"
        )

    def test_documents_monitor_numpy(self):
        s6 = _read_section6("IntegrityGuardian")
        assert "numpy" in s6.lower() or "monitor_numpy" in s6, (
            "CIC drift: IntegrityGuardian Section 6 missing monitor_numpy method"
        )


class TestModelArtifactFetcherSection6:
    """ModelArtifactFetcher CIC must document SHA-256, legacy, and state_dict behavior."""

    def test_documents_sha256_integrity(self):
        s6 = _read_section6("ModelArtifactFetcher")
        assert "sha256" in s6.lower() or "integrity" in s6.lower(), (
            "CIC drift: ModelArtifactFetcher Section 6 missing SHA-256 integrity check"
        )

    def test_documents_legacy_deprecation(self):
        s6 = _read_section6("ModelArtifactFetcher")
        assert "legacy" in s6.lower(), (
            "CIC drift: ModelArtifactFetcher Section 6 missing legacy full-object deprecation"
        )

    def test_documents_state_dict(self):
        s6 = _read_section6("ModelArtifactFetcher")
        assert "state_dict" in s6.lower() or "weights_only" in s6.lower(), (
            "CIC drift: ModelArtifactFetcher Section 6 missing state_dict/weights_only behavior"
        )


class TestDataFetcherSection6:
    """DataFetcher CIC must document blueprint raise behavior."""

    def test_documents_blueprint_raise(self):
        s6 = _read_section6("DataFetcher")
        has_raise = "raise" in s6.lower() or "valueerror" in s6.lower()
        has_blueprint = "blueprint" in s6.lower()
        assert has_raise and has_blueprint, (
            "CIC drift: DataFetcher Section 6 missing blueprint-source-missing "
            "raise behavior (changed from skip in C-50)"
        )
