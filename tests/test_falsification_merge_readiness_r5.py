"""
Falsification stubs for merge-readiness claim (R-5 hardening PR).

Soft falsification F4-03: CIC governance drift — two documentation gaps
discovered during merge-readiness audit.
"""


class TestF403CICGovernanceDrift:
    """CIC documents must reflect actual code behavior post-hardening."""

    def test_volume_handler_cic_lists_flip_hardening_tests(self):
        """VolumeHandler.md §10 must reference test_flip_symmetry_hardening.py
        and test_falsification_flip_hardening.py."""
        with open("docs/CICs/VolumeHandler.md") as f:
            content = f.read()
        assert "test_flip_symmetry_hardening" in content, (
            "VolumeHandler CIC §10 missing test_flip_symmetry_hardening.py"
        )
        assert "test_falsification_flip_hardening" in content, (
            "VolumeHandler CIC §10 missing test_falsification_flip_hardening.py"
        )

    def test_pfa_cic_lists_convention_mismatch_failure_mode(self):
        """PredictionFrameAssembler.md §6 must list convention mismatch as
        a failure mode (raises ValueError when signal or provider is not
        NORTH_UP)."""
        with open("docs/CICs/PredictionFrameAssembler.md") as f:
            content = f.read()
        section_6_start = content.find("## 6")
        section_7_start = content.find("## 7")
        section_6 = content[section_6_start:section_7_start]
        assert "convention" in section_6.lower(), (
            "PredictionFrameAssembler CIC §6 missing convention mismatch "
            "failure mode — code raises ValueError but CIC doesn't document it"
        )
