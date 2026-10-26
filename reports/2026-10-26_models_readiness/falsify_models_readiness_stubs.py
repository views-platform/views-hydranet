"""RED-BY-DESIGN stubs for /falsify "models is also ready for the pod" (2026-10-26).

⛔ DELIBERATELY NOT NAMED test_*.py.  `reports/` is gitignored and force-added; pytest's
`python_files = *_test.py` would collect a module that imports a 661 MB archive and a
views-models checkout.  Run it explicitly:

    pytest reports/2026-10-26_models_readiness/falsify_models_readiness_stubs.py

Three hard falsifications and three soft.  Each stub fails TODAY and is the thing that must
go green before the claim "models is ready for the pod" can be reasserted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

MODELS = Path.home() / "Documents/scripts/views_platform/views-models"
HF_CFG = "models/heavy_freighter/configs/config_hyperparameters.py"
AS_RUN_SHA = "c8ae1803"


def _show(sha: str, path: str) -> str:
    """Read a views-models file AT A SHA.  ⛔ Never the working tree — §7.7b's rule, written
    after a stale branch produced two false findings in one day."""
    return subprocess.run(
        ["git", "-C", str(MODELS), "show", f"{sha}:{path}"],
        capture_output=True, text=True, check=True,
    ).stdout


# ─────────────────────────── HARD ───────────────────────────

def test_H1_pod_preflight_can_read_the_config_it_gates() -> None:
    """HARD-1.  pod_preflight.load_config_keys execs the config with no `__file__` in the
    namespace (pod_preflight.py:91).  heavy_freighter at c8ae1803 uses
    `Path(__file__).resolve()` at module level (:21, the trajectory_log_path helper PR #603
    added), so the gate dies with NameError on the one config it exists to gate.

    Published 0.1.7 — the wheel the pod installs — is byte-identical and carries the defect.
    §7.7b's committed claim "at the correct sha the pre-flight passes on both clauses" is
    refuted by execution, not by reasoning.
    """
    from views_hydranet.infrastructure.pod_preflight import load_config_keys

    cfg = Path(__file__).parent / "_hf_at_c8ae1803.py"
    cfg.write_text(_show(AS_RUN_SHA, HF_CFG))
    keys = load_config_keys(cfg)          # ⛔ raises NameError today
    assert "trajectory_log_path" in keys


def test_H2_P4_estimator_does_not_need_the_gate_posterior() -> None:
    """HARD-2.  §7.6 says the readout "CANNOT BE COMPUTED WITHOUT STEP 2", naming the `by_*`
    gate posterior as the class whose absence voids P-4 silently.  d4_mu_on_the_grid.py's
    usage line is `<path to calibration_model_*.pt>`: it rebuilds mu from the CHECKPOINT by
    its own forward pass.  P-3 and P-4 are body quantities.  The gate posterior is not on
    their critical path; `$OUT/artifact/` is.

    This stub asserts the prereg says what the code does.  It fails until §7.6 is corrected.
    """
    prereg = (
        Path(__file__).parents[1] / "2026-10-21_structure_penalty_arm_prereg.md"
    ).read_text()
    assert "THIS ARM'S READOUT CANNOT BE COMPUTED WITHOUT STEP 2" not in prereg, (
        "\u00a77.6 still attributes P-4's void to step 2 (the classes outside $OUT); P-4's "
        "estimator reads the checkpoint, which pod_run_model.sh:397 archives into $OUT/artifact"
    )


def test_H3_the_arm_installs_the_same_hydranet_the_control_ran() -> None:
    """HARD-3.  The control ran views-hydranet==0.1.7 (its delivered ENVIRONMENT, at
    git c8ae1803).  pod_run_model.sh:172 hardcodes `views-hydranet~=0.1.1` — latest 0.1.x —
    with no env seam, on a script that is on the never-modify list.  Publishing 0.1.8 (which
    the HARD-1 fix and C-365 both require) silently makes the arm a two-variable experiment.
    """
    runner = _show(AS_RUN_SHA, "tools/podrun/pod_run_model.sh")
    assert "views-hydranet==0.1.7" in runner or "HYDRANET_SPEC" in runner, (
        "the arm's code version is decided by PyPI at install time, not by the repo"
    )


# ─────────────────────────── SOFT ───────────────────────────

def test_S1_the_rescue_tool_is_on_the_branch_the_pod_clones() -> None:
    """SOFT-1.  pod_sweep_outputs.sh is not on `development`; it is on PR #589 (open), whose
    head is the stale branch that already caused two false findings.  The pod clones the arm's
    branch off `development`, so the tool will not be there.  pod_run_model.sh:241 sets the
    precedent — it DIES when tools/collapse is absent — and there is no such guard here.
    """
    listed = subprocess.run(
        ["git", "-C", str(MODELS), "ls-tree", "-r", "--name-only", "origin/development",
         "--", "tools/podrun/"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "pod_sweep_outputs.sh" in listed


def test_S2_runbook_step_1_is_executable_as_written() -> None:
    """SOFT-2.  §7.6 step 1 gives the sweep no arguments and names exit codes 2 and 1.  The
    script REQUIRES `--since <ref>` and a `<model_name>` and exits 3 (usage) without them —
    an exit code the prereg never mentions.  Its own header names the right reference: the
    trajectory CSV, whose mtime brackets training.
    """
    prereg = (
        Path(__file__).parents[1] / "2026-10-21_structure_penalty_arm_prereg.md"
    ).read_text()
    assert "--since" in prereg, "step 1 cannot be run as written"


@pytest.mark.parametrize("horizon", [18])
def test_S3_the_arm_gate_comes_home_at_the_control_s_horizon(horizon: int) -> None:
    """SOFT-3.  tools/podrun/slice_gate_posterior.py is MERGED on development and is the
    committed answer to the gate-posterior loss — §7.6 does not mention it.  It slices h=1
    only.  The control's rescued archive is the FULL by_* tree (117 entries = 13 origins x 3
    targets x 3 files, 6.55 GB raw) and the control's gate readout is at h=18.  Using the
    merged tool yields a gate that is not comparable to the control's; the hand rescue costs
    ~630 MB over a 43-48 KB/s link.  Neither option is pre-registered.
    """
    slicer = _show(AS_RUN_SHA, "tools/podrun/slice_gate_posterior.py")
    assert f"h={horizon}" in slicer or "--horizon" in slicer, (
        "the merged slicer is h=1 only; the control's readout is h=18"
    )
