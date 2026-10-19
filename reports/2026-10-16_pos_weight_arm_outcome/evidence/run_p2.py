"""Run the committed P2 harness on a truncated_nb arm.

The ONLY change is vehicle.EXPECTED_SIDECAR["output_distribution"]: the committed gate hardcodes
violet_visitor's 'nb', and these arms are 'truncated_nb' (r1_readout.py:69 asserts the same value
for amber_kestrel). Every statistic -- binning, implied_w, the branch test -- is the committed
file, executed unmodified.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vehicle

assert vehicle.EXPECTED_SIDECAR["output_distribution"] == "nb", "gate already changed; stop"
vehicle.EXPECTED_SIDECAR["output_distribution"] = "truncated_nb"
sys.argv = ["p2_gate_calibration.py", sys.argv[1]]
runpy.run_path(str(Path(__file__).parent / "p2_gate_calibration.py"), run_name="__main__")
