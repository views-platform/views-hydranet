"""The per-arm SETUP audit (`scripts/arm_postflight.py`).

It runs inside `run_queue.sh`'s verify hook, whose exit code already stops the queue — so a false
PASS lets a 29-hour queue keep spending on broken output, and a false FAIL halts a healthy one.
Both directions are therefore pinned. The positive direction is *also* checked against four real
completed arms (see the dossier's `03` §D), because a guard whose passing case has only been seen
on synthetic fixtures is barely a guard.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.arm_postflight import audit_arm

HEADER = "target,model,h,N,n_event,AP,crps_all,act_ratio\n"
ARM = "someone_somewhere"
HZ = (1, 18)


def _score(rows) -> str:
    return HEADER + "".join(rows)


def _row(h, n=170430, n_event=1343, ap=0.33, crps=0.13):
    return f"sb,{ARM},{h},{n},{n_event},{ap},{crps},0.007\n"


def _good(tmp: Path, arm: str = ARM) -> Path:
    res = tmp / "results"
    res.mkdir(exist_ok=True)
    (res / f"score_{arm}.csv").write_text(_score([_row(h) for h in HZ]))
    (res / f"score_{arm}_use_real.csv").write_text(_score([_row(h, ap=0.49) for h in HZ]))
    for name in (f"ap_ci_{arm}.json", f"ret_ci_{arm}.json"):
        (res / name).write_text(json.dumps({"18": {"ap": 0.33, "mde": 0.016}}))
    (res / f"FLOORGATE_{arm}_PASS").write_text("FLOOR GATE: PASS\n")
    return res


def test_intact_arm_passes(tmp_path):
    assert audit_arm(_good(tmp_path), ARM, horizons=HZ) == []


def test_missing_artifact_is_caught(tmp_path):
    res = _good(tmp_path)
    (res / f"score_{ARM}_use_real.csv").unlink()
    assert any("missing artifact" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_empty_artifact_is_caught(tmp_path):
    """A truncated write leaves a 0-byte file that `-s` tests elsewhere would also reject."""
    res = _good(tmp_path)
    (res / f"ap_ci_{ARM}.json").write_text("")
    assert any("empty artifact" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_missing_floor_gate_token_is_caught(tmp_path):
    res = _good(tmp_path)
    (res / f"FLOORGATE_{ARM}_PASS").unlink()
    assert any("floor-gate" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_failed_floor_gate_is_caught(tmp_path):
    """A floor-gate FAIL means the vehicle cannot show an effect — C-299's three lost days."""
    res = _good(tmp_path)
    (res / f"FLOORGATE_{ARM}_PASS").unlink()
    (res / f"FLOORGATE_{ARM}_FAIL").write_text("FLOOR GATE: FAIL\n")
    assert any("floor gate FAILED" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_nan_in_a_score_field_is_caught(tmp_path):
    res = _good(tmp_path)
    (res / f"score_{ARM}.csv").write_text(_score([_row(1), _row(18, ap=float("nan"))]))
    assert any("non-finite AP" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_missing_horizon_is_caught(tmp_path):
    res = _good(tmp_path)
    (res / f"score_{ARM}.csv").write_text(_score([_row(1)]))
    assert any("missing horizons" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_support_mismatch_against_the_reference_is_caught(tmp_path):
    """THE quiet one: differing N means the arms are not scored on the same cells.

    Every paired comparison downstream would then be invalid rather than merely wrong, and nothing
    else in the pipeline notices — `ap_diff_origin_block_ci` refuses, but only if it is reached.
    """
    res = _good(tmp_path)
    ref = tmp_path / "ref.csv"
    ref.write_text(_score([_row(h, n=169000) for h in HZ]))
    problems = audit_arm(res, ARM, horizons=HZ, reference=ref)
    assert any("not scored on the same support" in p for p in problems)


def test_inconsistent_N_within_an_arm_is_caught(tmp_path):
    res = _good(tmp_path)
    (res / f"score_{ARM}.csv").write_text(_score([_row(1), _row(18, n=999)]))
    assert any("N is not constant" in p for p in audit_arm(res, ARM, horizons=HZ))


def test_degenerate_mde_is_caught(tmp_path):
    """An mde of 0 would make any effect look significant against `3 x MDE`."""
    res = _good(tmp_path)
    (res / f"ap_ci_{ARM}.json").write_text(json.dumps({"18": {"ap": 0.33, "mde": 0.0}}))
    assert any("non-positive or non-finite mde" in p for p in audit_arm(res, ARM, horizons=HZ))
