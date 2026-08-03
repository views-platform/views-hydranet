"""Pre-run disk-headroom guard (C-154).

The 6-run hurdle-NB sweep silently truncated S3_seed4's eval when the volume filled mid-run. This
guard aborts a run BEFORE it writes ~2.5 GB of predictions if free space is below a configured
budget. Opt-in: a None budget is a no-op (default-off => byte-identical for every existing model).
ADR-005 Green/Red taxonomy.
"""

from unittest import mock

import pytest

from views_hydranet.utils.disk_guard import assert_disk_headroom

_DU = "views_hydranet.utils.disk_guard.shutil.disk_usage"


def _usage(free_gb: float):
    """Mimic shutil.disk_usage's (total, used, free) — only .free is read."""
    free = int(free_gb * 1024**3)
    return mock.Mock(total=10 * free, used=0, free=free)


def test_green_sufficient_space_proceeds():
    """Green: free >= budget — proceeds silently (no raise)."""
    with mock.patch(_DU, return_value=_usage(50.0)):
        assert_disk_headroom(15.0, path=".")  # must not raise


def test_red_insufficient_space_aborts():
    """Red: free < budget — fail loud (the S3_seed4 truncation must never recur silently)."""
    with mock.patch(_DU, return_value=_usage(5.0)):
        with pytest.raises(RuntimeError, match="disk headroom"):
            assert_disk_headroom(15.0, path=".")


def test_none_budget_is_noop_and_never_touches_disk():
    """Opt-in contract: a None budget is a no-op — it must not even call disk_usage, so the
    default-off path is byte-identical to pre-#107 behaviour for every existing run."""
    with mock.patch(_DU) as du:
        assert_disk_headroom(None, path=".")
        du.assert_not_called()
