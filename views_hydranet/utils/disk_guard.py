"""Pre-run disk-headroom guard (C-154).

The 6-run hurdle-NB sweep silently truncated S3_seed4's eval when the volume filled mid-run. A run
writes ~2.5 GB of predictions (plus diagnostics) per origin-set; this guard aborts **before** those
writes if free space is below a configured budget. Opt-in: a ``None`` budget is a no-op, so the
default behaviour is unchanged (byte-identical) for every existing model/run.

Reports and aborts (fail loud) — it does NOT delete anything; cleanup stays a human decision.
"""

from __future__ import annotations

import logging
import shutil

logger = logging.getLogger(__name__)


def assert_disk_headroom(min_free_gb: float | None, path: str = ".", *, log=logger) -> None:
    """Raise ``RuntimeError`` if free space at ``path`` is below ``min_free_gb`` GiB.

    No-op when ``min_free_gb`` is ``None`` (the opt-in default — it does not even stat the disk).

    Args:
        min_free_gb: required free space in GiB, or ``None`` to disable the guard.
        path: any path on the target volume (free space is per-filesystem). Defaults to cwd.
        log: logger for the (info) headroom report.
    """
    if min_free_gb is None:
        return
    free_gb = shutil.disk_usage(path).free / 1024**3
    log.info(
        "disk-headroom check: %.1f GB free at %r (budget %.1f GB)", free_gb, path, min_free_gb
    )
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient disk headroom: {free_gb:.1f} GB free at {path!r} < required "
            f"{min_free_gb:.1f} GB (C-154). A run writes ~2.5 GB per origin-set and "
            f"would otherwise truncate silently (as the 6-run sweep did to S3_seed4) — free space "
            f"before re-running. No files were deleted."
        )
