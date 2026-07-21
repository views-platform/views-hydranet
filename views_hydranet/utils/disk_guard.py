"""Pre-run disk-headroom guard (C-154).

The 6-run hurdle-NB sweep silently truncated S3_seed4's eval when the volume filled mid-run. A run
writes ~2.5 GB of predictions (plus diagnostics) per origin-set; this guard aborts **before** those
writes if free space is below a configured budget. Opt-in: a ``None`` budget is a no-op, so the
default behaviour is unchanged (byte-identical) for every existing model/run.

Reports and aborts (fail loud) — it does NOT delete anything; cleanup stays a human decision.
"""

from __future__ import annotations

import logging
import math
import os
import shutil

import numpy as np

logger = logging.getLogger(__name__)

# Fraction of currently-available RAM a single allocation may claim before the guard fails loud.
# Leaves headroom for the framework's own working set + the numpy→PredictionFrame copies.
_RAM_HEADROOM_FRACTION = 0.85


def _available_ram_bytes() -> int | None:
    """Currently-available physical RAM in bytes, or ``None`` where it cannot be measured.

    Uses POSIX ``sysconf`` (Linux/macOS) so no third-party dependency is needed. Returns ``None``
    on platforms/builds without the required keys, so the caller degrades to a warn-and-skip.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        return None


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


def assert_cube_fits(
    *shapes: tuple[int, ...],
    dtype=np.float32,
    budget_gb: float | None = None,
    log=logger,
) -> None:
    """Raise ``RuntimeError`` if allocating ``shapes`` would exhaust memory (ADR-067 D×K guard).

    The D×K posterior cube grows with ``S = n_posterior_samples * n_head_samples`` and can blow up
    the ``[T,H,W,C,S]`` allocation (the 37GB-cache-scar class). Called **before** ``np.zeros`` so
    an oversize request fails loud instead of thrashing/OOM-killing.

    Two independent checks:
    - **Auto (always on):** the summed cube bytes must fit within ``_RAM_HEADROOM_FRACTION`` of the
      machine's *currently-available* RAM — so it adapts to a 32 GB laptop vs a 256 GB server with
      nothing to configure. Skipped (warn only) where free RAM cannot be measured.
    - **Budget (opt-in):** if ``budget_gb`` is set, the cube must also be ≤ ``budget_gb`` — a lower
      per-run cap (e.g. on a shared host), mirroring ``min_free_disk_gb``. ``None`` disables it.

    Args:
        *shapes: one or more array shapes that will be allocated together (e.g. the magnitude and
            probability cubes).
        dtype: the allocation dtype (its ``itemsize`` sizes the estimate). Defaults to float32.
        budget_gb: optional hard per-run ceiling in GiB, or ``None`` to disable the opt-in cap.
        log: logger for the headroom report / skip warning.
    """
    itemsize = np.dtype(dtype).itemsize
    need_bytes = sum(int(math.prod(shape)) for shape in shapes) * itemsize
    need_gb = need_bytes / 1024**3

    if budget_gb is not None and need_gb > budget_gb:
        raise RuntimeError(
            f"Posterior cube too large for the configured budget: {need_gb:.2f} GB needed > "
            f"{budget_gb:.2f} GB (max_posterior_cube_gb). Lower K/D or raise the budget."
        )

    avail = _available_ram_bytes()
    if avail is None:
        log.warning(
            "assert_cube_fits: could not measure available RAM on this platform; "
            "skipping the auto memory guard (cube needs %.2f GB).",
            need_gb,
        )
        return
    avail_gb = avail / 1024**3
    log.info(
        "cube-fit check: %.2f GB needed vs %.2f GB available RAM (headroom %.2f)",
        need_gb,
        avail_gb,
        _RAM_HEADROOM_FRACTION,
    )
    if need_bytes > avail * _RAM_HEADROOM_FRACTION:
        raise RuntimeError(
            f"Insufficient RAM for the posterior cube: {need_gb:.2f} GB needed > "
            f"{_RAM_HEADROOM_FRACTION:.0%} of {avail_gb:.2f} GB available. The D×K cube "
            f"(S = n_posterior_samples × n_head_samples) would exhaust memory — lower "
            f"n_head_samples/n_posterior_samples or use a larger machine. No files were written."
        )
