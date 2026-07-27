#!/usr/bin/env python
"""probe_inverse_transform_memory.py — line-isolate the C-116 eval/publish OOM.

CPU-only, no GPU/model/eval/network. Builds a representative forecast-shape volume
[T=36, H=180, W=180, C=11, S] float32 and measures PEAK process RSS across
`FeatureScaler.inverse_transform_volume` (the suspect: a full `work_data = vh.data.copy()`
at feature_scaler.py:221, which holds original + copy of the S-scaled volume at once).

Sweeps S ∈ {3, 6, 8} (each freed before the next; RAM-guarded so the probe can't OOM the box),
then measures an in-place variant (no `.copy()`) at S=8 to quantify the fix. Decisive:
does peak RSS scale with S and approach the observed ~16 GB, and does in-place cut it?

Run:  conda run -n views-hydranet-env python scripts/probe_inverse_transform_memory.py
"""

import gc
import threading
import time

import numpy as np

from views_hydranet.utils.config_initializer import TRANSFORMS
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.volume_handler import VolumeHandler

T, H, W, C = 36, 180, 180, 11
PRED_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
# 11 channels: 2 identity + 3 features (actuals) + 3 preds + 3 filler — only the
# .copy() of the whole array matters for memory; naming just needs to be valid.
CHANNEL_MAP = [
    "month_id",
    "priogrid_gid",
    "lr_sb_best",
    "lr_ns_best",
    "lr_os_best",
    "pred_lr_sb_best",
    "pred_lr_ns_best",
    "pred_lr_os_best",
    "f0",
    "f1",
    "f2",
]
assert len(CHANNEL_MAP) == C


def _vmrss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024  # KB → GB
    return float("nan")


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


class _PeakSampler:
    """Poll VmRSS in a thread; capture the high-water mark during a block."""

    def __init__(self, interval=0.01):
        self.interval, self.peak, self._run = interval, 0.0, False

    def __enter__(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while self._run:
            self.peak = max(self.peak, _vmrss_gb())
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._run = False
        self._t.join()
        self.peak = max(self.peak, _vmrss_gb())


def _make_vh(S: int) -> VolumeHandler:
    data = np.zeros((T, H, W, C, S), dtype=np.float32)
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C", "S"),
        channel_map=CHANNEL_MAP,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=("month_id", "priogrid_gid"),
        feature_cols=tuple(PRED_TARGETS),
        spatial_offset=(0, 0),
    )


def _make_scaler() -> FeatureScaler:
    sc = FeatureScaler(config={"transformations": {"log1p": list(PRED_TARGETS)}})
    sc._is_fitted = True
    return sc


def _inverse_in_place(vh: VolumeHandler) -> None:
    """Candidate fix: invert each pred channel ON vh.data — NO full-volume copy."""
    _, inverse = TRANSFORMS["log1p"]
    c_idx = vh.get_axis_idx("C")
    data = vh.data
    for i, name in enumerate(vh.channel_map):
        base = name.removeprefix("pred_")
        if base in PRED_TARGETS and not name.startswith("by_"):
            slc = [slice(None)] * data.ndim
            slc[c_idx] = i
            data[tuple(slc)] = inverse(data[tuple(slc)])


def main() -> None:
    vol_gb = lambda s: T * H * W * C * s * 4 / 1e9  # noqa: E731
    print(f"volume [T={T},H={H},W={W},C={C},S] float32 → {vol_gb(1):.2f} GB per sample-slice")
    base = _vmrss_gb()
    print(f"baseline RSS: {base:.2f} GB | MemAvailable: {_mem_available_gb():.1f} GB")
    print("per-S: copy_peak = extra RSS during inverse_transform_volume; inplace_peak = the fix\n")

    scaler = _make_scaler()
    for S in (3, 6, 8):
        if _mem_available_gb() < 12:
            print(f"{S:>3}  SKIPPED — MemAvailable < 12 GB (probe self-guard)")
            continue
        # --- current path: inverse_transform_volume (full .copy()) ---
        vh = _make_vh(S)
        before = _vmrss_gb()
        with _PeakSampler() as ps:
            result = scaler.inverse_transform_volume(vh)
        copy_peak = ps.peak - before
        net_after = _vmrss_gb() - before
        del vh, result
        gc.collect()

        # --- candidate fix: in-place, no copy ---
        vh2 = _make_vh(S)
        b2 = _vmrss_gb()
        with _PeakSampler() as ps2:
            _inverse_in_place(vh2)
        inplace_peak = ps2.peak - b2
        del vh2
        gc.collect()

        print(
            f"S={S} vol={vol_gb(S):.2f}G copy_peak={copy_peak:.2f}G "
            f"inplace_peak={inplace_peak:.2f}G net_after={net_after:.2f}G"
        )

    print(
        "\nRead: 'copy_path_peak' = extra RSS held during inverse_transform_volume "
        "(orig + .copy() + temporaries). If it ≈ 2× vol_GB and scales with S, the "
        ".copy() is the C-116 hog; compare 'inplace_peak' (the fix) — it should be ~1× or less."
    )


if __name__ == "__main__":
    main()
