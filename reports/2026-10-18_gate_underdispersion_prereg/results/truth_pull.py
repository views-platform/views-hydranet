"""FRESH datafactory pull of truth for the F11 window, months 457-469, region=land.

Standing rule: never validate on cached disk data. The Phase-1 cache
(reports/2026-10-12_underfire_diagnosis/pull_land_all3_121_504.parquet, 2026-10-12) is
read too, and the two are COMPARED rather than one substituted for the other.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
DIAG = HN / "reports/2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(HN))
from vehicle import TARGETS  # noqa: E402

OUT = Path(__file__).parent / "truth_fresh_457_469.parquet"
M0, M1 = 457, 469

print("TARGETS:", TARGETS)
if OUT.exists():
    print("fresh pull already on disk; reusing this session's own pull")
    fresh = pd.read_parquet(OUT).reset_index()
else:
    from datafactory_query import load_dataset
    from datafactory_query.defaults import DEFAULT_REMOTE
    fresh = load_dataset(region="land", start=M0, end=M1, features=TARGETS,
                         output_format="dataframe", data_dir=DEFAULT_REMOTE.zarr_url)
    fresh.to_parquet(OUT)
    fresh = fresh.reset_index()
print("fresh rows:", len(fresh), "months:", sorted(fresh.month_id.unique()))

cache = pd.read_parquet(DIAG / "pull_land_all3_121_504.parquet").reset_index()
cache = cache[(cache.month_id >= M0) & (cache.month_id <= M1)]
print("cache rows:", len(cache))

f = fresh.set_index(["month_id", "priogrid_id"]).sort_index()
c = cache.set_index(["month_id", "priogrid_id"]).sort_index()
print("index identical:", f.index.equals(c.index))
for t in TARGETS.values() if isinstance(TARGETS, dict) else TARGETS:
    if t not in f.columns:
        continue
    a, b = f[t].to_numpy(), c.reindex(f.index)[t].to_numpy()
    print(f"  {t:28s} act_true fresh={float((a>0).mean()):.8f} "
          f"cache={float((b>0).mean()):.8f}  exact_match={float(np.mean(a==b)):.6f}")
