"""Build the VALIDATION-partition truth (months 505-552) the ruler needs but does not have.

The frozen v2 truth covers **calibration only** (months 121-504). The acceptance run scores on the
validation partition — deliberately, because calibration is the partition that SELECTED these eight
configurations, so scoring there would grade the choice on the data that made it. But no validation
truth artifact existed, which surfaced only when the first scored run raised
`KeyError: (505, 62356)`.

Same fetch path, same region, same features as `tier_a_parity._fetch` — the provenance the
calibration artifact records. Frozen to disk with its sha256, for the same reason: a re-pull is not
byte-stable across GED vintages, so the artifact is the reference, not the query.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "v2_truth_validation.parquet"
START, END = 505, 552
REGION = "africa_me_legacy"
# Taken from tier_a_parity.DEFAULT_FEATURE_MAP, which is what built the calibration truth —
# NOT guessed. A first attempt invented `*_sum_nokgi` names and the backend rejected them.
RENAME = {
    "ged_sb_best": "lr_sb_best",
    "ged_ns_best": "lr_ns_best",
    "ged_os_best": "lr_os_best",
}
FEATURES = list(RENAME)


def main() -> int:
    import datafactory_query as d

    z = d.defaults.DEFAULT_REMOTE.zarr_url
    last = d.get_last_valid_month_id(zarr_url=z)
    print(f"remote last_valid_month_id = {last}", flush=True)
    if last < END:
        print(f"REFUSING: remote has only {last}, validation needs {END}. A partial truth would "
              f"silently score fewer months than the criteria assume.")
        return 1
    df = d.load_dataset(region=REGION, start=START, end=END, features=FEATURES,
                        output_format="dataframe", data_dir=z)
    df = df.rename(columns=RENAME)
    df.index = df.index.set_names(
        ["priogrid_id" if n and n.startswith("priogrid") else n for n in df.index.names]
    )
    # the ruler indexes (month_id, priogrid_id) in that order
    if df.index.names[0] != "month_id":
        df = df.reorder_levels(["month_id", "priogrid_id"]).sort_index()
    months = df.index.get_level_values(0)
    cells = df.index.get_level_values(1).nunique()
    print(f"months {months.min()}-{months.max()} ({months.nunique()}) x {cells} cells "
          f"= {len(df)} rows", flush=True)
    if months.min() != START or months.max() != END:
        print("REFUSING: month span is not the validation partition.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT}\nsha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
