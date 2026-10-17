import tarfile, numpy as np, pandas as pd
from pathlib import Path
C = Path("/home/simon/Documents/scripts/views_platform/views-models/models/copper_kestrel/data/generated/calibration_delivery_20261016")
p = sorted(C.glob("parquet/*.parquet"))[0]
df = pd.read_parquet(p)
print("parquet:", p.name, df.shape)
print("  index:", df.index.names, " columns:", list(df.columns))
print(df.head(3))
print("\n  horizons present:", sorted(set(df.reset_index().month_id))[:5], "...")
print("\n=== draws archive contents (first 12 entries)")
import zstandard, io
dec = zstandard.ZstdDecompressor()
with open(C/"draws"/sorted(x.name for x in (C/"draws").iterdir())[0], "rb") as fh:
    with dec.stream_reader(fh) as r:
        with tarfile.open(fileobj=io.BufferedReader(r), mode="r|") as t:
            for i, m in enumerate(t):
                if i >= 12: break
                print("   ", m.name, m.size)
