"""T0 VERIFICATION on the model's actually-emitted posterior, not a reconstruction.

T0 attributed composed zero mass from ONE forward pass. This archive holds the draws the collapse
itself consumed — same run, same process (MANIFEST `source` matches the artefact timestamp). If the
two disagree, T0's reported number is wrong.

⛔ These draws are from the RETRAINED model (4acbf4d3c2b457b0…), NOT the delivered one
(d92dcff463bd310f…). They do not make the delivery's frames II/III reproducible.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
y = np.load(D / "y_pred.npy", mmap_mode="r")
ids = np.load(D / "identifiers.npz")
print("y_pred", y.shape, y.dtype, " identifier keys:", list(ids.keys()))
for k in ids:
    print("   %-14s %s" % (k, ids[k].shape))

SCRATCH = ("/tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-hydranet/"
           "f7a7ca8c-4eb9-45a0-8923-271b8be2a39c/scratchpad")
t1 = pd.read_parquet(f"{SCRATCH}/t1_fits.parquet")
cells = t1.priogrid_id.to_numpy()

# locate the month/unit axes from the identifiers
mon = ids["time"]
unit = ids["unit"]
mon, unit = np.asarray(mon).ravel(), np.asarray(unit).ravel()
print("\nrows %d  months %d (%d..%d)  cells %d"
      % (len(mon), len(np.unique(mon)), mon.min(), mon.max(), len(np.unique(unit))))

h1 = mon.min()
sel = (mon == h1) & np.isin(unit, cells)
Y = np.asarray(y[sel])                      # [n_cells, S]
print("h1 = month %d · selected %d of the 100 cells · S = %d" % (h1, sel.sum(), Y.shape[1]))

pos = (Y > 0).mean(1)
print("\n" + "=" * 70)
print("COMPOSED POSITIVE-DRAW FRACTION, from the emitted posterior (h1, sb)")
print("=" * 70)
print("  retrained model, THIS archive      median %.4f   (n=%d cells)"
      % (np.median(pos), len(pos)))
print("  retrained model, T0 single pass    median 0.4714   <- the number T0 reported")
print("  DELIVERED model (2026-09-29)       median 0.4062")
print("  observed history (conflictology)   median 0.7639")
print()
print("  cells with ALL %d draws zero: %d of %d" % (Y.shape[1], int((pos == 0).sum()), len(pos)))
print("  agreement with T0's reconstruction: %+.4f" % (np.median(pos) - 0.4714))
