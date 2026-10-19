import numpy as np, subprocess
from pathlib import Path
VM = Path("/home/simon/Documents/scripts/views_platform/views-models")
for name, root, stem, arch in (
    ("amber ", VM/"models/amber_kestrel/data/generated/calibration_delivery_20261013", "predictions_calibration_20261013_010739", "predictions_calibration_20261013_010739_lr.tar.zst"),
    ("copper", VM/"models/copper_kestrel/data/generated/calibration_delivery_20261016", "predictions_calibration_20261016_120217", "predictions_calibration_20261016_120217_lr.tar.zst")):
    m = f"{stem}/origin_0/lr_sb_best"
    subprocess.run(["tar","--use-compress-program=unzstd","-xf",str(root/"draws"/arch),
                    f"{m}/y_pred.npy", f"{m}/identifiers.npz"], cwd="/tmp/lrwork", check=True)
    p = Path("/tmp/lrwork")/m
    y = np.load(p/"y_pred.npy"); ids = np.load(p/"identifiers.npz", allow_pickle=True)
    t = np.asarray(ids["time"]).ravel()
    h1 = (t == 457)
    nz = y[h1][y[h1] > 0]
    print(f"{name}: origin_0 h=1 all land -- P(emit>0)={float((y[h1]>0).mean()):.6f}  "
          f"mean emitted {float(y[h1].mean()):.4f}  mean|nonzero {float(nz.mean()) if nz.size else 0:.3f}  "
          f"total mass {float(y[h1].sum()/y[h1].shape[1]):.1f}")
    for f in (p/"y_pred.npy", p/"identifiers.npz"): f.unlink()
