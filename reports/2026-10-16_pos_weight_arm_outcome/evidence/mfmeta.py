import json
from pathlib import Path
VM = Path("/home/simon/Documents/scripts/views_platform/views-models")
for name, root in (("amber", VM/"models/amber_kestrel/data/generated/calibration_delivery_20261013/metricframes"),
                   ("copper", VM/"models/copper_kestrel/data/generated/calibration_delivery_20261016/metricframes")):
    print(name)
    for d in sorted(Path(root).rglob("metricframe_lr_sb_best")):
        if d.is_dir():
            print("  ", json.loads((d/"metadata.json").read_text()))
