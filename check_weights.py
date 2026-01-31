import torch
import sys
from pathlib import Path

def check_model(path):
    print(f"Checking model: {path}")
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
        has_nan = False
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"  {name} contains non-finite values!")
                has_nan = True
        if not has_nan:
            print("  All weights are finite.")
    except Exception as e:
        print(f"  Failed to load model: {e}")

if len(sys.argv) > 1:
    check_model(sys.argv[1])
else:
    # Try to find the latest model in artifacts
    import glob
    models = glob.glob("artifacts/*.pt")
    if models:
        latest_model = max(models, key=lambda x: Path(x).stat().st_mtime)
        check_model(latest_model)
    else:
        print("No models found in artifacts/")
