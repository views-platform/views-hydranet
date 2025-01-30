import torch

def setup_device():
    # Set the device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device   # not sure you need to return it, but it might be useful for debugging
