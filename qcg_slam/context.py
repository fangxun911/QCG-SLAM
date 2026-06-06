"""Shared runtime context for QCG-SLAM modules."""

import torch

# The original script uses CUDA-only execution. Keep the default aligned with
# that behavior and let the runner update it during startup.
device = torch.device("cuda")


def set_device(new_device):
    """Set the device used by helper modules."""
    global device
    if isinstance(new_device, str):
        device = torch.device(new_device)
    else:
        device = new_device
