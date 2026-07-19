"""Optimizer construction helpers."""

import torch


def initialize_optimizer(params, lrs_dict, tracking):
    """Create the Adam optimizer for tracking or mapping parameters."""
    lrs = lrs_dict
    param_groups = []
    for key, value in params.items():
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            continue
        if key not in lrs:
            raise KeyError(f"Missing learning rate for trainable parameter {key}")
        param_groups.append({"params": [value], "name": key, "lr": lrs[key]})
    if tracking:
        return torch.optim.Adam(param_groups)
    return torch.optim.Adam(param_groups, eps=1e-15)
