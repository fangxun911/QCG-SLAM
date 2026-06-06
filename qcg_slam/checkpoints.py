"""Checkpoint loading and saving helpers."""

import os

import numpy as np
import torch

from utils.common_utils import save_params_ckpt

from qcg_slam.keyframes import make_keyframe


def load_checkpoint_state(config, dataset, params, variables, device):
    """Load a checkpoint and rebuild keyframes/ground-truth pose history."""
    keyframe_list = []
    keyframe_time_indices = []
    gt_w2c_all_frames = []

    if not config['load_checkpoint']:
        return params, variables, keyframe_list, keyframe_time_indices, \
            gt_w2c_all_frames, 0

    checkpoint_time_idx = config['checkpoint_time_idx']
    print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
    ckpt_path = os.path.join(config['workdir'], config['run_name'],
                             f"params{checkpoint_time_idx}.npz")
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = {
        k: torch.tensor(params[k], device=device).float().requires_grad_(True)
        for k in params.keys()
    }
    variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0],
                                             device=device).float()
    variables['means2D_gradient_accum'] = torch.zeros(
        params['means3D'].shape[0], device=device).float()
    variables['denom'] = torch.zeros(params['means3D'].shape[0],
                                     device=device).float()
    variables['timestep'] = torch.zeros(params['means3D'].shape[0],
                                        device=device).float()
    # Load the keyframe time idx list
    keyframe_time_indices = np.load(
        os.path.join(config['workdir'], config['run_name'],
                     f"keyframe_time_indices{checkpoint_time_idx}.npy"))
    keyframe_time_indices = keyframe_time_indices.tolist()

    # Update the ground truth poses list
    for time_idx in range(checkpoint_time_idx):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(gt_pose)
        gt_w2c_all_frames.append(gt_w2c)

        # Initialize Keyframe List
        if time_idx in keyframe_time_indices:
            color = color.permute(2, 0, 1) / 255
            depth = depth.permute(2, 0, 1)
            keyframe_list.append(
                make_keyframe(params, time_idx, color, depth, device))

    return params, variables, keyframe_list, keyframe_time_indices, \
        gt_w2c_all_frames, checkpoint_time_idx


def save_checkpoint_state(config, params, keyframe_time_indices, time_idx):
    """Save frame checkpoint and matching keyframe index list."""
    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
    save_params_ckpt(params, ckpt_output_dir, time_idx)
    np.save(
        os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"),
        np.array(keyframe_time_indices))
