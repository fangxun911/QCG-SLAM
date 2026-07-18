"""Checkpoint loading and saving helpers."""

import os

import numpy as np
import torch

from utils.common_utils import save_params_ckpt

from qcg_slam.keyframes import make_keyframe
from qcg_slam.parameters import surface_normals_from_rotations


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
        k: torch.tensor(value, device=device).float().requires_grad_(
            k != 'surface_normals')
        for k, value in params.items()
    }
    expected_scale_dims = 1 if config['gaussian_distribution'] == \
        'isotropic' else 3
    if params['log_scales'].ndim != 2 or params['log_scales'].shape[
            1] != expected_scale_dims:
        raise ValueError(
            f"Checkpoint log_scales shape {tuple(params['log_scales'].shape)} "
            f"does not match {config['gaussian_distribution']} configuration")
    if config['gaussian_distribution'] == 'anisotropic':
        num_points = params['means3D'].shape[0]
        if 'surface_normals' not in params:
            params['surface_normals'] = surface_normals_from_rotations(
                params['unnorm_rotations'])
        if params['surface_normals'].shape != (num_points, 3):
            raise ValueError(
                "Checkpoint surface_normals must have shape "
                f"({num_points}, 3), got {tuple(params['surface_normals'].shape)}"
            )
        params['surface_normals'] = torch.nn.functional.normalize(
            params['surface_normals'].detach(), dim=1)
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
