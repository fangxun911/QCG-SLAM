"""Gaussian parameter initialization and storage helpers."""

import numpy as np
import torch

from qcg_slam import context as slam_context


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist,
                      gaussian_distribution):
    """Initialize Gaussian and camera parameters for optimization."""
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
    # 原始3D高斯是R S S^T R^T，现在改成 r，那么 R = 单位值，S = log(sqrt(r ** 2))
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
    # logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float,
    # device=slam_context.device)
    logit_opacities = torch.ones(
        (num_pts, 1), dtype=torch.float, device=slam_context.device) * 2.1
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(
            f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative
    # to the first frame
    # 将第一帧的相机位姿设置为单位值
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    # 将所有参数变成可学习的参数
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(
                torch.tensor(v, device=slam_context.device).float().contiguous(
                ).requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(
                v.float().contiguous().requires_grad_(True)).to(
                    slam_context.device)

    variables = {
        'max_2D_radius':
            torch.zeros(params['means3D'].shape[0],
                        device=slam_context.device).float(),
        'means2D_gradient_accum':
            torch.zeros(params['means3D'].shape[0],
                        device=slam_context.device).float(),
        'denom':
            torch.zeros(params['means3D'].shape[0],
                        device=slam_context.device).float(),
        'timestep':
            torch.zeros(params['means3D'].shape[0],
                        device=slam_context.device).float()
    }

    return params, variables


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    """Initialize trainable parameters for newly added Gaussians."""
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
    logit_opacities = torch.ones(
        (num_pts, 1), dtype=torch.float, device=slam_context.device) * 2.0
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(
            f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(
                torch.tensor(v, device=slam_context.device).float().contiguous(
                ).requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(
                v.float().contiguous().requires_grad_(True)).to(
                    slam_context.device)
    # 此函数和initialize_params函数的区别是：该函数只初始化新的点云的信息，不生成相机位姿信息和variables信息
    return params


def initialize_finer_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    """Initialize trainable parameters for fine-level Gaussians."""
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
    logit_opacities = torch.ones(
        (num_pts, 1), dtype=torch.float, device=slam_context.device) * 0
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(
            f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(
                torch.tensor(v, device=slam_context.device).float().contiguous(
                ).requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(
                v.float().contiguous().requires_grad_(True)).to(
                    slam_context.device)
    # 此函数和initialize_params函数的区别是：该函数只初始化新的点云的信息，不生成相机位姿信息和variables信息
    return params


def convert_params_to_store(params):
    """Detach parameters into tensors suitable for checkpoint storage."""
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store
