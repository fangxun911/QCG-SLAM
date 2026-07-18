"""Gaussian parameter initialization and storage helpers."""

import numpy as np
import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation

from qcg_slam import context as slam_context


def surface_normals_from_rotations(rotations):
    """Return the world-space local z axis for wxyz rotations."""
    return build_rotation(F.normalize(rotations.detach(), dim=1))[:, :, 2]


def _validate_gaussian_geometry(scales, rotations, num_points,
                                gaussian_distribution):
    expected_scale_dims = 1 if gaussian_distribution == "isotropic" else 3
    if gaussian_distribution not in ("isotropic", "anisotropic"):
        raise ValueError(
            f"Unknown gaussian_distribution {gaussian_distribution}")
    if scales.shape != (num_points, expected_scale_dims):
        raise ValueError(
            f"Expected {gaussian_distribution} scales with shape "
            f"({num_points}, {expected_scale_dims}), got {tuple(scales.shape)}")
    if rotations.shape != (num_points, 4):
        raise ValueError(
            f"Expected rotations with shape ({num_points}, 4), got "
            f"{tuple(rotations.shape)}")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Gaussian scales must be finite and strictly positive")
    if not torch.isfinite(rotations).all() or (
            torch.linalg.vector_norm(rotations, dim=1) <= 0).any():
        raise ValueError("Gaussian rotations must be finite non-zero quaternions")
    return scales, torch.nn.functional.normalize(rotations, dim=1)


def _initialize_gaussian_params(point_cloud, scales, rotations, opacity_logit,
                                gaussian_distribution):
    num_points = point_cloud.shape[0]
    scales, rotations = _validate_gaussian_geometry(
        scales, rotations, num_points, gaussian_distribution)
    return {
        'means3D': point_cloud[:, :3],
        'rgb_colors': point_cloud[:, 3:6],
        'unnorm_rotations': rotations,
        'logit_opacities': torch.full(
            (num_points, 1),
            opacity_logit,
            dtype=torch.float,
            device=slam_context.device),
        'log_scales': torch.log(scales),
    }


def _make_trainable(params):
    for key, value in params.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=slam_context.device).float()
        params[key] = torch.nn.Parameter(
            value.float().contiguous().requires_grad_(True)).to(
                slam_context.device)
    return params


def initialize_params(init_pt_cld, num_frames, init_scales, init_rotations,
                      gaussian_distribution):
    """Initialize Gaussian and camera parameters for optimization."""
    params = _initialize_gaussian_params(init_pt_cld, init_scales,
                                         init_rotations, 2.1,
                                         gaussian_distribution)

    # Initialize a single gaussian trajectory to model the camera poses relative
    # to the first frame
    # 将第一帧的相机位姿设置为单位值
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    # 将所有参数变成可学习的参数
    params = _make_trainable(params)
    if gaussian_distribution == "anisotropic":
        params['surface_normals'] = surface_normals_from_rotations(
            params['unnorm_rotations'])

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


def initialize_new_params(new_pt_cld, init_scales, init_rotations,
                          gaussian_distribution):
    """Initialize trainable parameters for newly added Gaussians."""
    params = _initialize_gaussian_params(new_pt_cld, init_scales,
                                         init_rotations, 2.0,
                                         gaussian_distribution)
    params = _make_trainable(params)
    # 此函数和initialize_params函数的区别是：该函数只初始化新的点云的信息，不生成相机位姿信息和variables信息
    return params


def initialize_finer_params(new_pt_cld, init_scales, init_rotations,
                            gaussian_distribution):
    """Initialize trainable parameters for fine-level Gaussians."""
    params = _initialize_gaussian_params(new_pt_cld, init_scales,
                                         init_rotations, 0.0,
                                         gaussian_distribution)
    params = _make_trainable(params)
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
