"""Initial frame setup for QCG-SLAM."""

import torch

from utils.recon_helpers import setup_camera

from qcg_slam.parameters import initialize_params
from qcg_slam.pointcloud import get_quadtree_pointcloud


def initialize_first_timestep(dataset,
                              num_frames,
                              scene_radius_depth_ratio,
                              mean_sq_dist_method,
                              densify_dataset=None,
                              gaussian_distribution=None,
                              scene_name=None):
    """Initialize cameras, point cloud, and parameters from the first frame."""
    # Get RGB-D Data & Camera Parameters
    # 这里的pose是一个4x4矩阵
    color, depth, quadtree, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera: cam 是一个 diff_gaussian_rasterization 中的 Camera 类，里面存放了相机参数信息
    cam = setup_camera(color.shape[2], color.shape[1],
                       intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1],
                                   densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    # 这里能不能用一些预测 depth 的模型把空洞的 depth 补齐呢？
    mask = (depth[0, :, :] > 0)  # Mask out invalid depth values
    # mask = mask.reshape(-1)
    # mean3_sq_dist 是一个数组，每个3D点对应其中的一个值，也就是文章里的r ** 2
    init_pt_cld, mean3_sq_dist = get_quadtree_pointcloud(
        color,
        depth,
        quadtree,
        densify_intrinsics,
        w2c,
        mask=mask,
        compute_mean_sq_dist=True,
        mean_sq_dist_method=mean_sq_dist_method,
        time_idx=0,
        scene_name=scene_name)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames,
                                          mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting
    # Densification
    # replica 里的 scene_radius_depth_ratio = 3
    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio

    if densify_dataset is not None:
        return (params, variables, intrinsics, w2c, cam, densify_intrinsics,
                densify_cam)
    else:
        return params, variables, intrinsics, w2c, cam
