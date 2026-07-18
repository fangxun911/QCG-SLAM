"""Gaussian densification helpers."""

import torch

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.slam_external import build_rotation
from utils.slam_helpers import (
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    transform_to_frame,
)

from qcg_slam import context as slam_context
from qcg_slam.parameters import (
    initialize_new_params,
    initialize_finer_params,
    surface_normals_from_rotations,
)
from qcg_slam.pointcloud import get_pointcloud, get_quadtree_pointcloud


def add_coarse_gaussians(params, variables, curr_data, sil_thres, time_idx,
                         mean_sq_dist_method, gaussian_distribution,
                         scene_name, surface_init_config=None):
    """Add coarse Gaussians for unobserved silhouette regions."""
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params,
                                               time_idx,
                                               gaussians_grad=False,
                                               camera_grad=False)
    # rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data['w2c'], transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(
        **depth_sil_rendervar)
    # im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)  # < 0.5
    # Check for new foreground objects by using GT depth
    # gt_depth = curr_data['depth'][0, :, :]
    # render_depth = depth_sil[0, :, :]
    # depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    # non_presence_depth_mask = (render_depth > gt_depth) * (depth_error >
    # 50*depth_error.median())
    # Determine non-presence mask
    # non_presence_mask = non_presence_sil_mask | non_presence_depth_mask #
    # silhouette低、深度不合理的像素的并集的mask
    # coarse densification 只填补没观察到的区域，深度差异大的地方在fine densification填补
    non_presence_mask = non_presence_sil_mask
    # # Flatten mask
    # non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    # 如果mask是全0，那就不用添加新高斯了
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(
            params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4, device=slam_context.device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        # non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        non_presence_mask = non_presence_mask & valid_depth_mask  # 输入没有展平的mask

        new_pt_cld, init_scales, init_rotations = get_quadtree_pointcloud(
            curr_data['im'],
            curr_data['depth'],
            curr_data['quadtree'],
            curr_data['intrinsics'],
            curr_w2c,
            mask=non_presence_mask,
            compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method,
            time_idx=time_idx,
            scene_name=scene_name,
            gaussian_distribution=gaussian_distribution,
            surface_init_config=surface_init_config)
        # print("new quadtree points: ", new_pt_cld.shape[0], "\n")
        if new_pt_cld.shape[0] != 0:
            new_params = initialize_new_params(new_pt_cld, init_scales,
                                               init_rotations,
                                               gaussian_distribution)
            if gaussian_distribution == "anisotropic":
                new_surface_normals = surface_normals_from_rotations(
                    init_rotations)
            for k, v in new_params.items():
                # 将新生成的高斯点拼接到已有的高斯点集合里，params[k]是原来的，v是新的，通过cat拼接到一起
                params[k] = torch.nn.Parameter(
                    torch.cat((params[k], v), dim=0).requires_grad_(True))
            if gaussian_distribution == "anisotropic":
                params['surface_normals'] = torch.cat(
                    (params['surface_normals'], new_surface_normals), dim=0)
            num_pts = params['means3D'].shape[0]
            variables['means2D_gradient_accum'] = torch.zeros(
                num_pts, device=slam_context.device).float()
            variables['denom'] = torch.zeros(
                num_pts, device=slam_context.device).float()
            variables['max_2D_radius'] = torch.zeros(
                num_pts, device=slam_context.device).float()
            new_timestep = time_idx * torch.ones(
                new_pt_cld.shape[0], device=slam_context.device).float()
            variables['timestep'] = torch.cat(
                (variables['timestep'], new_timestep), dim=0)

    return params, variables


def add_fine_gaussians(params, variables, curr_data, sil_thres, color_thres,
                       time_idx, mean_sq_dist_method, gaussian_distribution,
                       surface_init_config=None):
    """Add fine Gaussians for color, depth, and silhouette residuals."""
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params,
                                               time_idx,
                                               gaussians_grad=False,
                                               camera_grad=False)
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data['w2c'], transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(
        **depth_sil_rendervar)
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)  # sil_thres = 0.5
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    im_error = torch.mean(torch.abs(curr_data['im'] - im),
                          dim=0) * (gt_depth > 0)

    non_presence_depth_mask = (render_depth > gt_depth) * (
        depth_error > 50 * depth_error.median())
    non_presence_color_mask = (im_error > color_thres)

    non_presence_mask = (non_presence_color_mask | non_presence_depth_mask |
                         non_presence_sil_mask)
    # non_presence_mask = non_presence_depth_mask | non_presence_sil_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(
            params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda(slam_context.device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, init_scales, init_rotations = get_pointcloud(
            curr_data['im'],
            curr_data['depth'],
            curr_data['intrinsics'],
            curr_w2c,
            mask=non_presence_mask,
            compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method,
            gaussian_distribution=gaussian_distribution,
            surface_init_config=surface_init_config)
        new_params = initialize_finer_params(new_pt_cld, init_scales,
                                             init_rotations,
                                             gaussian_distribution)
        if gaussian_distribution == "anisotropic":
            new_surface_normals = surface_normals_from_rotations(
                init_rotations)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(
                torch.cat((params[k], v), dim=0).requires_grad_(True))
        if gaussian_distribution == "anisotropic":
            params['surface_normals'] = torch.cat(
                (params['surface_normals'], new_surface_normals), dim=0)
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(
            num_pts, device=slam_context.device).float()
        variables['denom'] = torch.zeros(num_pts,
                                         device=slam_context.device).float()
        variables['max_2D_radius'] = torch.zeros(
            num_pts, device=slam_context.device).float()
        new_timestep = time_idx * torch.ones(
            new_pt_cld.shape[0], device=slam_context.device).float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),
                                          dim=0)

    return params, variables
