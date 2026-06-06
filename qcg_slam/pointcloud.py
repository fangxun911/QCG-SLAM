"""RGB-D to point-cloud conversion helpers."""

import torch

from qcg_slam import context as slam_context


def get_quadtree_pointcloud(color,
                            depth,
                            quadtree,
                            intrinsics,
                            w2c,
                            transform_pts=True,
                            mask=None,
                            compute_mean_sq_dist=False,
                            mean_sq_dist_method="projective",
                            time_idx=None,
                            scene_name=None,
                            params=None):
    """Build a colored point cloud from quadtree leaf pixels."""
    with torch.no_grad():
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]

        # Compute indices of pixels
        quadtree_tensor = quadtree.detach()
        # quadtree_tensor = torch.tensor(quadtree).to(slam_context.device)
        # [N, 4]维度，左上和右下坐标
        # x_grid 和 y_grid: 四叉树每个叶节点的中心点坐标
        x_grid = torch.div(quadtree_tensor[:, 0] + quadtree_tensor[:, 2],
                           2,
                           rounding_mode='floor')  # 宽坐标
        y_grid = torch.div(quadtree_tensor[:, 1] + quadtree_tensor[:, 3],
                           2,
                           rounding_mode='floor')  # 高坐标
        xx = (x_grid - cx) / fx
        yy = (y_grid - cy) / fy
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        # depth_z = depth[0].reshape(-1)
        depth_z = depth[0, y_grid, x_grid]  # [C, H, W] 所以先y后x
        # if mask is not None:
        #     mask = mask[y_grid, x_grid] # mask 原本是 (H, W)

        # Initialize point cloud: pts_cam 是相机坐标系下的3D点
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(pts_cam.shape[0],
                                  1,
                                  device=slam_context.device).float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]  # pts 是世界坐标系下的3D点
        else:
            pts = pts_cam

        # Compute mean squared distance for initializing the scale of the
        # Gaussians
        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                # Projective Geometry (this is fast, farther -> larger radius)
                scale_gaussian = depth_z / ((fx + fy) / 2)
                # 一个像素->多个像素
                if params is None:
                    # 不考虑旧高斯的位置
                    node_duijiaoxian = torch.sqrt(
                        (quadtree_tensor[:, 2] - quadtree_tensor[:, 0])**2 +
                        (quadtree_tensor[:, 3] -
                         quadtree_tensor[:, 1])**2).detach()
                    node_radius = node_duijiaoxian * 0.5
                    # node_radius = torch.maximum(node_radius,
                    # torch.ones_like(node_radius))
                    scale_gaussian = scale_gaussian * node_radius
                    mean3_sq_dist = scale_gaussian**2
                    if mask is not None:
                        mask = mask[y_grid, x_grid]
                        mean3_sq_dist = mean3_sq_dist[mask]
                    del quadtree_tensor, node_duijiaoxian, node_radius
                    torch.cuda.empty_cache()
                else:
                    node_duijiaoxian = torch.sqrt(
                        (quadtree_tensor[:, 2] - quadtree_tensor[:, 0])**2 +
                        (quadtree_tensor[:, 3] - quadtree_tensor[:, 1])**2)
                    node_radius = node_duijiaoxian * 0.5
                    # 这里的 scale_gaussian_tree 是四叉树叶节点的半径r
                    scale_gaussian_tree = scale_gaussian * node_radius
                    # 下面考虑最近邻距离
                    # k = min(params['log_scales'].shape[0], int(2 * 10 ** 10 /
                    # pts.shape[0])) # 没有那么多算力处理这么多点，所以取后k个点
                    # old_radius = torch.exp(
                    #     params['log_scales'][-k:].detach()).max(dim=1).values
                    # old_points = params['means3D'][-k:].detach()
                    if mask is not None:
                        mask = mask[y_grid, x_grid]
                        new_points = pts[mask].detach()
                        # print("new quadtree points: ", new_points.shape[0],
                        # "~~~~~\n")
                        if new_points.shape[0] == 0:
                            return new_points, scale_gaussian**2
                        k = min(
                            params['log_scales'].shape[0],
                            int(10**9 /
                                new_points.shape[0]))  # 没有那么多算力处理这么多点，所以取后k个点
                        old_radius = torch.exp(
                            params['log_scales'][-k:].detach()).max(
                                dim=1).values
                        old_points = params['means3D'][-k:].detach()
                        # torch.save({"new_points": new_points, "old_points":
                        # old_points, "old_radius": old_radius},
                        # "cal_distance.1122.pt")
                        distances = torch.cdist(new_points, old_points)
                        scale_gaussian_tree_mask = scale_gaussian_tree[
                            mask].detach()
                        scale_gaussian_mask = scale_gaussian[mask].detach()
                    else:
                        distances = torch.cdist(pts, old_points)
                    closest_distance = torch.min(distances, dim=1)
                    neighbor_distance = closest_distance.values - old_radius[
                        closest_distance.indices]
                    # 综合这3者
                    scale_gaussian = torch.maximum(
                        torch.minimum(scale_gaussian_tree_mask,
                                      neighbor_distance), scale_gaussian_mask)
                    mean3_sq_dist = scale_gaussian**2
                    del distances, closest_distance, neighbor_distance
                    del quadtree_tensor
                    torch.cuda.empty_cache()

            else:
                raise ValueError(
                    f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

        # Colorize point cloud
        # cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) ->
        # (H, W, C) -> (H * W, C)
        cols = color[:, y_grid, x_grid]
        # 给点云上色，后3列为rgb值
        point_cld = torch.cat((pts, torch.transpose(cols, 0, 1)), -1)

        if mask is not None:
            # mask = mask[y_grid, x_grid] # mask 原本是 (H, W)
            point_cld = point_cld[mask]
            # if compute_mean_sq_dist:
            #     mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            # torch.cuda.empty_cache()
            return point_cld, mean3_sq_dist
        else:
            # torch.cuda.empty_cache()
            return point_cld


def get_pointcloud(color,
                   depth,
                   intrinsics,
                   w2c,
                   transform_pts=True,
                   mask=None,
                   compute_mean_sq_dist=False,
                   mean_sq_dist_method="projective"):
    """Build a colored point cloud from dense RGB-D pixels."""
    width, height = color.shape[2], color.shape[1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]

    # Compute indices of pixels
    # 这里是在像素平面生成网格
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=slam_context.device).float(),
        torch.arange(height, device=slam_context.device).float(),
        indexing='xy')
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud: pts_cam 是相机坐标系下的3D点
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1,
                              device=slam_context.device).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]  # pts 是世界坐标系下的3D点
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((fx + fy) / 2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(
                f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(
        -1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    # 给点云上色，后3列为rgb值
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    # print("new small points: ", point_cld.shape[0], "~~~~~\n")
    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld
