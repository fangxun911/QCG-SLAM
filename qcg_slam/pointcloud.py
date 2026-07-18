"""RGB-D to Gaussian seed conversion helpers."""

import torch
import torch.nn.functional as F

from utils.slam_helpers import matrix_to_quaternion

from qcg_slam import context as slam_context


DEFAULT_SURFACE_INIT_CONFIG = {
    "normal_window": 5,
    "fallback_normal_window": 3,
    "depth_abs_thresh": 0.02,
    "depth_rel_thresh": 0.02,
    "min_plane_points": 6,
    "max_planarity_ratio": 0.05,
    "min_view_cos": 0.2,
    "normal_scale_min_ratio": 0.05,
    "normal_scale_max_ratio": 0.25,
    "node_min_valid_fraction": 0.5,
    "node_min_inlier_fraction": 0.8,
    "geometry_batch_size": 32768,
    "min_scale": 1e-6,
}


def _surface_init_config(config):
    merged = DEFAULT_SURFACE_INIT_CONFIG.copy()
    if config is not None:
        merged.update(config)
    for key in ("normal_window", "fallback_normal_window"):
        window = int(merged[key])
        if window < 3 or window % 2 == 0:
            raise ValueError(f"{key} must be an odd integer >= 3")
        merged[key] = window
    merged["geometry_batch_size"] = int(merged["geometry_batch_size"])
    merged["min_plane_points"] = int(merged["min_plane_points"])
    return merged


def _identity_rotations(num_points, device, dtype):
    rotations = torch.zeros((num_points, 4), device=device, dtype=dtype)
    rotations[:, 0] = 1.0
    return rotations


def _unproject(depth_values, x_values, y_values, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return torch.stack(((x_values - cx) * depth_values / fx,
                        (y_values - cy) * depth_values / fy, depth_values),
                       dim=-1)


def _masked_median(values, valid):
    filled = torch.where(valid, values,
                         torch.full_like(values, float("inf")))
    sorted_values = torch.sort(filled, dim=1).values
    counts = valid.sum(dim=1)
    indices = torch.div((counts - 1).clamp_min(0),
                        2,
                        rounding_mode="floor").unsqueeze(1)
    medians = sorted_values.gather(1, indices).squeeze(1)
    return torch.where(counts > 0, medians, torch.zeros_like(medians))


def _estimate_local_planes_batch(depth, intrinsics, center_x, center_y,
                                 window, config):
    height, width = depth.shape[1:]
    half_window = window // 2
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-half_window,
                     half_window + 1,
                     device=depth.device),
        torch.arange(-half_window,
                     half_window + 1,
                     device=depth.device),
        indexing="ij")
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)
    patch_x = center_x[:, None] + offset_x
    patch_y = center_y[:, None] + offset_y
    in_bounds = ((patch_x >= 0) & (patch_x < width) & (patch_y >= 0) &
                 (patch_y < height))
    safe_x = patch_x.clamp(0, width - 1)
    safe_y = patch_y.clamp(0, height - 1)
    patch_depth = depth[0, safe_y, safe_x]
    center_depth = depth[0, center_y, center_x]
    depth_gate = torch.maximum(
        torch.full_like(center_depth, config["depth_abs_thresh"]),
        center_depth * config["depth_rel_thresh"])
    valid = (in_bounds & (patch_depth > 0) &
             (torch.abs(patch_depth - center_depth[:, None]) <=
              depth_gate[:, None]))

    patch_points = _unproject(patch_depth, patch_x.to(depth.dtype),
                              patch_y.to(depth.dtype), intrinsics)
    spatial_sigma = float(max(half_window, 1))
    spatial_weights = torch.exp(
        -(offset_x.to(depth.dtype)**2 + offset_y.to(depth.dtype)**2) /
        (2.0 * spatial_sigma**2))
    weights = spatial_weights * valid.to(depth.dtype)
    weight_sum = weights.sum(dim=1).clamp_min(config["min_scale"])
    centroid = (patch_points * weights[..., None]).sum(
        dim=1) / weight_sum[:, None]
    centered = patch_points - centroid[:, None, :]
    covariance = torch.einsum("bs,bsi,bsj->bij", weights, centered,
                              centered) / weight_sum[:, None, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    normal = eigenvectors[:, :, 0]

    center_points = _unproject(center_depth, center_x.to(depth.dtype),
                               center_y.to(depth.dtype), intrinsics)
    flip = torch.sum(normal * center_points, dim=1) > 0
    normal = torch.where(flip[:, None], -normal, normal)

    signed_residual = torch.sum(
        (patch_points - center_points[:, None, :]) * normal[:, None, :],
        dim=-1)
    residual_median = _masked_median(signed_residual, valid)
    residual_mad = _masked_median(
        torch.abs(signed_residual - residual_median[:, None]), valid)
    residual_sigma = 1.4826 * residual_mad

    denominator = eigenvalues[:, 1] + eigenvalues[:, 2]
    planarity_ratio = eigenvalues[:, 0] / denominator.clamp_min(
        config["min_scale"]**2)
    reliable = (valid.sum(dim=1) >= config["min_plane_points"])
    reliable &= denominator > config["min_scale"]**2
    reliable &= planarity_ratio <= config["max_planarity_ratio"]
    reliable &= torch.isfinite(normal).all(dim=1)
    reliable &= torch.isfinite(residual_sigma)
    return center_points, normal, residual_sigma, reliable


def _estimate_local_planes(depth, intrinsics, center_x, center_y, config,
                           allow_fallback):
    batch_size = config["geometry_batch_size"]
    outputs = []
    for start in range(0, center_x.shape[0], batch_size):
        end = min(start + batch_size, center_x.shape[0])
        outputs.append(
            _estimate_local_planes_batch(depth, intrinsics,
                                         center_x[start:end],
                                         center_y[start:end],
                                         config["normal_window"], config))
    center_points = torch.cat([output[0] for output in outputs], dim=0)
    normals = torch.cat([output[1] for output in outputs], dim=0)
    residual_sigmas = torch.cat([output[2] for output in outputs], dim=0)
    reliable = torch.cat([output[3] for output in outputs], dim=0)

    if allow_fallback and (~reliable).any():
        fallback_indices = torch.nonzero(~reliable, as_tuple=False).squeeze(1)
        fallback_outputs = []
        for start in range(0, fallback_indices.shape[0], batch_size):
            indices = fallback_indices[start:start + batch_size]
            fallback_outputs.append(
                _estimate_local_planes_batch(
                    depth, intrinsics, center_x[indices], center_y[indices],
                    config["fallback_normal_window"], config))
        fallback_normals = torch.cat(
            [output[1] for output in fallback_outputs], dim=0)
        fallback_residuals = torch.cat(
            [output[2] for output in fallback_outputs], dim=0)
        fallback_reliable = torch.cat(
            [output[3] for output in fallback_outputs], dim=0)
        accepted_indices = fallback_indices[fallback_reliable]
        normals[accepted_indices] = fallback_normals[fallback_reliable]
        residual_sigmas[accepted_indices] = fallback_residuals[
            fallback_reliable]
        reliable[accepted_indices] = True

    if allow_fallback and (~reliable).any():
        fallback_indices = torch.nonzero(~reliable, as_tuple=False).squeeze(1)
        normals[fallback_indices] = -F.normalize(
            center_points[fallback_indices], dim=1)
        residual_sigmas[fallback_indices] = 0.0
        reliable[fallback_indices] = True

    return center_points, normals, residual_sigmas, reliable


def _triangle_moments(vertex_a, vertex_b, vertex_c):
    cross = torch.cross(vertex_b - vertex_a,
                        vertex_c - vertex_a,
                        dim=1)
    area = 0.5 * torch.linalg.vector_norm(cross, dim=1)
    centroid = (vertex_a + vertex_b + vertex_c) / 3.0
    deviations = torch.stack((vertex_a - centroid, vertex_b - centroid,
                              vertex_c - centroid),
                             dim=1)
    covariance = torch.einsum("bki,bkj->bij", deviations,
                              deviations) / 12.0
    return area, centroid, covariance


def _footprint_geometry_batch(center_points, normals, residual_sigmas,
                              pixel_bounds, intrinsics, w2c, config):
    x_min = pixel_bounds[:, 0] - 0.5
    y_min = pixel_bounds[:, 1] - 0.5
    x_max = pixel_bounds[:, 2] - 0.5
    y_max = pixel_bounds[:, 3] - 0.5
    corner_x = torch.stack((x_min, x_max, x_max, x_min), dim=1)
    corner_y = torch.stack((y_min, y_min, y_max, y_max), dim=1)
    ones = torch.ones_like(corner_x)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    rays = torch.stack(((corner_x - cx) / fx, (corner_y - cy) / fy,
                        ones),
                       dim=-1)
    denominator = torch.sum(normals[:, None, :] * rays, dim=-1)
    ray_norm = torch.linalg.vector_norm(rays, dim=-1)
    incidence = torch.abs(denominator) / ray_norm.clamp_min(
        config["min_scale"])
    safe_denominator = torch.where(
        torch.abs(denominator) > config["min_scale"], denominator,
        torch.ones_like(denominator))
    distance = torch.sum(normals * center_points,
                         dim=1)[:, None] / safe_denominator
    corners = rays * distance[..., None]

    area_1, centroid_1, covariance_1 = _triangle_moments(
        corners[:, 0], corners[:, 1], corners[:, 2])
    area_2, centroid_2, covariance_2 = _triangle_moments(
        corners[:, 0], corners[:, 2], corners[:, 3])
    total_area = area_1 + area_2
    safe_area = total_area.clamp_min(config["min_scale"]**2)
    footprint_centroid = (
        area_1[:, None] * centroid_1 +
        area_2[:, None] * centroid_2) / safe_area[:, None]
    delta_1 = centroid_1 - footprint_centroid
    delta_2 = centroid_2 - footprint_centroid
    footprint_covariance = (
        area_1[:, None, None] *
        (covariance_1 + delta_1[:, :, None] * delta_1[:, None, :]) +
        area_2[:, None, None] *
        (covariance_2 + delta_2[:, :, None] * delta_2[:, None, :])
    ) / safe_area[:, None, None]

    _, eigenvectors = torch.linalg.eigh(footprint_covariance)
    tangent_x = eigenvectors[:, :, 2]
    tangent_x = tangent_x - torch.sum(
        tangent_x * normals, dim=1)[:, None] * normals
    tangent_x = F.normalize(tangent_x, dim=1)
    reference_x = torch.tensor([1.0, 0.0, 0.0],
                               device=normals.device,
                               dtype=normals.dtype).expand_as(normals)
    reference_x = reference_x - torch.sum(
        reference_x * normals, dim=1)[:, None] * normals
    weak_reference = torch.linalg.vector_norm(reference_x, dim=1) < 1e-4
    reference_y = torch.tensor([0.0, 1.0, 0.0],
                               device=normals.device,
                               dtype=normals.dtype).expand_as(normals)
    reference_y = reference_y - torch.sum(
        reference_y * normals, dim=1)[:, None] * normals
    reference_x = torch.where(weak_reference[:, None], reference_y,
                              reference_x)
    reference_x = F.normalize(reference_x, dim=1)
    tangent_x = torch.where(
        (torch.sum(tangent_x * reference_x, dim=1) < 0)[:, None], -tangent_x,
        tangent_x)
    tangent_y = F.normalize(torch.cross(normals, tangent_x, dim=1), dim=1)
    tangent_x = F.normalize(torch.cross(tangent_y, normals, dim=1), dim=1)

    variance_x = torch.einsum("bi,bij,bj->b", tangent_x,
                              footprint_covariance, tangent_x)
    variance_y = torch.einsum("bi,bij,bj->b", tangent_y,
                              footprint_covariance, tangent_y)
    scale_x = torch.sqrt(variance_x.clamp_min(config["min_scale"]**2))
    scale_y = torch.sqrt(variance_y.clamp_min(config["min_scale"]**2))
    min_tangent_scale = torch.minimum(scale_x, scale_y)
    min_normal_scale = torch.maximum(
        min_tangent_scale * config["normal_scale_min_ratio"],
        torch.full_like(min_tangent_scale, config["min_scale"]))
    max_normal_scale = min_tangent_scale * config[
        "normal_scale_max_ratio"]
    scale_z = torch.maximum(
        min_normal_scale, torch.minimum(residual_sigmas, max_normal_scale))
    scales = torch.stack((scale_x, scale_y, scale_z), dim=1)

    rotation_camera = torch.stack((tangent_x, tangent_y, normals), dim=-1)
    c2w = torch.linalg.inv(w2c)
    rotation_world = torch.einsum("ij,bjk->bik", c2w[:3, :3],
                                  rotation_camera)
    rotations = F.normalize(matrix_to_quaternion(rotation_world), dim=1)
    rotations = torch.where((rotations[:, 0] < 0)[:, None], -rotations,
                            rotations)
    means_world = torch.einsum("ij,bj->bi", c2w[:3, :3],
                               footprint_centroid) + c2w[:3, 3]

    valid = (incidence >= config["min_view_cos"]).all(dim=1)
    valid &= (distance > 0).all(dim=1)
    valid &= total_area > config["min_scale"]**2
    valid &= torch.isfinite(means_world).all(dim=1)
    valid &= torch.isfinite(scales).all(dim=1)
    valid &= torch.isfinite(rotations).all(dim=1)
    return means_world, scales, rotations, valid


def _footprint_geometry(center_points, normals, residual_sigmas,
                        pixel_bounds, intrinsics, w2c, config):
    batch_size = config["geometry_batch_size"]
    outputs = []
    for start in range(0, center_points.shape[0], batch_size):
        end = min(start + batch_size, center_points.shape[0])
        outputs.append(
            _footprint_geometry_batch(center_points[start:end],
                                      normals[start:end],
                                      residual_sigmas[start:end],
                                      pixel_bounds[start:end], intrinsics, w2c,
                                      config))
    return tuple(
        torch.cat([output[index] for output in outputs], dim=0)
        for index in range(4))


def build_surface_gaussian_geometry(depth,
                                    intrinsics,
                                    w2c,
                                    center_pixels,
                                    pixel_bounds,
                                    surface_init_config=None,
                                    allow_fallback=False):
    """Build world-space means, scales, and rotations from image footprints."""
    config = _surface_init_config(surface_init_config)
    if center_pixels.shape[0] == 0:
        empty_means = depth.new_empty((0, 3))
        empty_scales = depth.new_empty((0, 3))
        empty_rotations = depth.new_empty((0, 4))
        empty_valid = torch.empty((0,), device=depth.device, dtype=torch.bool)
        return empty_means, empty_scales, empty_rotations, empty_valid
    center_x = center_pixels[:, 0].long()
    center_y = center_pixels[:, 1].long()
    center_points, normals, residual_sigmas, reliable = _estimate_local_planes(
        depth, intrinsics, center_x, center_y, config, allow_fallback)
    means, scales, rotations, footprint_valid = _footprint_geometry(
        center_points, normals, residual_sigmas, pixel_bounds, intrinsics, w2c,
        config)
    return means, scales, rotations, reliable & footprint_valid


def _quadtree_plane_consistency(depth, intrinsics, pixel_bounds,
                                center_points, normals, config):
    fractions = torch.tensor([0.0, 0.5, 1.0],
                             device=depth.device,
                             dtype=depth.dtype)
    x_min = pixel_bounds[:, 0]
    y_min = pixel_bounds[:, 1]
    x_span = (pixel_bounds[:, 2] - pixel_bounds[:, 0] - 1).clamp_min(0)
    y_span = (pixel_bounds[:, 3] - pixel_bounds[:, 1] - 1).clamp_min(0)
    sample_y, sample_x = torch.meshgrid(fractions, fractions, indexing="ij")
    sample_x = torch.round(x_min[:, None] +
                           x_span[:, None] * sample_x.reshape(1, -1)).long()
    sample_y = torch.round(y_min[:, None] +
                           y_span[:, None] * sample_y.reshape(1, -1)).long()
    sample_depth = depth[0, sample_y, sample_x]
    valid_depth = sample_depth > 0
    sample_points = _unproject(sample_depth, sample_x.to(depth.dtype),
                               sample_y.to(depth.dtype), intrinsics)
    residual = torch.abs(
        torch.sum((sample_points - center_points[:, None, :]) *
                  normals[:, None, :],
                  dim=-1))
    center_depth = center_points[:, 2]
    threshold = torch.maximum(
        torch.full_like(center_depth, config["depth_abs_thresh"]),
        center_depth * config["depth_rel_thresh"])
    inliers = valid_depth & (residual <= threshold[:, None])
    valid_fraction = valid_depth.to(depth.dtype).mean(dim=1)
    inlier_fraction = inliers.sum(dim=1).to(depth.dtype) / valid_depth.sum(
        dim=1).clamp_min(1).to(depth.dtype)
    return ((valid_fraction >= config["node_min_valid_fraction"]) &
            (inlier_fraction >= config["node_min_inlier_fraction"]))


def _anisotropic_quadtree_gaussians(color, depth, quadtree, intrinsics, w2c,
                                    mask, surface_init_config):
    config = _surface_init_config(surface_init_config)
    bounds = quadtree.detach().to(device=depth.device, dtype=depth.dtype)
    center_x = torch.div(bounds[:, 0] + bounds[:, 2] - 1,
                         2,
                         rounding_mode="floor").long()
    center_y = torch.div(bounds[:, 1] + bounds[:, 3] - 1,
                         2,
                         rounding_mode="floor").long()
    candidate = depth[0, center_y, center_x] > 0
    if mask is not None:
        candidate &= mask[center_y, center_x]
    bounds = bounds[candidate]
    center_x = center_x[candidate]
    center_y = center_y[candidate]
    center_pixels = torch.stack((center_x, center_y), dim=1)
    if center_pixels.shape[0] == 0:
        return (depth.new_empty((0, 6)), depth.new_empty((0, 3)),
                depth.new_empty((0, 4)))

    center_points, normals, residual_sigmas, reliable = _estimate_local_planes(
        depth, intrinsics, center_x, center_y, config, allow_fallback=False)
    consistent = _quadtree_plane_consistency(depth, intrinsics, bounds,
                                             center_points, normals, config)
    means, scales, rotations, footprint_valid = _footprint_geometry(
        center_points, normals, residual_sigmas, bounds, intrinsics, w2c,
        config)
    valid = reliable & consistent & footprint_valid
    colors = color[:, center_y, center_x].transpose(0, 1)
    point_cloud = torch.cat((means[valid], colors[valid]), dim=1)
    return point_cloud, scales[valid], rotations[valid]


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
                            params=None,
                            gaussian_distribution="isotropic",
                            surface_init_config=None):
    """Build Gaussian seeds from quadtree leaf footprints."""
    del time_idx, scene_name
    with torch.no_grad():
        if gaussian_distribution == "anisotropic":
            if not transform_pts or not compute_mean_sq_dist:
                raise ValueError(
                    "anisotropic quadtree seeds require transformed points and scales"
                )
            return _anisotropic_quadtree_gaussians(
                color, depth, quadtree, intrinsics, w2c, mask,
                surface_init_config)
        if gaussian_distribution != "isotropic":
            raise ValueError(
                f"Unknown gaussian_distribution {gaussian_distribution}")

        bounds = quadtree.detach()
        center_x = torch.div(bounds[:, 0] + bounds[:, 2],
                             2,
                             rounding_mode="floor").long()
        center_y = torch.div(bounds[:, 1] + bounds[:, 3],
                             2,
                             rounding_mode="floor").long()
        depth_z = depth[0, center_y, center_x]
        points_camera = _unproject(depth_z, center_x.to(depth.dtype),
                                   center_y.to(depth.dtype), intrinsics)
        if transform_pts:
            c2w = torch.linalg.inv(w2c)
            points = torch.einsum("ij,bj->bi", c2w[:3, :3],
                                  points_camera) + c2w[:3, 3]
        else:
            points = points_camera

        selected = torch.ones(center_x.shape[0],
                              device=center_x.device,
                              dtype=torch.bool)
        if mask is not None:
            selected &= mask[center_y, center_x]
        colors = color[:, center_y, center_x].transpose(0, 1)
        point_cloud = torch.cat((points, colors), dim=1)[selected]
        if not compute_mean_sq_dist:
            return point_cloud
        if mean_sq_dist_method != "projective":
            raise ValueError(
                f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

        pixel_scale = depth_z / ((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
        node_diagonal = torch.sqrt((bounds[:, 2] - bounds[:, 0])**2 +
                                   (bounds[:, 3] - bounds[:, 1])**2)
        scales = pixel_scale * node_diagonal * 0.5
        if params is not None and selected.any():
            new_points = points[selected]
            max_old_points = int(10**9 / max(new_points.shape[0], 1))
            num_old_points = min(params["log_scales"].shape[0],
                                 max_old_points)
            if num_old_points > 0:
                old_scales = torch.exp(
                    params["log_scales"][-num_old_points:].detach()).max(
                        dim=1).values
                old_points = params["means3D"][-num_old_points:].detach()
                distances = torch.cdist(new_points, old_points)
                closest_distance, closest_index = torch.min(distances, dim=1)
                neighbor_scale = closest_distance - old_scales[closest_index]
                selected_pixel_scale = pixel_scale[selected]
                scales[selected] = torch.maximum(
                    torch.minimum(scales[selected], neighbor_scale),
                    selected_pixel_scale)
        scales = scales[selected, None]
        rotations = _identity_rotations(point_cloud.shape[0], depth.device,
                                        depth.dtype)
        return point_cloud, scales, rotations


def _anisotropic_dense_gaussians(color, depth, intrinsics, w2c, mask,
                                 surface_init_config):
    height, width = depth.shape[1:]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy")
    selected = depth[0] > 0
    if mask is not None:
        selected &= mask.reshape(height, width)
    center_x = x_grid[selected]
    center_y = y_grid[selected]
    center_pixels = torch.stack((center_x, center_y), dim=1)
    pixel_bounds = torch.stack((center_x, center_y, center_x + 1,
                                center_y + 1),
                               dim=1).to(depth.dtype)
    means, scales, rotations, valid = build_surface_gaussian_geometry(
        depth,
        intrinsics,
        w2c,
        center_pixels,
        pixel_bounds,
        surface_init_config=surface_init_config,
        allow_fallback=True)
    colors = color[:, center_y, center_x].transpose(0, 1)
    point_cloud = torch.cat((means[valid], colors[valid]), dim=1)
    return point_cloud, scales[valid], rotations[valid]


def get_pointcloud(color,
                   depth,
                   intrinsics,
                   w2c,
                   transform_pts=True,
                   mask=None,
                   compute_mean_sq_dist=False,
                   mean_sq_dist_method="projective",
                   gaussian_distribution="isotropic",
                   surface_init_config=None):
    """Build Gaussian seeds from dense RGB-D pixels."""
    with torch.no_grad():
        if gaussian_distribution == "anisotropic":
            if not transform_pts or not compute_mean_sq_dist:
                raise ValueError(
                    "anisotropic dense seeds require transformed points and scales"
                )
            return _anisotropic_dense_gaussians(color, depth, intrinsics, w2c,
                                                mask, surface_init_config)
        if gaussian_distribution != "isotropic":
            raise ValueError(
                f"Unknown gaussian_distribution {gaussian_distribution}")

        height, width = depth.shape[1:]
        x_grid, y_grid = torch.meshgrid(
            torch.arange(width, device=slam_context.device).float(),
            torch.arange(height, device=slam_context.device).float(),
            indexing="xy")
        depth_z = depth[0].reshape(-1)
        points_camera = _unproject(depth_z, x_grid.reshape(-1),
                                   y_grid.reshape(-1), intrinsics)
        if transform_pts:
            c2w = torch.linalg.inv(w2c)
            points = torch.einsum("ij,bj->bi", c2w[:3, :3],
                                  points_camera) + c2w[:3, 3]
        else:
            points = points_camera
        colors = color.permute(1, 2, 0).reshape(-1, 3)
        point_cloud = torch.cat((points, colors), dim=1)

        selected = None
        if mask is not None:
            selected = mask.reshape(-1)
            point_cloud = point_cloud[selected]
        if not compute_mean_sq_dist:
            return point_cloud
        if mean_sq_dist_method != "projective":
            raise ValueError(
                f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        scales = depth_z / ((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
        if selected is not None:
            scales = scales[selected]
        scales = scales[:, None]
        rotations = _identity_rotations(point_cloud.shape[0], depth.device,
                                        depth.dtype)
        return point_cloud, scales, rotations
