"""Regularization for surface-aligned anisotropic Gaussians."""

import math

import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation
from utils.slam_helpers import matrix_to_quaternion


def _is_enabled(params, config):
    return (config is not None and config.get("enabled", False) and
            params["log_scales"].shape[1] == 3 and
            "surface_normals" in params)


def surface_regularization_losses(params, visible, config):
    """Return thickness and reference-normal losses for visible Gaussians."""
    if not _is_enabled(params, config):
        return {}

    visible = visible.detach()
    if not visible.any():
        zero = params["log_scales"].sum() * 0.0
        return {"surface_thickness": zero, "surface_normal": zero}

    log_scales = params["log_scales"][visible]
    log_tangent_min = torch.minimum(log_scales[:, 0],
                                    log_scales[:, 1]).detach()
    log_ratio = log_scales[:, 2] - log_tangent_min
    log_min_ratio = math.log(config["min_normal_to_tangent_ratio"])
    log_max_ratio = math.log(config["max_normal_to_tangent_ratio"])
    below = F.relu(log_min_ratio - log_ratio)
    above = F.relu(log_ratio - log_max_ratio)
    thickness_loss = (below.square() + above.square()).mean()

    rotations = build_rotation(
        F.normalize(params["unnorm_rotations"][visible], dim=1))
    current_normals = rotations[:, :, 2]
    reference_normals = F.normalize(
        params["surface_normals"][visible].detach(), dim=1)
    cosine = torch.sum(current_normals * reference_normals,
                       dim=1).clamp(-1.0, 1.0)
    normal_loss = (1.0 - cosine).mean()
    return {
        "surface_thickness": thickness_loss,
        "surface_normal": normal_loss,
    }


def project_surface_scales(params, config):
    """Hard-project local z scale into the configured tangent-relative range."""
    if not _is_enabled(params, config):
        return
    with torch.no_grad():
        log_scales = params["log_scales"]
        log_tangent_min = torch.minimum(log_scales[:, 0], log_scales[:, 1])
        lower = log_tangent_min + math.log(
            config["min_normal_to_tangent_ratio"])
        upper = log_tangent_min + math.log(
            config["max_normal_to_tangent_ratio"])
        projected = torch.maximum(log_scales[:, 2], lower)
        projected = torch.minimum(projected, upper)
        log_scales[:, 2].copy_(projected)


def project_surface_normals(params, config):
    """Project local z axes into a cone around their initialization normals."""
    if not _is_enabled(params, config):
        return
    with torch.no_grad():
        quaternions = F.normalize(params["unnorm_rotations"], dim=1)
        rotations = build_rotation(quaternions)
        current_normals = rotations[:, :, 2]
        reference_normals = F.normalize(params["surface_normals"], dim=1)
        cosine = torch.sum(current_normals * reference_normals,
                           dim=1).clamp(-1.0, 1.0)
        max_angle = math.radians(config["max_normal_deviation_degrees"])
        outside = cosine < math.cos(max_angle)
        if not outside.any():
            return

        current_rotation = rotations[outside]
        current_normal = current_normals[outside]
        reference_normal = reference_normals[outside]
        tangent_direction = current_normal - torch.sum(
            current_normal * reference_normal,
            dim=1)[:, None] * reference_normal
        tangent_norm = torch.linalg.vector_norm(tangent_direction, dim=1)
        fallback_direction = current_rotation[:, :, 0]
        fallback_direction = fallback_direction - torch.sum(
            fallback_direction * reference_normal,
            dim=1)[:, None] * reference_normal
        tangent_direction = torch.where(
            (tangent_norm < 1e-6)[:, None], fallback_direction,
            tangent_direction)
        tangent_direction = F.normalize(tangent_direction, dim=1)
        projected_normal = (math.cos(max_angle) * reference_normal +
                            math.sin(max_angle) * tangent_direction)

        projected_x = current_rotation[:, :, 0]
        projected_x = projected_x - torch.sum(
            projected_x * projected_normal,
            dim=1)[:, None] * projected_normal
        weak_x = torch.linalg.vector_norm(projected_x, dim=1) < 1e-6
        fallback_x = current_rotation[:, :, 1]
        fallback_x = fallback_x - torch.sum(
            fallback_x * projected_normal,
            dim=1)[:, None] * projected_normal
        projected_x = torch.where(weak_x[:, None], fallback_x, projected_x)
        projected_x = F.normalize(projected_x, dim=1)
        projected_y = F.normalize(
            torch.cross(projected_normal, projected_x, dim=1), dim=1)
        projected_x = F.normalize(
            torch.cross(projected_y, projected_normal, dim=1), dim=1)
        projected_rotation = torch.stack(
            (projected_x, projected_y, projected_normal), dim=-1)
        projected_quaternion = F.normalize(
            matrix_to_quaternion(projected_rotation), dim=1)
        current_quaternion = quaternions[outside]
        projected_quaternion = torch.where(
            (torch.sum(projected_quaternion * current_quaternion, dim=1) <
             0)[:, None], -projected_quaternion, projected_quaternion)
        outside_indices = torch.nonzero(outside, as_tuple=False).squeeze(1)
        params["unnorm_rotations"].index_copy_(0, outside_indices,
                                               projected_quaternion)


def project_surface_geometry(params, config):
    """Apply hard thickness and normal-orientation constraints."""
    project_surface_scales(params, config)
    project_surface_normals(params, config)
