import math

import torch

from qcg_slam import context as slam_context
from qcg_slam.pointcloud import (
    build_surface_gaussian_geometry,
    get_pointcloud,
    get_quadtree_pointcloud,
)
from qcg_slam.parameters import initialize_new_params
from utils.slam_external import build_rotation


def _surface_config():
    return {
        "geometry_batch_size": 64,
        "depth_abs_thresh": 0.02,
        "depth_rel_thresh": 0.02,
    }


def _camera(width=15, height=15):
    return torch.tensor([[100.0, 0.0, (width - 1) / 2],
                         [0.0, 100.0, (height - 1) / 2],
                         [0.0, 0.0, 1.0]])


def test_front_parallel_pixel_matches_surface_moments():
    depth = torch.full((1, 15, 15), 2.0)
    intrinsics = _camera()
    means, scales, rotations, valid = build_surface_gaussian_geometry(
        depth,
        intrinsics,
        torch.eye(4),
        torch.tensor([[7, 7]]),
        torch.tensor([[7.0, 7.0, 8.0, 8.0]]),
        surface_init_config=_surface_config(),
        allow_fallback=False,
    )

    rotation = build_rotation(rotations)[0]
    expected_sigma = 2.0 / 100.0 / math.sqrt(12.0)
    assert valid.item()
    assert torch.allclose(means[0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-5)
    assert torch.allclose(scales[0, :2],
                          torch.full((2,), expected_sigma),
                          atol=1e-5)
    assert scales[0, 2] < scales[0, 0]
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(rotation), torch.ones(()), atol=1e-5)


def test_slanted_plane_normal_and_world_rotation():
    height = width = 15
    intrinsics = _camera(width, height)
    y_grid, x_grid = torch.meshgrid(torch.arange(height).float(),
                                    torch.arange(width).float(),
                                    indexing="ij")
    normal_camera = torch.tensor([0.2, 0.1, -0.9746794])
    point_camera = torch.tensor([0.0, 0.0, 2.0])
    plane_offset = torch.dot(normal_camera, point_camera)
    rays = torch.stack(((x_grid - 7.0) / 100.0,
                        (y_grid - 7.0) / 100.0,
                        torch.ones_like(x_grid)),
                       dim=-1)
    depth = (plane_offset / torch.sum(rays * normal_camera, dim=-1))[None]
    angle = math.pi / 3.0
    camera_to_world = torch.tensor([[math.cos(angle), 0.0,
                                     math.sin(angle), 0.2],
                                    [0.0, 1.0, 0.0, -0.1],
                                    [-math.sin(angle), 0.0,
                                     math.cos(angle), 0.3],
                                    [0.0, 0.0, 0.0, 1.0]])
    w2c = torch.linalg.inv(camera_to_world)
    means, scales, rotations, valid = build_surface_gaussian_geometry(
        depth,
        intrinsics,
        w2c,
        torch.tensor([[7, 7]]),
        torch.tensor([[7.0, 7.0, 9.0, 9.0]]),
        surface_init_config=_surface_config(),
        allow_fallback=False,
    )

    rotation = build_rotation(rotations)[0]
    expected_world_normal = camera_to_world[:3, :3] @ normal_camera
    expected_world_point = (camera_to_world[:3, :3] @ point_camera +
                            camera_to_world[:3, 3])
    assert valid.item()
    assert torch.abs(torch.dot(rotation[:, 2], expected_world_normal)) > 0.999
    assert torch.abs(
        torch.dot(expected_world_normal,
                  means[0] - expected_world_point)) < 1e-5
    assert torch.all(scales > 0)
    assert torch.allclose(torch.det(rotation), torch.ones(()), atol=1e-5)


def test_depth_boundary_quadtree_node_is_skipped():
    slam_context.set_device("cpu")
    depth = torch.ones((1, 9, 9))
    depth[:, :, 4:] = 2.0
    color = torch.zeros((3, 9, 9))
    quadtree = torch.tensor([[0, 0, 9, 9]])
    point_cloud, scales, rotations = get_quadtree_pointcloud(
        color,
        depth,
        quadtree,
        _camera(9, 9),
        torch.eye(4),
        mask=depth[0] > 0,
        compute_mean_sq_dist=True,
        gaussian_distribution="anisotropic",
        surface_init_config=_surface_config(),
    )
    assert point_cloud.shape == (0, 6)
    assert scales.shape == (0, 3)
    assert rotations.shape == (0, 4)


def test_dense_invalid_normal_uses_view_facing_fallback():
    slam_context.set_device("cpu")
    depth = torch.zeros((1, 5, 5))
    depth[0, 2, 2] = 2.0
    color = torch.ones((3, 5, 5))
    point_cloud, scales, rotations = get_pointcloud(
        color,
        depth,
        torch.tensor([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0],
                      [0.0, 0.0, 1.0]]),
        torch.eye(4),
        mask=depth[0] > 0,
        compute_mean_sq_dist=True,
        gaussian_distribution="anisotropic",
        surface_init_config=_surface_config(),
    )
    assert point_cloud.shape == (1, 6)
    assert torch.isfinite(point_cloud).all()
    assert torch.isfinite(scales).all()
    assert torch.isfinite(rotations).all()
    assert scales.shape == (1, 3)
    assert torch.all(scales > 0)


def test_isotropic_dense_initialization_keeps_legacy_scale_shape():
    slam_context.set_device("cpu")
    depth = torch.full((1, 3, 3), 2.0)
    color = torch.zeros((3, 3, 3))
    mask = torch.zeros((3, 3), dtype=torch.bool)
    mask[:, 1] = True
    point_cloud, scales, rotations = get_pointcloud(
        color,
        depth,
        torch.tensor([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0],
                      [0.0, 0.0, 1.0]]),
        torch.eye(4),
        mask=mask,
        compute_mean_sq_dist=True,
        gaussian_distribution="isotropic",
    )
    assert point_cloud.shape == (3, 6)
    assert scales.shape == (3, 1)
    assert rotations.shape == (3, 4)
    assert torch.allclose(scales, torch.full((3, 1), 0.02))
    assert torch.allclose(rotations[:, 0], torch.ones(3))


def test_anisotropic_parameters_reject_scalar_scales():
    slam_context.set_device("cpu")
    point_cloud = torch.zeros((1, 6))
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    try:
        initialize_new_params(point_cloud, torch.ones((1, 1)), rotations,
                              "anisotropic")
    except ValueError as error:
        assert "shape (1, 3)" in str(error)
    else:
        raise AssertionError("anisotropic scalar scales must be rejected")
