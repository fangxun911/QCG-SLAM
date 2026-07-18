import math
import tempfile

import numpy as np
import torch

from qcg_slam import context as slam_context
from qcg_slam.checkpoints import load_checkpoint_state
from qcg_slam.optimization import initialize_optimizer
from qcg_slam.parameters import initialize_params
from qcg_slam.surface_regularization import (
    project_surface_normals,
    project_surface_scales,
    surface_regularization_losses,
)
from utils.slam_external import cat_params_to_optimizer, remove_points


def _config():
    return {
        "enabled": True,
        "min_normal_to_tangent_ratio": 0.05,
        "max_normal_to_tangent_ratio": 0.25,
        "max_normal_deviation_degrees": 15.0,
        "thickness_weight": 0.1,
        "normal_weight": 0.05,
    }


def test_surface_losses_penalize_thickness_and_normal_drift():
    half_sqrt_two = math.sqrt(0.5)
    params = {
        "log_scales": torch.nn.Parameter(torch.zeros((1, 3))),
        "unnorm_rotations": torch.nn.Parameter(
            torch.tensor([[half_sqrt_two, half_sqrt_two, 0.0, 0.0]])),
        "surface_normals": torch.tensor([[0.0, 0.0, 1.0]]),
    }
    losses = surface_regularization_losses(params, torch.tensor([True]),
                                            _config())
    assert losses["surface_thickness"] > 0
    assert torch.allclose(losses["surface_normal"], torch.ones(()), atol=1e-6)

    total = losses["surface_thickness"] + losses["surface_normal"]
    total.backward()
    assert params["log_scales"].grad[0, 2] > 0
    assert torch.isfinite(params["unnorm_rotations"].grad).all()
    assert torch.linalg.vector_norm(params["unnorm_rotations"].grad) > 0


def test_scale_projection_enforces_surface_ratio_bounds():
    params = {
        "log_scales": torch.nn.Parameter(
            torch.log(torch.tensor([[2.0, 1.0, 2.0],
                                    [4.0, 2.0, 0.01]]))),
        "surface_normals": torch.tensor([[0.0, 0.0, 1.0],
                                         [0.0, 0.0, 1.0]]),
    }
    project_surface_scales(params, _config())
    scales = torch.exp(params["log_scales"])
    ratio = scales[:, 2] / torch.minimum(scales[:, 0], scales[:, 1])
    assert torch.allclose(ratio, torch.tensor([0.25, 0.05]), atol=1e-6)


def test_normal_projection_preserves_rotation_inside_reference_cone():
    half_sqrt_two = math.sqrt(0.5)
    params = {
        "log_scales": torch.nn.Parameter(torch.zeros((1, 3))),
        "unnorm_rotations": torch.nn.Parameter(
            torch.tensor([[half_sqrt_two, half_sqrt_two, 0.0, 0.0]])),
        "surface_normals": torch.tensor([[0.0, 0.0, 1.0]]),
    }
    project_surface_normals(params, _config())
    from utils.slam_external import build_rotation
    rotation = build_rotation(params["unnorm_rotations"].detach())[0]
    angle = torch.rad2deg(torch.acos(rotation[2, 2].clamp(-1.0, 1.0)))
    assert angle <= 15.001
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(rotation), torch.ones(()), atol=1e-5)


def test_reference_normals_are_frozen_and_not_optimized():
    slam_context.set_device("cpu")
    point_cloud = torch.zeros((2, 6))
    scales = torch.tensor([[1.0, 0.5, 0.1], [0.8, 0.4, 0.1]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]])
    params, _ = initialize_params(point_cloud, 1, scales, rotations,
                                  "anisotropic")
    assert params["surface_normals"].shape == (2, 3)
    assert not params["surface_normals"].requires_grad

    learning_rates = {
        "means3D": 0.0,
        "rgb_colors": 0.0,
        "unnorm_rotations": 0.001,
        "logit_opacities": 0.0,
        "log_scales": 0.001,
        "cam_unnorm_rots": 0.0,
        "cam_trans": 0.0,
    }
    optimizer = initialize_optimizer(params, learning_rates, tracking=False)
    group_names = {group["name"] for group in optimizer.param_groups}
    assert "surface_normals" not in group_names


def test_reference_normals_follow_gaussian_addition_and_removal():
    slam_context.set_device("cpu")
    point_cloud = torch.zeros((2, 6))
    scales = torch.tensor([[1.0, 0.5, 0.1], [0.8, 0.4, 0.1]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]])
    params, variables = initialize_params(point_cloud, 1, scales, rotations,
                                          "anisotropic")
    learning_rates = {
        "means3D": 0.0,
        "rgb_colors": 0.0,
        "unnorm_rotations": 0.001,
        "logit_opacities": 0.0,
        "log_scales": 0.001,
        "cam_unnorm_rots": 0.0,
        "cam_trans": 0.0,
    }
    optimizer = initialize_optimizer(params, learning_rates, tracking=False)
    new_params = {
        key: value[:1].detach().clone()
        for key, value in params.items()
        if key not in ("cam_unnorm_rots", "cam_trans")
    }
    params = cat_params_to_optimizer(new_params, params, optimizer)
    assert params["means3D"].shape[0] == 3
    assert params["surface_normals"].shape == (3, 3)
    assert not params["surface_normals"].requires_grad

    for key in ("means2D_gradient_accum", "denom", "max_2D_radius",
                "timestep"):
        variables[key] = torch.zeros(3)
    params, variables = remove_points(torch.tensor([False, True, False]),
                                      params, variables, optimizer)
    assert params["means3D"].shape[0] == 2
    assert params["surface_normals"].shape == (2, 3)
    assert not params["surface_normals"].requires_grad


def test_old_anisotropic_checkpoint_derives_reference_normals():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_params = {
            "means3D": np.zeros((1, 3), dtype=np.float32),
            "rgb_colors": np.zeros((1, 3), dtype=np.float32),
            "unnorm_rotations": np.array([[1.0, 0.0, 0.0, 0.0]],
                                         dtype=np.float32),
            "logit_opacities": np.zeros((1, 1), dtype=np.float32),
            "log_scales": np.zeros((1, 3), dtype=np.float32),
            "cam_unnorm_rots": np.array([[[1.0], [0.0], [0.0], [0.0]]],
                                         dtype=np.float32),
            "cam_trans": np.zeros((1, 3, 1), dtype=np.float32),
        }
        np.savez(f"{directory}/params0.npz", **checkpoint_params)
        np.save(f"{directory}/keyframe_time_indices0.npy",
                np.array([], dtype=np.int64))
        config = {
            "load_checkpoint": True,
            "checkpoint_time_idx": 0,
            "workdir": directory,
            "run_name": "",
            "gaussian_distribution": "anisotropic",
        }
        variables = {}
        params, variables, _, _, _, _ = load_checkpoint_state(
            config, None, {}, variables, torch.device("cpu"))
        assert params["surface_normals"].shape == (1, 3)
        assert torch.allclose(params["surface_normals"],
                              torch.tensor([[0.0, 0.0, 1.0]]))
        assert not params["surface_normals"].requires_grad
