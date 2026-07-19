"""Densification orchestration for online QCG-SLAM mapping."""

import os
from dataclasses import dataclass

from utils.slam_external import densify

from qcg_slam.datasets import get_dataset
from qcg_slam.gaussians import add_coarse_gaussians, add_fine_gaussians


@dataclass
class DensificationResources:
    """Optional dataset and camera state for a densification resolution."""

    dataset: object = None
    camera: object = None
    intrinsics: object = None


def load_densification_dataset(gradslam_data_cfg, dataset_config, device):
    """Load the optional dataset used at a dedicated densification resolution."""
    return get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["densification_image_height"],
        desired_width=dataset_config["densification_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )


def _prepare_coarse_densification_data(curr_data, time_idx, resources):
    """Prepare current-frame data for pre-coarse densification."""
    if resources.dataset is None:
        return curr_data

    frame = resources.dataset[time_idx]
    densify_color, densify_depth = frame[:2]
    densify_color = densify_color.permute(2, 0, 1) / 255
    densify_depth = densify_depth.permute(2, 0, 1)
    densify_quadtree = frame[2] if len(frame) >= 5 else curr_data["quadtree"]
    return {
        "cam": resources.camera,
        "im": densify_color,
        "depth": densify_depth,
        "quadtree": densify_quadtree,
        "id": time_idx,
        "intrinsics": resources.intrinsics,
        "w2c": curr_data["w2c"],
        "iter_gt_w2c_list": curr_data["iter_gt_w2c_list"],
    }


def add_coarse_frame_gaussians(
    params,
    variables,
    curr_data,
    time_idx,
    config,
    resources,
    wandb_run,
    wandb_time_step,
):
    """Add silhouette-based Gaussians immediately before coarse mapping."""
    if not config["mapping"]["add_new_gaussians"] or time_idx == 0:
        return params, variables

    densification_data = _prepare_coarse_densification_data(
        curr_data, time_idx, resources
    )
    params, variables = add_coarse_gaussians(
        params,
        variables,
        densification_data,
        config["mapping"]["sil_thres"],
        time_idx,
        config["mean_sq_dist_method"],
        config["gaussian_distribution"],
        config["data"]["sequence"],
        config["surface_init"],
    )
    if config["use_wandb"]:
        wandb_run.log(
            {
                "Mapping/Number of Gaussians": params["means3D"].shape[0],
                "Mapping/step": wandb_time_step,
            }
        )
    return params, variables


def add_fine_frame_gaussians(params, variables, curr_data, time_idx, config):
    """Add residual-based Gaussians between coarse and fine mapping."""
    if not config["mapping"]["add_new_gaussians"]:
        return params, variables

    return add_fine_gaussians(
        params,
        variables,
        curr_data,
        config["mapping"]["sil_thres"],
        config["mapping"]["color_thres"],
        time_idx,
        config["mean_sq_dist_method"],
        config["gaussian_distribution"],
        config["surface_init"],
    )


def apply_gradient_densification(
    params,
    variables,
    optimizer,
    iteration,
    config,
    wandb_run,
    wandb_mapping_step,
):
    """Apply optional gradient densification inside a mapping iteration."""
    if not config["mapping"]["use_gaussian_splatting_densification"]:
        return params, variables

    params, variables = densify(
        params,
        variables,
        optimizer,
        iteration,
        config["mapping"]["densify_dict"],
    )
    if config["use_wandb"]:
        wandb_run.log(
            {
                "Mapping/Number of Gaussians - Densification": params[
                    "means3D"
                ].shape[0],
                "Mapping/step": wandb_mapping_step,
            }
        )
    return params, variables
