"""Online and global mapping orchestration for QCG-SLAM."""

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from utils.common_utils import save_params_ckpt
from utils.eval_helpers import report_loss, report_progress
from utils.slam_external import prune_gaussians

from qcg_slam.densification import (
    add_coarse_frame_gaussians,
    add_fine_frame_gaussians,
    apply_gradient_densification,
)
from qcg_slam.keyframes import estimated_w2c_from_params, select_fine_mapping_frame
from qcg_slam.losses import get_loss
from qcg_slam.optimization import initialize_optimizer
from qcg_slam.surface_regularization import project_surface_geometry


def _make_mapping_data(curr_data, frame_idx, color, depth):
    """Build the RGB-D input dictionary for one mapping iteration."""
    return {
        "cam": curr_data["cam"],
        "im": color,
        "depth": depth,
        "id": frame_idx,
        "intrinsics": curr_data["intrinsics"],
        "w2c": curr_data["w2c"],
        "iter_gt_w2c_list": curr_data["iter_gt_w2c_list"][: frame_idx + 1],
    }


def _select_stage_data(
    stage,
    iteration,
    time_idx,
    curr_data,
    keyframe_time_indices,
    keyframe_list,
):
    """Select current-frame data for coarse mapping or sampled data for fine."""
    if stage == "coarse":
        frame_idx = time_idx
        color = curr_data["im"]
        depth = curr_data["depth"]
    else:
        frame_idx, color, depth = select_fine_mapping_frame(
            iteration,
            time_idx,
            keyframe_time_indices,
            keyframe_list,
            curr_data["im"],
            curr_data["depth"],
        )
    return _make_mapping_data(curr_data, frame_idx, color, depth)


def _prune_mapping_gaussians(
    params,
    variables,
    optimizer,
    iteration,
    config,
    wandb_run,
    wandb_mapping_step,
):
    """Apply optional pruning inside a mapping iteration."""
    if not config["mapping"]["prune_gaussians"]:
        return params, variables

    params, variables = prune_gaussians(
        params,
        variables,
        optimizer,
        iteration,
        config["mapping"]["pruning_dict"],
    )
    if config["use_wandb"]:
        wandb_run.log(
            {
                "Mapping/Number of Gaussians - Pruning": params["means3D"].shape[0],
                "Mapping/step": wandb_mapping_step,
            }
        )
    return params, variables


def _report_mapping_iteration(
    params,
    iteration_data,
    iteration,
    progress_bar,
    time_idx,
    config,
    wandb_run,
    wandb_mapping_step,
):
    """Report one coarse or fine mapping iteration."""
    if not config["report_iter_progress"]:
        progress_bar.update(1)
        return

    progress_kwargs = {}
    if config["use_wandb"]:
        progress_kwargs = {
            "wandb_run": wandb_run,
            "wandb_step": wandb_mapping_step,
            "wandb_save_qual": config["wandb"]["save_qual"],
        }
    report_progress(
        params,
        iteration_data,
        iteration + 1,
        progress_bar,
        iteration_data["id"],
        sil_thres=config["mapping"]["sil_thres"],
        mapping=True,
        online_time_idx=time_idx,
        **progress_kwargs,
    )


def _run_mapping_stage(
    stage,
    params,
    variables,
    optimizer,
    curr_data,
    time_idx,
    config,
    runtime_stats,
    keyframe_time_indices,
    keyframe_list,
    wandb_run,
    wandb_mapping_step,
):
    """Run one complete coarse or fine mapping stage."""
    configured_iterations = config["mapping"][f"{stage}_num_iters"]
    first_frame_extra = 50 if stage == "coarse" else 100
    num_iterations = configured_iterations
    if time_idx == 0 and configured_iterations > 0:
        num_iterations += first_frame_extra

    progress_bar = None
    if num_iterations > 0:
        progress_bar = tqdm(
            range(num_iterations),
            desc=f"{stage.title()} Mapping Time Step: {time_idx}",
        )

    for iteration in range(num_iterations):
        iter_start_time = time.time()
        iteration_data = _select_stage_data(
            stage,
            iteration,
            time_idx,
            curr_data,
            keyframe_time_indices,
            keyframe_list,
        )
        loss, variables, losses = get_loss(
            params,
            iteration_data,
            variables,
            iteration_data["id"],
            config["mapping"]["loss_weights"],
            config["mapping"]["use_sil_for_loss"],
            config["mapping"]["sil_thres"],
            config["mapping"]["use_l1"],
            config["mapping"]["ignore_outlier_depth_loss"],
            mapping=True,
            surface_regularization=config["surface_regularization"],
        )
        if config["use_wandb"]:
            wandb_mapping_step = report_loss(
                losses, wandb_run, wandb_mapping_step, mapping=True
            )

        loss.backward()
        with torch.no_grad():
            params, variables = _prune_mapping_gaussians(
                params,
                variables,
                optimizer,
                iteration,
                config,
                wandb_run,
                wandb_mapping_step,
            )
            params, variables = apply_gradient_densification(
                params,
                variables,
                optimizer,
                iteration,
                config,
                wandb_run,
                wandb_mapping_step,
            )
            optimizer.step()
            project_surface_geometry(params, config["surface_regularization"])
            optimizer.zero_grad(set_to_none=True)
            _report_mapping_iteration(
                params,
                iteration_data,
                iteration,
                progress_bar,
                time_idx,
                config,
                wandb_run,
                wandb_mapping_step,
            )
        runtime_stats.add_mapping_iter(time.time() - iter_start_time)

    if progress_bar is not None:
        progress_bar.close()
    return params, variables, wandb_mapping_step


def _report_mapping_result(
    params,
    curr_data,
    time_idx,
    config,
    wandb_run,
    wandb_time_step,
):
    """Report the final mapping result at the configured frame interval."""
    if not (
        time_idx == 0
        or (time_idx + 1) % config["report_global_progress_every"] == 0
    ):
        return

    try:
        progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
        with torch.no_grad():
            progress_kwargs = {}
            if config["use_wandb"]:
                progress_kwargs = {
                    "wandb_run": wandb_run,
                    "wandb_step": wandb_time_step,
                    "wandb_save_qual": config["wandb"]["save_qual"],
                    "global_logging": True,
                }
            report_progress(
                params,
                curr_data,
                1,
                progress_bar,
                time_idx,
                sil_thres=config["mapping"]["sil_thres"],
                mapping=True,
                online_time_idx=time_idx,
                **progress_kwargs,
            )
        progress_bar.close()
    except BaseException:
        ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
        save_params_ckpt(params, ckpt_output_dir, time_idx)
        print("Failed to evaluate trajectory.")


def map_frame(
    params,
    variables,
    curr_data,
    time_idx,
    config,
    device,
    runtime_stats,
    keyframe_time_indices,
    keyframe_list,
    densification_resources,
    wandb_run,
    wandb_time_step,
    wandb_mapping_step,
):
    """Run densification and mapping for one scheduled frame."""
    if time_idx != 0 and (time_idx + 1) % config["map_every"] != 0:
        return params, variables, wandb_mapping_step

    # 1. Fill unobserved regions before coarse mapping.
    params, variables = add_coarse_frame_gaussians(
        params,
        variables,
        curr_data,
        time_idx,
        config,
        densification_resources,
        wandb_run,
        wandb_time_step,
    )

    with torch.no_grad():
        estimated_w2c_from_params(params, time_idx, device)

    coarse_optimizer = initialize_optimizer(
        params, config["mapping"]["coarse_lrs"], tracking=False
    )
    mapping_start_time = time.time()
    # 2. Optimize the current frame at the coarse stage.
    params, variables, wandb_mapping_step = _run_mapping_stage(
        "coarse",
        params,
        variables,
        coarse_optimizer,
        curr_data,
        time_idx,
        config,
        runtime_stats,
        keyframe_time_indices,
        keyframe_list,
        wandb_run,
        wandb_mapping_step,
    )

    # 3. Fill color/depth/silhouette residuals before fine mapping.
    params, variables = add_fine_frame_gaussians(
        params, variables, curr_data, time_idx, config
    )

    fine_optimizer = initialize_optimizer(
        params, config["mapping"]["fine_lrs"], tracking=False
    )
    # 4. Optimize the current frame and sampled keyframes at the fine stage.
    params, variables, wandb_mapping_step = _run_mapping_stage(
        "fine",
        params,
        variables,
        fine_optimizer,
        curr_data,
        time_idx,
        config,
        runtime_stats,
        keyframe_time_indices,
        keyframe_list,
        wandb_run,
        wandb_mapping_step,
    )
    runtime_stats.add_mapping_frame(time.time() - mapping_start_time)

    _report_mapping_result(
        params, curr_data, time_idx, config, wandb_run, wandb_time_step
    )
    return params, variables, wandb_mapping_step


def run_global_mapping(
    params,
    variables,
    keyframe_list,
    gt_w2c_all_frames,
    config,
    camera,
    intrinsics,
    first_frame_w2c,
):
    """Run the optional post-SLAM keyframe mapping pass."""
    if "global_optimization" not in config:
        config["global_optimization"] = False

    total_time = 0.0
    if not config["global_optimization"]:
        return params, variables, total_time

    start_time = time.time()
    optimizer = initialize_optimizer(
        params, config["mapping"]["global_lrs"], tracking=False
    )
    for global_time_idx in tqdm(range(config["global_times"] * len(keyframe_list))):
        selected_idx = np.random.randint(0, len(keyframe_list))
        keyframe = keyframe_list[selected_idx]
        frame_idx = keyframe["id"]
        iteration_data = {
            "cam": camera,
            "im": keyframe["color"],
            "depth": keyframe["depth"],
            "id": frame_idx,
            "intrinsics": intrinsics,
            "w2c": first_frame_w2c,
            "iter_gt_w2c_list": gt_w2c_all_frames[: frame_idx + 1],
        }
        loss, variables, _ = get_loss(
            params,
            iteration_data,
            variables,
            frame_idx,
            config["mapping"]["loss_weights"],
            config["mapping"]["use_sil_for_loss"],
            config["mapping"]["sil_thres"],
            config["mapping"]["use_l1"],
            config["mapping"]["ignore_outlier_depth_loss"],
            mapping=True,
            surface_regularization=config["surface_regularization"],
        )
        loss.backward()
        with torch.no_grad():
            if config["mapping"]["prune_gaussians"]:
                params, variables = prune_gaussians(
                    params,
                    variables,
                    optimizer,
                    global_time_idx,
                    config["mapping"]["pruning_dict_global_optimization"],
                )
            optimizer.step()
            project_surface_geometry(params, config["surface_regularization"])
            optimizer.zero_grad(set_to_none=True)

    total_time = time.time() - start_time
    print("Total Global Optimization Time: %f s" % total_time)
    return params, variables, total_time
