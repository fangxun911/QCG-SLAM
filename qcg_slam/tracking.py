"""Camera tracking helpers for the QCG-SLAM frame loop."""

import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.common_utils import save_params_ckpt
from utils.eval_helpers import report_loss, report_progress
from utils.recon_helpers import setup_camera
from utils.slam_helpers import matrix_to_quaternion

from qcg_slam.datasets import get_dataset
from qcg_slam.losses import get_loss
from qcg_slam.optimization import initialize_optimizer


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    """Initialize the current camera pose estimate for tracking."""
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            prev_rot1 = F.normalize(
                params["cam_unnorm_rots"][..., curr_time_idx - 1].detach()
            )
            prev_rot2 = F.normalize(
                params["cam_unnorm_rots"][..., curr_time_idx - 2].detach()
            )
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params["cam_unnorm_rots"][..., curr_time_idx] = new_rot.detach()

            prev_tran1 = params["cam_trans"][..., curr_time_idx - 1].detach()
            prev_tran2 = params["cam_trans"][..., curr_time_idx - 2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params["cam_trans"][..., curr_time_idx] = new_tran.detach()
        else:
            params["cam_unnorm_rots"][..., curr_time_idx] = params[
                "cam_unnorm_rots"
            ][..., curr_time_idx - 1].detach()
            params["cam_trans"][..., curr_time_idx] = params["cam_trans"][
                ..., curr_time_idx - 1
            ].detach()

    return params


def load_tracking_dataset(gradslam_data_cfg, dataset_config, device):
    """Load the optional dataset used at a dedicated tracking resolution."""
    return get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["tracking_image_height"],
        desired_width=dataset_config["tracking_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )


def initialize_tracking_camera(tracking_dataset, first_frame_w2c):
    """Build camera state for a dedicated tracking-resolution dataset."""
    tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
    tracking_color = tracking_color.permute(2, 0, 1) / 255
    tracking_intrinsics = tracking_intrinsics[:3, :3]
    tracking_camera = setup_camera(
        tracking_color.shape[2],
        tracking_color.shape[1],
        tracking_intrinsics.cpu().numpy(),
        first_frame_w2c.detach().cpu().numpy(),
    )
    return tracking_intrinsics, tracking_camera


def _prepare_tracking_data(
    curr_data,
    curr_gt_w2c,
    time_idx,
    tracking_dataset,
    tracking_camera,
    tracking_intrinsics,
    first_frame_w2c,
):
    """Prepare one RGB-D frame at the configured tracking resolution."""
    if tracking_dataset is None:
        return curr_data

    tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
    tracking_color = tracking_color.permute(2, 0, 1) / 255
    tracking_depth = tracking_depth.permute(2, 0, 1)
    return {
        "cam": tracking_camera,
        "im": tracking_color,
        "depth": tracking_depth,
        "id": time_idx,
        "intrinsics": tracking_intrinsics,
        "w2c": first_frame_w2c,
        "iter_gt_w2c_list": curr_gt_w2c,
    }


def _optimize_camera_pose(
    params,
    variables,
    tracking_data,
    time_idx,
    config,
    eval_dir,
    runtime_stats,
    wandb_run,
    wandb_time_step,
    wandb_tracking_step,
):
    """Optimize or assign the camera pose for a single frame."""
    tracking_start_time = time.time()

    if time_idx > 0 and not config["tracking"]["use_gt_poses"]:
        optimizer = initialize_optimizer(
            params, config["tracking"]["lrs"], tracking=True
        )
        candidate_cam_unnorm_rot = (
            params["cam_unnorm_rots"][..., time_idx].detach().clone()
        )
        candidate_cam_tran = params["cam_trans"][..., time_idx].detach().clone()
        current_min_loss = float(1e20)
        iteration = 0
        do_continue_slam = False
        num_iters_tracking = config["tracking"]["num_iters"]
        progress_bar = tqdm(
            range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}"
        )

        while True:
            iter_start_time = time.time()
            loss, variables, losses = get_loss(
                params,
                tracking_data,
                variables,
                time_idx,
                config["tracking"]["loss_weights"],
                config["tracking"]["use_sil_for_loss"],
                config["tracking"]["sil_thres"],
                config["tracking"]["use_l1"],
                config["tracking"]["ignore_outlier_depth_loss"],
                tracking=True,
                plot_dir=eval_dir,
                visualize_tracking_loss=config["tracking"][
                    "visualize_tracking_loss"
                ],
                tracking_iteration=iteration,
            )
            if config["use_wandb"]:
                wandb_tracking_step = report_loss(
                    losses, wandb_run, wandb_tracking_step, tracking=True
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_unnorm_rot = (
                        params["cam_unnorm_rots"][..., time_idx].detach().clone()
                    )
                    candidate_cam_tran = (
                        params["cam_trans"][..., time_idx].detach().clone()
                    )

                if config["report_iter_progress"]:
                    progress_kwargs = {}
                    if config["use_wandb"]:
                        progress_kwargs = {
                            "wandb_run": wandb_run,
                            "wandb_step": wandb_tracking_step,
                            "wandb_save_qual": config["wandb"]["save_qual"],
                        }
                    report_progress(
                        params,
                        tracking_data,
                        iteration + 1,
                        progress_bar,
                        time_idx,
                        sil_thres=config["tracking"]["sil_thres"],
                        tracking=True,
                        **progress_kwargs,
                    )
                else:
                    progress_bar.update(1)

            runtime_stats.add_tracking_iter(time.time() - iter_start_time)
            iteration += 1
            if iteration == num_iters_tracking:
                if (
                    losses["depth"] < config["tracking"]["depth_loss_thres"]
                    and config["tracking"]["use_depth_loss_thres"]
                ):
                    break
                if (
                    config["tracking"]["use_depth_loss_thres"]
                    and not do_continue_slam
                ):
                    do_continue_slam = True
                    progress_bar = tqdm(
                        range(config["tracking"]["num_iters"]),
                        desc=f"Tracking Time Step: {time_idx}",
                    )
                    num_iters_tracking += config["tracking"]["num_iters"]
                    if config["use_wandb"]:
                        wandb_run.log(
                            {
                                "Tracking/Extra Tracking Iters Frames": time_idx,
                                "Tracking/step": wandb_time_step,
                            }
                        )
                else:
                    break

        progress_bar.close()
        with torch.no_grad():
            params["cam_unnorm_rots"][..., time_idx] = candidate_cam_unnorm_rot
            params["cam_trans"][..., time_idx] = candidate_cam_tran
    elif time_idx > 0 and config["tracking"]["use_gt_poses"]:
        with torch.no_grad():
            rel_w2c = tracking_data["iter_gt_w2c_list"][-1]
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
            params["cam_trans"][..., time_idx] = rel_w2c_tran

    runtime_stats.add_tracking_frame(time.time() - tracking_start_time)
    return params, variables, wandb_tracking_step


def _report_tracking_result(
    params,
    tracking_data,
    time_idx,
    config,
    wandb_run,
    wandb_time_step,
):
    """Report the final tracking result at the configured frame interval."""
    if not (
        time_idx == 0
        or (time_idx + 1) % config["report_global_progress_every"] == 0
    ):
        return

    try:
        progress_bar = tqdm(
            range(1), desc=f"Tracking Result Time Step: {time_idx}"
        )
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
                tracking_data,
                1,
                progress_bar,
                time_idx,
                sil_thres=config["tracking"]["sil_thres"],
                tracking=True,
                **progress_kwargs,
            )
        progress_bar.close()
    except BaseException:
        ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
        save_params_ckpt(params, ckpt_output_dir, time_idx)
        print("Failed to evaluate trajectory.")


def track_frame(
    params,
    variables,
    curr_data,
    curr_gt_w2c,
    time_idx,
    config,
    eval_dir,
    runtime_stats,
    wandb_run,
    wandb_time_step,
    wandb_tracking_step,
    first_frame_w2c,
    tracking_dataset=None,
    tracking_camera=None,
    tracking_intrinsics=None,
):
    """Run all tracking work for one frame and return the updated state."""
    tracking_data = _prepare_tracking_data(
        curr_data,
        curr_gt_w2c,
        time_idx,
        tracking_dataset,
        tracking_camera,
        tracking_intrinsics,
        first_frame_w2c,
    )

    if time_idx > 0:
        params = initialize_camera_pose(
            params, time_idx, forward_prop=config["tracking"]["forward_prop"]
        )

    params, variables, wandb_tracking_step = _optimize_camera_pose(
        params,
        variables,
        tracking_data,
        time_idx,
        config,
        eval_dir,
        runtime_stats,
        wandb_run,
        wandb_time_step,
        wandb_tracking_step,
    )
    _report_tracking_result(
        params,
        tracking_data,
        time_idx,
        config,
        wandb_run,
        wandb_time_step,
    )
    return params, variables, wandb_tracking_step
