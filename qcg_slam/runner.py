"""Main QCG-SLAM RGB-D tracking and mapping pipeline."""

import os
import time

import numpy as np
import torch
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import load_dataset_config
from utils.common_utils import save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress
from utils.eval_helpers import eval as eval_slam
from utils.recon_helpers import setup_camera
from utils.slam_external import prune_gaussians, densify
from utils.slam_helpers import matrix_to_quaternion

from qcg_slam.checkpoints import load_checkpoint_state, save_checkpoint_state
from qcg_slam.config import prepare_config, prepare_dataset_config
from qcg_slam import context as slam_context
from qcg_slam.datasets import get_dataset
from qcg_slam.gaussians import add_coarse_gaussians, add_fine_gaussians
from qcg_slam.initialization import initialize_first_timestep
from qcg_slam.keyframes import (
    estimated_w2c_from_params,
    make_keyframe,
    select_fine_mapping_frame,
    should_add_keyframe,
)
from qcg_slam.losses import get_loss
from qcg_slam.optimization import initialize_camera_pose, initialize_optimizer
from qcg_slam.runtime import RuntimeStats, report_runtime_stats


def rgbd_slam(config: dict):
    """Run the QCG-SLAM RGB-D tracking and mapping pipeline."""
    # Print Config
    print("Loading Config:")
    config = prepare_config(config)
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    # 创建 eval 文件夹
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Init WandB
    wandb_run = None
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(
            project=config['wandb']['project'],
            #    entity=config['wandb']['entity'],
            group=config['wandb']['group'],
            name=config['wandb']['name'],
            config=config)

    # Get Device
    device = torch.device('cuda')
    slam_context.set_device(device)
    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    # dataset_config 里有 gradslam_data_cfg: 是一个yaml文件，里面放了相机的参数
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:  # return 一个 dict
        gradslam_data_cfg = load_dataset_config(
            dataset_config["gradslam_data_cfg"])
    dataset_config, separate_densification_res, separate_tracking_res = (
        prepare_dataset_config(dataset_config))

    # Poses are relative to the first frame，第一帧的pose是单位矩阵
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config[
            "basedir"],  # ./data/Replica or ./data/Scannet++/data
        sequence=os.path.basename(dataset_config["sequence"]),  # scene_name
        start=dataset_config["start"],  # 0
        end=dataset_config["end"],  # -1
        stride=dataset_config["stride"],  # 1
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],  # 默认 False
        use_train_split=dataset_config["use_train_split"],  # 默认 True
        embedding_dim=dataset_config["quadtree_contrast_threshold"],
    )
    # 获取数据集数目
    num_frames = dataset_config["num_frames"]  # -1 scannetpp里有个场景是360
    if num_frames == -1:
        num_frames = len(dataset)

    # Init separate dataloader for densification if required
    if separate_densification_res:
        # 也就 desired_height 和 desired_width 不一样，但是数据上是一样的
        densify_dataset = get_dataset(
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
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(
                dataset,
                num_frames,
                config['scene_radius_depth_ratio'],
                config['mean_sq_dist_method'],
                densify_dataset=densify_dataset,
                gaussian_distribution=config['gaussian_distribution'])
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = (
            initialize_first_timestep(
                dataset,
                num_frames,
                config['scene_radius_depth_ratio'],
                config['mean_sq_dist_method'],
                gaussian_distribution=config['gaussian_distribution'],
                scene_name=config['data']['sequence']))

    # Init separate dataloader for tracking if required
    if separate_tracking_res:
        tracking_dataset = get_dataset(
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
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(
            2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2],
                                    tracking_color.shape[1],
                                    tracking_intrinsics.cpu().numpy(),
                                    first_frame_w2c.detach().cpu().numpy())

    # Init Variables to keep track of ground truth poses, keyframes, and
    # runtimes
    params, variables, keyframe_list, keyframe_time_indices, \
        gt_w2c_all_frames, checkpoint_time_idx = load_checkpoint_state(
            config, dataset, params, variables, device)
    runtime_stats = RuntimeStats()

    # Iterate over Scan: time_idx [0, ..., num_frames]
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, quadtree, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data （color得是RGB 归一化格式）
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {
            'cam': cam,
            'im': color,
            'depth': depth,
            'quadtree': quadtree,
            'id': iter_time_idx,
            'intrinsics': intrinsics,
            'w2c': first_frame_w2c,
            'iter_gt_w2c_list': curr_gt_w2c
        }

        # Initialize Data for Tracking
        if separate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {
                'cam': tracking_cam,
                'im': tracking_color,
                'depth': tracking_depth,
                'id': iter_time_idx,
                'intrinsics': tracking_intrinsics,
                'w2c': first_frame_w2c,
                'iter_gt_w2c_list': curr_gt_w2c
            }
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        coarse_num_iters_mapping = config['mapping']['coarse_num_iters']
        fine_num_iters_mapping = config['mapping']['fine_num_iters']

        # Initialize the camera pose for the current frame
        # 根据匀速假设，更新相机位姿信息
        if time_idx > 0:
            params = initialize_camera_pose(
                params,
                time_idx,
                forward_prop=config['tracking']['forward_prop'])

        # Tracking
        tracking_start_time = time.time()
        # 第0帧不进行位姿优化，且全程不用真实位姿
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params,
                                             config['tracking']['lrs'],
                                             tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][
                ..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][...,
                                                     time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking),
                                desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(
                    params,
                    tracking_curr_data,
                    variables,
                    iter_time_idx,
                    config['tracking']['loss_weights'],
                    config['tracking']['use_sil_for_loss'],
                    config['tracking']['sil_thres'],
                    config['tracking']['use_l1'],
                    config['tracking']['ignore_outlier_depth_loss'],
                    tracking=True,
                    plot_dir=eval_dir,
                    visualize_tracking_loss=config['tracking']
                    ['visualize_tracking_loss'],
                    tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses,
                                                      wandb_run,
                                                      wandb_tracking_step,
                                                      tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][
                            ..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][
                            ..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:  # False
                        if config['use_wandb']:
                            report_progress(
                                params,
                                tracking_curr_data,
                                iter + 1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config['tracking']['sil_thres'],
                                tracking=True,
                                wandb_run=wandb_run,
                                wandb_step=wandb_tracking_step,
                                wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(
                                params,
                                tracking_curr_data,
                                iter + 1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config['tracking']['sil_thres'],
                                tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                runtime_stats.add_tracking_iter(iter_end_time - iter_start_time)
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    # print(losses['depth'])
                    if losses['depth'] < config['tracking'][
                            'depth_loss_thres'] and config['tracking'][
                                'use_depth_loss_thres']:
                        break
                    # 如果没达到 depth_loss_thres 的话，迭代次数翻倍，继续循环
                    elif config['tracking'][
                            'use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True  # 最多只翻倍一次，防止陷入死循环
                        progress_bar = tqdm(
                            range(config['tracking']['num_iters']),
                            desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = num_iters_tracking + config[
                            'tracking']['num_iters']
                        if config['use_wandb']:
                            wandb_run.log({
                                "Tracking/Extra Tracking Iters Frames":
                                    time_idx,
                                "Tracking/step":
                                    wandb_time_step
                            })
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation 更新相机位姿参数
            with torch.no_grad():
                params['cam_unnorm_rots'][...,
                                          time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        runtime_stats.add_tracking_frame(tracking_end_time -
                                         tracking_start_time)

        # 每 report_global_progress_every 帧报告一次
        if time_idx == 0 or (time_idx +
                             1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(
                    range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(
                            params,
                            tracking_curr_data,
                            1,
                            progress_bar,
                            iter_time_idx,
                            sil_thres=config['tracking']['sil_thres'],
                            tracking=True,
                            wandb_run=wandb_run,
                            wandb_step=wandb_time_step,
                            wandb_save_qual=config['wandb']['save_qual'],
                            global_logging=True)
                    else:
                        report_progress(
                            params,
                            tracking_curr_data,
                            1,
                            progress_bar,
                            iter_time_idx,
                            sil_thres=config['tracking']['sil_thres'],
                            tracking=True)
                progress_bar.close()
            except BaseException:
                ckpt_output_dir = os.path.join(config["workdir"],
                                               config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping（slam肯定是每帧都建图）
        if time_idx == 0 or (time_idx + 1) % config['map_every'] == 0:
            # Densification （第0帧不densify，因为已经初始化高斯基元了）
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if separate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[
                        time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {
                        'cam': densify_cam,
                        'im': densify_color,
                        'depth': densify_depth,
                        'id': time_idx,
                        'intrinsics': densify_intrinsics,
                        'w2c': first_frame_w2c,
                        'iter_gt_w2c_list': curr_gt_w2c
                    }
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the
                # Silhouette在这里加一个bool位，判断是否将本帧视为关键帧
                params, variables = add_coarse_gaussians(
                    params, variables, densify_curr_data,
                    config['mapping']['sil_thres'], time_idx,
                    config['mean_sq_dist_method'],
                    config['gaussian_distribution'], config['data']['sequence'])
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({
                        "Mapping/Number of Gaussians": post_num_pts,
                        "Mapping/step": wandb_time_step
                    })

            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_w2c = estimated_w2c_from_params(params, time_idx, device)

            # Reset Optimizer & Learning Rates for Full Map Optimization
            # Coarse lrs
            optimizer = initialize_optimizer(params,
                                             config['mapping']['coarse_lrs'],
                                             tracking=False)

            # Mapping
            mapping_start_time = time.time()
            if coarse_num_iters_mapping > 0 and time_idx == 0:
                coarse_num_iters_mapping_1 = coarse_num_iters_mapping + 50
                progress_bar = tqdm(
                    range(coarse_num_iters_mapping_1),
                    desc=f"Coarse Mapping Time Step: {time_idx}")
            elif coarse_num_iters_mapping > 0:
                coarse_num_iters_mapping_1 = coarse_num_iters_mapping
                progress_bar = tqdm(
                    range(coarse_num_iters_mapping_1),
                    desc=f"Coarse Mapping Time Step: {time_idx}")
            # Coarse Mapping
            for iter in range(coarse_num_iters_mapping_1):
                iter_start_time = time.time()

                # Use Current Frame Data
                iter_time_idx = time_idx
                iter_color = color
                iter_depth = depth

                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx + 1]
                iter_data = {
                    'cam': cam,
                    'im': iter_color,
                    'depth': iter_depth,
                    'id': iter_time_idx,
                    'intrinsics': intrinsics,
                    'w2c': first_frame_w2c,
                    'iter_gt_w2c_list': iter_gt_w2c
                }
                # Loss for current frame
                loss, variables, losses = get_loss(
                    params,
                    iter_data,
                    variables,
                    iter_time_idx,
                    config['mapping']['loss_weights'],
                    config['mapping']['use_sil_for_loss'],
                    config['mapping']['sil_thres'],
                    config['mapping']['use_l1'],
                    config['mapping']['ignore_outlier_depth_loss'],
                    mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses,
                                                     wandb_run,
                                                     wandb_mapping_step,
                                                     mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(
                            params, variables, optimizer, iter,
                            config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({
                                "Mapping/Number of Gaussians - Pruning":
                                    params['means3D'].shape[0],
                                "Mapping/step":
                                    wandb_mapping_step
                            })
                    # Gaussian-Splatting's Gradient-based Densification
                    # 不用use_gaussian_splatting_densification
                    if config['mapping'][
                            'use_gaussian_splatting_densification']:
                        params, variables = densify(
                            params, variables, optimizer, iter,
                            config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({
                                "Mapping/Number of Gaussians - Densification":
                                    params['means3D'].shape[0],
                                "Mapping/step":
                                    wandb_mapping_step
                            })
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(
                                params,
                                iter_data,
                                iter + 1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config['mapping']['sil_thres'],
                                wandb_run=wandb_run,
                                wandb_step=wandb_mapping_step,
                                wandb_save_qual=config['wandb']['save_qual'],
                                mapping=True,
                                online_time_idx=time_idx)
                        else:
                            report_progress(
                                params,
                                iter_data,
                                iter + 1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config['mapping']['sil_thres'],
                                mapping=True,
                                online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                runtime_stats.add_mapping_iter(iter_end_time - iter_start_time)
            if coarse_num_iters_mapping_1 > 0:
                progress_bar.close()
            # End of Coarse Mapping

            torch.cuda.empty_cache()

            # finer densification
            params, variables = add_fine_gaussians(
                params, variables, curr_data, config['mapping']['sil_thres'],
                config['mapping']['color_thres'], time_idx,
                config['mean_sq_dist_method'], config['gaussian_distribution'])

            # Finer lrs
            optimizer = initialize_optimizer(params,
                                             config['mapping']['fine_lrs'],
                                             tracking=False)

            # if fine_num_iters_mapping > 0:
            # progress_bar = tqdm(range(fine_num_iters_mapping), desc=f"Fine
            # Mapping Time Step: {time_idx}")
            if fine_num_iters_mapping > 0 and time_idx == 0:
                fine_num_iters_mapping_1 = fine_num_iters_mapping + 100
                progress_bar = tqdm(range(fine_num_iters_mapping_1),
                                    desc=f"Fine Mapping Time Step: {time_idx}")
            elif fine_num_iters_mapping > 0:
                fine_num_iters_mapping_1 = fine_num_iters_mapping
                progress_bar = tqdm(range(fine_num_iters_mapping_1),
                                    desc=f"Fine Mapping Time Step: {time_idx}")
            # Fine Mapping
            for iter in range(fine_num_iters_mapping_1):
                iter_start_time = time.time()

                iter_time_idx, iter_color, iter_depth = (
                    select_fine_mapping_frame(iter, time_idx,
                                              keyframe_time_indices,
                                              keyframe_list, color, depth))

                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx + 1]
                iter_data = {
                    'cam': cam,
                    'im': iter_color,
                    'depth': iter_depth,
                    'id': iter_time_idx,
                    'intrinsics': intrinsics,
                    'w2c': first_frame_w2c,
                    'iter_gt_w2c_list': iter_gt_w2c
                }
                # Loss for current frame
                loss, variables, losses = get_loss(
                    params,
                    iter_data,
                    variables,
                    iter_time_idx,
                    config['mapping']['loss_weights'],
                    config['mapping']['use_sil_for_loss'],
                    config['mapping']['sil_thres'],
                    config['mapping']['use_l1'],
                    config['mapping']['ignore_outlier_depth_loss'],
                    mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses,
                                                     wandb_run,
                                                     wandb_mapping_step,
                                                     mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(
                            params, variables, optimizer, iter,
                            config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({
                                "Mapping/Number of Gaussians - Pruning":
                                    params['means3D'].shape[0],
                                "Mapping/step":
                                    wandb_mapping_step
                            })
                    # Gaussian-Splatting's Gradient-based Densification
                    # 不用use_gaussian_splatting_densification
                    if config['mapping'][
                            'use_gaussian_splatting_densification']:
                        params, variables = densify(
                            params, variables, optimizer, iter,
                            config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({
                                "Mapping/Number of Gaussians - Densification":
                                    params['means3D'].shape[0],
                                "Mapping/step":
                                    wandb_mapping_step
                            })
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(
                                params,
                                iter_data,
                                iter + 1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config['mapping']['sil_thres'],
                                wandb_run=wandb_run,
                                wandb_step=wandb_mapping_step,
                                wandb_save_qual=config['wandb']['save_qual'],
                                mapping=True,
                                online_time_idx=time_idx)
                        else:
                            report_progress(
                                params,
                                iter_data,
                                iter + 1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config['mapping']['sil_thres'],
                                mapping=True,
                                online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                runtime_stats.add_mapping_iter(iter_end_time - iter_start_time)
            if fine_num_iters_mapping_1 > 0:
                progress_bar.close()
            # End of Fine Mapping

            # Update the runtime numbers
            mapping_end_time = time.time()
            runtime_stats.add_mapping_frame(mapping_end_time -
                                            mapping_start_time)

            if time_idx == 0 or (
                    time_idx + 1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(
                        range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(
                                params,
                                curr_data,
                                1,
                                progress_bar,
                                time_idx,
                                sil_thres=config['mapping']['sil_thres'],
                                wandb_run=wandb_run,
                                wandb_step=wandb_time_step,
                                wandb_save_qual=config['wandb']['save_qual'],
                                mapping=True,
                                online_time_idx=time_idx,
                                global_logging=True)
                        else:
                            report_progress(
                                params,
                                curr_data,
                                1,
                                progress_bar,
                                time_idx,
                                sil_thres=config['mapping']['sil_thres'],
                                mapping=True,
                                online_time_idx=time_idx)
                    progress_bar.close()
                except BaseException:
                    ckpt_output_dir = os.path.join(config["workdir"],
                                                   config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

        # Add frame to keyframe list
        # 增加关键帧（第一帧、距离上一个关键帧已经隔了keyframe_every帧、倒数第2帧）
        # if ((time_idx == 0) or (not is_KeyFrame and
        # (time_idx-keyframe_time_indices[-1]) % config['keyframe_every'] == 0)
        # or \
        # (time_idx == num_frames-2) or is_KeyFrame) and (not
        # torch.isinf(curr_gt_w2c[-1]).any()) and (not
        # torch.isnan(curr_gt_w2c[-1]).any()):
        # if ((time_idx == 0) or (time_idx == num_frames-2) or is_KeyFrame) and
        # \
        # (not torch.isinf(curr_gt_w2c[-1]).any()) and (not
        # torch.isnan(curr_gt_w2c[-1]).any()):
        # if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0)
        # or \
        # (time_idx == num_frames-2)) and (not
        # torch.isinf(curr_gt_w2c[-1]).any()) and (not
        # torch.isnan(curr_gt_w2c[-1]).any()):
        if should_add_keyframe(time_idx, config['keyframe_every'], curr_gt_w2c):
            with torch.no_grad():
                keyframe_list.append(
                    make_keyframe(params, time_idx, color, depth, device))
                keyframe_time_indices.append(time_idx)

        # Checkpoint every iteration
        if config['save_checkpoints'] and time_idx % config[
                "checkpoint_interval"] == 0:
            save_checkpoint_state(config, params, keyframe_time_indices,
                                  time_idx)

        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    if 'global_optimization' not in config:
        config['global_optimization'] = False
    total_global_optimization_time = 0.0
    if config['global_optimization']:
        global_optimization_start_time = time.time()
        # Global Optimization
        optimizer = initialize_optimizer(params,
                                         config['mapping']['global_lrs'],
                                         tracking=False)
        for global_time_idx in tqdm(
                range(config['global_times'] * len(keyframe_list))):
            selected_rand_keyframe_idx = np.random.randint(
                0, len(keyframe_list))
            iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
            iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
            iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx + 1]
            iter_data = {
                'cam': cam,
                'im': iter_color,
                'depth': iter_depth,
                'id': iter_time_idx,
                'intrinsics': intrinsics,
                'w2c': first_frame_w2c,
                'iter_gt_w2c_list': iter_gt_w2c
            }
            # Loss for current frame
            loss, variables, losses = get_loss(
                params,
                iter_data,
                variables,
                iter_time_idx,
                config['mapping']['loss_weights'],
                config['mapping']['use_sil_for_loss'],
                config['mapping']['sil_thres'],
                config['mapping']['use_l1'],
                config['mapping']['ignore_outlier_depth_loss'],
                mapping=True)
            loss.backward()
            with torch.no_grad():
                # Prune Gaussians
                if config['mapping']['prune_gaussians']:
                    params, variables = prune_gaussians(
                        params, variables, optimizer, global_time_idx,
                        config['mapping']['pruning_dict_global_optimization'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_global_optimization_time = time.time(
        ) - global_optimization_start_time
        print("Total Global Optimization Time: %f s" %
              (total_global_optimization_time))

    # Compute Average Runtimes
    report_runtime_stats(config, eval_dir, wandb_run, runtime_stats,
                         total_global_optimization_time)

    # Add Camera Parameters to Save them
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    params['timestep'] = variables['timestep']

    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval_slam(dataset,
                      params,
                      num_frames,
                      eval_dir,
                      sil_thres=config['mapping']['sil_thres'],
                      wandb_run=wandb_run,
                      wandb_save_qual=config['wandb']['eval_save_qual'],
                      mapping_iters=config['mapping']['num_iters'],
                      add_new_gaussians=config['mapping']['add_new_gaussians'],
                      eval_every=config['eval_every'])
        else:
            eval_slam(dataset,
                      params,
                      num_frames,
                      eval_dir,
                      sil_thres=config['mapping']['sil_thres'],
                      mapping_iters=config['mapping']['num_iters'],
                      add_new_gaussians=config['mapping']['add_new_gaussians'],
                      eval_every=config['eval_every'])

    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()
