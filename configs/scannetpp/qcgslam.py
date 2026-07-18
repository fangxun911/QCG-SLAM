import os
from os.path import join as p_join
import datetime

primary_device = "cuda"

scenes = [
    "1a3100752b",
    "7c31a42404",
    "8b5caf3398",
    "85251de7d1",
    "b20a261fdf",
    "d3ba8b4232",
    "e01b287af5",
    "f34d532901",
]

seed = 0

# Export SCENE env variable before running
os.environ["SCENE"] = "2"

# Train Split Eval
use_train_split = True

# # Novel View Synthesis Eval
# use_train_split = False

scene_num_frames = [436, -1, -1, -1, 360, 958, -1, -1]

scene_name = scenes[int(os.environ["SCENE"])]
num_frames = scene_num_frames[int(os.environ["SCENE"])]

map_every = 1
keyframe_every = 1
tracking_iters = 300
coarse_mapping_iters = 15
fine_mapping_iters = 45
mapping_iters = coarse_mapping_iters + fine_mapping_iters

group_name = "ScanNet++"
now = datetime.datetime.now().strftime("%m%d%H%M")
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,  # Mapping every nth frame
    keyframe_every=keyframe_every,  # Keyframe every nth frame
    report_global_progress_every=50,  # Report Global Progress every nth frame
    eval_every=1,  # Evaluate every nth frame (at end of SLAM)
    # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    scene_radius_depth_ratio=3,
    # Mean-squared distance method: "projective" or "knn".
    mean_sq_dist_method="projective",
    # Gaussian covariance: "isotropic" or "anisotropic".
    gaussian_distribution="isotropic",
    global_optimization=True,  # 是否全局优化
    global_times=20,  # 全局优化轮数
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,  # Save Checkpoints
    checkpoint_interval=5,  # Checkpoint Interval
    use_wandb=False,
    wandb=dict(
        entity="CITLab",
        project="QCGSLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    swanlab=dict(
        entity="CITLab",
        project="QCGSLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=False,
    ),
    data=dict(
        dataset_name="scannetpp",
        basedir="./data/ScanNet++/data",
        sequence=scene_name,
        ignore_bad=False,
        use_train_split=use_train_split,
        desired_image_height=584,
        desired_image_width=876,
        start=0,
        end=-1,
        stride=1,
        num_frames=num_frames,
        quadtree_contrast_threshold=0.01,  # 四叉树对比度阈值要求
    ),
    tracking=dict(
        use_gt_poses=False,  # Use GT Poses for Tracking
        forward_prop=True,  # Forward Propagate Poses
        visualize_tracking_loss=False,  # Visualize Tracking Diff Images
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_depth_loss_thres=True,
        # Double tracking iterations if this threshold is not met.
        depth_loss_thres=10000,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0007,
            cam_trans=0.0025,
        ),
    ),
    mapping=dict(
        coarse_num_iters=coarse_mapping_iters,
        fine_num_iters=fine_mapping_iters,
        num_iters=coarse_mapping_iters + fine_mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,  # For Addition of new Gaussians
        color_thres=0.25,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,  # mapping 要算所有像素的 loss
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=1.0,
            depth=0.5,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        fine_lrs=dict(
            means3D=0.0002,
            rgb_colors=0.005,
            unnorm_rotations=0.002,
            logit_opacities=0.1,
            log_scales=0.002,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        coarse_lrs=dict(
            means3D=0.0003,
            rgb_colors=0.0075,
            unnorm_rotations=0.003,
            logit_opacities=0.15,
            log_scales=0.003,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        global_lrs=dict(
            means3D=0.00001,
            rgb_colors=0.00025,
            unnorm_rotations=0.0001,
            logit_opacities=0.005,
            log_scales=0.0001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.000,
        ),
        prune_gaussians=True,  # Prune Gaussians during Mapping
        # Tune based on the number of mapping iterations.
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            prune_big=False,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,  # Doesn't consider iter 0
        ),
        # Tune based on the number of mapping iterations.
        pruning_dict_global_optimization=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=4000,
            prune_big=False,
            prune_every=200,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,  # Doesn't consider iter 0
        ),
        # Use Gaussian Splatting-based densification during mapping.
        use_gaussian_splatting_densification=False,
        # Tune based on the number of mapping iterations.
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=3000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,  # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode="color",  # ['color', 'depth' or 'centers']
        # Offset final-recon view camera back by 0.5 units.
        offset_first_viz_cam=True,
        show_sil=False,  # Show Silhouette instead of RGB
        visualize_cams=True,  # Visualize Camera Frustums and Trajectory
        viz_w=600,
        viz_h=340,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=5,  # FPS for Online Recon Viz
        # Enter interactive mode after online recon viz.
        enter_interactive_post_online=True,
    ),
)
