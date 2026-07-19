"""Main QCG-SLAM RGB-D tracking and mapping pipeline."""

import os

from pprint import pprint
import numpy as np
import torch
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import load_dataset_config
from utils.common_utils import save_params
from utils.eval_helpers import eval as eval_slam

from qcg_slam.checkpoints import load_checkpoint_state, save_checkpoint_state
from qcg_slam.config import prepare_config, prepare_dataset_config
from qcg_slam import context as slam_context
from qcg_slam.datasets import get_dataset
from qcg_slam.densification import (
    DensificationResources,
    load_densification_dataset,
)
from qcg_slam.initialization import initialize_first_timestep
from qcg_slam.keyframes import make_keyframe, should_add_keyframe
from qcg_slam.mapping import map_frame, run_global_mapping
from qcg_slam.runtime import RuntimeStats, report_runtime_stats
from qcg_slam.tracking import (
    initialize_tracking_camera,
    load_tracking_dataset,
    track_frame,
)


class RGBDSLAMRunner:
    """Object-oriented entry point for the QCG-SLAM RGB-D pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = None
        self.eval_dir = None
        self.wandb_run = None
        self.wandb_time_step = 0
        self.wandb_tracking_step = 0
        self.wandb_mapping_step = 0
        self.device = None
        self.dataset_config = None
        self.gradslam_data_cfg = None
        self.dataset = None
        self.num_frames = None
        self.separate_densification_res = False
        self.separate_tracking_res = False
        self.densify_dataset = None
        self.densify_intrinsics = None
        self.densify_cam = None
        self.tracking_dataset = None
        self.tracking_intrinsics = None
        self.tracking_cam = None
        self.params = None
        self.variables = None
        self.intrinsics = None
        self.first_frame_w2c = None
        self.cam = None
        self.keyframe_list = []
        self.keyframe_time_indices = []
        self.gt_w2c_all_frames = []
        self.checkpoint_time_idx = 0
        self.runtime_stats = None
        self.total_global_optimization_time = 0.0

    def run(self):
        """Run the configured RGB-D SLAM pipeline."""
        self.prepare()
        self.load_datasets()
        self.initialize_state()
        self.run_frame_loop()
        self.run_global_optimization()
        self.finalize()

    def prepare(self):
        """Prepare config, output directories, logging, and device state."""
        print("Loading Config:")
        self.config = prepare_config(self.config)
        pprint(self.config, sort_dicts=False, width=100)

        self.output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        self.eval_dir = os.path.join(self.output_dir, "eval")
        os.makedirs(self.eval_dir, exist_ok=True)

        if self.config["use_wandb"]:
            self.wandb_run = wandb.init(
                project=self.config["wandb"]["project"],
                #    entity=self.config['wandb']['entity'],
                group=self.config["wandb"]["group"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )

        self.device = torch.device("cuda")
        slam_context.set_device(self.device)

    def load_datasets(self):
        """Load primary and optional tracking/densification datasets."""
        print("Loading Dataset ...")
        self.dataset_config = self.config["data"]
        if "gradslam_data_cfg" not in self.dataset_config:
            self.gradslam_data_cfg = {}
            self.gradslam_data_cfg["dataset_name"] = self.dataset_config["dataset_name"]
        else:
            self.gradslam_data_cfg = load_dataset_config(
                self.dataset_config["gradslam_data_cfg"]
            )
        (
            self.dataset_config,
            self.separate_densification_res,
            self.separate_tracking_res,
        ) = prepare_dataset_config(self.dataset_config)

        self.dataset = get_dataset(
            config_dict=self.gradslam_data_cfg,
            basedir=self.dataset_config["basedir"],
            sequence=os.path.basename(self.dataset_config["sequence"]),
            start=self.dataset_config["start"],
            end=self.dataset_config["end"],
            stride=self.dataset_config["stride"],
            desired_height=self.dataset_config["desired_image_height"],
            desired_width=self.dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=self.dataset_config["ignore_bad"],
            use_train_split=self.dataset_config["use_train_split"],
            embedding_dim=self.dataset_config["quadtree_contrast_threshold"],
        )
        self.num_frames = self.dataset_config["num_frames"]
        if self.num_frames == -1:
            self.num_frames = len(self.dataset)

        if self.separate_densification_res:
            self.densify_dataset = load_densification_dataset(
                self.gradslam_data_cfg, self.dataset_config, self.device
            )

        if self.separate_tracking_res:
            self.tracking_dataset = load_tracking_dataset(
                self.gradslam_data_cfg, self.dataset_config, self.device
            )

    def initialize_state(self):
        """Initialize Gaussian, camera, checkpoint, and runtime state."""
        if self.separate_densification_res:
            (
                self.params,
                self.variables,
                self.intrinsics,
                self.first_frame_w2c,
                self.cam,
                self.densify_intrinsics,
                self.densify_cam,
            ) = initialize_first_timestep(
                self.dataset,
                self.num_frames,
                self.config["scene_radius_depth_ratio"],
                self.config["mean_sq_dist_method"],
                densify_dataset=self.densify_dataset,
                gaussian_distribution=self.config["gaussian_distribution"],
                surface_init_config=self.config["surface_init"],
            )
        else:
            (
                self.params,
                self.variables,
                self.intrinsics,
                self.first_frame_w2c,
                self.cam,
            ) = initialize_first_timestep(
                self.dataset,
                self.num_frames,
                self.config["scene_radius_depth_ratio"],
                self.config["mean_sq_dist_method"],
                gaussian_distribution=self.config["gaussian_distribution"],
                scene_name=self.config["data"]["sequence"],
                surface_init_config=self.config["surface_init"],
            )
            self.densify_intrinsics = self.intrinsics

        if self.separate_tracking_res:
            self.tracking_intrinsics, self.tracking_cam = initialize_tracking_camera(
                self.tracking_dataset, self.first_frame_w2c
            )

        (
            self.params,
            self.variables,
            self.keyframe_list,
            self.keyframe_time_indices,
            self.gt_w2c_all_frames,
            self.checkpoint_time_idx,
        ) = load_checkpoint_state(
            self.config, self.dataset, self.params, self.variables, self.device
        )
        self.runtime_stats = RuntimeStats()

    def run_frame_loop(self):
        """Run per-frame tracking, mapping, keyframe, and checkpoint work."""
        config = self.config
        dataset = self.dataset
        num_frames = self.num_frames
        checkpoint_time_idx = self.checkpoint_time_idx
        params = self.params
        variables = self.variables
        keyframe_list = self.keyframe_list
        keyframe_time_indices = self.keyframe_time_indices
        gt_w2c_all_frames = self.gt_w2c_all_frames
        runtime_stats = self.runtime_stats
        device = self.device
        cam = self.cam
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        tracking_dataset = self.tracking_dataset
        tracking_cam = self.tracking_cam
        tracking_intrinsics = self.tracking_intrinsics
        densification_resources = DensificationResources(
            dataset=self.densify_dataset,
            camera=self.densify_cam,
            intrinsics=self.densify_intrinsics,
        )
        eval_dir = self.eval_dir
        wandb_run = self.wandb_run
        wandb_time_step = self.wandb_time_step
        wandb_tracking_step = self.wandb_tracking_step
        wandb_mapping_step = self.wandb_mapping_step

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
                "cam": cam,
                "im": color,
                "depth": depth,
                "quadtree": quadtree,
                "id": iter_time_idx,
                "intrinsics": intrinsics,
                "w2c": first_frame_w2c,
                "iter_gt_w2c_list": curr_gt_w2c,
            }

            params, variables, wandb_tracking_step = track_frame(
                params=params,
                variables=variables,
                curr_data=curr_data,
                curr_gt_w2c=curr_gt_w2c,
                time_idx=time_idx,
                config=config,
                eval_dir=eval_dir,
                runtime_stats=runtime_stats,
                wandb_run=wandb_run,
                wandb_time_step=wandb_time_step,
                wandb_tracking_step=wandb_tracking_step,
                first_frame_w2c=first_frame_w2c,
                tracking_dataset=tracking_dataset,
                tracking_camera=tracking_cam,
                tracking_intrinsics=tracking_intrinsics,
            )

            params, variables, wandb_mapping_step = map_frame(
                params=params,
                variables=variables,
                curr_data=curr_data,
                time_idx=time_idx,
                config=config,
                device=device,
                runtime_stats=runtime_stats,
                keyframe_time_indices=keyframe_time_indices,
                keyframe_list=keyframe_list,
                densification_resources=densification_resources,
                wandb_run=wandb_run,
                wandb_time_step=wandb_time_step,
                wandb_mapping_step=wandb_mapping_step,
            )

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
            if should_add_keyframe(time_idx, config["keyframe_every"], curr_gt_w2c):
                with torch.no_grad():
                    keyframe_list.append(
                        make_keyframe(params, time_idx, color, depth, device)
                    )
                    keyframe_time_indices.append(time_idx)

            # Checkpoint every iteration
            if (
                config["save_checkpoints"]
                and time_idx % config["checkpoint_interval"] == 0
            ):
                save_checkpoint_state(config, params, keyframe_time_indices, time_idx)

            # Increment WandB Time Step
            if config["use_wandb"]:
                wandb_time_step += 1

            torch.cuda.empty_cache()

        self.params = params
        self.variables = variables
        self.keyframe_list = keyframe_list
        self.keyframe_time_indices = keyframe_time_indices
        self.gt_w2c_all_frames = gt_w2c_all_frames
        self.runtime_stats = runtime_stats
        self.wandb_time_step = wandb_time_step
        self.wandb_tracking_step = wandb_tracking_step
        self.wandb_mapping_step = wandb_mapping_step

    def run_global_optimization(self):
        """Run the optional post-SLAM keyframe optimization pass."""
        (
            self.params,
            self.variables,
            self.total_global_optimization_time,
        ) = run_global_mapping(
            params=self.params,
            variables=self.variables,
            keyframe_list=self.keyframe_list,
            gt_w2c_all_frames=self.gt_w2c_all_frames,
            config=self.config,
            camera=self.cam,
            intrinsics=self.intrinsics,
            first_frame_w2c=self.first_frame_w2c,
        )

    def finalize(self):
        """Report runtime, evaluate final parameters, and save outputs."""
        report_runtime_stats(
            self.config,
            self.eval_dir,
            self.wandb_run,
            self.runtime_stats,
            self.total_global_optimization_time,
        )

        self.params["intrinsics"] = self.intrinsics.detach().cpu().numpy()
        self.params["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
        self.params["org_width"] = self.dataset_config["desired_image_width"]
        self.params["org_height"] = self.dataset_config["desired_image_height"]
        self.params["gt_w2c_all_frames"] = []
        for gt_w2c_tensor in self.gt_w2c_all_frames:
            self.params["gt_w2c_all_frames"].append(
                gt_w2c_tensor.detach().cpu().numpy()
            )
        self.params["gt_w2c_all_frames"] = np.stack(
            self.params["gt_w2c_all_frames"], axis=0
        )
        self.params["keyframe_time_indices"] = np.array(self.keyframe_time_indices)
        self.params["timestep"] = self.variables["timestep"]

        with torch.no_grad():
            if self.config["use_wandb"]:
                eval_slam(
                    self.dataset,
                    self.params,
                    self.num_frames,
                    self.eval_dir,
                    sil_thres=self.config["mapping"]["sil_thres"],
                    wandb_run=self.wandb_run,
                    wandb_save_qual=self.config["wandb"]["eval_save_qual"],
                    mapping_iters=self.config["mapping"]["num_iters"],
                    add_new_gaussians=self.config["mapping"]["add_new_gaussians"],
                    eval_every=self.config["eval_every"],
                )
            else:
                eval_slam(
                    self.dataset,
                    self.params,
                    self.num_frames,
                    self.eval_dir,
                    sil_thres=self.config["mapping"]["sil_thres"],
                    mapping_iters=self.config["mapping"]["num_iters"],
                    add_new_gaussians=self.config["mapping"]["add_new_gaussians"],
                    eval_every=self.config["eval_every"],
                )

        save_params(self.params, self.output_dir)

        if self.config["use_wandb"]:
            wandb.finish()


def rgbd_slam(config: dict):
    return RGBDSLAMRunner(config).run()


def _run_rgbd_slam_pipeline(config: dict):
    """Backward-compatible wrapper around the runner class."""
    return RGBDSLAMRunner(config).run()
