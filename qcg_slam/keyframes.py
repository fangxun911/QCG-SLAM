"""Keyframe helpers."""

import numpy as np
import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation


def estimated_w2c_from_params(params, time_idx, device):
    """Build the estimated world-to-camera matrix for a frame."""
    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][...,
                                                         time_idx].detach())
    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
    curr_w2c = torch.eye(4, device=device).float()
    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
    curr_w2c[:3, 3] = curr_cam_tran
    return curr_w2c


def make_keyframe(params, time_idx, color, depth, device):
    """Create a keyframe record for the current estimated pose."""
    return {
        'id': time_idx,
        'est_w2c': estimated_w2c_from_params(params, time_idx, device),
        'color': color,
        'depth': depth,
    }


def should_add_keyframe(time_idx, keyframe_every, curr_gt_w2c):
    """Return whether the current frame should be appended as a keyframe."""
    return ((time_idx) % keyframe_every == 0 and
            (not torch.isinf(curr_gt_w2c[-1]).any()) and
            (not torch.isnan(curr_gt_w2c[-1]).any()))


def select_fine_mapping_frame(iter_idx, time_idx, keyframe_time_indices,
                              keyframe_list, color, depth):
    """Select the current frame or a prior keyframe for fine mapping."""
    if iter_idx % 10 == 0:  # 定期选择当前帧
        selected_rand_keyframe_idx = len(keyframe_time_indices)
    else:
        candidate_frame_list = np.append(np.array(keyframe_time_indices),
                                         time_idx)
        probability_list = np.exp(-candidate_frame_list**2 / 2 / 5000 / 5000)
        cumulative_f_gaussian = np.cumsum(probability_list) / np.sum(
            probability_list)
        random_number = np.random.rand()
        selected_rand_keyframe_idx = np.searchsorted(cumulative_f_gaussian,
                                                     random_number)

    if selected_rand_keyframe_idx == len(keyframe_time_indices):
        return time_idx, color, depth

    keyframe = keyframe_list[selected_rand_keyframe_idx]
    return keyframe['id'], keyframe['color'], keyframe['depth']
