"""Render and diagnose the first-frame Gaussian initialization.

This intentionally stops before coarse/fine mapping so that the saved images
show only the Gaussians produced by ``initialize_first_timestep``.
"""

import argparse
import copy
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from qcg_slam.runner import RGBDSLAMRunner
from utils.common_utils import seed_everything
from utils.slam_external import build_rotation
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params2depthplussilhouette,
    transformed_params2rendervar,
)


def _load_config(path):
    module_name = f"initialization_debug_{os.getpid()}"
    experiment = SourceFileLoader(module_name, path).load_module()
    config = copy.deepcopy(experiment.config)
    config["load_checkpoint"] = False
    config["use_wandb"] = False
    return config


def _rgb_uint8(image):
    image = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.round(image * 255.0).astype(np.uint8)


def _save_rgb(path, image):
    Image.fromarray(_rgb_uint8(image), mode="RGB").save(path)


def _psnr(render, target, mask=None):
    squared_error = (render - target).square()
    if mask is not None:
        squared_error = squared_error[:, mask]
    mse = squared_error.mean().clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _depth_limits(depth, valid):
    values = depth[valid]
    if values.size == 0:
        return 0.0, 1.0
    return tuple(np.quantile(values, [0.02, 0.98]).tolist())


def _surface_metrics(params, intrinsics, w2c, gt_depth):
    means = params["means3D"].detach()
    ones = torch.ones_like(means[:, :1])
    camera_means = (w2c @ torch.cat((means, ones), dim=1).T).T[:, :3]
    z = camera_means[:, 2]
    projected_x = intrinsics[0, 0] * camera_means[:, 0] / z + intrinsics[0, 2]
    projected_y = intrinsics[1, 1] * camera_means[:, 1] / z + intrinsics[1, 2]
    x = projected_x.round().long()
    y = projected_y.round().long()
    height, width = gt_depth.shape
    in_bounds = ((z > 0) & (x >= 0) & (x < width) &
                 (y >= 0) & (y < height))
    safe_x = x.clamp(0, width - 1)
    safe_y = y.clamp(0, height - 1)
    sampled_depth = gt_depth[safe_y, safe_x]
    valid = in_bounds & (sampled_depth > 0)
    residual = torch.abs(z[valid] - sampled_depth[valid])

    rotations_world = build_rotation(
        torch.nn.functional.normalize(params["unnorm_rotations"].detach(),
                                      dim=1))
    normals_world = rotations_world[:, :, 2]
    normals_camera = torch.einsum("ij,bj->bi", w2c[:3, :3], normals_world)
    view_direction = torch.nn.functional.normalize(camera_means, dim=1)
    view_cosine = torch.abs((normals_camera * view_direction).sum(dim=1))

    scales = torch.exp(params["log_scales"].detach())
    tangent_min = torch.minimum(scales[:, 0], scales[:, 1])
    thickness_ratio = scales[:, 2] / tangent_min

    def percentile(values, q):
        if values.numel() == 0:
            return None
        return float(torch.quantile(values.float(), q).item())

    return {
        "projected_centers_with_valid_depth": int(valid.sum().item()),
        "center_depth_residual_median_m": percentile(residual, 0.5),
        "center_depth_residual_p90_m": percentile(residual, 0.9),
        "center_depth_residual_p99_m": percentile(residual, 0.99),
        "abs_normal_view_cosine_median": percentile(view_cosine, 0.5),
        "normal_to_min_tangent_scale_ratio_median": percentile(
            thickness_ratio, 0.5),
        "normal_to_min_tangent_scale_ratio_max": (
            float(thickness_ratio.max().item()) if thickness_ratio.numel()
            else None),
    }


def _save_overview(path, gt, render, rgb_error, silhouette, gt_depth,
                   render_depth, depth_error, metrics):
    valid_depth = gt_depth > 0
    depth_vmin, depth_vmax = _depth_limits(gt_depth, valid_depth)
    depth_error_values = depth_error[valid_depth & (silhouette > 0.5)]
    error_vmax = (float(np.quantile(depth_error_values, 0.95))
                  if depth_error_values.size else 0.1)
    error_vmax = max(error_vmax, 1e-4)

    figure, axes = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True)
    axes[0, 0].imshow(gt)
    axes[0, 0].set_title("Ground-truth RGB")
    axes[0, 1].imshow(render)
    axes[0, 1].set_title(f"Initialization render\nPSNR {metrics['psnr_full_db']:.2f} dB")
    rgb_plot = axes[0, 2].imshow(rgb_error, cmap="magma", vmin=0.0, vmax=0.5)
    axes[0, 2].set_title("Mean absolute RGB error")
    figure.colorbar(rgb_plot, ax=axes[0, 2], fraction=0.046)

    sil_plot = axes[1, 0].imshow(silhouette, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(
        f"Silhouette\ncoverage > 0.5: {metrics['coverage_silhouette_gt_0_5']:.1%}")
    figure.colorbar(sil_plot, ax=axes[1, 0], fraction=0.046)
    depth_plot = axes[1, 1].imshow(
        np.ma.masked_where(silhouette <= 1e-4, render_depth),
        cmap="turbo", vmin=depth_vmin, vmax=depth_vmax)
    axes[1, 1].set_title("Normalized rendered depth")
    figure.colorbar(depth_plot, ax=axes[1, 1], fraction=0.046, label="m")
    error_plot = axes[1, 2].imshow(
        np.ma.masked_where(~valid_depth, depth_error),
        cmap="magma", vmin=0.0, vmax=error_vmax)
    axes[1, 2].set_title(
        f"Absolute depth error\nmedian {metrics['render_depth_l1_median_m']:.4f} m")
    figure.colorbar(error_plot, ax=axes[1, 2], fraction=0.046, label="m")
    for axis in axes.flat:
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Render first-frame Gaussians before any optimization")
    parser.add_argument("experiment", help="Path to the training config")
    parser.add_argument(
        "--output",
        help="Output directory (default: <workdir>/<run_name>/initialization_debug)")
    parser.add_argument(
        "--tangent-scale-multiplier",
        type=float,
        default=1.0,
        help=("Diagnostic-only multiplier for local x/y scales; the default "
              "1.0 renders the unmodified initialization"))
    args = parser.parse_args()

    if args.tangent_scale_multiplier <= 0:
        parser.error("--tangent-scale-multiplier must be positive")

    config = _load_config(args.experiment)
    seed_everything(seed=config["seed"])
    if args.output:
        output_dir = args.output
    else:
        suffix = "initialization_debug"
        if args.tangent_scale_multiplier != 1.0:
            factor = f"{args.tangent_scale_multiplier:.4f}".replace(".", "p")
            suffix = f"initialization_debug_tangent_x_{factor}"
        output_dir = os.path.join(config["workdir"], config["run_name"],
                                  suffix)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    runner = RGBDSLAMRunner(config)
    runner.prepare()
    runner.load_datasets()
    runner.initialize_state()

    color, depth, _, _, _ = runner.dataset[0]
    target = color.permute(2, 0, 1) / 255.0
    gt_depth = depth.permute(2, 0, 1)[0]
    params = runner.params
    if args.tangent_scale_multiplier != 1.0:
        with torch.no_grad():
            params["log_scales"][:, :2] += np.log(
                args.tangent_scale_multiplier)

    with torch.no_grad():
        transformed = transform_to_frame(
            params, 0, gaussians_grad=False, camera_grad=False)
        render, radii, _ = Renderer(raster_settings=runner.cam)(
            **transformed_params2rendervar(params, transformed))
        depth_silhouette, _, _ = Renderer(raster_settings=runner.cam)(
            **transformed_params2depthplussilhouette(
                params, runner.first_frame_w2c, transformed))

    silhouette = depth_silhouette[1]
    normalized_depth = depth_silhouette[0] / silhouette.clamp_min(1e-8)
    valid_gt = gt_depth > 0
    visible_valid = valid_gt & (silhouette > 0.5)
    depth_error = torch.abs(normalized_depth - gt_depth)
    rgb_error = torch.mean(torch.abs(render - target), dim=0)

    metrics = {
        "scene": config["data"]["sequence"],
        "gaussian_distribution": config["gaussian_distribution"],
        "diagnostic_tangent_scale_multiplier": args.tangent_scale_multiplier,
        "num_gaussians": int(params["means3D"].shape[0]),
        "num_renderer_visible_gaussians": int((radii > 0).sum().item()),
        "initial_opacity": float(torch.sigmoid(
            params["logit_opacities"][0, 0]).item()),
        "psnr_full_db": _psnr(render, target),
        "psnr_valid_depth_db": _psnr(render, target, valid_gt),
        "coverage_silhouette_gt_0_5": float(
            (silhouette[valid_gt] > 0.5).float().mean().item()),
        "coverage_silhouette_gt_0_9": float(
            (silhouette[valid_gt] > 0.9).float().mean().item()),
        "render_depth_l1_mean_m": float(
            depth_error[visible_valid].mean().item()),
        "render_depth_l1_median_m": float(
            depth_error[visible_valid].median().item()),
        "render_depth_l1_p90_m": float(
            torch.quantile(depth_error[visible_valid], 0.9).item()),
    }
    metrics.update(_surface_metrics(params, runner.intrinsics,
                                    runner.first_frame_w2c, gt_depth))

    _save_rgb(os.path.join(output_dir, "ground_truth_rgb.png"), target)
    _save_rgb(os.path.join(output_dir, "initialization_render_rgb.png"), render)
    gt_np = _rgb_uint8(target)
    render_np = _rgb_uint8(render)
    rgb_error_np = rgb_error.cpu().numpy()
    silhouette_np = silhouette.cpu().numpy()
    gt_depth_np = gt_depth.cpu().numpy()
    normalized_depth_np = normalized_depth.cpu().numpy()
    depth_error_np = depth_error.cpu().numpy()

    plt.imsave(os.path.join(output_dir, "initialization_silhouette.png"),
               silhouette_np, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(os.path.join(output_dir, "initialization_rgb_error.png"),
               rgb_error_np, cmap="magma", vmin=0.0, vmax=0.5)
    depth_vmin, depth_vmax = _depth_limits(gt_depth_np, gt_depth_np > 0)
    plt.imsave(
        os.path.join(output_dir, "initialization_render_depth.png"),
        np.ma.masked_where(silhouette_np <= 1e-4, normalized_depth_np),
        cmap="turbo", vmin=depth_vmin, vmax=depth_vmax)
    depth_error_valid = depth_error_np[(gt_depth_np > 0) &
                                       (silhouette_np > 0.5)]
    depth_error_vmax = (float(np.quantile(depth_error_valid, 0.95))
                        if depth_error_valid.size else 0.1)
    plt.imsave(
        os.path.join(output_dir, "initialization_depth_error.png"),
        np.ma.masked_where(gt_depth_np <= 0, depth_error_np),
        cmap="magma", vmin=0.0, vmax=max(depth_error_vmax, 1e-4))
    _save_overview(
        os.path.join(output_dir, "initialization_overview.png"), gt_np,
        render_np, rgb_error_np, silhouette_np, gt_depth_np,
        normalized_depth_np, depth_error_np, metrics)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved initialization diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
