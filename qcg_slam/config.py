"""Configuration normalization helpers."""


def prepare_config(config):
    """Fill default config values expected by the runner."""
    # replica 配置文件里没有 use_depth_loss_thres；scannetpp有，如果没达到
    # thres，tracking轮数翻倍
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    # replica scannetpp配置文件里没有 visualize_tracking_loss
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    # replica scannetpp配置文件里 gaussian_distribution = isotropic
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    surface_defaults = {
        "normal_window": 5,
        "fallback_normal_window": 3,
        "depth_abs_thresh": 0.02,
        "depth_rel_thresh": 0.02,
        "min_plane_points": 6,
        "max_planarity_ratio": 0.05,
        "min_view_cos": 0.2,
        "normal_scale_min_ratio": 0.05,
        "normal_scale_max_ratio": 0.25,
        "node_min_valid_fraction": 0.5,
        "node_min_inlier_fraction": 0.8,
        "geometry_batch_size": 32768,
        "min_scale": 1e-6,
    }
    surface_defaults.update(config.get("surface_init", {}))
    config["surface_init"] = surface_defaults
    regularization_defaults = {
        "enabled": False,
        "min_normal_to_tangent_ratio": 0.05,
        "max_normal_to_tangent_ratio": 0.25,
        "max_normal_deviation_degrees": 15.0,
        "thickness_weight": 0.1,
        "normal_weight": 0.05,
    }
    regularization_defaults.update(config.get("surface_regularization", {}))
    min_ratio = regularization_defaults["min_normal_to_tangent_ratio"]
    max_ratio = regularization_defaults["max_normal_to_tangent_ratio"]
    if not 0 < min_ratio <= max_ratio:
        raise ValueError(
            "surface_regularization ratios must satisfy 0 < min <= max")
    max_deviation = regularization_defaults["max_normal_deviation_degrees"]
    if not 0 < max_deviation <= 90:
        raise ValueError(
            "surface_regularization max normal deviation must be in (0, 90]")
    if regularization_defaults["thickness_weight"] < 0 or \
            regularization_defaults["normal_weight"] < 0:
        raise ValueError("surface_regularization weights must be non-negative")
    config["surface_regularization"] = regularization_defaults
    return config


def prepare_dataset_config(dataset_config):
    """Fill dataset defaults and report separate resolution usage."""
    # replica 配置文件里没有 ignore_bad，这里默认为 False;scannet++是False
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    # replica 配置文件里没有 use_train_split，这里默认为 True，scannetpp是True
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True

    # densification 和 tracking 过程的图像尺寸都是和原图一样的
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config[
            "desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config[
            "desired_image_width"]
        separate_densification_res = False
    else:
        separate_densification_res = (
            dataset_config["densification_image_height"] !=
            dataset_config["desired_image_height"] or
            dataset_config["densification_image_width"] !=
            dataset_config["desired_image_width"])

    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config[
            "desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config[
            "desired_image_width"]
        separate_tracking_res = False
    else:
        separate_tracking_res = (dataset_config["tracking_image_height"] !=
                                 dataset_config["desired_image_height"] or
                                 dataset_config["tracking_image_width"] !=
                                 dataset_config["desired_image_width"])

    return dataset_config, separate_densification_res, separate_tracking_res
