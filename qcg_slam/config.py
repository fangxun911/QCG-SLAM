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
