"""Runs QCG-SLAM RGB-D Gaussian SLAM training and evaluation."""

import argparse
import os
import shutil
from importlib.machinery import SourceFileLoader
from qcg_slam.runner import rgbd_slam
from utils.common_utils import seed_everything


def main():
    """Load an experiment configuration and run QCG-SLAM."""

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config["seed"])

    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config["load_checkpoint"]:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)


if __name__ == "__main__":
    main()
