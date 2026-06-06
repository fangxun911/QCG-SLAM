"""Runs QCG-SLAM RGB-D Gaussian SLAM training and evaluation."""

import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)


def main():
    import setproctitle

    from qcg_slam.runner import rgbd_slam
    from utils.common_utils import seed_everything

    setproctitle.setproctitle("QCG-SLAM")

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(os.path.basename(args.experiment),
                                  args.experiment).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])

    # Create Results Directory and Copy Config 保存最终结果的目录地址
    results_dir = os.path.join(experiment.config["workdir"],
                               experiment.config["run_name"])
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)


if __name__ == "__main__":
    main()
