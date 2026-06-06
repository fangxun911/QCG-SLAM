# Third-Party Notices

This repository contains code derived from or adapted from third-party
projects. Those portions remain subject to their original license terms.

## SplaTAM

QCG-SLAM is based on SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D
SLAM.

- Project: https://github.com/spla-tam/SplaTAM
- License: BSD 3-Clause
- Copyright: Copyright (c) 2023, Nikhil Varma Keetha

The top-level `LICENSE` file retains the SplaTAM BSD 3-Clause license notice and
adds a copyright notice for QCG-SLAM modifications.

## GraphDECO / Inria Gaussian Splatting

Some files contain code copied or adapted from GraphDECO/Inria Gaussian
Splatting code, including files with explicit GraphDECO/Inria copyright headers
such as:

- `utils/slam_external.py`
- `utils/gs_external.py`
- `utils/graphics_utils.py`

Those portions are not covered by the BSD 3-Clause license in the same way as
the SplaTAM-derived code. They are subject to the Gaussian Splatting license,
which allows research and evaluation use and restricts commercial use without
prior explicit consent from the licensors.

A copy of the Gaussian Splatting license is included at:

- `third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md`

