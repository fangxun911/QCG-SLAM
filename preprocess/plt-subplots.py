import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

x_list = np.arange(1, 2000)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].plot(x_list, 2 * x_list)
axs[0, 0].set_title("RGB PSNR")
axs[0, 0].set_xlabel("Time Step")
axs[0, 0].set_ylabel("PSNR")
axs[0, 1].plot(x_list, np.log(x_list))
axs[0, 1].set_title("Depth L1")
axs[0, 1].set_xlabel("Time Step")
axs[0, 1].set_ylabel("L1 (cm)")
axs[1, 0].plot(x_list, np.sqrt(x_list))
axs[1, 0].set_title("RGB SSIM")
axs[1, 0].set_xlabel("Time Step")
axs[1, 0].set_ylabel("SSIM")
axs[1, 1].plot(x_list, np.power(x_list, 2))
axs[1, 1].set_title("RGB LPIPS")
axs[1, 1].set_xlabel("Time Step")
axs[1, 1].set_ylabel("LPIPS")

avg_psnr, avg_ssim, avg_lpips, avg_l1, ate_rmse = (31.35, 0.97, 0.12, 0.008,
                                                   0.004)
fig.suptitle(
    "Average PSNR: {:.2f}, Average SSIM: {:.2f}, Average LPIPS: {:.2f}, "
    "Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm, ".format(
        avg_psnr, avg_ssim, avg_lpips, avg_l1 * 100, ate_rmse * 100),
    fontsize=12)
plt.savefig("Metrics.png", bbox_inches='tight')
plt.close()
