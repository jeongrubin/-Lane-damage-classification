"""Build an Original / Ground Truth / Prediction comparison panel for the
U-Net damage segmentation model, sampled from data/segmentation/test.
"""
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.unet import UNet

random.seed(1)
DATA_DIR = "data/segmentation/test"
WEIGHTS = "lane_unet.pth"
OUT_PATH = "assets/segmentation/unet_gt_compare.png"
N_DAMAGED = 5
N_NORMAL = 1


def overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.5):
    colored = np.zeros_like(img_bgr)
    colored[mask == 1] = color
    return cv2.addWeighted(img_bgr, 1, colored, alpha, 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=2, base=32)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    model.to(device).eval()

    files = sorted(os.listdir(os.path.join(DATA_DIR, "images")))
    damaged = [f for f in files if f.startswith("C_") or f.startswith("F_")]
    normal = [f for f in files if f.startswith("A_")]
    random.shuffle(damaged)
    random.shuffle(normal)
    picked = damaged[:N_DAMAGED] + normal[:N_NORMAL]

    rows = []
    with torch.no_grad():
        for fn in picked:
            img = cv2.imread(os.path.join(DATA_DIR, "images", fn))
            gt = (cv2.imread(os.path.join(DATA_DIR, "masks", fn), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            pred = torch.argmax(model(tensor).squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            rows.append((
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(overlay(img, gt), cv2.COLOR_BGR2RGB),
                cv2.cvtColor(overlay(img, pred), cv2.COLOR_BGR2RGB),
            ))

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.3 * n))
    fig.suptitle("U-Net: Original / Ground Truth / Prediction (test set)", fontsize=13)
    col_titles = ["Original", "Ground Truth", "Prediction"]
    for i, (orig, gt, pred) in enumerate(rows):
        for j, im in enumerate([orig, gt, pred]):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=110)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
