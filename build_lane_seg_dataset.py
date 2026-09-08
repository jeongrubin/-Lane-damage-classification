"""Build train/val/test splits with rasterized damage masks from LabelMe JSON.

Source layout expected (LabelMe polygon export, one folder per class):
    raw_labels/A/*.png, *.json   (class A: no damage instances)
    raw_labels/C/*.png, *.json   (class C: moderate damage)
    raw_labels/F/*.png, *.json   (class F: severe damage)

Each JSON's shapes list contains one outer polygon labeled with the class
name (A/C/F) plus zero or more damage-instance polygons labeled "1". Only
the "1" polygons become the segmentation ground truth — class A images
have none, so their masks are all-background by construction.

Output:
    data/segmentation/{train,val,test}/images/*.png
    data/segmentation/{train,val,test}/masks/*.png   (0/255 binary)
"""
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

random.seed(42)

SRC_DIR = "raw_labels"
OUT_DIR = "data/segmentation"
RESIZE = (256, 256)
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def build_mask(json_path, w, h):
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for shape in d["shapes"]:
        if shape["label"] == "1":
            pts = [tuple(p) for p in shape["points"]]
            if len(pts) >= 3:
                draw.polygon(pts, outline=1, fill=1)
    return mask


def main():
    records = []
    for cls in ["A", "C", "F"]:
        cls_dir = os.path.join(SRC_DIR, cls)
        for jf in sorted(f for f in os.listdir(cls_dir) if f.endswith(".json")):
            base = os.path.splitext(jf)[0]
            img_path = os.path.join(cls_dir, base + ".png")
            if os.path.exists(img_path):
                records.append((cls, img_path, os.path.join(cls_dir, jf)))
    print(f"총 라벨링된 이미지: {len(records)}")

    by_class = {"A": [], "C": [], "F": []}
    random.shuffle(records)
    for r in records:
        by_class[r[0]].append(r)

    splits = {"train": [], "val": [], "test": []}
    for cls, items in by_class.items():
        n = len(items)
        n_val = max(1, int(n * VAL_FRAC))
        n_test = max(1, int(n * TEST_FRAC))
        n_train = n - n_val - n_test
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    for split, items in splits.items():
        img_dir = os.path.join(OUT_DIR, split, "images")
        mask_dir = os.path.join(OUT_DIR, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for cls, img_path, json_path in items:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            mask = build_mask(json_path, w, h)
            out_name = f"{cls}_{os.path.splitext(os.path.basename(img_path))[0]}".replace(" ", "_")
            img.resize(RESIZE, Image.BILINEAR).save(os.path.join(img_dir, out_name + ".png"))
            mask_arr = (np.array(mask.resize(RESIZE, Image.NEAREST)) * 255).astype(np.uint8)
            Image.fromarray(mask_arr).save(os.path.join(mask_dir, out_name + ".png"))
        counts = ", ".join(f"{c}={sum(1 for x in items if x[0] == c)}" for c in ["A", "C", "F"])
        print(f"{split}: {len(items)}장 ({counts})")


if __name__ == "__main__":
    main()
