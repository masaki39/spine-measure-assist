"""
Simple ONNX inference example for exported heatmap model.
Usage:
  uv run python train/infer_onnx.py --model best.onnx --image sample_image.npy --json sample_landmarks.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dataset import LANDMARK_ORDER, _percentile_clip_norm, _resize_with_padding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", required=True, help=".npy image path")
    p.add_argument("--json", help="Optional landmarks json to compare")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    return p.parse_args()


def preprocess(img_np, resize):
    if img_np.ndim == 3:
        img_np = img_np[0]
    img_np = _percentile_clip_norm(img_np)
    t = torch.from_numpy(img_np).unsqueeze(0)  # (1,H,W)
    t, scale, pad_x, pad_y = _resize_with_padding(t, tuple(resize))
    return t.unsqueeze(0), scale, pad_x, pad_y  # (1,1,H,W), ...


def postprocess_heatmaps(hm: np.ndarray):
    # hm: (1, L, H, W)
    hm = hm[0]
    coords = []
    for c in hm:
        idx = np.argmax(c)
        y, x = np.unravel_index(idx, c.shape)
        coords.append((float(x), float(y)))
    return coords


def main():
    args = parse_args()
    ort_sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    img_np = np.load(args.image)
    inp_t, scale, pad_x, pad_y = preprocess(img_np, args.resize)
    ort_out = ort_sess.run(None, {"image": inp_t.numpy()})
    coords = postprocess_heatmaps(ort_out[0])

    print("Predicted coords (x,y) in resized space:")
    for name, (x, y) in zip(LANDMARK_ORDER, coords):
        print(f"  {name}: ({x:.1f}, {y:.1f})")

    if args.json and os.path.exists(args.json):
        with open(args.json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        gt_orig = [(meta["landmarks_ijk"][k]["i"], meta["landmarks_ijk"][k]["j"]) for k in LANDMARK_ORDER]
        gt_resized = [(x * scale + pad_x, y * scale + pad_y) for (x, y) in gt_orig]
        print("\nGround truth in resized space (IJK scaled):")
        for name, (x, y) in zip(LANDMARK_ORDER, gt_resized):
            print(f"  {name}: ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
